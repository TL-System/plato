# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn as nn
from timm.utils import get_state_dict

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

"""
def copy_file(source_path, target_path):
    if source_path.startswith("manifold://") and target_path.startswith("manifold://"):
        copy(source_path, target_path, overwrite=True)
    else:
        pass
"""


def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    logger.info(
        f"==============> Resuming form {config.MODEL.RESUME}...................."
    )
    if config.MODEL.RESUME.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location="cpu", check_hash=True
        )
    else:
        with open(config.MODEL.RESUME, "rb") as fp:
            checkpoint = torch.load(fp, map_location="cpu")

    checkpoint["model"] = checkpoint["model_ema"]

    for key in model.state_dict().keys():
        if "attn_mask" in key:  # and key in checkpoint['model'].keys():
            checkpoint["model"][key] = model.state_dict()[key]
        if "relative_position_index" in key:
            checkpoint["model"][key] = model.state_dict()[key]
        if "relative_position_bias_table" in key:
            if checkpoint["model"][key].shape[0] != model.state_dict()[key].shape[0]:
                pos_bias_table = checkpoint["model"][key]
                old_window_size = int(pos_bias_table.shape[0] ** 0.5)
                new_window_size = int(model.state_dict()[key].shape[0] ** 0.5)
                num_head = pos_bias_table.shape[1]

                new_pos_bias_table = torch.nn.functional.interpolate(
                    pos_bias_table.permute(1, 0).reshape(
                        1, num_head, old_window_size, old_window_size
                    ),
                    size=(new_window_size, new_window_size),
                    mode="bicubic",
                    align_corners=False,
                )
                checkpoint["model"][key] = new_pos_bias_table.reshape(
                    num_head, -1
                ).permute(1, 0)

    msg = model.load_state_dict(checkpoint["model"], strict=False)

    logger.info(msg)
    max_accuracy = 0.0
    if (
        not config.EVAL_MODE
        and "optimizer" in checkpoint
        and "lr_scheduler" in checkpoint
        and "epoch" in checkpoint
    ):
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint["epoch"] + 1
        config.freeze()
        if (
            "amp" in checkpoint
            and config.AMP_OPT_LEVEL != "O0"
            and checkpoint["config"].AMP_OPT_LEVEL != "O0"
        ):
            amp.load_state_dict(checkpoint["amp"])
        logger.info(
            f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})"
        )
        if "max_accuracy" in checkpoint:
            max_accuracy = checkpoint["max_accuracy"]

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy


def save_checkpoint(
    config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger, model_ema
):
    save_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "max_accuracy": max_accuracy,
        "epoch": epoch,
        "config": config,
        "model_ema": get_state_dict(model_ema),
    }
    if config.AMP_OPT_LEVEL != "O0":
        save_state["amp"] = amp.state_dict()

    save_path = os.path.join(config.OUTPUT, f"ckpt.pth")
    logger.info(f"{save_path} saving......")
    with open(save_path, "wb") as fp:
        torch.save(save_state, fp)

    """
    if (epoch + 1) % 10 == 0:
        save_path_dup = os.path.join(config.OUTPUT, f'ckpt_{epoch+1}.pth')
        # copy_file(save_path, save_path_dup)
    """

    logger.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    latest_checkpoint = os.path.join(output_dir, f"ckpt.pth")
    if os.path.isfile(latest_checkpoint):
        resume_file = latest_checkpoint
    else:
        resume_file = None
    # checkpoints = os.listdir(output_dir)
    # checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    # print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    # if len(checkpoints) > 0:
    #    latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
    #    print(f"The latest checkpoint founded: {latest_checkpoint}")
    #    resume_file = latest_checkpoint
    # else:
    #    resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_master_process():
    return get_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def get_parallel_model(model, device):
    if get_world_size() > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device], find_unused_parameters=True
        )
    else:
        raise NotImplementedError
    return model


def reduce_dict(input_dict, average=True):
    """
    Reduce the values in the dictionary from all processes so that process with rank
    0 has the reduced results.

    Args:
        input_dict (dict): inputs to be reduced. All the values must be scalar CUDA Tensor.
        average (bool): whether to do average or sum

    Returns:
        a dict with the same keys as input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.reduce(values, dst=0)
        if dist.get_rank() == 0 and average:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict
