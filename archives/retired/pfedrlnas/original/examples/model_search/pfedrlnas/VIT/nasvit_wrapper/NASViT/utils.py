# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# --------------------------------------------------------
# Modified from Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

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


def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    logger.info(
        f"==============> Resuming form {config.MODEL.RESUME}...................."
    )
    if config.MODEL.RESUME.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location="cpu", check_hash=True
        )
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location="cpu")

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

                # print(pos_bias_table.permute(1, 0).reshape(1, num_head, old_window_size, old_window_size).shape, (new_window_size, new_window_size), 'shape')
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
    config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger, model_ema=None
):
    save_state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "max_accuracy": max_accuracy,
        "epoch": epoch,
        "config": config,
    }
    #               'model_ema': get_state_dict(model_ema)}
    if config.AMP_OPT_LEVEL != "O0":
        save_state["amp"] = amp.state_dict()

    save_path = os.path.join(config.OUTPUT, f"ckpt.pth")
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
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
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith("pth")]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max(
            [os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime
        )
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt
