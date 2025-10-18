# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import gc
import math
import os
import random
import sys
import time
import warnings
from collections import defaultdict
from copy import deepcopy

import misc.attentive_nas_eval as attentive_nas_eval
import misc.logger as logging
import numpy as np
import timm as timm
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from misc.config import get_config
from misc.loss_ops import AdaptiveLossSoft
from misc.lr_scheduler import build_scheduler
from misc.optimizer import build_optimizer
from misc.utils import (
    auto_resume_helper,
    get_grad_norm,
    load_checkpoint,
    reduce_tensor,
    save_checkpoint,
)
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import AverageMeter, ModelEma, accuracy

import models
from data import build_loader

try:
    from apex import amp
except ImportError:
    amp = None

logger = logging.get_logger(__name__)


def parse_option():
    parser = argparse.ArgumentParser(
        "Swin Transformer training and evaluation script", add_help=False
    )
    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--working-dir",
        type=str,
        required=False,
        default="./",
        help="root dir for models and logs",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs="+",
    )

    # easy config modification
    parser.add_argument("--batch-size", type=int, help="batch size for single GPU")
    parser.add_argument("--resume", type=str, default="", help="resume path")
    parser.add_argument(
        "--accumulation-steps", type=int, help="gradient accumulation steps"
    )
    parser.add_argument(
        "--use-checkpoint",
        action="store_true",
        help="whether to use gradient checkpointing to save memory",
    )
    parser.add_argument(
        "--amp-opt-level",
        type=str,
        default="O1",
        choices=["O0", "O1", "O2"],
        help="mixed precision opt level, if O0, no amp is used",
    )
    parser.add_argument("--tag", help="tag of experiment")
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--throughput", action="store_true", help="Test throughput only"
    )

    # distributed data parallel settings
    parser.add_argument(
        "--machine-rank", default=0, type=int, help="machine rank, distributed setting"
    )
    parser.add_argument(
        "--num-machines",
        default=1,
        type=int,
        help="number of nodes, distributed setting",
    )
    parser.add_argument(
        "--workflow-run-id", default="", type=str, help="fblearner job id"
    )
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:10001",
        type=str,
        help="init method, distributed setting",
    )

    args, unparsed = parser.parse_known_args()

    # setup the work dir
    args.output = args.working_dir
    args.tag = args.workflow_run_id or args.tag  # override settings

    config = get_config(args)
    return args, config


def _setup_worker_env(gpu, ngpus_per_node, config):
    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    config.defrost()
    config.RANK = gpu + config.machine_rank * ngpus_per_node  # across machines
    config.WORLD_SIZE = ngpus_per_node * config.num_nodes
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=config.dist_url,
        world_size=config.WORLD_SIZE,
        rank=config.RANK,
    )
    config.gpu = gpu
    config.LOCAL_RANK = gpu
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    assert dist.get_world_size() == config.WORLD_SIZE, (
        "DDP is not properply initialized."
    )
    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = (
        config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    )
    linear_scaled_warmup_lr = (
        config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    )
    linear_scaled_min_lr = (
        config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    )
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = (
            linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        )
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    # Setup logging format.
    logging.setup_logging(
        os.path.join(config.OUTPUT, "stdout.log"),
        "a" if config.workflow_run_id else "w",
    )

    # backup the config
    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())


def adaptive_clip_grad(
    named_parameters, supernet_gn, clip_factor=0.01, eps=1e-3, norm_type=2.0
):
    # if isinstance(parameters, torch.Tensor):
    #     parameters = [parameters]
    for name, p in named_parameters:
        if p.grad is None:
            continue
        p_data = p.detach()
        g_data = p.grad.detach()

        if name in supernet_gn.keys():
            max_norm = 1.0 * supernet_gn[name]
        else:
            max_norm = 1.0 / 4.0
        grad_norm = unitwise_norm(g_data, norm_type=norm_type)
        new_grads = 4 * (max_norm + 1e-4) / grad_norm * g_data
        p.grad.detach().copy_(new_grads)


def main_worker(gpu, ngpus_per_node, config):
    _setup_worker_env(gpu, ngpus_per_node, config)

    model = models.model_factory.create_model(config)
    model.cuda()
    logger.info(str(model))
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = (
        build_loader(config)
    )

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = models.model_factory.create_model(config)
    model.cuda()
    logger.info(str(model))

    model_ema = ModelEma(model, decay=0.99985, device="", resume="")

    optimizer = build_optimizer(config, model)
    if config.AMP_OPT_LEVEL != "O0":
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=config.AMP_OPT_LEVEL
        )
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[config.LOCAL_RANK],
        broadcast_buffers=False,
        find_unused_parameters=True,
    )
    model_without_ddp = model.module

    teachers = []

    teacher = timm.create_model("swsl_resnext101_32x4d", pretrained=True)
    teacher = teacher.cuda()
    teacher.eval()
    teacher = torch.nn.parallel.DistributedDataParallel(
        teacher, device_ids=[config.LOCAL_RANK], broadcast_buffers=False
    )
    teachers.append(teacher)

    teacher = timm.create_model("swsl_resnext101_32x8d", pretrained=True)
    teacher = teacher.cuda()
    teacher.eval()
    teacher = torch.nn.parallel.DistributedDataParallel(
        teacher, device_ids=[config.LOCAL_RANK], broadcast_buffers=False
    )
    teachers.append(teacher)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, "flops"):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.AUG.MIXUP > 0.0:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(
                    f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}"
                )
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f"auto resuming from {resume_file}")
        else:
            logger.info(f"no checkpoint found in {config.OUTPUT}, ignoring auto resume")

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(
            config, model_without_ddp, optimizer, lr_scheduler, logger
        )
        model_ema.ema = deepcopy(model_without_ddp)
        # validate(config, data_loader_train, data_loader_val, model)
        # logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if config.EVAL_MODE:
            return

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    validate(config, data_loader_train, data_loader_val, model)
    logger.info("Start training")
    start_time = time.time()
    supernet_gn = defaultdict(float)

    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        if epoch > 100:
            optimizer.proj_student = False  # switch
        data_loader_train.sampler.set_epoch(epoch)

        supernet_gn = train_one_epoch(
            config,
            model,
            criterion,
            data_loader_train,
            optimizer,
            epoch,
            mixup_fn,
            lr_scheduler,
            teachers,
            model_ema,
            supernet_gn,
        )
        if dist.get_rank() == 0 and (
            epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)
        ):
            save_checkpoint(
                config,
                epoch,
                model_without_ddp,
                max_accuracy,
                optimizer,
                lr_scheduler,
                logger,
                model_ema,
            )

        validate(config, data_loader_train, data_loader_val, model)
        if epoch % 30 == 0:
            validate(config, data_loader_train, data_loader_val, model_ema.ema)
        print("inner", optimizer.grad_inner)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))


def train_one_epoch(
    config,
    model,
    criterion,
    data_loader,
    optimizer,
    epoch,
    mixup_fn,
    lr_scheduler,
    teacher=None,
    model_ema=None,
    supernet_gradnorm=None,
    drop=False,
):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    data_time = AverageMeter()
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    ce_criterion = nn.CrossEntropyLoss()
    teacher_criterion = AdaptiveLossSoft(alpha_min=-1.0, alpha_max=1.0)

    start = time.time()
    end = time.time()

    cfgs = []
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        data_time.update(time.time() - end)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if teacher == None:
            pass
        else:
            with torch.no_grad():
                temp = 5.0
                t_soft_logits = (temp * teacher[0](samples)).softmax(dim=-1)
                t_soft_logits1 = (temp * teacher[1](samples)).softmax(dim=-1)

        # max
        cfg = model.module.sample_max_subnet()
        outputs, supernet_features = model(samples)
        ce_loss = criterion(outputs, t_soft_logits / 2.0 + t_soft_logits1 / 2.0)

        if not math.isfinite(ce_loss.item()):
            pass
        ce_loss.backward()

        g_value = ce_loss.item()
        optimizer.g_value = g_value
        optimizer.first_step(zero_grad=True)

        with torch.no_grad():
            supernet_features = [item.detach() for item in supernet_features]
            soft_logits = outputs.clone().detach()  # .softmax(dim=-1)
            # multi_soft_logits = torch.cat((t_soft_logits, t_soft_logits, soft_logits.softmax(dim=-1)), dim=0)

        sandwich_rule = getattr(config, "sandwich_rule", True)
        num_subnet_training = max(2, getattr(config, "num_arch_training", 2))
        model.module.set_dropout_rate(0.0, 0.0, True)

        # renew cfg list
        if idx % config.TRAIN.SUBNET_REPEAT_SAMPLE == 0:
            cfgs = []

        for arch_id in range(1, num_subnet_training):
            if arch_id == 1 and sandwich_rule:
                model.module.sample_min_subnet()
            else:
                if idx % config.TRAIN.SUBNET_REPEAT_SAMPLE == 0:
                    cfg = model.module.sample_active_subnet()
                    cfgs.append(cfg)
                else:
                    cfg = cfgs[arch_id - 1]
                    model.module.set_active_subnet(
                        cfg["resolution"],
                        cfg["width"],
                        cfg["depth"],
                        cfg["kernel_size"],
                        cfg["expand_ratio"],
                    )

            outputs, subnet_features = model(samples)
            loss = teacher_criterion(outputs, soft_logits)  # .softmax(dim=-1))

            if not math.isfinite(loss.item()):
                pass  # continue
            loss.backward()

        if config.TRAIN.CLIP_GRAD:
            # adaptive_clip_grad(model.named_parameters(), supernet_gradnorm)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.TRAIN.CLIP_GRAD
            )

        optimizer.g_constraint = 1.8
        optimizer.second_step(zero_grad=True)
        optimizer.zero_grad()

        lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        if model_ema is not None:
            model_ema.update(model)

        loss_meter.update(ce_loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]["lr"]
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f"Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t"
                f"eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t"
                f"data {data_time.val:.4f} ({data_time.avg:.4f})\t"
                f"batch {batch_time.val:.4f} ({batch_time.avg:.4f})\t"
                f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                f"grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t"
                f"mem {memory_used:.0f}MB"
            )

    epoch_time = time.time() - start
    logger.info(
        f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}"
    )

    return supernet_gradnorm


@torch.no_grad()
def validate(config, train_loader, valid_loader, model):
    subnets_to_be_evaluated = {
        # 'attentive_nas_random_net': {}
        "attentive_nas_min_net": {},
        "attentive_nas_max_net": {},
    }

    criterion = nn.CrossEntropyLoss()

    attentive_nas_eval.validate(
        subnets_to_be_evaluated,
        train_loader,
        valid_loader,
        model,
        criterion,
        config,
        logger,
        bn_calibration=True,
    )


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(
            f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}"
        )
        return


if __name__ == "__main__":
    _, config = parse_option()

    cudnn.deterministic = True
    warnings.warn(
        "You have chosen to seed training. "
        "This will turn on the CUDNN deterministic setting, "
        "which can slow down your training considerably! "
        "You may see unexpected behavior when restarting "
        "from checkpoints."
    )

    random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    ngpus_per_node = 1  #  torch.cuda.device_count()
    # Use torch.multiprocessing.spawn to launch distributed processes: the
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
