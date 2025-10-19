# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import time

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import models

from .flops_counter import count_net_flops_and_params
from .progress import AverageMeter, ProgressMeter, accuracy


def log_helper(summary, logger=None):
    if logger:
        logger.info(summary)
    else:
        print(summary)


def validate_one_subnet(
    val_loader,
    subnet,
    criterion,
    args,
    logger=None,
):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    log_helper("evaluating...", logger)
    # evaluation
    end = time.time()

    subnet.cuda(args.gpu)
    subnet.eval()  # freeze again all running stats

    for batch_idx, (images, target) in enumerate(val_loader):
        images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = subnet(images)
        loss = criterion(output, target).item()

        # measure accuracy
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.size(0)

        if args.distributed and getattr(args, "distributed_val", True):
            corr1, corr5, loss = acc1 * batch_size, acc5 * batch_size, loss * batch_size
            stats = torch.tensor([corr1, corr5, loss, batch_size], device=args.gpu)
            dist.barrier()  # synchronizes all processes
            dist.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
            corr1, corr5, loss, batch_size = stats.tolist()
            acc1, acc5, loss = corr1 / batch_size, corr5 / batch_size, loss / batch_size

        top1.update(acc1, batch_size)
        top5.update(acc5, batch_size)
        losses.update(loss, batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)

        if batch_idx % args.print_freq == 0:
            progress.display(batch_idx, logger)

    log_helper(
        " * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}, Top1: {top1.sum}/{top1.count}".format(
            top1=top1, top5=top5
        ),
        logger,
    )

    # compute flops
    if getattr(subnet, "module", None):
        resolution = subnet.module.resolution
    else:
        resolution = subnet.resolution
    data_shape = (1, 3, resolution, resolution)

    flops, params = count_net_flops_and_params(subnet, data_shape)
    return float(top1.avg), float(top5.avg), float(losses.avg), flops, params
