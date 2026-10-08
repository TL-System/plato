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

from .comm import reduce_eval_results
from .imagenet_eval import log_helper, validate_one_subnet
from .progress import AverageMeter, ProgressMeter, accuracy


def validate(
    subnets_to_be_evaluated,
    train_loader,
    val_loader,
    model,
    criterion,
    args,
    logger,
    bn_calibration=True,
):
    supernet = (
        model.module
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model
    )

    results = []
    top1_list, top5_list = [], []
    with torch.no_grad():
        for net_id in subnets_to_be_evaluated:
            if net_id == "attentive_nas_min_net":
                supernet.sample_min_subnet()
            elif net_id == "attentive_nas_max_net":
                supernet.sample_max_subnet()
            elif net_id.startswith("attentive_nas_random_net"):
                supernet.sample_active_subnet()
            else:
                supernet.set_active_subnet(
                    subnets_to_be_evaluated[net_id]["resolution"],
                    subnets_to_be_evaluated[net_id]["width"],
                    subnets_to_be_evaluated[net_id]["depth"],
                    subnets_to_be_evaluated[net_id]["kernel_size"],
                    subnets_to_be_evaluated[net_id]["expand_ratio"],
                )

            subnet = supernet.get_active_subnet()
            subnet_cfg = supernet.get_active_subnet_settings()
            subnet.cuda(args.gpu)

            if bn_calibration:
                subnet.eval()
                subnet.reset_running_stats_for_calibration()

                # estimate running mean and running statistics
                logger.info("Calirating bn running statistics")
                for batch_idx, (images, _) in enumerate(train_loader):
                    if batch_idx >= args.post_bn_calibration_batch_num:
                        break
                    if getattr(args, "use_clean_images_for_subnet_training", False):
                        _, images = images
                    images = images.cuda(args.gpu, non_blocking=True)
                    subnet(images)  # forward only

            acc1, acc5, loss, flops, params = validate_one_subnet(
                val_loader, subnet, criterion, args, logger
            )
            top1_list.append(acc1)
            top5_list.append(acc5)

            head_dim = 8
            func = (
                lambda x: x[0] ** 2
                * (
                    x[1] ** 2 * 6
                    + x[1] ** 2 * 8
                    + x[0] ** 2 * x[1] * 5
                    + 3 * 3 * x[1] * 4
                )
                + (x[1] // head_dim) ** 2 * x[0] ** 4 * 2
            )
            func = func
            flops += (
                (subnet_cfg["depth"][3] - 1)
                * func([subnet_cfg["resolution"] // 16, subnet_cfg["width"][4], 2])
                / 1e6
            )
            flops += (
                (subnet_cfg["depth"][4] - 1)
                * func([subnet_cfg["resolution"] // 32, subnet_cfg["width"][5], 1])
                / 1e6
            )
            flops += (
                (subnet_cfg["depth"][5] - 1)
                * func([subnet_cfg["resolution"] // 32, subnet_cfg["width"][6], 1])
                / 1e6
            )

            if subnet_cfg["resolution"] % 64 == 0:
                flops += (
                    func([subnet_cfg["resolution"] // 64, subnet_cfg["width"][7], 1])
                    / 1e6
                    * (subnet_cfg["depth"][6] - 1)
                )
            else:
                flops += (
                    func(
                        [subnet_cfg["resolution"] // 64 + 1, subnet_cfg["width"][7], 1]
                    )
                    / 1e6
                    * (subnet_cfg["depth"][6] - 1)
                )
            summary = str(
                {
                    "net_id": net_id,
                    "mode": "evaluate",
                    "epoch": getattr(args, "curr_epoch", -1),
                    "acc1": acc1,
                    "acc5": acc5,
                    "loss": loss,
                    "flops": flops,  # incorrect
                    "params": params,
                    **subnet_cfg,
                }
            )

            if args.distributed and getattr(args, "distributed_val", True):
                logger.info(summary)
                results += [summary]
            else:
                group = reduce_eval_results(summary, args.gpu)
                results += group
                for rec in group:
                    logger.info(rec)
    return results
