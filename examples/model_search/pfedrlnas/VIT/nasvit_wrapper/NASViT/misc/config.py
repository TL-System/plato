# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os

import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = [""]

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 128
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = "./datasets/imagenet"
# Dataset name
_C.DATA.DATASET = "imagenet"
# Input image size
_C.DATA.IMG_SIZE = 224
# Interpolation to resize image (random, bilinear, bicubic)
# _C.DATA.INTERPOLATION = 'bilinear'
_C.DATA.INTERPOLATION = "bicubic"
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
_C.DATA.ZIP_MODE = False
# Cache Data in Memory, could be overwritten by command line argument
_C.DATA.CACHE_MODE = "part"
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = False
# Number of data loading threads
_C.DATA.NUM_WORKERS = 6  # 24
# Prefetch factor
_C.DATA.PREFETCH_FACTOR = 4  # 12

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = "vit"
# Model name
_C.MODEL.NAME = "vit"
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = "ckpt_360.pth"
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 1000
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 0
_C.TRAIN.SUBNET_REPEAT_SAMPLE = 1
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False
_C.TRAIN.USE_CONV_PROJ = True
_C.TRAIN.MAX = 192
# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = "cosine"
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = "adamw"
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = "rand-n1-m1-mstd0.5-inc1"
# Random erase prob
_C.AUG.REPROB = 0.01
# Random erase mode
_C.AUG.REMODE = "pixel"
# Random erase count
_C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0.01
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = "batch"

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Mixed precision opt level, if O0, no amp is used ('O0', 'O1', 'O2')
# overwritten by command line argument
_C.AMP_OPT_LEVEL = ""
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ""
# Tag of experiment, overwritten by command line argument
_C.TAG = "default"
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 50
_C.print_freq = 50
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0

_C.distributed = True
_C.multiprocessing_distributed = True
_C.arch = "attentive_nas_dynamic_model"
_C.sandwich_rule = True
_C.inplace_distill = True
_C.post_bn_calibration_batch_num = 64
_C.num_arch_training = 4

_C.supernet_config = CN()
_C.supernet_config.use_v3_head = True
_C.supernet_config.resolutions = [192, 224, 256, 288]
_C.supernet_config.first_conv = CN()
_C.supernet_config.first_conv.c = [16, 24]
_C.supernet_config.first_conv.act_func = "swish"
_C.supernet_config.first_conv.s = 2

_C.supernet_config.mb1 = CN()
_C.supernet_config.mb1.c = [16, 24]
_C.supernet_config.mb1.d = [1, 2]
_C.supernet_config.mb1.k = [3, 5]
_C.supernet_config.mb1.t = [1]
_C.supernet_config.mb1.s = 1
_C.supernet_config.mb1.act_func = "swish"
_C.supernet_config.mb1.se = False

_C.supernet_config.mb2 = CN()
_C.supernet_config.mb2.c = [24, 32]
_C.supernet_config.mb2.d = [3, 4, 5]
_C.supernet_config.mb2.k = [3, 5]
_C.supernet_config.mb2.t = [4, 5, 6]
_C.supernet_config.mb2.s = 2
_C.supernet_config.mb2.act_func = "swish"
_C.supernet_config.mb2.se = False

_C.supernet_config.mb3 = CN()
_C.supernet_config.mb3.c = [32, 40]
_C.supernet_config.mb3.d = [3, 4, 5, 6]
_C.supernet_config.mb3.k = [3]
_C.supernet_config.mb3.t = [4, 5, 6]
_C.supernet_config.mb3.s = 2
_C.supernet_config.mb3.act_func = "swish"
_C.supernet_config.mb3.se = True

_C.supernet_config.mb4 = CN()
_C.supernet_config.mb4.c = [64, 72]
_C.supernet_config.mb4.d = [3, 4, 5, 6]
_C.supernet_config.mb4.k = [3]
_C.supernet_config.mb4.t = [4, 5, 6]
_C.supernet_config.mb4.s = 2
_C.supernet_config.mb4.act_func = "swish"
_C.supernet_config.mb4.se = False

_C.supernet_config.mb5 = CN()
_C.supernet_config.mb5.c = [112, 120, 128]
_C.supernet_config.mb5.d = [3, 4, 5, 6, 7, 8, 9]
_C.supernet_config.mb5.k = [3]
_C.supernet_config.mb5.t = [4, 5, 6]
_C.supernet_config.mb5.s = 2
_C.supernet_config.mb5.act_func = "swish"
_C.supernet_config.mb5.se = True


_C.supernet_config.mb6 = CN()
_C.supernet_config.mb6.c = [160, 168, 176, 184]  # [168, 176, 184, 192]
_C.supernet_config.mb6.d = [3, 4, 5, 6, 7, 8]
_C.supernet_config.mb6.k = [3]
_C.supernet_config.mb6.t = [6]
_C.supernet_config.mb6.s = 1
_C.supernet_config.mb6.act_func = "swish"
_C.supernet_config.mb6.se = True

_C.supernet_config.mb7 = CN()
_C.supernet_config.mb7.c = [208, 216, 224]
_C.supernet_config.mb7.d = [3, 4, 5, 6]
_C.supernet_config.mb7.k = [3]
_C.supernet_config.mb7.t = [6]
_C.supernet_config.mb7.s = 2
_C.supernet_config.mb7.act_func = "swish"
_C.supernet_config.mb7.se = True

_C.supernet_config.last_conv = CN()
_C.supernet_config.last_conv.c = [1792, 1984]
_C.supernet_config.last_conv.act_func = "swish"

_C.sync_bn = False  # True


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, "r") as f:
        yaml_cfg = yaml.load(f, Loader=yaml.SafeLoader)

    for cfg in yaml_cfg.setdefault("BASE", [""]):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print("=> merge config from {}".format(cfg_file))
    if cfg_file.startswith("manifold://"):
        config.merge_from_file(pathmgr.get_local_path(cfg_file))
    else:
        config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    # merge from specific arguments
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size

    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.accumulation_steps:
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if args.use_checkpoint:
        config.TRAIN.USE_CHECKPOINT = True
    if args.amp_opt_level:
        config.AMP_OPT_LEVEL = args.amp_opt_level
    if args.output:
        config.OUTPUT = args.output
    if args.tag:
        config.TAG = args.tag
    if args.eval:
        config.EVAL_MODE = True
    if args.throughput:
        config.THROUGHPUT_MODE = True

    ## set local rank for distributed training
    # will be set later
    # config.LOCAL_RANK = args.local_rank

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    if not os.path.isdir(config.OUTPUT):
        os.makedirs(config.OUTPUT)

    # update DDP settings
    config.machine_rank = args.machine_rank
    config.num_nodes = args.num_machines
    config.dist_url = args.dist_url
    config.workflow_run_id = args.workflow_run_id

    # auto-resume is enabled by default for cloud jobs
    if config.workflow_run_id:
        config.TRAIN.AUTO_RESUME = True

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config
