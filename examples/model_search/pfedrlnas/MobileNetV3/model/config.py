"""
Configuration file for MobileNetV3 search space.
"""

from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = [""]

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = "mobilenetv3"
# Model name
_C.MODEL.NAME = "mobilenetv3"
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


_C.supernet_config = CN()
_C.supernet_config.use_v3_head = True
_C.supernet_config.resolutions = [224]
_C.supernet_config.first_conv = CN()
_C.supernet_config.first_conv.c = [16]
_C.supernet_config.first_conv.act_func = "h_swish"
_C.supernet_config.first_conv.s = 2

_C.supernet_config.mb0 = CN()
_C.supernet_config.mb0.c = [16]
_C.supernet_config.mb0.d = [1, 2]
_C.supernet_config.mb0.k = [3, 5]
_C.supernet_config.mb0.t = [1]
_C.supernet_config.mb0.s = 2
_C.supernet_config.mb0.act_func = "h_swish"
_C.supernet_config.mb0.se = False

_C.supernet_config.mb1 = CN()
_C.supernet_config.mb1.c = [16, 24]
_C.supernet_config.mb1.d = [1, 2]
_C.supernet_config.mb1.k = [3, 5]
_C.supernet_config.mb1.t = [1]
_C.supernet_config.mb1.s = 2
_C.supernet_config.mb1.act_func = "h_swish"
_C.supernet_config.mb1.se = True

_C.supernet_config.mb2 = CN()
_C.supernet_config.mb2.c = [24, 40]
_C.supernet_config.mb2.d = [1, 2]
_C.supernet_config.mb2.k = [3, 5]
_C.supernet_config.mb2.t = [3, 4, 5]
_C.supernet_config.mb2.s = 2
_C.supernet_config.mb2.act_func = "h_swish"
_C.supernet_config.mb2.se = False

_C.supernet_config.mb3 = CN()
_C.supernet_config.mb3.c = [40, 80]
_C.supernet_config.mb3.d = [2, 3, 4]
_C.supernet_config.mb3.k = [3, 5]
_C.supernet_config.mb3.t = [3, 4, 5]
_C.supernet_config.mb3.s = 2
_C.supernet_config.mb3.act_func = "h_swish"
_C.supernet_config.mb3.se = True

_C.supernet_config.mb4 = CN()
_C.supernet_config.mb4.c = [48, 112]
_C.supernet_config.mb4.d = [2, 3, 4]
_C.supernet_config.mb4.k = [3, 5]
_C.supernet_config.mb4.t = [3, 4, 5]
_C.supernet_config.mb4.s = 1
_C.supernet_config.mb4.act_func = "h_swish"
_C.supernet_config.mb4.se = True

_C.supernet_config.mb5 = CN()
_C.supernet_config.mb5.c = [96, 160]
_C.supernet_config.mb5.d = [2, 3, 4]
_C.supernet_config.mb5.k = [3, 5]
_C.supernet_config.mb5.t = [3, 4, 5]
_C.supernet_config.mb5.s = 2
_C.supernet_config.mb5.act_func = "h_swish"
_C.supernet_config.mb5.se = True

_C.supernet_config.last_conv = CN()
_C.supernet_config.last_conv.c = [1280]
_C.supernet_config.last_conv.act_func = "h_swish"

_C.sync_bn = False  # True


def get_config():
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    # update_config(config, args)

    return config
