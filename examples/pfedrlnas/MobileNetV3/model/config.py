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
