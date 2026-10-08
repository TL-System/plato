"""
Helped functions in PerFedRLNAS only applicable for search space: NASVIT
"""

from nasvit_wrapper.config import _C as config
from timm.loss import LabelSmoothingCrossEntropy
from torch import optim


def get_nasvit_loss_criterion():
    """Get timm Label Smoothing Cross Entropy, only NASVIT needs this."""
    return LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    """If the module name is in skip list, do not use weight decay."""
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if (  # pylint: disable=too-many-boolean-expressions
            len(param.shape) == 1
            or name.endswith(".bias")
            or "rescale" in name
            or "bn" in name
            or (name in skip_list)
            or check_keywords_in_name(name, skip_keywords)
        ):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [
        {"params": has_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def check_keywords_in_name(name, keywords=()):
    """Check whether module name is in keywords list."""
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


def get_optimizer(model):
    """Get a specific optimizer where only assigned parts of model weights use weight decay."""

    skip = {"rescale", "bn", "absolute_pos_embed"}
    skip_keywords = {"relative_position_bias_table"}
    if hasattr(model, "no_weight_decay"):
        skip = model.no_weight_decay()
    if hasattr(model, "no_weight_decay_keywords"):
        skip_keywords = model.no_weight_decay_keywords()

    # add weight decay before gamma (double check!!)
    parameters = set_weight_decay(model, skip, skip_keywords)
    base_opt = optim.AdamW
    optimizer = base_opt(
        parameters,
        eps=config.TRAIN.OPTIMIZER.EPS,
        betas=config.TRAIN.OPTIMIZER.BETAS,
        lr=config.TRAIN.BASE_LR,
        weight_decay=config.TRAIN.WEIGHT_DECAY,
    )
    return optimizer
