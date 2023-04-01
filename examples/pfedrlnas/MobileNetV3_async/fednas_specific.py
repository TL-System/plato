"""
Helped functions in PerFedRLNAS only applicable for search space: NASVIT
"""
from timm.loss import LabelSmoothingCrossEntropy

from model.config import get_config


def get_nasvit_loss_criterion():
    """Get timm Label Smoothing Cross Entropy."""
    return LabelSmoothingCrossEntropy(smoothing=get_config().MODEL.LABEL_SMOOTHING)
