import copy
import numbers

import torch

from plato.config import Config


@torch.no_grad()
def reconfigure_for_class_attack(model, target_classes=None):
    """Reconfigure the model for class attack."""
    model = copy.deepcopy(model)
    cls_to_obtain = wrap_indices(target_classes)

    *_, l_w, l_b = model.parameters()

    # linear weight
    masked_weight = torch.zeros_like(l_w)
    masked_weight[cls_to_obtain] = Config().algorithm.class_multiplier
    l_w.copy_(masked_weight)

    # linear bias
    masked_bias = torch.ones_like(l_b) * Config().algorithm.bias_multiplier
    masked_bias[cls_to_obtain] = l_b[cls_to_obtain]
    l_b.copy_(masked_bias)

    return model.cpu().state_dict()


@torch.no_grad()
def reconfigure_for_feature_attack(
    model,
    feature_val,
    feature_loc,
    target_classes=None,
    allow_reset_param_weights=False,
):
    """Reconfigure the model for feature attack."""
    model = copy.deepcopy(model)
    cls_to_obtain = wrap_indices(target_classes)
    feature_loc = wrap_indices(feature_loc)

    if allow_reset_param_weights and Config().algorithm.reset_param_weights:
        feat_multiplier = 1
    else:
        feat_multiplier = Config().algorithm.feat_multiplier

    *_, l_w, l_b = model.parameters()

    masked_weight = torch.zeros_like(l_w)
    masked_weight[cls_to_obtain, feature_loc] = feat_multiplier
    l_w.copy_(masked_weight)

    masked_bias = torch.ones_like(l_b) * Config().algorithm.bias_multiplier
    masked_bias[cls_to_obtain] = -feature_val * feat_multiplier
    l_b.copy_(masked_bias)

    return model.cpu().state_dict()


def reconstruct_feature(shared_grad, shared_weights, cls_to_obtain):
    """Reconstruct features."""
    # Use weight or delta updates
    if shared_weights is not None:
        shared_grad = shared_weights
    weights = shared_grad[-2]
    bias = shared_grad[-1]
    grads_fc_debiased = weights / bias[:, None]

    if bias[cls_to_obtain] != 0:
        return grads_fc_debiased[cls_to_obtain]

    return torch.zeros_like(grads_fc_debiased[0])


def wrap_indices(indices):
    """Wrap indices."""
    if isinstance(indices, numbers.Number):
        return [indices]
    else:
        return list(indices)


def check_with_tolerance(value, feature_list, threshold=1e-3):
    """Check features with tolerance."""
    for i in feature_list:
        if abs(value - i) < threshold:
            return True

    return False
