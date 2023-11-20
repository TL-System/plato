import torch
import numbers
import copy


@torch.no_grad()
def reconfigure_for_class_attack(
    model, target_classes=None, class_multiplier=0.5, bias_multiplier=1000
):
    model = copy.deepcopy(model)
    cls_to_obtain = wrap_indices(target_classes)

    *_, l_w, l_b = model.parameters()

    # linear weight
    masked_weight = torch.zeros_like(l_w)
    masked_weight[cls_to_obtain] = class_multiplier
    l_w.copy_(masked_weight)

    # linear bias
    masked_bias = torch.ones_like(l_b) * bias_multiplier
    masked_bias[cls_to_obtain] = l_b[cls_to_obtain]
    l_b.copy_(masked_bias)

    return model


def wrap_indices(indices):
    if isinstance(indices, numbers.Number):
        return [indices]
    else:
        return list(indices)
