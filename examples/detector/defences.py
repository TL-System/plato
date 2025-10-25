import logging
from collections import OrderedDict
from typing import Mapping, Sequence

import numpy as np
import torch

from plato.config import Config

registered_defences = {}

WeightMapping = Mapping[str, torch.Tensor]


def get():
    defence_type = (
        Config().server.defence_type
        if hasattr(Config().server, "defence_type")
        else None
    )

    if defence_type is None:
        logging.info("No defence is applied.")
        return lambda x: x

    if defence_type in registered_defences:
        registered_defence = registered_defences[defence_type]
        logging.info(f"Clients perform {defence_type} defence.")
        return registered_defence

    raise ValueError(f"No such defence: {defence_type}")


def _flatten_single_weight(weight: WeightMapping) -> torch.Tensor:
    """Flatten a single model weight dictionary into a 1-D tensor."""
    flattened_parts = [param.reshape(-1) for param in weight.values()]
    if not flattened_parts:
        raise ValueError("Cannot flatten empty weight mapping.")
    return torch.cat(flattened_parts)


def flatten_weights(weights: Sequence[WeightMapping]) -> torch.Tensor:
    flattened_rows = [_flatten_single_weight(weight) for weight in weights]
    if not flattened_rows:
        return torch.empty((0, 0))
    return torch.stack(flattened_rows)


def reconstruct_weight(
    flattened_weight: torch.Tensor, reference: WeightMapping
) -> OrderedDict[str, torch.Tensor]:
    """Reconstruct model weights from a flattened tensor."""
    start_index = 0
    reconstructed: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, weight_value in reference.items():
        length = weight_value.numel()
        reconstructed[name] = (
            flattened_weight[start_index : start_index + length]
            .reshape(weight_value.shape)
            .to(weight_value.device)
        )
        start_index += length
    return reconstructed


def median(weights_attacked: Sequence[WeightMapping]) -> OrderedDict[str, torch.Tensor]:
    """Aggregate weight updates from the clients using median."""
    if len(weights_attacked) == 0:
        logging.info("Median defence received no client updates.")
        return OrderedDict()

    flattened = flatten_weights(weights_attacked)
    median_delta_vector = torch.median(flattened, dim=0).values
    median_update = reconstruct_weight(median_delta_vector, weights_attacked[0])

    logging.info("Finished Median server aggregation.")
    return median_update
