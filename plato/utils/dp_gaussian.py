"""
A Gaussian mechanism used in Axiothea to ensure differential privacy.
"""

from collections import OrderedDict
import math

import torch
import numpy as np

from plato.config import Config


def gaussian_mechanism(weights, channel_type):
    """"Apply Gaussian mechanism."""
    clipped_weights, clipping_bound = clip_weights(weights)

    # Add Gaussian noise to clipped weights
    clipped_weights_with_noise = add_gaussian_noise(clipped_weights,
                                                    clipping_bound,
                                                    channel_type)

    return clipped_weights_with_noise


def clip_weights(weights):
    """Clip weights for using Gaussian mechanisum."""

    clipped_weights = OrderedDict()

    norm_list = [
        torch.linalg.norm(weight.float()).item()
        for _, weight in weights.items()
    ]
    clipping_bound = np.median(norm_list).item()

    # Compute clipped weights
    for name, weight in weights.items():
        weight_norm = torch.linalg.norm(weight.float()).item()
        if clipping_bound == 0:
            clipped_weights[name] = weight
        else:
            clipped_weights[name] = weight / max(1,
                                                 weight_norm / clipping_bound)

    return clipped_weights, clipping_bound


def add_gaussian_noise(clipped_weights, clipping_bound, channel_type):
    """Add Gaussian noise to clipped model weights."""

    epsilon = Config().algorithm.dp_epsilon
    delta = Config().algorithm.dp_delta

    weights_with_noise = OrderedDict()

    if channel_type == 'client_uplink' or channel_type == 'edge_server_uplink':
        if channel_type == 'client_uplink':
            # parameter d
            parameter = Config().data.partition_size
        if channel_type == 'edge_server_uplink':
            # parameter c
            parameter = Config().clients.per_round / Config(
            ).algorithm.total_silos

        for name, weight in clipped_weights.items():
            standard_deviation = 2 * clipping_bound / parameter / epsilon * math.sqrt(
                2 * math.log(1.25 / delta))
            # Computes noise and adds it to the weight
            noise = np.random.normal(loc=0.0,
                                     scale=standard_deviation,
                                     size=weight.shape)
            weights_with_noise[name] = (weight + noise).float()

    if channel_type == 'central_downlink' or channel_type == 'edge_server_downlink':
        if channel_type == 'edge_server_downlink':
            # parameter d
            parameter = Config().data.partition_size
            parameter_ = Config().clients.per_round / Config(
            ).algorithm.total_silos
            round_num = Config().algorithm.local_rounds
        if channel_type == 'central_downlink':
            # parameter c
            parameter = Config().clients.per_round / Config(
            ).algorithm.total_silos
            parameter_ = Config().algorithm.total_silos
            round_num = Config().trainer.rounds

        if round_num <= math.sqrt(parameter_):
            # No need to add noise to ensure differential privacy
            weights_with_noise = clipped_weights
        else:
            for name, weight in clipped_weights.items():
                standard_deviation = 2 * clipping_bound / parameter / parameter_ / epsilon * math.sqrt(
                    2 * math.log(1.25 / delta) *
                    (round_num * round_num - parameter_))

                # Computes noise and adds it to the weight
                noise = np.random.normal(loc=0.0,
                                         scale=standard_deviation,
                                         size=weight.shape)
                weights_with_noise[name] = (weight + noise).float()

    return weights_with_noise
