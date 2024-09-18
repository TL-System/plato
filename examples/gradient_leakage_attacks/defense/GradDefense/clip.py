import math

import numpy as np
import torch
from defense.GradDefense.perturb import slicing
from utils.pseudorandom import ReservoirSample, getGause


def noise(
    dy_dx: list,
    sensitivity: list,
    slices_num: int,
    perturb_slices_num: int,
    noise_intensity: float,
):
    (
        layer_dims_pool,
        layer_params_num_pool,
        layer_params_num_gcd,
        slice_indices,
        slice_params_indice,
    ) = slicing(dy_dx[0], sensitivity, slices_num)

    # Pseudorandom sample slices
    sampled_slices = []
    rs = ReservoirSample(perturb_slices_num)
    for item in range(slices_num):
        sampled_slices = rs.feed(item)

    # Flatten gradients
    gradients_flatten_pool = []
    for sample_id in range(len(dy_dx)):
        gradients_flatten = []
        for layer in dy_dx[sample_id]:
            layer_flatten = torch.flatten(layer).cpu().numpy()
            gradients_flatten.extend(layer_flatten)
        # print(len(gradients_flatten))
        gradients_flatten_pool.append(gradients_flatten)

    # Gradients clipping
    for sampled_slice in sorted(sampled_slices):
        params_start_indice = slice_params_indice[sampled_slice]
        params_end_indice = slice_params_indice[sampled_slice + 1]
        for sample_id in range(len(dy_dx)):
            norm = np.linalg.norm(
                gradients_flatten_pool[sample_id][
                    params_start_indice : params_end_indice + 1
                ]
            )
            clipping_rate = max(1, norm / noise_intensity)
            gradients_flatten_pool[sample_id][
                params_start_indice : params_end_indice + 1
            ] = [
                x / clipping_rate
                for x in gradients_flatten_pool[sample_id][
                    params_start_indice : params_end_indice + 1
                ]
            ]

    gradients_flatten = np.mean(gradients_flatten_pool, axis=0)

    for sampled_slice in sorted(sampled_slices):
        params_start_indice = slice_params_indice[sampled_slice]
        params_end_indice = slice_params_indice[sampled_slice + 1]
        for params_index in range(params_start_indice, params_end_indice + 1, 1):
            gradients_flatten[params_index] += getGause(scale=noise_intensity)

    gradients_perturbed = []
    params_start_indice = 0
    params_end_indice = 0
    # Recover gradients
    for layer_index in range(len(layer_dims_pool)):
        if layer_index == 0:
            params_start_indice = 0
            params_end_indice = layer_params_num_pool[layer_index] - 1
        else:
            params_start_indice = params_end_indice + 1
            params_end_indice = (
                params_start_indice + layer_params_num_pool[layer_index] - 1
            )

        # print(params_start_indice, params_end_indice)
        # print(layer_dims_pool[layer_index])

        layer_gradient = (
            torch.from_numpy(
                np.array(gradients_flatten[params_start_indice : params_end_indice + 1])
            ).float()
        ).reshape(layer_dims_pool[layer_index])

        gradients_perturbed.append(layer_gradient)

    return gradients_perturbed
