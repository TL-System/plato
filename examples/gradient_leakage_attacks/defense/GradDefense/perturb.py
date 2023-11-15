import math

import numpy as np
import torch
from utils.pseudorandom import ReservoirSample, getGause


def compute_gcd(layer_params_num_pool: list):
    gcd_ = 0
    for layer_index in range(len(layer_params_num_pool)):
        if layer_index == 0:
            gcd_ = layer_params_num_pool[layer_index]
        else:
            gcd_ = math.gcd(gcd_, layer_params_num_pool[layer_index])
    return gcd_


def compute_atom_size(layer_dims_pool: list):
    layer_params_num_pool = []
    for layer_dims in layer_dims_pool:
        layer_params_num = 1
        for dim in layer_dims:
            layer_params_num *= dim
        layer_params_num_pool.append(layer_params_num)

    # Compute greatest common divisor
    layer_params_num_gcd = compute_gcd(layer_params_num_pool)

    return layer_params_num_pool, layer_params_num_gcd


def slicing(dy_dx: list, sensitivity: list, slices_num: int):
    layer_dims_pool = []
    for layer_gradient in dy_dx:
        layer_dims = list((_ for _ in layer_gradient.shape))
        layer_dims_pool.append(layer_dims)

    layer_params_num_pool, layer_params_num_gcd = compute_atom_size(layer_dims_pool)

    sens_weighted_atom_pool = []
    for layer_index in range(len(layer_params_num_pool)):
        layer_params_num = layer_params_num_pool[layer_index]
        layer_sensitivity = sensitivity[layer_index]

        layer_atom_num = int(layer_params_num / layer_params_num_gcd)
        layer_atom_sens = layer_sensitivity / layer_atom_num
        sens_weighted_atom_pool.extend([layer_atom_sens] * layer_atom_num)

    avg_slice_sens = sum(sensitivity) / slices_num

    # last_fc_atom_start_index = sum(layer_params_num_pool[:-2]) / layer_params_num_gcd
    # print(sum(sens_weighted_atom_pool[:int(last_fc_atom_start_index)]))
    # print(sum(sens_weighted_atom_pool[int(last_fc_atom_start_index):]))

    # Simple bin-packing
    slice_params_indice = []
    slice_params_indice.append(0)
    slice_indices = []
    atom_index_start = 0
    for slice_index in range(slices_num):
        sens_cache = 0
        slice_atom_indices = []
        for atom_index in range(atom_index_start, len(sens_weighted_atom_pool), 1):
            if slice_index == slices_num - 1:
                # TODO distribute fc_layer atoms average on each slice,
                #     but now that, all are at the tail slice.
                slice_atom_indices.append(atom_index)
                continue
            sens_cache += sens_weighted_atom_pool[atom_index]
            if sens_cache < avg_slice_sens:
                slice_atom_indices.append(atom_index)
            else:
                atom_index_start = atom_index
                slice_params_indice.append(atom_index * layer_params_num_gcd - 1)
                break

        slice_indices.append(slice_atom_indices)

    slice_params_indice.append(len(sens_weighted_atom_pool) * layer_params_num_gcd - 1)
    # print(slice_params_indice)

    return (
        layer_dims_pool,
        layer_params_num_pool,
        layer_params_num_gcd,
        slice_indices,
        slice_params_indice,
    )


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
    ) = slicing(dy_dx, sensitivity, slices_num)

    # Pseudorandom sample slices
    sampled_slices = []
    rs = ReservoirSample(perturb_slices_num)
    for item in range(slices_num):
        sampled_slices = rs.feed(item)

    # Flatten gradients
    gradients_flatten = []
    for layer in dy_dx:
        layer_flatten = (torch.flatten(layer)).numpy()
        gradients_flatten.extend(layer_flatten)
    # print(len(gradients_flatten))

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
