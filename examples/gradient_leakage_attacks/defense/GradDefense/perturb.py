"""
Gradient perturbation of GradDefense

Reference:
Wang et al., "Protect Privacy from Gradient Leakage Attack in Federated Learning," INFOCOM 2022.
https://github.com/wangjunxiao/GradDefense
"""

import math
import random

import numpy as np
import torch

random.seed(0)


def get_gause(scale: float, e: float = 0):
    """Obtain gaussian distribution."""
    return random.gauss(e, scale)


class ReservoirSample:
    """Class for Reservoir samples."""

    def __init__(self, size):
        self._size = size
        self._counter = 0
        self._sample = []

    def feed(self, item):
        """Generate Reservoir samples."""
        self._counter += 1
        if len(self._sample) < self._size:
            self._sample.append(item)
            return self._sample
        rand_int = random.randint(1, self._counter)
        if rand_int <= self._size:
            self._sample[rand_int - 1] = item
        return self._sample


def compute_gcd(layer_params_num_pool: list):
    """Compute GCD."""
    gcd_ = 0
    for layer_index, layer in enumerate(layer_params_num_pool):
        if layer_index == 0:
            gcd_ = layer
        else:
            gcd_ = math.gcd(gcd_, layer)
    return gcd_


def compute_atom_size(layer_dims_pool: list):
    """Compute atom size."""
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
    """Slice gradients."""
    layer_dims_pool = []
    for layer_gradient in dy_dx:
        layer_dims = list((_ for _ in layer_gradient.shape))
        layer_dims_pool.append(layer_dims)

    layer_params_num_pool, layer_params_num_gcd = compute_atom_size(layer_dims_pool)

    sens_weighted_atom_pool = []
    for layer_index, layer in enumerate(layer_params_num_pool):
        layer_params_num = layer
        layer_sensitivity = sensitivity[layer_index]

        layer_atom_num = int(layer_params_num / layer_params_num_gcd)
        layer_atom_sens = layer_sensitivity / layer_atom_num
        sens_weighted_atom_pool.extend([layer_atom_sens] * layer_atom_num)

    avg_slice_sens = sum(sensitivity) / slices_num

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
                # but now that, all are at the tail slice.
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
    """Gradients perturbation using GradDefense."""
    (
        layer_dims_pool,
        layer_params_num_pool,
        _,
        _,
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

    for sampled_slice in sorted(sampled_slices):
        params_start_indice = slice_params_indice[sampled_slice]
        params_end_indice = slice_params_indice[sampled_slice + 1]
        for params_index in range(params_start_indice, params_end_indice + 1, 1):
            gradients_flatten[params_index] += get_gause(scale=noise_intensity)

    gradients_perturbed = []
    params_start_indice = 0
    params_end_indice = 0

    # Recover gradients
    for layer_index, _ in enumerate(layer_dims_pool):
        if layer_index == 0:
            params_start_indice = 0
            params_end_indice = layer_params_num_pool[layer_index] - 1
        else:
            params_start_indice = params_end_indice + 1
            params_end_indice = (
                params_start_indice + layer_params_num_pool[layer_index] - 1
            )

        layer_gradient = (
            torch.from_numpy(
                np.array(gradients_flatten[params_start_indice : params_end_indice + 1])
            ).float()
        ).reshape(layer_dims_pool[layer_index])

        gradients_perturbed.append(layer_gradient)

    return gradients_perturbed


def noise_with_clip(
    dy_dx: list,
    sensitivity: list,
    slices_num: int,
    perturb_slices_num: int,
    noise_intensity: float,
):
    """Gradients clipping using GradDefense."""
    (
        layer_dims_pool,
        layer_params_num_pool,
        _,
        _,
        slice_params_indice,
    ) = slicing(dy_dx[0], sensitivity, slices_num)

    # Pseudorandom sample slices
    sampled_slices = []
    rs = ReservoirSample(perturb_slices_num)
    for item in range(slices_num):
        sampled_slices = rs.feed(item)

    # Flatten gradients
    gradients_flatten_pool = []
    for _, sample in enumerate(dy_dx):
        gradients_flatten = []
        for layer in sample:
            layer_flatten = torch.flatten(layer).cpu().numpy()
            gradients_flatten.extend(layer_flatten)
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
            gradients_flatten[params_index] += get_gause(scale=noise_intensity)

    gradients_perturbed = []
    params_start_indice = 0
    params_end_indice = 0
    # Recover gradients
    for layer_index, _ in enumerate(layer_dims_pool):
        if layer_index == 0:
            params_start_indice = 0
            params_end_indice = layer_params_num_pool[layer_index] - 1
        else:
            params_start_indice = params_end_indice + 1
            params_end_indice = (
                params_start_indice + layer_params_num_pool[layer_index] - 1
            )

        layer_gradient = (
            torch.from_numpy(
                np.array(gradients_flatten[params_start_indice : params_end_indice + 1])
            ).float()
        ).reshape(layer_dims_pool[layer_index])

        gradients_perturbed.append(layer_gradient)

    return gradients_perturbed
