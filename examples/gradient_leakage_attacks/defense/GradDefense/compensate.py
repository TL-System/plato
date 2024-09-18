"""
Gradient compensating of GradDefense

Reference:
Wang et al., "Protect Privacy from Gradient Leakage Attack in Federated Learning," INFOCOM 2022.
https://github.com/wangjunxiao/GradDefense
"""

import math

import numpy as np
import torch


def get_factor(num):
    """Calculate factors."""
    factors = []
    for_times = int(math.sqrt(num))
    for i in range(for_times + 1)[1:]:
        if num % i == 0:
            factors.append(i)
            t = int(num / i)
            if not t == i:
                factors.append(t)
    return factors


def get_matrix_size(total_params_num: int, q: float):
    """Calculate matrix size."""
    gradients_matrix_v = math.sqrt(q * total_params_num)

    for factor in sorted(get_factor(total_params_num)):
        if factor >= gradients_matrix_v:
            gradients_matrix_v = factor
            break

    gradients_matrix_w = int(total_params_num / gradients_matrix_v)

    assert isinstance(gradients_matrix_v, int)
    assert isinstance(gradients_matrix_w, int)
    assert gradients_matrix_v * gradients_matrix_w == total_params_num

    real_q = gradients_matrix_v / gradients_matrix_w

    return gradients_matrix_v, gradients_matrix_w, real_q


def get_covariance_matrix(matrix):
    """Calculate covariance matrix."""
    return np.cov(matrix, rowvar=0)


def denoise(gradients: list, scale: float, q: float):
    """Denoise gradients."""
    layer_dims_pool = []
    for layer in gradients:
        layer_dims = list((_ for _ in layer.shape))
        layer_dims_pool.append(layer_dims)

    layer_params_num_pool = []
    for layer_dims in layer_dims_pool:
        layer_params_num = 1
        for dim in layer_dims:
            layer_params_num *= dim
        layer_params_num_pool.append(layer_params_num)

    total_params_num = 0
    for layer_params_num in layer_params_num_pool:
        total_params_num += layer_params_num

    gradients_matrix_v, gradients_matrix_w, real_q = get_matrix_size(
        total_params_num=total_params_num, q=q
    )

    # Flatten gradients
    gradients_flatten = []
    for layer in gradients:
        layer_flatten = (torch.flatten(layer)).cpu().numpy()
        gradients_flatten.extend(layer_flatten)

    matrix_c = np.array(gradients_flatten).reshape(
        gradients_matrix_v, gradients_matrix_w
    )
    covmatrix_ctc = get_covariance_matrix(matrix_c)

    lamda_min = ((1 - 1 / math.sqrt(real_q)) ** 2) * (scale**2)
    lamda_max = ((1 + 1 / math.sqrt(real_q)) ** 2) * (scale**2)

    eigen_vals, eigen_vecs = np.linalg.eig(covmatrix_ctc)

    n_index = []
    for index, eigen_val in enumerate(eigen_vals):
        if eigen_val <= lamda_min or eigen_val >= lamda_max:
            n_index.append(index)
    n_eigen_vecs = eigen_vecs[:, n_index]

    low_data = np.dot(matrix_c, n_eigen_vecs)
    high_data = np.dot(low_data, n_eigen_vecs.T)

    compensated_gradients_flatten = high_data.flatten()

    gradients_compensated = []
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
                np.array(
                    compensated_gradients_flatten[
                        params_start_indice : params_end_indice + 1
                    ]
                )
            )
        ).reshape(layer_dims_pool[layer_index])

        gradients_compensated.append(layer_gradient)

    return gradients_compensated
