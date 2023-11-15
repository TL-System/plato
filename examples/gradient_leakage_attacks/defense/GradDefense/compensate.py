import math

import numpy as np
import torch


def getFactor(num):
    factors = []
    for_times = int(math.sqrt(num))
    for i in range(for_times + 1)[1:]:
        if num % i == 0:
            factors.append(i)
            t = int(num / i)
            if not t == i:
                factors.append(t)
    return factors


def getMatrixSize(total_params_num: int, Q: float):
    gradients_matrix_v = math.sqrt(Q * total_params_num)

    for factor in sorted(getFactor(total_params_num)):
        if factor >= gradients_matrix_v:
            gradients_matrix_v = factor
            break

    gradients_matrix_w = int(total_params_num / gradients_matrix_v)

    assert isinstance(gradients_matrix_v, int)
    assert isinstance(gradients_matrix_w, int)
    assert gradients_matrix_v * gradients_matrix_w == total_params_num

    real_Q = gradients_matrix_v / gradients_matrix_w

    return gradients_matrix_v, gradients_matrix_w, real_Q


def getCovarianceMatrix(matrix):
    return np.cov(matrix, rowvar=0)


def denoise(gradients: list, scale: float, Q: float):
    layer_dims_pool = []
    for layer in gradients:
        layer_dims = list((_ for _ in layer.shape))
        layer_dims_pool.append(layer_dims)

    # print(layer_dims_pool)

    layer_params_num_pool = []
    for layer_dims in layer_dims_pool:
        layer_params_num = 1
        for dim in layer_dims:
            layer_params_num *= dim
        layer_params_num_pool.append(layer_params_num)

    # print(layer_params_num_pool)

    total_params_num = 0
    for layer_params_num in layer_params_num_pool:
        total_params_num += layer_params_num

    # print(total_params_num)

    gradients_matrix_v, gradients_matrix_w, real_Q = getMatrixSize(
        total_params_num=total_params_num, Q=Q
    )

    # print (gradients_matrix_v, gradients_matrix_w, real_Q)

    # Flatten gradients
    gradients_flatten = []
    for layer in gradients:
        layer_flatten = (torch.flatten(layer)).cpu().numpy()
        gradients_flatten.extend(layer_flatten)

    # print(len(gradients_flatten))

    matrix_C = np.array(gradients_flatten).reshape(
        gradients_matrix_v, gradients_matrix_w
    )
    covmatrix_CTC = getCovarianceMatrix(matrix_C)

    # print(covmatrix_CTC.shape)

    lamda_min = ((1 - 1 / math.sqrt(real_Q)) ** 2) * (scale**2)
    lamda_max = ((1 + 1 / math.sqrt(real_Q)) ** 2) * (scale**2)

    eigen_vals, eigen_vecs = np.linalg.eig(covmatrix_CTC)

    n_index = []
    for index in range(len(eigen_vals)):
        if eigen_vals[index] <= lamda_min or eigen_vals[index] >= lamda_max:
            n_index.append(index)
    n_eigen_vecs = eigen_vecs[:, n_index]

    lowData = np.dot(matrix_C, n_eigen_vecs)
    highData = np.dot(lowData, n_eigen_vecs.T)

    # print(n_eigen_vecs.shape)
    # print(lowData.shape)
    # print(highData.shape)

    compensated_gradients_flatten = highData.flatten()

    gradients_compensated = []
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
                np.array(
                    compensated_gradients_flatten[
                        params_start_indice : params_end_indice + 1
                    ]
                )
            )
        ).reshape(layer_dims_pool[layer_index])

        gradients_compensated.append(layer_gradient)

    return gradients_compensated
