"""This is the file conv2circulant from https://github.com/JunyiZhu-AI/R-GAP/blob/main/conv2circulant.py"""

import numpy as np


def generate_coordinates(x_shape, kernel, stride, padding):
    assert len(x_shape) == 4
    assert len(kernel.shape) == 4
    assert x_shape[1] == kernel.shape[1]
    k_i, k_j = kernel.shape[-2:]
    x_i, x_j = np.array(x_shape[-2:]) + 2 * padding
    y_i, y_j = (x_i - k_i) // stride + 1, (x_j - k_j) // stride + 1
    kernel = kernel.reshape(kernel.shape[0], -1)
    circulant_w = []
    for f in range(len(kernel)):
        circulant_row = []
        for u in range(len(kernel[f])):
            c = u // (k_i * k_j)
            h = (u - c * k_i * k_j) // k_j
            w = u - c * k_i * k_j - h * k_j
            rows = np.array(range(0, x_i - k_i + 1, stride)) + h
            cols = np.array(range(0, x_j - k_j + 1, stride)) + w
            circulant_unit = []
            for row in range(len(rows)):
                for col in range(len(cols)):
                    circulant_unit.append(
                        [f * y_i * y_j + row * y_j + col, c * x_i * x_j + rows[row] * x_j + cols[col]]
                    )
            circulant_row.append(circulant_unit)
        circulant_w.append(circulant_row)
    return np.array(circulant_w), x_shape[1] * x_i * x_j, kernel.shape[0] * y_i * y_j


def circulant_w(x_len, kernel, coors, y_len):
    weights = np.zeros([y_len, x_len], dtype=np.float32)
    kernel = kernel.reshape(kernel.shape[0], -1)
    for coor, f in list(zip(coors, kernel)):
        for c, v in list(zip(coor, f)):
            for h, w in c:
                assert weights[h, w] == 0
                weights[h, w] = v
    return weights


def aggregate_g(k, x_len, coors):
    k = k.squeeze()
    A_mat = []
    for coor in coors:
        A_row = []
        for c in coor:
            A_unit = np.zeros(shape=x_len, dtype=np.float32)
            for i in c:
                assert A_unit[i[1]] == 0
                A_unit[i[1]] = k[i[0]]
            A_row.append(A_unit)
        A_mat.append(A_row)
    A_mat = np.array(A_mat)
    return A_mat.reshape(-1, A_mat.shape[-1])
