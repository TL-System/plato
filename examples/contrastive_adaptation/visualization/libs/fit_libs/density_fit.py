import os
import numpy as np

from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture


def fit_kernel_density(
    xlsx_data,
    fit_data_name,
    kernel_type,
    num_points,
    with_bound=True,
):
    fit_data = xlsx_data[fit_data_name]
    fit_data.dropna(inplace=True)

    fit_data = fit_data[:, np.newaxis]

    density_func = KernelDensity(kernel=kernel_type,
                                 bandwidth=0.2).fit(fit_data)

    generated_points = density_func.sample(n_samples=num_points)

    if with_bound:
        for point_idx in range(len(generated_points)):
            point = generated_points[point_idx]
            if point >= 1:
                generated_points[point_idx] = 1 - np.random.rand(1) * 0.02

    return generated_points


def fit_mixgaussian_density(xlsx_data,
                            fit_data_name,
                            num_points,
                            n_components,
                            with_bound=True):
    fit_data = xlsx_data[fit_data_name]
    fit_data.dropna(inplace=True)
    fit_data = fit_data[:, np.newaxis]
    fit_data = fit_data.reshape(-1, 1)

    density_gm = GaussianMixture(n_components=n_components).fit(fit_data)

    generated_points = density_gm.sample(n_samples=num_points)[0]

    generated_points = generated_points.reshape(-1)

    if with_bound:
        for point_idx in range(len(generated_points)):
            point = generated_points[point_idx]
            if point >= 1:
                generated_points[point_idx] = 1 - np.random.rand(1) * 0.02

    return generated_points


def fit_mixgaussian_2D_density(xlsx_data, fit_data_name=None, n_components=2):
    if fit_data_name is not None:
        D_points = xlsx_data[[fit_data_name[0], fit_data_name[1]]]
    else:
        D_points = xlsx_data[["X", "Y"]]

    D_points = D_points.dropna()

    D_points_array = D_points.to_numpy()

    density_gm = GaussianMixture(n_components=n_components).fit(D_points_array)

    return density_gm


def sample_mixgaussian_2D(density_gm,
                          num_points,
                          dec_round=2,
                          with_bound=True):
    generated_points = density_gm.sample(n_samples=num_points)[0]

    if with_bound:
        for point_idx in range(len(generated_points)):
            point = generated_points[point_idx]
            if point[0] >= 1:
                generated_points[point_idx][0] = 1 - np.random.rand(1) * 0.02
            if point[1] >= 1:
                generated_points[point_idx][1] = 1 - np.random.rand(1) * 0.02
    generated_points = np.around(generated_points, decimals=dec_round)
    return generated_points


def sample_mixgaussian_2D_on_dt(density_gm, required_dt, anchor_col=0):
    # generate points whose data in 'anchor_dim' is the sames as the 'required_dt'

    anchor_points = required_dt[:, anchor_col]
    expected_generated_points = np.zeros_like(required_dt)

    inserted_pos = 0
    searched_poses = list()
    while inserted_pos <= len(required_dt) - 1:
        for generate_circle in range(1000):
            generated_points = sample_mixgaussian_2D(density_gm=density_gm,
                                                     num_points=40000,
                                                     dec_round=1,
                                                     with_bound=False)
            extracted_poses = list()

            for idx in range(len(anchor_points)):
                if idx in searched_poses:
                    continue

                expected_anchor_point = anchor_points[idx]

                #search the generated_points
                for idj in range(len(generated_points)):
                    if idj in extracted_poses:  # skip the positions extracted
                        continue

                    if expected_anchor_point == generated_points[idj,
                                                                 anchor_col]:
                        expected_generated_points[
                            inserted_pos] = generated_points[idj, :]

                        inserted_pos += 1
                        extracted_poses.append(idj)
                        searched_poses.append(idx)
                        break
            if len(expected_generated_points) == len(required_dt):
                break

    return expected_generated_points