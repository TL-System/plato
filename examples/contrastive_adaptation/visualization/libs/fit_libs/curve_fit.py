'''
In this part, we convert the digital points of training curve to the actual one
'''

import numpy as np

import pandas as pd

from .utils import find_value_position


def fit_curve_once(
    xlsx_data,
    fit_data_name,
    expected_xs,
    around_value=1,
    fit_level=0,
    x_scale=1,
    x_shift=0,
):

    # load the data
    fit_data_X = xlsx_data[fit_data_name + "_X"]
    fit_data_Y = xlsx_data[fit_data_name + "_Y"]

    fit_data_X.dropna(inplace=True)
    fit_data_Y.dropna(inplace=True)

    fit_data_X = np.around(fit_data_X, decimals=around_value)

    if fit_level != 0:
        fit_points = np.polyfit(fit_data_X, fit_data_Y, fit_level)

        fit_func = np.poly1d(fit_points)

        expected_xs = expected_xs
        expected_ys = fit_func(expected_xs)

    else:
        expected_xs = fit_data_X.to_numpy(dtype=int)
        expected_ys = fit_data_Y.to_numpy(dtype=float)

    expected_xs = expected_xs * x_scale
    expected_xs = expected_xs - x_shift
    expected_xs = expected_xs.astype(int)
    return expected_xs, expected_ys


def fir_curve_segments(
        xlsx_data,
        fit_data_name,
        segments,  # e.g. [(start_point), (end_point)]
        segments_points,  # e.g. [50, 60]
        segments_levels,  # e.g. [4, 5]
        around_value=1,
        x_scale=1,
        x_shift=0):

    # load the data
    fit_data_X = xlsx_data[fit_data_name + "_X"]
    fit_data_Y = xlsx_data[fit_data_name + "_Y"]

    fit_data_X.dropna(inplace=True)
    fit_data_Y.dropna(inplace=True)

    fit_data_X = np.around(fit_data_X, decimals=around_value)

    segments_fitted_curves = list()
    for seg_idx in range(len(segments)):
        seg_range = segments[seg_idx]
        seg_points = segments_points[seg_idx]
        seg_level = segments_levels[seg_idx]

        segment_required_xpoints = np.linspace(seg_range[0], seg_range[1],
                                               seg_points)

        segment_start_x_pos = find_value_position(fit_data_X, seg_range[0])
        segment_end_x_pos = find_value_position(fit_data_X, seg_range[1])

        segment_fit_xpoints = fit_data_X[segment_start_x_pos:segment_end_x_pos]
        segment_fit_ypoints = fit_data_Y[segment_start_x_pos:segment_end_x_pos]

        if seg_level != 0:
            fit_points = np.polyfit(segment_fit_xpoints, segment_fit_ypoints,
                                    seg_level)

            fit_func = np.poly1d(fit_points)

            expected_xs = segment_required_xpoints
            expected_ys = fit_func(expected_xs)

        else:
            expected_xs = segment_fit_xpoints.to_numpy(dtype=int)
            expected_ys = segment_fit_ypoints.to_numpy(dtype=float)

        segments_fitted_curves.append([expected_xs, expected_ys])

    # concat segments
    concated_segs_curve_X = list()
    concated_segs_curve_Y = list()
    for seg_curves_idx in range(len(segments_fitted_curves)):
        seg_curve_XY_points = segments_fitted_curves[seg_curves_idx]

        if seg_curves_idx == 0:
            pass
        else:
            pre_seg_curve_XY_points = segments_fitted_curves[seg_curves_idx -
                                                             1]

            diff = seg_curve_XY_points[1][0] - pre_seg_curve_XY_points[1][-1]
            seg_curve_XY_points[1][
                0] = pre_seg_curve_XY_points[1][-1] + diff / 2

        concated_segs_curve_X.append(seg_curve_XY_points[0][:-1])
        concated_segs_curve_Y.append(seg_curve_XY_points[1][:-1])

    concated_Xs = np.concatenate(concated_segs_curve_X)
    concated_Ys = np.concatenate(concated_segs_curve_Y)
    concated_Xs = concated_Xs * x_scale
    concated_Xs = concated_Xs - x_shift
    return concated_Xs, concated_Ys
