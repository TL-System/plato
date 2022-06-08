import numpy as np

from scipy import interpolate


def interp1d_curves(xlsx_data,
                    fit_data_name,
                    around_value=1,
                    expected_xs=None,
                    fit_level=None,
                    x_scale=1,
                    x_shift=0):
    """ This functions fit the function for separated points """
    # load the data
    fit_data_X = xlsx_data[fit_data_name + "_X"]
    fit_data_Y = xlsx_data[fit_data_name + "_Y"]

    fit_data_X.dropna(inplace=True)
    fit_data_Y.dropna(inplace=True)

    fit_data_X = np.around(fit_data_X, decimals=around_value)

    if expected_xs is None:
        min_x = fit_data_X.min()
        max_x = fit_data_X.max()

        expected_xs = np.arange(min_x + 1, max_x, 2)

    if fit_level != None:
        interp_func = interpolate.interp1d(fit_data_X,
                                           fit_data_Y,
                                           kind=fit_level)

        expected_xs = expected_xs
        expected_ys = interp_func(expected_xs)

    else:
        expected_xs = fit_data_X.to_numpy(dtype=int)
        expected_ys = fit_data_Y.to_numpy(dtype=float)

    expected_xs = expected_xs * x_scale
    expected_xs = expected_xs - x_shift
    expected_xs = expected_xs.astype(int)

    return expected_xs, expected_ys


def interp_spline_curves(xlsx_data,
                         fit_data_name,
                         around_value=1,
                         expected_xs=None,
                         fit_level=None,
                         x_scale=1,
                         x_shift=0,
                         expected_xs_type=int):
    # load the data
    fit_data_X = xlsx_data[fit_data_name + "_X"]
    fit_data_Y = xlsx_data[fit_data_name + "_Y"]

    fit_data_X.dropna(inplace=True)
    fit_data_Y.dropna(inplace=True)

    fit_data_X = np.around(fit_data_X, decimals=around_value)

    if expected_xs is None:
        min_x = fit_data_X.min()
        max_x = fit_data_X.max()

        expected_xs = np.arange(min_x + 1, max_x, 2)

    if fit_level != None:
        # obtain the cooeficits
        knots, b_spline_coefficients, dg_spline = interpolate.splrep(
            fit_data_X, fit_data_Y, k=fit_level, s=0)

        interp_spline = interpolate.BSpline(knots,
                                            b_spline_coefficients,
                                            dg_spline,
                                            extrapolate=False)

        expected_xs = expected_xs
        expected_ys = interp_spline(expected_xs)

    else:
        expected_xs = fit_data_X.to_numpy(dtype=expected_xs_type)
        expected_ys = fit_data_Y.to_numpy(dtype=float)

    expected_xs = expected_xs * x_scale
    expected_xs = expected_xs - x_shift
    expected_xs = expected_xs.astype(expected_xs_type)

    return expected_xs, expected_ys