import numpy as np


def noise_add_small(y, noise_size):
    small_noise = 0.005 * np.random.normal(size=noise_size)
    ydata = y + small_noise
    return ydata


def noise_add_medium(y, noise_size):
    medium_noise = 0.01 * np.random.normal(size=noise_size)
    ydata = y - np.absolute(medium_noise)
    return ydata


def noise_add_large(y, noise_size):
    large_noise = 0.03 * np.random.normal(size=noise_size)
    ydata = y - np.absolute(large_noise)
    return ydata


def add_noise(y, noise_size, noise_probs):

    noise_add_select = np.random.choice(3, 1, p=noise_probs)
    noise_add_select = noise_add_select[0]
    if noise_add_select == 0:
        ydata = noise_add_small(y, noise_size)
    if noise_add_select == 1:
        ydata = noise_add_medium(y, noise_size)
    if noise_add_select == 2:
        ydata = noise_add_large(y, noise_size)

    return ydata


def add_noise_elementwise(y, noise_probs, add_range=None):
    ydata = y
    for i in range(y.size):
        y_i = y[i]
        if add_range is None or i in add_range:
            ydata[i] = add_noise(y_i, 1, noise_probs)
        else:
            continue

    return ydata


def add_noise_segment(
    x,
    y,
    noise_segments_probs,
):

    ydata = y
    for i in range(y.size):
        x_i = x[i]
        y_i = y[i]
        for seg_inx in range(len(noise_segments_probs)):
            seg_info = noise_segments_probs[seg_inx]
            segment_noise_prob = seg_info[0]
            segment_range = seg_info[1]

            if x_i > segment_range[0] and x_i < segment_range[-1]:
                ydata[i] = add_noise(y_i, 1, segment_noise_prob)
            else:
                continue

    return ydata
