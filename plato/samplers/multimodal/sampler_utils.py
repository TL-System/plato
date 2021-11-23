"""
Useful tools used for samplers


"""

import numpy as np


def create_dirichlet_skew(total_size,
                          concentration,
                          number_partitions,
                          min_partition_size=None):
    """ Create the distribution skewness based on the dirichlet distribution """
    if min_partition_size is not None:
        min_size = 0
        while min_size < min_partition_size:
            proportions = np.random.dirichlet(
                np.repeat(concentration, number_partitions))

            proportions = proportions / proportions.sum()
            min_size = np.min(proportions * total_size)

    else:
        proportions = np.random.dirichlet(
            np.repeat(concentration, number_partitions))

    return proportions
