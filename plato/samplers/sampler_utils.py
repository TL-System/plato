"""
Useful tools used for implementing samplers

"""

import numpy as np


def extend_indices(indices, required_total_size):
    """ Extend the indices to obtain the required total size  
         by duplicating the indices """
    # add extra samples to make it evenly divisible, if needed
    if len(indices) < required_total_size:
        while len(indices) < required_total_size:
            indices += indices[:(required_total_size - len(indices))]
    else:
        indices = indices[:required_total_size]
    assert len(indices) == required_total_size

    return indices


def create_dirichlet_skew(
        total_size,  # the totoal size to generate partitions
        concentration,  # the beta of the dirichlet dictribution
        number_partitions,  # number of partitions
        min_partition_size=None,  # minimum required size for partitions
        is_extend_total_size=False):
    """ Create the distribution skewness based on the dirichlet distribution
    
        Note:
            is_extend_total_size (boolean) determines whether to generate the
             partitions satisfying min_partition_size by directly extending
             the total data size. 
    """
    if min_partition_size is not None:
        if not is_extend_total_size:
            min_size = 0
            while min_size < min_partition_size:
                proportions = np.random.dirichlet(
                    np.repeat(concentration, number_partitions))

                proportions = proportions / proportions.sum()
                min_size = np.min(proportions * total_size)

        else:  # extend the total size to satisfy the minimum requirement
            minimum_proportion_bound = float(min_partition_size / total_size)

            proportions = np.random.dirichlet(
                np.repeat(concentration, number_partitions))

            proportions = proportions / proportions.sum()

            # set the proportion to satisfy the minimum size
            def set_min_bound(pro):
                if pro > minimum_proportion_bound:
                    return pro
                else:
                    return minimum_proportion_bound

            proportions = list(map(lambda pro: set_min_bound(pro),
                                   proportions))

    else:
        proportions = np.random.dirichlet(
            np.repeat(concentration, number_partitions))

    return proportions
