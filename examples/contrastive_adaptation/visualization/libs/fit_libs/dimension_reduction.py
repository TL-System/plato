"""
Implementation of the visualization of the encoded features.

For details example, please access
https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html
and
https://scikit-learn.org/stable/auto_examples/neighbors/plot_nca_dim_reduction.html#sphx-glr-auto-examples-neighbors-plot-nca-dim-reduction-py

"""

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import pandas as pd

dim_reduction_methods_pool = {"pca": PCA, "tsne": TSNE}


def learn_dimension_reduction(
    method_name,
    train_vectors,
    test_vectors,
    test_labels,
    config=None,
):
    """ Fit the dimension reduction method with the train set. """
    defined_algorithm = dim_reduction_methods_pool[method_name](config)
    defined_algorithm.fit(train_vectors)
    test_embedded_2d = defined_algorithm.transform(test_vectors)

    test_embedded_2d_df = pd.DataFrame(test_embedded_2d,
                                       columns=['Dim1', 'Dim2'])
    test_embedded_2d_df['label'] = test_labels

    return test_embedded_2d_df
