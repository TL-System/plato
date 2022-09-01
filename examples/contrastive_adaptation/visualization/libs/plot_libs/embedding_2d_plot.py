"""
Visualization of

"""

import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import cm
import numpy as np
import seaborn as sns


# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))
    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range

    return starts_from_zero / value_range


def get_pca_visualization(axis,
                          encoded_data,
                          encoded_data_label,
                          prepared_cmap,
                          embedding_type="PCA",
                          markerscale=2,
                          legend_fontsize=12):
    #pca = PCA(n_components=2)
    pca = TSNE(n_components=2, init='pca', n_iter=3000)

    pca_proj = pca.fit_transform(encoded_data)

    pca_proj[:, 0] = scale_to_01_range(pca_proj[:, 0])
    pca_proj[:, 1] = scale_to_01_range(pca_proj[:, 1])

    unique_labels = np.unique(encoded_data_label)
    unique_labels = [int(lb) for lb in unique_labels.tolist()]

    # Plot those points as a scatter plot and label them based on the pred labels

    for lab in unique_labels:
        indices = encoded_data_label == lab
        axis.scatter(pca_proj[indices, 0],
                     pca_proj[indices, 1],
                     color=np.array(prepared_cmap(lab)).reshape(1, 4),
                     label=lab,
                     alpha=0.5)
    axis.legend(fontsize=legend_fontsize, markerscale=2)


def visualize_2d_embeddings(self,
                            embedded_2d_df,
                            save_path=None,
                            save_prefix="2d_embedding"):
    """ Visualize the embedding in 2D space. """

    # plot the embedding as the scatter points
    fig, ax = plt.subplots()
    ax = sns.scatterplot('Dim1',
                         'Dim2',
                         data=embedded_2d_df,
                         palette='tab10',
                         hue='label',
                         linewidth=0,
                         alpha=0.6,
                         ax=ax)

    # plot the joint distribution
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    g = sns.jointplot('Dim1', 'Dim2', data=embedded_2d_df, kind="hex")
    plt.subplots_adjust(top=0.95)

    # save the two plots
    fig.savefig(os.path.join(save_path, save_prefix + '_2d_points.png'))
    fig.savefig(os.path.join(save_path, save_prefix + '_2d_points.pdf'),
                format='pdf')

    g.savefig(os.path.join(save_path, save_prefix + '_joint_2d_points.png'))
    g.savefig(os.path.join(save_path, save_prefix + '_joint_2d_points.pdf'),
              format='pdf')
