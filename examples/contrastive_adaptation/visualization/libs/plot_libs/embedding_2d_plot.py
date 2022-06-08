"""
Visualization of

"""

import os
import matplotlib.pyplot as plt

import seaborn as sns


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
