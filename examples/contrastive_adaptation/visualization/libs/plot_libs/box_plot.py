"""
The implementation of plotting the boxplot
Please access examples from matplotlib
https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.boxplot.html

"""
#matplotlib.rcParams['text.usetex'] = True

import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .axes_set import set_ticker


def plot_one_stream_boxes(ax,
                          samples,
                          color,
                          boxes_position,
                          boxes_label=None):
    """ Plot one stream boxes for a sequence of samples.

        In general, these sequence of samples derive from
            one source. For example, each box corresponds samples
            of one time-slot.

        Args:
            ax (axes): the axes is about to be plot on.
            samples (list): a sequence of samples, in which
                each itme is a 1D-array containing the samples
                for corresponding box.
                length is N.
            color (list): the color of plot boxes
                if len(color) is 1, all boxes share one color,
                    otherwise len(color) == N
            boxes_position (list): the positions (x-axis) of where to
                plot boxes.
                len(boxes_position) == N
    """
    num_samples_part = len(samples)
    color = [color] if isinstance(color, str) else color

    assert len(color) == 1 or len(color) == len(num_samples_part)
    assert len(samples) == len(boxes_position)
    color = color * num_samples_part if len(color) == 1 else color

    # rectangular box plot
    bplot = ax.boxplot(
        samples,
        #vert=True,  # vertical box alignment
        notch=True,  # notch shape
        patch_artist=True,  # fill with color
        widths=0.2,
        positions=boxes_position,
        labels=boxes_label)  # will be used to label x-ticks

    for patch_idx, patch in enumerate(bplot['boxes']):
        patch.set_facecolor(color[patch_idx])

    return bplot


def plot_group_boxes(groups_samples,
                     groups_colors,
                     groups_boxes_positions,
                     center_pos,
                     groups_label,
                     xticklabels,
                     fig_style="seaborn-paper",
                     font_size="xx-small",
                     yscale=0.05,
                     xlim=[0.3, 7.5],
                     ylim=[0.1, 1],
                     save_file_path=None,
                     save_file_name=None):
    """ Plot a group of boxes while each group corresponds to one data stream.

        groups_samples (list): a nested list containing multiple data streams,
            each item is a list containing samples of the data stream.
            The structure of the samples is consistent with that in
            'plot_one_stream_boxes' function.
            len(groups_samples) == N.

        groups_colors (list): a nested list containing colors for  multiple data streams,
            each item is a list containing colors (aka. a list) for boxes of this
            data stream.
            len(groups_samples) == N.

        groups_boxes_positions (list): a nested list containig positions for data stream's
            boxes.
    """
    with plt.style.context(fig_style):
        whole_fig, whole_axs = plt.subplots(1, 1)  # figsize=(10, 10)
    plt.rcParams['legend.title_fontsize'] = font_size

    for group_idx, group_samples in enumerate(groups_samples):

        plot_one_stream_boxes(
            ax=whole_axs,
            samples=group_samples,
            color=groups_colors[group_idx],
            boxes_position=groups_boxes_positions[group_idx],
            boxes_label=None,
        )

    set_ticker(whole_axs, yscale=yscale, y_grid="major", y_type="%.2f")

    whole_axs.set_ylabel(r"Accuracy (%)", fontsize=16, weight='bold')
    whole_axs.set_xlabel('Communication rounds', fontsize=16, weight='bold')

    whole_axs.set_xticks(center_pos)
    whole_axs.set_ylim(ylim)
    whole_axs.set_xlim(xlim)

    plt.setp(whole_axs, xticks=center_pos, xticklabels=xticklabels)

    # where some data has already been plotted to ax
    handles, labels = whole_axs.get_legend_handles_labels()

    # manually define a new patch
    patches = [
        mpatches.Patch(color=groups_colors[group_idx][0],
                       label=groups_label[group_idx])
        for group_idx, _ in enumerate(groups_samples)
    ]
    handles = handles + patches

    # plot the legend
    plt.legend(handles=handles, fontsize=10)

    whole_fig.tight_layout()

    save_file_path = os.getcwd() if save_file_path is None else save_file_path
    save_file_name = "group_boxes" if save_file_name is None else save_file_name

    plt.savefig(os.path.join(save_file_path, save_file_name + '.png'))
    plt.savefig(os.path.join(save_file_path, save_file_name + '.pdf'),
                format='pdf')


def plot_points(ax, samples, color, marker_type, label, vol=10):
    X = samples[:, 0]
    Y = samples[:, 1]
    ax.scatter(X, Y, c=color, s=vol, alpha=1, marker=marker_type, label=label)
