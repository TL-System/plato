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
from libs.utils import positions_compute


def plot_one_stream_boxes(ax,
                          samples,
                          color,
                          boxes_position,
                          box_width,
                          flier_points_type="+"):
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
        sym=flier_points_type,
        patch_artist=True,  # fill with color
        widths=box_width,
        positions=boxes_position)  # will be used to label x-ticks

    for patch_idx, patch in enumerate(bplot['boxes']):
        patch.set_facecolor(color[patch_idx])

    return bplot


def plot_items_stream_boxes(whole_ax,
                            items_samples,
                            items_colors,
                            plot_center_positions,
                            items_legend_label,
                            box_width=1,
                            box_interval=0.5):
    """ Plotting stream boxes for a set of items

        Args:
            items_samples (list): a nested list containing multiple data streams for
                items to be plotted,
                each element in this list should be an array that contains one stream
                of samples to plot as boxes.

            items_colors (list): a list containing colors for items,

            plot_center_positions (list): a list containing the center positions to
                plot these items.

    """
    num_of_items = len(items_samples)
    plotted_groups_holder = []
    for item_idx, item_samples in enumerate(items_samples):
        boxes_position = positions_compute.get_item_plot_positions(
            item_idx=item_idx,
            num_plot_items=num_of_items,
            center_positions=plot_center_positions,
            plot_width=box_width + box_interval)

        plot_boxes_holder = plot_one_stream_boxes(
            ax=whole_ax,
            samples=item_samples,
            color=items_colors[item_idx],
            boxes_position=boxes_position,
            box_width=box_width,
        )
        plotted_groups_holder.append(plot_boxes_holder)

    order = list(range(num_of_items))

    whole_ax.legend([plotted_groups_holder[i]["boxes"][0] for i in order],
                    [items_legend_label[i] for i in order],
                    loc='best',
                    prop={
                        'size': 12,
                        'weight': "bold"
                    })

    return whole_ax


def plot_points(ax, samples, color, marker_type, label, vol=10):
    X = samples[:, 0]
    Y = samples[:, 1]
    ax.scatter(X, Y, c=color, s=vol, alpha=1, marker=marker_type, label=label)


def group_box_fig_save(whole_fig, save_file_path=None, save_file_name=None):

    whole_fig.tight_layout()

    save_file_path = os.getcwd() if save_file_path is None else save_file_path
    save_file_name = "group_boxes" if save_file_name is None else save_file_name

    plt.savefig(os.path.join(save_file_path, save_file_name + '.png'))
    plt.savefig(os.path.join(save_file_path, save_file_name + '.pdf'),
                format='pdf')
