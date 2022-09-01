"""
The implementation of plotting the bars

"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator

from libs.utils import positions_compute


def plot_text_values(axis, values, barplot, color, fontweight, skip_zero=True):
    """
    Attach a text label above each bar displaying its height
    """
    for rect_idx, rect in enumerate(barplot):
        height = rect.get_height()
        value = values[rect_idx]
        if value == 0 and skip_zero:
            continue
        axis.text(x=rect.get_x() + rect.get_width() / 2.,
                  y=height + 1,
                  s=value,
                  ha='center',
                  va='bottom',
                  color=color,
                  fontweight=fontweight,
                  fontsize="small")


def plot_one_stream_bars(axis, values, bars_position, bar_width, legend_label,
                         bars_color, bars_alpha, bars_hatch):
    """ Plot one stream bars for a sequence of data.

        In general, these sequence of samples derive from
            one source. For example, each box corresponds samples
            of one time-slot.

        Args:
            ax (axes): the axes is about to be plot on.
            values (list): a sequence of values, in which
                each itme is a value presenting the y value.
                length is N.
            bars_color (list): the color of plot boxes
                if len(color) is 1, all boxes share one color,
                    otherwise len(color) == N
            bars_position (list): the positions (x-axis) of where to
                plot boxes.
                len(boxes_position) == N
    """

    assert len(values) == len(bars_position)

    barplot = axis.bar(bars_position,
                       values,
                       bar_width,
                       label=legend_label,
                       color=bars_color,
                       alpha=bars_alpha,
                       hatch=bars_hatch)

    return barplot


def plot_items_stream_bars(
        whole_ax,
        items_values,  # a groups of values to plot
        items_legend_labels,  # the label presented in legend
        groups_labels,  # the label denotes the name of the group
        plot_center_positions,
        items_colors,  # the color to plot the corresponding item
        items_bar_hatchs,
        bar_width=0.15,
        bar_interval=0.0,
        legend_font_size=15,
        show_text_value=True):
    """ Plotting stream bars for a set of items

    """

    num_of_items = len(items_values)

    for item_idx, item_values in enumerate(items_values):
        legend_label = items_legend_labels[item_idx]
        bars_color = items_colors[item_idx]
        bars_hatch = items_bar_hatchs[item_idx]
        # x + (w * (1 - n) / 2) + i * w
        bars_position = positions_compute.get_item_plot_positions(
            item_idx=item_idx,
            num_plot_items=num_of_items,
            center_positions=plot_center_positions,
            plot_width=bar_width + bar_interval)

        # if the item_samples is an array, i.e., it contains only
        # one set of samples.
        if not isinstance(item_values, list) and not isinstance(
                item_values, np.ndarray):
            item_values = [item_values]

        barplot = plot_one_stream_bars(axis=whole_ax,
                                       values=item_values,
                                       bars_position=bars_position,
                                       bar_width=bar_width,
                                       legend_label=legend_label,
                                       bars_color=bars_color,
                                       bars_alpha=0.5,
                                       bars_hatch=bars_hatch)
        if show_text_value:
            plot_text_values(axis=whole_ax,
                             values=item_values,
                             barplot=barplot,
                             color="black",
                             fontweight="normal")

    # order = list(range(len(items_legend_labels)))
    # print(items_legend_labels)
    # whole_ax.legend([handles[i] for i in order], [labels[i] for i in order],
    #                 loc='best',
    #                 prop={
    #                     'size': legend_font_size,
    #                     'weight': 'bold'
    #                 })
    if groups_labels is not None:
        whole_ax.set_xticks(plot_center_positions,
                            groups_labels,
                            weight='bold')

    return whole_ax


def group_bar_fig_save(whole_fig, save_file_path=None, save_file_name=None):

    whole_fig.tight_layout()

    save_file_path = os.getcwd() if save_file_path is None else save_file_path
    save_file_name = "group_boxes" if save_file_name is None else save_file_name

    plt.savefig(os.path.join(save_file_path, save_file_name + '.png'))
    plt.savefig(os.path.join(save_file_path, save_file_name + '.pdf'),
                format='pdf')
