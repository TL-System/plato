"""
Computing the position for plottings such as the bar plot, box plot.

"""
import numpy as np


def get_item_plot_positions(item_idx, num_plot_items, center_positions,
                            plot_width):
    """ Getting the positions to plot one item in multiple centers.

        For examples, one items has four data to be plotted in four places
        defined by the 'center_positions'.
        Then, there are a total 'num_plot_items' (such as 10) numbers of such items.
        Therefore, in one place with corresponding center position, 10 itmes will be plotted.

        Then, in one center position with index n:
            - the plot position of the item with 'item_idx' in these items should be computed to
              ensure that the center of these plotted items is the  by center_positions[n].


        Finally, this function is to compute where to plot the item with the 'item_idx' in all
        required 'center_positions'

        Return:
            item_plot_positions (list): a list in which each item with index j
                                presents where to plot the item in the j-th center
                                position.
    """
    if not isinstance(center_positions, np.ndarray):
        center_positions = np.array(center_positions)

    assert center_positions.ndim == 1

    item_plot_positions = center_positions + (plot_width *
                                              (1 - num_plot_items) /
                                              2) + item_idx * plot_width
    return item_plot_positions
