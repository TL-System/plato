import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator, FormatStrFormatter


def set_ticker(ax,
               xscale=None,
               yscale=None,
               sub_xscale=None,
               sub_yscale=None,
               x_type="%.1f",
               y_type="%.1f",
               x_grid="major",
               y_grid="minor"):
    if xscale is not None:
        # modify major scale
        # set x major tick labels to multiples of xyscale
        xmajorLocator = MultipleLocator(xscale)
        # set the position of the main tick label
        ax.xaxis.set_major_locator(xmajorLocator)

    if yscale is not None:
        # set y major tick labels to multiples of xyscale
        ymajorLocator = MultipleLocator(yscale)
        ax.yaxis.set_major_locator(ymajorLocator)

    if x_type is not None:
        # format the x-axis label text
        xmajorFormatter = FormatStrFormatter(x_type)
        ax.xaxis.set_major_formatter(xmajorFormatter)
    if y_type is not None:
        # Format the y-axis label text
        ymajorFormatter = FormatStrFormatter(y_type)
        ax.yaxis.set_major_formatter(ymajorFormatter)

    if sub_xscale is not None:
        # set x-axis minor tick labels to multiples of sub_xyscale
        xminorLocator = MultipleLocator(sub_xscale)
        # set the position of the minor tick labels, without label text formatting
        ax.xaxis.set_minor_locator(xminorLocator)

    if sub_yscale is not None:
        # set y-axis minor tick labels to multiples of sub_xyscale
        yminorLocator = MultipleLocator(sub_yscale)
        ax.yaxis.set_minor_locator(yminorLocator)

    # open grid
    if x_grid is not None:
        # The grid of the x-axis uses the major ticks
        ax.xaxis.grid(True, which=x_grid)
    if y_grid is not None:
        # The grid of the y-axis uses the minor scale
        ax.yaxis.grid(True, which=y_grid)

    # remove top border
    ax.spines['top'].set_visible(False)
    # remove right border
    ax.spines['right'].set_visible(False)
    # set the line style
    ax.grid(linestyle="--")

    return ax
