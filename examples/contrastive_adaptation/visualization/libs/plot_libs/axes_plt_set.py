import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator, FormatStrFormatter


def set_plt_shape():
    plt.subplots_adjust(left=0.133,
                        bottom=0.13,
                        right=0.952,
                        top=0.96,
                        hspace=0.2,
                        wspace=0.2)


def set_ticker(
        ax,
        xlim=None,
        ylim=None,
        xscale=None,
        yscale=None,
        sub_xscale=None,
        sub_yscale=None,
        x_type="%.1f",
        y_type="%.1f",
        x_grid="major",
        y_grid="major",  # minor
        grid_alpha=0.5,
        remove_borders=[],
        remove_ticks=['x', 'y']):
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

    # x, y range
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    for tick in remove_ticks:
        if 'x' == tick:
            ax.set_xticks([])
            ax.set_xticks([], minor=True)
        if 'y' == tick:
            ax.set_yticks([])
            ax.set_yticks([], minor=True)
    # open grid
    if x_grid is not None:
        # The grid of the x-axis uses the major ticks
        ax.xaxis.grid(True,
                      which=x_grid,
                      linestyle="--",
                      color='lightgrey',
                      alpha=grid_alpha)
    else:
        ax.xaxis.grid(False)
    if y_grid is not None:
        # The grid of the y-axis uses the minor scale
        ax.yaxis.grid(True,
                      which=y_grid,
                      linestyle="--",
                      color='lightgrey',
                      alpha=grid_alpha)

    else:
        ax.yaxis.grid(False)

    # remove top border
    for border in remove_borders:
        ax.spines[border].set_visible(False)

    return ax


def set_legend(ax, font_size=12, font_weight="normal"):

    handles, labels = plt.gca().get_legend_handles_labels()

    order = list(range(len(labels)))
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc='best',
              prop={
                  'size': font_size,
                  'weight': font_weight
              })
