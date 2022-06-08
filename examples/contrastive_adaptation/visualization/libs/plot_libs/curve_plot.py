import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition)
from mpl_toolkits.axes_grid1.inset_locator import TransformedBbox, BboxPatch, BboxConnector


def mark_inset(parent_axes,
               inset_axes,
               loc1a=1,
               loc1b=1,
               loc2a=2,
               loc2b=2,
               **kwargs):
    rect = TransformedBbox(inset_axes.viewLim, parent_axes.transData)

    pp = BboxPatch(rect, fill=False, **kwargs)
    parent_axes.add_patch(pp)

    p1 = BboxConnector(inset_axes.bbox, rect, loc1=loc1a, loc2=loc1b, **kwargs)
    inset_axes.add_patch(p1)
    p1.set_clip_on(False)
    p2 = BboxConnector(inset_axes.bbox, rect, loc1=loc2a, loc2=loc2b, **kwargs)
    inset_axes.add_patch(p2)
    p2.set_clip_on(False)

    return pp, p1, p2


def plot_curves(cur_ax, x_y_points, labels, colors, line_styles, line_widths):
    num_plots = len(x_y_points)
    for plot_idx in range(num_plots):
        x_points = x_y_points[plot_idx][0]
        y_points = x_y_points[plot_idx][1]
        label = labels[plot_idx]
        color = colors[plot_idx]
        line_sty = line_styles[plot_idx]
        line_width = line_widths[plot_idx]

        cur_ax.plot(
            x_points,
            y_points,
            label=label,
            color=color,
            #marker='o', markersize=5
            linestyle=line_sty,
            linewidth=line_width)


def plot_curves_with_inset(cur_ax,
                           x_y_points,
                           labels,
                           colors,
                           line_styles,
                           line_widths,
                           sub_region=[],
                           axes_info=[0.1, 0.1, 0.4, 0.4]):
    # Note: axes_info here defines the shape of the axes
    #   0.1, 0.1 means the left lower corner of the axes lies in the
    # 0.1 and 0.1 position of the orginal axes (cur_ax) this is
    # the relative position
    #   0.4 and 0.4 is the relative width and height

    num_plots = len(x_y_points)
    # Create a set of inset Axes: these should fill the bounding box allocated to
    # them.
    ax2 = plt.axes([0, 0, 1, 1])
    # Manually set the position and relative size of the inset axes within ax1
    ip = InsetPosition(cur_ax, axes_info)
    ax2.set_axes_locator(ip)
    # Mark the region corresponding to the inset axes on ax1 and draw lines
    # in grey linking the two axes.
    # the locs here define which corner of the axs figure connected to the anchor
    # 'upper right'  : 1, 'upper left'   : 2, 'lower left'   : 3, 'lower right'  : 4
    mark_inset(cur_ax,
               ax2,
               loc1a=1,
               loc1b=4,
               loc2a=2,
               loc2b=3,
               fc="none",
               ec="0.5")
    for plot_idx in range(num_plots):
        x_points = x_y_points[plot_idx][0]
        y_points = x_y_points[plot_idx][1]
        label = labels[plot_idx]
        color = colors[plot_idx]
        line_sty = line_styles[plot_idx]
        line_width = line_widths[plot_idx]

        cur_ax.plot(x_points,
                    y_points,
                    label=label,
                    color=color,
                    linestyle=line_sty,
                    linewidth=line_width)

        ax2.plot(x_points,
                 y_points,
                 color=color,
                 linestyle=line_sty,
                 linewidth=line_width)
    ax2.set_xlim(sub_region[0], sub_region[1])  # apply the x-limits
    ax2.set_ylim(sub_region[2], sub_region[3])  # apply the y-limits
