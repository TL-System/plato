import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter  # useful for `logit` scale

# Fixing random state for reproducibility
np.random.seed(19680801)

# make up some data in the interval ]0, 1[


x = np.arange(0, 1, 0.01)

y = 1 / x + 10 * x


# plot with various axes scales
plt.figure()

# linear
plt.plot(x, y)
plt.yscale("linear")
# plt.title("geometric")
plt.grid(True)


# Format the minor tick labels of the y-axis into empty strings with
# `NullFormatter`, to avoid cumbering the axis with too many labels.
plt.gca().yaxis.set_minor_formatter(NullFormatter())
# Adjust the subplot layout, because the logit one may take more space
# than usual, due to y-tick labels like "1 - 10^{-3}"
plt.subplots_adjust(
    top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35
)
plt.arrow(
    0.4,
    20,
    -0.059,
    -12,
    length_includes_head=True,
    head_width=0.04,
    head_length=2,
    fc="r",
    ec="r",
)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig("geometric_opt_2d.pdf")
