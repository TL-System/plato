# surface plot for 2d objective function
from numpy import arange
from numpy import meshgrid
import numpy
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

# objective function
def objective(x, y):

    result = x + 1.0 / x + 10 * y + 1 / y

    temp1 = numpy.tri(len(x), len(x)) * numpy.inf
    temp2 = numpy.tril(temp1)
    temp3 = numpy.rot90(temp2, k=1)
    temp4 = numpy.rot90(numpy.tri(len(x), len(x)), k=3)

    temp = temp4 + temp3

    print(result)
    return temp * result


def constraint(x):
    return 1 - x


# define range for input
r_min, r_max = 0.01, 1.01
# sample input range uniformly at 0.1 increments
xaxis = arange(r_min, r_max, 0.01)  # )
yaxis = arange(r_min, r_max, 0.01)  # 0.015)
# create a mesh from the axis
x, y = meshgrid(xaxis, yaxis)
# compute targets
results = objective(x, y)
# results2 = constraint(x, y)

xx, zz = numpy.meshgrid(xaxis, range(200))
yy = constraint(xx)


# zz = meshgrid(range(-1, 20, 1))

# create a surface plot with the jet color scheme
figure = pyplot.figure()
axis = figure.gca(projection="3d")

axis.plot_surface(x, y, results, cmap="rainbow")
axis.plot_surface(xx, yy, zz, alpha=0.5)
axis.view_init(30, -20)  # 190)

# pyplot.show()
# show the plot
pyplot.savefig("geometric_opt.pdf")
