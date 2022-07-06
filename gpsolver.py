import mosek
import numpy as np
from cvxopt import matrix, log, exp, solvers, sparse
"""
This is an example for geomatrix solover with full parameters.
It's used to solve the following optimization questions.

Minimize 	q_1^{-1} + q_2^{-1} + q_1 + q_2

Subject to 	q_1 + q_2 = 1
	 	-q_1      < 0
		-q_2 	  < 0
"""

num = 5
alpha = 1
BigA = 1
aggre_weight = np.arange(num)  # p_i
local_gradient_bound = np.arange(num)  # G_i
local_staleness = np.arange(num)  # \tau_i

aggre_weight_square = np.square(aggre_weight)  # p_i^2
local_gradient_bound_square = np.square(local_gradient_bound)  # G_i^2

F1_params = alpha * np.multiply(aggre_weight_square,
                                local_gradient_bound_square)  # p_i^2 * G_i^2
F1 = matrix(np.eye(num) * F1_params)

F2_params = BigA * np.multiply(local_staleness,
                               local_gradient_bound)  # \tau_i * G_i
F2 = matrix(-1 * np.eye(num) * F2_params)
F = sparse([[F1, F2]])
#F = matrix( [[-1., 0., 0., 1., 0., 0.],
#             [0., -1., 0., 0., 1., 0.],
#	     [0., 0., -1., 0., 0., 1.]])
#g = log( matrix( [1.0, 1.0, 1.0, 1.0, 1.0,1.0]) )
g = log(matrix(np.ones(2 * num)))
K = [2 * num]
G = matrix(-1 * np.eye(num))  #None

h = matrix(np.zeros((5, 1)))  #None

A1 = matrix([[1.]])
A = matrix([[1.]])
for i in range(num - 1):
    A = sparse([[A], [A1]])

#A = matrix([[1.], [1.], [1.]])
#print("!!", A)
#A = matrix(np.ones((3, 1), dtype=np.float64))

b = matrix([1.])
sol = solvers.gp(K, F, g, G, h, A, b, solver='mosek')['x']
print("sol: ", sol)
