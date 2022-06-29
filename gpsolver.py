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
F1 = matrix(np.eye(3))
F2 = matrix(-1 * np.eye(3))
F = sparse([[F1, F2]])
#F = matrix( [[-1., 0., 0., 1., 0., 0.],
#             [0., -1., 0., 0., 1., 0.],
#	     [0., 0., -1., 0., 0., 1.]])
#g = log( matrix( [1.0, 1.0, 1.0, 1.0, 1.0,1.0]) )
g = log(matrix(np.ones(6)))
test = 3
K = [2 * test]
G = None
h = None

A1 = matrix([[1.]])
A = matrix([[1.]])
for i in range(2):
    A = sparse([[A], [A1]])

#A = matrix([[1.], [1.], [1.]])
#print("!!", A)
#A = matrix(np.ones((3, 1), dtype=np.float64))

b = matrix([1.])
sol = solvers.gp(K, F, g, G, h, A, b, solver='mosek')['x']
print("sol: ", sol)
