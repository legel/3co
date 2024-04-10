import numpy as np
from qpsolvers import solve_qp


# 0.5 x^T P x + q^Tx, s.t. Gx <= h, Ax=b, lb <= x <= up
# v : view point

v = np.array([-8., 3., 1.])

triangle = np.array([
    [3., 0., 0.],
    [4., 0., 0.],
    [4., 2., 0.]
])
a = triangle[0]
b = triangle[1]
c = triangle[2]

# P
P = np.array([
    [a**2, 1.,   1.],
    [0.,   b**2, 0.],
    [0.,   0.,   c**2]
])
print(P)

# q
q = np.array([-2.0 * a * v[0], -2.0*b*v[1], -2.0*c*v[2]])


# Ax=b
A = np.array([
    [1., 1., 1.],
    [0., 0., 0.],
    [0., 0., 0.]
])

b = np.array([1.])

# Gx <= h
G = np.array([
    [-1., 0.,  0.],
    [ 0., -1., 0.],
    [ 0., 0.,  -1.]
])

h = np.array([0., 0., 0.,])


x = solve_qp(P, q, G, h, A, b, solver="osqp")

print(x)