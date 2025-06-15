import cvxpy as cp
from cvxpy.error import SolverError
import numpy as np
import sys

import math

import jax
from jax import numpy as jnp


@jax.jit
def barrier_filter_linear(grad_x, B0, c):
    p = grad_x.T @ B0
    return -c * p / (jnp.dot(p, p))


def barrier_filter_quadratic_two(P, p, c, initialize, control_bias_term=np.zeros((2,))):
    def is_neg_def(x):
        # Check if a matrix is PSD
        return np.all(np.real(np.linalg.eigvals(x)) < 0)

    # CVX faces numerical difficulties otherwise
    check_nd = is_neg_def(P)

    # Check if P is PD
    if(check_nd):
        u = cp.Variable((2))
        u.value = np.array(initialize)
        P = np.array(P)
        p = np.array(p)

        prob = cp.Problem(cp.Minimize(1.0 * cp.square(u[0] + control_bias_term[0]) + 1.0 * cp.square(u[1] + control_bias_term[1])),
                          [cp.quad_form(u, P) + p.T @ u + c >= 0])
        try:
            prob.solve(verbose=False, warm_start=True)
        except SolverError:
            pass

    if(not check_nd or u[0] is None or prob.status not in ["optimal", "optimal_inaccurate"]):
        u = cp.Variable((2))
        u.value = np.array(initialize)
        p = np.array(p)
        prob = cp.Problem(cp.Minimize(1.0 * cp.square(u[0]) + 1.0 * cp.square(u[1])),
                          [p @ u + c >= 0])
        try:
            prob.solve(verbose=False, warm_start=True)
        except SolverError:
            pass

    if prob.status not in ["optimal", "optimal_inaccurate"] or u[0] is None:
        print("Quadratic Solver failed.")
        return np.array([0., 0.])
    return np.array([u[0].value, u[1].value])

def barrier_filter_quadratic_eight(P, p, c, initialize, control_bias_term=np.zeros((8,))):
    def is_neg_def(x):
        # Check if a matrix is PSD
        return np.all(np.real(np.linalg.eigvals(x)) < 0)

    # CVX faces numerical difficulties otherwise
    check_nd = is_neg_def(P)

    # Check if P is PD
    if(check_nd):
        u = cp.Variable((8))
        u.value = np.array(initialize)
        P = np.array(P)
        p = np.array(p)

        prob = cp.Problem(cp.Minimize(1.0 * cp.square(u[0] + control_bias_term[0]) + 1.0 * cp.square(u[1] + control_bias_term[1])
                                      + 1.0 * cp.square(u[2] + control_bias_term[2]) + 1.0 * cp.square(u[3] + control_bias_term[3])
                                      + 1.0 * cp.square(u[4] + control_bias_term[4]) + 1.0 * cp.square(u[5] + control_bias_term[5])
                                      + 1.0 * cp.square(u[6] + control_bias_term[6]) + 1.0 * cp.square(u[7] + control_bias_term[7])),
                          [cp.quad_form(u, P) + p.T @ u + c >= 0])
        try:
            prob.solve(verbose=False, warm_start=True)
        except SolverError:
            pass

    if(not check_nd or u[0] is None or prob.status not in ["optimal", "optimal_inaccurate"]):
        u = cp.Variable((8))
        u.value = np.array(initialize)
        p = np.array(p)
        prob = cp.Problem(cp.Minimize(1.0 * cp.square(u[0] + control_bias_term[0]) + 1.0 * cp.square(u[1] + control_bias_term[1])
                                      + 1.0 * cp.square(u[2] + control_bias_term[2]) + 1.0 * cp.square(u[3] + control_bias_term[3])
                                      + 1.0 * cp.square(u[4] + control_bias_term[4]) + 1.0 * cp.square(u[5] + control_bias_term[5])
                                      + 1.0 * cp.square(u[6] + control_bias_term[6]) + 1.0 * cp.square(u[7] + control_bias_term[7])),
                          [p @ u + c >= 0])
        try:
            prob.solve(verbose=False, warm_start=True)
        except SolverError:
            pass

    if prob.status not in ["optimal", "optimal_inaccurate"] or u[0] is None:
        return np.array([0., 0., 0., 0., 0., 0., 0., 0.])
    return np.array([u[0].value, u[1].value, u[2].value, u[3].value, u[4].value, u[5].value, u[6].value, u[7].value])
