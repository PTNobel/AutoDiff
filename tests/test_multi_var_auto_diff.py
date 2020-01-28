#!/usr/bin/python3

import auto_diff
import numpy as np


def test(f, x, u, df_dx, df_du):
    f_xu = f(x, u)

    with auto_diff.AutoDiff(x, u) as (x, u):
        y, (J_fx, J_fu) = auto_diff.get_value_and_jacobians(f(x, u))

    try:
        np.testing.assert_allclose(f_xu, y, atol=1e-12)
    except AssertionError as e:
        print("Expected value of", f_xu)
        print("Got", y)
        raise e

    try:
        np.testing.assert_allclose(df_dx, J_fx, atol=1e-12)
    except AssertionError as e:
        print("Expected derivative of", df_dx)
        print("Got", J_fx)
        raise e

    try:
        np.testing.assert_allclose(df_du, J_fu, atol=1e-12)
    except AssertionError as e:
        print("Expected derivative of", df_du)
        print("Got", J_fu)
        raise e
    print("Test Succeeded")


A = np.array([[5, 6., 3., 1.],
              [2, 3.,  5,  4],
              [np.pi, np.pi/2, np.e, np.exp(2)]])

B = np.array([[4, 2., 1.5],
              [.25, 2.5,  9],
              [np.e, 0.0, np.exp(0.5)]])

x = np.array([[.6, .8, .3, .4]]).T
u = np.array([[.2, 8.3, .5]]).T

test(lambda x, u: A @ x + B @ u, x, u, A, B)
