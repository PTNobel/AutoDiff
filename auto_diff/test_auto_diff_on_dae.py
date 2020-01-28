from . import AutoDiff, get_value_and_jacobian
import numpy as np


def assert_func(expect, actual):
    try:
        np.testing.assert_allclose(expect, actual, atol=1e-12)
    except AssertionError as e:
        print("Expected value of", expect)
        print("Got", actual)
        raise e


def test_auto_diff_on_DAE(DAE, x, u=None):
    disable_u = False
    if u is None:
        u = np.ndarray((0, 0))
        disable_u = True

    f_eval = DAE.f(x, u)
    df_dx_eval = DAE.df_dx(x, u)
    df_du_eval = DAE.df_du(x, u)
    q_eval = DAE.q(x, u)
    dq_dx_eval = DAE.dq_dx(x, u)
    dq_du_eval = DAE.dq_du(x, u)

    with AutoDiff(x) as x_vv:
        f_test, df_dx_test = get_value_and_jacobian(DAE.f(x_vv, u))
        q_test, dq_dx_test = get_value_and_jacobian(DAE.q(x_vv, u))
    assert_func(f_eval, f_test)
    assert_func(df_dx_eval, df_dx_test)
    assert_func(q_eval, q_test)
    assert_func(dq_dx_eval, dq_dx_test)

    if disable_u:
        print("Success")
        return
    with AutoDiff(u) as u_vv:
        f_test, df_du_test = get_value_and_jacobian(DAE.f(x, u_vv))
        q_test, dq_du_test = get_value_and_jacobian(DAE.q(x, u_vv))

    assert_func(f_eval, f_test)
    assert_func(df_du_eval, df_du_test)
    assert_func(q_eval, q_test)
    assert_func(dq_du_eval, dq_du_test)

    print("Success")
