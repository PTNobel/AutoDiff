import unittest

import auto_diff
import numpy as np

class AutoDiffUnitTesting(unittest.TestCase):
    def _assertAllClose(self, actual, desired, rtol=1e-07, atol=1e-12, equal_nan=True):
        np.testing.assert_allclose(actual, desired, rtol, atol, equal_nan)


class TestSingleVariableSparseAutoDiff(AutoDiffUnitTesting):
    def _test_helper(self, f, x, df_dx, debug=False):
        if debug:
            breakpoint()

        input_x = x
        f_x = f(input_x)
        with auto_diff.SparseAutoDiff(input_x) as x:
            y, Jf = auto_diff.get_value_and_jacobian(f(x))

        self._assertAllClose(y, f_x)
        self._assertAllClose(Jf, df_dx)
    
        # Some bugs only appeared with rectangular Jacobians.
        A = np.random.rand(input_x.shape[0], 3 * input_x.shape[0])
        b = np.random.rand(input_x.shape[0], 1)
        x = np.linalg.lstsq(A, input_x - b, rcond=None)[0]

        df_dx = df_dx @ A

        with auto_diff.SparseAutoDiff(x) as x:
            y, Jf = auto_diff.get_value_and_jacobian(f(A @ x + b))
    
        self._assertAllClose(y, f_x)
        self._assertAllClose(Jf, df_dx)

    def _test_out(self, f, x, df_dx, debug=False):
        if debug:
            breakpoint()
        input_x = x
        f_x = f(input_x)

        with auto_diff.SparseAutoDiff(input_x) as x:
            out_dest = np.ndarray(f_x.shape)
            f(x, out=out_dest)
            y, Jf = auto_diff.get_value_and_jacobian(out_dest)

        self._assertAllClose(f_x, y)
        self._assertAllClose(Jf, df_dx)

    def test_add_with_out(self):
        def f(x):
            y = np.sqrt(x)
            out = np.ndarray((3, 1))
            np.add(x, y, out=out)
            return out
        x = np.array([[2.], [4.], [9.0]])
        df_dx = np.array([[1 + 0.5 / np.sqrt(2.), 0.0, 0.0],
                            [0.0, 1 + 1./4., 0.0],
                            [0.0, 0.0, 1 + 1./6.]])
        self._test_helper(f, x, df_dx)

    def test_multiply_with_out(self):
        def f(x):
            y = np.sqrt(x)
            out = np.ndarray((3, 1))
            np.multiply(x, y, out=out)
            return out
        x = np.array([[2.], [4.], [9.0]])
        df_dx = np.array([[np.sqrt(2) + 1 / np.sqrt(2.), 0.0, 0.0],
                        [0.0, 2 + 4 * 1./4., 0.0],
                        [0.0, 0.0, 3 + 9 * 1./6.]])
        self._test_helper(f, x, df_dx)
    
    def test_abs(self):
        f = np.abs
        x = np.array([[2.], [-2.], [0.0]])
        df_dx = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]])
        # x = np.array([[2.], [-2.], [4.0]])
        # df_dx = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
        with self.assertWarns(UserWarning, msg='abs of a near-zero number, derivative is ill-defined'):
            self._test_helper(f, x, df_dx)
            self._test_out(f, x, df_dx)

    def test_sqrt(self):
        f = np.sqrt
        x = np.array([[2.], [4.], [9.0]])
        df_dx = np.array([[0.5 / np.sqrt(2.), 0.0, 0.0],
                        [0.0, 1./4., 0.0],
                        [0.0, 0.0, 1./6.]])
        # x = np.array([[2.], [-2.], [4.0]])
        # df_dx = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
        self._test_helper(f, x, df_dx)
        self._test_out(f, x, df_dx)
        
    def test_sin(self):
        f = np.sin
        x = np.array([[np.pi], [-np.pi/2], [np.pi/4]])
        df_dx = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0, 0, np.sqrt(2) / 2]])
        self._test_helper(f, x, df_dx)
        self._test_out(f, x, df_dx)
        
       
    def test_cos(self):
        f = np.cos
        x = np.array([[np.pi], [-np.pi/2], [np.pi/4]])
        df_dx = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0, 0, -np.sqrt(2) / 2]])
        self._test_helper(f, x, df_dx)
        self._test_out(f, x, df_dx)
        
    def test_tan(self):
        f = np.tan
        x = np.array([[np.pi], [-np.pi/3], [np.pi/4]])
        df_dx = np.array([[1.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0, 0, 2.0]])
        self._test_helper(f, x, df_dx)
        self._test_out(f, x, df_dx)

    def test_tanh(self):
        f = np.tanh
        x = np.array([[np.log(2)], [-np.log(3)], [0.0]])
        df_dx = np.array([[0.64, 0.0, 0.0], [0.0, 0.36, 0.0], [0, 0, 1.0]])
        self._test_helper(f, x, df_dx)
        self._test_out(f, x, df_dx)

    def test_sinh(self):
        f = np.sinh
        x = np.array([[np.log(2)], [-np.log(3)], [0.0]])
        df_dx = np.array([[1.25, 0.0, 0.0], [0.0, 5 / 3, 0.0], [0, 0, 1.0]])
        self._test_helper(f, x, df_dx)
        self._test_out(f, x, df_dx)

    def test_cosh(self):
        f = np.cosh
        x = np.array([[np.log(2)], [-np.log(3)], [0.0]])
        df_dx = np.array([[2.25/3, 0.0, 0.0], [0.0, -4/3, 0.0], [0, 0, 0.0]])
        self._test_helper(f, x, df_dx)
        self._test_out(f, x, df_dx)

    def test_arctanh(self):
        f = np.arctanh
        x = np.array([[np.sqrt(1/4)], [0.5], [0.0]])
        df_dx = np.array([[4/3, 0.0, 0.0], [0.0, 1/(1 - 0.5**2), 0.0], [0, 0, 1.0]])
        self._test_helper(f, x, df_dx)
        self._test_out(f, x, df_dx)

    def test_arccosh(self):
        f = np.arccosh
        x = np.array([[np.sqrt(5)], [np.sqrt(10)], [np.sqrt(17)]])
        df_dx = np.array([[1/2, 0.0, 0.0], [0.0, 1/3, 0.0], [0, 0, 1.0/4]])
        self._test_helper(f, x, df_dx)
        self._test_out(f, x, df_dx)

    def test_arcsinh(self):
        f = np.arcsinh
        x = np.array([[np.sqrt(3)], [np.sqrt(8)], [np.sqrt(15)]])
        df_dx = np.array([[1/2, 0.0, 0.0], [0.0, 1/3, 0.0], [0, 0, 1.0/4]])
        self._test_helper(f, x, df_dx)
        self._test_out(f, x, df_dx)

    def test_arcsin(self):
        f = np.arcsin
        x = np.array([[0], [np.sqrt(2)/2], [1/2]])
        df_dx = np.array([[1.0, 0.0, 0.0],
                        [0.0, np.sqrt(2), 0.0],
                        [0, 0, 2 / np.sqrt(3)]])
        self._test_helper(f, x, df_dx)
        self._test_out(f, x, df_dx)
    
    def test_arccos(self):
        f = np.arccos
        x = np.array([[0], [np.sqrt(2)/2], [1/2]])
        df_dx = np.array([[-1.0, 0.0, 0.0],
                        [0.0, -np.sqrt(2), 0.0],
                        [0, 0, -2 / np.sqrt(3)]])
        self._test_helper(f, x, df_dx)
        self._test_out(f, x, df_dx)
        
    def test_arctan(self):
        f = np.arctan
        x = np.array([[-1.0], [99999], [1.0]])
        df_dx = np.array([[0.5, 0.0, 0.0],
                        [0.0, 1.0002e-10, 0.0],
                        [0, 0, 1/2]])
        self._test_helper(f, x, df_dx)
        self._test_out(f, x, df_dx)
        
    def test_log(self):
        f = np.log
        x = np.array([[1.0], [0.5], [2.5]])
        df_dx = np.diag([1.0, 2, .4])
        self._test_helper(f, x, df_dx)
        self._test_out(f, x, df_dx)
        
    def test_log2(self):
        f = np.log2
        x = np.array([[1.0], [0.5], [2.5]])
        df_dx = np.diag([1.0, 2, .4]) / np.log(2)
        self._test_helper(f, x, df_dx)
        self._test_out(f, x, df_dx)
        
    def test_log10(self):
        f = np.log10
        x = np.array([[1.0], [0.5], [2.5]])
        df_dx = np.diag([1.0, 2, .4]) / np.log(10)
        self._test_helper(f, x, df_dx)
        self._test_out(f, x, df_dx)
        
    def test_log1p(self):
        f = np.log1p
        x = np.array([[1.0], [-0.5], [1.5]])
        df_dx = np.diag([.5, 2, .4])
        self._test_helper(f, x, df_dx)
        self._test_out(f, x, df_dx)
        
    def test_negative(self):
        f = np.negative
        x = np.array([[1.0], [-0.5], [1.5]])
        df_dx = -np.eye(3)
        self._test_helper(f, x, df_dx)
        self._test_out(f, x, df_dx)
        
    def test_positive(self):
        f = np.positive
        x = np.array([[1.0], [-0.5], [1.5]])
        df_dx = np.eye(3)
        self._test_helper(f, x, df_dx)
        self._test_out(f, x, df_dx)
        
    def test_decomposing_x(self):
        def f(x):
            x_1, x_2, x_3 = x
            return np.array([x_1 + x_2 + x_3])

        x = np.array([[-1.0], [2.0], [3.0]])
        df_dx = np.array([[1, 1, 1]])
        self._test_helper(f, x, df_dx)

        def f(x):
            x_1, x_2, x_3 = x
            return np.array([x_1 - x_2 - 2 * x_3])

        x = np.array([[-1.0], [2.0], [3.0]])
        df_dx = np.array([[1, -1, -2]])
        self._test_helper(f, x, df_dx)

        def f(x):
            x_1, x_2, x_3 = x
            return np.array([x_1 * x_2 - 2. * x_3 - x_1 * 3.,
                            x_2 / x_3 - x_2 / 2. + 3. / x_3])
            

        x = np.array([[-1.0], [6.0], [3.0]])
        df_dx = np.array([[3.0, -1, -2], [0, .3333333333 - 0.5, -6 / 9.0 - 1 / 3.0]])
        self._test_helper(f, x, df_dx)

        def f(x):
            x_1, x_2 = x
            return np.array([x_1**2., np.e**x_2, x_1**x_2])

        x = np.array([[3.0], [3.0]])
        df_dx = np.array([[6.0, 0.0], [0.0, np.exp(3)], [27.0, 27.0 * np.log(3)]])

    def test_constant(self):
        def f(x):
            return np.array([[0], [1], [2.0]])

        x = np.array([[2.0]])
        df_dx = np.array([[0], [0], [0.0]])
        self._test_helper(f, x, df_dx)

    def test_matrixmul(self):
        A = np.array([[1.0, 4.0, 7.0], [5.0, 7.0, -200]])
        x = np.array([[2.0], [3.0], [-4.0]])
        self._test_helper(lambda x: A @ x, x, A)

    def test_affine(self):
        A = np.array([[1.0, 4.0, 7.0], [5.0, 7.0, -200]])
        b = np.array([[3.0], [-np.pi]])
        x = np.array([[2.0], [3.0], [-4.0]])
        self._test_helper(lambda x: A @ x + b, x, A)

    def test_exp_of_affine(self):
        A = np.array([[1.0, -2.0, 7.0], [5.0, 7.0, 1]])
        b = np.array([[48.0], [-8.0]])
        x = np.array([[2.0], [1.0], [-7.0]])
        k = A @ x + b
        [y_1], [y_2] = np.exp(k)
        df_dx = np.diag([y_1, y_2]) @ A
        self._test_helper(lambda x: np.exp(A @ x + b), x, df_dx)

    def test_assign_scalar(self):
        def f(x):
            C = 1.0e-7;
            retval = C * x
            for i in range(3):
                retval[i] = 0
            return retval
        
        x = np.array([[4.0], [3.0], [6.0], [7.0]])
        df_dx = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0, 0], [0, 0, 0, 0], [0,0, 0, 1.0e-7]])
        self._test_helper(f, x, df_dx)

    def test_assign_vector(self):
        def f(x, u):
            C = 1.0e-7;
            retval = C * x
            for i in range(3):
                retval[i] = u
            return retval
        
        x = np.array([[4.0], [3.0], [6.0], [7.0]])
        df_dx = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0, 0], [0, 0, 0, 0], [0,0, 0, 1.0e-7]])
        u = np.array([[1.0]])
        self._test_helper(lambda x: f(x, u), x, df_dx)




class TestMultipleVariableSparseAutoDiff(AutoDiffUnitTesting):
    def _test_helper(self, f, x, u, df_dx, df_du, debug=False):
        if debug:
            breakpoint()

        f_xu = f(x, u)

        input_x = x
        input_u = u

        with auto_diff.SparseAutoDiff(x, u) as (x, u):
            y, (J_fx, J_fu) = auto_diff.get_value_and_jacobians(f(x, u))
        self._assertAllClose(y, f_xu)
        self._assertAllClose(J_fx, df_dx)
        self._assertAllClose(J_fu, df_du)

        u = input_u
        with auto_diff.SparseAutoDiff(input_x) as x:
            y, J_fx = auto_diff.get_value_and_jacobian(f(x, u))
        self._assertAllClose(y, f_xu)
        self._assertAllClose(J_fx, df_dx)

        x = input_x
        with auto_diff.SparseAutoDiff(input_u) as u:
            y, J_fu = auto_diff.get_value_and_jacobian(f(x, u))
        self._assertAllClose(y, f_xu)
        self._assertAllClose(J_fu, df_du)

        # Some bugs only appeared with rectangular Jacobians.
        A_x = np.random.rand(input_x.shape[0], 3 * input_x.shape[0])
        b_x = np.random.rand(input_x.shape[0], 1)
        affine_x = np.linalg.lstsq(A_x, input_x - b_x, rcond=None)[0]

        A_u = np.random.rand(input_u.shape[0], 3 * input_x.shape[0])
        b_u = np.random.rand(input_u.shape[0], 1)
        affine_u = np.linalg.lstsq(A_u, input_u - b_u, rcond=None)[0]

        df_dx = df_dx @ A_x
        df_du = df_du @ A_u

        with auto_diff.SparseAutoDiff(affine_x, affine_u) as (x, u):
            y, (J_fx, J_fu) = auto_diff.get_value_and_jacobians(f(A_x @ x + b_x, A_u @ u + b_u))
    
        self._assertAllClose(y, f_xu)
        self._assertAllClose(J_fx, df_dx)
        self._assertAllClose(J_fu, df_du)

        with auto_diff.SparseAutoDiff(affine_x) as x:
            y, J_fx = auto_diff.get_value_and_jacobian(f(A_x @ x + b_x, A_u @ affine_u + b_u))
    
        self._assertAllClose(y, f_xu)
        self._assertAllClose(J_fx, df_dx)

        with auto_diff.SparseAutoDiff(affine_u) as u:
            y, J_fu = auto_diff.get_value_and_jacobian(f(A_x @ affine_x + b_x, A_u @ u + b_u))
    
        self._assertAllClose(y, f_xu)
        self._assertAllClose(J_fu, df_du)

    def test_linear(self):
        A = np.array([[5, 6., 3., 1.],
                      [2, 3.,  5,  4],
                      [np.pi, np.pi/2, np.e, np.exp(2)]])

        B = np.array([[4, 2., 1.5],
                      [.25, 2.5,  9],
                      [np.e, 0.0, np.exp(0.5)]])

        x = np.array([[.6, .8, .3, .4]]).T
        u = np.array([[.2, 8.3, .5]]).T

        self._test_helper(lambda x, u: A @ x + B @ u, x, u, A, B)

    def test_add_and_mulitply(self):
        def f(x, u):
            A = np.array([[2., -1], [-1, 2]])
            B = np.array([[1.] , [0]])
            return A @ x + B * u

        x = np.array([[5.], [3.]])
        u = np.array([[3.]])
        self._test_helper(f, x, u, np.array([[2, -1], [-1, 2.]]), np.array([[1], [0.]]))

    def test_addition_broadcasting_and_reshape(self):
        def f(x, u):
            y = x.T + u
            return y.reshape((6, 1))

        x = np.array([[5.], [2.]])
        u = np.array([[3.], [7.], [11.]])
        df_dx = np.array([[1., 0.],
                          [0., 1.],
                          [1., 0.],
                          [0., 1.],
                          [1., 0.],
                          [0., 1.]])
        
        df_du = np.array([[1., 0., 0.],
                          [1., 0., 0.],
                          [0., 1., 0.],
                          [0., 1., 0.],
                          [0., 0., 1.],
                          [0., 0., 1.]])
        self._test_helper(f, x, u, df_dx, df_du)

    def test_subtraction_broadcasting_and_reshape(self):
        def f(x, u):
            y = np.subtract(x.T, u)
            return y.reshape((6, 1))

        x = np.array([[5.], [2.]])
        u = np.array([[3.], [7.], [11.]])
        df_dx = np.array([[1., 0.],
                          [0., 1.],
                          [1., 0.],
                          [0., 1.],
                          [1., 0.],
                          [0., 1.]])
        df_du = np.array([[-1., 0., 0.],
                          [-1., 0., 0.],
                          [0., -1., 0.],
                          [0., -1., 0.],
                          [0., 0., -1.],
                          [0., 0., -1.]])

        self._test_helper(f, x, u, df_dx, df_du)

    def test_division_broadcasting_and_reshape(self):
        def f(x, u):
            y = np.divide(x.T, u)
            return y.reshape((4, 1))

        x = np.array([[5.], [2.]])
        u = np.array([[3.], [11.]])
        df_dx = np.array([[1./3, 0.],
                          [0., 1./3],
                          [1./11, 0.],
                          [0., 1./11]])
        df_du = np.array([[-5/9., 0.],
                          [-2/9., 0.],
                          [0., -5/121.],
                          [0., -2/121.]])

        self._test_helper(f, x, u, df_dx, df_du)


    def test_float_power_broadcasting_and_reshape(self):
        def f(x, u):
            y = np.float_power(x.T, u)
            return y.reshape((4, 1))

        x = np.array([[5.], [2.]])
        u = np.array([[3.], [4.]])
        df_dx = np.array([[3 * 5.**2, 0.],
                          [0., 3 * 2.**2],
                          [4 * 5.**3, 0.],
                          [0., 4 * 2.**3]])
        df_du = np.array([[np.log(5) * 5**3, 0.],
                          [np.log(2) * 2**3, 0.],
                          [0., np.log(5) * 5**4],
                          [0., np.log(2) * 2**4]])

        self._test_helper(f, x, u, df_dx, df_du)

    def test_power_broadcasting_and_reshape(self):
        def f(x, u):
            y = np.power(x.T, u)
            return y.reshape((4, 1))

        x = np.array([[5.], [2.]])
        u = np.array([[3.], [4.]])
        df_dx = np.array([[3 * 5.**2, 0.],
                          [0., 3 * 2.**2],
                          [4 * 5.**3, 0.],
                          [0., 4 * 2.**3]])
        df_du = np.array([[np.log(5) * 5**3, 0.],
                          [np.log(2) * 2**3, 0.],
                          [0., np.log(5) * 5**4],
                          [0., np.log(2) * 2**4]])

        self._test_helper(f, x, u, df_dx, df_du)

class TestArbitraryShapeSparseAutoDiff(AutoDiffUnitTesting):
    def _test_helper(self, f, x, df_dx, debug=False):
        if debug:
            breakpoint()

        input_x = x
        f_x = f(input_x)
        with auto_diff.SparseAutoDiff(input_x) as x:
            ad_f_x = f(x)
            y, Jf = ad_f_x.val, ad_f_x.der

        self._assertAllClose(y, f_x)
        self._assertAllClose(Jf, df_dx)

    def _test_transpose(self):
        # Testing transpose requires accessing internals as it enforces the output
        # being a column vector
        print("TODO: Write a test of transpose")
        # f = lambda x: x.T
        # x = np.array([[np.pi], [-np.pi/2], [np.pi/4]])
        # df_dx = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0, 0, 1.0]])
        # test(f, x, df_dx, 'transpose')

    def _test_2d_matrix_1d_vector(self):
        A = np.array([[1, 5., 0.], [3., 6., 2.]])
        x = np.array([0., 1., 2.])
        self._test_helper(lambda x: A @ x, x, A)


if __name__ == '__main__':
    unittest.main()
