import auto_diff
import numpy as np


def test(f, x, df_dx, name=None):
    input_x = x
    f_x = f(x)
    if name is None:
        name = str(f)
    print("Testing ", name, "on x = \n", x)
    with auto_diff.AutoDiff(x) as x:
        y, Jf = auto_diff.get_value_and_jacobian(f(x))
    try:
        np.testing.assert_allclose(f_x, y, atol=1e-12)
    except AssertionError as e:
        print("Expected value of", f_x)
        print("Got", y)
        raise e

    try:
        np.testing.assert_allclose(df_dx, Jf, atol=1e-12)
    except AssertionError as e:
        print("Expected derivative of", df_dx)
        print("Got", Jf)
        raise e

    print("Passed.")

    # Some bugs only appeared with rectangular Jacobians.
    print("Testing affine transform of", name)
    A = np.random.rand(input_x.shape[0], 3 * input_x.shape[0])
    b = np.random.rand(input_x.shape[0], 1)
    # x = np.linalg.lstsq(A, input_x - b,
    #                    rcond=None)[-1].reshape((3 * input_x.shape[0], 1))
    x = np.linalg.lstsq(A, input_x - b, rcond=None)[0]
    df_dx = df_dx @ A
    with auto_diff.AutoDiff(x) as x:
        y, Jf = auto_diff.get_value_and_jacobian(f(A @ x + b))

    try:
        np.testing.assert_allclose(f_x, y, atol=1e-12)
    except AssertionError as e:
        print("Expected value of", f_x)
        print("Got", y)
        raise e
    try:
        np.testing.assert_allclose(df_dx, Jf, atol=1e-12)
    except AssertionError as e:
        print("Expected derivative of", df_dx)
        print("Got", Jf)
        raise e

    print("Passed.")


def f(x):
    y = np.sqrt(x)
    out = np.ndarray((3, 1))
    np.add(x, y, out=out)
    return out
x = np.array([[2.], [4.], [9.0]])
df_dx = np.array([[1 + 0.5 / np.sqrt(2.), 0.0, 0.0],
                  [0.0, 1 + 1./4., 0.0],
                  [0.0, 0.0, 1 + 1./6.]])
test(f, x, df_dx, "test np.add(..., out=...)")


def f(x):
    y = np.sqrt(x)
    out = np.ndarray((3, 1))
    np.multiply(x, y, out=out)
    return out
x = np.array([[2.], [4.], [9.0]])
df_dx = np.array([[np.sqrt(2) + 1 / np.sqrt(2.), 0.0, 0.0],
                  [0.0, 2 + 4 * 1./4., 0.0],
                  [0.0, 0.0, 3 + 9 * 1./6.]])
test(f, x, df_dx, "test np.multiply(..., out=...)")

f = np.abs
x = np.array([[2.], [-2.], [0.0]])
df_dx = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]])
# x = np.array([[2.], [-2.], [4.0]])
# df_dx = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
test(f, x, df_dx)

f = np.sqrt
x = np.array([[2.], [4.], [9.0]])
df_dx = np.array([[0.5 / np.sqrt(2.), 0.0, 0.0],
                  [0.0, 1./4., 0.0],
                  [0.0, 0.0, 1./6.]])
# x = np.array([[2.], [-2.], [4.0]])
# df_dx = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
test(f, x, df_dx)

f = np.sin
x = np.array([[np.pi], [-np.pi/2], [np.pi/4]])
df_dx = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0, 0, np.sqrt(2) / 2]])
test(f, x, df_dx)

# Testing transpose requires accessing internals as it enforces the output
# being a column vector
print("TODO: Write a test of transpose")
# f = lambda x: x.T
# x = np.array([[np.pi], [-np.pi/2], [np.pi/4]])
# df_dx = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0, 0, 1.0]])
# test(f, x, df_dx, 'transpose')

f = np.cos
x = np.array([[np.pi], [-np.pi/2], [np.pi/4]])
df_dx = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0, 0, -np.sqrt(2) / 2]])
test(f, x, df_dx)


f = np.tan
x = np.array([[np.pi], [-np.pi/3], [np.pi/4]])
df_dx = np.array([[1.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0, 0, 2.0]])
test(f, x, df_dx)

f = np.arcsin
x = np.array([[0], [np.sqrt(2)/2], [1/2]])
df_dx = np.array([[1.0, 0.0, 0.0],
                  [0.0, np.sqrt(2), 0.0],
                  [0, 0, 2 / np.sqrt(3)]])
test(f, x, df_dx)

f = np.arccos
x = np.array([[0], [np.sqrt(2)/2], [1/2]])
df_dx = np.array([[-1.0, 0.0, 0.0],
                  [0.0, -np.sqrt(2), 0.0],
                  [0, 0, -2 / np.sqrt(3)]])
test(f, x, df_dx)


f = np.arctan
x = np.array([[-1.0], [99999], [1.0]])
df_dx = np.array([[0.5, 0.0, 0.0],
                  [0.0, 1.0002e-10, 0.0],
                  [0, 0, 1/2]])
test(f, x, df_dx)


f = np.log
x = np.array([[1.0], [0.5], [2.5]])
df_dx = np.diag([1.0, 2, .4])
test(f, x, df_dx)


f = np.log2
x = np.array([[1.0], [0.5], [2.5]])
df_dx = np.diag([1.0, 2, .4]) / np.log(2)
test(f, x, df_dx)


f = np.log10
x = np.array([[1.0], [0.5], [2.5]])
df_dx = np.diag([1.0, 2, .4]) / np.log(10)
test(f, x, df_dx)


f = np.log1p
x = np.array([[1.0], [-0.5], [1.5]])
df_dx = np.diag([.5, 2, .4])
test(f, x, df_dx)


f = np.negative
x = np.array([[1.0], [-0.5], [1.5]])
df_dx = -np.eye(3)
test(f, x, df_dx)


f = np.positive
x = np.array([[1.0], [-0.5], [1.5]])
df_dx = np.eye(3)
test(f, x, df_dx)


def f(x):
    x_1, x_2, x_3 = x
    return np.array([x_1 + x_2 + x_3])


x = np.array([[-1.0], [2.0], [3.0]])
df_dx = np.array([[1, 1, 1]])
test(f, x, df_dx, "sum of x")


def f(x):
    x_1, x_2, x_3 = x
    return np.array([x_1 - x_2 - 2 * x_3])


x = np.array([[-1.0], [2.0], [3.0]])
df_dx = np.array([[1, -1, -2]])
test(f, x, df_dx, "subtraction and scalar multiplication")


def f(x):
    x_1, x_2, x_3 = x
    return np.array([x_1 * x_2 - 2. * x_3 - x_1 * 3.,
                     x_2 / x_3 - x_2 / 2. + 3. / x_3])


x = np.array([[-1.0], [6.0], [3.0]])
df_dx = np.array([[3.0, -1, -2], [0, .3333333333 - 0.5, -6 / 9.0 - 1 / 3.0]])
test(f, x, df_dx, "subtraction, addition, multiplication, division")


def f(x):
    x_1, x_2 = x
    return np.array([x_1**2., np.e**x_2, x_1**x_2])


x = np.array([[3.0], [3.0]])
df_dx = np.array([[6.0, 0.0], [0.0, np.exp(3)], [27.0, 27.0 * np.log(3)]])


def f(x):
    return np.array([[0], [1], [2.0]])


x = np.array([[2.0]])
df_dx = np.array([[0], [0], [0.0]])
test(f, x, df_dx, "constant function")


A = np.array([[1.0, 4.0, 7.0], [5.0, 7.0, -200]])
x = np.array([[2.0], [3.0], [-4.0]])
test(lambda x: A @ x, x, A, "linear function")


A = np.array([[1.0, 4.0, 7.0], [5.0, 7.0, -200]])
b = np.array([[3.0], [-np.pi]])
x = np.array([[2.0], [3.0], [-4.0]])
test(lambda x: A @ x + b, x, A, "affine function")


A = np.array([[1.0, -2.0, 7.0], [5.0, 7.0, 1]])
b = np.array([[48.0], [-8.0]])
x = np.array([[2.0], [1.0], [-7.0]])
k = A @ x + b
[y_1], [y_2] = np.exp(k)
df_dx = np.diag([y_1, y_2]) @ A
test(lambda x: np.exp(A @ x + b), x, df_dx, "exp of affine function")
