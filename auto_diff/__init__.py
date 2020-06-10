"""
An automatic differentiation library for Python+NumPy

# How To Use:
There are two public elements of the API, AutoDiff and `get_value_and_jacobian`
AutoDiff is a context manager and must be entered with a with statement.
The __enter__ method returns a new version of x that must be used to instead of
the x passed as a parameter to the AutoDiff constructor.

Finally while in the with block, you can call get_value_and_jacobian on the
output of a function to get the derivative. This may be called multiple times.

If you are using get_value_and_jacobian, x must be a 2D column vector, and
the value you must be parsing for the derivative must also be a 2D column
vector. In most other cases, how to convert to a Jacobian Matrix is
non-obvious. If you wish to deal with those cases see the paragraph after the
example.

Examples:
    import auto_diff
    import numpy as np

    # Define a function f(x), where you want f() differentiated wrt x
    # x and f() should be numpy arrays.
    # f() can have other arguments, eg, f(x, u)
    # Define the input vector, x, to be a numpy array.
    # Then, evaluate f() and its Jacobian, do:
    #   with auto_diff.AutoDiff(x) as x:
    #       f_eval = f(x, u)
    #       y, Jf = auto_diff.get_value_and_jacobian(f_eval)
    # y is the value of f(x, u) and Jf is the Jacobian of f with respect to x.

    # scalar function example
    def f(x):
        return(x*x);
    x = 3;
    with auto_diff.AutoDiff(x) as x:
        f_eval = f(x);
        y, Jf = auto_diff.get_value_and_jacobian(f_eval);

    # vector function example 1
    def f(x):
        retval = np.zeros((3,1));
        retval[0] = 1*x[0] + 2*x[1] + 3*x[2];
        retval[1] = 4*x[0] + 5*x[1] + 6*x[2];
        retval[2] = 7*x[0] + 8*x[1] + 9*x[2];
        return(retval);
    x = np.array([[1],[2],[3]]);
    with auto_diff.AutoDiff(x) as x:
        f_eval = f(x);
        y, Jf = auto_diff.get_value_and_jacobian(f_eval);

    # vector function example 2: f(x) = A x
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    def f(x):
        # return(A*x);
        return(A@x);
    x = np.array([[1],[2],[3]]);
    with auto_diff.AutoDiff(x) as x:
        f_eval = f(x);
        y, Jf = auto_diff.get_value_and_jacobian(f_eval);

We can also differentiate functions from arbitrarily shaped numpy arrays to
arbitrarily shaped outputs. Let y = f(x), where x is a numpy array of shape
x.shape, and y is is the output of the function we wish to differentiate, f.

We can then access a numpy array of shape (*y.shape, *x.shape), by accessing
y.der. This represents the gradients of each component of y with respect to x.
To find the gradient of the norm of a vector x, for example one can do

    import auto_diff
    import numpy as np
    x = np.array([[np.pi], [3.0], [17.0]])

    with auto_diff.AutoDiff(x) as x:
        print(np.linalg.norm(x).der)

# Restrictions

Restrictions that can be removed by emailing `parthnobel@berkeley.edu` with an
explanation of what you're doing and ideally a code snippet for testing.
- No kwargs are currently supported for any masked function or any function
    used to construct vectors. (Except out for np.add)
- Only the functions listed and documented in `masked_functions` are supported.
- When constructing an object that isn't a constant, you must construct the
    numpy array with one of:
    * zeros
    * eye
    * identity
    * ndarray
    * array
    * ones
    * ones_like
    * zeros_like
    * empty
    * empty_like
    * full
    * full_like
Don't stress too much about this, you'll almost certainly get an error if this
isn't the case, and for pure functions this shouldn't be an issue.

Restrictions:
* You must import numpy and use that object, rather then do something like
``from numpy import ...``, where ``...`` is either * or just function names.

Crashes, Bug Reports, and Feedback:
Email `parthnobel@berkeley.edu`

Prerequisite:
A version of NumPy >= 1.17 may be required. Bugs on older versions have always
raised errors, so there should be nothing to worry about.

Author: Parth Nobel (Github: /PTNobel, parthnobel@berkeley.edu)
Version: 0.2

"""

from . import true_np
from . import vecvalder_funcs_and_ufuncs as masked_functions
from . import sparsevecvalder_funcs_and_ufuncs as sparse_masked_functions

from .numpy_masking import AutoDiff
from .numpy_masking import _active_auto_diffs

from .sparse_numpy_masking import SparseAutoDiff

def get_value_and_jacobians(x):
    """Extracts values and jacobians from the value of a function,

    This only supports column vectors being differentiated
    with respect to a column vector.
    """
    import numpy as np
    if isinstance(x, true_np.ndarray):
        x = np.array(x)
    
    jacobians = []
    total = 0

    if isinstance(_active_auto_diffs[-1], SparseAutoDiff):
        for vec in _active_auto_diffs[-1].vecs:
            jacobians.append(x.der[:, total:total + np.size(vec)])
            total += np.size(vec)

        return x.val, tuple(jacobians)

    shared_axes = tuple(slice(None) for _ in x.val.shape)
    for vec in _active_auto_diffs[-1].vecs:
        idx = (*shared_axes, slice(total, total + vec.shape[0]), slice(None))
        jacobians.append(
            x.der[idx].reshape((x.val.shape[0], vec.shape[0])))
        total += vec.shape[0]

    return x.val, tuple(jacobians)


def get_value_and_jacobian(x):
    import numpy as np
    if isinstance(x, true_np.ndarray):
        x = np.array(x)
    elif isinstance(_active_auto_diffs[-1], SparseAutoDiff):
        return x.val, jacobian(x)
    if x.der.shape[-1] == 0:
        assert x.der.shape[2] == 1
        return x.val, true_np.ndarray((-1, 0))
    return x.val, jacobian(x)


def jacobian(x):
    import numpy as np
    if isinstance(x, true_np.ndarray):
        x = np.array(x)

    if isinstance(_active_auto_diffs[-1], SparseAutoDiff):
        return x.der

    if x.der.shape[-1] == 0:
        assert x.der.shape[2] == 1
        return true_np.ndarray((x.der.shape[0], 0))

    i, j, k, n = x.der.shape
    assert j == 1 == n
    return x.der.reshape((i, k))
