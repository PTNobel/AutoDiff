# auto_diff
An automatic differentiation library for Python+NumPy

## How To Use
There are four public elements of the API:

 * `AutoDiff` is a context manager and must be entered with a with statement.
The `__enter__` method returns a new version of x that must be used to instead of the x passed as a parameter to the `AutoDiff` constructor.

 * `value`, `jacobian`, `get_value_and_jacobian`, these functions, which must be
 called in an `AutoDiff` context, extract the value, Jacobian, or both from a
 dependent variable.

If you are using `get_value_and_jacobian`, x must be a 2D column vector, and
the value you must be parsing for the derivative must also be a 2D column
vector. In most other cases, how to convert to a Jacobian Matrix is
non-obvious. If you wish to deal with those cases see the paragraph after the
example.

### Example
```python
import auto_diff
import numpy as np

# Define a function f
# f can have other constant arguments, if they are constant wrt x
# Define the input vector, x

with auto_diff.AutoDiff(x) as x:
    f_eval = f(x, u)
    y, Jf = auto_diff.get_value_and_jacobian(f_eval)

# y is the value of f(x, u) and Jf is the Jacobian of f with respect to x.
```

We can also differentiate functions from arbitrarily shaped numpy arrays to
arbitrarily shaped outputs. Let `y = f(x)`, where `x` is a numpy array of shape
`x.shape`, and `y` is is the output of the function we wish to differentiate, `f`.

We can then access a numpy array of shape `(*y.shape, *x.shape)`, by accessing
`y.der`. This represents the gradients of each component of `y` with respect to
`x`. To find the gradient of the norm of a vector x, for example one can do

```python
import auto_diff
import numpy as np
x = np.array([[np.pi], [3.0], [17.0]])

with auto_diff.AutoDiff(x) as x:
    print(np.linalg.norm(x).der)
```

## Restrictions

* You must import numpy and use that object, rather then do something like
``from numpy import ...``, where ``...`` is either `*` or just function names.

Crashes, Bug Reports, and Feedback:
Email `parthnobel@berkeley.edu`


There are missing features right now. I'm working on them, feel free to email me
if you want something prioritized.

## Prerequisite
A version of NumPy >= 1.17 may be required. Bugs on older versions have always
raised errors, so there should be nothing to worry about.

Author: Parth Nobel (Github: /PTNobel, parthnobel@berkeley.edu)
Version: 0.3


