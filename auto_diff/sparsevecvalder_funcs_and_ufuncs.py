from .sparsevecvalder import register, SparseVecValDer
import warnings
import scipy.sparse
from . import true_np as np
import numpy as modded_np

cls = SparseVecValDer


# On 5-Stage Ring Oscillator example from PyMAPP, using this instead of
# np.ndindex reduced a 200 s simulation to 150 s.
def _ndindex(shape):
    state = [0 for _ in shape]
    total = 0
    stop_condition = np.product(shape)
    while total < stop_condition:
        total += 1
        yield tuple(state)
        for i in range(-1, -1 -len(shape), -1):
            state[i] += 1
            if state[i] == shape[i]:
                state[i] = 0
            else:
                break


class ScalarAsArrayWrapper:
    __slots__ = 'scalar', 'shape', 'flat'

    def __init__(self, scalar):
        self.scalar = scalar
        self.shape = ()
        self.flat = self

    def __getitem__(self, idx):
        return self.scalar

    def __array_function__(self, func, types, args, kwargs):
        if func is np.size:
            return 1
        else:
            return NotImplemented


class PairedIterator:
    sentinel = object()

    __slots__ = 'left_iter', 'right_iter', '_left_value', '_right_value'

    def __init__(self, left_iter, right_iter):
        self.left_iter = left_iter
        self.right_iter = right_iter
        self._left_value = PairedIterator.sentinel
        self._right_value = PairedIterator.sentinel

    def advance_both(self):
        try:
            self.advance_left()
        finally:
            self.advance_right()

    @property
    def left_value(self):
        if self._left_value is PairedIterator.sentinel:
            self.advance_left()

        if self._left_value is PairedIterator.sentinel:
            raise StopIteration

        return self._left_value

    @property
    def right_value(self):
        if self._right_value is PairedIterator.sentinel:
            self.advance_right()

        if self._right_value is PairedIterator.sentinel:
            raise StopIteration

        return self._right_value

    def advance_left(self):
        try:
            self._left_value = next(self.left_iter)
        except StopIteration:
            self._left_value = PairedIterator.sentinel
            raise

    def advance_right(self):
        try:
            self._right_value = next(self.right_iter)
        except StopIteration:
            self._right_value = PairedIterator.sentinel
            raise

_empty_lil = scipy.sparse.lil_matrix([[]])
def _combine_lil_rows(combine_func, combine_left, combine_right, x, y, out_idx, out_data):
    paired_iter = PairedIterator(zip(x.rows[0], x.data[0]), zip(y.rows[0], y.data[0]))
    out_idx.clear()
    out_data.clear()
    try:
        while True:
            if paired_iter.left_value[0] == paired_iter.right_value[0]:
                out_idx.append(paired_iter.left_value[0])
                out_data.append(combine_func(paired_iter.left_value[1], paired_iter.right_value[1]))
                paired_iter.advance_both()
            elif paired_iter.left_value[0] < paired_iter.right_value[0]:
                out_idx.append(paired_iter.left_value[0])
                out_data.append(combine_left(paired_iter.left_value[1]))
                paired_iter.advance_left()
            elif paired_iter.left_value[0] > paired_iter.right_value[0]:
                out_idx.append(paired_iter.right_value[0])
                out_data.append(combine_right(paired_iter.right_value[1]))
                paired_iter.advance_right()
    except StopIteration:
        pass

    try:
        # consume left
        while True:
            out_idx.append(paired_iter.left_value[0])
            out_data.append(combine_left(paired_iter.left_value[1]))
            paired_iter.advance_left()
    except StopIteration:
        pass

    try:
        # consume right
        while True:
            out_idx.append(paired_iter.right_value[0])
            out_data.append(combine_right(paired_iter.right_value[1]))
            paired_iter.advance_right()
    except StopIteration:
        pass




class FeaturelessVecValDer:
    __slots__ = 'val', 'der'
    def __init__(self, val, der):
        self.val = val
        self.der = der
    """
    @property
    def val(self):
        return self.__val

    @property
    def der(self):
        return self.__der
    """

def _derivative_independent_comparison_ufunc(fn):
    def wrapped_function(x1, x2, **kwargs):
        if isinstance(x1, cls) and isinstance(x2, cls):
            return fn(x1.val, x2.val, **kwargs)
        elif isinstance(x1, cls):
            return fn(x1.val, x2, **kwargs)
        elif isinstance(x2, cls):
            return fn(x1, x2.val, **kwargs)
        else:
            raise RuntimeError("This should not be occuring.")

    wrapped_function.__name__ = fn.__name__
    return wrapped_function


_none_vec_val_der = FeaturelessVecValDer(None, None)


def _simple_index_to_square_index(i, shape):
    backwards_out = list()
    for axis_dim in reversed(shape):
        backwards_out.append(i % axis_dim)
        i //= axis_dim

    return tuple(reversed(backwards_out))


def _square_index_to_simple_index(idx, shape):
    assert len(idx) == len(shape)
    out = 0
    multiplier = 1
    for i, axis_dim in zip(reversed(idx), reversed(shape)):
        out += i * multiplier
        multiplier *= axis_dim
    return out


# This is probably a good C/Cython candidate
def _chain_rule(f_x, dx, out=None):
    r"""This implements the chain rule for the elementwise application  of
        f: R -> R
    Meant to implement for f: R -> R, g: R^n -> R, x \in R^n
    \nabla f(g(x)) = f'(g(x)) \nabla g(x)
    """
    if out is None:
        out = scipy.sparse.lil_matrix(dx.shape)
    # we're about to overwrite each element. If we do compression of the for
    # loop in the future be sure to switch to np.zeros.
    flat_f_x = f_x.flat
    for i, (f_x_i, grad_of_x, idx_of_grad) in enumerate(zip(flat_f_x, dx.data, dx.rows)):
        for j, partial_x in zip(idx_of_grad, grad_of_x):
            out[i, j] = f_x_i * partial_x

    return out


# Add support for where=
def _add_out_support(mutable_out=False):
    def decorator(fn):
        def fn_with_out(*args, out=None, where=True, **kwargs):
            if where is not True:
                raise NotImplementedError("Have yet to implement where support")
            if out is None and not mutable_out:
                out = _none_vec_val_der
            elif out is None and mutable_out:
                out = FeaturelessVecValDer(None, None)
            elif isinstance(out, tuple):
                if len(out) == 1:
                    out = out[0]
                else:
                    out = FeaturelessVecValDer(tuple(arg.val for arg in out),
                                            tuple(arg.der for arg in out))
            elif isinstance(out, cls):
                pass
            else:
                raise RuntimeError("""Unknown type passed as out parameter.""")

            return fn(*args, out=out, **kwargs)

        fn_with_out.__name__ = fn.__name__
        return fn_with_out

    if isinstance(mutable_out, bool):
        return decorator
    else:
        value = mutable_out
        mutable_out = False
        return decorator(value)


def _generate_two_argument_broadcasting_function(name, combine_val, combine_der, combine_der_left, combine_der_right):
    """
    * combine_val: Callable[[x, y], out] | Callable[[float, float], float]
    * combine_der: Callable[[x, y, dx, dy], dout] | Callable[[float, float, float, float], float]
    * combine_der_left: Callable[[x, y, dx], dout] | Callable[[float, float, float], float]
    * combine_der_right: Callable[[x, y, dy], dout] | Callable[[float, float, float], float]
    """
    @_add_out_support
    def fn(x1, x2, /, out):
        if np.isscalar(x1):
            x1 = ScalarAsArrayWrapper(x1)

        if np.isscalar(x2):
            x2 = ScalarAsArrayWrapper(x2)

        x1_index = np.arange(np.size(x1)).reshape(x1.shape)
        x2_index = np.arange(np.size(x2)).reshape(x2.shape)

        broadcast_obj = np.broadcast(x1_index, x2_index)
        if out.val is None:
            out = modded_np.empty(broadcast_obj.shape)

        idx_iterator = _ndindex(out.val.shape)

        if isinstance(x1, cls) and isinstance(x2, cls):
            x1_flat = x1.val.flat
            x2_flat = x2.val.flat
            x1_der = x1.der
            x2_der = x2.der
        elif isinstance(x1, cls):
            x1_flat = x1.val.flat
            x2_flat = x2.flat
            x1_der = x1.der
            x2_der = ScalarAsArrayWrapper(_empty_lil)
        elif isinstance(x2, cls):
            x1_flat = x1.flat
            x2_flat = x2.val.flat
            x1_der = ScalarAsArrayWrapper(_empty_lil)
            x2_der = x2.der
        else:
            raise RuntimeError("This should not be occuring.")

        for idx, (x1_elem_idx, x2_elem_idx) in zip(idx_iterator, broadcast_obj):
            x1_val_idx = x1_flat[x1_elem_idx]
            x2_val_idx = x2_flat[x2_elem_idx]
            out.val[idx] = combine_val(x1_val_idx, x2_val_idx)
            out_idx = _square_index_to_simple_index(idx, out.val.shape)
            _combine_lil_rows(
                lambda dx, dy: combine_der(x1_val_idx, x2_val_idx, dx, dy),
                lambda dx: combine_der_left(x1_val_idx, x2_val_idx, dx),
                lambda dy: combine_der_right(x1_val_idx, x2_val_idx, dy),
                x1_der[x1_elem_idx], x2_der[x2_elem_idx],
                out.der.rows[out_idx], out.der.data[out_idx])
        return cls(out.val, out.der)

    fn.__name__ = name
    return fn


@register(np.transpose)
def transpose(a, axes=None):

    assert isinstance(a, cls)
    val = np.transpose(a.val, axes)

    if len([dim for dim in a.shape if dim > 1]) <= 1:
        # Then np.transpose doesn't change any thing I hope?
        return cls(val, scipy.sparse.lil_matrix(a.der))

    val_idx = np.arange(np.size(a.val)).reshape(a.val.shape)

    val_idx_shuffle = np.transpose(val_idx, axes)
    
    der = a.der[val_idx_shuffle.flat]
    return cls(val, der)


# Tested
@register(np.absolute)
@_add_out_support
def absolute(x, /, out):
    """WARNING: Computes the derivative of |x| at 0 as 0."""
    val = np.absolute(x.val, out=out.val)
    der_mask = np.sign(x.val)

    small_mask = (val <= 1e-12)
    if (small_mask).any():
        warnings.warn('abs of a near-zero number, derivative is ill-defined')
        der_mask[small_mask] = 0

    der = _chain_rule(der_mask, x.der, out=out.der)
    return cls(val, der)


abs = absolute


# Tested
@register(np.sqrt)
@_add_out_support
def sqrt(x, /, out):
    val = np.sqrt(x.val, out=out.val)
    return cls(val, _chain_rule(0.5 / val, x.der, out=out.der))


# Tested
@register(np.sin)
@_add_out_support
def sin(x, /, out):
    return cls(np.sin(x.val, out=out.val),
               _chain_rule(np.cos(x.val), x.der, out=out.der))


# Tested
@register(np.cos)
@_add_out_support
def cos(x, /, out):
    return cls(np.cos(x.val, out=out.val),
               _chain_rule(-np.sin(x.val), x.der, out=out.der))


# Tested
@register(np.tan)
@_add_out_support
def tan(x, /, out):
    return cls(np.tan(x.val, out=out.val),
               _chain_rule(1.0/(np.cos(x.val)**2), x.der, out=out.der))


# Tested
@register(np.arcsin)
@_add_out_support
def arcsin(x, /, out):
    return cls(np.arcsin(x.val, out=out.val),
               _chain_rule(1 / np.sqrt(1 - x.val**2), x.der, out=out.der))


# Tested
@register(np.arccos)
@_add_out_support
def arccos(x, /, out):
    return cls(np.arccos(x.val, out=out.val),
               _chain_rule(-1.0 / np.sqrt(1 - x.val**2), x.der, out=out.der))


# Tested
@register(np.arctan)
@_add_out_support
def arctan(x, /, out):
    return cls(np.arctan(x.val, out=out.val),
               _chain_rule(1.0 / (1 + x.val**2), x.der, out=out.der))


@register(np.tanh)
@_add_out_support
def tanh(x, /, out):
    return cls(np.tanh(x.val, out=out.val),
               _chain_rule(1.0/(np.cosh(x.val)**2), x.der, out=out.der))

@register(np.sinh)
@_add_out_support
def sinh(x, /, out):
    return cls(np.sinh(x.val, out=out.val),
               _chain_rule(np.cosh(x.val), x.der, out=out.der))


@register(np.cosh)
@_add_out_support
def cosh(x, /, out):
    return cls(np.cosh(x.val, out=out.val),
               _chain_rule(np.sinh(x.val), x.der, out=out.der))

@register(np.arccosh)
@_add_out_support
def arccosh(x, /, out):
    return cls(np.arccosh(x.val, out=out.val),
               _chain_rule(1 / np.sqrt(x.val**2 -1), x.der, out=out.der))

@register(np.arcsinh)
@_add_out_support
def arcsinh(x, /, out):
    return cls(np.arcsinh(x.val, out=out.val),
               _chain_rule(1 / np.sqrt(x.val**2 + 1), x.der, out=out.der))


@register(np.arctanh)
@_add_out_support
def arctanh(x, /, out):
    return cls(np.arctanh(x.val, out=out.val),
               _chain_rule(1 / (1 - x.val**2), x.der, out=out.der))


# Tested
@register(np.exp)
@_add_out_support
def exp(x, /, out):
    val = np.exp(x.val, out=out.val)
    return cls(val, _chain_rule(val, x.der, out=out.der))


# Tested
@register(np.log)
@_add_out_support
def log(x, /, out):
    return cls(np.log(x.val, out=out.val),
               _chain_rule(1.0 / x.val, x.der, out=out.der))


# Tested
@register(np.log2)
@_add_out_support
def log2(x, /, out):
    return cls(np.log2(x.val, out=out.val),
            _chain_rule(1.0 / np.log(2) / x.val, x.der, out=out.der))


# Tested
@register(np.log10)
@_add_out_support
def log10(x, /, out):
    return cls(np.log10(x.val, out=out.val),
            _chain_rule(1.0 / np.log(10) / x.val, x.der, out=out.der))


# Tested
@register(np.log1p)
@_add_out_support
def log1p(x, /, out):
    return cls(np.log1p(x.val, out=out.val),
            _chain_rule(1.0 / (1 + x.val), x.der, out=out.der))


# Tested
@register(np.negative)
@_add_out_support(mutable_out=True)
def negative(x, /, out):
    if out.der is None:
        out.der = scipy.sparse.lil_matrix(x.der.shape)

    for i, (grad_idx, grad_x) in enumerate(zip(x.der.rows, x.der.data)):
        for j, dx in zip(grad_idx, grad_x):
            out.der[i, j] = -dx

    return cls(np.negative(x.val, out=out.val), out.der)


# Tested
@register(np.positive)
@_add_out_support(True)
def positive(x, /, out):
    if out.der is None:
        out.der = scipy.sparse.lil_matrix(x.der.shape)

    for i, (grad_idx, grad_x) in enumerate(zip(x.der.rows, x.der.data)):
        for j, dx in zip(grad_idx, grad_x):
            out.der[i, j] = dx

    return cls(np.positive(x.val, out=out.val), out.der)



# Tested
add = _generate_two_argument_broadcasting_function('add',
    lambda x, y: x + y,
    lambda _x, _y, dx, dy: dx + dy,
    lambda _x, _y, dx: dx,
    lambda _x, _y, dy: dy
)
register(np.add)(add)

# Tested
subtract = _generate_two_argument_broadcasting_function('subtract',
    lambda x, y: x - y,
    lambda _x, _y, dx, dy: dx - dy,
    lambda _x, _y, dx: dx,
    lambda _x, _y, dy: -dy,
)
register(np.subtract)(subtract)

# Tested
multiply = _generate_two_argument_broadcasting_function('multiply',
    lambda x, y: x * y,
    lambda x, y, dx, dy: x * dy + y * dx,
    lambda _x, y, dx: y * dx,
    lambda x, _y, dy: x * dy,
)
register(np.multiply)(multiply)

# Tested
true_divide = _generate_two_argument_broadcasting_function('true_divide',
    lambda x, y: x / y,
    lambda x, y, dx, dy:  (y * dx - x * dy)/(y**2),
    lambda _x, y, dx:  (y * dx)/(y**2),
    lambda x, y, dy:  (-x * dy)/(y**2),
)
register(np.true_divide)(true_divide)

# Tested
float_power = _generate_two_argument_broadcasting_function('float_power',
    lambda x, y: np.float_power(x, y),
    lambda x, y, dx, dy: (x**(y - 1)) * (y * dx + x * np.log(x) * dy),
    lambda x, y, dx: (x**(y - 1)) * y * dx,
    lambda x, y, dy: (x**y) * np.log(x) * dy,
)
register(np.float_power)(float_power)

# Tested
power = _generate_two_argument_broadcasting_function('power',
    lambda x, y: np.power(x,y),
    lambda x, y, dx, dy: (x**(y - 1)) * (y * dx + x * np.log(x) * dy),
    lambda x, y, dx: (x**(y - 1)) * y * dx,
    lambda x, y, dy: (x**y) * np.log(x) * dy,
)
register(np.power)(power)


def _matmul_internal_gradient_compute_valder_valder(i, k, x1, x2, grad_matrix, entry_index):
    """
        x1: SparseVecValDer
        x2: SparseVecValDer
    """
    for j in range(x1.shape[1]):
        x2_der_idx = _square_index_to_simple_index((j, k), x2.val.shape)
        x1_der_idx = _square_index_to_simple_index((i, j), x1.val.shape)
        x1_ij = x1.val[i, j] 
        x2_jk = x2.val[j, k]
        for grad_idx, x2_der_value in zip(x2.der.rows[x2_der_idx], x2.der.data[x2_der_idx]):
            grad_matrix[entry_index, grad_idx] += x1_ij * x2_der_value

        for grad_idx, x1_der_value in zip(x1.der.rows[x1_der_idx], x1.der.data[x1_der_idx]):
            grad_matrix[entry_index, grad_idx] += x1_der_value * x2_jk


def _matmul_internal_gradient_compute_valder_ndarray(i, k, x1, x2, grad_matrix, entry_index):
    """
        x1: SparseVecValDer
        x2: np.ndarray
    """
    for j in range(x1.shape[1]):
        x1_der_idx = _square_index_to_simple_index((i, j), x1.val.shape)
        x2_jk = x2[j, k]

        for grad_idx, x1_der_value in zip(x1.der.rows[x1_der_idx], x1.der.data[x1_der_idx]):
            grad_matrix[entry_index, grad_idx] += x1_der_value * x2_jk


def _matmul_internal_gradient_compute_ndarray_valder(i, k, x1, x2, grad_matrix, entry_index):
    """
        x1: np.ndarray
        x2: SparseVecValDer
    """
    for j in range(x1.shape[1]):
        x2_der_idx = _square_index_to_simple_index((j, k), x2.val.shape)
        x1_ij = x1[i, j] 

        for grad_idx, x2_der_value in zip(x2.der.rows[x2_der_idx], x2.der.data[x2_der_idx]):
            grad_matrix[entry_index, grad_idx] += x1_ij * x2_der_value

# Partially Tested
# TODO: Add support for broadcasting.
@register(np.matmul)
@_add_out_support
def matmul(x1, x2, /, out):
    # C/Cython candidate
    if isinstance(x1, cls) and isinstance(x2, cls):
        val = np.matmul(x1.val, x2.val, out=out.val)

        if out.der is None:
            der = scipy.sparse.lil_matrix((np.size(val), x1.der.shape[1]))
        else:
            der = out.der
            for idx, der_vals in zip(der.rows, der.data):
                idx.clear()
                der_vals.clear()

        for i, k in _ndindex(val.shape):
            ik_comb = _square_index_to_simple_index((i, k), val.shape)
            _matmul_internal_gradient_compute_valder_valder(i, k, x1, x2, der, ik_comb)

        return cls(val, der)

    elif isinstance(x1, cls):
        val = np.matmul(x1.val, x2, out=out.val)
        if out.der is None:
            der = scipy.sparse.lil_matrix((np.size(val), x1.der.shape[1]))
        else:
            der = out.der

        for i, k in _ndindex(val.shape):
            ik_comb = _square_index_to_simple_index((i, k), val.shape)
            _matmul_internal_gradient_compute_valder_ndarray(i, k, x1, x2, der, ik_comb)

        return cls(val, der)

    elif isinstance(x2, cls):
        val = np.matmul(x1, x2.val, out=out.val)
        if out.der is None:
            der = scipy.sparse.lil_matrix((np.size(val), x2.der.shape[1]))
        else:
            der = out.der

        for i, k in _ndindex(val.shape):
            ik_comb = _square_index_to_simple_index((i, k), val.shape)
            _matmul_internal_gradient_compute_ndarray_valder(i, k, x1, x2, der, ik_comb)

        return cls(val, der)

    else:
        raise RuntimeError("This should not be occuring.")


@register(np.linalg.norm)
def norm(x, ord=None):
    assert ord is None or ord == 2
    return (x.T @ x)[0, 0]


#TODO np.vstack, np.hstack


# Comparison methods
equal = _derivative_independent_comparison_ufunc(np.equal)
register(np.equal)(equal)

less = _derivative_independent_comparison_ufunc(np.less)
register(np.less)(less)

greater = _derivative_independent_comparison_ufunc(np.greater)
register(np.greater)(greater)

less_equal = _derivative_independent_comparison_ufunc(np.less_equal)
register(np.less_equal)(less_equal)

greater_equal = _derivative_independent_comparison_ufunc(np.greater_equal)
register(np.greater_equal)(greater_equal)

not_equal = _derivative_independent_comparison_ufunc(np.not_equal)
register(np.not_equal)(not_equal)

@register(np.isfinite)
def isfinite(x, /, out=None, *args, **kwargs):
    return np.isfinite(x.val, out=out, *args, **kwargs)

@register(np.size)
def size(x, axis=None):
    return np.size(x.val, axis=axis)
