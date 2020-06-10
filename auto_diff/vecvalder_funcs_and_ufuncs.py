from .vecvalder import register, VecValDer
import warnings
from . import true_np as np
import numpy as modded_np

cls = VecValDer


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


class FeaturelessVecValDer:
    __slots__ = 'val', 'der'
    def __init__(self, val, der):
        self.val = val
        self.der = der


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


# This is probably a good C/Cython candidate
def _chain_rule(f_x, dx, out=None):
    r"""This implements the chain rule for the elementwise application  of
        f: R -> R
    Meant to implement for f: R -> R, g: R^n -> R, x \in R^n
    \nabla f(g(x)) = f'(g(x)) \nabla g(x)
    """
    if out is None:
        out = np.ndarray(dx.shape)  # Uninitialized memory is fine because
    # we're about to overwrite each element. If we do compression of the for
    # loop in the future be sure to switch to np.zeros.
    for index, y in np.ndenumerate(f_x):
        out[index] = y * dx[index]
    return out


# Add support for where=
def _add_out_support(fn):
    def fn_with_out(*args, out=None, where=True, **kwargs):
        if where is not True:
            raise NotImplementedError("Have yet to implement where support")
        if out is None:
            out = _none_vec_val_der
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
                    #Please email parthnobel@berkeley.edu with sample code.""")

        return fn(*args, out=out, **kwargs)

    fn_with_out.__name__ = fn.__name__
    return fn_with_out


@register(np.transpose)
def transpose(a, axes=None):
    assert isinstance(a, cls)
    val = np.transpose(a.val, axes)
    dims = len(val.shape)
    if axes is None:
        axes = reversed(range(dims))
    der = np.transpose(a.der, (*axes, *range(dims, 2 * dims)))
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
@_add_out_support
def negative(x, /, out):
    return cls(np.negative(x.val, out=out.val), np.negative(x.der, out=out.der))


# Tested
@register(np.positive)
@_add_out_support
def positive(x, /, out):
    return cls(np.positive(x.val, out=out.val), np.positive(x.der, out=out.der))


# Tested
@register(np.add)
@_add_out_support
def add(x1, x2, /, out):
    broadcast_obj = np.broadcast(x1, x2)
    if out.val is None:
        out = modded_np.empty(broadcast_obj.shape)
    idx_iterator = _ndindex(out.val.shape)
 
    if isinstance(x1, cls) and isinstance(x2, cls):
        for idx, (x1_elem, x2_elem) in zip(idx_iterator, broadcast_obj):
            out.val[idx] = x1_elem.val + x2_elem.val
            out.der[idx] = x1_elem.der + x2_elem.der
        return cls(out.val, out.der)

    elif isinstance(x1, cls):
        for idx, (x1_elem, x2_elem) in zip(idx_iterator, broadcast_obj):
            out.val[idx] = x1_elem.val + x2_elem
            out.der[idx] = x1_elem.der
        return cls(out.val, out.der)

    elif isinstance(x2, cls):
        for idx, (x1_elem, x2_elem) in zip(idx_iterator, broadcast_obj):
            out.val[idx] = x1_elem + x2_elem.val
            out.der[idx] = x2_elem.der
        return cls(out.val, out.der)
    else:
        raise RuntimeError("This should not be occuring.")

# Tested
@register(np.subtract)
@_add_out_support
def subtract(x1, x2, /, out):
    broadcast_obj = np.broadcast(x1, x2)
    if out.val is None:
        out = modded_np.empty(broadcast_obj.shape)
    idx_iterator = _ndindex(out.val.shape)
 
    if isinstance(x1, cls) and isinstance(x2, cls):
        for idx, (x1_elem, x2_elem) in zip(idx_iterator, broadcast_obj):
            out.val[idx] = x1_elem.val - x2_elem.val
            out.der[idx] = x1_elem.der - x2_elem.der
        return cls(out.val, out.der)

    elif isinstance(x1, cls):
        for idx, (x1_elem, x2_elem) in zip(idx_iterator, broadcast_obj):
            out.val[idx] = x1_elem.val - x2_elem
            out.der[idx] = x1_elem.der
        return cls(out.val, out.der)

    elif isinstance(x2, cls):
        for idx, (x1_elem, x2_elem) in zip(idx_iterator, broadcast_obj):
            out.val[idx] = x1_elem - x2_elem.val
            out.der[idx] = -x2_elem.der
        return cls(out.val, out.der)
    else:
        raise RuntimeError("This should not be occuring.")


# Tested
@register(np.multiply)
@_add_out_support
def multiply(x1, x2, /, out):
    broadcast_obj = np.broadcast(x1, x2)
    if out.val is None:
        out = modded_np.empty(broadcast_obj.shape)
    idx_iterator = _ndindex(out.val.shape)
 
    if isinstance(x1, cls) and isinstance(x2, cls):
        for idx, (x1_elem, x2_elem) in zip(idx_iterator, broadcast_obj):
            out.val[idx] = x1_elem.val * x2_elem.val
            out.der[idx] = x1_elem.der * x2_elem.val + x1_elem.val * x2_elem.der
        return cls(out.val, out.der)

    elif isinstance(x1, cls):
        for idx, (x1_elem, x2_elem) in zip(idx_iterator, broadcast_obj):
            out.val[idx] = x1_elem.val * x2_elem
            out.der[idx] = x1_elem.der * x2_elem
        return cls(out.val, out.der)

    elif isinstance(x2, cls):
        for idx, (x1_elem, x2_elem) in zip(idx_iterator, broadcast_obj):
            out.val[idx] = x1_elem * x2_elem.val
            out.der[idx] = x1_elem * x2_elem.der
        return cls(out.val, out.der)

    else:
        raise RuntimeError("This should not be occuring.")


# Tested
@register(np.true_divide)
@_add_out_support
def true_divide(x1, x2, /, out):
    broadcast_obj = np.broadcast(x1, x2)
    if out.val is None:
        out = modded_np.empty(broadcast_obj.shape)
    idx_iterator = _ndindex(out.val.shape)
 
    if isinstance(x1, cls) and isinstance(x2, cls):
        for idx, (x1_elem, x2_elem) in zip(idx_iterator, broadcast_obj):
            out.val[idx] = x1_elem.val / x2_elem.val
            out.der[idx] = (x1_elem.der * x2_elem.val
                            - x1_elem.val * x2_elem.der) / x2_elem.val**2
        return cls(out.val, out.der)

    elif isinstance(x1, cls):
        for idx, (x1_elem, x2_elem) in zip(idx_iterator, broadcast_obj):
            out.val[idx] = x1_elem.val / x2_elem
            out.der[idx] = x1_elem.der * x2_elem / x2_elem**2
        return cls(out.val, out.der)

    elif isinstance(x2, cls):
        for idx, (x1_elem, x2_elem) in zip(idx_iterator, broadcast_obj):
            out.val[idx] = x1_elem / x2_elem.val
            out.der[idx] = -x1_elem * x2_elem.der / x2_elem.val**2
        return cls(out.val, out.der)

    else:
        raise RuntimeError("This should not be occuring.")


@register(np.float_power)
@_add_out_support
def float_power(x1, x2, /, out):
    broadcast_obj = np.broadcast(x1, x2)
    if out.val is None:
        out = modded_np.empty(broadcast_obj.shape)
    idx_iterator = _ndindex(out.val.shape)
 
    if isinstance(x1, cls) and isinstance(x2, cls):
        for idx, (x1_elem, x2_elem) in zip(idx_iterator, broadcast_obj):
            out.val[idx] = np.float_power(x1_elem.val, x2_elem.val)
            out.der[idx] = (x1_elem.val**(x2_elem.val - 1))* (x2_elem.val * x1_elem.der + x1_elem.val * np.log(x1_elem.val) * x2_elem.der)
        return cls(out.val, out.der)

    elif isinstance(x1, cls):
        for idx, (x1_elem, x2_elem) in zip(idx_iterator, broadcast_obj):
            out.val[idx] = np.float_power(x1_elem.val, x2_elem)
            out.der[idx] = (x1_elem.val**(x2_elem-1))* x2_elem * x1_elem.der
        return cls(out.val, out.der)

    elif isinstance(x2, cls):
        for idx, (x1_elem, x2_elem) in zip(idx_iterator, broadcast_obj):
            out.val[idx] = np.float_power(x1_elem, x2_elem.val)
            out.der[idx] = (x1_elem**(x2_elem.val)) * np.log(x1_elem) * x2_elem.der
        return cls(out.val, out.der)

    else:
        raise RuntimeError("This should not be occuring.")


@register(np.power)
@_add_out_support
def power(x1, x2, /, out):
    broadcast_obj = np.broadcast(x1, x2)
    if out.val is None:
        out = modded_np.empty(broadcast_obj.shape)
    idx_iterator = _ndindex(out.val.shape)
 
    if isinstance(x1, cls) and isinstance(x2, cls):
        for idx, (x1_elem, x2_elem) in zip(idx_iterator, broadcast_obj):
            out.val[idx] = np.power(x1_elem.val, x2_elem.val)
            out.der[idx] = (x1_elem.val**(x2_elem.val - 1))* (x2_elem.val * x1_elem.der + x1_elem.val * np.log(x1_elem.val) * x2_elem.der)
        return cls(out.val, out.der)

    elif isinstance(x1, cls):
        for idx, (x1_elem, x2_elem) in zip(idx_iterator, broadcast_obj):
            out.val[idx] = np.power(x1_elem.val, x2_elem)
            out.der[idx] = (x1_elem.val**(x2_elem-1))* x2_elem * x1_elem.der
        return cls(out.val, out.der)

    elif isinstance(x2, cls):
        for idx, (x1_elem, x2_elem) in zip(idx_iterator, broadcast_obj):
            out.val[idx] = np.power(x1_elem, x2_elem.val)
            out.der[idx] = (x1_elem**(x2_elem.val)) * np.log(x1_elem) * x2_elem.der
        return cls(out.val, out.der)

    else:
        raise RuntimeError("This should not be occuring.")

# Partially Tested
# TODO: Add support for broadcasting.
@register(np.matmul)
@_add_out_support
def matmul(x1, x2, /, out):
    if isinstance(x1, cls) and isinstance(x2, cls):
        val = np.matmul(x1.val, x2.val, out=out.val)
        # Find a way to write der as numpy primitives.
        # C/Cython candidate
        if out.der is None:
            der = np.ndarray((*val.shape, *x1.der.shape[-2:]))
        else:
            der = out.der
        for i, k in _ndindex(val.shape):
            der[i, k] = sum(
                x1.val[i, j] * x2.der[j, k] + x1.der[i, j] * x2.val[j, k]
                for j in range(x1.shape[1]))
        return cls(val, der)

    elif isinstance(x1, cls):
        val = np.matmul(x1.val, x2, out=out.val)
        # Find a way to write der as numpy primitives.
        # C/Cython candidate
        if out.der is None:
            der = np.ndarray((*val.shape, *x1.der.shape[-2:]))
        else:
            der = out.der
        for i, k in _ndindex(val.shape):
            der[i, k] = sum(x1.der[i, j] * x2[j, k]
                            for j in range(x1.shape[1]))
        return cls(val, der)

    elif isinstance(x2, cls):
        val = np.matmul(x1, x2.val, out=out.val)
        # Find a way to write der as numpy primitives.
        # C/Cython candidate
        if out.der is None:
            der = np.ndarray((*val.shape, *x2.der.shape[-2:]))
        else:
            der = out.der
        for i, k in _ndindex(val.shape):
            der[i, k] = sum(x1[i, j] * x2.der[j, k]
                            for j in range(x1.shape[1]))
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
def isfinite(x):
    return np.isfinite(x.val)
