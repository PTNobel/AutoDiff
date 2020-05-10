#!/usr/bin/python3
import numpy as np
from . import true_np

_HANDLED_FUNCS_AND_UFUNCS = {}

def _defer_to_val(f):
    def fn(self, *args, **kwargs):
        return getattr(self.val, f)(*args, **kwargs)
    fn.__name__ = f
    return fn

class VecValDer(np.lib.mixins.NDArrayOperatorsMixin):
    __slots__ = 'val', 'der'

    def __init__(self, val, der):
        self.val = np.asanyarray(val)
        self.der = np.asanyarray(der)

    @property
    def T(self):
        return np.transpose(self)

    @property
    def shape(self):
        return self.val.shape

    def transpose(self, *axes):
        if len(axes) == 0:
            axes = None
        return np.transpose(self, axes)

    all = _defer_to_val('all')
    any = _defer_to_val('any')
    argmax = _defer_to_val('argmax')
    argmin = _defer_to_val('argmin')
    argpartition = _defer_to_val('argpartition')
    argsort = _defer_to_val('argsort')
    nonzero = _defer_to_val('nonzero')
    
    def copy(self):
        return VecValDer(self.val.copy(), self.der.copy())

    def fill(self, value):
        if isinstance(value, VecValDer):
            self.val.fill(value.val)
            self.der[:] = value.der
        else:
            self.val.fill(value)
            self.der.fill(0.0)

    def reshape(self, shape):
        der_dim_shape = self.der.shape[len(self.val.shape):]
        new_der_shape = shape + der_dim_shape
        return VecValDer(self.val.reshape(shape), self.der.reshape(new_der_shape))

    def trace(self, *args, **kwargs):
        return np.trace(*args, **kwargs)

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        if method == '__call__' and ufunc in _HANDLED_FUNCS_AND_UFUNCS:
            return _HANDLED_FUNCS_AND_UFUNCS[ufunc](*args, **kwargs)
        else:
            return NotImplemented
            # Note, Parth considered speculative execution, but concluded
            # that the lack of guarantees about correctness and avoiding side
            # effects wasn't worth it.

    def __array_function__(self, func, types, args, kwargs):
        if func in _HANDLED_FUNCS_AND_UFUNCS:
            return _HANDLED_FUNCS_AND_UFUNCS[func](*args, **kwargs)
        else:
            return NotImplemented

    def __setitem__(self, key, value):
        if isinstance(value, VecValDer):
            self.val[key] = value.val
            self.der[key] = value.der
        else:
            self.val[key] = value
            self.der[key] = 0

    def __getitem__(self, key):
        return VecValDer(self.val[key], self.der[key])

    def __len__(self):
        return len(self.val)

    # Tested
    def __iter__(self):
        return (VecValDer(val, der)
                for val, der in zip(iter(self.val), iter(self.der)))

    def __repr__(self):
        return f"VecValDer(\n{self.val},\n {self.der})"

    def __str__(self):
        return str(self.val)


def register(np_fn):
    assert np_fn not in _HANDLED_FUNCS_AND_UFUNCS

    def decorator(f):
        assert np_fn.__name__ == f.__name__
        _HANDLED_FUNCS_AND_UFUNCS[np_fn] = f
        return f
    return decorator
