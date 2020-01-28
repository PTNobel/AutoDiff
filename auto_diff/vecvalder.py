#!/usr/bin/python3
import numpy as np
from . import true_np

_HANDLED_FUNCS_AND_UFUNCS = {}


class VecValDer(np.lib.mixins.NDArrayOperatorsMixin):
    __slots__ = 'val', 'der'

    def __init__(self, val, der):
        self.val = val
        self.der = der

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
            # We're being passed a scalar? Or something weird?
            # Like vec[:] = other_vec?
            value = np.asarray(value)
            assert self.val[key].shape == value.shape
            self.val[key] = value
            self.der[key] = true_np.zeros(self.der[key].shape)

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
