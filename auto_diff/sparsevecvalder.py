#!/usr/bin/python3
import numpy as np

_HANDLED_FUNCS_AND_UFUNCS = {}

def _defer_to_val(f):
    def fn(self, *args, **kwargs):
        return getattr(self.val, f)(*args, **kwargs)
    fn.__name__ = f
    return fn

class SparseVecValDer(np.lib.mixins.NDArrayOperatorsMixin):
    __slots__ = 'val', 'der'

    def __init__(self, val, der):
        self.val = np.asanyarray(val)
        self.der = der

    @property
    def T(self):
        return np.transpose(self)

    @property
    def shape(self):
        return self.val.shape

    @property
    def dtype(self):
        return self.val.dtype


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
        return SparseVecValDer(self.val.copy(), self.der.copy())

    def fill(self, value):
        if isinstance(value, SparseVecValDer):
            self.val.fill(value.val)
            self.der[:] = value.der
        else:
            self.val.fill(value)
            self.der[:] = 0.0

    def reshape(self, shape):
        return SparseVecValDer(self.val.reshape(shape), self.der)

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
            #
            # My idea for speculative execution, was we could just run the ufunc
            # on self.val, and if the return value had a dtype that was not f64,
            # then it is safe to assume that there is no effect on gradients
            # and therefore that we can return it. If it is f64, we assume that
            # we're dropping data about derivatives and therefore should return
            # NotImplemented.
            #
            # If all ufuncs were pure, this would be safe. However, ufuncs can
            # mutate arrays or modify some other type of state, and there is no
            # way to walkback execution of these functions, therefore we can't
            # safely speculate.
            #
            # If MyPy ever finishes designing NumPy stubs then perhaps we could
            # use those to auto-generate (_defer_to_val) implementations or
            # something.

    def __array_function__(self, func, types, args, kwargs):
        if func in _HANDLED_FUNCS_AND_UFUNCS:
            return _HANDLED_FUNCS_AND_UFUNCS[func](*args, **kwargs)
        else:
            return NotImplemented

    def __setitem__(self, key, value):
        if isinstance(value, SparseVecValDer):
            self.val[key] = value.val
            if value.der.shape[0] == 1:
                for i in np.arange(np.size(self.val)
                        ).reshape(self.val.shape)[key].flat:
                    self.der[i] = value.der
            else:
                ctr = 0
                for i in np.arange(np.size(self.val)
                        ).reshape(self.val.shape)[key].flat:
                    self.der[i] = value.der[ctr]
                    ctr += 1
        else:
            self.val[key] = value
            for i in np.arange(np.size(self.val)).reshape(self.val.shape)[key].flat:
                self.der[i] = 0

    def __getitem__(self, key):
        der = self.der[np.arange(np.size(self.val)).reshape(self.val.shape)[key].flat]

        return SparseVecValDer(self.val[key], der)


    def __len__(self):
        return len(self.val)

    # Tested
    def __iter__(self):
        return (SparseVecValDer(val, der)
                for val, der in zip(iter(self.val), iter(self.der)))

    def __repr__(self):
        return f"SparseVecValDer(\n{self.val},\n {self.der})"

    def __str__(self):
        return str(self.val)


def register(np_fn):
    assert np_fn not in _HANDLED_FUNCS_AND_UFUNCS

    def decorator(f):
        assert np_fn.__name__ == f.__name__
        _HANDLED_FUNCS_AND_UFUNCS[np_fn] = f
        return f
    return decorator
