import numpy as np
from .vecvalder import VecValDer
from . import true_np


_active_auto_diffs = []

# FYI *_like can be handled as functions on their parameters, but I decided not
# to because it feels better to mask them.
_list_of_masked_functions = [
    'zeros',
    'eye',
    'identity',
    'ndarray',
    'array',
    'ones',
    'ones_like',
    'zeros_like',
    'empty',
    'empty_like',
    'full',
    'full_like',

    'ndindex',
    'broadcast_to',
]


def _swap_numpy_methods(new):
    output = {fn_name: getattr(np, fn_name)
              for fn_name in _list_of_masked_functions}

    # Should unwind on key error in fetching new[fn_name]
    # Or at least verify entries for all.
    for fn_name in _list_of_masked_functions:
        new[fn_name]
        # Raise an error if new does not include every function we're masking
        # We want to do this before we start modifying anything to avoid the
        # need to keep track of state in order to unwind from an error.
        # This loop also is much more memory efficient then a list or dict
        # comprehension. Especially as we add more methods. Consuming a
        # generator expression, would require a for loop and save nothing.

    for fn_name in _list_of_masked_functions:
        setattr(np, fn_name, new[fn_name])

    return output


# This should probably be never used internally outside of masking.
# Just saying.
class WithTrueNumpy:
    true_np_dict = {fn_name: getattr(true_np, fn_name)
                    for fn_name in _list_of_masked_functions}

    def __enter__(self):
        self.old = _swap_numpy_methods(self.true_np_dict)

    def __exit__(self, *args):
        _swap_numpy_methods(self.old)


# TODO: Add vstack hstack stack dstack concatenate vsplit block
class AutoDiff:
    def __init__(self, *vecs):
        self.vecs = vecs
        self.x = np.vstack(vecs)

    def __enter__(self):
        _active_auto_diffs.append(self)
        new_np = {fn_name: getattr(self, fn_name)
                  for fn_name in _list_of_masked_functions}

        self.old_nps = _swap_numpy_methods(new_np)

        val = np.asarray(self.x)
        der = true_np.zeros((*val.shape, *val.shape))
        for i in np.ndindex(val.shape):
            der[(*i, *i)] = 1.0

        out = VecValDer(val, der)

        if len(self.vecs) > 1:
            out_vecs = []
            total = 0
            for vec in self.vecs:
                out_vecs.append(out[total:total + vec.shape[0], :])
                total += vec.shape[0]

            return tuple(out_vecs)
        else:
            return out

    def __exit__(self, type, value, traceback):
        _active_auto_diffs.remove(self)
        _swap_numpy_methods(self.old_nps)

    def _build_vec_val_der(self, val):
        der = true_np.zeros((*val.shape, *self.x.shape))
        return VecValDer(val, der)

    def zeros(self, shape):
        val = true_np.zeros(shape)
        return self._build_vec_val_der(val)

    def ndarray(self, shape):
        val = true_np.ndarray(shape)
        return self._build_vec_val_der(val)

    def _parse_list_of_items(self, obj):
        val_rows = []
        der_rows = []
        for item in obj:
            if isinstance(item, VecValDer):
                val_rows.append(item.val)
                der_rows.append(item.der)
            elif hasattr(item, '__array_interface__') \
                    or hasattr(item, '__array__'):
                val_rows.append(item)
                der_rows.append(true_np.zeros(
                    (*item.shape, *self.x.shape)))
            elif np.isscalar(item):
                val_rows.append(item)
                der_rows.append(true_np.zeros(self.x.shape))
            else:
                sub_val_rows, sub_der_rows = self._parse_list_of_items(item)
                val_rows.append(sub_val_rows)
                der_rows.append(sub_der_rows)

        return val_rows, der_rows

    def array(self, obj):
        if isinstance(obj, VecValDer):
            return VecValDer(obj.val, obj.der)
        elif hasattr(obj, '__array_interface__') \
                or hasattr(obj, '__array__'):
            val = true_np.array(obj)
            return self._build_vec_val_der(val)
        elif np.isscalar(obj):
            val = true_np.array(obj)
            return self._build_vec_val_der(val)
        else:
            val_rows, der_rows = self._parse_list_of_items(obj)
            val = true_np.array(val_rows)
            der = true_np.array(der_rows)
            return VecValDer(val, der)

    def eye(self, N, M=None, k=0):
        val = true_np.eye(N, M, k)
        return self._build_vec_val_der(val)

    def identity(self, n):
        val = true_np.identity(n)
        return self._build_vec_val_der(val)

    def ones(self, shape):
        val = true_np.ones(shape)
        return self._build_vec_val_der(val)

    def ones_like(self, a):
        if isinstance(a, VecValDer):
            val = true_np.ones_like(a.val)
        else:
            val = true_np.ones_like(a)

        return self._build_vec_val_der(val)

    def zeros_like(self, a):
        if isinstance(a, VecValDer):
            val = true_np.zeros_like(a.val)
        else:
            val = true_np.zeros_like(a)

        return self._build_vec_val_der(val)

    def empty_like(self, a):
        if isinstance(a, VecValDer):
            val = true_np.empty_like(a.val)
        else:
            val = true_np.empty_like(a)

        return self._build_vec_val_der(val)

    def full(self, shape, fill_value):
        val = true_np.full(shape, fill_value)
        return self._build_vec_val_der(val)

    def empty(self, shape):
        val = true_np.empty(shape)
        return self._build_vec_val_der(val)

    def full_like(self, a, fill_value):
        if isinstance(a, VecValDer):
            val = true_np.full_like(a.val)
        else:
            val = true_np.full_like(a)

        return self._build_vec_val_der(val)

    def ndindex(self, *shape):
        with WithTrueNumpy():
            return np.ndindex(*shape)

    def broadcast_to(self, array, shape, subok=False):
        with WithTrueNumpy():
            return np.broadcast_to(array, shape, subok)
