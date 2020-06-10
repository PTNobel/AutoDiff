import numpy as np
import scipy.sparse
from .sparsevecvalder import SparseVecValDer
from . import true_np
from . import numpy_masking


def _vstack_lil_matrix(matrices):
    M = sum(m.shape[0] for m in matrices)
    N = (s := set(m.shape[1] for m in matrices)).pop()
    assert len(s) == 0, "Mixing SparseVecValDer's from multiple sessions?"
    # [N] = set(m.shape[1] for m in matrices)

    output = scipy.sparse.lil_matrix((M, N))
    total = 0
    for m in matrices:
        output[total:total + m.shape[0]] = m
        total += m.shape[0]

    return output


# TODO: Check for any kwargs, if you've recieved any, immediately defer to the
# default function. Add documentation.

# TODO: Add vstack hstack stack dstack concatenate vsplit block
class SparseAutoDiff:
    def __init__(self, *vecs):
        self.vecs = vecs
        self.x = np.vstack(vecs)

    def __enter__(self):
        numpy_masking._active_auto_diffs.append(self)
        new_np = {fn_name: getattr(self, fn_name)
                  for fn_name in numpy_masking._list_of_masked_functions}

        self.old_nps = numpy_masking._swap_numpy_methods(new_np)

        val = np.asarray(self.x)
        # der = scipy.sparse.identity(np.size(val), format='lil')
        der = scipy.sparse.lil_matrix((np.size(val), np.size(val)))
        for i in range(np.size(val)):
            der[i, i] = 1.0

        out = SparseVecValDer(val, der)

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
        numpy_masking._active_auto_diffs.remove(self)
        numpy_masking._swap_numpy_methods(self.old_nps)

    def _build_vec_val_der(self, val):
        der = scipy.sparse.lil_matrix((np.size(val), np.size(self.x)))
        return SparseVecValDer(val, der)

    def zeros(self, shape, **kwargs):
        if len(kwargs) > 0:
            return true_np.zeros(shape, **kwargs)
        val = true_np.zeros(shape)
        return self._build_vec_val_der(val)

    ndarray = np.ndarray

    def _parse_list_of_items(self, obj):
        val_rows = []
        der_rows = []
        for item in obj:
            if isinstance(item, SparseVecValDer):
                val_rows.append(item.val)
                der_rows.append(item.der)
            elif hasattr(item, '__array_interface__') \
                    or hasattr(item, '__array__'):
                val_rows.append(item)
                der_rows.append(
                        scipy.sparse.lil_matrix((np.size(item), np.size(self.x))))

            elif np.isscalar(item):
                val_rows.append(item)
                der_rows.append(scipy.sparse.lil_matrix((1, np.size(self.x))))
            else:
                sub_val_rows, sub_der_rows = self._parse_list_of_items(item)
                val_rows.append(sub_val_rows)
                der_rows.append(_vstack_lil_matrix(sub_der_rows))

        return val_rows, der_rows

    def array(self, obj, **kwargs):
        if len(kwargs) > 0:
            return true_np.array(obj, **kwargs)

        if isinstance(obj, SparseVecValDer):
            return SparseVecValDer(obj.val, obj.der)

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
            der = _vstack_lil_matrix(der_rows)
            return SparseVecValDer(val, der)

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
        if isinstance(a, SparseVecValDer):
            val = true_np.ones_like(a.val)
        else:
            val = true_np.ones_like(a)

        return self._build_vec_val_der(val)

    def zeros_like(self, a):
        if isinstance(a, SparseVecValDer):
            val = true_np.zeros_like(a.val)
        else:
            val = true_np.zeros_like(a)

        return self._build_vec_val_der(val)

    def empty_like(self, a):
        if isinstance(a, SparseVecValDer):
            val = true_np.empty_like(a.val)
        else:
            val = true_np.empty_like(a)

        return self._build_vec_val_der(val)

    def full(self, shape, fill_value):
        val = true_np.full(shape, fill_value)
        return self._build_vec_val_der(val)

    def empty(self, shape, dtype=np.float64):
        if dtype is np.float64:
            val = true_np.empty(shape)
            return self._build_vec_val_der(val)
        else:
            return true_np.empty(shape, dtype=dtype)

    def full_like(self, a, fill_value):
        if isinstance(a, SparseVecValDer):
            val = true_np.full_like(a.val)
        else:
            val = true_np.full_like(a)

        return self._build_vec_val_der(val)

    def ndindex(self, *shape):
        with numpy_masking.WithTrueNumpy():
            return np.ndindex(*shape)

    def broadcast_to(self, array, shape, subok=False):
        with numpy_masking.WithTrueNumpy():
            return np.broadcast_to(array, shape, subok)
