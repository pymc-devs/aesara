import numpy as np


try:
    import scipy.sparse

    imported_scipy = True
except ImportError:
    imported_scipy = False


import aesara
from aesara.tensor.type import TensorType


def _is_sparse(x):
    """

    Returns
    -------
    boolean
        True iff x is a L{scipy.sparse.spmatrix} (and not a L{numpy.ndarray}).

    """
    if not isinstance(x, (scipy.sparse.spmatrix, np.ndarray, tuple, list)):
        raise NotImplementedError(
            "this function should only be called on "
            "sparse.scipy.sparse.spmatrix or "
            "numpy.ndarray, not,",
            x,
        )
    return isinstance(x, scipy.sparse.spmatrix)


class SparseType(TensorType):
    """
    Fundamental way to create a sparse node.

    Parameters
    ----------
    dtype : numpy dtype string such as 'int64' or 'float64' (among others)
        Type of numbers in the matrix.
    format: str
        The sparse storage strategy.

    Returns
    -------
    An empty SparseVariable instance.

    Notes
    -----
    As far as I can tell, L{scipy.sparse} objects must be matrices, i.e.
    have dimension 2.

    """

    if imported_scipy:
        format_cls = {
            "csr": scipy.sparse.csr_matrix,
            "csc": scipy.sparse.csc_matrix,
            "bsr": scipy.sparse.bsr_matrix,
        }
    dtype_specs_map = {
        "float32": (float, "npy_float32", "NPY_FLOAT32"),
        "float64": (float, "npy_float64", "NPY_FLOAT64"),
        "uint8": (int, "npy_uint8", "NPY_UINT8"),
        "int8": (int, "npy_int8", "NPY_INT8"),
        "uint16": (int, "npy_uint16", "NPY_UINT16"),
        "int16": (int, "npy_int16", "NPY_INT16"),
        "uint32": (int, "npy_uint32", "NPY_UINT32"),
        "int32": (int, "npy_int32", "NPY_INT32"),
        "uint64": (int, "npy_uint64", "NPY_UINT64"),
        "int64": (int, "npy_int64", "NPY_INT64"),
        "complex128": (complex, "aesara_complex128", "NPY_COMPLEX128"),
        "complex64": (complex, "aesara_complex64", "NPY_COMPLEX64"),
    }
    ndim = 2

    # Will be set to SparseVariable SparseConstant later.
    Variable = None
    Constant = None

    def __init__(self, format, dtype, broadcastable=None, name=None):
        if not imported_scipy:
            raise Exception(
                "You can't make SparseType object when SciPy is not available."
            )

        if not isinstance(format, str):
            raise TypeError("The sparse format parameter must be a string")

        if format in self.format_cls:
            self.format = format
        else:
            raise NotImplementedError(
                f'unsupported format "{format}" not in list',
            )

        if broadcastable is None:
            broadcastable = [False, False]

        super().__init__(dtype, broadcastable, name=name)

    def clone(self, dtype=None, broadcastable=None):
        new = super().clone(dtype=dtype, broadcastable=broadcastable)
        new.format = self.format
        return new

    def filter(self, value, strict=False, allow_downcast=None):
        if (
            isinstance(value, self.format_cls[self.format])
            and value.dtype == self.dtype
        ):
            return value
        if strict:
            raise TypeError(
                f"{value} is not sparse, or not the right dtype (is {value.dtype}, "
                f"expected {self.dtype})"
            )
        # The input format could be converted here
        if allow_downcast:
            sp = self.format_cls[self.format](value, dtype=self.dtype)
        else:
            sp = self.format_cls[self.format](value)
            if str(sp.dtype) != self.dtype:
                raise NotImplementedError(
                    f"Expected {self.dtype} dtype but got {sp.dtype}"
                )
        if sp.format != self.format:
            raise NotImplementedError()
        return sp

    @staticmethod
    def may_share_memory(a, b):
        # This is Fred suggestion for a quick and dirty way of checking
        # aliasing .. this can potentially be further refined (ticket #374)
        if _is_sparse(a) and _is_sparse(b):
            return (
                SparseType.may_share_memory(a, b.data)
                or SparseType.may_share_memory(a, b.indices)
                or SparseType.may_share_memory(a, b.indptr)
            )
        if _is_sparse(b) and isinstance(a, np.ndarray):
            a, b = b, a
        if _is_sparse(a) and isinstance(b, np.ndarray):
            if (
                np.may_share_memory(a.data, b)
                or np.may_share_memory(a.indices, b)
                or np.may_share_memory(a.indptr, b)
            ):
                # currently we can't share memory with a.shape as it is a tuple
                return True
        return False

    def make_variable(self, name=None):
        return self.Variable(self, name=name)

    def __eq__(self, other):
        return super().__eq__(other) and other.format == self.format

    def __hash__(self):
        return super().__hash__() ^ hash(self.format)

    def __str__(self):
        return f"Sparse[{self.dtype}, {self.format}]"

    def __repr__(self):
        return f"Sparse[{self.dtype}, {self.format}]"

    def values_eq_approx(self, a, b, eps=1e-6):
        # WARNING: equality comparison of sparse matrices is not fast or easy
        # we definitely do not want to be doing this un-necessarily during
        # a FAST_RUN computation..
        if not scipy.sparse.issparse(a) or not scipy.sparse.issparse(b):
            return False
        diff = abs(a - b)
        if diff.nnz == 0:
            return True
        # Built-in max from python is not implemented for sparse matrix as a
        # reduction. It returns a sparse matrix which cannot be compared to a
        # scalar. When comparing sparse to scalar, no exceptions is raised and
        # the returning value is not consistent. That is why it is apply to a
        # numpy.ndarray.
        return max(diff.data) < eps

    def values_eq(self, a, b):
        # WARNING: equality comparison of sparse matrices is not fast or easy
        # we definitely do not want to be doing this un-necessarily during
        # a FAST_RUN computation..
        return (
            scipy.sparse.issparse(a)
            and scipy.sparse.issparse(b)
            and abs(a - b).sum() == 0.0
        )

    def is_valid_value(self, a):
        return scipy.sparse.issparse(a) and (a.format == self.format)

    def get_shape_info(self, obj):
        obj = self.filter(obj)
        assert obj.indices.dtype == "int32"
        assert obj.indptr.dtype == "int32"
        return (obj.shape, obj.data.size, obj.indices.size, obj.indptr.size, obj.nnz)

    def get_size(self, shape_info):
        return (
            shape_info[1] * np.dtype(self.dtype).itemsize
            + (shape_info[2] + shape_info[3]) * np.dtype("int32").itemsize
        )

    def value_zeros(self, shape):
        matrix_constructor = getattr(scipy.sparse, f"{self.format}_matrix", None)

        if matrix_constructor is None:
            raise ValueError(f"Sparse matrix type {self.format} not found in SciPy")

        return matrix_constructor(shape, dtype=self.dtype)


# Register SparseType's C code for ViewOp.
aesara.compile.register_view_op_c_code(
    SparseType,
    """
    Py_XDECREF(%(oname)s);
    %(oname)s = %(iname)s;
    Py_XINCREF(%(oname)s);
    """,
    1,
)
