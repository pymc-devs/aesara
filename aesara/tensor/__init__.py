"""Define the tensor toplevel"""


__docformat__ = "restructuredtext en"

from functools import singledispatch
from typing import TYPE_CHECKING, Any, Callable, List, NoReturn, Optional, Union


if TYPE_CHECKING:
    from aesara.graph.basic import Apply, Variable


def as_tensor_variable(
    x: Any, name: Optional[str] = None, ndim: Optional[int] = None, **kwargs
) -> Callable:
    """Convert `x` into the appropriate ``TensorType``.

    This function is often used by ``make_node`` methods of ``Op`` subclasses
    to turn ndarrays, numbers, ``Scalar`` instances, ``Apply`` instances and
    ``TensorType`` instances into valid input list elements.

    Parameters
    ----------
    x
        The object to be converted into a ``Variable`` type. A
        ``numpy.ndarray`` argument will not be copied, but a list of numbers
        will be copied to make an ``numpy.ndarray``.
    name
        If a new ``Variable`` instance is created, it will be named with this
        string.
    ndim
        Return a ``Variable`` with this many dimensions.
    dtype
        The dtype to use for the resulting ``Variable``.  If `x` is already
        a ``Variable`` type, then the dtype will not be changed.

    Raises
    ------
    TypeError
        If `x` cannot be converted to a ``TensorType`` Variable.

    """
    return _as_tensor_variable(x, name, ndim, **kwargs)


@singledispatch
def _as_tensor_variable(
    x: "Union[int, List, Apply, Variable]",
    name: Optional[str],
    ndim: Optional[int],
    **kwargs,
) -> NoReturn:
    raise NotImplementedError(f"Cannot convert {x} to a tensor variable.")


import aesara.tensor.exceptions
from aesara.gradient import consider_constant, grad, hessian, jacobian
from aesara.tensor import sharedvar  # adds shared-variable constructors
from aesara.tensor import (
    basic_opt,
    blas,
    blas_c,
    blas_scipy,
    nlinalg,
    nnet,
    opt_uncanonicalize,
    xlogx,
)
from aesara.tensor.basic import *
from aesara.tensor.blas import batched_dot, batched_tensordot
from aesara.tensor.extra_ops import (
    bartlett,
    bincount,
    broadcast_shape,
    broadcast_shape_iter,
    broadcast_to,
    cumprod,
    cumsum,
    diff,
    fill_diagonal,
    fill_diagonal_offset,
    ravel_multi_index,
    repeat,
    squeeze,
    unique,
    unravel_index,
)
from aesara.tensor.io import *
from aesara.tensor.math import *
from aesara.tensor.shape import (
    reshape,
    shape,
    shape_padaxis,
    shape_padleft,
    shape_padright,
    specify_shape,
)

# We import as `_shared` instead of `shared` to avoid confusion between
# `aesara.shared` and `tensor._shared`.
from aesara.tensor.sort import argsort, argtopk, sort, topk, topk_and_argtopk
from aesara.tensor.subtensor import *
from aesara.tensor.type import (
    TensorType,
    bcol,
    bmatrix,
    brow,
    bscalar,
    btensor3,
    btensor4,
    btensor5,
    btensor6,
    btensor7,
    bvector,
    ccol,
    cmatrix,
    col,
    cols,
    complex_matrix_types,
    complex_scalar_types,
    complex_types,
    complex_vector_types,
    crow,
    cscalar,
    ctensor3,
    ctensor4,
    ctensor5,
    ctensor6,
    ctensor7,
    cvector,
    dcol,
    dcols,
    dmatrices,
    dmatrix,
    drow,
    drows,
    dscalar,
    dscalars,
    dtensor3,
    dtensor3s,
    dtensor4,
    dtensor4s,
    dtensor5,
    dtensor5s,
    dtensor6,
    dtensor6s,
    dtensor7,
    dtensor7s,
    dvector,
    dvectors,
    fcol,
    fcols,
    float_matrix_types,
    float_scalar_types,
    float_types,
    float_vector_types,
    fmatrices,
    fmatrix,
    frow,
    frows,
    fscalar,
    fscalars,
    ftensor3,
    ftensor3s,
    ftensor4,
    ftensor4s,
    ftensor5,
    ftensor5s,
    ftensor6,
    ftensor6s,
    ftensor7,
    ftensor7s,
    fvector,
    fvectors,
    icol,
    icols,
    imatrices,
    imatrix,
    int_matrix_types,
    int_scalar_types,
    int_types,
    int_vector_types,
    irow,
    irows,
    iscalar,
    iscalars,
    itensor3,
    itensor3s,
    itensor4,
    itensor4s,
    itensor5,
    itensor5s,
    itensor6,
    itensor6s,
    itensor7,
    itensor7s,
    ivector,
    ivectors,
    lcol,
    lcols,
    lmatrices,
    lmatrix,
    lrow,
    lrows,
    lscalar,
    lscalars,
    ltensor3,
    ltensor3s,
    ltensor4,
    ltensor4s,
    ltensor5,
    ltensor5s,
    ltensor6,
    ltensor6s,
    ltensor7,
    ltensor7s,
    lvector,
    lvectors,
    matrices,
    matrix,
    row,
    rows,
    scalar,
    scalars,
    tensor,
    tensor3,
    tensor3s,
    tensor4,
    tensor4s,
    tensor5,
    tensor5s,
    tensor6,
    tensor6s,
    tensor7,
    tensor7s,
    values_eq_approx_always_true,
    vector,
    vectors,
    wcol,
    wrow,
    wscalar,
    wtensor3,
    wtensor4,
    wtensor5,
    wtensor6,
    wtensor7,
    wvector,
    zcol,
    zmatrix,
    zrow,
    zscalar,
    ztensor3,
    ztensor4,
    ztensor5,
    ztensor6,
    ztensor7,
    zvector,
)
from aesara.tensor.type_other import *
