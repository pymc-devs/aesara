from collections.abc import Collection

import numpy as np

import aesara
from aesara.assert_op import Assert
from aesara.gradient import (
    DisconnectedType,
    _float_zeros_like,
    disconnected_type,
    grad_undefined,
)
from aesara.graph.basic import Apply, equal_computations
from aesara.graph.op import COp, Op
from aesara.graph.params_type import ParamsType
from aesara.graph.type import EnumList, Generic
from aesara.misc.safe_asarray import _asarray
from aesara.scalar import int32 as int_t
from aesara.scalar import upcast
from aesara.tensor import basic as aet
from aesara.tensor.exceptions import NotScalarConstantError
from aesara.tensor.math import abs_
from aesara.tensor.math import all as aet_all
from aesara.tensor.math import eq, ge, lt, maximum, minimum, or_, prod
from aesara.tensor.math import sum as aet_sum
from aesara.tensor.subtensor import advanced_inc_subtensor1, set_subtensor
from aesara.tensor.type import (
    TensorType,
    dvector,
    int_dtypes,
    int_vector_types,
    integer_dtypes,
    vector,
)
from aesara.utils import LOCAL_BITWIDTH, PYTHON_INT_BITWIDTH


class CpuContiguous(COp):
    """
    Check to see if the input is c-contiguous,
    if it is, do nothing, else return a contiguous array.
    """

    __props__ = ()
    view_map = {0: [0]}
    check_input = False

    def make_node(self, x):
        x_ = aet.as_tensor_variable(x)
        return Apply(self, [x_], [x_.type()])

    def perform(self, node, inputs, output_storage):
        (x,) = inputs
        y = output_storage[0]
        # if the ouput is contiguous do nothing, else copy
        # the input
        if not x.flags["C_CONTIGUOUS"]:
            x = x.copy()
        assert x.flags["C_CONTIGUOUS"]
        y[0] = x

    def grad(self, inputs, dout):
        return [aet.as_tensor_variable(dout[0])]

    def c_code(self, node, name, inames, onames, sub):
        (x,) = inames
        (y,) = onames
        code = (
            """
            if (!PyArray_CHKFLAGS(%(x)s, NPY_ARRAY_C_CONTIGUOUS)){
                // check to see if output is contiguous first
                if (%(y)s != NULL &&
                    PyArray_CompareLists(PyArray_DIMS(%(y)s), PyArray_DIMS(%(x)s), PyArray_NDIM(%(x)s)) &&
                    PyArray_CHKFLAGS(%(y)s, NPY_ARRAY_C_CONTIGUOUS)){
                    PyArray_CopyInto(%(y)s, %(x)s);
                }
                else{
                    Py_XDECREF(%(y)s);
                    %(y)s = PyArray_GETCONTIGUOUS(%(x)s);
                }
            }
            else{
                Py_XINCREF(%(x)s);
                Py_XDECREF(%(y)s);
                %(y)s = %(x)s;
            }
            """
            % locals()
        )
        return code

    def c_code_cache_version(self):
        return (1,)


cpu_contiguous = CpuContiguous()


class SearchsortedOp(COp):
    """Wrapper of numpy.searchsorted.

    For full documentation, see :func:`searchsorted`.

    See Also
    --------
    searchsorted : numpy-like function to use the SearchsortedOp

    """

    params_type = Generic()
    __props__ = ("side",)
    check_input = False

    def __init__(self, side="left"):
        if side == "left" or side == "right":
            self.side = side
        else:
            raise ValueError(f"'{side}' is an invalid value for keyword 'side'")

    def get_params(self, node):
        return self.side

    def make_node(self, x, v, sorter=None):
        x = aet.as_tensor(x, ndim=1)
        v = aet.as_tensor(v)
        out_type = v.type.clone(dtype="int64")
        if sorter is None:
            return Apply(self, [x, v], [out_type()])
        else:
            sorter = aet.as_tensor(sorter, ndim=1)
            if PYTHON_INT_BITWIDTH == 32 and sorter.dtype == "int64":
                raise TypeError(
                    "numpy.searchsorted with Python 32bit do not support a"
                    " sorter of int64."
                )
            if sorter.type not in int_vector_types:
                raise TypeError("sorter must be an integer vector", sorter.type)
            return Apply(self, [x, v, sorter], [out_type()])

    def infer_shape(self, fgraph, node, shapes):
        return [shapes[1]]

    def perform(self, node, inputs, output_storage, params):
        x = inputs[0]
        v = inputs[1]
        if len(node.inputs) == 3:
            sorter = inputs[2]
        else:
            sorter = None
        z = output_storage[0]

        z[0] = np.searchsorted(x, v, side=params, sorter=sorter).astype(
            node.outputs[0].dtype
        )

    def c_support_code_struct(self, node, name):
        return f"""
            int right_{name};
        """

    def c_init_code_struct(self, node, name, sub):
        side = sub["params"]
        fail = sub["fail"]
        return (
            """
            PyObject* tmp_%(name)s = PyUnicode_FromString("right");
            if (tmp_%(name)s == NULL)
                %(fail)s;
            right_%(name)s = PyUnicode_Compare(%(side)s, tmp_%(name)s);
            Py_DECREF(tmp_%(name)s);
        """
            % locals()
        )

    def c_code(self, node, name, inames, onames, sub):
        sorter = None
        if len(node.inputs) == 3:
            x, v, sorter = inames
        else:
            x, v = inames
        if not sorter:
            sorter = "NULL"
        (z,) = onames
        fail = sub["fail"]

        return (
            """
            Py_XDECREF(%(z)s);
            %(z)s = (PyArrayObject*) PyArray_SearchSorted(%(x)s, (PyObject*) %(v)s,
                                                          right_%(name)s ? NPY_SEARCHLEFT : NPY_SEARCHRIGHT, (PyObject*) %(sorter)s);
            if (!%(z)s)
                %(fail)s;
            if (PyArray_TYPE(%(z)s) != NPY_INT64){
                PyObject * tmp = PyArray_Cast(%(z)s, NPY_INT64);
                Py_XDECREF(%(z)s);
                %(z)s = (PyArrayObject*) tmp;
            }
        """
            % locals()
        )

    def c_code_cache_version(self):
        return (2,)

    def grad(self, inputs, output_gradients):
        num_ins = len(inputs)
        if num_ins == 3:
            x, v, sorter = inputs
        else:
            x, v = inputs

        x_grad = _float_zeros_like(x)
        v_grad = _float_zeros_like(v)
        if num_ins == 3:
            return [x_grad, v_grad, disconnected_type()]
        else:
            return [x_grad, v_grad]


def searchsorted(x, v, side="left", sorter=None):
    """Find indices where elements should be inserted to maintain order.

    Wrapping of numpy.searchsorted. Find the indices into a sorted array
    `x` such that, if the corresponding elements in `v` were inserted
    before the indices, the order of `x` would be preserved.

    Parameters
    ----------
    x: 1-D tensor (array-like)
        Input array. If `sorter` is None, then it must be sorted in
        ascending order, otherwise `sorter` must be an array of indices
        which sorts it.
    v: tensor (array-like)
        Contains the values to be inserted into `x`.
    side: {'left', 'right'}, optional.
        If 'left' (default), the index of the first suitable
        location found is given. If 'right', return the last such index. If
        there is no suitable index, return either 0 or N (where N is the length
        of `x`).
    sorter: 1-D tensor of integers (array-like), optional
        Contains indices that sort array `x` into ascending order.
        They are typically the result of argsort.

    Returns
    -------
    indices : tensor of integers (int64)
        Array of insertion points with the same shape as `v`.

    See Also
    --------
    `numpy.searchsorted <https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.searchsorted.html>`_

    Notes
    -----
    * Binary search is used to find the required insertion points.
    * This Op is working **only on CPU** currently.

    Examples
    --------
    >>> from aesara import tensor as aet
    >>> from aesara.tensor import extra_ops
    >>> x = aet.dvector()
    >>> idx = x.searchsorted(3)
    >>> idx.eval({x: [1,2,3,4,5]})
    array(2)
    >>> extra_ops.searchsorted([1,2,3,4,5], 3).eval()
    array(2)
    >>> extra_ops.searchsorted([1,2,3,4,5], 3, side='right').eval()
    array(3)
    >>> extra_ops.searchsorted([1,2,3,4,5], [-10, 10, 2, 3]).eval()
    array([0, 5, 1, 2])

    .. versionadded:: 0.9

    """
    return SearchsortedOp(side=side)(x, v, sorter)


class CumOp(COp):
    # See function cumsum/cumprod for docstring

    __props__ = ("axis", "mode")
    check_input = False
    params_type = ParamsType(
        c_axis=int_t, mode=EnumList(("MODE_ADD", "add"), ("MODE_MUL", "mul"))
    )

    def __init__(self, axis=None, mode="add"):
        if mode not in ("add", "mul"):
            raise ValueError(f'{type(self).__name__}: Unknown mode "{mode}"')
        self.axis = axis
        self.mode = mode

    c_axis = property(lambda self: np.MAXDIMS if self.axis is None else self.axis)

    def make_node(self, x):
        x = aet.as_tensor_variable(x)
        out_type = x.type()

        if self.axis is None:
            out_type = vector(dtype=x.dtype)  # Flatten
        elif self.axis >= x.ndim or self.axis < -x.ndim:
            raise ValueError(f"axis(={self.axis}) out of bounds")

        return Apply(self, [x], [out_type])

    def perform(self, node, inputs, output_storage, params):
        x = inputs[0]
        z = output_storage[0]
        if self.mode == "add":
            z[0] = np.cumsum(x, axis=self.axis)
        else:
            z[0] = np.cumprod(x, axis=self.axis)

    def grad(self, inputs, output_gradients):
        (x,) = inputs
        (gi,) = output_gradients

        if self.axis is None:
            if self.mode == "add":
                return [cumsum(gi[::-1])[::-1].reshape(x.shape)]
            elif self.mode == "mul":
                fx = cumprod(x, axis=self.axis)
                return [cumsum((fx * gi)[::-1])[::-1].reshape(x.shape) / x]
            else:
                raise NotImplementedError(
                    f'{type(self).__name__}: unknown gradient for mode "{self.mode}"'
                )

        reverse_slicing = [slice(None, None, None)] * gi.ndim
        reverse_slicing[self.axis] = slice(None, None, -1)
        reverse_slicing = tuple(reverse_slicing)
        # We need to reverse the gradients along ``self.axis``,
        #  compute cumsum, then reverse again
        if self.mode == "add":
            return [cumsum(gi[reverse_slicing], self.axis)[reverse_slicing]]
        elif self.mode == "mul":
            fx = cumprod(x, axis=self.axis)
            return [cumsum((fx * gi)[reverse_slicing], self.axis)[reverse_slicing] / x]
        else:
            raise NotImplementedError(
                f'{type(self).__name__}: unknown gradient for mode "{self.mode}"'
            )

    def infer_shape(self, fgraph, node, shapes):
        if self.axis is None:
            return [(prod(shapes[0]),)]  # Flatten

        return shapes

    def c_code(self, node, name, inames, onames, sub):
        (x,) = inames
        (z,) = onames
        axis = self.axis
        fail = sub["fail"]
        params = sub["params"]

        code = (
            """
                int axis = %(params)s->c_axis;
                if (axis == 0 && PyArray_NDIM(%(x)s) == 1)
                    axis = NPY_MAXDIMS;
                npy_intp shape[1] = { PyArray_SIZE(%(x)s) };
                if(axis == NPY_MAXDIMS && !(%(z)s && PyArray_DIMS(%(z)s)[0] == shape[0]))
                {
                    Py_XDECREF(%(z)s);
                    %(z)s = (PyArrayObject*) PyArray_SimpleNew(1, shape, PyArray_TYPE((PyArrayObject*) py_%(x)s));
                }

                else if(axis != NPY_MAXDIMS && !(%(z)s && PyArray_CompareLists(PyArray_DIMS(%(z)s), PyArray_DIMS(%(x)s), PyArray_NDIM(%(x)s))))
                {
                    Py_XDECREF(%(z)s);
                    %(z)s = (PyArrayObject*) PyArray_SimpleNew(PyArray_NDIM(%(x)s), PyArray_DIMS(%(x)s), PyArray_TYPE(%(x)s));
                }

                if (!%(z)s)
                    %(fail)s;
                {

                    PyObject * t = NULL;
                    if(%(params)s->mode == MODE_ADD)
                        t = PyArray_CumSum(
                            %(x)s, axis,
                            PyArray_TYPE(%(x)s), %(z)s);
                    else if(%(params)s->mode == MODE_MUL)
                        t = PyArray_CumProd(
                            %(x)s, axis,
                            PyArray_TYPE(%(x)s), %(z)s);

                    if (!t){
                       %(fail)s;
                    }
                    // Because PyArray_CumSum/CumProd returns a newly created reference on t.
                    Py_XDECREF(t);
                }
            """
            % locals()
        )

        return code

    def c_code_cache_version(self):
        return (8,)

    def __str__(self):
        return f"{self.__class__.__name__}{{{self.axis}, {self.mode}}}"


def cumsum(x, axis=None):
    """Return the cumulative sum of the elements along a given axis.

    Wraping of numpy.cumsum.

    Parameters
    ----------
    x
        Input tensor variable.
    axis
        The axis along which the cumulative sum is computed.
        The default (None) is to compute the cumsum over the flattened array.


    .. versionadded:: 0.7

    """
    return CumOp(axis=axis, mode="add")(x)


def cumprod(x, axis=None):
    """Return the cumulative product of the elements along a given axis.

    Wraping of numpy.cumprod.

    Parameters
    ----------
    x
        Input tensor variable.

    axis
        The axis along which the cumulative product is computed.
        The default (None) is to compute the cumprod over the flattened array.


    .. versionadded:: 0.7

    """
    return CumOp(axis=axis, mode="mul")(x)


# CumsumOp and CumprodOp are for compatibility with old version,
# just in case unpickling an Aesara function with old Ops.
class CumsumOp(Op):
    __props__ = ("axis",)

    def __new__(typ, *args, **kwargs):
        obj = object.__new__(CumOp, *args, **kwargs)
        obj.mode = "add"
        return obj


class CumprodOp(Op):
    __props__ = ("axis",)

    def __new__(typ, *args, **kwargs):
        obj = object.__new__(CumOp, *args, **kwargs)
        obj.mode = "mul"
        return obj


class DiffOp(Op):
    # See function diff for docstring

    __props__ = ("n", "axis")

    def __init__(self, n=1, axis=-1):
        self.n = n
        self.axis = axis
        # numpy return a view in that case.
        # TODO, make an optimization that remove this op in this case.
        if n == 0:
            self.view_map = {0: [0]}

    def make_node(self, x):
        x = aet.as_tensor_variable(x)
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        z = output_storage[0]
        z[0] = np.diff(x, n=self.n, axis=self.axis)

    def grad(self, inputs, outputs_gradients):
        inputs = inputs[0]

        if inputs.ndim != 1:
            raise NotImplementedError(
                "Grad is not implemented for inputs with"
                "number of dimension other than 1."
            )

        z = outputs_gradients[0]

        def _grad_helper(z):
            pre = aet.concatenate([[0.0], z])
            app = aet.concatenate([z, [0.0]])
            return pre - app

        for k in range(self.n):
            z = _grad_helper(z)
        return [z]

    def infer_shape(self, fgraph, node, ins_shapes):
        i0_shapes = ins_shapes[0]
        out_shape = list(i0_shapes)
        out_shape[self.axis] = out_shape[self.axis] - self.n
        return [out_shape]


def diff(x, n=1, axis=-1):
    """Calculate the n-th order discrete difference along given axis.

    The first order difference is given by out[i] = a[i + 1] - a[i]
    along the given axis, higher order differences are calculated by
    using diff recursively. Wraping of numpy.diff.

    Parameters
    ----------
    x
        Input tensor variable.

    n
        The number of times values are differenced, default is 1.

    axis
        The axis along which the difference is taken, default is the last axis.


    .. versionadded:: 0.6

    """
    return DiffOp(n=n, axis=axis)(x)


def bincount(x, weights=None, minlength=None, assert_nonneg=False):
    """Count number of occurrences of each value in array of ints.

    The number of bins (of size 1) is one larger than the largest
    value in x. If minlength is specified, there will be at least
    this number of bins in the output array (though it will be longer
    if necessary, depending on the contents of x). Each bin gives the
    number of occurrences of its index value in x. If weights is
    specified the input array is weighted by it, i.e. if a value n
    is found at position i, out[n] += weight[i] instead of out[n] += 1.

    Parameters
    ----------
    x : 1 dimension, nonnegative ints
    weights : array of the same shape as x with corresponding weights.
        Optional.
    minlength : A minimum number of bins for the output array.
        Optional.
    assert_nonneg : A flag that inserts an assert_op to check if
        every input x is nonnegative.
        Optional.


    .. versionadded:: 0.6

    """
    if x.ndim != 1:
        raise TypeError("Inputs must be of dimension 1.")

    if assert_nonneg:
        assert_op = Assert("Input to bincount has negative values!")
        x = assert_op(x, aet_all(x >= 0))

    max_value = aet.cast(x.max() + 1, "int64")

    if minlength is not None:
        max_value = maximum(max_value, minlength)

    # Note: we do not use inc_subtensor(out[x], ...) in the following lines,
    # since out[x] raises an exception if the indices (x) are int8.
    if weights is None:
        out = aet.zeros([max_value], dtype=x.dtype)
        out = advanced_inc_subtensor1(out, 1, x)
    else:
        out = aet.zeros([max_value], dtype=weights.dtype)
        out = advanced_inc_subtensor1(out, weights, x)
    return out


def squeeze(x, axis=None):
    """
    Remove broadcastable dimensions from the shape of an array.

    It returns the input array, but with the
    broadcastable dimensions removed. This is
    always `x` itself or a view into `x`.

    .. versionadded:: 0.6

    Parameters
    ----------
    x
        Input data, tensor variable.

    axis : None or int or tuple of ints, optional

        Selects a subset of the single-dimensional entries in the
        shape. If an axis is selected with shape entry greater than
        one, an error is raised.

    Returns
    -------
    object
        `x` without its broadcastable dimensions.

    """
    if axis is None:
        axis = range(x.ndim)
    elif not isinstance(axis, Collection):
        axis = (axis,)

    view = x.dimshuffle(
        [i for i in range(x.ndim) if not x.broadcastable[i] or i not in axis]
    )
    return view


def compress(condition, x, axis=None):
    """
    Return selected slices of an array along given axis.

    It returns the input tensor, but with selected slices along a given axis
    retained. If no axis is provided, the tensor is flattened.
    Corresponds to numpy.compress

    .. versionadded:: 0.7

    Parameters
    ----------
    x
        Input data, tensor variable.
    condition
         1 dimensional array of non-zero and zero values
         corresponding to indices of slices along a selected axis.

    Returns
    -------
    object
        `x` with selected slices.

    """
    indices = aet.flatnonzero(condition)
    return x.take(indices, axis=axis)


class RepeatOp(Op):
    # See the repeat function for docstring

    __props__ = ("axis",)

    def __init__(self, axis=None):
        self.axis = axis

    def make_node(self, x, repeats):
        x = aet.as_tensor_variable(x)
        repeats = aet.as_tensor_variable(repeats)

        if repeats.dtype not in integer_dtypes:
            raise TypeError("repeats.dtype must be an integer.")

        # Some dtypes are not supported by numpy's implementation of repeat.
        # Until another one is available, we should fail at graph construction
        # time, not wait for execution.
        ptr_bitwidth = LOCAL_BITWIDTH
        if ptr_bitwidth == 64:
            numpy_unsupported_dtypes = ("uint64",)
        if ptr_bitwidth == 32:
            numpy_unsupported_dtypes = ("uint32", "int64", "uint64")

        if repeats.dtype in numpy_unsupported_dtypes:
            raise TypeError(
                (
                    "dtypes %s are not supported by numpy.repeat "
                    "for the 'repeats' parameter, " % str(numpy_unsupported_dtypes)
                ),
                repeats.dtype,
            )

        if self.axis is None:
            broadcastable = [False]
        else:
            try:
                const_reps = aet.get_scalar_constant_value(repeats)
            except NotScalarConstantError:
                const_reps = None
            if const_reps == 1:
                broadcastable = x.broadcastable
            else:
                broadcastable = list(x.broadcastable)
                broadcastable[self.axis] = False

        out_type = TensorType(x.dtype, broadcastable)

        return Apply(self, [x, repeats], [out_type()])

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        repeats = inputs[1]
        z = output_storage[0]
        z[0] = np.repeat(x, repeats=repeats, axis=self.axis)

    def connection_pattern(self, node):

        return [[True], [False]]

    def grad(self, inputs, gout):
        (x, repeats) = inputs
        (gz,) = gout
        if repeats.ndim == 0:
            if self.axis is None:
                axis = x.ndim
            else:
                if self.axis >= 0:
                    axis = self.axis + 1
                else:
                    axis = self.axis + x.ndim + 1

            shape = [x.shape[k] for k in range(x.ndim)]
            shape.insert(axis, repeats)

            return [gz.reshape(shape, x.ndim + 1).sum(axis=axis), DisconnectedType()()]
        elif repeats.ndim == 1:
            # For this implementation, we would need to specify the length
            # of repeats in order to split gz in the right way to sum
            # the good part.
            raise NotImplementedError()
        else:
            raise ValueError()

    def infer_shape(self, fgraph, node, ins_shapes):
        i0_shapes = ins_shapes[0]
        repeats = node.inputs[1]
        out_shape = list(i0_shapes)

        # uint64 shape are not supported.
        dtype = None
        if repeats.dtype in ["uint8", "uint16", "uint32"]:
            dtype = "int64"
        if self.axis is None:
            if repeats.ndim == 0:
                if len(i0_shapes) == 0:
                    out_shape = [repeats]
                else:
                    res = 1
                    for d in i0_shapes:
                        res = res * d
                    out_shape = (res * repeats,)
            else:
                out_shape = [aet_sum(repeats, dtype=dtype)]
        else:
            if repeats.ndim == 0:
                out_shape[self.axis] = out_shape[self.axis] * repeats
            else:
                out_shape[self.axis] = aet_sum(repeats, dtype=dtype)
        return [out_shape]


def repeat(x, repeats, axis=None):
    """Repeat elements of an array.

    It returns an array which has the same shape as `x`, except
    along the given axis. The axis is used to speficy along which
    axis to repeat values. By default, use the flattened input
    array, and return a flat output array.

    The number of repetitions for each element is `repeat`.
    `repeats` is broadcasted to fit the length of the given `axis`.

    Parameters
    ----------
    x
        Input data, tensor variable.
    repeats
        int, scalar or tensor variable
    axis : int, optional

    See Also
    --------
    tensor.tile

    .. versionadded:: 0.6

    """
    repeats = aet.as_tensor_variable(repeats)

    if repeats.ndim > 1:
        raise ValueError("The dimension of repeats should not exceed 1.")

    if repeats.ndim == 1 and not repeats.broadcastable[0]:
        return RepeatOp(axis=axis)(x, repeats)
    else:
        if repeats.ndim == 1:
            repeats = repeats[0]

        if x.dtype == "uint64":
            raise TypeError("repeat doesn't support dtype uint64")

        if axis is None:
            axis = 0
            x = x.flatten()
        else:
            if axis >= x.ndim:
                raise ValueError("Axis should not exceed x.ndim-1.")
            if axis < 0:
                axis = x.ndim + axis

        shape = [x.shape[i] for i in range(x.ndim)]

        # shape_ is the shape of the intermediate tensor which has
        # an additional dimension comparing to x. We use alloc to
        # allocate space for this intermediate tensor to replicate x
        # along that additional dimension.
        shape_ = shape[:]
        shape_.insert(axis + 1, repeats)

        # shape is now the shape of output, where shape[axis] becomes
        # shape[axis]*repeats.
        shape[axis] = shape[axis] * repeats

        # dims_ is the dimension of that intermediate tensor.
        dims_ = list(np.arange(x.ndim))
        dims_.insert(axis + 1, "x")

        # After the original tensor is duplicated along the additional
        # dimension, we reshape it to the expected output shape, and
        # return the output z.
        z = aet.alloc(x.dimshuffle(*dims_), *shape_).reshape(shape)
        return z


class Bartlett(Op):
    # See function bartlett for docstring
    __props__ = ()

    def make_node(self, M):
        M = aet.as_tensor_variable(M)
        if M.ndim != 0:
            raise TypeError(f"{self.__class__.__name__} only works on scalar input")
        elif M.dtype not in integer_dtypes:
            # dtype is an Aesara attribute here
            raise TypeError(f"{self.__class__.__name__} only works on integer input")
        return Apply(self, [M], [dvector()])

    def perform(self, node, inputs, out_):
        M = inputs[0]
        (out,) = out_
        out[0] = np.bartlett(M)

    def infer_shape(self, fgraph, node, in_shapes):
        temp = node.inputs[0]
        M = aet.switch(lt(temp, 0), aet.cast(0, temp.dtype), temp)
        return [[M]]

    def grad(self, inputs, output_grads):
        return [None for i in inputs]


bartlett_ = Bartlett()


# I create a function only to have the doc show well.
def bartlett(M):
    """
    An instance of this class returns the Bartlett spectral window in the
    time-domain. The Bartlett window is very similar to a triangular window,
    except that the end points are at zero. It is often used in signal
    processing for tapering a signal, without generating too much ripple in
    the frequency domain.

    .. versionadded:: 0.6

    Parameters
    ----------
    M : integer scalar
        Number of points in the output window. If zero or less,
        an empty vector is returned.

    Returns
    -------
    vector of doubles
        The triangular window, with the maximum value normalized to one
        (the value one appears only if the number of samples is odd), with
        the first and last samples equal to zero.

    """
    return bartlett_(M)


class FillDiagonal(Op):
    # See function fill_diagonal for docstring
    __props__ = ()

    def infer_shape(self, fgraph, node, in_shapes):
        return [in_shapes[0]]

    def make_node(self, a, val):
        a = aet.as_tensor_variable(a)
        val = aet.as_tensor_variable(val)
        if a.ndim < 2:
            raise TypeError(
                "%s: first parameter must have at least"
                " two dimensions" % self.__class__.__name__
            )
        elif val.ndim != 0:
            raise TypeError(
                f"{self.__class__.__name__}: second parameter must be a scalar"
            )
        val = aet.cast(val, dtype=upcast(a.dtype, val.dtype))
        if val.dtype != a.dtype:
            raise TypeError(
                "%s: type of second parameter must be the same as"
                " the first's" % self.__class__.__name__
            )
        return Apply(self, [a, val], [a.type()])

    def perform(self, node, inputs, output_storage):
        a = inputs[0].copy()
        val = inputs[1]
        if a.ndim == 2:
            # numpy.fill_diagonal up to date(including 1.6.2) have a
            # bug for tall matrix.
            # For 2-d arrays, we accept rectangular ones.
            step = a.shape[1] + 1
            end = a.shape[1] * a.shape[1]
            # Write the value out into the diagonal.
            a.flat[:end:step] = val
        else:
            np.fill_diagonal(a, val)

        output_storage[0][0] = a

    def grad(self, inp, cost_grad):
        """
        Notes
        -----
        The gradient is currently implemented for matrices only.

        """
        a, val = inp
        grad = cost_grad[0]
        if a.dtype.startswith("complex"):
            return [None, None]
        elif a.ndim > 2:
            raise NotImplementedError(
                "%s: gradient is currently implemented"
                " for matrices only" % self.__class__.__name__
            )
        wr_a = fill_diagonal(grad, 0)  # valid for any number of dimensions
        # diag is only valid for matrices
        wr_val = aet.diag(grad).sum()
        return [wr_a, wr_val]


fill_diagonal_ = FillDiagonal()


# I create a function only to have the doc show well.
def fill_diagonal(a, val):
    """
    Returns a copy of an array with all
    elements of the main diagonal set to a specified scalar value.

    .. versionadded:: 0.6

    Parameters
    ----------
    a
        Rectangular array of at least two dimensions.
    val
        Scalar value to fill the diagonal whose type must be
        compatible with that of array 'a' (i.e. 'val' cannot be viewed
        as an upcast of 'a').

    Returns
    -------
    array
        An array identical to 'a' except that its main diagonal
        is filled with scalar 'val'. (For an array 'a' with a.ndim >=
        2, the main diagonal is the list of locations a[i, i, ..., i]
        (i.e. with indices all identical).)

    Support rectangular matrix and tensor with more than 2 dimensions
    if the later have all dimensions are equals.



    """
    return fill_diagonal_(a, val)


class FillDiagonalOffset(Op):
    # See function fill_diagonal_offset for docstring
    __props__ = ()

    def infer_shape(self, fgraph, node, in_shapes):
        return [in_shapes[0]]

    def make_node(self, a, val, offset):
        a = aet.as_tensor_variable(a)
        val = aet.as_tensor_variable(val)
        offset = aet.as_tensor_variable(offset)
        if a.ndim != 2:
            raise TypeError(
                "%s: first parameter must have exactly"
                " two dimensions" % self.__class__.__name__
            )
        elif val.ndim != 0:
            raise TypeError(
                f"{self.__class__.__name__}: second parameter must be a scalar"
            )
        elif offset.ndim != 0:
            raise TypeError(
                f"{self.__class__.__name__}: third parameter must be a scalar"
            )
        val = aet.cast(val, dtype=upcast(a.dtype, val.dtype))
        if val.dtype != a.dtype:
            raise TypeError(
                "%s: type of second parameter must be the same"
                " as the first's" % self.__class__.__name__
            )
        elif offset.dtype not in integer_dtypes:
            raise TypeError(
                f"{self.__class__.__name__}: type of third parameter must be as integer"
                " use aesara.tensor.cast( input, 'int32/int64')"
            )

        return Apply(self, [a, val, offset], [a.type()])

    def perform(self, node, inputs, output_storage):
        a = inputs[0].copy()
        val = inputs[1]
        offset = inputs[2]
        height, width = a.shape

        """
        Notes
        -----
        The fill_diagonal only support rectangular matrix. The output
        of tall matrix is "wrapped", which is an option in numpy 1.9.0
        but was regarded as a bug in numpy 1.6.2. Here I implement the
        fill_diagonal_offset with unwrapped output, so fill_diagonal_offset
        supports tall matrix.(This make a little difference between the output
        of fill_diagonal and fill_diagonal_offset only in the case of tall
        matrix)

        """
        if offset >= 0:
            start = offset
            num_of_step = min(min(width, height), width - offset)
        else:
            start = -offset * a.shape[1]
            num_of_step = min(min(width, height), height + offset)
        step = a.shape[1] + 1
        end = start + step * num_of_step
        # Write the value out into the diagonal.
        a.flat[start:end:step] = val

        output_storage[0][0] = a

    def grad(self, inp, cost_grad):
        """
        Notes
        -----
        The gradient is currently implemented for matrices only.
        """
        a, val, offset = inp
        grad = cost_grad[0]
        height, width = grad.shape

        if a.dtype.startswith("complex"):
            return [None, None]

        # only valid for matrices
        wr_a = fill_diagonal_offset(grad, 0, offset)

        offset_abs = abs_(offset)
        pos_offset_flag = ge(offset, 0)
        neg_offset_flag = lt(offset, 0)
        min_wh = minimum(width, height)

        start = offset * pos_offset_flag + offset_abs * width * neg_offset_flag
        num_of_step = minimum(
            min_wh, width * pos_offset_flag + height * neg_offset_flag - offset_abs
        )

        step = a.shape[1] + 1
        end = start + step * num_of_step

        # input of slice should be integer
        start = aet.cast(start, "int32")
        step = aet.cast(step, "int32")
        end = aet.cast(end, "int32")

        wr_val = grad.flatten()[start:end:step].sum()

        wr_offset = grad_undefined(
            self,
            2,
            offset,
            "offset is not defined for non-integer offset so"
            " fill_diagonal_offset(a,val,offset+eps) is undefined",
        )

        return [wr_a, wr_val, wr_offset]


fill_diagonal_offset_ = FillDiagonalOffset()


def fill_diagonal_offset(a, val, offset):
    """
    Returns a copy of an array with all
    elements of the main diagonal set to a specified scalar value.

    Parameters
    ----------
    a
        Rectangular array of two dimensions.
    val
        Scalar value to fill the diagonal whose type must be
        compatible with that of array 'a' (i.e. 'val' cannot be viewed
        as an upcast of 'a').
    offset
        Scalar value Offset of the diagonal from the main
        diagonal. Can be positive or negative integer.

    Returns
    -------
    array
        An array identical to 'a' except that its offset diagonal
        is filled with scalar 'val'. The output is unwrapped.

    """
    return fill_diagonal_offset_(a, val, offset)


def to_one_hot(y, nb_class, dtype=None):
    """
    Return a matrix where each row correspond to the one hot
    encoding of each element in y.

    Parameters
    ----------
    y
        A vector of integer value between 0 and nb_class - 1.
    nb_class : int
        The number of class in y.
    dtype : data-type
        The dtype of the returned matrix. Default floatX.

    Returns
    -------
    object
        A matrix of shape (y.shape[0], nb_class), where each row ``i`` is
        the one hot encoding of the corresponding ``y[i]`` value.

    """
    ret = aet.zeros((y.shape[0], nb_class), dtype=dtype)
    ret = set_subtensor(ret[aet.arange(y.shape[0]), y], 1)
    return ret


class Unique(Op):
    """
    Wraps numpy.unique. This op is not implemented on the GPU.

    Examples
    --------
    >>> import numpy as np
    >>> import aesara

    >>> x = aesara.tensor.vector()
    >>> f = aesara.function([x], Unique(True, True, False)(x))
    >>> f([1, 2., 3, 4, 3, 2, 1.])
    [array([ 1.,  2.,  3.,  4.]), array([0, 1, 2, 3]), array([0, 1, 2, 3, 2, 1, 0])]

    >>> y = aesara.tensor.matrix()
    >>> g = aesara.function([y], Unique(True, True, False)(y))
    >>> g([[1, 1, 1.0], (2, 3, 3.0)])
    [array([ 1.,  2.,  3.]), array([0, 3, 4]), array([0, 0, 0, 1, 2, 2])]

    """

    __props__ = ("return_index", "return_inverse", "return_counts", "axis")

    def __init__(
        self, return_index=False, return_inverse=False, return_counts=False, axis=None
    ):
        self.return_index = return_index
        self.return_inverse = return_inverse
        self.return_counts = return_counts
        self.axis = axis
        numpy_ver = [int(n) for n in np.__version__.split(".")[:2]]
        if self.axis is not None and bool(numpy_ver < [1, 13]):
            raise RuntimeError(
                "Numpy version = "
                + np.__version__
                + f". Option 'axis={axis}' works starting from version 1.13.0."
            )

    def make_node(self, x):
        x = aet.as_tensor_variable(x)
        self_axis = self.axis
        if self_axis is None:
            broadcastable = [False]
        else:
            if self_axis < 0:
                self_axis += len(x.broadcastable)
            if self_axis < 0 or self_axis >= len(x.broadcastable):
                raise RuntimeError(
                    "Unique axis `{}` is outside of input ndim = "
                    "{}.".format(self.axis, len(x.broadcastable))
                )
            broadcastable = [
                b if axis != self_axis else False
                for axis, b in enumerate(x.broadcastable)
            ]
        outputs = [TensorType(broadcastable=broadcastable, dtype=x.dtype)()]
        typ = TensorType(broadcastable=[False], dtype="int64")
        if self.return_index:
            outputs.append(typ())
        if self.return_inverse:
            outputs.append(typ())
        if self.return_counts:
            outputs.append(typ())
        return Apply(self, [x], outputs)

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        z = output_storage
        param = {}
        if self.return_index:
            param["return_index"] = True
        if self.return_inverse:
            param["return_inverse"] = True
        if self.return_counts:
            param["return_counts"] = True
        if self.axis is not None:
            param["axis"] = self.axis
        outs = np.unique(x, **param)
        if (
            (not self.return_inverse)
            and (not self.return_index)
            and (not self.return_counts)
        ):
            z[0][0] = outs
        else:
            for i in range(len(outs)):
                z[i][0] = outs[i]

    def infer_shape(self, fgraph, node, i0_shapes):
        ret = fgraph.shape_feature.default_infer_shape(fgraph, node, i0_shapes)
        if self.axis is not None:
            self_axis = self.axis
            ndim = len(i0_shapes[0])
            if self_axis < 0:
                self_axis += ndim
            if self_axis < 0 or self_axis >= ndim:
                raise RuntimeError(
                    f"Unique axis `{self.axis}` is outside of input ndim = {ndim}."
                )
            ret[0] = tuple(
                [fgraph.shape_feature.shape_ir(i, node.outputs[0]) for i in range(ndim)]
            )
        if self.return_inverse:
            if self.axis is None:
                shape = (prod(i0_shapes[0]),)
            else:
                shape = (i0_shapes[0][self_axis],)
            if self.return_index:
                ret[2] = shape
                return ret
            ret[1] = shape
            return ret
        return ret

    def __setstate__(self, state):
        self.__dict__.update(state)
        # For backwards compatibility with pickled instances of Unique that
        # did not have the axis parameter specified
        if "axis" not in state:
            self.axis = None


def unique(
    ar, return_index=False, return_inverse=False, return_counts=False, axis=None
):
    """Find the unique elements of an array.

    Returns the sorted unique elements of an array. There are three optional
    outputs in addition to the unique elements:

    * the indices of the input array that give the unique values
    * the indices of the unique array that reconstruct the input array
    * the number of times each unique value comes up in the input array
    """
    return Unique(return_index, return_inverse, return_counts, axis)(ar)


class UnravelIndex(Op):
    __props__ = ("order",)

    def __init__(self, order="C"):
        assert order in ("C", "F")
        self.order = order

    def make_node(self, indices, dims):
        indices = aet.as_tensor_variable(indices)
        dims = aet.as_tensor_variable(dims)

        if indices.dtype not in int_dtypes:
            raise TypeError(
                f"'{indices.dtype}' object cannot be interpreted as an index"
            )
        if dims.dtype not in int_dtypes:
            raise TypeError(f"'{dims.dtype}' object cannot be interpreted as an index")
        if dims.ndim != 1:
            raise TypeError("dims must be a 1D array")

        return Apply(
            self,
            [indices, dims],
            [
                TensorType(dtype="int64", broadcastable=(False,) * indices.ndim)()
                for i in range(aet.get_vector_length(dims))
            ],
        )

    def infer_shape(self, fgraph, node, input_shapes):
        return [input_shapes[0]] * len(node.outputs)

    def perform(self, node, inp, out):
        indices, dims = inp
        res = np.unravel_index(indices, dims, order=self.order)
        assert len(res) == len(out)
        for i in range(len(out)):
            ret = _asarray(res[i], node.outputs[0].dtype)
            if ret.base is not None:
                # NumPy will return a view when it can.
                # But we don't want that.
                ret = ret.copy()
            out[i][0] = ret


def unravel_index(indices, dims, order="C"):
    """
    Converts a flat index or array of flat indices into a tuple
    of coordinate arrays.

    Parameters
    ----------
    indices : Aesara or NumPy array
        An integer array whose elements are indices into the flattened
        version of an array of dimensions ``dims``.
    dims : tuple of ints
        The shape of the array to use for unraveling ``indices``.
    order : {'C', 'F'}, optional
        Determines whether the indices should be viewed as indexing in
        row-major (C-style) or column-major (Fortran-style) order.

    Returns
    -------
    unraveled_coords : tuple of ndarray
        Each array in the tuple has the same shape as the ``indices``
        array.

    See Also
    --------
    ravel_multi_index

    """
    res = UnravelIndex(order=order)(indices, dims)
    if not isinstance(res, (list, tuple)):
        return (res,)
    else:
        return tuple(res)


class RavelMultiIndex(Op):
    __props__ = ("mode", "order")

    def __init__(self, mode="raise", order="C"):
        assert mode in ("raise", "wrap", "clip")
        assert order in ("C", "F")
        self.mode = mode
        self.order = order

    def make_node(self, *inp):
        multi_index = [aet.as_tensor_variable(i) for i in inp[:-1]]
        dims = aet.as_tensor_variable(inp[-1])

        for i in multi_index:
            if i.dtype not in int_dtypes:
                raise TypeError(f"'{i.dtype}' object cannot be interpreted as an index")
        if dims.dtype not in int_dtypes:
            raise TypeError(f"'{dims.dtype}' object cannot be interpreted as an index")
        if dims.ndim != 1:
            raise TypeError("dims must be a 1D array")

        return Apply(
            self,
            multi_index + [dims],
            [TensorType(dtype="int64", broadcastable=(False,) * multi_index[0].ndim)()],
        )

    def infer_shape(self, fgraph, node, input_shapes):
        return [input_shapes[0]]

    def perform(self, node, inp, out):
        multi_index, dims = inp[:-1], inp[-1]
        res = np.ravel_multi_index(multi_index, dims, mode=self.mode, order=self.order)
        out[0][0] = _asarray(res, node.outputs[0].dtype)


def ravel_multi_index(multi_index, dims, mode="raise", order="C"):
    """
    Converts a tuple of index arrays into an array of flat
    indices, applying boundary modes to the multi-index.

    Parameters
    ----------
    multi_index : tuple of Aesara or NumPy arrays
        A tuple of integer arrays, one array for each dimension.
    dims : tuple of ints
        The shape of array into which the indices from ``multi_index`` apply.
    mode : {'raise', 'wrap', 'clip'}, optional
        Specifies how out-of-bounds indices are handled.  Can specify
        either one mode or a tuple of modes, one mode per index.
        * 'raise' -- raise an error (default)
        * 'wrap' -- wrap around
        * 'clip' -- clip to the range
        In 'clip' mode, a negative index which would normally
        wrap will clip to 0 instead.
    order : {'C', 'F'}, optional
        Determines whether the multi-index should be viewed as
        indexing in row-major (C-style) or column-major
        (Fortran-style) order.

    Returns
    -------
    raveled_indices : Aesara array
        An array of indices into the flattened version of an array
        of dimensions ``dims``.

    See Also
    --------
    unravel_index

    """
    if not isinstance(multi_index, (tuple, list)):
        raise TypeError("multi_index must be a tuple or a list.")
    args = tuple(multi_index) + (dims,)
    return RavelMultiIndex(mode=mode, order=order)(*args)


def broadcast_shape(*arrays, **kwargs):
    """Compute the shape resulting from broadcasting arrays.

    Parameters
    ----------
    *arrays: `TensorVariable`s
        The tensor variables, or their shapes (as tuples),
        for which the broadcast shape is computed.
    arrays_are_shapes: bool (Optional)
        Indicates whether or not the `arrays` contains shape tuples.
        If you use this approach, make sure that the broadcastable dimensions
        are (scalar) constants with the value `1` or `1` exactly.

    """
    return broadcast_shape_iter(arrays, **kwargs)


def broadcast_shape_iter(arrays, **kwargs):
    """Compute the shape resulting from broadcasting arrays.

    Parameters
    ----------
    arrays: Iterable[TensorVariable] or Iterable[Tuple[Variable]]
        An iterable of tensors, or a tuple of shapes (as tuples),
        for which the broadcast shape is computed.
        XXX: Do not call this with a generator/iterator; this function will not
        make copies!
    arrays_are_shapes: bool (Optional)
        Indicates whether or not the `arrays` contains shape tuples.
        If you use this approach, make sure that the broadcastable dimensions
        are (scalar) constants with the value `1` or `1` exactly.

    """
    one = aesara.scalar.ScalarConstant(aesara.scalar.int64, 1)

    arrays_are_shapes = kwargs.pop("arrays_are_shapes", False)
    if arrays_are_shapes:
        max_dims = max(len(a) for a in arrays)

        array_shapes = [
            (one,) * (max_dims - len(a))
            + tuple(one if getattr(sh, "value", sh) == 1 else sh for sh in a)
            for a in arrays
        ]
    else:
        max_dims = max(a.ndim for a in arrays)

        array_shapes = [
            (one,) * (max_dims - a.ndim)
            + tuple(one if bcast else sh for sh, bcast in zip(a.shape, a.broadcastable))
            for a in arrays
        ]

    result_dims = []

    for dim_shapes in zip(*array_shapes):
        non_bcast_shapes = [shape for shape in dim_shapes if shape != one]

        if len(non_bcast_shapes) > 0:
            # Either there's only one non-broadcastable dimensions--and that's
            # what determines the dimension size, or there are multiple
            # non-broadcastable dimensions that must be equal
            i_dim = non_bcast_shapes.pop()

            potentially_unequal_dims = [
                dim
                for dim in non_bcast_shapes
                # TODO FIXME: This is a largely deficient means of comparing graphs
                # (and especially shapes)
                if not equal_computations([i_dim], [dim])
            ]

            if potentially_unequal_dims:
                # In this case, we can't tell whether or not the dimensions are
                # equal, so we'll need to assert their equality and move the error
                # handling to evaluation time.
                assert_dim = Assert("Could not broadcast dimensions")
                eq_condition = aet_all(
                    [
                        or_(eq(dim, one), eq(i_dim, dim))
                        for dim in potentially_unequal_dims
                    ]
                )
                eq_condition = or_(eq(i_dim, one), eq_condition)
                result_dims.append(assert_dim(i_dim, eq_condition))
            else:
                result_dims.append(i_dim)
        else:
            # Every array was broadcastable in this dimension
            result_dims.append(one)

    return tuple(result_dims)


class BroadcastTo(Op):

    view_map = {0: [0]}

    def __call__(self, a, shape, **kwargs):
        return super().__call__(a, *shape, **kwargs)

    def make_node(self, a, *shape):
        a = aet.as_tensor_variable(a)
        shape = aet.as_tensor_variable(shape, ndim=1)

        shape, bcast = aet.alloc_validate_shape(shape)

        out = type(a.type)(dtype=a.type.dtype, broadcastable=bcast)()

        # Attempt to prevent in-place operations on this view-based output
        out.tag.indestructible = True

        return Apply(self, [a] + shape, [out])

    def perform(self, node, inputs, output_storage):
        a, *shape = inputs
        z = output_storage[0]
        z[0] = np.broadcast_to(a, shape)

    def grad(self, inputs, outputs_gradients):
        a, *shape = inputs
        (dout,) = outputs_gradients

        # Determine the dimensions that were added by broadcasting
        new_dims = list(range(dout.ndim - a.ndim))

        d_wrt_a = broadcast_to(dout, shape).sum(axis=new_dims)

        # Determine the dimensions that were broadcast
        _, shape_bcast = aet.alloc_validate_shape(shape)
        bcast_sums = [
            i
            for i, (a_b, s_b) in enumerate(zip(a.broadcastable, shape_bcast[-a.ndim :]))
            if a_b and not s_b
        ]

        if bcast_sums:
            d_wrt_a = d_wrt_a.sum(axis=bcast_sums, keepdims=True)

        return [d_wrt_a] + [
            grad_undefined(self, i, shp) for i, shp in enumerate(shape, 1)
        ]

    def infer_shape(self, fgraph, node, ins_shapes):
        return [node.inputs[1:]]

    def __str__(self):
        return f"{type(self).__name__}"

    def __repr__(self):
        return f"{type(self).__name__}()"


broadcast_to = BroadcastTo()
