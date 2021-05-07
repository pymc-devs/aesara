import warnings

import numpy as np

from aesara import config, printing
from aesara import scalar as aes
from aesara.gradient import DisconnectedType
from aesara.graph.basic import Apply, Variable
from aesara.graph.op import COp, Op
from aesara.graph.params_type import ParamsType
from aesara.graph.type import Generic
from aesara.misc.safe_asarray import _asarray
from aesara.printing import pprint
from aesara.scalar.basic import BinaryScalarOp
from aesara.tensor.basic import (
    alloc,
    arange,
    as_tensor_variable,
    cast,
    concatenate,
    constant,
    patternbroadcast,
    stack,
    switch,
)
from aesara.tensor.elemwise import (
    CAReduce,
    CAReduceDtype,
    DimShuffle,
    Elemwise,
    scalar_elemwise,
)
from aesara.tensor.shape import shape
from aesara.tensor.type import (
    complex_dtypes,
    continuous_dtypes,
    discrete_dtypes,
    int_dtypes,
    integer_dtypes,
    tensor,
    uint_dtypes,
)
from aesara.tensor.type_other import NoneConst
from aesara.tensor.utils import as_list
from aesara.tensor.var import TensorConstant, _tensor_py_operators


if int(config.tensor__cmp_sloppy) > 1:
    # This config variable is a quick-and-dirty way to get low-precision
    # comparisons.  For a more precise setting of these tolerances set
    # them explicitly in your user code by assigning, for example,
    # "aesara.tensor.math.float32_atol = ..."

    # When config.tensor__cmp_sloppy>1 we are even more sloppy. This is
    # useful to test the GPU as they don't use extended precision and
    # this cause some difference bigger then the normal sloppy.
    float16_atol = 1e-2
    float16_rtol = 5e-2

    float32_atol = 5e-4
    float32_rtol = 1e-3

    float64_rtol = 1e-4
    float64_atol = 1e-3
elif int(config.tensor__cmp_sloppy):
    float16_atol = 5e-3
    float16_rtol = 1e-2

    float32_atol = 1e-4
    float32_rtol = 1e-3

    float64_rtol = 1e-4
    float64_atol = 1e-3
else:
    # If you change those value in test don't forget to put them back
    # when the test end.  Don't forget the case when the test fail.
    float16_atol = 1e-3
    float16_rtol = 1e-3

    float32_atol = 1e-5
    float32_rtol = 1e-5

    # defaults in numpy.allclose
    # Don't be more strict then numpy rtol
    # It cause useless error.
    float64_rtol = 1.0000000000000001e-05
    float64_atol = 1e-8


def _get_atol_rtol(a, b):
    tiny = ("float16",)
    narrow = ("float32", "complex64")
    if (str(a.dtype) in tiny) or (str(b.dtype) in tiny):
        atol = float16_atol
        rtol = float16_rtol
    elif (str(a.dtype) in narrow) or (str(b.dtype) in narrow):
        atol = float32_atol
        rtol = float32_rtol
    else:
        atol = float64_atol
        rtol = float64_rtol
    return atol, rtol


def _allclose(a, b, rtol=None, atol=None):
    a = np.asarray(a)
    b = np.asarray(b)
    atol_, rtol_ = _get_atol_rtol(a, b)
    if rtol is not None:
        rtol_ = rtol
    if atol is not None:
        atol_ = atol

    return np.allclose(a, b, atol=atol_, rtol=rtol_)


class MaxAndArgmax(COp):
    """
    Calculate the max and argmax over a given axis or over all axes.

    """

    nin = 2  # tensor, axis
    nout = 2  # max val, max idx
    E_axis = "invalid axis"
    params_type = Generic()
    __props__ = ("axis",)
    _f16_ok = True

    def __init__(self, axis):
        assert isinstance(axis, list)
        self.axis = tuple(axis)

    def get_params(self, node):
        return self.axis

    def make_node(self, x):
        x = as_tensor_variable(x)

        # We keep the original broadcastable flags for dimensions on which
        # we do not perform the max / argmax.
        all_axes = set(self.axis)
        broadcastable = [
            b for i, b in enumerate(x.type.broadcastable) if i not in all_axes
        ]
        inputs = [x]
        outputs = [
            tensor(x.type.dtype, broadcastable, name="max"),
            tensor("int64", broadcastable, name="argmax"),
        ]
        return Apply(self, inputs, outputs)

    def perform(self, node, inp, outs, params):
        x = inp[0]
        axes = params
        max, max_idx = outs
        if axes is None:
            axes = tuple(range(x.ndim))
        else:
            axes = tuple(int(ax) for ax in axes)
        max[0] = _asarray(np.max(x, axes), dtype=node.outputs[0].dtype)
        # Numpy does not support multiple axes for argmax
        # Work around
        keep_axes = np.array([i for i in range(x.ndim) if i not in axes], dtype="int64")
        # Not-reduced axes in front
        transposed_x = np.transpose(x, np.concatenate((keep_axes, axes)))
        kept_shape = transposed_x.shape[: len(keep_axes)]
        reduced_shape = transposed_x.shape[len(keep_axes) :]

        # Numpy.prod returns 1.0 when arg is empty, so we cast it to int64
        # Otherwise reshape would complain citing float arg
        new_shape = kept_shape + (np.prod(reduced_shape, dtype="int64"),)
        reshaped_x = transposed_x.reshape(new_shape)

        max_idx[0] = _asarray(np.argmax(reshaped_x, axis=-1), dtype="int64")

    def c_code(self, node, name, inp, out, sub):
        if len(self.axis) != 1 and len(self.axis) != node.inputs[0].ndim:
            raise NotImplementedError(
                "NumPy C-API can compute max and argmax only for 1 axis or for all axes."
            )
        x = inp[0]
        axis = sub["params"]
        max, argmax = out
        fail = sub["fail"]
        ret = """
        #if PY_MAJOR_VERSION >= 3
            #ifndef PyInt_AS_LONG
                #define PyInt_AS_LONG PyLong_AS_LONG
            #endif
        #endif

        int axis;

        if (PyTuple_GET_SIZE(%(axis)s) == PyArray_NDIM(%(x)s)) {
            axis = NPY_MAXDIMS;
        } else if(PyTuple_GET_SIZE(%(axis)s) == 1) {
            PyObject* axis_object = PyTuple_GET_ITEM(%(axis)s, 0);
            axis = (int)PyInt_AS_LONG(axis_object);
            if (axis > PyArray_NDIM(%(x)s)-1 || axis < -PyArray_NDIM(%(x)s)) {
                PyErr_SetString(PyExc_ValueError,
                "MaxAndArgmax: bad axis argument");
                %(fail)s
            }
        } else {
            PyErr_SetString(PyExc_NotImplementedError,
            "MaxAndArgmax: NumPy C-API can compute max and argmax only for 1 axis or for all axes.");
            %(fail)s
        }

        Py_CLEAR(%(max)s);
        Py_CLEAR(%(argmax)s);//todo pass them as out parameter.

        %(max)s = (PyArrayObject*)PyArray_Max(%(x)s, axis, NULL);
        if (%(max)s == NULL) {
            %(fail)s;
        }
        if (!PyArray_CheckExact(%(max)s)) {
            %(max)s = (PyArrayObject*)PyArray_FromAny((PyObject*)%(max)s, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);
            if(%(max)s == NULL){
                %(fail)s;
            }
        }

        %(argmax)s = (PyArrayObject*)PyArray_ArgMax(%(x)s, axis, NULL);
        if (%(argmax)s == NULL) {
            Py_CLEAR(%(max)s);
            %(fail)s;
        }
        if (!PyArray_CheckExact(%(argmax)s)) {
            %(argmax)s = (PyArrayObject*)PyArray_FromAny((PyObject*)%(argmax)s, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);
            if(%(argmax)s == NULL){
                %(fail)s;
            }
        }
        if (PyArray_TYPE(%(argmax)s) != NPY_INT64) {
            PyObject * tmp = PyArray_Cast(%(argmax)s, NPY_INT64);
            if (NULL == tmp){
                %(fail)s;
            }
            Py_DECREF(%(argmax)s);
            %(argmax)s = (PyArrayObject*)tmp;
        }
        """
        return ret % locals()

    def c_code_cache_version(self):
        return (5,)

    def infer_shape(self, fgraph, node, shapes):
        ishape = shapes[0]
        rval = tuple(
            ishape[i]
            for (i, b) in enumerate(node.inputs[0].type.broadcastable)
            if i not in self.axis
        )
        return [rval, rval]

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return [None, None]
        if len(self.axis) != 1:
            raise ValueError("R_op supported for arg_max only for " "one axis!")
        if self.axis[0] > 1:
            raise ValueError("R_op supported for arg_max only when " " axis is 0 or 1")
        if inputs[0].ndim != 2:
            raise ValueError(
                "R_op supported for arg_max only when " " input is a matrix"
            )
        max_vals, max_pos = self.make_node(*inputs).outputs
        if self.axis[0] == 0:
            return [eval_points[0][max_pos, arange(eval_points[0].shape[1])], None]
        else:
            return [eval_points[0][arange(eval_points[0].shape[0]), max_pos], None]

    def grad(self, inp, grads):
        # The strict sense mathematical gradient of the maximum function is
        # not calculated here for it is not defined at every point where some
        # coordinates are identical. However, since the latter set has null
        # Lebesgue measure, the result may be interpreted as weak gradient.

        # @note: This function should work correctly for L{vector}s.
        # (x, y), (gz, gw)
        # gz*dz/dx + gw*dw/dx, gz*dz/dy + gw*dw/dy
        # gMax * dMax/dx + gArgMax * dArgMax/dx,
        # gMax * dMax/daxis + gArgMax * dArgMax/daxis
        # g_max has one less dimension than x, so you need to complete
        # g_max to x's shape when axis=0 the broadcasting mechanism
        # does it automatically
        x = inp[0]
        axis = as_tensor_variable(self.axis)
        g_max, g_max_idx = grads

        g_max_disconnected = isinstance(g_max.type, DisconnectedType)
        g_max_idx_disconnected = isinstance(g_max_idx.type, DisconnectedType)

        # if the op is totally disconnected, so are its inputs
        if g_max_disconnected and g_max_idx_disconnected:
            return [DisconnectedType()(), DisconnectedType()()]

        # if the max is disconnected but the argmax is not,
        # the gradient on its inputs is zero
        if g_max_disconnected:
            return [x.zeros_like()]
        if NoneConst.equals(axis):
            axis_ = list(range(x.ndim))
        else:
            axis_ = axis
        xmax = max(x, axis_)

        # Raise the g_max and xmax to the same number of dim as the input.
        pattern = []
        out_dim = 0
        if NoneConst.equals(axis):
            # We are taking the max/argmax over all dimensions.
            axis = None
        for i in range(x.ndim):
            if axis is None or i in axis.data:
                pattern.append("x")
            else:
                pattern.append(out_dim)
                out_dim += 1
        g_max_pad = DimShuffle(g_max.broadcastable, pattern)(g_max)
        xmax_pad = DimShuffle(xmax.broadcastable, pattern)(xmax)

        # Set the grad to the correct position.
        g_x = eq(xmax_pad, x) * g_max_pad
        return (g_x,)


class Argmax(COp):
    """
    Calculate the argmax over a given axis or over all axes.
    """

    nin = 2  # tensor, axis
    nout = 1
    E_axis = "invalid axis"
    __props__ = ("axis",)
    _f16_ok = True

    params_type = ParamsType(c_axis=aes.int64)

    def __init__(self, axis):
        if axis is not None:
            axis = tuple(axis)
        self.axis = tuple(axis)

    def get_params(self, node):
        if self.axis is not None and len(self.axis) == 1:
            c_axis = np.int64(self.axis[0])
        else:
            # The value here doesn't matter, it won't be used
            c_axis = np.int64(-1)
        return self.params_type.get_params(c_axis=c_axis)

    def make_node(self, x, axis=None):
        x = as_tensor_variable(x)
        if self.axis is None:
            all_axes = list(range(x.ndim))
        else:
            all_axes = self.axis
        inputs = [x]

        # We keep the original broadcastable flags for dimensions on which
        # we do not perform the argmax.
        broadcastable = [
            b for i, b in enumerate(x.type.broadcastable) if i not in all_axes
        ]
        outputs = [tensor("int64", broadcastable, name="argmax")]
        return Apply(self, inputs, outputs)

    def prepare_node(self, node, storage_map, compute_map, impl):
        if len(node.inputs) == 2:
            raise ValueError(
                "You are trying to compile a graph with an old Argmax node.  Either reoptimize your graph or rebuild it to get the new node format."
            )

    def perform(self, node, inp, outs, params):
        (x,) = inp
        axes = self.axis
        (max_idx,) = outs
        if axes is None:
            axes = tuple(range(x.ndim))

        # Numpy does not support multiple axes for argmax
        # Work around
        keep_axes = np.array([i for i in range(x.ndim) if i not in axes], dtype="int64")
        # Not-reduced axes in front
        transposed_x = np.transpose(x, np.concatenate((keep_axes, axes)))
        kept_shape = transposed_x.shape[: len(keep_axes)]
        reduced_shape = transposed_x.shape[len(keep_axes) :]
        new_shape = kept_shape + (np.prod(reduced_shape),)
        reshaped_x = transposed_x.reshape(new_shape)

        max_idx[0] = _asarray(np.argmax(reshaped_x, axis=-1), dtype="int64")

    def c_code(self, node, name, inp, out, sub):
        (x,) = inp
        (argmax,) = out
        fail = sub["fail"]
        params = sub["params"]
        if self.axis is None:
            axis_code = "axis = NPY_MAXDIMS;"
        else:
            if len(self.axis) > 1:
                raise NotImplementedError()
            # params is only used here for now
            axis_code = (
                """
            axis = %(params)s->c_axis;
            if(axis > PyArray_NDIM(%(x)s)-1 || axis < -PyArray_NDIM(%(x)s)){
                PyErr_SetString(PyExc_ValueError,
                "Argmax, bad axis argument");
                %(fail)s
            }
            """
                % locals()
            )
        ret = """
        int axis;

        Py_CLEAR(%(argmax)s);//todo pass them as out parameter.
        %(axis_code)s

        %(argmax)s = (PyArrayObject*)PyArray_ArgMax(%(x)s, axis, NULL);
        if(%(argmax)s == NULL){
            %(fail)s;
        }
        if(!PyArray_CheckExact(%(argmax)s)){
            %(argmax)s = (PyArrayObject*)PyArray_FromAny((PyObject*)%(argmax)s, NULL, 0, 0, NPY_ARRAY_ENSUREARRAY, NULL);
            if(%(argmax)s == NULL){
                %(fail)s;
            }
        }
        if(PyArray_TYPE(%(argmax)s) != NPY_INT64){
            PyObject * tmp = PyArray_Cast(%(argmax)s, NPY_INT64);
            if (NULL == tmp){
                %(fail)s;
            }
            Py_DECREF(%(argmax)s);
            %(argmax)s = (PyArrayObject*)tmp;
        }
        """
        return ret % locals()

    def c_code_cache_version(self):
        return (1,)

    def infer_shape(self, fgraph, node, shapes):
        (ishape,) = shapes
        if self.axis is None:
            return [()]
        rval = tuple(
            [
                ishape[i]
                for (i, b) in enumerate(node.inputs[0].type.broadcastable)
                if i not in self.axis
            ]
        )
        return [rval]

    def grad(self, inp, grads):
        (x,) = inp

        return [x.zeros_like()]


def makeKeepDims(x, y, axis):
    """
    Reintroduces in y with length one the axes of x which have been left out
    in a prior reduction of x. With this option, the resulting tensor will
    broadcast correctly against the original tensor x.

    """
    x = as_tensor_variable(x)
    y = as_tensor_variable(y)

    if axis is None:
        axis = list(range(x.type.ndim))
    elif isinstance(axis, (int, np.integer)):
        axis = [axis]
    elif isinstance(axis, np.ndarray) and axis.ndim == 0:
        axis = [int(axis)]
    else:
        axis = [int(a) for a in axis]
    newaxis = []
    for a in axis:
        if not isinstance(a, int):
            raise ValueError("keepdims option can be used only with constant axis")
        if a < 0:
            a += x.type.ndim
        newaxis.append(a)
    i = 0
    new_dims = []
    for j, _ in enumerate(x.type.broadcastable):
        if j in newaxis:
            new_dims.append("x")
        else:
            new_dims.append(i)
            i += 1
    return DimShuffle(y.type.broadcastable, new_dims)(y)


def check_and_normalize_axes(x, axis):
    """Check axes, normalize and convert them to a Python list of integers.

    Parameters
    ----------
    x: TensorVariable
    axis: int, tuple or list of integers

    Returns
    -------
    axis: list of integers
        Return an empty list if argument is None.

    """
    x = as_tensor_variable(x)
    if axis is None:
        axis = []
    elif isinstance(axis, (int, np.integer)) or (
        isinstance(axis, np.ndarray) and axis.ndim == 0
    ):
        axis = [int(axis)]
    elif isinstance(axis, (tuple, list, np.ndarray)):
        axis = [int(i) for i in axis]
    elif isinstance(axis, Variable):
        if NoneConst.equals(axis):
            axis = []
        elif not isinstance(axis, TensorConstant):
            raise TypeError(f"Computation needs a constant axis. Got {axis}")
        else:
            assert axis.dtype in integer_dtypes
            if isinstance(axis.data, (int, np.integer)) or (
                isinstance(axis.data, np.ndarray) and axis.data.ndim == 0
            ):
                axis = [int(axis.data)]
            elif isinstance(axis.data, (list, np.ndarray)):
                axis = [int(i) for i in axis.data]
    else:
        raise TypeError(
            f"Axis must be an integer, tuple, list of integers or a TensorVariable. Got {axis}"
        )
    if len(axis) > 0:
        for i in range(len(axis)):
            if axis[i] < 0:
                axis[i] += x.type.ndim
            if axis[i] < 0 or axis[i] >= x.type.ndim:
                raise ValueError(
                    f"Computation needs a valid axis number for {int(x.type.ndim)}-D tensor. Got {int(axis[i])}"
                )
        axis = list(set(axis))
        axis.sort()
    return axis


def max_and_argmax(a, axis=None, keepdims=False):
    """
    Returns maximum elements and their indices obtained by iterating over
    given axis.

    When axis is None (the default value), the max is performed
    over the flattened tensor.

    Parameters
    ----------
    keepdims : bool
        If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the result
        will broadcast correctly against the original tensor.

    """
    # Check axis and convert it to a Python list of integers.
    # Axis will be used as an op param of MaxAndArgmax.
    a = as_tensor_variable(a)
    axis = check_and_normalize_axes(a, axis)
    if len(axis) == 0:
        axis = list(range(a.type.ndim))
    out, argout = MaxAndArgmax(axis)(a)

    if keepdims:
        out = makeKeepDims(a, out, axis)
        argout = makeKeepDims(a, argout, axis)
    return [out, argout]


class NonZeroCAReduce(CAReduce):
    def _c_all(self, node, name, inames, onames, sub):
        decl, checks, alloc, loop, end = super()._c_all(node, name, inames, onames, sub)

        # We add an additional check for zero-sized dimensions (This seems like
        # something that could enabled in `elemwise_cgen.make_checks`.)
        iname = inames[0]

        axis = self.axis
        if axis is None:
            axis = list(range(len(node.inputs[0].type.broadcastable)))

        pattern = [0] * len(node.inputs[0].broadcastable)
        for i in axis:
            pattern[i] = 1

        pattern_ = str(pattern)[1:-1]

        decl += f"""int tosum[]={{{pattern_}}};"""
        alloc += f"""
                for(int i=0;i<PyArray_NDIM({iname});i++){{
                    if(PyArray_DIMS({iname})[i]==0 && tosum[i]){{
                        PyErr_Format(PyExc_ValueError,
                            "Input of CAReduce{{{node.op.scalar_op}}} has zero-size on axis %%d",i);
                        {sub["fail"]};
                    }}
                }}
                """
        return decl, checks, alloc, loop, end


class Max(NonZeroCAReduce):
    nfunc_spec = ("max", 1, 1)

    def __init__(self, axis):
        super().__init__(aes.scalar_maximum, axis)


class Min(NonZeroCAReduce):
    nfunc_spec = ("min", 1, 1)

    def __init__(self, axis):
        super().__init__(aes.scalar_minimum, axis)


def max(x, axis=None, keepdims=False):
    """
    Returns maximum elements obtained by iterating over given axis.

    When axis is None (the default value), the max is performed
    over the flattened tensor.

    Parameters
    ----------
    keepdims: bool
        If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the result
        will broadcast correctly against the original tensor.

    Notes
    -----
    We return an error as numpy when we reduce a dim with a shape of 0.

    """

    # We have a choice of implementing this call with the
    # CAReduce op or the MaxAndArgmax op.

    # MaxAndArgmax supports grad and Rop, so we prefer to use that.
    # CAReduce is faster, but optimizations will replace MaxAndArgmax[0]
    # with CAReduce at compile time, so at this stage the important
    # thing is supporting all user interface features, not speed.
    # Some cases can be implemented only with CAReduce.

    # We thus prefer to use MaxAndArgmax, if possible. It does not
    # support all axis arguments, so we may need to fall back to CAReduce.

    try:
        out = max_and_argmax(x, axis)[0]
    except Exception:
        out = Max(axis)(x)

    if keepdims:
        out = makeKeepDims(x, out, axis)
    return out


def argmax(x, axis=None, keepdims=False):
    """
    Returns indices of maximum elements obtained by iterating over given axis.

    When axis is None (the default value), the argmax is performed
    over the flattened tensor.

    Parameters
    ----------
    keepdims : bool
        If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the result
        will broadcast correctly against the original tensor.

    """
    argout = max_and_argmax(x, axis)[1]

    if keepdims:
        argout = makeKeepDims(x, argout, axis)
    return argout


def min(x, axis=None, keepdims=False):
    """
    Returns minimum elements obtained by iterating over given axis.

    When axis is None (the default value), the min is performed
    over the flattened tensor.

    Parameters
    ----------
    keepdims: bool
        If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the result
        will broadcast correctly against the original tensor.

    """
    x = as_tensor_variable(x)
    str_x_type = str(x.dtype)
    if str_x_type.startswith("float") or str_x_type in int_dtypes:
        return -max(-x, axis=axis, keepdims=keepdims)
    elif str_x_type in uint_dtypes:
        itype = np.iinfo(x.dtype)
        max_val = np.array(itype.max, dtype=itype.dtype)
        return max_val - max(max_val - x, axis=axis, keepdims=keepdims)
    elif str_x_type == "bool":
        return ~max(~x, axis=axis, keepdims=keepdims)
    else:
        # Be careful about unsigned integers, complex
        raise NotImplementedError()


def argmin(x, axis=None, keepdims=False):
    """
    Returns indices of minimum elements obtained by iterating over given axis.

    When axis is None (the default value), the argmin is performed
    over the flattened tensor.

    Parameters
    ----------
    keepdims: bool
        If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the result
        will broadcast correctly against the original tensor.

    """
    x = as_tensor_variable(x)
    str_x_type = str(x.dtype)
    if str_x_type.startswith("float") or str_x_type in int_dtypes:
        return argmax(-x, axis=axis, keepdims=keepdims)
    elif str_x_type in uint_dtypes:
        itype = np.iinfo(x.dtype)
        return argmax(itype.max - x, axis=axis, keepdims=keepdims)
    elif str_x_type == "bool":
        return argmax(~x, axis=axis, keepdims=keepdims)
    else:
        # Be careful about unsigned integers, complex
        raise NotImplementedError()


def smallest(*args):
    """
    Return the [elementwise] smallest of a variable number of arguments.

    Like python's min.

    """
    if len(args) == 2:
        a, b = args
        return switch(a < b, a, b)
    else:
        return min(stack(args), axis=0)


def largest(*args):
    """
    Return the [elementwise] largest of a variable number of arguments.

    Like python's max.

    """
    if len(args) == 2:
        a, b = args
        return switch(a > b, a, b)
    else:
        return max(stack(args), axis=0)


@scalar_elemwise
def lt(a, b):
    """a < b"""


@scalar_elemwise
def gt(a, b):
    """a > b"""


@scalar_elemwise
def le(a, b):
    """a <= b"""


@scalar_elemwise
def ge(a, b):
    """a >= b"""


@scalar_elemwise
def eq(a, b):
    """a == b"""


@scalar_elemwise
def neq(a, b):
    """a != b"""


@scalar_elemwise
def isnan(a):
    """isnan(a)"""


# Rename isnan to isnan_ to allow to bypass it when not needed.
# glibc 2.23 don't allow isnan on int, so we remove it from the graph.
isnan_ = isnan


def isnan(a):
    """isnan(a)"""
    a = as_tensor_variable(a)
    if a.dtype in discrete_dtypes:
        return alloc(
            np.asarray(False, dtype="bool"), *[a.shape[i] for i in range(a.ndim)]
        )
    return isnan_(a)


@scalar_elemwise
def isinf(a):
    """isinf(a)"""


# Rename isnan to isnan_ to allow to bypass it when not needed.
# glibc 2.23 don't allow isnan on int, so we remove it from the graph.
isinf_ = isinf


def isinf(a):
    """isinf(a)"""
    a = as_tensor_variable(a)
    if a.dtype in discrete_dtypes:
        return alloc(
            np.asarray(False, dtype="bool"), *[a.shape[i] for i in range(a.ndim)]
        )
    return isinf_(a)


def allclose(a, b, rtol=1.0e-5, atol=1.0e-8, equal_nan=False):
    """
    Implement Numpy's ``allclose`` on tensors.

    ``absolute(a - b) <= (atol + rtol * absolute(b))``

    Parameters
    ----------
    a : tensor
        Input to compare.
    b : tensor
        Input to compare.
    rtol : float
        The relative tolerance parameter.
    atol : float
        The absolute tolerance parameter.
    equal_nan: bool
        Whether to consider nan's in the same place to be close.

    Returns
    -------
    bool
        A boolean value (of type int8 returned by the tensor elementwise `all`
        function) whether all elements in a and b are in the tolerance range
        defined above.

    Notes
    -----
    Not a symmetric equation. See Numpy's documentation.

    """
    return all(isclose(a, b, rtol, atol, equal_nan))


def isclose(a, b, rtol=1.0e-5, atol=1.0e-8, equal_nan=False):
    """
    Implements Numpy's ``isclose`` on tensors.

    The tolerance values are positive, typically very small numbers. The
    relative difference (`rtol` * abs(`b`)) and the absolute difference
    `atol` are added together to compare against the absolute difference
    between `a` and `b`.

    ``absolute(a - b) <= (atol + rtol * absolute(b))``

    Parameters
    ----------
    a : tensor
        Input to compare.
    b : tensor
        Input to compare.
    rtol : float
        The relative tolerance parameter.
    atol : float
        The absolute tolerance parameter.
    equal_nan : bool
        Whether to consider nan's in the same place to be close

    Returns
    -------
    int8
        A boolean (int8) array where two arrays are element-wise equal
        within a tolerance.

    Notes
    -----
    Not a symmetric equation. See Numpy's documentation.

    Examples
    --------
    >>> import aesara
    >>> import numpy as np
    >>> a = _asarray([1e10, 1e-7], dtype="float64")
    >>> b = _asarray([1.00001e10, 1e-8], dtype="float64")
    >>> aesara.tensor.isclose(a, b).eval()
    array([1, 0], dtype=int8)
    >>> a = _asarray([1e10, 1e-8], dtype="float64")
    >>> b = _asarray([1.00001e10, 1e-9], dtype="float64")
    >>> aesara.tensor.isclose(a, b).eval()
    array([1, 1], dtype=int8)
    >>> a = _asarray([1e10, 1e-8], dtype="float64")
    >>> b = _asarray([1.0001e10, 1e-9], dtype="float64")
    >>> aesara.tensor.isclose(a, b).eval()
    array([0, 1], dtype=int8)
    >>> a = _asarray([1.0, np.nan], dtype="float64")
    >>> b = _asarray([1.0, np.nan], dtype="float64")
    >>> aesara.tensor.isclose(a, b).eval()
    array([1, 0], dtype==int8)
    >>> a = _asarray([1.0, np.nan], dtype="float64")
    >>> b = _asarray([1.0, np.nan], dtype="float64")
    >>> aesara.tensor.isclose(a, b, equal_nan=True).eval()
    array([1, 1], dtype==int8)
    >>> a = _asarray([1.0, np.inf], dtype="float64")
    >>> b = _asarray([1.0, -np.inf], dtype="float64")
    >>> aesara.tensor.isclose(a, b).eval()
    array([1, 0], dtype==int8)
    >>> a = _asarray([1.0, np.inf], dtype="float64")
    >>> b = _asarray([1.0, np.inf], dtype="float64")
    >>> aesara.tensor.isclose(a, b).eval()
    array([1, 1], dtype==int8)

    """
    # close will be an int8 array of 1 where within tolerance
    # and 0 where not within tolerance or there was a nan or inf value.
    diff = abs(a - b)
    tolerance = atol + rtol * abs(b)
    close_prelim = le(diff, tolerance)

    a_nan = isnan(a)
    b_nan = isnan(b)
    nans = bitwise_or(a_nan, b_nan)

    a_inf = isinf(a)
    b_inf = isinf(b)
    infs = bitwise_or(a_inf, b_inf)

    nans_or_infs = bitwise_or(nans, infs)

    # close is now an array of 0's except where elements are not nan or inf
    # and are within the tolerance.
    close = bitwise_and(close_prelim, bitwise_not(nans_or_infs))

    # deal with signed inf values. this will make an array inf_eq of 0's
    # except where inf values have the same sign.
    both_infs = bitwise_and(a_inf, b_inf)
    inf_signs_eq = eq(a_inf * sgn(a), b_inf * sgn(b))
    inf_eq = bitwise_and(both_infs, inf_signs_eq)

    # now create the potential result combining close and inf_eq
    close_with_infs = bitwise_or(close, inf_eq)

    # deal with comparing nan's.
    if equal_nan:
        both_nans = bitwise_and(a_nan, b_nan)
        return bitwise_or(close_with_infs, both_nans)
    # otherwise nan's aren't considered close.
    else:
        return close_with_infs


##########################
# Bit-wise
##########################


@scalar_elemwise
def and_(a, b):
    """bitwise a & b"""


bitwise_and = and_  # numpy name for it


@scalar_elemwise
def or_(a, b):
    """bitwise a | b"""


bitwise_or = or_  # numpy name for it


@scalar_elemwise
def xor(a, b):
    """bitwise a ^ b"""


bitwise_xor = xor  # numpy name for it


@scalar_elemwise
def invert(a):
    """bitwise ~a"""


bitwise_not = invert  # numpy alias for it

##########################
# Math
##########################


@scalar_elemwise
def abs_(a):
    """|`a`|

    TensorVariable overloads the `TensorVariable.__abs__` operator so that
    this function is called when you type abs(a).

    """


pprint.assign(abs_, printing.PatternPrinter(("|%(0)s|", -1000)))


@scalar_elemwise
def exp(a):
    """e^`a`"""


@scalar_elemwise
def exp2(a):
    """2^`a`"""


@scalar_elemwise
def expm1(a):
    """e^`a` - 1"""


@scalar_elemwise
def neg(a):
    """-a"""


# numpy.reciprocal does integer division on integer inputs
# (which is not very interesting)
@scalar_elemwise
def inv(a):
    """1.0/a"""


@scalar_elemwise
def log(a):
    """base e logarithm of a"""


@scalar_elemwise
def log2(a):
    """base 2 logarithm of a"""


@scalar_elemwise
def log10(a):
    """base 10 logarithm of a"""


@scalar_elemwise
def log1p(a):
    """log(1+a)"""


@scalar_elemwise
def sgn(a):
    """sign of a"""


@scalar_elemwise
def ceil(a):
    """ceiling of a"""


@scalar_elemwise
def floor(a):
    """floor of a"""


@scalar_elemwise
def trunc(a):
    """trunc of a"""


def iround(a, mode=None):
    """cast(round(a,mode),'int64')"""
    return cast(round(a, mode), "int64")


def round(a, mode=None):
    """round_mode(a) with mode in [half_away_from_zero, half_to_even].
    Default to half_to_even."""
    if mode is None:
        mode = "half_to_even"
        if config.warn__round:
            warnings.warn(
                "aesara.tensor.round() changed its default from"
                " `half_away_from_zero` to `half_to_even` to have"
                " the same default as NumPy. Use the Aesara flag"
                " `warn__round=False` to disable this warning."
            )
    if mode == "half_away_from_zero":
        return round_half_away_from_zero(a)
    elif mode == "half_to_even":
        return round_half_to_even(a)
    else:
        raise Exception(f"round mode {mode} is not implemented.")


@scalar_elemwise
def round_half_to_even(a):
    """round_half_to_even(a)"""


@scalar_elemwise
def round_half_away_from_zero(a):
    """round_half_away_from_zero(a)"""


@scalar_elemwise
def sqr(a):
    """square of a"""


# alias to sqr, included to maintain similarity with numpy interface
square = sqr


def cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None):
    """Calculate the covariance matrix.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, :math:`m = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element :math:`C_{ij}` is the covariance of
    :math:`x_i` and :math:`x_j`. The element :math:`C_{ii}` is the variance
    of :math:`x_i`. Code and docstring ported from numpy.

    Parameters
    ==========
    m : array_like
        A 2-D array containing multiple variables and observations.
        Each row of `m` represents a variable, and each column is
        observations of all those variables.
    y : array_like, optional
        An additional set of variables and observations. `y` has the same form
        as that of `m`.
    rowvar : bool, optional
        If `rowvar` is True (default), then each row represents a
        variable, with observations in the columns. Otherwise, the relationship
        is transposed: each column represents a variable, while the rows
        contain observations.
    bias : bool, optional
        Default normalization (False) is by ``(N - 1)``, where ``N`` is the
        number of observations given (unbiased estimate). If `bias` is True, then
        normalization is by ``N``. These values can be overridden by using the
        keyword ``ddof``.
    ddof : int, optional
        If not ``None`` the default value implied by `bias` is overridden.
        The default value is ``None``.

    Returns
    =======
    out : The covariance matrix of the variables.

    """

    if fweights is not None:
        raise NotImplementedError("fweights are not implemented")
    if aweights is not None:
        raise NotImplementedError("aweights are not implemented")

    if not rowvar and m.shape[0] != 1:
        m = m.T

    if y is not None:
        if not rowvar and y.shape[0] != 1:
            y = y.T
        m = concatenate((m, y), axis=0)

    if ddof is None:
        if not bias:
            ddof = 1
        else:
            ddof = 0

    # Determine the normalization
    fact = m.shape[1] - ddof

    m -= m.mean(axis=1, keepdims=1)
    c = m.dot(m.T)
    c *= constant(1) / fact
    return c.squeeze()


@scalar_elemwise
def sqrt(a):
    """square root of a"""


@scalar_elemwise
def deg2rad(a):
    """convert degree a to radian"""


@scalar_elemwise
def rad2deg(a):
    """convert radian a to degree"""


@scalar_elemwise
def cos(a):
    """cosine of a"""


@scalar_elemwise
def arccos(a):
    """arccosine of a"""


@scalar_elemwise
def sin(a):
    """sine of a"""


@scalar_elemwise
def arcsin(a):
    """arcsine of a"""


@scalar_elemwise
def tan(a):
    """tangent of a"""


@scalar_elemwise
def arctan(a):
    """arctangent of a"""


@scalar_elemwise
def arctan2(a, b):
    """arctangent of a / b"""


@scalar_elemwise
def cosh(a):
    """hyperbolic cosine of a"""


@scalar_elemwise
def arccosh(a):
    """hyperbolic arc cosine of a"""


@scalar_elemwise
def sinh(a):
    """hyperbolic sine of a"""


@scalar_elemwise
def arcsinh(a):
    """hyperbolic arc sine of a"""


@scalar_elemwise
def tanh(a):
    """hyperbolic tangent of a"""


@scalar_elemwise
def arctanh(a):
    """hyperbolic arc tangent of a"""


@scalar_elemwise
def erf(a):
    """error function"""


@scalar_elemwise
def erfc(a):
    """complementary error function"""


@scalar_elemwise
def erfcx(a):
    """scaled complementary error function"""


@scalar_elemwise
def erfinv(a):
    """inverse error function"""


@scalar_elemwise
def erfcinv(a):
    """inverse complementary error function"""


@scalar_elemwise
def gamma(a):
    """gamma function"""


@scalar_elemwise
def gammaln(a):
    """log gamma function"""


@scalar_elemwise
def psi(a):
    """derivative of log gamma function"""


@scalar_elemwise
def tri_gamma(a):
    """second derivative of the log gamma function"""


@scalar_elemwise
def chi2sf(x, k):
    """chi squared survival function"""


@scalar_elemwise
def gammainc(k, x):
    """Regularized lower gamma function"""


@scalar_elemwise
def gammaincc(k, x):
    """Regularized upper gamma function"""


@scalar_elemwise
def gammau(k, x):
    """Upper incomplete gamma function."""


@scalar_elemwise
def gammal(k, x):
    """Lower incomplete gamma function."""


@scalar_elemwise
def j0(x):
    """Bessel function of the first kind of order 0."""


@scalar_elemwise
def j1(x):
    """Bessel function of the first kind of order 1."""


@scalar_elemwise
def jv(v, x):
    """Bessel function of the first kind of order v (real)."""


@scalar_elemwise
def i0(x):
    """Modified Bessel function of the first kind of order 0."""


@scalar_elemwise
def i1(x):
    """Modified Bessel function of the first kind of order 1."""


@scalar_elemwise
def iv(v, x):
    """Modified Bessel function of the first kind of order v (real)."""


@scalar_elemwise
def sigmoid(x):
    """Logistic sigmoid function (1 / (1 + exp(x)), also known as expit or inverse logit"""


expit = sigmoid


@scalar_elemwise
def softplus(x):
    """Compute log(1 + exp(x)), also known as softplus or log1pexp"""


@scalar_elemwise
def real(z):
    """Return real component of complex-valued tensor `z`"""


_tensor_py_operators.real = property(real)


@scalar_elemwise
def imag(z):
    """Return imaginary component of complex-valued tensor `z`"""


_tensor_py_operators.imag = property(imag)


@scalar_elemwise
def angle(z):
    """Return polar-coordinate angle of complex-valued tensor `z`"""


@scalar_elemwise  # numpy.complex cannot build tensors
def complex(real, imag):
    """Return complex-valued tensor with `real` and `imag` components"""


@scalar_elemwise
def conj(z):
    """Return the complex conjugate of `z`."""


@scalar_elemwise
def complex_from_polar(abs, angle):
    """Return complex-valued tensor from polar coordinate specification."""


class Mean(CAReduce):
    def __init__(self, axis=None):
        super().__init__(aes.add, axis)
        assert self.axis is None or len(self.axis) == 1

    def __str__(self):
        if self.axis is not None:
            return "Mean{%s}" % (", ".join(str(x) for x in self.axis))
        else:
            return "Mean"

    def _output_dtype(self, idtype):
        # we want to protect against overflow
        return "float64"

    def perform(self, node, inp, out):
        (input,) = inp
        (output,) = out
        if self.axis is None:
            axis = None
        else:
            axis = self.axis[0]
        # numpy.asarray is needed as otherwise we can end up with a
        # numpy scalar.
        output[0] = np.asarray(np.mean(input, dtype="float64", axis=axis))

    def c_code(self, node, name, inames, onames, sub):

        ret = super().c_code(node, name, inames, onames, sub)

        if self.axis is not None:
            return ret

        # TODO: c_code perform support only axis is None
        return (
            ret
            + f"""
  *((double *)PyArray_DATA({onames[0]})) /= PyArray_SIZE({inames[0]});
  """
        )


# TODO: implement the grad. When done and tested, you can make this the default
# version.
#    def grad(self, (x,), (gout,)):
#      import pdb;pdb.set_trace()
#      return grad(mean(x, self.axis, op=False),[x])


def mean(input, axis=None, dtype=None, op=False, keepdims=False, acc_dtype=None):
    """
    Computes the mean value along the given axis(es) of a tensor `input`.

    Parameters
    ----------
    axis : None or int or (list of int) (see `Sum`)
        Compute the mean along this axis of the tensor.
        None means all axes (like numpy).
    dtype: None or string
        Dtype to cast the result of the inner summation into.
        For instance, by default, a sum of a float32 tensor will be
        done in float64 (acc_dtype would be float64 by default),
        but that result will be casted back in float32.
    keepdims: bool
        If this is set to True, the axes which are reduced are
        left in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original tensor.
    acc_dtype: None or string
        Dtype to use for the inner summation. This will not
        necessarily be the dtype of the output (in particular
        if it is a discrete (int/uint) dtype, the output will
        be in a float type). If None, then we use the same rules as `sum()`.

    Notes
    -----
    For gpu, if you specify dtype=float32, everything will be done on the gpu.

    """
    input = as_tensor_variable(input)
    if op:
        if dtype not in (None, "float64"):
            raise NotImplementedError(
                "The Mean op does not support the dtype argument, "
                "and will always use float64. If you want to specify "
                "the dtype, call tensor.mean(..., op=False).",
                dtype,
            )
        if acc_dtype not in (None, "float64"):
            raise NotImplementedError(
                "The Mean op does not support the acc_dtype argument, "
                "and will always use float64. If you want to specify "
                "acc_dtype, call tensor.mean(..., op=False).",
                dtype,
            )
        out = Mean(axis)(input)
        if keepdims:
            out = makeKeepDims(input, out, axis)
        return out

    if dtype is not None:
        # The summation will be done with the specified dtype.
        # sum() will complain if it is not suitable.
        sum_dtype = dtype
    else:
        sum_dtype = None
        # float16 overflows on the cast way too often
        if input.dtype == "float16":
            sum_dtype = "float32"

    s = sum(input, axis=axis, dtype=sum_dtype, keepdims=keepdims, acc_dtype=acc_dtype)
    shp = shape(input)

    # Cast shp into a float type
    # TODO Once we have a consistent casting policy, we could simply
    # use true_div.
    if s.dtype in ("float16", "float32", "complex64"):
        shp = cast(shp, "float32")
    else:
        shp = cast(shp, "float64")

    if axis is None:
        axis = list(range(input.ndim))
    elif isinstance(axis, (int, np.integer)):
        axis = [axis]
    elif isinstance(axis, np.ndarray) and axis.ndim == 0:
        axis = [int(axis)]
    else:
        axis = [int(a) for a in axis]

    # This sequential division will possibly be optimized by Aesara:
    for i in axis:
        s = true_div(s, shp[i])

    # This can happen when axis is an empty list/tuple
    if s.dtype != shp.dtype and s.dtype in discrete_dtypes:
        s = cast(s, shp.dtype)

    if dtype == "float16" or (dtype is None and input.dtype == "float16"):
        s = cast(s, "float16")
    s.name = "mean"
    return s


def var(input, axis=None, ddof=0, keepdims=False, corrected=False):
    """
    Computes the variance along the given axis(es) of a tensor `input`.

    Parameters
    ----------
    axis: None or int or (list of int) (see `Sum`)
        Compute the variance along this axis of the tensor.
        None means all axes (like numpy).
    ddof: Degrees of freedom; 0 would compute the ML estimate, 1 would compute
        the unbiased estimate.
    keepdims : bool
        If this is set to True, the axes which are reduced are
        left in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original tensor.
    corrected : bool
        If this is set to True, the 'corrected_two_pass' algorithm is
        used to compute the variance.
        Refer : http://www.cs.yale.edu/publications/techreports/tr222.pdf

    Notes
    -----
    Default uses the two-pass algorithm (reference below).
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Two-pass_algorithm
    Also supports 'corrected_two_pass' algorithm (using the 'corrected' flag)
    which is numerically more stable. There exist other implementations that
    offer better stability, but probably slower.

    """

    if isinstance(ddof, (bool)):
        raise ValueError(
            "Parameter keepdims is now at index 3: (input, \
                          axis=None, ddof=0, keepdims=False, corrected=False)"
        )

    input_ndim = input.type.ndim
    if axis is None:
        axis = list(range(input_ndim))
    elif isinstance(axis, (int, np.integer)):
        axis = [axis]
    elif isinstance(axis, np.ndarray) and axis.ndim == 0:
        axis = [int(axis)]
    else:
        axis = [int(a) for a in axis]

    # compute the axis-wise mean
    mean_input = mean(input, axis, keepdims=True)

    # center the input
    centered_input = input - mean_input

    # return the mean sqr
    two = constant(2, dtype=centered_input.dtype)
    if ddof == 0:
        v = mean((centered_input ** two), axis, keepdims=keepdims)
    else:
        shp = shape(input) - ddof
        v = sum((centered_input ** two), axis=axis, keepdims=keepdims)
        for i in axis:
            v = true_div(v, shp[i])

    # use 'corrected_two_pass' algorithm
    if corrected:
        if ddof == 0:
            error = mean(centered_input, axis, keepdims=keepdims) ** 2
        else:
            shp = shape(input) - ddof
            shp_inp = shape(input)
            error = sum(centered_input, axis=axis, keepdims=keepdims) ** 2
            for i in axis:
                error = true_div(error, shp[i] * shp_inp[i])
        v = v - error

    v.name = "var"
    return v


def std(input, axis=None, ddof=0, keepdims=False, corrected=False):
    """
    Computes the standard deviation along the given axis(es) of a tensor `input`.

    Parameters
    ----------
    axis: None or int or (list of int) (see `Sum`)
        Compute the variance along this axis of the tensor.
        None means all axes (like numpy).
    ddof: Degrees of freedom; 0 would compute the ML estimate, 1 would compute
        the unbiased estimate.
    keepdims : bool
        If this is set to True, the axes which are reduced are
        left in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original tensor.
    corrected : bool
        If this is set to True, the 'corrected_two_pass' algorithm is
        used to compute the variance.
        Refer : http://www.cs.yale.edu/publications/techreports/tr222.pdf

    Notes
    -----
    It calls 'var()' and 'var()' uses the two-pass algorithm (reference below).
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Two-pass_algorithm
    Function 'var()' also supports 'corrected_two_pass' algorithm (using the
    'corrected' flag) which is numerically more stable. There exist other
    implementations that offer better stability, but probably slower.

    """

    if isinstance(ddof, (bool)):
        raise ValueError(
            "Parameter keepdims is now at index 3: (input, \
                          axis=None, ddof=0, keepdims=False, corrected=False)"
        )

    ret = sqrt(
        var(input=input, axis=axis, ddof=ddof, keepdims=keepdims, corrected=corrected)
    )
    ret.name = "std"
    return ret


@scalar_elemwise(symbolname="scalar_maximum")
def maximum(x, y):
    """elemwise maximum. See max for the maximum in one tensor"""
    # see decorator for function body


@scalar_elemwise(symbolname="scalar_minimum")
def minimum(x, y):
    """elemwise minimum. See min for the minimum in one tensor"""
    # see decorator for function body


def divmod(x, y):
    """elementvise divmod, using floor_div and mod_check"""
    return floor_div(x, y), mod_check(x, y)


@scalar_elemwise
def add(a, *other_terms):
    """elementwise addition"""
    # see decorator for function body


@scalar_elemwise
def sub(a, b):
    """elementwise subtraction"""
    # see decorator for function body


@scalar_elemwise
def mul(a, *other_terms):
    """elementwise multiplication"""
    # see decorator for function body


@scalar_elemwise
def true_div(a, b):
    """elementwise [true] division (inverse of multiplication)"""
    # see decorator for function body


@scalar_elemwise
def int_div(a, b):
    """elementwise [floor] division (inverse of multiplication)"""
    # see decorator for function body


# floor_div and int_div are the same thing
floor_div = int_div


def ceil_intdiv(a, b):
    """
    Safely compute ceil(float_division(a, b)).

    Works for all dtypes, but mostly useful when a and b are int.

    """
    # If a and b are int with not many significant bits, we could
    # cast them to float to avoid doing the modulo. We do not know if this
    # is faster or not. But this is not safe for int64 as the cast will
    # lose precision.
    # e.g.: cast(cast(a, scalar.upcast(a, 'float32')) / b, aes.upcast(a, b))

    # We cast for the case when a and b are uint*. Otherwise neq will
    # force their upcast to int.
    div = int_div(a, b)
    ret = cast(neq(a % b, 0), div.dtype) + div
    assert ret.dtype == aes.upcast(div.owner.inputs[0], div.owner.inputs[1])
    return ret


def mod_check(x, y):
    """Make sure we do not try to use complex numbers."""
    if (
        as_tensor_variable(x).dtype in complex_dtypes
        or as_tensor_variable(y).dtype in complex_dtypes
    ):
        # Currently forbidden.
        raise aes.Mod.complex_error
    else:
        return mod(x, y)


@scalar_elemwise
def mod(a, b):
    """elementwise modulo"""
    # see decorator for function body


@scalar_elemwise
def pow(a, b):
    """elementwise power"""
    # see decorator for function body


@scalar_elemwise
def clip(x, min, max):
    """
    Clip x to be between min and max.

    Note that when `x` is equal to the boundaries, the output is considered
    to be `x`, so at these points, the gradient of the cost wrt the output
    will be propagated to `x`, not to `min` nor `max`. In other words,
    on these points, the gradient wrt `x` will be equal to the gradient wrt
    the output, and the gradient wrt `min` and `max` will be zero.

    """
    # see decorator for function body
    # for grep: clamp, bound


pprint.assign(add, printing.OperatorPrinter("+", -2, "either"))
pprint.assign(mul, printing.OperatorPrinter("*", -1, "either"))
pprint.assign(sub, printing.OperatorPrinter("-", -2, "left"))
pprint.assign(neg, printing.OperatorPrinter("-", 0, "either"))
pprint.assign(true_div, printing.OperatorPrinter("/", -1, "left"))
pprint.assign(int_div, printing.OperatorPrinter("//", -1, "left"))
pprint.assign(pow, printing.OperatorPrinter("**", 1, "right"))


class Dot(Op):
    """
    Computes the dot product of two variables. For two matrices, this is
    equivalent to matrix multiplication. For two vectors, this is the inner
    product.

    Notes
    -----
    Matrix-matrix products are sometimes optimized to Dot22 or Gemm ops
    (see tensor.blas).
    Vector-vector products are sometimes optimized to Ger or CGer (see
    tensor.blas).
    Matrix-vector products are sometimes optimized to Gemv, CGemv (see
    tensor.blas).

    """

    __props__ = ()

    # the rationale for Dot22 is related to getting GEMM Ops into the
    # graph.  See Dot22 in tensor.blas for details.

    def make_node(self, *inputs):
        inputs = list(map(as_tensor_variable, inputs))

        if len(inputs) != 2:
            raise TypeError(f"Two arguments required, {len(inputs)} given ")
        if inputs[0].ndim not in (1, 2):
            raise TypeError(
                "Input 0 (0-indexed) must have ndim of "
                f"1 or 2, {int(inputs[0].ndim)} given. Consider calling "
                "aesara.tensor.dot instead."
            )
        if inputs[1].ndim not in (1, 2):
            raise TypeError(
                "Input 1 (0-indexed) must have ndim of "
                f"1 or 2, {int(inputs[1].ndim)} given. Consider calling "
                "aesara.tensor.dot instead."
            )

        i_broadcastables = [input.type.broadcastable for input in inputs]
        bx, by = i_broadcastables
        if len(by) == 2:  # y is a matrix
            bz = bx[:-1] + by[-1:]
        elif len(by) == 1:  # y is vector
            bz = bx[:-1]

        i_dtypes = [input.type.dtype for input in inputs]
        outputs = [tensor(aes.upcast(*i_dtypes), bz)]
        return Apply(self, inputs, outputs)

    def perform(self, node, inp, out):
        x, y = inp
        (z,) = out

        # the asarray is here because dot between two vectors
        # gives a numpy float object but we need to return a 0d
        # ndarray
        z[0] = np.asarray(np.dot(x, y))

    def grad(self, inp, grads):

        x, y = inp
        (gz,) = grads
        xdim, ydim, gdim = x.type.ndim, y.type.ndim, gz.type.ndim

        # grad is scalar, so x is vector and y is vector
        if gdim == 0:
            xgrad = gz * y
            ygrad = gz * x

        # x is vector, y is matrix, grad is vector
        elif xdim == 1 and ydim == 2:
            xgrad = dot(gz, y.T)
            ygrad = outer(x.T, gz)

        # x is matrix, y is vector, grad is vector
        elif xdim == 2 and ydim == 1:
            xgrad = outer(gz, y.T)
            ygrad = dot(x.T, gz)

        # x is matrix, y is matrix, grad is matrix
        elif xdim == ydim == 2:
            xgrad = dot(gz, y.T)
            ygrad = dot(x.T, gz)

        # If x or y contain broadcastable dimensions but only one of
        # them know that a matching dimensions is broadcastable, the
        # above code don't always return the right broadcast pattern.
        # This cause problem down the road. See gh-1461.
        if xgrad.broadcastable != x.broadcastable:
            xgrad = patternbroadcast(xgrad, x.broadcastable)
        if ygrad.broadcastable != y.broadcastable:
            ygrad = patternbroadcast(ygrad, y.broadcastable)

        rval = xgrad, ygrad

        for elem in rval:
            assert elem.dtype.find("float") != -1

        return rval

    def R_op(self, inputs, eval_points):
        # R_op for a \dot b evaluted at c for a and d for b is
        # simply c \dot b + a \dot d

        assert len(inputs) == 2
        assert len(eval_points) == 2
        if eval_points[0] is None and eval_points[1] is None:
            return [None]

        if eval_points[0]:
            t1 = self(eval_points[0], inputs[1])
        if eval_points[1]:
            t2 = self(inputs[0], eval_points[1])

        if eval_points[0] and eval_points[1]:
            return [t1 + t2]
        elif eval_points[0]:
            return [t1]
        else:
            return [t2]

    def infer_shape(self, fgraph, node, shapes):
        xshp, yshp = shapes
        x, y = node.inputs

        # vector / vector
        if x.ndim == 1 and y.ndim == 1:
            return [()]
        # matrix / vector
        if x.ndim == 2 and y.ndim == 1:
            return [xshp[:-1]]
        # vector / matrix
        if x.ndim == 1 and y.ndim == 2:
            return [yshp[-1:]]
        # matrix / matrix
        if x.ndim == 2 and y.ndim == 2:
            return [xshp[:-1] + yshp[-1:]]
        raise NotImplementedError()

    def __str__(self):
        return "dot"


_dot = Dot()
pprint.assign(
    _dot, printing.OperatorPrinter(printing.special["middle_dot"], -1, "left")
)


def dot(l, r):
    """Return a symbolic dot product.

    This is designed to work with both sparse and dense tensors types.
    """

    if not isinstance(l, Variable):
        l = as_tensor_variable(l)

    if not isinstance(r, Variable):
        r = as_tensor_variable(r)

    try:
        res = l.__dot__(r)
        if res is NotImplemented:
            raise NotImplementedError
    except (NotImplementedError, AttributeError, TypeError):
        res = r.__rdot__(l)
        if res is NotImplemented:
            raise NotImplementedError()

    return res


def dense_dot(a, b):
    """
    Computes the dot product of two variables.

    For two matrices, this is equivalent to matrix multiplication.
    For two vectors, this is the inner product.
    When one variable is a scalar, this is like elementwise multiplication.
    For N dimensions, this is a sum product over the last axis
    of the first array and the second-to-last axis of the second array:

        dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])

    Note that this dot function does one of three things, in the following
    sequence:

        1.  If either a or b is scalar, it returns the elementwise product
            without calling the Aesara Dot op.

        2.  If either a or b has more than 2 dimensions, it calls Aesara's
            tensordot function with appropriate axes. The tensordot function
            expresses high-dimensional dot products in terms of 2D matrix
            multiplications, so it may be possible to futherize optimize for
            performance.

        3.  If both a and b have either 1 or 2 dimensions, it calls Aesara's
            Dot op on a and b.

    Notes
    -----
    Matrix-matrix products are sometimes optimized to Dot22 or Gemm ops
    (see tensor.blas).
    Vector-vector products are sometimes optimized to Ger or CGer (see
    tensor.blas).
    Matrix-vector products are sometimes optimized to Gemv, CGemv (see
    tensor.blas).

    """
    a, b = as_tensor_variable(a), as_tensor_variable(b)

    if a.ndim == 0 or b.ndim == 0:
        return a * b
    elif a.ndim > 2 or b.ndim > 2:
        return tensordot(a, b, [[a.ndim - 1], [np.maximum(0, b.ndim - 2)]])
    else:
        return _dot(a, b)


def _tensordot_as_dot(a, b, axes, dot, batched):
    """
    Reduces a tensor dot product to a matrix or vector dot product. Based
    on code from Tijmen Tieleman's gnumpy
    (http://www.cs.toronto.edu/~tijmen/gnumpy.html).

    Please see the documentation of tensordot for the meaning of the a, b
    and axes arguments.

    :param dot: a function that accepts two symbolic variables and computes
                the appropriate dot product (e.g. dot, batched_dot)
    :type dot: function

    :param batched: whether to treat the first axis of a and b as a batch
                    axis.  If so, this axis will be preserved in the output,
                    allowing this function to be used also for batched
                    tensor dot products.
    :type batched: boolean

    :returns: a tensor with shape equal to the concatenation of a's shape
              (less any dimensions that were summed over) and b's shape
              (less the first dimension and any dimensions that were summed
              over).
    :rtype: symbolic tensor
    """
    a, b = as_tensor_variable(a), as_tensor_variable(b)

    if not np.isscalar(axes) and len(axes) != 2:
        raise ValueError(
            "Axes should be an integer or a "
            "list/tuple of len 2 ({axes} was provided)"
        )

    # if 'axes' is a number of axes to multiply and sum over (trailing axes
    # of a, leading axes of b), we can just reshape and use dot.
    elif np.isscalar(axes):
        axes = int(axes)

        for operand_name, operand in (("a", a), ("b", b)):
            if axes > operand.ndim:
                raise ValueError(
                    f"axes can not be larger than the dimension of {operand_name} "
                    f"({operand_name}.ndim={operand.ndim}, axes={axes})"
                )
            if batched and axes == operand.ndim:
                raise ValueError(
                    "axes to sum over must not include the batch axis "
                    f"of {operand_name} ({operand_name}.ndim={operand.ndim}, axes={axes})"
                )

        batch_axes = 1 if batched else 0
        a_outaxes = slice(0, a.ndim - axes)
        b_outaxes = slice(batch_axes + axes, b.ndim)
        outshape = concatenate([a.shape[a_outaxes], b.shape[b_outaxes]])
        outbcast = a.broadcastable[a_outaxes] + b.broadcastable[b_outaxes]
        outndim = len(outbcast)

        a_shape = [1] * 2
        b_shape = [1] * 2

        # compute total size of summed axes
        for i in range(0, axes):
            a_shape[1] *= a.shape[-(i + 1)]
            b_shape[0] *= b.shape[batch_axes + i]
        # compute total size of other axes
        for i in range(0, a.ndim - axes - batch_axes):
            a_shape[0] *= a.shape[batch_axes + i]
        for i in range(0, b.ndim - axes - batch_axes):
            b_shape[1] *= b.shape[-(i + 1)]

        if batched:
            a_shape.insert(0, a.shape[0])
            b_shape.insert(0, b.shape[0])

        a_reshaped = a.reshape(a_shape)
        b_reshaped = b.reshape(b_shape)

        out_reshaped = dot(a_reshaped, b_reshaped)
        out = out_reshaped.reshape(outshape, outndim)
        # Make sure the broadcastable pattern of the result is correct,
        # since some shape information can be lost in the reshapes.
        return patternbroadcast(out, outbcast)

    # if 'axes' is a list, transpose a and b such that the summed axes of a
    # are last and the summed axes of b are first.
    else:
        axes = [as_list(axes_) for axes_ in axes]

        if len(axes[0]) != len(axes[1]):
            raise ValueError("Axes elements must have the same length.")

        for i, (operand_name, operand) in enumerate((("a", a), ("b", b))):
            if len(axes[i]) > operand.ndim:
                raise ValueError(
                    f"axes[{i}] should be array_like with length less than "
                    f"the dimensions of {operand_name} ({operand_name}.ndim={operand.ndim}, len(axes[0])={len(axes[i])})."
                )
            if len(axes[i]) > 0 and np.max(axes[i]) >= operand.ndim:
                raise ValueError(
                    f"axes[{i}] contains dimensions greater than or equal "
                    f"to {operand_name}.ndim ({operand_name}.ndim={operand.ndim}, max(axes[0])={np.max(np.array(axes[i]))})."
                )
            if batched and 0 in axes[i]:
                raise ValueError(
                    "axes to sum over must not contain the batch axis "
                    f"(axes[{i}]={axes[i]})"
                )

        batch_axes = [0] if batched else []
        other_axes = [
            [x for x in range(operand.ndim) if x not in axes[i] and x not in batch_axes]
            for i, operand in enumerate((a, b))
        ]

        a_shuffled = a.dimshuffle(batch_axes + other_axes[0] + axes[0])
        b_shuffled = b.dimshuffle(batch_axes + axes[1] + other_axes[1])

        # now that a and b are in the right order, recur with integer axes
        return _tensordot_as_dot(
            a_shuffled, b_shuffled, len(axes[0]), dot=dot, batched=batched
        )


def tensordot(a, b, axes=2):
    """
    Compute a generalized dot product over provided axes.

    Given two tensors a and b, tensordot computes a generalized dot product over
    the provided axes. Aesara's implementation reduces all expressions to
    matrix or vector dot products and is based on code from Tijmen Tieleman's
    gnumpy (http://www.cs.toronto.edu/~tijmen/gnumpy.html).

    Parameters
    ----------
    a: symbolic tensor
        The first tensor variable.
    b: symbolic tensor
        The second tensor variable
    axes: int or array-like of length 2
        If an integer, the number of axes to sum over.
        If an array, it must have two array elements containing the axes
        to sum over in each tensor.

        Note that the default value of 2 is not guaranteed to work
        for all values of a and b, and an error will be raised if
        that is the case. The reason for keeping the default is to
        maintain the same signature as numpy's tensordot function
        (and np.tensordot raises analogous errors for non-compatible
        inputs).

        If an integer i, it is converted to an array containing
        the last i dimensions of the first tensor and the first
        i dimensions of the second tensor:
            axes = [list(range(a.ndim - i, b.ndim)), list(range(i))]

        If an array, its two elements must contain compatible axes
        of the two tensors. For example, [[1, 2], [2, 0]] means sum
        over the 2nd and 3rd axes of a and the 3rd and 1st axes of b.
        (Remember axes are zero-indexed!) The 2nd axis of a and the
        3rd axis of b must have the same shape; the same is true for
        the 3rd axis of a and the 1st axis of b.

    Returns
    -------
    symbolic tensor
        A tensor with shape equal to the concatenation of a's shape
        (less any dimensions that were summed over) and b's shape
        (less any dimensions that were summed over).

    Examples
    --------
    It may be helpful to consider an example to see what tensordot does.
    Aesara's implementation is identical to NumPy's. Here a has shape (2, 3, 4)
    and b has shape (5, 6, 4, 3). The axes to sum over are [[1, 2], [3, 2]] --
    note that a.shape[1] == b.shape[3] and a.shape[2] == b.shape[2]; these axes
    are compatible. The resulting tensor will have shape (2, 5, 6) -- the
    dimensions that are not being summed:

    >>> a = np.random.random((2,3,4))
    >>> b = np.random.random((5,6,4,3))

    #tensordot
    >>> c = np.tensordot(a, b, [[1,2],[3,2]])

    #loop replicating tensordot
    >>> a0, a1, a2 = a.shape
    >>> b0, b1, _, _ = b.shape
    >>> cloop = np.zeros((a0,b0,b1))

    #loop over non-summed indices -- these exist
    #in the tensor product.
    >>> for i in range(a0):
    ...     for j in range(b0):
    ...         for k in range(b1):
    ...             #loop over summed indices -- these don't exist
    ...             #in the tensor product.
    ...             for l in range(a1):
    ...                 for m in range(a2):
    ...                     cloop[i,j,k] += a[i,l,m] * b[j,k,m,l]

    >>> np.allclose(c, cloop)
    true

    This specific implementation avoids a loop by transposing a and b such that
    the summed axes of a are last and the summed axes of b are first. The
    resulting arrays are reshaped to 2 dimensions (or left as vectors, if
    appropriate) and a matrix or vector dot product is taken. The result is
    reshaped back to the required output dimensions.

    In an extreme case, no axes may be specified. The resulting tensor
    will have shape equal to the concatenation of the shapes of a and b:

    >>> c = np.tensordot(a, b, 0)
    >>> print(a.shape)
    (2,3,4)
    >>> print(b.shape)
    (5,6,4,3)
    >>> print(c.shape)
    (2,3,4,5,6,4,3)

    See the documentation of numpy.tensordot for more examples.

    """
    return _tensordot_as_dot(a, b, axes, dot=dot, batched=False)


def outer(x, y):
    """Return vector-vector outer product.

    If an input isn't a vector, we flatten it first.

    """
    if x.ndim != 1:
        x = x.flatten()
    if y.ndim != 1:
        y = y.flatten()
    return dot(x.dimshuffle(0, "x"), y.dimshuffle("x", 0))


class All(CAReduce):
    """Applies `logical and` to all the values of a tensor along the
    specified axis(es).

    """

    __props__ = ("axis",)
    nfunc_spec = ("all", 1, 1)

    def __init__(self, axis=None):
        super().__init__(aes.and_, axis)

    def _output_dtype(self, idtype):
        return "bool"

    def __str__(self):
        if self.axis is None:
            return "All"
        else:
            return "All{%s}" % ", ".join(map(str, self.axis))

    def make_node(self, input):
        input = as_tensor_variable(input)
        if input.dtype != "bool":
            input = neq(input, 0)
        ret = super().make_node(input)
        return ret

    def grad(self, inp, grads):
        (x,) = inp
        return [x.zeros_like(config.floatX)]


class Any(CAReduce):
    """Applies `bitwise or` to all the values of a tensor along the
    specified axis(es).

    """

    __props__ = ("axis",)
    nfunc_spec = ("any", 1, 1)

    def __init__(self, axis=None):
        super().__init__(aes.or_, axis)

    def _output_dtype(self, idtype):
        return "bool"

    def __str__(self):
        if self.axis is None:
            return "Any"
        else:
            return "Any{%s}" % ", ".join(map(str, self.axis))

    def make_node(self, input):
        input = as_tensor_variable(input)
        if input.dtype != "bool":
            input = neq(input, 0)
        ret = super().make_node(input)
        return ret

    def grad(self, inp, grads):
        (x,) = inp
        return [x.zeros_like(config.floatX)]


class Sum(CAReduceDtype):
    """
    Sums all the values of a tensor along the specified axis(es).

    Equivalent to `CAReduceDtype(scalar.add, axis=axis, dtype=dtype)`,
    with the difference that this defines the gradient of sum wrt its
    tensor input.

    Parameters
    ----------
    axis
        Axis(es) along which the tensor should be summed
        (use None to sum over all axes, and a list or tuple to sum along more
        than one axis).

    dtype
        The dtype of the internal accumulator and returned
        tensor. If None, then we use the default dtype which is the same as the
        input tensor's dtype except when:
        - the input dtype is a signed integer of precision < 64 bit, in
        which case we use int64
        - the input dtype is an unsigned integer of precision < 64 bit, in
        which case we use uint64
        This value does not depend on the value of "acc_dtype".

    acc_dtype
        The dtype of the internal accumulator.
        If None (default), we use the dtype in the list below,
        or the input dtype if its precision is higher:
        - for int dtypes, we use at least int64;
        - for uint dtypes, we use at least uint64;
        - for float dtypes, we use at least float64;
        - for complex dtypes, we use at least complex128.

    """

    __props__ = ("axis", "dtype", "acc_dtype")
    nfunc_spec = ("sum", 1, 1)

    def __init__(self, axis=None, dtype=None, acc_dtype=None):
        super().__init__(aes.add, axis=axis, dtype=dtype, acc_dtype=acc_dtype)

    def __str__(self):
        name = self.__class__.__name__
        axis = ""
        if self.axis is not None:
            axis = ", ".join(str(x) for x in self.axis)
            axis = f"axis=[{axis}], "
        return f"{name}{{{axis}acc_dtype={self.acc_dtype}}}"

    def L_op(self, inp, out, grads):
        (x,) = inp

        if out[0].dtype not in continuous_dtypes:
            return [x.zeros_like(dtype=config.floatX)]

        (gz,) = grads
        gz = as_tensor_variable(gz)
        axis = self.axis
        if axis is None:
            axis = list(range(x.type.ndim))
        if axis == ():
            return (gz,)
        new_dims = []
        i = 0
        for j, _ in enumerate(x.type.broadcastable):
            if j in axis:
                new_dims.append("x")
            else:
                new_dims.append(i)
                i += 1
        ds_op = DimShuffle(gz.type.broadcastable, new_dims)
        gx = Elemwise(aes.second)(x, ds_op(gz))
        return [gx]

    def R_op(self, inputs, eval_points):
        # There is just one element in inputs and eval_points, the axis are
        # part of self
        if None in eval_points:
            return [None]
        return self(*eval_points, **dict(return_list=True))


def sum(input, axis=None, dtype=None, keepdims=False, acc_dtype=None):
    """
    Computes the sum along the given axis(es) of a tensor `input`.

    When axis is None (the default value), the sum is performed
    over the flattened tensor.

    For full documentation see `Sum`.
    In particular please pay attention to the important warning when using
    a custom acc_dtype.

    Parameters
    ----------
    keepdims: bool
        If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the result
        will broadcast correctly against the original tensor.

    """

    out = Sum(axis=axis, dtype=dtype, acc_dtype=acc_dtype)(input)

    if keepdims:
        out = makeKeepDims(input, out, axis)
    return out


pprint.assign(Sum(), printing.FunctionPrinter("sum"))


class Prod(CAReduceDtype):
    """
    Multiplies all the values of a tensor along the specified axis(es).

    Equivalent to `CAReduce(scalar.mul, axis = axis)`, with the
    difference that this defines the gradient of prod wrt its tensor
    input.

    """

    __props__ = ("axis", "dtype", "acc_dtype")
    nfunc_spec = ("prod", 1, 1)

    def __init__(self, axis=None, dtype=None, acc_dtype=None, no_zeros_in_input=False):
        super().__init__(aes.mul, axis=axis, dtype=dtype, acc_dtype=acc_dtype)
        self.no_zeros_in_input = no_zeros_in_input

    def __setstate__(self, dct):
        super().__setstate__(dct)
        # Add default value to be able to reload old pickled objects.
        if "no_zeros_in_input" not in dct:
            self.no_zeros_in_input = False

    def L_op(self, inp, out, grads):
        """
        The grad of this Op could be very easy, if it is was not for the case
        where zeros are present in a given "group" (ie. elements reduced
        together to form the product).

        If no zeros are found in the elements of the product, then the
        partial derivative of the product relative to one of the elements
        (one of the inputs) is simply the product of the other elements.
        That's easy to see from the chain rule.

        Now the trick (with no zeros) is to take the overall product, then
        for every original element, the partial derivative is given by
        this product divided by the element itself (which equals the product
        of the other terms). This is easy to do by broadcasting the original
        product.

        (Note that we also need to broadcast-multiply by the
        "incoming gradient", ie. the gradient of the cost relative to the
        output/product).

        With zeros, things get more complicated. For a given group, we have 3
        cases:

        * No zeros in the group. Use previous trick.
        * If only one zero is present, then the gradient for that element is
            non-zero, but is zero for all others.
        * If more than one zero is present, then all the derivatives are zero.

        For the last two cases (with 1 or more zeros), we can't use the
        division trick, as this gives divisions by 0.

        Implementing that case-by-case logic is not as trivial, so a bunch of
        hacks are piled down here to do it. Notably, for the "only one zero"
        case, there's a special Op that computes the product of the elements
        in the group, minus the zero (see `ProdWithoutZeros`). The trick is then
        to use the division trick for groups with no zero, to use the
        `ProdWithoutZeros` op where there's only one zero, and to output a
        derivative of zero for any element part of a group with more than
        one zero.

        I do this by first counting the number of zeros in each group (see the
        `aet.eq` bits), then taking this or that behavior (see `aet.switch`)
        based on the result of this count.

        """
        (prod_in,) = inp
        (gz,) = grads

        if out[0].dtype in discrete_dtypes or self.acc_dtype in discrete_dtypes:
            # There is an int conversion in the way
            return [prod_in.zeros_like(dtype=config.floatX)]

        # Prepare the broadcasting that is used everywhere to broadcast
        # over the original groups (ie. broadcast over the elements of a given
        # product)
        gz = as_tensor_variable(gz)
        axis = self.axis
        if axis is None:
            axis = list(range(prod_in.type.ndim))
        if axis == ():
            return (gz,)
        new_dims = []
        i = 0
        for j, _ in enumerate(prod_in.type.broadcastable):
            if j in axis:
                new_dims.append("x")
            else:
                new_dims.append(i)
                i += 1

        # result of the product, broadcastable over groups
        prod_out = self(prod_in).dimshuffle(new_dims)
        # incoming gradient, broadcastable over groups
        gz = gz.dimshuffle(new_dims)

        # division trick if we don't have zeros. This will contain
        # NaNs to be eliminated in the `aet.switch` if we do have zeros.
        grad_case_without_zeros = gz * prod_out / prod_in

        if self.no_zeros_in_input:
            # this handles inputs with zeros, but only certain input shapes
            return [grad_case_without_zeros]
        else:

            where_zeros = eq(prod_in, 0.0)
            sum_where_zeros = sum(where_zeros, axis=self.axis)
            groups_with_single_zero = eq(sum_where_zeros, 1).dimshuffle(new_dims)
            # tensor with 0 everywhere except for those places where
            # a 0 part of a group with a single zero was to be found
            where_single_zero = groups_with_single_zero * where_zeros
            # further optimization to avoid computing ProdWithoutZeros
            # if the incoming gradient is 0
            where_gz_not_zero = neq(gz, 0.0)
            # only take ProdWithoutZeros for the groups with single zeros
            # with non-null incoming gradient
            where_to_take_prod_without_zeros = (
                groups_with_single_zero * where_gz_not_zero
            )
            # preprocess the original input so that we set 0 everywhere
            # except for groups that contain a single zero, to avoid computing
            # multiplications on other groups
            prod_without_zeros_in = where_to_take_prod_without_zeros * prod_in
            # TODO: put lazy switch here, if it'd work
            # this is pretty efficient already (no multiplication if 0), but
            # it'd be even better if we had a lazy if per element
            prod_without_zeros = ProdWithoutZeros(axis=self.axis)(prod_without_zeros_in)
            prod_without_zeros = prod_without_zeros.dimshuffle(new_dims)

            groups_without_zeros = eq(sum_where_zeros, 0).dimshuffle(new_dims)

            final_grad = switch(
                groups_without_zeros,
                grad_case_without_zeros,
                switch(where_single_zero, prod_without_zeros, 0.0) * gz,
            )

            return [final_grad]

    def c_code_cache_version(self):
        return (1,)


def prod(
    input,
    axis=None,
    dtype=None,
    keepdims=False,
    acc_dtype=None,
    no_zeros_in_input=False,
):
    """
    Computes the product along the given axis(es) of a tensor `input`.

    When axis is None (the default value), the product is performed
    over the flattened tensor.

    For full documentation see ``tensor.elemwise.Prod``.

    Parameters
    ----------
    keepdims: bool
        If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option, the result
        will broadcast correctly against the original tensor.

    """

    out = Prod(
        axis, dtype=dtype, acc_dtype=acc_dtype, no_zeros_in_input=no_zeros_in_input
    )(input)

    if keepdims:
        out = makeKeepDims(input, out, axis)
    return out


class MulWithoutZeros(BinaryScalarOp):
    # "identity" here is zero, as in Reduce we don't want to start
    # with reducing (1, something_else): this leads to the erroneous
    # case where a vector of zeros is reduced by binary reductions
    # of (1, 0), which always ends up as 1 (ie. the result for
    # the c version, for the product of [0,0,0], is 1.0)

    identity = 0.0
    commutative = True
    associative = True

    def impl(self, x, y):
        if x == 0:
            return y
        if y == 0:
            return x
        return x * y

    def c_code(self, node, name, inp, out, sub):
        x, y = inp
        (z,) = out
        return (
            "%(z)s = ((%(x)s == 0) ? (%(y)s) : "
            + "((%(y)s == 0) ? (%(x)s) : ((%(y)s)*(%(x)s))) );"
        ) % locals()

    def c_code_cache_version(self):
        return (1,)


mul_without_zeros = MulWithoutZeros(aes.upcast_out, name="mul_without_zeros")


class ProdWithoutZeros(CAReduceDtype):

    __props__ = ("axis", "dtype", "acc_dtype")

    def __init__(self, axis=None, dtype=None, acc_dtype=None):
        super().__init__(mul_without_zeros, axis=axis, dtype=dtype, acc_dtype=acc_dtype)

    def grad(self, inp, grads):
        from aesara.gradient import grad_not_implemented

        (a,) = inp
        a_grad = grad_not_implemented(
            self,
            0,
            a,
            "2nd derivatives of `product(a)` is not currently supported."
            "If `a` is guaranteed to contains no zeros, use "
            "`product(a, no_zeros_in_input=True)`.",
        )
        return [a_grad]


def any(x, axis=None, keepdims=False):
    out = Any(axis)(x)

    if keepdims:
        out = makeKeepDims(x, out, axis)
    return out


def all(x, axis=None, keepdims=False):
    out = All(axis)(x)

    if keepdims:
        out = makeKeepDims(x, out, axis)
    return out


def ptp(a, axis=None):
    """
    Range of values (maximum - minimum) along an axis.

    The name of the function comes from the acronym for peak to peak.

    Parameters
    ----------
    a
        Input tensor.
    axis
        Axis along which to find the peaks. By default, flatten the array.

    Returns
    -------
    array
        A new array holding the result.

    """

    a = as_tensor_variable(a)

    out = max(a, axis) - min(a, axis)

    return out


def power(x, y):
    return x ** y


__all__ = [
    "max_and_argmax",
    "max",
    "argmax",
    "min",
    "argmin",
    "smallest",
    "largest",
    "lt",
    "gt",
    "le",
    "ge",
    "eq",
    "neq",
    "isnan",
    "isinf",
    "allclose",
    "isclose",
    "and_",
    "bitwise_and",
    "or_",
    "bitwise_or",
    "xor",
    "bitwise_xor",
    "invert",
    "bitwise_not",
    "abs_",
    "exp",
    "exp2",
    "expm1",
    "neg",
    "inv",
    "log",
    "log2",
    "log10",
    "log1p",
    "sgn",
    "ceil",
    "floor",
    "trunc",
    "iround",
    "round",
    "round_half_to_even",
    "round_half_away_from_zero",
    "sqr",
    "square",
    "cov",
    "sqrt",
    "deg2rad",
    "rad2deg",
    "cos",
    "arccos",
    "sin",
    "arcsin",
    "tan",
    "arctan",
    "arctan2",
    "cosh",
    "arccosh",
    "sinh",
    "arcsinh",
    "tanh",
    "arctanh",
    "erf",
    "erfc",
    "erfcx",
    "erfinv",
    "erfcinv",
    "gamma",
    "gammaln",
    "psi",
    "tri_gamma",
    "chi2sf",
    "gammainc",
    "gammaincc",
    "gammau",
    "gammal",
    "j0",
    "j1",
    "jv",
    "i0",
    "i1",
    "iv",
    "sigmoid",
    "expit",
    "softplus",
    "real",
    "imag",
    "angle",
    "complex",
    "conj",
    "complex_from_polar",
    "sum",
    "prod",
    "mean",
    "var",
    "std",
    "std",
    "maximum",
    "minimum",
    "divmod",
    "add",
    "sub",
    "mul",
    "true_div",
    "int_div",
    "floor_div",
    "ceil_intdiv",
    "mod",
    "pow",
    "clip",
    "dot",
    "dense_dot",
    "tensordot",
    "outer",
    "any",
    "all",
    "ptp",
    "power",
]
