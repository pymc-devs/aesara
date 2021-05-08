import operator
import warnings
from functools import reduce, singledispatch
from numbers import Number
from textwrap import indent
from typing import Union

import numba
import numpy as np
import scipy
import scipy.special
from llvmlite.llvmpy.core import Type as llvm_Type
from numba import types
from numba.core.errors import TypingError
from numba.cpython.unsafe.tuple import tuple_setitem
from numba.extending import box
from numpy.core.multiarray import normalize_axis_index

from aesara.compile.ops import DeepCopyOp, ViewOp
from aesara.graph.basic import Apply
from aesara.graph.fg import FunctionGraph
from aesara.graph.type import Type
from aesara.link.utils import (
    compile_function_src,
    fgraph_to_python,
    get_name_for_object,
)
from aesara.scalar.basic import (
    Cast,
    Clip,
    Composite,
    Identity,
    Scalar,
    ScalarOp,
    Second,
)
from aesara.tensor.basic import (
    Alloc,
    AllocDiag,
    AllocEmpty,
    ARange,
    ExtractDiag,
    Eye,
    Join,
    MakeVector,
    Rebroadcast,
    ScalarFromTensor,
    TensorFromScalar,
)
from aesara.tensor.elemwise import CAReduce, DimShuffle, Elemwise
from aesara.tensor.extra_ops import (
    Bartlett,
    BroadcastTo,
    CumOp,
    DiffOp,
    FillDiagonal,
    FillDiagonalOffset,
    RavelMultiIndex,
    Repeat,
    SearchsortedOp,
    Unique,
    UnravelIndex,
)
from aesara.tensor.math import Dot, MaxAndArgmax
from aesara.tensor.nlinalg import SVD, Det, Eig, Eigh, MatrixInverse, QRFull
from aesara.tensor.nnet.basic import LogSoftmax, Softmax
from aesara.tensor.nnet.sigm import ScalarSoftplus
from aesara.tensor.shape import Reshape, Shape, Shape_i, SpecifyShape
from aesara.tensor.slinalg import Cholesky, Solve
from aesara.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedIncSubtensor1,
    AdvancedSubtensor,
    AdvancedSubtensor1,
    IncSubtensor,
    Subtensor,
)
from aesara.tensor.type import TensorType, tensor
from aesara.tensor.type_other import MakeSlice


def get_numba_type(
    aesara_type: Type, layout: str = "A", force_scalar: bool = False
) -> numba.types.Type:
    """Create a Numba type object for a ``Type``."""

    if isinstance(aesara_type, TensorType):
        dtype = aesara_type.numpy_dtype
        numba_dtype = numba.from_dtype(dtype)
        if force_scalar:
            return numba_dtype
        return numba.types.Array(numba_dtype, aesara_type.ndim, layout)
    elif isinstance(aesara_type, Scalar):
        dtype = np.dtype(aesara_type.dtype)
        numba_dtype = numba.from_dtype(dtype)
        return numba_dtype
    else:
        raise NotImplementedError(f"Numba type not implemented for {aesara_type}")


def create_numba_signature(node: Apply, force_scalar: bool = False) -> numba.types.Type:
    """Create a Numba type for the signature of an ``Apply`` node."""
    input_types = []
    for inp in node.inputs:
        input_types.append(get_numba_type(inp.type, force_scalar=force_scalar))

    output_types = []
    for out in node.outputs:
        output_types.append(get_numba_type(out.type, force_scalar=force_scalar))

    if len(output_types) > 1:
        return numba.types.Tuple(output_types)(*input_types)
    elif len(output_types) == 1:
        return output_types[0](*input_types)
    else:
        return numba.types.void(*input_types)


def slice_new(self, start, stop, step):
    fnty = llvm_Type.function(self.pyobj, [self.pyobj, self.pyobj, self.pyobj])
    fn = self._get_function(fnty, name="PySlice_New")
    return self.builder.call(fn, [start, stop, step])


def enable_slice_boxing():
    """Enable boxing for Numba's native ``slice``s.

    TODO: this can be removed when https://github.com/numba/numba/pull/6939 is
    merged and a release is made.
    """

    @box(types.SliceType)
    def box_slice(typ, val, c):
        """Implement boxing for ``slice`` objects in Numba.

        This makes it possible to return an Numba's internal representation of a
        ``slice`` object as a proper ``slice`` to Python.
        """

        start = c.box(types.int64, c.builder.extract_value(val, 0))
        stop = c.box(types.int64, c.builder.extract_value(val, 1))
        if typ.has_step:
            step = c.box(types.int64, c.builder.extract_value(val, 2))
        else:
            step = c.pyapi.get_null_object()

        slice_val = slice_new(c.pyapi, start, stop, step)

        return slice_val

    @numba.extending.overload(operator.contains)
    def in_seq_empty_tuple(x, y):
        if isinstance(x, types.Tuple) and not x.types:
            return lambda x, y: False


enable_slice_boxing()


@numba.generated_jit(nopython=True)
def to_scalar(x):
    if isinstance(x, (numba.types.Number, numba.types.Boolean)):
        return lambda x: x
    elif isinstance(x, numba.types.Array):
        return lambda x: x.item()
    else:
        raise TypingError(f"{x} must be a scalar compatible type.")


def enable_slice_literals():
    """Enable lowering for ``SliceLiteral``s.

    TODO: This can be removed once https://github.com/numba/numba/pull/6996 is merged
    and a release is made.
    """
    from numba.core import types
    from numba.core.datamodel.models import SliceModel
    from numba.core.datamodel.registry import register_default
    from numba.core.imputils import lower_cast, lower_constant
    from numba.core.types.misc import SliceLiteral
    from numba.cpython.slicing import get_defaults

    register_default(numba.types.misc.SliceLiteral)(SliceModel)

    @property
    def key(self):
        return self.name

    SliceLiteral.key = key

    def make_slice_from_constant(context, builder, ty, pyval):
        sli = context.make_helper(builder, ty)
        lty = context.get_value_type(types.intp)

        (
            default_start_pos,
            default_start_neg,
            default_stop_pos,
            default_stop_neg,
            default_step,
        ) = [context.get_constant(types.intp, x) for x in get_defaults(context)]

        step = pyval.step
        if step is None:
            step_is_neg = False
            step = default_step
        else:
            step_is_neg = step < 0
            step = lty(step)

        start = pyval.start
        if start is None:
            if step_is_neg:
                start = default_start_neg
            else:
                start = default_start_pos
        else:
            start = lty(start)

        stop = pyval.stop
        if stop is None:
            if step_is_neg:
                stop = default_stop_neg
            else:
                stop = default_stop_pos
        else:
            stop = lty(stop)

        sli.start = start
        sli.stop = stop
        sli.step = step

        return sli._getvalue()

    @lower_constant(numba.types.SliceType)
    def constant_slice(context, builder, ty, pyval):
        if isinstance(ty, types.Literal):
            typ = ty.literal_type
        else:
            typ = ty

        return make_slice_from_constant(context, builder, typ, pyval)

    @lower_cast(numba.types.misc.SliceLiteral, numba.types.SliceType)
    def cast_from_literal(context, builder, fromty, toty, val):
        return make_slice_from_constant(
            context,
            builder,
            toty,
            fromty.literal_value,
        )


enable_slice_literals()


def create_tuple_creator(f, n):
    """Construct a compile-time ``tuple``-comprehension-like loop.

    See https://github.com/numba/numba/issues/2771#issuecomment-414358902
    """
    assert n > 0

    f = numba.njit(f)

    @numba.njit
    def creator(args):
        return (f(0, *args),)

    for i in range(1, n):

        @numba.njit
        def creator(args, creator=creator, i=i):
            return creator(args) + (f(i, *args),)

    return numba.njit(lambda *args: creator(args))


def create_tuple_string(x):
    args = ", ".join(x + ([""] if len(x) == 1 else []))
    return f"({args})"


@singledispatch
def numba_typify(data, dtype=None, **kwargs):
    return data


@singledispatch
def numba_funcify(op, node=None, storage_map=None, **kwargs):
    """Create a Numba compatible function from an Aesara `Op`."""

    warnings.warn(
        (f"Numba will use object mode to run {op}'s perform method"),
        UserWarning,
    )

    n_outputs = len(node.outputs)

    if n_outputs > 1:
        ret_sig = numba.types.Tuple([get_numba_type(o.type) for o in node.outputs])
    else:
        ret_sig = get_numba_type(node.outputs[0].type)

    @numba.njit
    def perform(*inputs):
        with numba.objmode(ret=ret_sig):
            outputs = [[None] for i in range(n_outputs)]
            op.perform(node, inputs, outputs)
            outputs = tuple([o[0] for o in outputs])
            if n_outputs == 1:
                ret = outputs[0]
            else:
                ret = outputs
        return ret

    return perform


@numba_funcify.register(FunctionGraph)
def numba_funcify_FunctionGraph(
    fgraph,
    node=None,
    fgraph_name="numba_funcified_fgraph",
    **kwargs,
):
    return fgraph_to_python(
        fgraph,
        numba_funcify,
        type_conversion_fn=numba_typify,
        fgraph_name=fgraph_name,
        **kwargs,
    )


@numba_funcify.register(ScalarOp)
def numba_funcify_ScalarOp(op, node, **kwargs):
    # TODO: Do we need to cache these functions so that we don't end up
    # compiling the same Numba function over and over again?

    scalar_func_name = op.nfunc_spec[0]

    if scalar_func_name.startswith("scipy."):
        func_package = scipy
        scalar_func_name = scalar_func_name.split(".", 1)[-1]
    else:
        func_package = np

    if "." in scalar_func_name:
        scalar_func = reduce(getattr, [scipy] + scalar_func_name.split("."))
    else:
        scalar_func = getattr(func_package, scalar_func_name)

    input_names = ", ".join([v.auto_name for v in node.inputs])

    global_env = {"scalar_func": scalar_func}

    scalar_op_fn_name = get_name_for_object(scalar_func)
    scalar_op_src = f"""
def {scalar_op_fn_name}({input_names}):
    return scalar_func({input_names})
    """
    scalar_op_fn = compile_function_src(scalar_op_src, scalar_op_fn_name, global_env)

    return numba.njit(scalar_op_fn)


@numba_funcify.register(Elemwise)
def numba_funcify_Elemwise(op, node, use_signature=False, identity=None, **kwargs):
    scalar_op_fn = numba_funcify(op.scalar_op, node, **kwargs)

    input_names = ", ".join([v.auto_name for v in node.inputs])

    if use_signature:
        signature = [create_numba_signature(node, force_scalar=True)]
    else:
        signature = []

    numba_vectorize = numba.vectorize(signature, identity=identity)
    global_env = {"scalar_op": scalar_op_fn, "numba_vectorize": numba_vectorize}

    elemwise_fn_name = f"elemwise_{get_name_for_object(scalar_op_fn)}"
    elemwise_src = f"""
@numba_vectorize
def {elemwise_fn_name}({input_names}):
    return scalar_op({input_names})
    """
    elemwise_fn = compile_function_src(elemwise_src, elemwise_fn_name, global_env)

    return elemwise_fn


def create_axis_reducer(
    reduce_fn: numba.np.ufunc.dufunc.DUFunc,
    identity: Union[np.ndarray, Number],
    axis: int,
    ndim: int,
    dtype: numba.types.Type,
    keepdims: bool = False,
) -> numba.core.dispatcher.Dispatcher:
    """Create a Numba JITed function that performs a NumPy reduction on a given axis.

    Parameters
    ==========
    reduce_fn:
        The Numba ``ufunc`` representing a binary op that can perform the
        reduction on arbitrary ``ndarray``s.
    identity:
        The identity value for the reduction.
    axis:
        The axis to reduce.
    ndim:
        The number of dimensions of the result.
    dtype:
        The data type of the result.
    keepdims:
        Determines whether or not the reduced dimension is retained.
    """
    if ndim > 1:

        if keepdims:

            @numba.njit(inline="always")
            def set_out_dims(x):
                return np.expand_dims(x, axis)

        else:

            @numba.njit(inline="always")
            def set_out_dims(x):
                return x

        res_shape_tuple_ctor = create_tuple_creator(
            lambda i, shape: shape[i] if i < axis else shape[i + 1], ndim - 1
        )

        reaxis_first = (axis,) + tuple(i for i in range(ndim) if i != axis)

        @numba.njit(boundscheck=False)
        def careduce_axis(x):
            res_shape = res_shape_tuple_ctor(x.shape)
            x_axis_first = x.transpose(reaxis_first)

            res = np.full(res_shape, to_scalar(identity), dtype=dtype)
            for m in range(x.shape[axis]):
                reduce_fn(res, x_axis_first[m], res)

            return set_out_dims(res)

    else:

        if keepdims:

            @numba.njit(inline="always")
            def set_out_dims(x):
                return np.array([x], dtype)

        else:

            @numba.njit(inline="always")
            def set_out_dims(x):
                return direct_cast(x, dtype)

        @numba.njit(boundscheck=False)
        def careduce_axis(x):
            res = to_scalar(identity)
            for val in x:
                res = reduce_fn(res, val)
            return set_out_dims(res)

    return careduce_axis


def create_multiaxis_reducer(
    reduce_fn, identity, axes, ndim, dtype, input_name="input"
):
    careduce_fn_name = f"careduce_{get_name_for_object(reduce_fn)}"
    careduce_axes_fns = ()
    to_reduce = reversed(sorted(axes))
    careduce_lines_src = []
    var_name = input_name
    for i, axis in enumerate(to_reduce):
        careduce_axes_fns += (
            create_axis_reducer(reduce_fn, identity, axis - i, ndim, dtype),
        )
        ndim -= 1
        last_var_name = var_name
        var_name = f"axis_{i}_res"
        careduce_lines_src.append(
            f"{var_name} = careduce_axes_fns[{i}]({last_var_name})"
        )

    careduce_assign_lines = indent("\n".join(careduce_lines_src), " " * 4)
    careduce_def_src = f"""
def {careduce_fn_name}({input_name}):
{careduce_assign_lines}
    return {var_name}
    """

    global_env = {"careduce_axes_fns": careduce_axes_fns}

    careduce_fn = compile_function_src(careduce_def_src, careduce_fn_name, global_env)
    return careduce_fn


@numba_funcify.register(CAReduce)
def numba_funcify_CAReduce(op, node, **kwargs):
    axes = op.axis
    if axes is None:
        axes = list(range(node.inputs[0].ndim))

    if hasattr(op, "acc_dtype") and op.acc_dtype is not None:
        acc_dtype = op.acc_dtype
    else:
        acc_dtype = node.outputs[0].type.dtype

    np_acc_dtype = np.dtype(acc_dtype)

    scalar_op_identity = np.asarray(op.scalar_op.identity, dtype=np_acc_dtype)

    acc_dtype = numba.np.numpy_support.from_dtype(np_acc_dtype)

    scalar_nfunc_spec = op.scalar_op.nfunc_spec

    # We construct a dummy `Apply` that has the minimum required number of
    # inputs for the scalar `Op`.  Without this, we would get a scalar function
    # with too few arguments.
    dummy_node = Apply(
        op,
        [tensor(acc_dtype, [False]) for i in range(scalar_nfunc_spec[1])],
        [tensor(acc_dtype, [False]) for o in range(scalar_nfunc_spec[2])],
    )
    elemwise_fn = numba_funcify_Elemwise(op, dummy_node, use_signature=True, **kwargs)

    input_name = get_name_for_object(node.inputs[0])
    ndim = node.inputs[0].ndim
    careduce_fn = create_multiaxis_reducer(
        elemwise_fn, scalar_op_identity, axes, ndim, acc_dtype, input_name=input_name
    )

    return numba.njit(careduce_fn)


@numba_funcify.register(Composite)
def numba_funcify_Composite(op, node, **kwargs):
    numba_impl = numba.njit(numba_funcify(op.fgraph, **kwargs))

    @numba.njit
    def composite(*args):
        return numba_impl(*args)[0]

    return composite


def create_index_func(node, objmode=False):
    """Create a Python function that assembles and uses an index on an array."""

    def convert_indices(indices, entry):
        if indices and isinstance(entry, Type):
            rval = indices.pop(0)
            return rval.auto_name
        elif isinstance(entry, slice):
            return (
                f"slice({convert_indices(indices, entry.start)}, "
                f"{convert_indices(indices, entry.stop)}, "
                f"{convert_indices(indices, entry.step)})"
            )
        elif isinstance(entry, type(None)):
            return "None"
        else:
            raise ValueError()

    set_or_inc = isinstance(
        node.op, (IncSubtensor, AdvancedIncSubtensor1, AdvancedIncSubtensor)
    )
    index_start_idx = 1 + int(set_or_inc)

    input_names = [v.auto_name for v in node.inputs]
    op_indices = list(node.inputs[index_start_idx:])
    idx_list = getattr(node.op, "idx_list", None)

    indices_creation_src = (
        tuple(convert_indices(op_indices, idx) for idx in idx_list)
        if idx_list
        else tuple(input_names[index_start_idx:])
    )

    if len(indices_creation_src) == 1:
        indices_creation_src = f"indices = ({indices_creation_src[0]},)"
    else:
        indices_creation_src = ", ".join(indices_creation_src)
        indices_creation_src = f"indices = ({indices_creation_src})"

    if set_or_inc:
        fn_name = "incsubtensor"
        if node.op.inplace:
            index_prologue = f"z = {input_names[0]}"
        else:
            index_prologue = f"z = np.copy({input_names[0]})"

        if node.inputs[1].ndim == 0:
            # TODO FIXME: This is a hack to get around a weird Numba typing
            # issue.  See https://github.com/numba/numba/issues/6000
            y_name = f"{input_names[1]}.item()"
        else:
            y_name = input_names[1]

        if node.op.set_instead_of_inc:
            index_body = f"z[indices] = {y_name}"
        else:
            index_body = f"z[indices] += {y_name}"
    else:
        fn_name = "subtensor"
        index_prologue = ""
        index_body = f"z = {input_names[0]}[indices]"

    if objmode:
        output_var = node.outputs[0]

        if not set_or_inc:
            # Since `z` is being "created" while in object mode, it's
            # considered an "outgoing" variable and needs to be manually typed
            output_sig = f"z='{output_var.dtype}[{', '.join([':'] * output_var.ndim)}]'"
        else:
            output_sig = ""

        index_body = f"""
    with objmode({output_sig}):
        {index_body}
        """

    subtensor_def_src = f"""
def {fn_name}({", ".join(input_names)}):
    {index_prologue}
    {indices_creation_src}
    {index_body}
    return z
    """

    return subtensor_def_src


@numba_funcify.register(Subtensor)
@numba_funcify.register(AdvancedSubtensor)
@numba_funcify.register(AdvancedSubtensor1)
def numba_funcify_Subtensor(op, node, **kwargs):

    subtensor_def_src = create_index_func(
        node, objmode=isinstance(op, AdvancedSubtensor)
    )

    global_env = {"np": np, "objmode": numba.objmode}

    subtensor_fn = compile_function_src(subtensor_def_src, "subtensor", global_env)

    return numba.njit(subtensor_fn)


@numba_funcify.register(IncSubtensor)
@numba_funcify.register(AdvancedIncSubtensor)
@numba_funcify.register(AdvancedIncSubtensor1)
def numba_funcify_IncSubtensor(op, node, **kwargs):

    incsubtensor_def_src = create_index_func(
        node, objmode=isinstance(op, AdvancedIncSubtensor)
    )

    global_env = {"np": np, "objmode": numba.objmode}

    incsubtensor_fn = compile_function_src(
        incsubtensor_def_src, "incsubtensor", global_env
    )

    return numba.njit(incsubtensor_fn)


@numba_funcify.register(DeepCopyOp)
def numba_funcify_DeepCopyOp(op, node, **kwargs):

    # Scalars are apparently returned as actual Python scalar types and not
    # NumPy scalars, so we need two separate Numba functions for each case.
    if node.outputs[0].type.ndim == 0:
        # TODO: Do we really need to compile a pass-through function like this?
        @numba.njit
        def deepcopyop(x):
            return x

    else:

        @numba.njit
        def deepcopyop(x):
            return x.copy()

    return deepcopyop


@numba_funcify.register(MakeSlice)
def numba_funcify_MakeSlice(op, **kwargs):
    @numba.njit
    def makeslice(*x):
        return slice(*x)

    return makeslice


@numba_funcify.register(MakeVector)
def numba_funcify_MakeVector(op, **kwargs):
    dtype = np.dtype(op.dtype)

    @numba.njit
    def makevector(*args):
        return np.array([a.item() for a in args], dtype=dtype)

    return makevector


@numba_funcify.register(Shape)
def numba_funcify_Shape(op, **kwargs):
    @numba.njit
    def shape(x):
        return np.asarray(np.shape(x))

    return shape


@numba_funcify.register(Shape_i)
def numba_funcify_Shape_i(op, **kwargs):
    i = op.i

    @numba.njit
    def shape_i(x):
        return np.shape(x)[i]

    return shape_i


@numba_funcify.register(TensorFromScalar)
def numba_funcify_TensorFromScalar(op, **kwargs):
    @numba.njit
    def tensor_from_scalar(x):
        return np.array(x)

    return tensor_from_scalar


@numba_funcify.register(ScalarFromTensor)
def numba_funcify_ScalarFromTensor(op, **kwargs):
    @numba.njit
    def scalar_from_tensor(x):
        return x.item()

    return scalar_from_tensor


@numba_funcify.register(AllocEmpty)
def numba_funcify_AllocEmpty(op, node, **kwargs):

    global_env = {"np": np, "to_scalar": to_scalar, "dtype": op.dtype}

    shape_var_names = [v.auto_name for v in node.inputs]
    shape_var_item_names = [f"{name}_item" for name in shape_var_names]
    shapes_to_items_src = indent(
        "\n".join(
            [
                f"{item_name} = to_scalar({shape_name})"
                for item_name, shape_name in zip(shape_var_item_names, shape_var_names)
            ]
        ),
        " " * 4,
    )

    alloc_def_src = f"""
def allocempty({", ".join(shape_var_names)}):
{shapes_to_items_src}
    scalar_shape = {create_tuple_string(shape_var_item_names)}
    return np.empty(scalar_shape, dtype)
    """

    alloc_fn = compile_function_src(alloc_def_src, "allocempty", global_env)

    return numba.njit(alloc_fn)


@numba_funcify.register(Alloc)
def numba_funcify_Alloc(op, node, **kwargs):

    global_env = {"np": np, "to_scalar": to_scalar}

    shape_var_names = [v.auto_name for v in node.inputs[1:]]
    shape_var_item_names = [f"{name}_item" for name in shape_var_names]
    shapes_to_items_src = indent(
        "\n".join(
            [
                f"{item_name} = to_scalar({shape_name})"
                for item_name, shape_name in zip(shape_var_item_names, shape_var_names)
            ]
        ),
        " " * 4,
    )

    alloc_def_src = f"""
def alloc(val, {", ".join(shape_var_names)}):
    val_np = np.asarray(val)
{shapes_to_items_src}
    scalar_shape = {create_tuple_string(shape_var_item_names)}
    res = np.empty(scalar_shape, dtype=val_np.dtype)
    res[...] = val_np
    return res
    """

    alloc_fn = compile_function_src(alloc_def_src, "alloc", global_env)

    return numba.njit(alloc_fn)


@numba_funcify.register(AllocDiag)
def numba_funcify_AllocDiag(op, **kwargs):
    offset = op.offset

    @numba.njit
    def allocdiag(v):
        return np.diag(v, k=offset)

    return allocdiag


@numba_funcify.register(Second)
def numba_funcify_Second(op, node, **kwargs):
    @numba.njit
    def second(x, y):
        return y

    return second


@numba_funcify.register(DimShuffle)
def numba_funcify_DimShuffle(op, **kwargs):
    shuffle = tuple(op.shuffle)
    drop = tuple(op.drop)
    augment = tuple(op.augment)
    inplace = op.inplace

    ndim_new_shape = len(shuffle) + len(augment)
    create_zeros_tuple = create_tuple_creator(lambda _: 0, ndim_new_shape)

    if len(shuffle) > 0:

        @numba.njit
        def populate_new_shape(i, j, new_shape, shuffle_shape):
            if i in augment:
                new_shape = tuple_setitem(new_shape, i, 1)
                return j, new_shape
            else:
                new_shape = tuple_setitem(new_shape, i, shuffle_shape[j])
                return j + 1, new_shape

    else:
        # When `len(shuffle) == 0`, the `shuffle_shape[j]` expression above is
        # is typed as `getitem(Tuple(), int)`, which has no implementation
        # (since getting an item from an empty sequence doesn't make sense).
        # To avoid this compile-time error, we omit the expression altogether.
        @numba.njit
        def populate_new_shape(i, j, new_shape, shuffle_shape):
            new_shape = tuple_setitem(new_shape, i, 1)
            return j, new_shape

    @numba.njit
    def dimshuffle_inner(x, shuffle):
        res = np.transpose(x, shuffle + drop)
        shuffle_shape = res.shape[: len(shuffle)]

        new_shape = create_zeros_tuple()

        j = 0
        for i in range(len(new_shape)):
            j, new_shape = populate_new_shape(i, j, new_shape, shuffle_shape)

        # FIXME: Numba's `array.reshape` only accepts C arrays.
        res_reshape = np.reshape(np.ascontiguousarray(res), new_shape)

        if not inplace:
            return res_reshape.copy()
        else:
            return res_reshape

    # Without the following wrapper function we would see this error:
    # E   No implementation of function Function(<built-in function getitem>) found for signature:
    # E
    # E    >>> getitem(UniTuple(int64 x 2), slice<a:b>)
    # E
    # E   There are 22 candidate implementations:
    # E      - Of which 22 did not match due to:
    # E      Overload of function 'getitem': File: <numerous>: Line N/A.
    # E        With argument(s): '(UniTuple(int64 x 2), slice<a:b>)':
    # E       No match.
    # ...(on this line)...
    # E           shuffle_shape = res.shape[: len(shuffle)]
    @numba.njit
    def dimshuffle(x):
        return dimshuffle_inner(x, shuffle)

    return dimshuffle


@numba_funcify.register(Rebroadcast)
def numba_funcify_Rebroadcast(op, **kwargs):
    op_axis = tuple(op.axis.items())

    @numba.njit
    def rebroadcast(x):
        for axis, value in numba.literal_unroll(op_axis):
            if value and x.shape[axis] != 1:
                raise ValueError(
                    ("Dimension in Rebroadcast's input was supposed to be 1")
                )
        return x

    return rebroadcast


@numba.extending.intrinsic
def direct_cast(typingctx, val, typ):

    if isinstance(typ, numba.types.TypeRef):
        casted = typ.instance_type
    elif isinstance(typ, numba.types.DTypeSpec):
        casted = typ.dtype
    else:
        casted = typ

    sig = casted(casted, typ)

    def codegen(context, builder, signature, args):
        val, _ = args
        context.nrt.incref(builder, signature.return_type, val)
        return val

    return sig, codegen


@numba_funcify.register(Cast)
def numba_funcify_Cast(op, **kwargs):

    dtype = np.dtype(op.o_type.dtype)
    dtype = numba.np.numpy_support.from_dtype(dtype)

    @numba.njit
    def cast(x):
        return direct_cast(x, dtype)

    return cast


@numba_funcify.register(Reshape)
def numba_funcify_Reshape(op, **kwargs):

    ndim = op.ndim
    # TODO: It might be possible/better to use
    # `numba.np.unsafe.ndarray.to_fixed_tuple` here instead
    create_zeros_tuple = create_tuple_creator(lambda _: 0, ndim)

    @numba.njit
    def reshape(x, shape):

        new_shape = create_zeros_tuple()

        for i in numba.literal_unroll(range(ndim)):
            new_shape = tuple_setitem(new_shape, i, shape[i])

        return np.reshape(x, new_shape)

    return reshape


@numba_funcify.register(SpecifyShape)
def numba_funcify_SpecifyShape(op, **kwargs):
    @numba.njit
    def specifyshape(x, shape):
        assert np.array_equal(x.shape, shape)
        return x

    return specifyshape


@numba_funcify.register(Identity)
@numba_funcify.register(ViewOp)
def numba_funcify_ViewOp(op, **kwargs):
    @numba.njit
    def viewop(x):
        return x

    return viewop


@numba_funcify.register(Clip)
def numba_funcify_Clip(op, **kwargs):
    @numba.njit
    def clip(_x, _min, _max):
        x = to_scalar(_x)
        min = to_scalar(_min)
        max = to_scalar(_max)
        return np.where(x < min, min, to_scalar(np.where(x > max, max, x)))

    return clip


@numba_funcify.register(ARange)
def numba_funcify_ARange(op, **kwargs):
    dtype = np.dtype(op.dtype)
    dtype = numba.np.numpy_support.from_dtype(dtype)

    @numba.njit
    def arange(start, stop, step):
        return np.arange(
            to_scalar(start), to_scalar(stop), to_scalar(step), dtype=dtype
        )

    return arange


@numba_funcify.register(Join)
def numba_funcify_Join(op, **kwargs):
    view = op.view

    if view != -1:
        # TODO: Where (and why) is this `Join.view` even being used?  From a
        # quick search, the answer appears to be "nowhere", so we should
        # probably just remove it.
        raise NotImplementedError("The `view` parameter to `Join` is not supported")

    @numba.njit
    def join(axis, *tensors):
        return np.concatenate(tensors, to_scalar(axis))

    return join


@numba_funcify.register(ExtractDiag)
def numba_funcify_ExtractDiag(op, **kwargs):
    offset = op.offset
    # axis1 = op.axis1
    # axis2 = op.axis2

    @numba.njit
    def extract_diag(x):
        return np.diag(x, k=offset)

    return extract_diag


@numba_funcify.register(Eye)
def numba_funcify_Eye(op, **kwargs):
    dtype = np.dtype(op.dtype)
    dtype = numba.np.numpy_support.from_dtype(dtype)

    @numba.njit
    def eye(N, M, k):
        return np.eye(to_scalar(N), to_scalar(M), to_scalar(k), dtype=dtype)

    return eye


@numba_funcify.register(Bartlett)
def numba_funcify_Bartlett(op, **kwargs):
    @numba.njit
    def bartlett(x):
        return np.bartlett(to_scalar(x))

    return bartlett


@numba_funcify.register(CumOp)
def numba_funcify_CumOp(op, node, **kwargs):
    axis = op.axis
    mode = op.mode
    ndim = node.outputs[0].ndim

    reaxis_first = (axis,) + tuple(i for i in range(ndim) if i != axis)

    if mode == "add":
        np_func = np.add
        identity = 0
    else:
        np_func = np.multiply
        identity = 1

    @numba.njit(boundscheck=False)
    def cumop(x):
        out_dtype = x.dtype
        if x.shape[axis] < 2:
            return x.astype(out_dtype)

        x_axis_first = x.transpose(reaxis_first)
        res = np.empty(x_axis_first.shape, dtype=out_dtype)

        for m in range(x.shape[axis]):
            if m == 0:
                np_func(identity, x_axis_first[m], res[m])
            else:
                np_func(res[m - 1], x_axis_first[m], res[m])

        return res.transpose(reaxis_first)

    return cumop


@numba_funcify.register(DiffOp)
def numba_funcify_DiffOp(op, node, **kwargs):
    n = op.n
    axis = op.axis
    ndim = node.inputs[0].ndim
    dtype = node.outputs[0].dtype

    axis = normalize_axis_index(axis, ndim)

    slice1 = [slice(None)] * ndim
    slice2 = [slice(None)] * ndim
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    slice1 = tuple(slice1)
    slice2 = tuple(slice2)

    op = np.not_equal if dtype == "bool" else np.subtract

    @numba.njit(boundscheck=False)
    def diffop(x):
        res = x.copy()

        for _ in range(n):
            res = op(res[slice1], res[slice2])

        return res

    return diffop


@numba_funcify.register(FillDiagonal)
def numba_funcify_FillDiagonal(op, **kwargs):
    @numba.njit
    def filldiagonal(a, val):
        np.fill_diagonal(a, val)
        return a

    return filldiagonal


@numba_funcify.register(FillDiagonalOffset)
def numba_funcify_FillDiagonalOffset(op, node, **kwargs):
    @numba.njit
    def filldiagonaloffset(a, val, offset):
        height, width = a.shape

        if offset >= 0:
            start = to_scalar(offset)
            num_of_step = min(min(width, height), width - offset)
        else:
            start = -to_scalar(offset) * a.shape[1]
            num_of_step = min(min(width, height), height + offset)

        step = a.shape[1] + 1
        end = start + step * num_of_step
        b = a.ravel()
        b[start:end:step] = val
        # TODO: This isn't implemented in Numba
        # a.flat[start:end:step] = val
        # return a
        return b.reshape(a.shape)

    return filldiagonaloffset


@numba_funcify.register(RavelMultiIndex)
def numba_funcify_RavelMultiIndex(op, node, **kwargs):

    mode = op.mode
    order = op.order

    if order != "C":
        raise NotImplementedError(
            "Numba does not implement `order` in `numpy.ravel_multi_index`"
        )

    if mode == "raise":

        @numba.njit
        def mode_fn(*args):
            raise ValueError("invalid entry in coordinates array")

    elif mode == "wrap":

        @numba.njit(inline="always")
        def mode_fn(new_arr, i, j, v, d):
            new_arr[i, j] = v % d

    elif mode == "clip":

        @numba.njit(inline="always")
        def mode_fn(new_arr, i, j, v, d):
            new_arr[i, j] = min(max(v, 0), d - 1)

    if node.inputs[0].ndim == 0:

        @numba.njit
        def ravelmultiindex(*inp):
            shape = inp[-1]
            arr = np.stack(inp[:-1])

            new_arr = arr.T.astype(np.float64).copy()
            for i, b in enumerate(new_arr):
                if b < 0 or b >= shape[i]:
                    mode_fn(new_arr, i, 0, b, shape[i])

            a = np.ones(len(shape), dtype=np.float64)
            a[: len(shape) - 1] = np.cumprod(shape[-1:0:-1])[::-1]
            return np.array(a.dot(new_arr.T), dtype=np.int64)

    else:

        @numba.njit
        def ravelmultiindex(*inp):
            shape = inp[-1]
            arr = np.stack(inp[:-1])

            new_arr = arr.T.astype(np.float64).copy()
            for i, b in enumerate(new_arr):
                for j, (d, v) in enumerate(zip(shape, b)):
                    if v < 0 or v >= d:
                        mode_fn(new_arr, i, j, v, d)

            a = np.ones(len(shape), dtype=np.float64)
            a[: len(shape) - 1] = np.cumprod(shape[-1:0:-1])[::-1]
            return a.dot(new_arr.T).astype(np.int64)

    return ravelmultiindex


@numba_funcify.register(Repeat)
def numba_funcify_Repeat(op, node, **kwargs):
    axis = op.axis

    use_python = False

    if axis is not None:
        use_python = True

    if use_python:

        warnings.warn(
            (
                "Numba will use object mode to allow the "
                "`axis` argument to `numpy.repeat`."
            ),
            UserWarning,
        )

        ret_sig = get_numba_type(node.outputs[0].type)

        @numba.njit
        def repeatop(x, repeats):
            with numba.objmode(ret=ret_sig):
                ret = np.repeat(x, repeats, axis)
            return ret

    else:
        repeats_ndim = node.inputs[1].ndim

        if repeats_ndim == 0:

            @numba.njit
            def repeatop(x, repeats):
                return np.repeat(x, repeats.item())

        else:

            @numba.njit
            def repeatop(x, repeats):
                return np.repeat(x, repeats)

    return repeatop


@numba_funcify.register(Unique)
def numba_funcify_Unique(op, node, **kwargs):
    axis = op.axis

    use_python = False

    if axis is not None:
        use_python = True

    return_index = op.return_index
    return_inverse = op.return_inverse
    return_counts = op.return_counts

    returns_multi = return_index or return_inverse or return_counts
    use_python |= returns_multi

    if not use_python:

        @numba.njit
        def unique(x):
            return np.unique(x)

    else:

        warnings.warn(
            (
                "Numba will use object mode to allow the "
                "`axis` and/or `return_*` arguments to `numpy.unique`."
            ),
            UserWarning,
        )

        if returns_multi:
            ret_sig = numba.types.Tuple([get_numba_type(o.type) for o in node.outputs])
        else:
            ret_sig = get_numba_type(node.outputs[0].type)

        @numba.njit
        def unique(x):
            with numba.objmode(ret=ret_sig):
                ret = np.unique(x, return_index, return_inverse, return_counts, axis)
            return ret

    return unique


@numba_funcify.register(UnravelIndex)
def numba_funcify_UnravelIndex(op, node, **kwargs):
    order = op.order

    if order != "C":
        raise NotImplementedError(
            "Numba does not support the `order` argument in `numpy.unravel_index`"
        )

    if len(node.outputs) == 1:

        @numba.njit(inline="always")
        def maybe_expand_dim(arr):
            return arr

    else:

        @numba.njit(inline="always")
        def maybe_expand_dim(arr):
            return np.expand_dims(arr, 1)

    @numba.njit
    def unravelindex(arr, shape):
        a = np.ones(len(shape), dtype=np.int64)
        a[1:] = shape[:0:-1]
        a = np.cumprod(a)[::-1]

        # Aesara actually returns a `tuple` of these values, instead of an
        # `ndarray`; however, this `ndarray` result should be able to be
        # unpacked into a `tuple`, so this discrepancy shouldn't really matter
        return ((maybe_expand_dim(arr) // a) % shape).T

    return unravelindex


@numba_funcify.register(SearchsortedOp)
def numba_funcify_Searchsorted(op, node, **kwargs):
    side = op.side

    use_python = False
    if len(node.inputs) == 3:
        use_python = True

    if use_python:
        warnings.warn(
            (
                "Numba will use object mode to allow the "
                "`sorter` argument to `numpy.searchsorted`."
            ),
            UserWarning,
        )

        ret_sig = get_numba_type(node.outputs[0].type)

        @numba.njit
        def searchsorted(a, v, sorter):
            with numba.objmode(ret=ret_sig):
                ret = np.searchsorted(a, v, side, sorter)
            return ret

    else:

        @numba.njit
        def searchsorted(a, v):
            return np.searchsorted(a, v, side)

    return searchsorted


@numba_funcify.register(BroadcastTo)
def numba_funcify_BroadcastTo(op, node, **kwargs):
    warnings.warn(
        ("Numba will use object mode to allow the " "use of `numpy.broadcast_to`."),
        UserWarning,
    )

    ret_sig = get_numba_type(node.outputs[0].type)

    @numba.njit
    def broadcastto(x, *shape):
        with numba.objmode(ret=ret_sig):
            ret = np.broadcast_to(x, shape)
        return ret

    return broadcastto


def int_to_float_fn(inputs, out_dtype):
    """Create a Numba function that converts integer and boolean ``ndarray``s to floats."""

    if any(i.type.numpy_dtype.kind in "ib" for i in inputs):

        args_dtype = np.dtype(f"f{out_dtype.itemsize}")

        @numba.njit(inline="always")
        def inputs_cast(x):
            return x.astype(args_dtype)

    else:

        @numba.njit(inline="always")
        def inputs_cast(x):
            return x

    return inputs_cast


@numba_funcify.register(Dot)
def numba_funcify_Dot(op, node, **kwargs):
    # Numba's `np.dot` does not support integer dtypes, so we need to cast to
    # float.

    out_dtype = node.outputs[0].type.numpy_dtype
    inputs_cast = int_to_float_fn(node.inputs, out_dtype)

    @numba.njit
    def dot(x, y):
        return np.dot(inputs_cast(x), inputs_cast(y)).astype(out_dtype)

    return dot


@numba_funcify.register(Softmax)
def numba_funcify_Softmax(op, node, **kwargs):

    x_at = node.inputs[0]
    x_dtype = x_at.type.numpy_dtype
    x_dtype = numba.np.numpy_support.from_dtype(x_dtype)

    # np.max(x, axis=1)
    reduce_max = create_axis_reducer(np.maximum, -np.inf, 1, x_at.ndim, x_dtype)
    # np.sum(x, axis=1)
    reduce_sum = create_axis_reducer(np.add, 0.0, 1, x_at.ndim, x_dtype)

    @numba.njit
    def softmax(x):
        z = np.expand_dims(reduce_max(x), -1)
        e_x = np.exp(x - z)
        w = np.expand_dims(reduce_sum(e_x), -1)
        sm = e_x / w
        return sm

    return softmax


@numba_funcify.register(LogSoftmax)
def numba_funcify_LogSoftmax(op, node, **kwargs):

    x_at = node.inputs[0]
    x_dtype = x_at.type.numpy_dtype
    x_dtype = numba.np.numpy_support.from_dtype(x_dtype)

    # np.max(x, axis=1)
    reduce_max = create_axis_reducer(np.maximum, -np.inf, 1, x_at.ndim, x_dtype)
    # np.sum(x, axis=1, keepdims=True)
    reduce_sum = create_axis_reducer(np.add, 0.0, 1, x_at.ndim, x_dtype, keepdims=True)

    @numba.njit
    def log_softmax(x):
        xdev = x - np.expand_dims(reduce_max(x), -1)
        lsm = xdev - np.log(reduce_sum(np.exp(xdev)))
        return lsm

    return log_softmax


@numba_funcify.register(ScalarSoftplus)
def numba_funcify_ScalarSoftplus(op, node, **kwargs):

    not_int8 = node.inputs[0].dtype not in ("int8", "uint8")

    @numba.njit
    def scalarsoftplus(x):
        if x < -37.0:
            return np.exp(x) if not_int8 else np.exp(x)
        elif x < 18.0:
            return np.log1p(np.exp(x)) if not_int8 else np.log1p(np.exp(x))
        elif x < 33.3:
            return x + np.exp(-x) if not_int8 else x + np.exp(-x)
        else:
            return x

    return scalarsoftplus


def create_axis_apply_fn(fn, axis, ndim, dtype):
    reaxis_first = tuple(i for i in range(ndim) if i != axis) + (axis,)

    @numba.njit(boundscheck=False)
    def axis_apply_fn(x):
        x_reaxis = x.transpose(reaxis_first)

        res = np.zeros(x_reaxis.shape[:-1], dtype=dtype)
        for m in np.ndindex(res.shape):
            v = fn(x_reaxis[m])
            res[m] = v
        return res

    return axis_apply_fn


@numba_funcify.register(MaxAndArgmax)
def numba_funcify_MaxAndArgmax(op, node, **kwargs):
    axis = op.axis
    x_at = node.inputs[0]
    x_dtype = x_at.type.numpy_dtype
    x_dtype = numba.np.numpy_support.from_dtype(x_dtype)
    x_ndim = x_at.ndim

    if x_ndim == 0:

        @numba.njit(inline="always")
        def maxandargmax(x):
            return x, 0

    else:

        if axis is None:
            axes = tuple(range(x_ndim))
        else:
            axes = tuple(int(ax) for ax in axis)

        # NumPy does not support multiple axes for argmax; this is a
        # work-around
        keep_axes = tuple(i for i in range(x_ndim) if i not in axes)

        reduce_max = numba.njit(
            create_multiaxis_reducer(np.maximum, -np.inf, axes, x_ndim, x_dtype)
        )
        reduced_x_ndim = x_ndim - len(axes) + 1
        argmax_axis = create_axis_apply_fn(
            np.argmax, reduced_x_ndim - 1, reduced_x_ndim, np.int64
        )

        reaxis_order = keep_axes + axes
        sl1 = slice(None, len(keep_axes))
        sl2 = slice(len(keep_axes), None)

        @numba.njit
        def maxandargmax(x):
            max_res = reduce_max(x)

            # Not-reduced axes in front
            transposed_x = np.ascontiguousarray(np.transpose(x, reaxis_order))
            kept_shape = transposed_x.shape[sl1]
            reduced_shape = transposed_x.shape[sl2]
            reduced_size = 1
            for s in reduced_shape:
                reduced_size *= s

            # Numpy.prod returns 1.0 when arg is empty, so we cast it to int64
            # Otherwise reshape would complain citing float arg
            new_shape = kept_shape + (reduced_size,)
            reshaped_x = transposed_x.reshape(new_shape)

            max_idx_res = argmax_axis(reshaped_x)

            return max_res, max_idx_res

    return maxandargmax


@numba_funcify.register(Cholesky)
def numba_funcify_Cholesky(op, node, **kwargs):
    lower = op.lower

    out_dtype = node.outputs[0].type.numpy_dtype

    if lower:

        inputs_cast = int_to_float_fn(node.inputs, out_dtype)

        @numba.njit
        def cholesky(a):
            return np.linalg.cholesky(inputs_cast(a)).astype(out_dtype)

    else:
        # TODO: Use SciPy's BLAS/LAPACK Cython wrappers.

        warnings.warn(
            (
                "Numba will use object mode to allow the "
                "`lower` argument to `scipy.linalg.cholesky`."
            ),
            UserWarning,
        )

        ret_sig = get_numba_type(node.outputs[0].type)

        @numba.njit
        def cholesky(a):
            with numba.objmode(ret=ret_sig):
                ret = scipy.linalg.cholesky(a, lower=lower).astype(out_dtype)
            return ret

    return cholesky


@numba_funcify.register(Solve)
def numba_funcify_Solve(op, node, **kwargs):

    if op.A_structure == "lower_triangular" or op.A_structure == "upper_triangular":

        lower = op.A_structure == "lower_triangular"

        warnings.warn(
            (
                "Numba will use object mode to allow the "
                "`compute_uv` argument to `numpy.linalg.svd`."
            ),
            UserWarning,
        )

        ret_sig = get_numba_type(node.outputs[0].type)

        @numba.njit
        def solve(a, b):
            with numba.objmode(ret=ret_sig):
                ret = scipy.linalg.solve_triangular(a, b, lower=lower)
            return ret

    else:
        out_dtype = node.outputs[0].type.numpy_dtype
        inputs_cast = int_to_float_fn(node.inputs, out_dtype)

        @numba.njit
        def solve(a, b):
            return np.linalg.solve(inputs_cast(a), inputs_cast(b)).astype(out_dtype)

    return solve


@numba_funcify.register(Det)
def numba_funcify_Det(op, node, **kwargs):

    out_dtype = node.outputs[0].type.numpy_dtype
    inputs_cast = int_to_float_fn(node.inputs, out_dtype)

    @numba.njit
    def det(x):
        return direct_cast(np.linalg.det(inputs_cast(x)), out_dtype)

    return det


@numba_funcify.register(Eig)
def numba_funcify_Eig(op, node, **kwargs):

    out_dtype_1 = node.outputs[0].type.numpy_dtype
    out_dtype_2 = node.outputs[1].type.numpy_dtype

    inputs_cast = int_to_float_fn(node.inputs, out_dtype_1)

    @numba.njit
    def eig(x):
        out = np.linalg.eig(inputs_cast(x))
        return (out[0].astype(out_dtype_1), out[1].astype(out_dtype_2))

    return eig


@numba_funcify.register(Eigh)
def numba_funcify_Eigh(op, node, **kwargs):
    uplo = op.UPLO

    if uplo != "L":

        warnings.warn(
            (
                "Numba will use object mode to allow the "
                "`UPLO` argument to `numpy.linalg.eigh`."
            ),
            UserWarning,
        )

        out_dtypes = tuple(o.type.numpy_dtype for o in node.outputs)
        ret_sig = numba.types.Tuple(
            [get_numba_type(node.outputs[0].type), get_numba_type(node.outputs[1].type)]
        )

        @numba.njit
        def eigh(x):
            with numba.objmode(ret=ret_sig):
                out = np.linalg.eigh(x, UPLO=uplo)
                ret = (out[0].astype(out_dtypes[0]), out[1].astype(out_dtypes[1]))
            return ret

    else:

        @numba.njit
        def eigh(x):
            return np.linalg.eigh(x)

    return eigh


@numba_funcify.register(MatrixInverse)
def numba_funcify_MatrixInverse(op, node, **kwargs):

    out_dtype = node.outputs[0].type.numpy_dtype
    inputs_cast = int_to_float_fn(node.inputs, out_dtype)

    @numba.njit
    def matrix_inverse(x):
        return np.linalg.inv(inputs_cast(x)).astype(out_dtype)

    return matrix_inverse


@numba_funcify.register(QRFull)
def numba_funcify_QRFull(op, node, **kwargs):
    mode = op.mode

    if mode != "reduced":
        warnings.warn(
            (
                "Numba will use object mode to allow the "
                "`mode` argument to `numpy.linalg.qr`."
            ),
            UserWarning,
        )

        if len(node.outputs) > 1:
            ret_sig = numba.types.Tuple([get_numba_type(o.type) for o in node.outputs])
        else:
            ret_sig = get_numba_type(node.outputs[0].type)

        @numba.njit
        def qr_full(x):
            with numba.objmode(ret=ret_sig):
                ret = np.linalg.qr(x, mode=mode)
            return ret

    else:

        out_dtype = node.outputs[0].type.numpy_dtype
        inputs_cast = int_to_float_fn(node.inputs, out_dtype)

        @numba.njit
        def qr_full(x):
            res = np.linalg.qr(inputs_cast(x))
            return res

    return qr_full


@numba_funcify.register(SVD)
def numba_funcify_SVD(op, node, **kwargs):
    full_matrices = op.full_matrices
    compute_uv = op.compute_uv

    if not compute_uv:

        warnings.warn(
            (
                "Numba will use object mode to allow the "
                "`compute_uv` argument to `numpy.linalg.svd`."
            ),
            UserWarning,
        )

        if len(node.outputs) > 1:
            ret_sig = numba.types.Tuple([get_numba_type(o.type) for o in node.outputs])
        else:
            ret_sig = get_numba_type(node.outputs[0].type)

        @numba.njit
        def svd(x):
            with numba.objmode(ret=ret_sig):
                ret = np.linalg.svd(x, full_matrices, compute_uv)
            return ret

    else:

        out_dtype = node.outputs[0].type.numpy_dtype
        inputs_cast = int_to_float_fn(node.inputs, out_dtype)

        @numba.njit
        def svd(x):
            return np.linalg.svd(inputs_cast(x), full_matrices)

    return svd
