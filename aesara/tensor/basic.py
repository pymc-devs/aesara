"""`Op` classes for working with ``numpy.ndarrays`` symbolically.

This module primarily defines `Op`s for the creation, conversion, and
manipulation of tensors.

"""

import builtins
import logging
import warnings
from collections import OrderedDict
from collections.abc import Sequence
from numbers import Number
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from typing_extensions import Protocol

import aesara
import aesara.scalar.sharedvar
from aesara import compile, config, printing
from aesara import scalar as aes
from aesara.assert_op import Assert, assert_op
from aesara.gradient import DisconnectedType, grad_not_implemented, grad_undefined
from aesara.graph.basic import Apply, Constant, Variable
from aesara.graph.op import COp, Op
from aesara.graph.params_type import ParamsType
from aesara.graph.type import CType
from aesara.misc.safe_asarray import _asarray
from aesara.printing import min_informative_str, pprint
from aesara.scalar import int32
from aesara.scalar.basic import ScalarConstant, ScalarVariable
from aesara.tensor import _as_tensor_variable, as_tensor_variable
from aesara.tensor.elemwise import DimShuffle, Elemwise, scalar_elemwise
from aesara.tensor.exceptions import EmptyConstantError, NotScalarConstantError
from aesara.tensor.shape import (
    Shape,
    Shape_i,
    reshape,
    shape,
    shape_padaxis,
    shape_padleft,
)
from aesara.tensor.type import (
    TensorType,
    discrete_dtypes,
    float_dtypes,
    int_dtypes,
    int_types,
    int_vector_types,
    integer_dtypes,
    tensor,
    uint_dtypes,
    values_eq_approx_always_true,
)
from aesara.tensor.var import TensorConstant, TensorVariable


_logger = logging.getLogger("aesara.tensor.basic")

__docformat__ = "restructuredtext en"


def check_equal_numpy(x: np.array, y: np.array) -> bool:
    """
    Return True iff x and y are equal.

    Checks the dtype and shape if x and y are numpy.ndarray instances.

    """
    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        return x.dtype == y.dtype and x.shape == y.shape and np.all(abs(x - y) < 1e-10)
    elif isinstance(x, np.random.RandomState) and isinstance(y, np.random.RandomState):
        return builtins.all(
            np.all(a == b) for a, b in zip(x.__getstate__(), y.__getstate__())
        )
    else:
        return x == y


compile.register_checker(check_equal_numpy)


def __oplist_tag(thing, tag):
    tags = getattr(thing, "__oplist_tags", [])
    tags.append(tag)
    thing.__oplist_tags = tags


@_as_tensor_variable.register(Apply)
def _as_tensor_Apply(
    x: Union[int, List, Apply, Variable], name: str, ndim: int, **kwargs
) -> Callable:
    # use Apply's default output mechanism
    if (x.op.default_output is None) and (len(x.outputs) != 1):
        raise TypeError(
            "Multi-output Op encountered. "
            "Retry using only one of the outputs directly."
        )

    x = x.default_output()

    return as_tensor_variable(x, name=name, ndim=ndim, **kwargs)


@_as_tensor_variable.register(ScalarVariable)
@_as_tensor_variable.register(ScalarConstant)
def _as_tensor_Scalar(x, name, ndim, **kwargs):
    return as_tensor_variable(tensor_from_scalar(x), name=name, ndim=ndim, **kwargs)


@_as_tensor_variable.register(Variable)
def _as_tensor_Variable(x, name, ndim, **kwargs):
    if not isinstance(x.type, TensorType):
        raise TypeError(
            "Tensor type field must be a TensorType; found {}.".format(type(x.type))
        )

    if ndim is None:
        return x

    if x.type.ndim > ndim:
        # strip off leading broadcastable dimensions
        first_non_broadcastable = [
            idx for idx in range(x.ndim) if not x.broadcastable[idx]
        ][0]
        x = x.dimshuffle(list(range(x.ndim))[first_non_broadcastable:])
        if x.ndim > ndim:
            raise ValueError(
                "Tensor of type {} could not be cast to have {} dimensions".format(
                    x.type, ndim
                )
            )
        return x
    elif x.type.ndim < ndim:
        return shape_padleft(x, n_ones=(ndim - x.type.ndim))
    else:
        return x


@_as_tensor_variable.register(list)
@_as_tensor_variable.register(tuple)
def _as_tensor_Sequence(x, name, ndim, dtype=None, **kwargs):

    if len(x) == 0:
        return constant(x, name=name, ndim=ndim, dtype=dtype)

    # If a sequence has `Variable`s in it, then we want
    # to customize the conversion to a tensor type.
    def extract_constants(i):
        if isinstance(i, Variable):
            if isinstance(i, Constant):
                return i.data
            else:
                raise TypeError
        else:
            return i

    try:
        x = type(x)(extract_constants(i) for i in x)
    except TypeError:
        if builtins.all(getattr(i, "ndim", None) == 0 for i in x) and (
            ndim is None or ndim == 1
        ):
            # In this instance, we have a sequence of constants with which we
            # want to construct a vector, so we can use `MakeVector` directly.
            if dtype is None:
                dtype = aes.upcast(*[i.dtype for i in x if hasattr(i, "dtype")])
            return MakeVector(dtype)(*x)

        # In this case, we have at least one non-`Constant` term, so we
        # couldn't get an underlying non-symbolic sequence of objects and we to
        # symbolically join terms.
        return stack(x)

    return constant(x, name=name, ndim=ndim, dtype=dtype)


@_as_tensor_variable.register(np.bool_)
@_as_tensor_variable.register(np.number)
@_as_tensor_variable.register(Number)
@_as_tensor_variable.register(np.ndarray)
def _as_tensor_numbers(x, name, ndim, dtype=None, **kwargs):
    return constant(x, name=name, ndim=ndim, dtype=dtype)


@_as_tensor_variable.register(bool)
def _as_tensor_bool(x, name, ndim, **kwargs):
    raise TypeError(
        "Cannot cast True or False as a tensor variable. Please use "
        "np.array(True) or np.array(False) if you need these constants. "
        "This error might be caused by using the == operator on "
        "Variables. v == w does not do what you think it does, "
        "use aesara.tensor.eq(v, w) instead."
    )


as_tensor = as_tensor_variable


def constant(
    x, name: Optional[str] = None, ndim: Optional[int] = None, dtype=None
) -> TensorConstant:
    """Return a `TensorConstant` with value `x`.

    Raises
    ------
    TypeError
        `x` could not be converted to a numpy.ndarray.
    ValueError
        `x` could not be expanded to have ndim dimensions.

    """
    if isinstance(x, TensorConstant):
        if (
            (name is None or x.name == name)
            and (ndim is None or x.ndim == ndim)
            and (dtype is None or x.dtype == dtype)
        ):
            return x
        else:
            x = x.data

    x_ = aes.convert(x, dtype=dtype)

    if ndim is not None:
        if x_.ndim < ndim:
            x_ = np.expand_dims(x_, axis=tuple(range(ndim - x_.ndim)))
        elif x_.ndim > ndim:
            try:
                x_ = np.squeeze(x_, axis=tuple(range(x_.ndim - ndim)))
            except np.AxisError:
                raise ValueError(
                    f"ndarray could not be cast to constant with {int(ndim)} dimensions"
                )

        assert x_.ndim == ndim

    ttype = TensorType(dtype=x_.dtype, broadcastable=[s == 1 for s in x_.shape])

    try:
        return TensorConstant(ttype, x_, name=name)
    except Exception:
        raise TypeError(f"Could not convert {x} to TensorType", type(x))


def _obj_is_wrappable_as_tensor(x) -> bool:
    try:
        constant(x)
        return True
    except TypeError:
        return False


def numpy_scalar(data) -> np.array:
    """Return a scalar stored in a numpy ndarray.

    Raises
    ------
    NotScalarConstantError
        If the numpy ndarray is not a scalar.
    EmptyConstantError

    """

    # handle case where data is numpy.array([])
    if data.ndim > 0 and (len(data.shape) == 0 or builtins.max(data.shape) == 0):
        assert np.all(np.array([]) == data)
        raise EmptyConstantError()
    try:
        complex(data)  # works for all numeric scalars
        return data
    except Exception:
        raise NotScalarConstantError(
            "v.data is non-numeric, non-scalar, or has more than one" " unique value",
            data,
        )


_scalar_constant_value_elemwise_ops = (
    aes.Cast,
    aes.Switch,
    aes.NEQ,
    aes.EQ,
    aes.LT,
    aes.GT,
    aes.LE,
    aes.GE,
    aes.Sub,
    aes.Add,
    aes.Mod,
    aes.Mul,
    aes.IntDiv,
    aes.TrueDiv,
    aes.ScalarMinimum,
    aes.ScalarMaximum,
)


def get_scalar_constant_value(
    orig_v: Optional[Union[int, float, np.ndarray, Constant]],
    elemwise: bool = True,
    only_process_constants: bool = False,
    max_recur: int = 10,
):
    """Return the constant scalar(0-D) value underlying variable `v`.

    If `v` is the output of dimshuffles, fills, allocs, rebroadcasts,
    cast, OutputGuard, DeepCopyOp, ScalarFromTensor, ScalarOp, Elemwise
    and some pattern with Subtensor, this function digs through them.

    If `v` is not some view of constant scalar data, then raise a
    NotScalarConstantError.

    Parameters
    ----------
    orig_v : int, float, np.ndarray, Constant or None
        The variable `v` where to find the constant scalar(0-D).
    elemwise : bool
        If False, we won't try to go into elemwise. So this call is faster.
        But we still investigate in Second Elemwise (as this is a substitute
        for Alloc)
    only_process_constants : bool
        If True, we only attempt to obtain the value of `orig_v` if it's
        directly constant and don't try to dig through dimshuffles, fills,
        allocs, and other to figure out its value.
    max_recur : int
        The maximum number of recursion.

    Notes
    -----
        There may be another function similar to this one in the code,
        but I'm not sure where it is.

    """
    v = orig_v
    while True:
        if v is None:
            # None is not a scalar (and many uses of this function seem
            # to depend on passing it None)
            raise NotScalarConstantError()

        if isinstance(v, (np.integer, int, float)):
            return np.asarray(v)

        if isinstance(v, np.ndarray):
            return numpy_scalar(v).copy()

        if isinstance(v, Constant):
            if getattr(v.tag, "unique_value", None) is not None:
                data = v.tag.unique_value
            else:
                data = v.data

            if isinstance(data, np.ndarray):
                return numpy_scalar(data).copy()
            else:
                return data

        if not only_process_constants and getattr(v, "owner", None) and max_recur > 0:
            max_recur -= 1
            if isinstance(
                v.owner.op,
                (
                    Alloc,
                    DimShuffle,
                    Rebroadcast,
                    # outputguard is only used in debugmode but we
                    # keep it here to avoid problems with old pickels.
                    compile.ops.OutputGuard,
                    compile.DeepCopyOp,
                ),
            ):
                v = v.owner.inputs[0]
                continue
            elif isinstance(v.owner.op, Shape_i):
                i = v.owner.op.i
                inp = v.owner.inputs[0]
                if isinstance(inp, Constant):
                    return np.asarray(np.shape(inp.data)[i])
                # The shape of a broadcastable dimension is 1
                if hasattr(inp.type, "broadcastable") and inp.type.broadcastable[i]:
                    return np.asarray(1)

            # Don't act as the constant_folding optimization here as this
            # fct is used too early in the optimization phase.  This would
            # mess with the stabilization optimization and be too slow.
            # We put all the scalar Ops used by get_canonical_form_slice()
            # to allow it to determine the broadcast pattern correctly.
            elif isinstance(v.owner.op, (ScalarFromTensor, TensorFromScalar)):
                v = v.owner.inputs[0]
                continue
            elif isinstance(v.owner.op, Assert):
                # check if all conditions are constant and true
                cond = [
                    get_scalar_constant_value(c, max_recur=max_recur)
                    for c in v.owner.inputs[1:]
                ]
                if builtins.all([0 == c.ndim and c != 0 for c in cond]):
                    v = v.owner.inputs[0]
                    continue
            elif isinstance(v.owner.op, aes.ScalarOp):
                if isinstance(v.owner.op, aes.Second):
                    # We don't need both input to be constant for second
                    shp, val = v.owner.inputs
                    v = val
                    continue
                if isinstance(v.owner.op, _scalar_constant_value_elemwise_ops):
                    const = [
                        get_scalar_constant_value(i, max_recur=max_recur)
                        for i in v.owner.inputs
                    ]
                    ret = [[None]]
                    v.owner.op.perform(v.owner, const, ret)
                    return ret[0][0].copy()
            # In fast_compile, we don't enable local_fill_to_alloc, so
            # we need to investigate Second as Alloc. So elemwise
            # don't disable the check for Second.
            elif isinstance(v.owner.op, Elemwise):
                if isinstance(v.owner.op.scalar_op, aes.Second):
                    # We don't need both input to be constant for second
                    shp, val = v.owner.inputs
                    v = val
                    continue
                elif elemwise and isinstance(
                    v.owner.op.scalar_op, _scalar_constant_value_elemwise_ops
                ):
                    const = [
                        get_scalar_constant_value(i, max_recur=max_recur)
                        for i in v.owner.inputs
                    ]
                    ret = [[None]]
                    v.owner.op.perform(v.owner, const, ret)
                    return ret[0][0].copy()
            elif (
                isinstance(v.owner.op, aesara.tensor.subtensor.Subtensor)
                and v.ndim == 0
            ):
                if isinstance(v.owner.inputs[0], TensorConstant):
                    cdata = tuple(v.owner.op.get_constant_idx(v.owner.inputs))
                    try:
                        return v.owner.inputs[0].data.__getitem__(cdata).copy()
                    except IndexError:
                        raise IndexError(
                            str(tuple(v.owner.op.idx_list))
                            + " is not a valid index into "
                            + str(v.owner.inputs[0].data)
                        )

                # The index list 'idx_list' should have length the same
                # shape as the input.
                # TODO: implement the case where we take a scalar in a matrix
                assert len(v.owner.op.idx_list) == v.owner.inputs[0].ndim

                # Needed to make better graph in this test in
                # aesara/tensor/tests/test_sharedvar.py:
                # test_shared_options.test_specify_shape_partial
                if (
                    v.owner.inputs[0].owner
                    and isinstance(v.owner.inputs[0].owner.op, Join)
                    and len(v.owner.op.idx_list) == 1
                ):
                    # Ensure the Join is joining only scalar variables (so that
                    # the constant value can be found at the same index as the
                    # one used in the sub-tensor).
                    if builtins.all(
                        var.ndim == 0 for var in v.owner.inputs[0].owner.inputs[1:]
                    ):
                        idx = v.owner.op.idx_list[0]
                        if isinstance(idx, CType):
                            idx = get_scalar_constant_value(
                                v.owner.inputs[1], max_recur=max_recur
                            )
                        # Note the '+ 1' is because the first argument to Join
                        # is the axis.
                        ret = v.owner.inputs[0].owner.inputs[idx + 1]
                        ret = get_scalar_constant_value(ret, max_recur=max_recur)
                        # join can cast implicitly its input in some case.
                        return _asarray(ret, dtype=v.type.dtype)
                    if builtins.all(
                        var.ndim == 1 for var in v.owner.inputs[0].owner.inputs[1:]
                    ):
                        idx = v.owner.op.idx_list[0]
                        if isinstance(idx, CType):
                            idx = get_scalar_constant_value(
                                v.owner.inputs[1], max_recur=max_recur
                            )
                        try:
                            # TODO: assert joined axis is 0.
                            length = 0
                            loop = False
                            for joined in v.owner.inputs[0].owner.inputs[1:]:
                                ll = get_vector_length(joined)
                                if idx < length + ll:
                                    v = joined[idx - length]
                                    loop = True
                                    break
                                length += ll
                            if loop:
                                continue
                        except TypeError:
                            pass
                        except ValueError:
                            pass

                elif (
                    v.owner.inputs[0].owner
                    and isinstance(v.owner.inputs[0].owner.op, MakeVector)
                    and
                    # MakeVector normally accept only scalar as input.
                    # We put this check in case there is change in the future
                    builtins.all(
                        var.ndim == 0 for var in v.owner.inputs[0].owner.inputs
                    )
                    and len(v.owner.op.idx_list) == 1
                ):

                    idx = v.owner.op.idx_list[0]
                    if isinstance(idx, CType):
                        idx = get_scalar_constant_value(
                            v.owner.inputs[1], max_recur=max_recur
                        )
                    # Python 2.4 does not support indexing with numpy.integer
                    # So we cast it.
                    idx = int(idx)
                    ret = v.owner.inputs[0].owner.inputs[idx]
                    ret = get_scalar_constant_value(ret, max_recur=max_recur)
                    # MakeVector can cast implicitly its input in some case.
                    return _asarray(ret, dtype=v.type.dtype)

                # This is needed when we take the grad as the Shape op
                # are not already changed into MakeVector
                owner = v.owner
                leftmost_parent = owner.inputs[0]
                if leftmost_parent.owner and isinstance(
                    leftmost_parent.owner.op, Shape
                ):
                    op = owner.op
                    idx_list = op.idx_list
                    idx = idx_list[0]
                    if isinstance(idx, CType):
                        idx = get_scalar_constant_value(
                            owner.inputs[1], max_recur=max_recur
                        )
                    grandparent = leftmost_parent.owner.inputs[0]
                    gp_broadcastable = grandparent.type.broadcastable
                    ndim = grandparent.type.ndim
                    if grandparent.owner and isinstance(
                        grandparent.owner.op, Rebroadcast
                    ):
                        ggp_broadcastable = grandparent.owner.inputs[0].broadcastable
                        l = [
                            b1 or b2
                            for b1, b2 in zip(ggp_broadcastable, gp_broadcastable)
                        ]
                        gp_broadcastable = tuple(l)

                    assert ndim == len(gp_broadcastable)

                    if not (idx < len(gp_broadcastable)):
                        msg = (
                            "get_scalar_constant_value detected "
                            f"deterministic IndexError: x.shape[{int(idx)}] "
                            f"when x.ndim={int(ndim)}."
                        )
                        if config.exception_verbosity == "high":
                            msg += f" x={min_informative_str(v)}"
                        else:
                            msg += f" x={v}"
                        raise ValueError(msg)

                    if gp_broadcastable[idx]:
                        return np.asarray(1)

                    if isinstance(grandparent, Constant):
                        return np.asarray(np.shape(grandparent.data)[idx])

        raise NotScalarConstantError(v)


#########################
# Casting Operations
#########################


class TensorFromScalar(Op):

    __props__ = ()

    def make_node(self, s):
        assert isinstance(s.type, aes.Scalar)
        return Apply(self, [s], [tensor(dtype=s.type.dtype, broadcastable=())])

    def perform(self, node, inp, out_):
        (s,) = inp
        (out,) = out_
        out[0] = np.asarray(s)

    def infer_shape(self, fgraph, node, in_shapes):
        return [()]

    def grad(self, inp, grads):
        (s,) = inp
        (dt,) = grads
        if s.type.dtype in float_dtypes:
            assert dt.type.dtype in float_dtypes
            return [scalar_from_tensor(dt)]

        # If the input dtype is an integer, then so is the output dtype,
        # and the "zero" gradient can be represented in that int dtype.
        # Currently, aesara.grad insists that the dtype of the returned
        # gradient has a float dtype, so we use floatX.
        if s.type.dtype in discrete_dtypes:
            return [s.zeros_like().astype(config.floatX)]

        raise NotImplementedError("grad not implemented for complex dtypes")


tensor_from_scalar = TensorFromScalar()


class ScalarFromTensor(COp):

    __props__ = ()

    def make_node(self, t):
        assert isinstance(t.type, TensorType)
        assert t.type.broadcastable == ()
        return Apply(
            self, [t], [aes.get_scalar_type(dtype=t.type.dtype).make_variable()]
        )

    def perform(self, node, inp, out_):
        (s,) = inp
        (out,) = out_
        out[0] = s.flatten()[0]

    def infer_shape(self, fgraph, node, in_shapes):
        return [()]

    def grad(self, inp, grads):
        (s,) = inp
        (dt,) = grads
        return [tensor_from_scalar(dt)]

    def R_op(self, inputs, eval_points):
        if None in eval_points:
            return [None]
        return self.make_node(*eval_points).outputs

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        fail = sub["fail"]
        return (
            """
        %(z)s = ((dtype_%(x)s*)(PyArray_DATA(%(x)s)))[0];
        """
            % locals()
        )

    def c_code_cache_version(self):
        return (1,)


scalar_from_tensor = ScalarFromTensor()


class Rebroadcast(COp):
    """
    Change the input's broadcastable fields in some predetermined way.

    See Also
    --------
    unbroadcast <aesara.tensor.unbroadcast>
    addbroadcast <aesara.tensor.addbroadcast>
    patternbroadcast <aesara.tensor.patternbroadcast>

    Notes
    -----
    Works inplace and works for CudaNdarrayType.

    Examples
    --------
    `Rebroadcast((0, True), (1, False))(x)` would make `x` broadcastable in
    axis 0 and not broadcastable in axis 1.

    """

    view_map = {0: [0]}
    _f16_ok = True
    # Mapping from Type to C code (and version) to use.
    # In the C code, the name of the input variable is %(iname)s,
    # the output variable is %(oname)s.
    c_code_and_version: Dict = {}

    check_input = False
    __props__ = ("axis",)
    _f16_ok = True

    def __init__(self, *axis):
        # Sort them to make sure we merge all possible case.
        items = sorted(axis)
        self.axis = OrderedDict(items)
        for axis, broad in self.axis.items():
            if not isinstance(axis, (np.integer, int)):
                raise TypeError(f"Rebroadcast needs integer axes. Got {axis}")

            if not isinstance(broad, (np.bool_, bool)):
                raise TypeError(
                    f"Rebroadcast needs bool for new broadcast pattern. Got {broad}"
                )

    def __hash__(self):
        # Need special __hash__ as dict aren't hashable.
        # no ambiguity because each item key is unique
        items = sorted(self.axis.items())
        return hash((type(self), tuple(items)))

    def __str__(self):
        if len(self.axis) == 0:
            broadcast_pattern = []
        else:
            broadcast_pattern = ["?" for i in range(1 + max(self.axis.keys()))]
        for k, v in self.axis.items():
            broadcast_pattern[k] = str(int(v))
        return f"{self.__class__.__name__}{{{','.join(broadcast_pattern)}}}"

    def make_node(self, x):
        if self.axis.keys() and (x.ndim <= max(self.axis.keys())):
            raise ValueError("Trying to rebroadcast non-existent dimension")
        t = x.type.clone(
            broadcastable=[
                self.axis.get(i, b) for i, b in enumerate(x.type.broadcastable)
            ]
        )
        return Apply(self, [x], [t()])

    def perform(self, node, inp, out_):
        (x,) = inp
        (out,) = out_
        for axis, value in self.axis.items():
            if value and x.shape[axis] != 1:
                raise ValueError(
                    f"Dimension {axis} in Rebroadcast's input was"
                    f" supposed to be 1 (got {x.shape[axis]} instead)"
                )
        out[0] = x

    def grad(self, inp, grads):
        (x,) = inp
        (gz,) = grads
        # restore the broadcasting pattern of the input
        return (
            Rebroadcast(
                *[
                    (axis, x.type.broadcastable[axis])
                    for axis, value in self.axis.items()
                ]
            )(gz),
        )

    def infer_shape(self, fgraph, node, ishapes):
        assert len(ishapes) == 1
        l = []
        one = aesara.tensor.basic.constant(1)
        for ax in range(len(ishapes[0])):
            if self.axis.get(ax, False):
                l.append(one)
            else:
                l.append(ishapes[0][ax])

        return [tuple(l)]

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return [None]
        return self(*eval_points, **dict(return_list=True))

    def c_code(self, node, nodename, inp, out, sub):
        (iname,) = inp
        (oname,) = out
        fail = sub["fail"]

        itype = node.inputs[0].type.__class__
        if itype in self.c_code_and_version:
            code, version = self.c_code_and_version[itype]
            final_code = ""
            for axis, value in self.axis.items():
                if value:
                    final_code += code % locals()
            return (
                final_code
                + f"""
            Py_XDECREF({oname});
            {oname} = {iname};
            Py_XINCREF({oname});
            """
            )
        raise NotImplementedError()

    def c_code_cache_version(self):
        version = []
        # If any of the c code is unversionned, we have to return ()
        # Else, we will return a list of (type name, version) pairs.
        for t, (c, v) in sorted(
            self.c_code_and_version.items(), key=lambda pair: str(pair[0])
        ):
            if not v:
                warnings.warn(
                    f"Type {t} has C code for Rebroadcast, but it "
                    "has no version. You should add a 'version' "
                    "keyword arg when calling "
                    "register_rebroadcast_c_code.",
                    stacklevel=2,
                )
                return ()
            version.append((str(t), v))

        if version:
            version.append(1)
        return tuple(version)


def register_rebroadcast_c_code(typ, code, version=()):
    """
    Tell Rebroadcast how to generate C code for an Aesara Type.

    typ : Aesara type
        It must be the Aesara class itself and not an instance of the class.
    code : C code
        That checks if the dimension %(axis)s is of shape 1 for the Aesara type
        'typ'. Use %(iname)s and %(oname)s for the input and output C variable
        names respectively, and %(axis)s for the axis that we need to check.
        This code is put in a loop for all axes.
    version
        A number indicating the version of the code, for cache.

    """
    Rebroadcast.c_code_and_version[typ] = (code, version)


register_rebroadcast_c_code(
    TensorType,
    """
    if(PyArray_DIMS(%(iname)s)[%(axis)s] != 1){
        PyErr_Format(PyExc_ValueError,
            "Dimension %(axis)s in Rebroadcast's input was"
            " supposed to be 1 (got %%d instead)",
            PyArray_DIMS(%(iname)s)[%(axis)s]);
        %(fail)s
    }
    """,
    version=1,
)


# to be removed as we get the epydoc routine-documenting thing going
# -JB 20080924
def _conversion(real_value, name):
    __oplist_tag(real_value, "casting")
    real_value.__module__ = "tensor.basic"
    pprint.assign(real_value, printing.FunctionPrinter(name))
    return real_value


# These _conver_to_<type> functions have leading underscores to indicate that
# they should not be called directly.  They do not perform sanity checks about
# what types you are casting to what.  That logic is implemented by the
# `cast()` function below.

_convert_to_bool = _conversion(Elemwise(aes.convert_to_bool), "bool")
"""Cast to boolean"""

_convert_to_int8 = _conversion(Elemwise(aes.convert_to_int8), "int8")
"""Cast to 8-bit integer"""

_convert_to_int16 = _conversion(Elemwise(aes.convert_to_int16), "int16")
"""Cast to 16-bit integer"""

_convert_to_int32 = _conversion(Elemwise(aes.convert_to_int32), "int32")
"""Cast to 32-bit integer"""

_convert_to_int64 = _conversion(Elemwise(aes.convert_to_int64), "int64")
"""Cast to 64-bit integer"""

_convert_to_uint8 = _conversion(Elemwise(aes.convert_to_uint8), "uint8")
"""Cast to unsigned 8-bit integer"""

_convert_to_uint16 = _conversion(Elemwise(aes.convert_to_uint16), "uint16")
"""Cast to unsigned 16-bit integer"""

_convert_to_uint32 = _conversion(Elemwise(aes.convert_to_uint32), "uint32")
"""Cast to unsigned 32-bit integer"""

_convert_to_uint64 = _conversion(Elemwise(aes.convert_to_uint64), "uint64")
"""Cast to unsigned 64-bit integer"""

_convert_to_float16 = _conversion(Elemwise(aes.convert_to_float16), "float16")
"""Cast to half-precision floating point"""

_convert_to_float32 = _conversion(Elemwise(aes.convert_to_float32), "float32")
"""Cast to single-precision floating point"""

_convert_to_float64 = _conversion(Elemwise(aes.convert_to_float64), "float64")
"""Cast to double-precision floating point"""

_convert_to_complex64 = _conversion(Elemwise(aes.convert_to_complex64), "complex64")
"""Cast to single-precision complex"""

_convert_to_complex128 = _conversion(Elemwise(aes.convert_to_complex128), "complex128")
"""Cast to double-precision complex"""

_cast_mapping = {
    "bool": _convert_to_bool,
    "int8": _convert_to_int8,
    "int16": _convert_to_int16,
    "int32": _convert_to_int32,
    "int64": _convert_to_int64,
    "uint8": _convert_to_uint8,
    "uint16": _convert_to_uint16,
    "uint32": _convert_to_uint32,
    "uint64": _convert_to_uint64,
    "float16": _convert_to_float16,
    "float32": _convert_to_float32,
    "float64": _convert_to_float64,
    "complex64": _convert_to_complex64,
    "complex128": _convert_to_complex128,
}


def cast(x, dtype):
    """Symbolically cast `x` to a Tensor of type `dtype`."""
    if dtype == "floatX":
        dtype = config.floatX

    _x = as_tensor_variable(x)
    if _x.type.dtype == dtype:
        return _x
    if _x.type.dtype.startswith("complex") and not dtype.startswith("complex"):
        raise TypeError(
            "Casting from complex to real is ambiguous: consider real(), "
            "imag(), angle() or abs()"
        )
    return _cast_mapping[dtype](x)


##########################
# Condition
##########################


@scalar_elemwise
def switch(cond, ift, iff):
    """if cond then ift else iff"""


where = switch
##########################
# Misc
##########################


# fill, _fill_inplace = _elemwise(aes.second, 'fill',
# """fill WRITEME (elemwise)""")
@scalar_elemwise
def second(a, b):
    """Create a matrix by filling the shape of a with b"""


fill = second
pprint.assign(fill, printing.FunctionPrinter("fill"))


def ones_like(model, dtype=None, opt: bool = False):
    """equivalent of numpy.ones_like
    Parameters
    ----------
    model : tensor
    dtype : data-type, optional
    opt : bool
        If True, we will return a constant instead of a graph when possible.
        Useful for Aesara optimization, not for user building a graph as this
        have the consequence that model isn't always in the graph.

    Returns
    -------
    tensor
        tensor the shape of model containing ones of the type of dtype.
    """
    if dtype is None:
        dtype = model.type.dtype
    ret = constant(1.0, dtype=dtype)
    if opt and ret.type == model.type:
        return ret
    return fill(model, ret)


def zeros_like(model, dtype=None, opt: bool = False):
    """equivalent of numpy.zeros_like
    Parameters
    ----------
    model : tensor
    dtype : data-type, optional
    opt : bool
        If True, we will return a constant instead of a graph when possible.
        Useful for Aesara optimization, not for user building a graph as this
        have the consequence that model isn't always in the graph.

    Returns
    -------
    tensor
        tensor the shape of model containing zeros of the type of dtype.
    """

    if dtype is None:
        dtype = model.type.dtype
    ret = constant(0.0, dtype=dtype)
    if opt and ret.type == model.type:
        return ret
    return fill(model, ret)


def zeros(shape: Union[np.ndarray, Sequence, TensorVariable], dtype=None):
    """Create a `TensorVariable` filled with zeros, closer to NumPy's syntax
    than ``alloc``.

    """
    if not isinstance(shape, (np.ndarray, Sequence, TensorVariable)):
        shape = [shape]
    if dtype is None:
        dtype = config.floatX
    return alloc(np.array(0, dtype=dtype), *shape)


def ones(
    shape: Union[np.ndarray, Sequence, TensorVariable, List],
    dtype: Optional[type] = None,
):
    """Create a `TensorVariable` filled with ones,
    closer to NumPy's syntax than ``alloc``."""
    if not isinstance(shape, (np.ndarray, Sequence, TensorVariable)):
        shape = [shape]
    if dtype is None:
        dtype = config.floatX
    return alloc(np.array(1, dtype=dtype), *shape)


class Nonzero(Op):
    """
    Return the indices of the elements that are non-zero.

    Parameters
    ----------
    a: array_like
        Input array.

    Returns
    -------
    indices: list
        A list containing the indices of the non-zero elements of `a`.

    See Also
    --------
    nonzero_values : Return the non-zero elements of the input array
    flatnonzero : Return the indices of the non-zero elements of the
        flattened input array.

    """

    __props__ = ()

    def make_node(self, a):
        a = as_tensor_variable(a)
        if a.ndim == 0:
            raise ValueError("Nonzero only supports non-scalar arrays.")
        output = [
            TensorType(dtype="int64", broadcastable=(False,))() for i in range(a.ndim)
        ]
        return Apply(self, [a], output)

    def perform(self, node, inp, out_):
        a = inp[0]

        result_tuple = np.nonzero(a)
        for i, res in enumerate(result_tuple):
            out_[i][0] = res.astype("int64")

    def grad(self, inp, grads):
        return [grad_undefined(self, 0, inp[0])]


_nonzero = Nonzero()


def nonzero(a: np.array, return_matrix: bool = False) -> Tuple:
    """
    Returns one of the following:

        If return_matrix is False (default, same as NumPy):
            A tuple of vector arrays such that the ith element of the jth array
            is the index of the ith non-zero element of the input array in the
            jth dimension.

        If return_matrix is True (same as Aesara Op):
            Returns a matrix of shape (ndim, number of nonzero elements) such
            that element (i,j) is the index in the ith dimension of the jth
            non-zero element.

    Parameters
    ----------
    a : array_like
        Input array.
    return_matrix : bool
        If True, returns a symbolic matrix. If False, returns a tuple of
        arrays. Defaults to False.

    Returns
    -------
    tuple of vectors or matrix

    See Also
    --------
    nonzero_values : Return the non-zero elements of the input array
    flatnonzero : Return the indices of the non-zero elements of the
        flattened input array.

    """
    res = _nonzero(a)
    if isinstance(res, list):
        res = tuple(res)
    else:
        res = (res,)

    if return_matrix:
        if len(res) > 1:
            return stack(res, 0)
        elif len(res) == 1:
            return shape_padleft(res[0])
    else:
        return res


def flatnonzero(a):
    """Return a vector of indices that are non-zero in the flattened version of `a`.

    Parameters
    ----------
    a : tensor
        Input tensor

    Returns
    -------
    vector
        Output vector, containing the indices of the elements of `a.flatten()`
        that are non-zero.

    See Also
    --------
    nonzero : Return the indices of the non-zero elements of the input array.
    nonzero_values : Return the non-zero elements of the input array

    """
    if a.ndim == 0:
        raise ValueError("Nonzero only supports non-scalar arrays.")
    return nonzero(a.flatten(), return_matrix=False)[0]


def nonzero_values(a):
    """Return a vector of non-zero elements contained in the input array.

    Parameters
    ----------
    a : tensor
        Input tensor

    Returns
    -------
    vector
        Output vector, containing the non-zero elements of a.

    See Also
    --------
    nonzero : Return the indices of the non-zero elements of the input array.
    flatnonzero : Return the indices of the non-zero elements of the
        flattened input array.

    """
    return a.flatten()[flatnonzero(a)]


class Tri(Op):

    __props__ = ("dtype",)

    def __init__(self, dtype=None):
        if dtype is None:
            dtype = config.floatX
        self.dtype = dtype

    def make_node(self, N, M, k):
        N = as_tensor_variable(N)
        M = as_tensor_variable(M)
        k = as_tensor_variable(k)
        return Apply(
            self,
            [N, M, k],
            [TensorType(dtype=self.dtype, broadcastable=(False, False))()],
        )

    def perform(self, node, inp, out_):
        N, M, k = inp
        (out,) = out_
        out[0] = np.tri(N, M, k, dtype=self.dtype)

    def infer_shape(self, fgraph, node, in_shapes):
        out_shape = [node.inputs[0], node.inputs[1]]
        return [out_shape]

    def grad(self, inp, grads):
        return [grad_undefined(self, i, inp[i]) for i in range(3)]


def tri(
    N: int, M: Optional[int] = None, k: Optional[int] = 0, dtype: Optional[type] = None
):
    """
    An array with ones at and below the given diagonal and zeros elsewhere.

    Parameters
    ----------
    N : int
        Number of rows in the array.
    M : int, optional
        Number of columns in the array.
        By default, `M` is taken equal to `N`.
    k : int, optional
        The sub-diagonal at and below which the array is filled.
        `k` = 0 is the main diagonal, while `k` < 0 is below it,
        and `k` > 0 is above.  The default is 0.
    dtype : dtype, optional
        Data type of the returned array. The default is None.

    Returns
    -------
    Array of shape (N, M)
        Array with its lower triangle filled with ones and zero elsewhere;
        in other words ``T[i,j] == 1`` for ``i <= j + k``, 0 otherwise.

    """
    if dtype is None:
        dtype = config.floatX
    if M is None:
        M = N
    op = Tri(dtype)
    return op(N, M, k)


def tril(m: np.array, k: Optional[int] = 0) -> np.array:
    """
    Lower triangle of an array.

    Return a copy of an array with elements above the `k`-th diagonal zeroed.

    Parameters
    ----------
    m : array_like, shape (M, N)
        Input array.
    k : int, optional
        Diagonal above which to zero elements.  `k = 0` (the default) is the
        main diagonal, `k < 0` is below it and `k > 0` is above.

    Returns
    -------
    array, shape (M, N)
        Lower triangle of `m`, of same shape and data-type as `m`.

    See Also
    --------
    triu : Same thing, only for the upper triangle.

    """
    return m * tri(m.shape[0], m.shape[1], k=k, dtype=m.dtype)


def triu(m, k: int = 0) -> np.array:
    """
    Upper triangle of an array.

    Return a copy of a matrix with the elements below the `k`-th diagonal
    zeroed.

    Parameters
    ----------
    m : array_like, shape (M, N)
        Input array.
    k : int
        Diagonal above which to zero elements.  `k = 0` (the default) is the
        main diagonal, `k < 0` is below it and `k > 0` is above.

    See Also
    --------
    tril : Lower triangle of an array.

    """
    return m * (
        constant(1, dtype=m.dtype) - tri(m.shape[0], m.shape[1], k=k - 1, dtype=m.dtype)
    )


class Eye(Op):

    __props__ = ("dtype",)

    def __init__(self, dtype=None):
        if dtype is None:
            dtype = config.floatX
        self.dtype = dtype

    def make_node(self, n, m, k):
        n = as_tensor_variable(n)
        m = as_tensor_variable(m)
        k = as_tensor_variable(k)
        assert n.ndim == 0
        assert m.ndim == 0
        assert k.ndim == 0
        return Apply(
            self,
            [n, m, k],
            [TensorType(dtype=self.dtype, broadcastable=(False, False))()],
        )

    def perform(self, node, inp, out_):
        n, m, k = inp
        (out,) = out_
        out[0] = np.eye(n, m, k, dtype=self.dtype)

    def infer_shape(self, fgraph, node, in_shapes):
        out_shape = [node.inputs[0], node.inputs[1]]
        return [out_shape]

    def grad(self, inp, grads):
        return [grad_undefined(self, i, inp[i]) for i in range(3)]


def eye(n: int, m: Optional[int] = None, k: Optional[int] = 0, dtype=None):
    """Return a 2-D array with ones on the diagonal and zeros elsewhere.

    Parameters
    ----------
    n : int
        Number of rows in the output.
    m : int, optional
        Number of columns in the output. If None, defaults to `N`.
    k : int, optional
        Index of the diagonal: 0 (the default) refers to the main diagonal,
        a positive value refers to an upper diagonal, and a negative value
        to a lower diagonal.
    dtype : data-type, optional
        Data-type of the returned array.

    Returns
    -------
    ndarray of shape (N,M)
        An array where all elements are equal to zero, except for the `k`-th
        diagonal, whose values are equal to one.

    """
    if dtype is None:
        dtype = config.floatX
    if m is None:
        m = n
    localop = Eye(dtype)
    return localop(n, m, k)


def identity_like(x):
    return eye(x.shape[0], x.shape[1], k=0, dtype=x.dtype)


def alloc_validate_shape(shape):
    sh = [as_tensor_variable(s) for s in shape]
    bcast = []
    for i, s in enumerate(sh):

        def err_str():
            if config.exception_verbosity == "high":
                return "\n" + min_informative_str(s)
            else:
                return str(s)

        if s.type.dtype not in integer_dtypes:
            s_as_str = err_str()
            raise TypeError(
                "Shape arguments to Alloc must be integers, "
                f"but argument {i} is not for apply node: {s_as_str}"
            )
        if s.ndim != 0:
            s_as_str = err_str()
            raise TypeError(
                "Each shape dimension to Alloc must be a scalar, ",
                f"but dimension {i} have {int(s.ndim)} dimensions for apply node: {s_as_str}",
            )

        # if s is constant 1, then we're broadcastable in that dim
        try:
            const_shp = get_scalar_constant_value(s)
        except NotScalarConstantError:
            const_shp = None
        bcast.append(1 == const_shp)
    return sh, bcast


class Alloc(COp):
    """Create a `TensorVariable` from an initial value and a desired shape.

    alloc(value, shape0, shape1, ..., shapeN)

    Returns an N-dimensional tensor initialized by a value, using something
    equivalent to

        z = numpy.zeros(shape, value.dtype)
        z += value

    The result has N dimensions, has the dtype of the given value, and is
    obtained by broadcasting value over the output array.

    This `Op` is used to replace ``fill`` during optimizations, because, after
    shapes are lifted, the first argument to ``fill`` can often be pruned from
    the graph.

    """

    _f16_ok = True
    __props__: Union[Tuple, Tuple[str, str]] = ()

    def validate_shape(self, shape):
        return alloc_validate_shape(shape)

    def make_node(self, value, *shape):
        v = as_tensor_variable(value)
        sh, bcast = alloc_validate_shape(shape)
        if v.ndim > len(sh):
            raise TypeError(
                "The Alloc value to use has more dimensions"
                " than the specified dimensions",
                v.ndim,
                len(sh),
            )
        otype = TensorType(dtype=v.dtype, broadcastable=bcast)
        return Apply(self, [v] + sh, [otype()])

    def perform(self, node, inputs, out_):
        (out,) = out_
        v = inputs[0]
        sh = tuple([int(i) for i in inputs[1:]])
        if out[0] is None or out[0].shape != sh:
            if v.size == 1 and v.item() == 0:
                out[0] = np.zeros(sh, dtype=v.dtype)
            else:
                out[0] = np.empty(sh, dtype=v.dtype)
                out[0][...] = v  # broadcast v to fill us up
        else:
            # reuse the allocated memory.
            out[0][...] = v  # broadcast v to fill us up

    def c_code(self, node, name, inp, out, sub):
        vv = inp[0]
        ndim = len(inp[1:])
        (zz,) = out
        fail = sub["fail"]

        code = f"""
            npy_intp shape[{ndim}];
            """

        # Initialize shape
        for i, shp_i in enumerate(inp[1:]):
            code += """
                shape[%(i)s] = ((dtype_%(shp_i)s*) PyArray_DATA(%(shp_i)s))[0];
                """ % dict(
                i=i, shp_i=shp_i
            )

        code += """
            int need_new_out = (NULL == %(zz)s);
            for (int i = 0; i < %(ndim)s; i++)
                need_new_out = (need_new_out
                                || (PyArray_DIMS(%(zz)s)[i] != shape[i]));

            if (need_new_out)
            {
                Py_XDECREF(%(zz)s);
                %(zz)s = (PyArrayObject*) PyArray_SimpleNew(%(ndim)s,
                    shape, PyArray_TYPE((PyArrayObject*) py_%(vv)s));
                if (!%(zz)s)
                {
                    PyErr_SetString(PyExc_MemoryError, "alloc failed");
                    %(fail)s
                }
            }

            // This function takes care of broadcasting
            if (PyArray_CopyInto(%(zz)s, %(vv)s) == -1)
              %(fail)s
            """ % dict(
            vv=vv, ndim=ndim, zz=zz, fail=fail
        )

        return code

    def c_code_cache_version(self):
        return (2,)

    def infer_shape(self, fgraph, node, input_shapes):
        return [node.inputs[1:]]

    def connection_pattern(self, node):

        rval = [[True]]

        for ipt in node.inputs[1:]:
            rval.append([False])

        return rval

    def grad(self, inputs, grads):
        x = inputs[0]
        gz = grads[0]
        n_axes_to_sum = gz.ndim - x.ndim
        # The number of dimensions added
        axis = list(range(n_axes_to_sum))
        # The broadcasted dimensions
        axis_broadcasted = []
        axis_kept = []
        for i, (ib, gb) in enumerate(
            zip(
                inputs[0].broadcastable,
                # We need the dimensions corresponding to x
                grads[0].broadcastable[-inputs[0].ndim :],
            )
        ):
            if ib and not gb:
                axis_broadcasted.append(i + n_axes_to_sum)
            else:
                axis_kept.append(i)
        gx = gz.sum(axis=axis + axis_broadcasted)
        if axis_broadcasted:
            new_order = ["x"] * x.ndim
            for idx, axis in enumerate(axis_kept):
                new_order[axis] = idx
            gx = gx.dimshuffle(new_order)
            # Dimshuffle to add back the broadcasted dims
        # The *elements* of the output are not connected to
        # the inputs that specify the shape. If you grow the
        # shape by epsilon, the existing elements do not
        # change.
        return [gx] + [DisconnectedType()() for i in inputs[1:]]

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return [None]
        return self(eval_points[0], *inputs[1:], **dict(return_list=True))

    def do_constant_folding(self, fgraph, node):
        clients = fgraph.clients[node.outputs[0]]

        if not clients:
            return False

        for client in clients:
            if client[0] == "output":
                # If the output is a constant, it will have to be deepcopied
                # each time the function is called.  So we do not fold.
                return False
            elif (
                # The following ops work inplace of their input id 0.
                client[1] == 0
                and isinstance(
                    client[0].op,
                    (
                        # Ops that will work inplace on the Alloc. So if they
                        # get constant_folded, they would copy the
                        # constant and this is less efficients.
                        # Not doing the constant folding could also lower
                        # the peak memory usage, as we the "constant" won't
                        # always exists.
                        aesara.tensor.subtensor.IncSubtensor,
                        aesara.tensor.subtensor.AdvancedIncSubtensor1,
                        aesara.tensor.subtensor.AdvancedIncSubtensor,
                        aesara.tensor.blas.Gemv,
                        aesara.tensor.blas_c.CGemv,
                        aesara.tensor.blas.Ger,
                        aesara.tensor.blas_c.CGer,
                        aesara.tensor.blas_scipy.ScipyGer,
                    ),
                )
            ):
                return False
            # If the clients is a transfer to the GPU, we don't want to
            # fold. We let the Alloc being moved to the GPU, then we
            # let the GPU algo decide if it need to fold it or not.
            elif client[0].op.__class__.__name__.lower().startswith("gpu"):
                return False
        return True


alloc = Alloc()
pprint.assign(alloc, printing.FunctionPrinter("alloc"))


def full(shape, fill_value, dtype=None):
    """Return a new array of given shape and type, filled with `fill_value`.

    See ``numpy.full``.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the new array, e.g., ``(2, 3)`` or ``2``.
    fill_value : scalar or array_like
        Fill value.
    dtype : data-type, optional
        The desired data-type for the array  The default, None, means
        `np.array(fill_value).dtype`.

    """
    fill_value = as_tensor_variable(fill_value)
    if dtype:
        fill_value = fill_value.astype(dtype)
    return alloc(fill_value, *shape)


class MakeVector(COp):
    """Concatenate a number of scalars together into a vector.

    This is a simple version of stack() that introduces far less cruft
    into the graph. Should work with 0 inputs. The constant_folding
    optimization will remove it.

    """

    __props__ = ("dtype",)

    def __init__(self, dtype: str = "int64"):
        self.dtype = dtype

    def make_node(self, *inputs):
        inputs = list(map(as_tensor_variable, inputs))
        if not all(a.type == inputs[0].type for a in inputs) or (
            len(inputs) > 0 and inputs[0].dtype != self.dtype
        ):
            dtype = aes.upcast(self.dtype, *[i.dtype for i in inputs])
            # upcast the input to the determined dtype,
            # but don't downcast anything
            assert dtype == self.dtype, (
                "The upcast of the inputs to MakeVector should match the "
                "dtype given in __init__."
            )
            if not all(self.dtype == cast(i, dtype=dtype).dtype for i in inputs):
                raise TypeError(
                    "MakeVector.make_node expected inputs"
                    f" upcastable to {self.dtype}. got {[i.dtype for i in inputs]}"
                )
            inputs = [cast(i, dtype=dtype) for i in inputs]
        assert all(self.dtype == a.dtype for a in inputs)
        assert all(a.ndim == 0 for a in inputs)

        if inputs:
            dtype = inputs[0].type.dtype
        else:
            dtype = self.dtype
        # bcastable = (len(inputs) == 1)
        bcastable = False
        otype = TensorType(broadcastable=(bcastable,), dtype=dtype)
        return Apply(self, inputs, [otype()])

    def perform(self, node, inputs, out_):
        (out,) = out_
        # not calling aesara._asarray as optimization
        if (out[0] is None) or (out[0].size != len(inputs)):
            out[0] = _asarray(inputs, dtype=node.outputs[0].dtype)
        else:
            # assume that out has correct dtype. there is no cheap way to check
            out[0][...] = inputs

    def c_code_cache_version(self):
        return (2,)

    def c_code(self, node, name, inp, out_, props):
        (out,) = out_
        # Shouldn't use PyArray_TYPE(inp[0]) for the dtype
        # when len(inp) == 0 (we need to support this case.
        # So there will be (1 * nb_dtype) + ((nb len(inp) - 1 ))
        # different c code with the following algo
        out_shape = len(inp)
        out_num = np.dtype(node.outputs[0].dtype).num
        # don't use dtype_%(out)s as when check_input=False, it isn't defined.
        out_dtype = node.outputs[0].type.dtype_specs()[1]
        if len(inp) > 0:
            assert self.dtype == node.inputs[0].dtype
            out_num = f"PyArray_TYPE({inp[0]})"

        ret = (
            """
        npy_intp dims[1];
        dims[0] = %(out_shape)s;
        if(!%(out)s || PyArray_DIMS(%(out)s)[0] != %(out_shape)s){
            Py_XDECREF(%(out)s);
            %(out)s = (PyArrayObject*)PyArray_EMPTY(1, dims, %(out_num)s, 0);
        }
        """
            % locals()
        )
        for idx, i in enumerate(inp):
            ret += (
                """
            *((%(out_dtype)s *)PyArray_GETPTR1(%(out)s, %(idx)s)) = *((%(out_dtype)s *) PyArray_DATA(%(i)s));
            """
                % locals()
            )
        return ret

    def infer_shape(self, fgraph, node, ishapes):
        return [(len(ishapes),)]

    def grad(self, inputs, output_gradients):
        # If the output is of an integer dtype, no gradient shall pass
        if self.dtype in discrete_dtypes:
            return [ipt.zeros_like().astype(config.floatX) for ipt in inputs]

        grads = []
        for i, inp in enumerate(inputs):
            grads.append(output_gradients[0][i])
        return grads

    def R_op(self, inputs, eval_points):
        if None in eval_points:
            return [None]
        return self.make_node(*eval_points).outputs


make_vector = MakeVector()


class Transfer(Protocol):
    _others: List


def transfer_decorator(func: Any) -> Transfer:
    return func


@transfer_decorator
def transfer(var, target: str):
    """
    Return a version of `var` transferred to `target`.

    `cpu` mean a TensorType (on the CPU).  Other types may define
    additional targets.

    Parameters
    ----------
    var : variable
        A aesara variable
    target : str
        The target of the transfer
    """
    if target == "cpu":
        return as_tensor_variable(var)
    else:
        for trans in transfer._others:
            res = trans(var, target)
            if res is not None:
                return res
    raise ValueError(f"Can't transfer to target {target}")


transfer._others = []


def register_transfer(fn: Callable):
    """
    Register a transfer function for alternative targets.

    Parameters
    ----------
    fn : callable
    """
    transfer._others.append(fn)


"""Create a duplicate of `a` (with duplicated storage)"""
tensor_copy = Elemwise(aes.identity)
pprint.assign(tensor_copy, printing.IgnorePrinter())


class Default(Op):
    """
    Takes an input x and a default value.

    If the input is not None, a reference to it is returned.
    If the input is None, a copy of the default value is returned instead.
    The input and the default must have exactly the same type.

    """

    view_map = {0: [0]}
    __props__ = ()

    def make_node(self, x, default):
        x, default = as_tensor_variable(x), as_tensor_variable(default)
        if x.type != default.type:
            raise TypeError("Both default() arguments must have same type", x, default)
        return Apply(self, [x, default], [default.type()])

    def perform(self, node, inp, out_):
        x, default = inp
        (out,) = out_
        if x is None:
            # why copy?  Aesara can't yet understand out[0] being a view of
            # either x or y, so we can be a view of x, but only a copy of y.
            out[0] = default.copy()
        else:
            out[0] = x


default = Default()


##########################
# View Operations
##########################


def extract_constant(x, elemwise: bool = True, only_process_constants: bool = False):
    """
    This function is basically a call to tensor.get_scalar_constant_value.

    The main difference is the behaviour in case of failure. While
    get_scalar_constant_value raises an TypeError, this function returns x,
    as a tensor if possible. If x is a ScalarVariable from a
    scalar_from_tensor, we remove the conversion. If x is just a
    ScalarVariable, we convert it to a tensor with tensor_from_scalar.

    """
    try:
        x = get_scalar_constant_value(x, elemwise, only_process_constants)
    except NotScalarConstantError:
        pass
    if isinstance(x, aes.ScalarVariable) or isinstance(
        x, aes.sharedvar.ScalarSharedVariable
    ):
        if x.owner and isinstance(x.owner.op, ScalarFromTensor):
            x = x.owner.inputs[0]
        else:
            x = tensor_from_scalar(x)
    return x


def transpose(x, axes: Optional[List] = None):
    """
    Reorder the dimensions of x. (Default: reverse them)

    This is a macro around dimshuffle that matches the numpy.transpose function.

    """
    if axes is None:
        axes = list(range((x.ndim - 1), -1, -1))
    ret = DimShuffle(x.broadcastable, axes)(x)
    if x.name and axes == list(range((x.ndim - 1), -1, -1)):
        ret.name = x.name + ".T"
    return ret


def split(x, splits_size, n_splits: int, axis: int = 0):
    the_split = Split(n_splits)
    return the_split(x, axis, splits_size)


class Split(COp):
    """Partition a `TensorVariable` along some axis.

    Examples
    --------
    >>> x = vector()
    >>> splits = lvector()
    You have to declare right away how many split_points there will be.
    >>> ra, rb, rc = split(x, splits, n_splits = 3, axis = 0)
    >>> f = function([x, splits], [ra, rb, rc])
    >>> a, b, c = f([0,1,2,3,4,5], [3, 2, 1])
    a == [0,1,2]
    b == [3, 4]
    c == [5]

    """

    len_splits: Optional[int] = None
    """A Split instance will have this many outputs, and require that
    the splits argument to `perform` have exactly this many elements.
    """
    __props__ = ("len_splits",)

    def __init__(self, len_splits):
        self.len_splits = int(len_splits)

    def __str__(self):
        return f"{self.__class__.__name__ }{{{self.len_splits}}}"

    def make_node(self, x, axis, splits):
        """WRITEME"""
        x = as_tensor_variable(x)
        axis = as_tensor_variable(axis)
        splits = as_tensor_variable(splits)

        if splits.type not in int_vector_types:
            raise TypeError("`splits` parameter must be tensors of integer type")
        if axis.type not in int_types:
            raise TypeError("`axis` parameter must be an integer scalar")

        #         # The following lines are necessary if we allow splits of zero
        #         if isinstance(axis, Constant):
        #             x = unbroadcast(x, int(axis.data))
        #         else:
        #             x = unbroadcast(x, *range(x.type.ndim))

        inputs = [x, axis, splits]
        outputs = [x.type() for i in range(self.len_splits)]

        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, outputs):
        x, axis, splits = inputs

        len_along_axis = x.shape[axis]

        if len(splits) != self.len_splits:
            raise ValueError("Length of `splits` is not equal to `len_splits`")
        if np.sum(splits) != len_along_axis:
            raise ValueError(
                f"The splits sum to {np.sum(splits)}; expected {len_along_axis}"
            )
        if builtins.any([nb < 0 for nb in splits]):
            raise ValueError(
                "Attempted to make an array with a " "negative number of elements"
            )

        # Checking is done, let's roll the splitting algorithm!
        # Basically we step along the given axis of x, extracting
        # subtensors of size splits[i] as we go along.

        general_key = [slice(None, None, None) for s in x.shape]
        lower_idx = 0
        for i in range(self.len_splits):
            upper_idx = lower_idx + splits[i]
            general_key[axis] = slice(lower_idx, upper_idx, None)
            outputs[i][0] = x.__getitem__(tuple(general_key)).copy()
            lower_idx = upper_idx

    def infer_shape(self, fgraph, node, in_shapes):
        axis = node.inputs[1]
        splits = node.inputs[2]
        shp_x, shp_axis, shp_splits = in_shapes
        out_shapes = []
        for i in range(self.len_splits):
            temp = as_tensor_variable(shp_x)
            temp = aesara.tensor.subtensor.set_subtensor(temp[axis], splits[i])
            temp = [temp[i] for i in range(len(shp_x))]
            out_shapes.append(temp)
        return out_shapes

    def grad(self, inputs, g_outputs):
        """Join the gradients along the axis that was used to split x."""
        x, axis, n = inputs
        outputs = self(*inputs, **dict(return_list=True))
        # If all the output gradients are disconnected, then so are the inputs
        if builtins.all([isinstance(g.type, DisconnectedType) for g in g_outputs]):
            return [
                DisconnectedType()(),
                grad_undefined(self, 1, axis),
                grad_undefined(self, 2, n),
            ]
        # Else, we have to make them zeros before joining them
        new_g_outputs = []
        for o, g in zip(outputs, g_outputs):
            if isinstance(g.type, DisconnectedType):
                new_g_outputs.append(o.zeros_like())
            else:
                new_g_outputs.append(g)

        return [
            join(axis, *new_g_outputs),
            grad_undefined(self, 1, axis),
            grad_undefined(self, 2, n),
        ]

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return [None for i in self.len_splits]
        return self.make_node(eval_points[0], *inputs[1:]).outputs

    def c_code_cache_version(self):
        return (2,)

    def c_support_code(self, **kwargs):
        return """
        /* Return 1 if output has the correct shape. */
        int split_output_shape_is_correct (
            PyArrayObject* output, PyArrayObject* array_to_split, int axis_to_split, npy_intp split_size
        ) {
            return
                PyArray_NDIM(output) == PyArray_NDIM(array_to_split)
                && memcmp(
                    PyArray_DIMS(output),
                    PyArray_DIMS(array_to_split),
                    axis_to_split * sizeof(npy_intp)
                ) == 0
                && memcmp(
                    PyArray_DIMS(output) + axis_to_split + 1,
                    PyArray_DIMS(array_to_split) + axis_to_split + 1,
                    (PyArray_NDIM(array_to_split) - axis_to_split - 1) * sizeof(npy_intp)
                ) == 0
                && split_size == PyArray_DIM(output, axis_to_split);
        }
        """

    def c_code(self, node, name, inputs, outputs, sub):
        if self.len_splits == 0:
            # There are no outputs, then nothing to do.
            return ""

        # outputs_pointers lists the addresses of the pointers to the outputs.
        outputs_pointers = "&" + (", &".join(outputs))
        x, axis, splits = inputs
        fail = sub["fail"]
        x_typenum = np.dtype(node.inputs[0].dtype).num
        x_itemsize = np.dtype(node.inputs[0].dtype).itemsize
        axis_dtype = node.inputs[1].type.dtype_specs()[1]
        splits_dtype = node.inputs[2].type.dtype_specs()[1]
        expected_splits_count = self.len_splits

        return (
            """
        int ndim = PyArray_NDIM(%(x)s);
        int axis = (int)(*(%(axis_dtype)s*)PyArray_GETPTR1(%(axis)s, 0));
        int splits_count = PyArray_DIM(%(splits)s, 0);
        npy_intp len_along_axis, sum_of_splits = 0, current_split_length = 0, current_split_start = 0;
        npy_intp* split_dims = NULL;
        PyObject* split_view = NULL;
        npy_intp data_offset;
        int i;
        PyArrayObject** outputs[] = {%(outputs_pointers)s};

        /* Check inputs. */

        if (splits_count != %(expected_splits_count)s) {
            PyErr_Format(PyExc_ValueError,
                "Split: splits count (%%d) != expected count (%%d).", splits_count, %(expected_splits_count)s);
            %(fail)s
        }

        if (axis < 0) {
            axis += ndim;
        }
        if (axis < 0 || axis >= ndim) {
            PyErr_Format(PyExc_IndexError, "Split: invalid axis %%d for a %%d-D array.", axis, ndim);
            %(fail)s
        }
        len_along_axis = PyArray_DIM(%(x)s, axis);

        for (i = 0; i < splits_count; ++i) {
            current_split_length = (npy_intp)(*(%(splits_dtype)s*)PyArray_GETPTR1(%(splits)s, i));
            if (current_split_length < 0) {
                PyErr_Format(PyExc_ValueError,
                    "Split: you try to take a negative number (%%ld) of elements.", current_split_length);
                %(fail)s
            }
            sum_of_splits += current_split_length;
        }
        if (sum_of_splits != len_along_axis) {
            PyErr_Format(PyExc_ValueError, "Split: the splits sums to %%ld, expected %%ld.", sum_of_splits, len_along_axis);
            %(fail)s
        }

        /* Check outputs. */

        split_dims = (npy_intp*) malloc(ndim * sizeof(npy_intp));
        if (split_dims == NULL) {
            PyErr_NoMemory();
            %(fail)s
        }

        memcpy(split_dims, PyArray_DIMS(%(x)s), ndim * sizeof(npy_intp));

        for (i = 0; i < splits_count; ++i) {
            PyArrayObject** output = outputs[i];
            current_split_length = (npy_intp) (* (%(splits_dtype)s*) PyArray_GETPTR1(%(splits)s, i));
            if (*output == NULL || !split_output_shape_is_correct(*output, %(x)s, axis, current_split_length)) {
                Py_XDECREF(*output);
                split_dims[axis] = current_split_length;
                *output = (PyArrayObject*)PyArray_EMPTY(ndim, split_dims, %(x_typenum)s, PyArray_IS_F_CONTIGUOUS(%(x)s));
                if (outputs == NULL) {
                    PyErr_SetString(PyExc_RuntimeError, "Split: unable to allocate an output.");
                    free(split_dims);
                    %(fail)s
                }
            }
        }

        /* Compute split. */

        for (i = 0; i < splits_count; ++i) {
            current_split_length = (npy_intp) (* (%(splits_dtype)s*) PyArray_GETPTR1(%(splits)s, i));
            data_offset = PyArray_STRIDE(%(x)s, axis) * current_split_start;
            split_dims[axis] = current_split_length;
            split_view = PyArray_New(&PyArray_Type,
                                    ndim, split_dims,
                                    %(x_typenum)s,
                                    PyArray_STRIDES(%(x)s),
                                    PyArray_BYTES(%(x)s) + data_offset,
                                    %(x_itemsize)s,
                                    PyArray_FLAGS(%(x)s),
                                    NULL);
            if (split_view == NULL) {
                PyErr_SetString(PyExc_RuntimeError, "Split: unable to create a view for a split.");
                free(split_dims);
                %(fail)s
            }
            if (PyArray_CopyInto(*outputs[i], (PyArrayObject*)split_view) != 0) {
                PyErr_SetString(PyExc_RuntimeError, "Split: unable to copy a split view into the output.");
                Py_XDECREF(split_view);
                free(split_dims);
                %(fail)s
            }
            Py_XDECREF(split_view);
            current_split_start += current_split_length;
        }

        free(split_dims);
        """
            % locals()
        )


def addbroadcast(x, *axes):
    """
    Make the input broadcastable in the specified axes.

    For example, addbroadcast(x, 0) will make the first dimension of
    x broadcastable. When performing the function, if the length of
    x along that dimension is not 1, a ValueError will be raised.

    We apply the opt here not to pollute the graph especially during
    the gpu optimization

    Parameters
    ----------
    x : tensor_like
        Input aesara tensor.
    axis : an int or an iterable object such as list or tuple of int values
        The dimension along which the tensor x should be broadcastable.
        If the length of x along these dimensions is not 1, a ValueError will
        be raised.

    Returns
    -------
    tensor
        A aesara tensor, which is broadcastable along the specified dimensions.

    """
    rval = Rebroadcast(*[(axis, True) for axis in axes])(x)
    return aesara.tensor.basic_opt.apply_rebroadcast_opt(rval)


def unbroadcast(x, *axes):
    """
    Make the input impossible to broadcast in the specified axes.

    For example, addbroadcast(x, 0) will make the first dimension
    of x broadcastable. When performing the function, if the length
    of x along that dimension is not 1, a ValueError will be raised.

    We apply the opt here not to pollute the graph especially during
    the gpu optimization

    Parameters
    ----------
    x : tensor_like
        Input aesara tensor.
    axis : an int or an iterable object such as list or tuple of int values
        The dimension along which the tensor x should be unbroadcastable.
        If the length of x along these dimensions is not 1, a ValueError will
        be raised.

    Returns
    -------
    tensor
        A aesara tensor, which is unbroadcastable along the specified dimensions.

    """
    rval = Rebroadcast(*[(axis, False) for axis in axes])(x)
    return aesara.tensor.basic_opt.apply_rebroadcast_opt(rval)


def patternbroadcast(x, broadcastable):
    """
    Make the input adopt a specific broadcasting pattern.

    Broadcastable must be iterable. For example,
    patternbroadcast(x, (True, False)) will make the first
    dimension of x broadcastable and the second dimension
    not broadcastable, so x will now be a row.

    We apply the opt here not to pollute the graph especially during the gpu
    optimization.

    Parameters
    ----------
    x : tensor_like
        Input aesara tensor.
    broadcastable : an iterable object such as list or tuple of bool values
        A set of boolean values indicating whether a dimension should be
        broadcastable or not. If the length of x along these dimensions is
        not 1, a ValueError will be raised.

    Returns
    -------
    tensor
        A aesara tensor, which is unbroadcastable along the specified dimensions.

    """
    rval = Rebroadcast(*[(i, broadcastable[i]) for i in range(len(broadcastable))])(x)
    return aesara.tensor.basic_opt.apply_rebroadcast_opt(rval)


class Join(COp):
    """
    Concatenate multiple `TensorVariable`s along some axis.

    The axis must be given as first argument. All tensors must have the same
    shape along all dimensions other than this axis.
    Of course, TensorVariable instances do not have a shape, so this error
    cannot be caught until runtime.  See `perform()`.

    See Also
    --------
    stack : For joins involving scalar values

    Examples
    --------
    >>> x, y, z = tensor.matrix(), tensor.matrix(), tensor.matrix()
    >>> u = tensor.vector()

    >>> r = join(0, x, y, z)
    >>> c = join(1, x, y, z)
    >>> join(2, x, y, z)    # WRONG: the axis has to be an index into the shape
    >>> join(0, x, u)       # WRONG: joined tensors must have the same rank

    """

    check_input = False
    __props__ = ("view",)

    def __init__(self, view=-1):
        self.view = view
        if view != -1:
            # since the first input is always the axis, the tensors
            # start from index 1.
            self.view_map = {0: [1 + view]}

    def __str__(self):
        if self.view == -1:
            return self.__class__.__name__
        else:
            return "{}{{{}}}".format(
                self.__class__.__name__,
                ", ".join(
                    "{}={!r}".format(p, getattr(self, p)) for p in self.__props__
                ),
            )

    def __setstate__(self, d):
        self.__dict__.update(d)
        if not hasattr(self, "view"):
            self.view = -1

    def make_node(self, *axis_and_tensors):
        """
        Parameters
        ----------
        axis: an Int or integer-valued Variable
        tensors
            A variable number (but not zero) of tensors to
            concatenate along the specified axis.  These tensors must have
            the same shape along all dimensions other than this axis.

        Returns
        -------
        A symbolic Variable
            It has the same ndim as the input tensors, and the most inclusive
            dtype.

        """
        axis, tens = axis_and_tensors[0], axis_and_tensors[1:]
        if not tens:
            raise ValueError("Cannot join an empty list of tensors")
        as_tensor_variable_args = [as_tensor_variable(x) for x in tens]

        dtypes = [x.type.dtype for x in as_tensor_variable_args]
        out_dtype = aes.upcast(*dtypes)

        def output_maker(bcastable):
            return tensor(dtype=out_dtype, broadcastable=bcastable)

        return self._make_node_internal(
            axis, tens, as_tensor_variable_args, output_maker
        )

    def _make_node_internal(self, axis, tens, as_tensor_variable_args, output_maker):
        if not builtins.all(targs.type.ndim for targs in as_tensor_variable_args):
            raise TypeError(
                "Join cannot handle arguments of dimension 0."
                " For joining scalar values, see @stack"
            )
        # Handle single-tensor joins immediately.
        if len(as_tensor_variable_args) == 1:
            bcastable = list(as_tensor_variable_args[0].type.broadcastable)
        else:
            # When the axis is fixed, a dimension should be
            # broadcastable if at least one of the inputs is
            # broadcastable on that dimension (see justification below),
            # except for the axis dimension.
            # Initialize bcastable all false, and then fill in some trues with
            # the loops.
            bcastable = [False] * len(as_tensor_variable_args[0].type.broadcastable)
            ndim = len(bcastable)
            # Axis can also be a constant
            if not isinstance(axis, int):
                try:
                    # Note : `get_scalar_constant_value` returns a ndarray not
                    # an int
                    axis = int(get_scalar_constant_value(axis))

                except NotScalarConstantError:
                    pass
            if isinstance(axis, int):
                # Basically, broadcastable -> length 1, but the
                # converse does not hold. So we permit e.g. T/F/T
                # joins, and if they fail at runtime they fail, but if
                # they don't then it means that the argument where
                # that broadcastable flag was False had length 1 along
                # this dimension, and therefore this dimension should
                # be broadcastable for the output.

                if axis < -ndim:
                    raise IndexError(
                        f"Join axis {int(axis)} out of bounds [0, {int(ndim)})"
                    )
                if axis < 0:
                    axis += ndim

                for x in as_tensor_variable_args:
                    for current_axis, bflag in enumerate(x.type.broadcastable):
                        # Constant negative axis can no longer be negative at
                        # this point. It safe to compare this way.
                        if current_axis == axis:
                            continue
                        if bflag:
                            bcastable[current_axis] = True
                try:
                    bcastable[axis] = False
                except IndexError:
                    raise ValueError(
                        'Join argument "axis" is out of range'
                        " (given input dimensions)"
                    )
            else:
                # When the axis may vary, no dimension can be guaranteed to be
                # broadcastable.
                bcastable = [False] * len(as_tensor_variable_args[0].type.broadcastable)

        if not builtins.all(
            [x.ndim == len(bcastable) for x in as_tensor_variable_args[1:]]
        ):
            raise TypeError(
                "Join() can only join tensors with the same " "number of dimensions."
            )

        inputs = [as_tensor_variable(axis)] + list(as_tensor_variable_args)
        if inputs[0].type not in int_types:
            raise TypeError(
                "Axis could not be cast to an integer type",
                axis,
                inputs[0].type,
                int_types,
            )

        outputs = [output_maker(bcastable)]

        node = Apply(self, inputs, outputs)
        return node

    def perform(self, node, axis_and_tensors, out_):
        (out,) = out_
        view = self.view
        axis, tens = axis_and_tensors[0], axis_and_tensors[1:]
        # we check these tensors for being empty.
        if (view != -1) and np.all(
            [tensor.shape[axis] == 0 for tensor in tens[0:view] + tens[view + 1 :]]
        ):
            out[0] = tens[view]

        else:
            ndim = tens[0].ndim
            if axis < -ndim:
                raise IndexError(
                    f"Join axis {int(axis)} out of bounds [0, {int(ndim)})"
                )

            out[0] = _asarray(
                np.concatenate(tens, axis=axis), dtype=node.outputs[0].type.dtype
            )

    def c_code_cache_version(self):
        return (5,)

    def c_code(self, node, name, inputs, outputs, sub):
        axis, tens = inputs[0], inputs[1:]
        view = self.view
        non_empty_tensor = tens[view]
        input_1 = tens[0]
        l = len(tens)
        (out,) = outputs
        fail = sub["fail"]
        adtype = node.inputs[0].type.dtype_specs()[1]
        copy_to_list = []

        for i, inp in enumerate(tens):
            copy_to_list.append(
                f"""Py_INCREF({inp});
                   PyList_SetItem(list, {i}, (PyObject*){inp});"""
            )

        copy_inputs_to_list = "\n".join(copy_to_list)
        n = len(tens)

        code = (
            """
        int axis = ((%(adtype)s *)PyArray_DATA(%(axis)s))[0];
        PyObject* list = PyList_New(%(l)s);
        %(copy_inputs_to_list)s
        int tensors_lens_sum;
        if(%(view)s != -1) {
            tensors_lens_sum = 0;

            for(int i=0; i < %(n)s; i++){
                tensors_lens_sum += PyArray_DIM((PyArrayObject *)(PyList_GetItem(list, i)), axis);
            }
            tensors_lens_sum -= PyArray_DIM(%(non_empty_tensor)s, axis);
        }
        if(%(view)s != -1 && tensors_lens_sum == 0) {
            Py_XDECREF(%(out)s);
            Py_INCREF(%(non_empty_tensor)s);
            %(out)s = %(non_empty_tensor)s;
        }else{
            //PyObject* PyArray_Concatenate(PyObject* obj, int axis)
            int ndim = PyArray_NDIM(%(input_1)s);
            if( axis < -ndim ){
                PyErr_Format(PyExc_IndexError,
                             "Join axis %%d out of bounds [0, %%d)", axis, ndim);
                %(fail)s
            }
            Py_XDECREF(%(out)s);
            %(out)s = (PyArrayObject *)PyArray_Concatenate(list, axis);
            Py_DECREF(list);
            if(!%(out)s){
                %(fail)s
            }
        }
        """
            % locals()
        )
        return code

    def R_op(self, inputs, eval_points):
        if None in eval_points[1:]:
            return [None]
        return self.make_node(inputs[0], *eval_points[1:]).outputs

    def grad(self, axis_and_tensors, grads):
        """The gradient wrt a join op is a `Split`, used to partition
        the gradient along the `axis` which was used for joining.
        """
        (gz,) = grads
        axis, tens = axis_and_tensors[0], axis_and_tensors[1:]

        rval = [grad_undefined(self, 0, axis)]

        dtypes = [as_tensor_variable(x).type.dtype for x in tens]
        out_dtype = aes.upcast(*dtypes)

        if "float" in out_dtype or "complex" in out_dtype:
            # assume that this is differentiable
            split = Split(len(tens))
            split_gz = split(gz, axis, stack([shape(x)[axis] for x in tens]))
            # If there is only one split, it might not be in a list.
            if not isinstance(split_gz, list):
                split_gz = [split_gz]
            # Split.make_node isn't always able to infer the right
            # broadcast. As the grad need to keep the information,
            # read it if needed.
            split_gz = [
                patternbroadcast(g, t.broadcastable) for t, g in zip(tens, split_gz)
            ]
            rval = rval + split_gz
        else:
            # the output has integer type, so the gradient through it
            # is 0
            rval = rval + [t.zeros_like(dtype=config.floatX) for t in tens]

        return rval

    def infer_shape(self, fgraph, node, ishapes):
        from aesara.tensor.math import eq, ge

        # ishapes[0] contains the size of the axis on which we join
        # Join op should get at least one input to join
        assert len(ishapes) > 1
        n_dim = len(ishapes[1])
        for shp in ishapes[1:]:
            assert shp is not None
            assert len(shp) == n_dim

        # The joining dimension could be negative, but we need it to be
        # in [0, n_dim) in the loop below.
        # An axis < -n_dim or >= ndim would be invalid, but this is
        # not checked here. An Assert op would be a way of addressing that,
        # but it may disrupt optimizations.
        join_dim = switch(ge(node.inputs[0], 0), node.inputs[0], node.inputs[0] + n_dim)
        out_shapes = []
        for dim in range(n_dim):
            # we have to deal with 2 possible cases in here :
            #   a) we are dealing with the dimension for which we join
            #     (called t_side from true side of the if, where the if
            #     compares current dimension with the joining dimension)
            #   b) a non joining dimension ( in which maybe a symbolic
            #      assertion can be used to make sure all tensors have
            #      the same number of elements on this non-joined dimension
            #      this is f_side
            # initialize
            t_side = ishapes[1][dim]
            f_side = ishapes[1][dim]
            # loop over tensors and sum for the joining dimension
            for shp in ishapes[2:]:
                t_side = t_side + shp[dim]
            # return the dimensions found
            out_shapes.append(switch(eq(dim, join_dim), t_side, f_side))

        return [tuple(out_shapes)]


join_ = Join()
pprint.assign(Join, printing.FunctionPrinter("join"))


def join(axis, *tensors_list):
    """
    Convenience function to concatenate `TensorType`s along the given axis.

    This function will not add the op in the graph when it is not useful.
    For example, in the case that the list of tensors to be concatenated
    is one, it will just return the tensor.

    Parameters
    ----------
    axis : int (symbolic or literal)
        On which dimension should the tensors be joined?  The `axis`
        must be a valid index into the shape of the tensors to be
        concatenated.
        The `axis` parameter may either be an integer or an object that
        can be converted to a scalar using `as_scalar`(`axis`). In the
        former case, the axis is fixed at construction, while in the
        latter it may vary over time depending on the value of the
        `axis` variable.
    tensors_list : list of TensorVariable (or list-like)
        A list of tensors to be concatenated along the given axis.
        The shapes of the tensors to be concatenated must be all
        identical, except in the dimension (`axis`) on which they are to
        be joined.
    """
    if len(tensors_list) == 1:
        return tensors_list[0]
    else:
        return join_(axis, *tensors_list)


def roll(x, shift: int, axis: Optional[int] = None):
    """
    Convenience function to roll TensorTypes along the given axis.

    Syntax copies numpy.roll function.

    Parameters
    ----------
    x : tensor_like
        Input tensor.
    shift : int (symbolic or literal)
        The number of places by which elements are shifted.
    axis : int (symbolic or literal), optional
        The axis along which elements are shifted. By default, the array
        is flattened before shifting, after which the original
        shape is restored.

    Returns
    -------
    tensor
        Output tensor, with the same shape as ``x``.

    """
    if axis is None:
        if x.ndim > 1:
            y = x.flatten()
            return roll(y, shift, axis=0).reshape(x.shape)
        else:
            axis = 0

    if axis < 0:
        axis += x.ndim

    # Shift may be larger than the size of the axis. If so, since the
    # roll operation is cyclic, we can take the shift modulo the size
    # of the axis
    shift = shift % x.shape[axis]

    # A slice of all elements in a dimension ':'
    allslice = slice(None)
    # List of slices describing the front half [:, :, shift:, :]
    front_slice = slice(-shift, None)
    front_list = [allslice] * axis + [front_slice] + [allslice] * (x.ndim - axis - 1)
    # List of slices describing the back half [:, :, :shift, :]
    end_slice = slice(0, -shift)
    end_list = [allslice] * axis + [end_slice] + [allslice] * (x.ndim - axis - 1)
    return join(axis, x.__getitem__(tuple(front_list)), x.__getitem__(tuple(end_list)))


def stack(*tensors, **kwargs):
    """Stack tensors in sequence on given axis (default is 0).

    Take a sequence of tensors and stack them on given axis to make a single
    tensor. The size in dimension `axis` of the result will be equal to the number
    of tensors passed.

    Note: The interface stack(*tensors) is deprecated, you should use
    stack(tensors, axis=0) insted.

    Parameters
    ----------
    tensors : list or tuple of tensors
        A list of tensors to be stacked.
    axis : int
        The index of the new axis. Default value is 0.

    Examples
    --------
    >>> a = aesara.tensor.type.scalar()
    >>> b = aesara.tensor.type.scalar()
    >>> c = aesara.tensor.type.scalar()
    >>> x = aesara.tensor.stack([a, b, c])
    >>> x.ndim # x is a vector of length 3.
    1
    >>> a = aesara.tensor.type.tensor4()
    >>> b = aesara.tensor.type.tensor4()
    >>> c = aesara.tensor.type.tensor4()
    >>> x = aesara.tensor.stack([a, b, c])
    >>> x.ndim # x is a 5d tensor.
    5
    >>> rval = x.eval(dict((t, np.zeros((2, 2, 2, 2))) for t in [a, b, c]))
    >>> rval.shape # 3 tensors are stacked on axis 0
    (3, 2, 2, 2, 2)
    >>> x = aesara.tensor.stack([a, b, c], axis=3)
    >>> x.ndim
    5
    >>> rval = x.eval(dict((t, np.zeros((2, 2, 2, 2))) for t in [a, b, c]))
    >>> rval.shape # 3 tensors are stacked on axis 3
    (2, 2, 2, 3, 2)
    >>> x = aesara.tensor.stack([a, b, c], axis=-2)
    >>> x.ndim
    5
    >>> rval = x.eval(dict((t, np.zeros((2, 2, 2, 2))) for t in [a, b, c]))
    >>> rval.shape # 3 tensors are stacked on axis -2
    (2, 2, 2, 3, 2)
    """
    # ---> Remove this when moving to the new interface:
    if not tensors and not kwargs:
        raise ValueError("No tensor arguments provided")

    if not kwargs and not isinstance(tensors[0], (list, tuple)):
        warnings.warn(
            "stack(*tensors) interface is deprecated, use"
            " stack(tensors, axis=0) instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        axis = 0
    elif "tensors" in kwargs:
        tensors = kwargs["tensors"]
        if "axis" in kwargs:
            axis = kwargs["axis"]
        else:
            axis = 0
    else:
        if len(tensors) == 2:
            axis = tensors[1]
        elif "axis" in kwargs:
            axis = kwargs["axis"]
        else:
            axis = 0
        tensors = tensors[0]
    # <--- Until here.

    if len(tensors) == 0:
        raise ValueError("No tensor arguments provided")

    # If all tensors are scalars of the same type, call make_vector.
    # It makes the graph simpler, by not adding DimShuffles and Rebroadcasts

    # This should be an optimization!
    # Doing it here make the graph less canonicalized
    # (more type need to be understood by all optimization)
    # And DebugMode can't detect error in this code as it is not in an
    # optimization.
    # See ticket #660
    if np.all(
        [  # in case there is direct int in tensors.
            isinstance(t, (np.number, float, int, builtins.complex))
            or (
                isinstance(t, Variable)
                and isinstance(t.type, TensorType)
                and t.ndim == 0
            )
            for t in tensors
        ]
    ):
        # in case there is direct int
        tensors = list(map(as_tensor_variable, tensors))
        dtype = aes.upcast(*[i.dtype for i in tensors])
        return aesara.tensor.basic_opt.MakeVector(dtype)(*tensors)
    return join(axis, *[shape_padaxis(t, axis) for t in tensors])


def concatenate(tensor_list: Union[Tuple, List], axis: int = 0):
    """Alias for `join`(axis, *tensor_list).

    This function is similar to `join`, but uses the signature of
    numpy's concatenate function.

    Parameters
    ----------
    tensor_list : list or tuple

    axis : int

    Raises
    ------
    TypeError
        The tensor_list must be a tuple or list.

    """
    # Check someone did not make the common mistake to do something like:
    #   c = concatenate(x, y)
    # instead of
    #   c = concatenate((x, y))
    if not isinstance(tensor_list, (tuple, list)):
        raise TypeError(
            "The 'tensors' argument must be either a tuple "
            "or a list, make sure you did not forget () or [] around "
            "arguments of concatenate.",
            tensor_list,
        )
    return join(axis, *tensor_list)


def get_vector_length(v):
    """Return the run-time length of a symbolic vector.

    Parameters
    ----------
    v
        A rank-1 TensorType variable.

    Raises
    ------
    TypeError
        `v` hasn't the proper type.
    ValueError
        No special case applies, the length is not known.
        In general this is not possible, but for a number of special cases
        the length can be determined at compile / graph-construction time.
        This function implements these special cases.

    """
    v = as_tensor_variable(v)
    if v.ndim != 1:
        raise TypeError(f"argument must be symbolic vector, got '{v}'")
    if v.type.broadcastable[0]:
        return 1
    if isinstance(v, aesara.tensor.sharedvar.TensorSharedVariable) and v.type.ndim == 1:
        return len(v.get_value())
    if isinstance(v, Constant) and v.type.ndim == 1:
        return len(v.data)
    if v.owner and isinstance(v.owner.op, aesara.tensor.basic_opt.MakeVector):
        return len(v.owner.inputs)
    if v.owner and isinstance(v.owner.op, Shape):
        return v.owner.inputs[0].type.ndim

    # We can skip `Op`s that don't affect the length, like unary `Elemwise`
    # `Op`s
    if (
        v.owner
        and isinstance(v.owner.op, Elemwise)
        and len(v.owner.inputs) == 1
        and len(v.owner.outputs) == 1
    ):
        return get_vector_length(v.owner.inputs[0])

    if v.owner and isinstance(v.owner.op, Join):
        axis, *arrays = v.owner.inputs
        try:
            axis = get_scalar_constant_value(axis)
            if axis != 0:
                raise ValueError()
            if not builtins.all(a.ndim == 1 for a in arrays):
                raise ValueError()
            return builtins.sum(get_vector_length(a) for a in arrays)
        except (ValueError, NotScalarConstantError):
            raise ValueError(f"Length of {v} cannot be determined")

    # If we take a slice, we know how many elements it will result in
    # TODO: We can cover more `*Subtensor` cases.
    if (
        v.owner
        and isinstance(v.owner.op, aesara.tensor.subtensor.Subtensor)
        and isinstance(v.owner.op.idx_list[0], slice)
    ):
        try:
            indices = aesara.tensor.subtensor.get_idx_list(
                v.owner.inputs, v.owner.op.idx_list
            )
            start = (
                None
                if indices[0].start is None
                else get_scalar_constant_value(indices[0].start)
            )
            stop = (
                None
                if indices[0].stop is None
                else get_scalar_constant_value(indices[0].stop)
            )
            step = (
                None
                if indices[0].step is None
                else get_scalar_constant_value(indices[0].step)
            )

            arg_len = get_vector_length(v.owner.inputs[0])
            return len(range(*slice(start, stop, step).indices(arg_len)))
        except (ValueError, NotScalarConstantError):
            raise ValueError(f"Length of {v} cannot be determined")

    raise ValueError(f"Length of {v} cannot be determined")


def horizontal_stack(*args):
    """
    Horizontally stack two L{TensorType}s.

    Stack two L{TensorType}s along the second axis (column wise). These
    L{TensorType}s must have the same shape along all dimensions but the
    second.

    """
    # Note: 'horizontal_stack' and 'vertical_stack' do not behave exactly like
    # Numpy's hstack and vstack functions. This is intended, because Numpy's
    # functions have potentially confusing/incoherent behavior (try them on 1D
    # arrays). If this is fixed in a future version of Numpy, it may be worth
    # trying to get closer to Numpy's way of doing things. In the meantime,
    # better keep different names to emphasize the implementation divergences.
    assert len(args) >= 2
    for arg in args:
        assert arg.type.ndim == 2
    return concatenate(args, axis=1)


def vertical_stack(*args):
    assert len(args) >= 2
    for arg in args:
        assert arg.type.ndim == 2
    return concatenate(args, axis=0)


class Flatten(COp):
    """
    Flatten a tensor.

    Flattens a tensor to `ndim` dimensions by preserving the leading
    ndim - 1 shape components.

    .. note:: The interface Flatten(Op) is deprecated, you should use flatten.
    """

    view_map = {0: [0]}

    check_input = False
    __props__ = ("ndim",)

    def __init__(self, ndim=1):
        warnings.warn(
            "Flatten class is deprecated, " "please use flatten method instead.",
            DeprecationWarning,
            stacklevel=4,
        )
        self.ndim = int(ndim)

    def __str__(self):
        return f"{self.__class__.__name__}{{{self.ndim}}}"

    def make_node(self, x):
        t_x = as_tensor_variable(x)
        if self.ndim < 1 or (x.ndim and self.ndim > x.ndim):
            raise ValueError(
                f"invalid output ndimensions ({self.ndim}) for tensor of "
                f"rank {t_x.ndim}"
            )

        # Infer the broadcastable pattern of the output. For every dimension
        # unaffected by the flatten, the broadcast flag should be unchanged.
        # For the dimension resulting from the collapse of other dimensions,
        # it should be broadcastable iff all the collapsed dimensions were
        # broadcastable.
        bcast_kept_dims = x.broadcastable[: self.ndim - 1]
        bcast_new_dim = builtins.all(x.broadcastable[self.ndim - 1 :])
        broadcastable = bcast_kept_dims + (bcast_new_dim,)

        return Apply(self, [t_x], [tensor(x.type.dtype, broadcastable)])

    def perform(self, node, inp, out_):
        (x,) = inp
        (out,) = out_
        ndim = self.ndim
        if ndim == 1:
            try:
                out[0] = x.reshape(x.size)
            except AttributeError:
                out[0] = x.reshape((np.prod(x.shape),))
        elif ndim == len(x.shape):
            out[0] = x
        else:
            newshape = x.shape[: ndim - 1] + (np.prod(x.shape[ndim - 1 :]),)
            out[0] = x.reshape(newshape)

    def infer_shape(self, fgraph, node, in_shapes):
        from aesara.tensor.math import prod

        (in_shp,) = in_shapes
        part1 = in_shp[: self.ndim - 1]
        part2 = in_shp[self.ndim - 1 :]

        if len(part2) > 1:
            part2 = (prod(part2, dtype="int64"),)
        elif len(part2) == 1:
            # We do not want to force an upcast of part2 if its length is 1
            pass
        else:
            if len(in_shp) == 0 and self.ndim == 1:
                part2 = (1,)
            else:
                raise ValueError(
                    f"invalid output ndimensions ({self.ndim}) for tensor "
                    f"of rank {len(in_shp)}"
                )

        out_shape = part1 + part2
        return [out_shape]

    def grad(self, inp, grads):
        (x,) = inp
        (g_out,) = grads
        return [reshape(g_out, shape(x), x.ndim)]

    def R_op(self, inputs, eval_points):
        if None in eval_points:
            return [None]
        return self.make_node(*eval_points).outputs

    def c_code_cache_version(self):
        return (1, 1)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (out,) = outputs
        ndim = self.ndim
        fail = sub["fail"]
        return (
            """
        if (%(ndim)s == PyArray_NDIM(%(x)s))
        {
            Py_XDECREF(%(out)s);
            Py_XINCREF(%(x)s);
            %(out)s = %(x)s;
        }
        else
        {
            Py_XDECREF(%(out)s);

            if (%(ndim)s == 1)
            {
                npy_intp size = PyArray_SIZE(%(x)s);
                PyArray_Dims newshape;
                newshape.ptr = &size;
                newshape.len = 1;
                %(out)s = (PyArrayObject*)PyArray_Newshape(%(x)s,
                                                           &newshape,
                                                           NPY_CORDER);
            }
            else
            {
                npy_intp *oldshape = PyArray_DIMS(%(x)s);
                npy_intp newshape_dims[%(ndim)s];

                int i;
                for (i = 0; i < %(ndim)s - 1; ++i)
                    newshape_dims[i] = oldshape[i];

                newshape_dims[i] = 1;

                for (int j = %(ndim)s - 1; j < PyArray_NDIM(%(x)s); ++j)
                    newshape_dims[i] *= oldshape[j];

                PyArray_Dims newshape;
                newshape.ptr = newshape_dims;
                newshape.len = %(ndim)s;
                %(out)s = (PyArrayObject*)PyArray_Newshape(%(x)s,
                                                           &newshape,
                                                           NPY_CORDER);
            }
        }
        if (!%(out)s)
        {
            //The error message should have been set by
            // PyArray_Newshape
            %(fail)s;
        }
        """
            % locals()
        )


def is_flat(
    var: TensorVariable, ndim: Optional[int] = None, outdim: Optional[int] = None
) -> bool:
    """
    Verifies the dimensionality of the var is equal to
    outdim. This method is usually called after flatten method on a
    variable, where the first outdim-1 dimension size(s) of the variable
    is kept intact, and the last dimension size of the variable is made
    equal to the multiplication of its remaining dimension size(s), such that
    the variable would end up with as many dimension as outdim.

    Parameters
    ----------
        var : aesara.tensor.var.TensorVariable
            the aesara var on which the dimensionality is checked.
        ndim : int or None

        outdim : int or None
            the expected dimensionality of var.

    Returns
    -------
    bool
        the comparison result of var's dim
        and the expected outdim.
    """
    if outdim is None and ndim is None:
        ndim = 1
    elif outdim is not None and ndim is not None:
        raise ValueError("You should only specify ndim")
    elif outdim is not None:
        warnings.warn("flatten outdim parameter is deprecated, use ndim instead.")
        ndim = outdim
    return var.ndim == ndim


def flatten(x: TensorVariable, ndim: int = 1):
    """Return a copy of the array collapsed into one dimension.

    Reshapes the variable `x` by keeping the first outdim-1 dimension size(s)
    of `x` the same, and making the last dimension size of `x` equal to the
    multiplication of its remaining dimension size(s).

    Parameters
    ----------
    x : aesara.tensor.var.TensorVariable
        The variable to be reshaped.
    ndim : int
        The number of dimensions of the returned variable
        The default value is ``1``.

    Returns
    -------
    aesara.tensor.var.TensorVariable
        the flattend variable with dimensionality of outdim
    """
    if ndim is None:
        ndim = 1

    # Any input variable can be flattened to have ndim of 1,
    # even if it's a scalar. Otherwise, ndim must be positive
    # and smaller than x.ndim.
    if ndim < 1 or (ndim > 1 and ndim > x.ndim):
        raise ValueError(f"ndim {ndim} out of bound [1, {x.ndim + 1})")

    if ndim > 1:
        dims = tuple(x.shape[: ndim - 1]) + (-1,)
    else:
        dims = (-1,)
    x_reshaped = x.reshape(dims)
    bcast_kept_dims = x.broadcastable[: ndim - 1]
    bcast_new_dim = builtins.all(x.broadcastable[ndim - 1 :])
    broadcastable = bcast_kept_dims + (bcast_new_dim,)
    x_reshaped = addbroadcast(x_reshaped, *[i for i in range(ndim) if broadcastable[i]])
    return x_reshaped


class Tile(Op):
    """
    Construct an array by repeating the input x according to reps pattern.

    .. note:: Deprecated
              Use tile() instead.

    Tiles its input according to reps. The length of reps is the number of
    dimension of x and contains the number of times to tile x in each
    dimension.

    See Also
    --------
    numpy.tile : http://docs.scipy.org/doc/numpy/reference/generated/numpy.tile.html

    """

    __props__ = ("ndim",)

    def __init__(self, ndim):
        self.ndim = ndim

    def __str__(self):
        return f"{self.__class__.__name__ }{{ndim={self.ndim}}}"

    def make_node(self, x, reps):
        warnings.warn(
            ("Tile op is deprecated, use tile function instead."), stacklevel=3
        )
        x = as_tensor_variable(x)
        reps = as_tensor_variable(reps)
        return Apply(self, [x, reps], [tensor(x.type.dtype, [False] * self.ndim)])

    def perform(self, node, inp, out_):
        x, reps = inp
        (out,) = out_
        res = np.tile(x, reps)
        if res.ndim != self.ndim:
            raise ValueError("Tile.perform produced incorrect number of dimensions")

        if (np.asarray(reps) == 1).all():
            # In that case, some NumPy version return a view!  As this
            # op isn't declared as inplace, we need to check that and
            # copy the data.
            if np.may_share_memory(res, x):
                res = res.copy()
        out[0] = res

    def infer_shape(self, fgraph, node, in_shapes):
        # Note: in contrast with numpy, it is assumed that x.shape and reps
        # have equal length;  see also tile function below

        # Note: if reps were to be allowed not to be a constant and x.shape
        # and reps to be unequal, the following block of code could be used:
        # prepend 1 to x.shape if needed
        # if self.ndim > x.ndim:
        # shp = concatenate(ones(self.ndim - x.ndim), shp)
        # prepend 1 to reps if needed
        # reps = concatenate(ones(self.ndim - reps.shape[0]), reps)

        x, reps = node.inputs
        shp = in_shapes[0]
        tiled_shp = shp * reps
        out_shape = []
        for i in range(self.ndim):
            out_shape.append(tiled_shp[i])
        return [out_shape]

    def grad(self, inp, grads):
        x, reps = inp
        (g_out,) = grads
        # return [tilegrad(x, reps, g_out), None]
        raise NotImplementedError()


def tile(x, reps: Union[int, List], ndim: Optional[int] = None):
    """
    Tile input array `x` according to `reps`.

    See the docstring of `numpy.tile` for details.

    'reps' can be constant integer (e.g. 3), constant vector(e.g. [2 3]),
    symbolic scalar (e.g. tensor.iscalar()), symbolic vector (e.g. tensor.ivector())
    or a list of symbolic scalar (e.g. [tensor.iscalar(), tensor.iscalar()]).

    ndim is the number of the dimensions of the output, if it is provided, ndim
    should be equal or larger than x.ndim and len(reps), otherwise, we will use
    max(x.ndim, len(reps)) as ndim. If reps is symbolic vector, the ndim has to
    be provided.

    """
    from aesara.tensor.math import ge

    if ndim is not None and ndim < x.ndim:
        raise ValueError("ndim should be equal or larger than x.ndim")

    # If reps is a scalar, integer or vector, we convert it to a list.
    if not isinstance(reps, (list, tuple)):
        reps_astensor = as_tensor_variable(reps)
        ndim_check = reps_astensor.ndim
        if reps_astensor.dtype not in discrete_dtypes:
            raise ValueError("elements of reps must be integer dtype")

        # The scalar/integer case
        if ndim_check == 0:
            reps = [reps]

        # The vector case
        elif ndim_check == 1:
            if ndim is None:
                raise ValueError(
                    "if reps is tensor.vector, you should specify " "the ndim"
                )
            else:
                offset = ndim - reps.shape[0]

                # assert that reps.shape[0] does not exceed ndim
                offset = assert_op(offset, ge(offset, 0))

                # if reps.ndim is less than x.ndim, we pad the reps with
                # "1" so that reps will have the same ndim as x.
                reps_ = [switch(i < offset, 1, reps[i - offset]) for i in range(ndim)]
                reps = reps_

        # For others, raise an error
        else:
            raise ValueError("the dimension of reps should not exceed 1")
    else:
        if ndim is not None and len(reps) > ndim:
            raise ValueError("len(reps) should be equal or less than ndim")
        if not np.all(
            [
                isinstance(r, int)
                or (isinstance(r, TensorVariable) and r.dtype in discrete_dtypes)
                for r in reps
            ]
        ):
            raise ValueError("elements of reps must be scalars of integer dtype")

    # If reps.ndim is less than x.ndim, we pad the reps with
    # "1" so that reps will have the same ndim as x
    reps = list(reps)
    if ndim is None:
        ndim = builtins.max(len(reps), x.ndim)
    if len(reps) < ndim:
        reps = [1] * (ndim - len(reps)) + reps

    _shape = [1] * (ndim - x.ndim) + [x.shape[i] for i in range(x.ndim)]
    alloc_shape = reps + _shape
    y = alloc(x, *alloc_shape)
    shuffle_ind = np.arange(ndim * 2).reshape(2, ndim)
    shuffle_ind = shuffle_ind.transpose().flatten()
    y = y.dimshuffle(*shuffle_ind)
    new_shapes = [sh * reps[i] for i, sh in enumerate(_shape)]
    y = y.reshape(new_shapes)

    return y


class ARange(Op):
    """Create an array containing evenly spaced values within a given interval.

    Parameters and behaviour are the same as numpy.arange().

    """

    __props__ = ("dtype",)

    def __init__(self, dtype):
        self.dtype = dtype

    def make_node(self, start, stop, step):
        start, stop, step = map(as_tensor_variable, (start, stop, step))
        assert start.ndim == 0
        assert stop.ndim == 0
        assert step.ndim == 0

        inputs = [start, stop, step]
        outputs = [tensor(self.dtype, (False,))]

        return Apply(self, inputs, outputs)

    @config.change_flags(warn_float64="ignore")
    def infer_shape(self, fgraph, node, i_shapes):
        from aesara.tensor.math import ceil, maximum

        # Note start, stop and step can be float numbers.
        start, stop, step = node.inputs

        def is_constant_value(var, value):
            try:
                v = get_scalar_constant_value(var)
                return np.all(v == value)
            except NotScalarConstantError:
                pass
            return False

        def upcast(var):
            if (
                var.dtype in integer_dtypes
                and
                # We do not want to cast uint64 to int64 as this can
                # loose information. If we upcast uint64 with int64,
                # this give float64. This is safer then checking for
                # uint64 in case we support [u]int128 or other in the
                # future.
                aes.upcast(var.dtype, "int64") == "int64"
            ):
                return cast(var, "int64")
            return var

        if is_constant_value(step, 1):
            if is_constant_value(start, 0):
                return [(cast(stop, "int64"),)]
            else:
                stop = upcast(stop)
                start = upcast(start)
                return [(maximum(cast(stop - start, "int64"), 0),)]
        else:
            stop = upcast(stop)
            start = upcast(start)
            return [
                (
                    maximum(
                        cast(ceil(cast((stop - start), "float64") / step), "int64"), 0
                    ),
                )
            ]

    def perform(self, node, inp, out_):
        start, stop, step = inp
        (out,) = out_
        start = start.item()
        stop = stop.item()
        step = step.item()
        out[0] = np.arange(start, stop, step, dtype=self.dtype)

    def connection_pattern(self, node):

        return [[True], [False], [True]]

    def L_op(self, inputs, outputs, grads):
        start, stop, step = inputs
        (gz,) = grads
        # `start` and `step` affect the output values
        # but the outputs are integers so there's
        # no gradient through them.
        # When they are not integers, the gradients are
        # as expressed below.
        # `stop` does not affect the output values,
        # just the output shape, so it is disconnected.

        if self.dtype in discrete_dtypes:
            return [
                start.zeros_like(dtype=config.floatX),
                DisconnectedType()(),
                step.zeros_like(dtype=config.floatX),
            ]
        else:
            num_steps_taken = outputs[0].shape[0]
            return [
                gz.sum(),
                DisconnectedType()(),
                (gz * arange(num_steps_taken, dtype=self.dtype)).sum(),
            ]

    def R_op(self, inputs, eval_points):
        return [None]


_arange = {}


def arange(start, stop=None, step=1, dtype=None):
    # If only one argument is provided, it is in fact the "stop" argument,
    # and start is 0.
    if stop is None:
        start, stop = 0, start

    start, stop, step = map(as_tensor_variable, (start, stop, step))
    # If dtype is not provided, infer it from the other arguments
    if dtype is None:
        dtype = aes.upcast(start.type.dtype, stop.type.dtype, step.type.dtype)
        # don't try to be stingy and byte-optimize, this leads to
        # overflow problems.
        if dtype in int_dtypes:
            dtype = "int64"
        if dtype in uint_dtypes:
            dtype = "uint64"
        if config.cast_policy in ("numpy", "numpy+floatX"):
            # We enforce numpy semantics, except in the special case where
            # `config.cast_policy` is 'numpy+floatX' and we want to use float32
            # rather than float64.
            # As an example, if `start`, `stop` and `step` are all int32,
            # `numpy.arange` returns an int64 array (on 64-bit platforms),
            # while the upcast above returns int32.
            numpy_dtype = np.arange(
                start=np.array(0, dtype=start.dtype),
                stop=np.array(1, dtype=stop.dtype),
                step=np.array(1, dtype=step.dtype),
            ).dtype
            if numpy_dtype != dtype:
                if (
                    config.cast_policy == "numpy+floatX"
                    and config.floatX == "float32"
                    and numpy_dtype == "float64"
                    and
                    # No explicit float64 in the three arguments?
                    builtins.all(
                        dt != "float64" for dt in [s.dtype for s in (start, stop, step)]
                    )
                ):
                    # We use float32 instead.
                    assert dtype != "float64"
                    dtype = "float32"
                else:
                    # We use the same dtype as numpy instead of the result of
                    # the upcast.
                    dtype = str(numpy_dtype)

    if dtype not in _arange:
        _arange[dtype] = ARange(dtype)
    return _arange[dtype](start, stop, step)


class _nd_grid:
    """Create a dense n-dimensional 'meshgrid' with equally spaced points.

    Used to create the instance ``mgrid`` and ``ogrid`` which act similarly
    to their numpy equivalents.

    Parameters
    ----------
    sparse : boolean, optional, default=True
        Specifying False leads to the equivalent of numpy's mgrid functionality.
        Specifying True leads to the equivalent of ogrid.

    Examples
    --------
    >>> a = aet.mgrid[0:5, 0:3]
    >>> a[0].eval()
    array([[0, 0, 0],
           [1, 1, 1],
           [2, 2, 2],
           [3, 3, 3],
           [4, 4, 4]], dtype=int8)
    >>> a[1].eval()
    array([[0, 1, 2],
           [0, 1, 2],
           [0, 1, 2],
           [0, 1, 2],
           [0, 1, 2]], dtype=int8)
    >>> b = aet.ogrid[0:5, 0:3]
    >>> b[0].eval()
    array([[0],
           [1],
           [2],
           [3],
           [4]], dtype=int8)
    >>> b[1].eval()
    array([[0, 1, 2, 3]], dtype=int8)

    """

    def __init__(self, sparse=False):
        self.sparse = sparse

    def __getitem__(self, *args):

        if isinstance(args[0], slice):
            sl = args[0]
            return arange(sl.start or 0, sl.stop, sl.step or 1)

        ndim = len(args[0])
        for sl in args[0]:
            if isinstance(sl.step, builtins.complex):
                raise NotImplementedError(
                    "Not implemented for slices " "whose step is complex"
                )
        ranges = [arange(sl.start or 0, sl.stop, sl.step or 1) for sl in args[0]]
        shapes = [
            tuple([1] * j + [r.shape[0]] + [1] * (ndim - 1 - j))
            for j, r in enumerate(ranges)
        ]
        ranges = [r.reshape(shape) for r, shape in zip(ranges, shapes)]
        if self.sparse:
            grids = ranges
        else:
            grids = []
            ones = [ones_like(r) for r in ranges]
            for i in range(ndim):
                grid = 1
                for j in range(ndim):
                    if j == i:
                        grid = grid * ranges[j]
                    else:
                        grid = grid * ones[j]
                grids.append(grid)
        return grids


mgrid = _nd_grid()
ogrid = _nd_grid(sparse=True)


class PermuteRowElements(Op):
    """Permute the elements of each row (inner-most dim) of a tensor.

    A permutation will be applied to every row (vector) of the input tensor x.
    Depending on the dimensionality of x and the permutation tensor y,
    different cases are possible.
    If y.ndim = 1, y is a single permutation, that will be applied to every
    vector of x. For instance, if x is a matrix, the same permutation will be
    applied to each row of x.
    If x.ndim = y.ndim, each row of x corresponds to a row of y, containing
    a permutation that will be applied to that row. For instance, if x and y
    are two matrices, a different permutation will be applied to each row of x.
    If x.ndim > y.ndim, y will be broadcasted to fit x, then each row (vector)
    of x will be reordered according to the corresponding row of y. (This is
    a generalization of the first case).
    If x.ndim = 1, every permutation in y will be applied to x, and the output
    will contain all the results.
    If x.ndim < y.ndim, x will be broadcasted to fit y, and different
    permutations contained in y will be applied to each vector in x. (This is
    a generalization of the previous case).

    If the "inverse" argument is True, the Op will perform the inverse
    permutation instead.
    """

    __props__ = ()

    def make_node(self, x, y, inverse):
        x = as_tensor_variable(x)
        y = as_tensor_variable(y)
        if inverse:  # as_tensor_variable does not accept booleans
            inverse = as_tensor_variable(1)
        else:
            inverse = as_tensor_variable(0)

        # y should contain integers
        assert y.type.dtype in integer_dtypes
        # Inverse should be an integer scalar
        assert inverse.type.ndim == 0 and inverse.type.dtype in integer_dtypes

        # Match shapes of x and y
        x_dim = x.type.ndim
        y_dim = y.type.ndim

        if x_dim > y_dim:
            y = shape_padleft(y, n_ones=(x_dim - y_dim))
        elif x_dim < y_dim:
            x = shape_padleft(x, n_ones=(y_dim - x_dim))

        # Compute the broadcastable pattern of the output
        out_broadcastable = [
            xb and yb for xb, yb in zip(x.type.broadcastable, y.type.broadcastable)
        ]
        out_type = tensor(dtype=x.type.dtype, broadcastable=out_broadcastable)

        inputlist = [x, y, inverse]
        outputlist = [out_type]
        return Apply(self, inputlist, outputlist)

    def _rec_perform(self, node, x, y, inverse, out, curdim):
        """Perform the permutation by doing a recursion over the input
        dimensions.

        For every dimension, starting with the leftmost, the right set of
        indices is determined (depending if broadcasting or not), then
        the function is recursively called on the appropriate subtensors.

        The terminal case is reached when the current tensors are vector,
        then the permutation contained in y is applied to x.

        Parameters
        ----------
        x: TensorVariable
            The input tensor, on which the permutation is applied.
        y: TensorVariable
            Tensor containing the permutations to apply.
        inverse: bool
            Whether to apply permutations or their inverse.
        out: TensorVariable
            Tensor storing the output result.
        curdim: int
            Counter of the current depth of recursion.

        """
        if len(x.shape) == 1:
            # Numpy advanced indexing works in this case
            if inverse:
                out[y] = x[:]
            else:
                out[:] = x[y]
        else:
            xs0 = x.shape[0]
            ys0 = y.shape[0]
            if xs0 == ys0:
                for i in range(xs0):
                    self._rec_perform(node, x[i], y[i], inverse, out[i], curdim + 1)
            elif ys0 == 1 and node.inputs[1].type.broadcastable[curdim]:
                # Broadcast y
                for i in range(xs0):
                    self._rec_perform(node, x[i], y[0], inverse, out[i], curdim + 1)
            elif xs0 == 1 and node.inputs[0].type.broadcastable[curdim]:
                # Broadcast x
                for i in range(ys0):
                    self._rec_perform(node, x[0], y[i], inverse, out[i], curdim + 1)
            else:
                raise ValueError(f"Dimension mismatch: {xs0}, {ys0}")

    def perform(self, node, inp, out):
        x, y, inverse = inp
        (outs,) = out
        x_s = x.shape
        y_s = y.shape
        assert len(x_s) == len(y_s)

        # Make sure the output is big enough
        out_s = []
        for xdim, ydim in zip(x_s, y_s):
            if xdim == ydim:
                outdim = xdim
            elif xdim == 1:
                outdim = ydim
            elif ydim == 1:
                outdim = xdim
            else:
                raise ValueError(f"Dimension mismatch: {xdim}, {ydim}")
            out_s.append(outdim)

        if outs[0] is None or outs[0].shape != out_s:
            outs[0] = np.empty(out_s, dtype=x.dtype)

        self._rec_perform(node, x, y, inverse, outs[0], curdim=0)

    def infer_shape(self, fgraph, node, in_shapes):
        from aesara.tensor.math import maximum

        shp_x = in_shapes[0]
        shp_y = in_shapes[1]
        assert len(shp_x) == len(shp_y)
        out_shape = []
        for i in range(len(shp_x)):
            out_shape.append(maximum(shp_x[i], shp_y[i]))
        return [out_shape]

    def grad(self, inp, grads):
        from aesara.tensor.math import Sum, eq

        x, y, inverse = inp
        (gz,) = grads
        # First, compute the gradient wrt the broadcasted x.
        # If 'inverse' is False (0), apply the inverse of y on gz.
        # Else, apply y on gz.
        gx = permute_row_elements(gz, y, eq(inverse, 0))

        # If x has been broadcasted along some axes, we need to sum
        # the gradient over these axes, but keep the dimension (as
        # broadcastable)
        broadcasted_dims = [
            dim
            for dim in range(gz.type.ndim)
            if x.type.broadcastable[dim] and not gz.type.broadcastable[dim]
        ]
        gx = Sum(axis=broadcasted_dims)(gx)

        # Sum(...) removed the dimensions in broadcasted_dims,
        # so we need to put them back.
        newdims = []
        i = 0
        for dim in range(gz.type.ndim):
            if dim in broadcasted_dims:
                newdims.append("x")
            else:
                newdims.append(i)
                i += 1

        gx = DimShuffle(gx.type.broadcastable, newdims)(gx)
        assert gx.type.broadcastable == x.type.broadcastable

        # if x is an integer type, then so is the output.
        # this means f(x+eps) = f(x) so the gradient with respect
        # to x is zero
        if x.type.dtype in discrete_dtypes:
            gx = x.zeros_like()

        # The elements of y and of inverse both affect the output,
        # so they are connected to the output,
        # and the transformation isn't defined if their values
        # are non-integer, so the gradient with respect to them is
        # undefined

        return [gx, grad_undefined(self, 1, y), grad_undefined(self, 1, inverse)]


_permute_row_elements = PermuteRowElements()


def permute_row_elements(x, y, inverse=0):
    return _permute_row_elements(x, y, inverse)


def inverse_permutation(perm):
    """Computes the inverse of permutations.

    Each row of input should contain a permutation of the first integers.

    """
    return permute_row_elements(
        arange(perm.shape[-1], dtype=perm.dtype), perm, inverse=True
    )


class ExtractDiag(Op):
    """
    Return specified diagonals.

    If x is 2-D, returns the diagonal of x with the given offset,
    i.e., the collection of elements of the form x[i, i+offset].
    If x has more than two dimensions, then the axes specified by
    axis1 and axis2 are used to determine the 2-D sub-array whose
    diagonal is returned. The shape of the resulting array can be
    determined by removing axis1 and axis2 and appending an index
    to the right equal to the size of the resulting diagonals.

    Parameters
    ----------
    x: A tensor variable with x.ndim >= 2.

    offset: Offset of the diagonal from the main diagonal.
        Can be positive or negative.
        Defaults to main diagonal (0).

    axis1: Axis to be used as the first axis of the 2-D
        sub-arrays from which the diagonals should be taken.
        Defaults to first axis (0).

    axis2: Axis to be used as the second axis of the 2-D
        sub-arrays from which the diagonals should be taken.
        Defaults to second axis (1).



    Returns
    -------
    array_of_diagonals:
        If x is 2-D, a 1-D array of the same type as a
        containing the diagonal is returned.
        If the dimension of x is greater than two, then an
        array of diagonals is returned, "packed" from left-most
        dimension to right-most (e.g., if x is 3-D, then the
        diagonals are "packed" along rows).



    Raises
    ------
    ValueError
        If the dimension of x is less than 2.


    See Also
    --------
    numpy.diagonal:
        https://docs.scipy.org/doc/numpy-dev/reference/generated/numpy.diagonal.html
    """

    __props__ = ("offset", "axis1", "axis2", "view")

    def __init__(self, offset=0, axis1=0, axis2=1, view=False):
        self.view = view
        if self.view:
            self.view_map = {0: [0]}
        self.offset = offset
        self.axis1 = axis1
        self.axis2 = axis2

    def make_node(self, x):
        x = as_tensor_variable(x)

        if x.ndim < 2:
            raise ValueError(
                "ExtractDiag needs an input with 2 or more " "dimensions", x
            )
        return Apply(
            self,
            [x],
            [x.type.__class__(dtype=x.dtype, broadcastable=[False] * (x.ndim - 1))()],
        )

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (z,) = outputs
        z[0] = x.diagonal(self.offset, self.axis1, self.axis2)
        if not self.view:
            z[0] = z[0].copy()

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout

        if x.ndim == 2:
            x = zeros_like(x)
            xdiag = AllocDiag(offset=self.offset)(gz)
            return [
                aesara.tensor.subtensor.set_subtensor(
                    x[: xdiag.shape[0], : xdiag.shape[1]], xdiag
                )
            ]
        else:
            warnings.warn("Gradient of ExtractDiag only works for matrices.")
            return [grad_not_implemented(self, 0, x)]

    def infer_shape(self, fgraph, node, shapes):
        from aesara.tensor.math import clip, minimum

        (in_shape,) = shapes
        dim1 = in_shape[self.axis1]
        dim2 = in_shape[self.axis2]
        out_shape = [
            d for i, d in enumerate(in_shape) if i not in (self.axis1, self.axis2)
        ]
        # The following logic is inspired by C code of PyArray_Diagonal().
        offset = self.offset
        if offset > 0:
            diag_size = clip(dim2 - offset, 0, dim1)
        elif offset < 0:
            diag_size = clip(dim1 + offset, 0, dim2)
        else:
            diag_size = minimum(dim1, dim2)
        out_shape.append(diag_size)
        return [tuple(out_shape)]

    def __setstate__(self, state):
        self.__dict__.update(state)

        if self.view:
            self.view_map = {0: [0]}

        if "offset" not in state:
            self.offset = 0
        if "axis1" not in state:
            self.axis1 = 0
        if "axis2" not in state:
            self.axis2 = 1


extract_diag = ExtractDiag()
# TODO: optimization to insert ExtractDiag with view=True


def diagonal(a, offset: int = 0, axis1: int = 0, axis2: int = 1):
    """
    A helper function for `ExtractDiag`. It accepts tensor with
    `ndim >= 2` as input. The name `diagonal` is just meant to keep it
    consistent with numpy.

    Parameters
    ----------
    a : symbolic tensor
    offset : int
        offset
    axis1 : int
    axis2 : int

    Returns
    -------
    tensor : symbolic tensor

    """
    return ExtractDiag(offset, axis1, axis2)(a)


class AllocDiag(Op):
    """An `Op` that copies a vector to the diagonal of an empty matrix.

    It does the inverse of `ExtractDiag`.
    """

    __props__ = ("offset", "axis1", "axis2")

    def __init__(self, offset: int = 0, axis1: int = 0, axis2: int = 1):
        """
        Parameters
        ----------
        offset: int
            Offset of the diagonal from the main diagonal defined by `axis1`
            and `axis2`. Can be positive or negative.  Defaults to main
            diagonal (i.e. 0).
        axis1: int
            Axis to be used as the first axis of the 2-D sub-arrays to which
            the diagonals will be allocated.  Defaults to first axis (i.e. 0).
        axis2: int
            Axis to be used as the second axis of the 2-D sub-arrays to which
            the diagonals will be allocated.  Defaults to second axis (i.e. 1).
        """
        self.offset = offset
        self.axis1 = axis1
        self.axis2 = axis2

    def make_node(self, diag):
        diag = as_tensor_variable(diag)
        if diag.type.ndim < 1:
            raise ValueError(
                "AllocDiag needs an input with 1 or more " "dimensions", diag.type
            )
        return Apply(
            self,
            [diag],
            [
                diag.type.__class__(
                    dtype=diag.dtype, broadcastable=[False] * (diag.ndim + 1)
                )()
            ],
        )

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (z,) = outputs

        axis1 = np.minimum(self.axis1, self.axis2)
        axis2 = np.maximum(self.axis1, self.axis2)
        offset = self.offset

        # Create array with one extra dimension for resulting matrix
        result_shape = x.shape[:-1] + (x.shape[-1] + abs(offset),) * 2
        result = np.zeros(result_shape, dtype=x.dtype)

        # Create slice for diagonal in final 2 axes
        idxs = np.arange(x.shape[-1])
        diagonal_slice = (len(result_shape) - 2) * [slice(None)] + [
            idxs + np.maximum(0, -offset),
            idxs + np.maximum(0, offset),
        ]

        # Fill in final 2 axes with x
        result[tuple(diagonal_slice)] = x

        if len(x.shape) > 1:
            # Re-order axes so they correspond to diagonals at axis1, axis2
            axes = list(range(len(x.shape[:-1])))
            last_idx = axes[-1]
            axes = axes[:axis1] + [last_idx + 1] + axes[axis1:]
            axes = axes[:axis2] + [last_idx + 2] + axes[axis2:]
            result = result.transpose(axes)

        z[0] = result

    def grad(self, inputs, gout):
        (gz,) = gout
        return [diagonal(gz, offset=self.offset, axis1=self.axis1, axis2=self.axis2)]

    def infer_shape(self, fgraph, nodes, shapes):
        (x_shape,) = shapes
        axis1 = np.minimum(self.axis1, self.axis2)
        axis2 = np.maximum(self.axis1, self.axis2)

        result_shape = list(x_shape[:-1])
        diag_shape = x_shape[-1] + abs(self.offset)
        result_shape = result_shape[:axis1] + [diag_shape] + result_shape[axis1:]
        result_shape = result_shape[:axis2] + [diag_shape] + result_shape[axis2:]
        return [tuple(result_shape)]

    def __setstate__(self, state):
        if "view_map" in state:
            del state["view_map"]

        self.__dict__.update(state)

        if "offset" not in state:
            self.offset = 0
        if "axis1" not in state:
            self.axis1 = 0
        if "axis2" not in state:
            self.axis2 = 1


def diag(v, k: int = 0):
    """
    A helper function for two ops: `ExtractDiag` and
    `AllocDiag`. The name `diag` is meant to keep it consistent
    with numpy. It both accepts tensor vector and tensor matrix.
    While the passed tensor variable `v` has `v.ndim>=2`, it builds a
    `ExtractDiag` instance, and returns a vector with its entries equal to
    `v`'s main diagonal; otherwise if `v.ndim` is `1`, it builds an `AllocDiag`
    instance, and returns a matrix with `v` at its k-th diaogonal.

    Parameters
    ----------
    v : symbolic tensor
    k : int
        offset

    Returns
    -------
    tensor : symbolic tensor

    """

    if v.ndim == 1:
        return AllocDiag(k)(v)
    elif v.ndim >= 2:
        return diagonal(v, offset=k)
    else:
        raise ValueError("Input must has v.ndim >= 1.")


def stacklists(arg):
    """
    Recursively stack lists of tensors to maintain similar structure.

    This function can create a tensor from a shaped list of scalars:

    Examples
    --------
    >>> from aesara.tensor import stacklists
    >>> from aesara.tensor.type import scalars, matrices
    >>> from aesara import function
    >>> a, b, c, d = scalars('abcd')
    >>> X = stacklists([[a, b], [c, d]])
    >>> f = function([a, b, c, d], X)
    >>> f(1, 2, 3, 4)
    array([[ 1.,  2.],
           [ 3.,  4.]], dtype=float32)

    We can also stack arbitrarily shaped tensors. Here we stack matrices into
    a 2 by 2 grid:

    >>> from numpy import ones
    >>> a, b, c, d = matrices('abcd')
    >>> X = stacklists([[a, b], [c, d]])
    >>> f = function([a, b, c, d], X)
    >>> x = ones((4, 4), 'float32')
    >>> f(x, x, x, x).shape
    (2, 2, 4, 4)

    """
    if isinstance(arg, (tuple, list)):
        return stack(list(map(stacklists, arg)))
    else:
        return arg


def swapaxes(y, axis1, axis2):
    "Swap the axes of a tensor."
    y = as_tensor_variable(y)
    ndim = y.ndim
    li = list(range(0, ndim))
    li[axis1], li[axis2] = li[axis2], li[axis1]
    return y.dimshuffle(li)


def choose(a, choices, out=None, mode: str = "raise"):
    """
    Construct an array from an index array and a set of arrays to choose from.

    First of all, if confused or uncertain, definitely look at the Examples -
    in its full generality, this function is less simple than it might seem
    from the following code description (below ndi = numpy.lib.index_tricks):

    np.choose(a,c) == np.array([c[a[I]][I] for I in ndi.ndindex(a.shape)]).

    But this omits some subtleties. Here is a fully general summary:

    Given an ``index`` array (a) of integers and a sequence of n arrays
    (choices), a and each choice array are first broadcast, as necessary,
    to arrays of a common shape; calling these Ba and
    Bchoices[i], i = 0,...,n-1 we have that, necessarily,
    Ba.shape == Bchoices[i].shape for each i.
    Then, a new array with shape Ba.shape is created as follows:

    - if mode=raise (the default), then, first of all, each element of a
      (and thus Ba) must be in the range [0, n-1]; now, suppose that
      i (in that range) is the value at the (j0, j1, ..., jm) position in Ba -
      then the value at the same position in the new array is the value in
      Bchoices[i] at that same position;

    - if mode=wrap, values in a (and thus Ba) may be any (signed) integer;
      modular arithmetic is used to map integers outside the range [0, n-1]
      back into that range; and then the new array is constructed as above;

    - if mode=clip, values in a (and thus Ba) may be any (signed) integer;
      negative integers are mapped to 0; values greater than n-1 are mapped
      to n-1; and then the new array is constructed as above.

    Parameters
    ----------
    a : int array
        This array must contain integers in [0, n-1], where n is the number of
        choices, unless mode=wrap or mode=clip, in which cases any integers
        are permissible.
    choices : sequence of arrays
        Choice arrays. a and all of the choices must be broadcastable to
        the same shape. If choices is itself an array (not recommended),
        then its outermost dimension (i.e., the one corresponding to
        choices.shape[0]) is taken as defining the ``sequence``.
    out : array, optional
        If provided, the result will be inserted into this array.
        It should be of the appropriate shape and dtype.
    mode : {``raise`` (default), ``wrap``, ``clip``}, optional
        Specifies how indices outside [0, n-1] will be treated:
        ``raise`` : an exception is raised
        ``wrap`` : value becomes value mod n
        ``clip`` : values < 0 are mapped to 0, values > n-1 are mapped to n-1

    Returns
    -------
    merged_array - array
        The merged result.

    Raises
    ------
    ValueError - shape mismatch
        If a and each choice array are not all broadcastable to the same shape.

    """
    # This is done to keep the same function signature then NumPy.
    assert out is None
    return Choose(mode)(a, choices)


class Choose(Op):
    __props__ = ("mode",)

    def __init__(self, mode):
        assert mode in ("raise", "wrap", "clip")
        self.mode = mode

    def infer_shape(self, fgraph, node, shapes):

        a_shape, choices_shape = shapes
        out_shape = aesara.tensor.extra_ops.broadcast_shape(
            a_shape, choices_shape[1:], arrays_are_shapes=True
        )

        return [out_shape]

    def make_node(self, a, choices):
        # Import here as it isn't imported by default and we can't
        # import at the top as it would cause circular import.
        import aesara.typed_list

        a = as_tensor_variable(a)
        if a.dtype not in discrete_dtypes:
            raise TypeError(
                f"choose first argument must have an [u]int* dtype. Got {a.dtype}."
            )

        # Only use make_list if choices have inconsistent shapes
        # otherwise use as_tensor_variable
        if isinstance(choices, (tuple, list)):
            choice = aesara.typed_list.make_list(choices)
        else:
            choice = as_tensor_variable(choices)
        (out_shape,) = self.infer_shape(
            None, None, [tuple(a.shape), tuple(shape(choice))]
        )

        bcast = []
        for s in out_shape:
            try:
                s_val = aesara.get_scalar_constant_value(s)
            except (NotScalarConstantError, AttributeError):
                s_val = None

            if s_val == 1:
                bcast.append(True)
            else:
                bcast.append(False)

        o = TensorType(choice.dtype, bcast)
        return Apply(self, [a, choice], [o()])

    def perform(self, node, inputs, outputs):
        (z,) = outputs
        a = inputs[0]
        choice = inputs[1]
        # TODO reuse out?
        z[0] = np.choose(a, choice, mode=self.mode)


class AllocEmpty(COp):
    """Implement Alloc on the cpu, but without initializing memory."""

    __props__: Union[Tuple[str], Tuple[str, str]] = ("dtype",)
    params_type = ParamsType(typecode=int32)

    # specify the type of the data
    def __init__(self, dtype):
        assert isinstance(dtype, str), dtype
        self.dtype = dtype.lower()

    @property
    def typecode(self):
        return np.dtype(self.dtype).num

    def make_node(self, *_shape):
        _shape, bcast = alloc_validate_shape(_shape)
        otype = TensorType(dtype=self.dtype, broadcastable=bcast)
        output = otype()

        output.tag.values_eq_approx = values_eq_approx_always_true
        # The outut can contain nan/inf.  output.type is a new
        # instance, so we can do this only for that variable.
        output.type.filter_checks_isfinite = False

        # We can't reuse filter_checks_isfinite as by default it is
        # False and it is set to true only in DebugMode.
        # We can't set it in the type as other make_node can reuse the type.
        # We can't set it in the variable as it isn't copied when we copy
        # the variale. So we set it in the tag.
        output.tag.nan_guard_mode_check = False
        return Apply(self, _shape, [output])

    def debug_perform(self, node, inputs, out_, params):
        self.perform(node, inputs, out_, params)
        out_[0][0].fill(-123456789)

    def perform(self, node, inputs, out_, params):
        (out,) = out_
        sh = tuple([int(i) for i in inputs])
        if out[0] is None or out[0].shape != sh:
            out[0] = np.empty(sh, dtype=self.dtype)

    def c_code(self, node, name, inputs, out_, sub):
        (out,) = out_
        fail = sub["fail"]
        shps = inputs
        nd = len(shps)
        params = sub["params"]
        str = f"npy_intp dims[{nd}];\n"
        for idx, sh in enumerate(shps):
            str += (
                "dims[%(idx)s] ="
                "((npy_intp)((dtype_%(sh)s*)"
                " PyArray_DATA(%(sh)s))[0]);\n" % locals()
            )

        # Validate that the output storage exists
        str += f"if({out}==NULL\n"
        for idx, sh in enumerate(shps):
            str += f"||PyArray_DIMS({out})[{idx}]!=dims[{idx}]"

        str += (
            """){
            /* Reference received to invalid output variable.
            Decrease received reference's ref count and allocate new
            output variable */
            Py_XDECREF(%(out)s);
            %(out)s = (PyArrayObject*)PyArray_EMPTY(%(nd)s,
                                                    dims,
                                                    %(params)s->typecode,
                                                    0);
            if (!%(out)s)
            {
                PyErr_SetString(PyExc_MemoryError, "alloc failed");
                %(fail)s;
            }
        }
        """
            % locals()
        )
        return str

    def infer_shape(self, fgraph, node, input_shapes):
        return [node.inputs]

    def c_code_cache_version(self):
        return (4,)

    def do_constant_folding(self, fgraph, node):
        return False

    def connection_pattern(self, node):
        return [[False] for i in node.inputs]

    def grad(self, inputs, grads):
        return [DisconnectedType()() for i in inputs]

    def R_op(self, inputs, eval_points):
        return [zeros(inputs, self.dtype)]


def empty(shape, dtype=None):
    """Return a new array of given shape and type, without initializing entries.

    See ``numpy.empty``.

    Parameters
    ----------
    shape : int or tuple of int
        Shape of the empty array, e.g., ``(2, 3)`` or ``2``.
    dtype : data-type, optional
        Desired output data-type for the array, e.g, `numpy.int8`. Default is
        `numpy.float64`.
    """
    if dtype is None:
        dtype = config.floatX
    return AllocEmpty(dtype)(*shape)


__all__ = [
    "choose",
    "swapaxes",
    "stacklists",
    "diag",
    "diagonal",
    "inverse_permutation",
    "permute_row_elements",
    "mgrid",
    "ogrid",
    "arange",
    "tile",
    "flatten",
    "is_flat",
    "vertical_stack",
    "horizontal_stack",
    "get_vector_length",
    "concatenate",
    "stack",
    "roll",
    "join",
    "patternbroadcast",
    "unbroadcast",
    "addbroadcast",
    "split",
    "transpose",
    "extract_constant",
    "default",
    "tensor_copy",
    "transfer",
    "alloc",
    "alloc_validate_shape",
    "identity_like",
    "eye",
    "triu",
    "tril",
    "tri",
    "nonzero_values",
    "flatnonzero",
    "nonzero",
    "ones",
    "zeros",
    "zeros_like",
    "ones_like",
    "fill",
    "second",
    "where",
    "switch",
    "cast",
    "scalar_from_tensor",
    "tensor_from_scalar",
    "get_scalar_constant_value",
    "constant",
    "as_tensor_variable",
    "as_tensor",
    "extract_diag",
    "full",
    "empty",
]
