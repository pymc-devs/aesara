import builtins
import operator
import pickle
import warnings
from copy import copy
from functools import reduce
from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import aesara.scalar as aes
from aesara.compile.debugmode import DebugMode
from aesara.compile.function import function
from aesara.compile.mode import get_default_mode
from aesara.compile.sharedvalue import shared
from aesara.configdefaults import config
from aesara.gradient import NullTypeGradError, grad, numeric_grad
from aesara.graph.basic import Variable, applys_between
from aesara.graph.fg import FunctionGraph
from aesara.link.c.basic import DualLinker
from aesara.misc.safe_asarray import _asarray
from aesara.tensor import blas, blas_c
from aesara.tensor.basic import (
    as_tensor_variable,
    constant,
    eye,
    get_scalar_constant_value,
    switch,
)
from aesara.tensor.elemwise import CAReduce, Elemwise
from aesara.tensor.math import (
    Argmax,
    Dot,
    MaxAndArgmax,
    Mean,
    Prod,
    ProdWithoutZeros,
    Sum,
    _dot,
    abs_,
    add,
    allclose,
    arccos,
    arccosh,
    arcsin,
    arcsinh,
    arctan,
    arctan2,
    arctanh,
    argmax,
    argmin,
    ceil,
    ceil_intdiv,
    clip,
    complex_from_polar,
    conj,
    cos,
    cosh,
    cov,
    deg2rad,
    dense_dot,
    dot,
    eq,
    exp,
    exp2,
    expm1,
    floor,
    inv,
    isclose,
    isinf,
    isnan,
    isnan_,
    log,
    log1p,
    log2,
    log10,
    max,
    max_and_argmax,
    maximum,
    mean,
    min,
    minimum,
    mod,
    mul,
    neg,
    neq,
    outer,
    power,
    ptp,
    rad2deg,
    round_half_away_from_zero,
    round_half_to_even,
    sgn,
    sigmoid,
    sin,
    sinh,
    smallest,
    sqr,
    sqrt,
    sub,
)
from aesara.tensor.math import sum as aet_sum
from aesara.tensor.math import tan, tanh, tensordot, true_div, trunc, var
from aesara.tensor.type import (
    TensorType,
    complex_dtypes,
    continuous_dtypes,
    cscalar,
    discrete_dtypes,
    dmatrix,
    dscalar,
    dtensor3,
    dvector,
    fmatrix,
    fscalar,
    fscalars,
    imatrix,
    iscalar,
    ivector,
    lscalar,
    matrices,
    matrix,
    scalar,
    tensor,
    tensor3,
    tensor4,
    values_eq_approx_remove_nan,
    vector,
    vectors,
    zvector,
)
from aesara.tensor.type_other import NoneConst
from tests import unittest_tools as utt
from tests.tensor.utils import (
    _bad_build_broadcast_binary_normal,
    _bad_runtime_broadcast_binary_normal,
    _bad_runtime_inv,
    _eps,
    _good_broadcast_binary_arctan2,
    _good_broadcast_binary_normal,
    _good_broadcast_div_mod_normal_float,
    _good_broadcast_div_mod_normal_float_no_complex,
    _good_broadcast_pow_normal_float,
    _good_broadcast_unary_arccosh,
    _good_broadcast_unary_arcsin,
    _good_broadcast_unary_arctanh,
    _good_broadcast_unary_normal,
    _good_broadcast_unary_normal_float_no_complex,
    _good_broadcast_unary_normal_float_no_empty_no_complex,
    _good_broadcast_unary_normal_no_complex,
    _good_broadcast_unary_positive,
    _good_broadcast_unary_tan,
    _good_broadcast_unary_wide,
    _good_inv,
    _grad_broadcast_binary_normal,
    _grad_broadcast_pow_normal,
    _grad_broadcast_unary_normal,
    _grad_broadcast_unary_normal_no_complex,
    _grad_broadcast_unary_normal_no_complex_no_corner_case,
    _grad_broadcast_unary_normal_noint,
    _grad_inv,
    _numpy_true_div,
    angle_eps,
    check_floatX,
    copymod,
    div_grad_rtol,
    eval_outputs,
    get_numeric_types,
    ignore_isfinite_mode,
    inplace_func,
    makeBroadcastTester,
    makeTester,
    rand,
    rand_nonzero,
    rand_ranged,
    randcomplex,
    randint,
    randuint32,
    upcast_float16_ufunc,
    upcast_int8_nfunc,
)


if config.mode == "FAST_COMPILE":
    mode_opt = "FAST_RUN"
else:
    mode_opt = get_default_mode()


TestAddBroadcast = makeBroadcastTester(
    op=add,
    expected=lambda *inputs: check_floatX(inputs, reduce(lambda x, y: x + y, inputs)),
    good=dict(
        three_inputs_same_shapes=(rand(2, 3), rand(2, 3), rand(2, 3)),
        three_inputs_same_shapes_uint=(
            randuint32(2, 3),
            randuint32(2, 3),
            randuint32(2, 3),
        ),
        four_inputs_broadcast=(rand(2, 3), rand(1, 3), rand(2, 1), rand(1, 1)),
        **_good_broadcast_binary_normal,
    ),
    bad_build=_bad_build_broadcast_binary_normal,
    bad_runtime=_bad_runtime_broadcast_binary_normal,
)


TestSubBroadcast = makeBroadcastTester(
    op=sub,
    expected=lambda x, y: check_floatX((x, y), x - y),
    good=_good_broadcast_binary_normal,
    bad_build=_bad_build_broadcast_binary_normal,
    bad_runtime=_bad_runtime_broadcast_binary_normal,
    grad=_grad_broadcast_binary_normal,
)


TestMaximumBroadcast = makeBroadcastTester(
    op=maximum,
    expected=lambda *inputs: check_floatX(inputs, np.maximum(*inputs)),
    good=_good_broadcast_binary_normal,
    bad_build=_bad_build_broadcast_binary_normal,
    bad_runtime=_bad_runtime_broadcast_binary_normal,
    grad=_grad_broadcast_binary_normal,
)


def test_maximum_minimum_grad():
    # Test the discontinuity point.
    # We decided that we only pass the gradient to the first input in that case.
    x, y = vectors("xy")
    for op in [maximum, minimum]:
        o = op(x, y)
        g = grad(o.sum(), [x, y])

        f = function([x, y], g)
        assert np.allclose(f([1], [1]), [[1], [0]])


TestMinimumBroadcast = makeBroadcastTester(
    op=minimum,
    expected=lambda *inputs: check_floatX(inputs, np.minimum(*inputs)),
    good=_good_broadcast_binary_normal,
    bad_build=_bad_build_broadcast_binary_normal,
    bad_runtime=_bad_runtime_broadcast_binary_normal,
    grad=_grad_broadcast_binary_normal,
)

TestMulBroadcast = makeBroadcastTester(
    op=mul,
    expected=lambda *inputs: check_floatX(inputs, reduce(lambda x, y: x * y, inputs)),
    good=dict(
        three_inputs_same_shapes=(rand(2, 3), rand(2, 3), rand(2, 3)),
        four_inputs_broadcast=(rand(2, 3), rand(1, 3), rand(2, 1), rand(1, 1)),
        **_good_broadcast_binary_normal,
    ),
    bad_build=_bad_build_broadcast_binary_normal,
    bad_runtime=_bad_runtime_broadcast_binary_normal,
    grad=dict(
        three_inputs_same_shapes=(rand(2, 3), rand(2, 3), rand(2, 3)),
        four_inputs_broadcast=(rand(2, 3), rand(1, 3), rand(2, 1), rand(1, 1)),
        **_grad_broadcast_binary_normal,
    ),
)

# Values are fixed, because the gradient evaluation in TestModBroadcast often
# fails when the inputs are close to each other (due to gradient discontinuity).
# fmt: off
_grad_broadcast_div_mod_normal = dict(
    same_shapes=(
        np.array([[-0.51157823, 0.02560825, -0.7482302], [0.05923786, -0.21001006, -0.66742722]]),
        np.array([[-0.02250197, -0.32979461, 0.32081774], [0.36419213, -0.54073201, 0.8932643]])
    ),
    scalar=(
        np.array([[0.32390696, -0.77305276, -0.66302977], [0.8214372, -0.31612823, -0.06294127]]),
        np.array([[-0.86904352]])
    ),
    row=(
        np.array([[0.89763688, -0.09403658, 0.05847774], [-0.00694876, -0.08999577, 0.19857154]]),
        np.array([[-0.47662978, 0.72692131, -0.18250251]])
    ),
    column=(
        np.array([[0.04506636, 0.05725927, -0.94947897], [0.39868416, -0.12655465, -0.87068554]]),
        np.array([[-0.39040176], [0.76164576]])
    ),
    # same_shapes=(rand(2, 3), rand_nonzero((2, 3))),
    # scalar=(rand(2, 3), rand_nonzero((1, 1))),
    # row=(rand(2, 3), rand_nonzero((1, 3))),
    # column=(rand(2, 3), rand_nonzero((2, 1))),
    # complex1=(randcomplex(2, 3), randcomplex_nonzero((2, 3))),
    # complex2=(randcomplex(2, 3), rand_nonzero((2, 3))),
    # complex3=(rand(2, 3), randcomplex_nonzero((2, 3))),
    # dtype_mixup_1=(rand(2, 3), randint_nonzero(2, 3)),
    # dtype_mixup_2=(randint_nonzero(2, 3), rand_nonzero((2, 3))),
    # empty1=(np.asarray([]), np.asarray([1.])),
    # empty2=(np.asarray([0]), np.asarray([])),
)
# fmt: on

TestTrueDivBroadcast = makeBroadcastTester(
    op=true_div,
    expected=_numpy_true_div,
    good=_good_broadcast_div_mod_normal_float_no_complex,
    grad=_grad_broadcast_div_mod_normal,
    grad_rtol=div_grad_rtol,
)

TestInvBroadcast = makeBroadcastTester(
    op=inv,
    expected=lambda x: upcast_int8_nfunc(np.true_divide)(np.int8(1), x),
    good=_good_inv,
    bad_runtime=_bad_runtime_inv,
    grad=_grad_inv,
    grad_rtol=div_grad_rtol,
)

TestCeilIntDivBroadcast = makeBroadcastTester(
    op=ceil_intdiv,
    expected=lambda x, y: check_floatX((x, y), (x // y) + ((x % y) != 0)),
    good=_good_broadcast_div_mod_normal_float_no_complex,
    name="CeilIntDiv",
    # As we implement this function with neq, the gradient returned is always 0.
    # grad=_grad_broadcast_div_mod_normal,
    # grad_rtol=div_grad_rtol,
)

TestModBroadcast = makeBroadcastTester(
    op=mod,
    expected=lambda x, y: np.asarray(x % y, dtype=aes.upcast(x.dtype, y.dtype)),
    good=copymod(_good_broadcast_div_mod_normal_float, ["complex1", "complex2"]),
    grad=_grad_broadcast_div_mod_normal,
    grad_eps=1e-5,
)

# Disable NAN checking for pow operator per issue #1780
TestPowBroadcast = makeBroadcastTester(
    op=pow,
    expected=lambda x, y: check_floatX((x, y), x ** y),
    good=_good_broadcast_pow_normal_float,
    grad=_grad_broadcast_pow_normal,
    name="Pow",
    mode=ignore_isfinite_mode,
)

TestAbsBroadcast = makeBroadcastTester(
    op=abs_,
    expected=lambda x: abs(x),
    good=_good_broadcast_unary_normal,
    grad=_grad_broadcast_unary_normal,
)

TestNegBroadcast = makeBroadcastTester(
    op=neg,
    expected=lambda x: -x,
    good=_good_broadcast_unary_normal,
    grad=_grad_broadcast_unary_normal,
)

TestSgnBroadcast = makeBroadcastTester(
    op=sgn,
    expected=np.sign,
    good=_good_broadcast_unary_normal_no_complex,
    grad=_grad_broadcast_unary_normal,
)

TestCeilBroadcast = makeBroadcastTester(
    op=ceil,
    expected=upcast_float16_ufunc(np.ceil),
    good=_good_broadcast_unary_normal_no_complex,
    grad=copymod(
        _grad_broadcast_unary_normal_noint,
        extra=[np.asarray([-2.5, -1.5, -1.51, 0.49, 0.98, 1.02], dtype=config.floatX)],
    ),
)

TestFloorBroadcast = makeBroadcastTester(
    op=floor,
    expected=upcast_float16_ufunc(np.floor),
    good=_good_broadcast_unary_normal_no_complex,
    grad=_grad_broadcast_unary_normal_noint,
)

TestTruncBroadcast = makeBroadcastTester(
    op=trunc,
    expected=upcast_float16_ufunc(np.trunc),
    good=_good_broadcast_unary_normal_no_complex,
)

TestRoundHalfToEvenBroadcast = makeBroadcastTester(
    op=round_half_to_even,
    expected=np.round,
    good=_good_broadcast_unary_normal_float_no_complex,
    grad=_grad_broadcast_unary_normal_no_complex_no_corner_case,
)

# np.vectorize don't handle correctly empty ndarray.
# see in their file numpy/lib/function_base.py in class vectorize.__call__
# This happen in float32 mode.
TestRoundHalfAwayFromZeroBroadcast = makeBroadcastTester(
    op=round_half_away_from_zero,
    expected=lambda a: aes.round_half_away_from_zero_vec(a),
    good=_good_broadcast_unary_normal_float_no_empty_no_complex,
    grad=_grad_broadcast_unary_normal_no_complex_no_corner_case,
)

TestSqrBroadcast = makeBroadcastTester(
    op=sqr,
    expected=np.square,
    good=_good_broadcast_unary_normal,
    grad=_grad_broadcast_unary_normal,
)

TestExpBroadcast = makeBroadcastTester(
    op=exp,
    expected=upcast_float16_ufunc(np.exp),
    good=dict(
        _good_broadcast_unary_normal,
        int8=[np.arange(-127, 89, dtype="int8")],
        uint8=[np.arange(0, 89, dtype="uint8")],
        uint16=[np.arange(0, 89, dtype="uint16")],
    ),
    grad=_grad_broadcast_unary_normal,
)

TestExp2Broadcast = makeBroadcastTester(
    op=exp2,
    expected=upcast_float16_ufunc(np.exp2),
    good=_good_broadcast_unary_normal,
    grad=_grad_broadcast_unary_normal,
)

TestExpm1Broadcast = makeBroadcastTester(
    op=expm1,
    expected=upcast_float16_ufunc(np.expm1),
    good=dict(
        _good_broadcast_unary_normal,
        int8=[np.arange(-127, 89, dtype="int8")],
        uint8=[np.arange(0, 89, dtype="uint8")],
        uint16=[np.arange(0, 89, dtype="uint16")],
    ),
    grad=_grad_broadcast_unary_normal,
)


_grad_broadcast_unary_positive = dict(
    normal=(rand_ranged(_eps, 5, (2, 3)),),
)

TestLogBroadcast = makeBroadcastTester(
    op=log,
    expected=upcast_float16_ufunc(np.log),
    good=_good_broadcast_unary_positive,
    grad=_grad_broadcast_unary_positive,
)

TestLog2Broadcast = makeBroadcastTester(
    op=log2,
    expected=upcast_float16_ufunc(np.log2),
    good=_good_broadcast_unary_positive,
    grad=_grad_broadcast_unary_positive,
)

TestLog10Broadcast = makeBroadcastTester(
    op=log10,
    expected=upcast_float16_ufunc(np.log10),
    good=_good_broadcast_unary_positive,
    grad=_grad_broadcast_unary_positive,
)

TestLog1pBroadcast = makeBroadcastTester(
    op=log1p,
    expected=upcast_float16_ufunc(np.log1p),
    good=_good_broadcast_unary_positive,
    grad=_grad_broadcast_unary_positive,
)

TestSqrtBroadcast = makeBroadcastTester(
    op=sqrt,
    expected=upcast_float16_ufunc(np.sqrt),
    good=_good_broadcast_unary_positive,
    grad=_grad_broadcast_unary_positive,
)

_grad_broadcast_unary_wide = dict(
    normal=(rand_ranged(-1000, 1000, (2, 3)),),
)

TestDeg2radBroadcast = makeBroadcastTester(
    op=deg2rad,
    expected=upcast_float16_ufunc(np.deg2rad),
    good=_good_broadcast_unary_normal_no_complex,
    grad=_grad_broadcast_unary_normal_no_complex,
    eps=angle_eps,
)

TestRad2degBroadcast = makeBroadcastTester(
    op=rad2deg,
    expected=upcast_float16_ufunc(np.rad2deg),
    good=_good_broadcast_unary_normal_no_complex,
    grad=_grad_broadcast_unary_normal_no_complex,
    eps=angle_eps,
)

TestSinBroadcast = makeBroadcastTester(
    op=sin,
    expected=upcast_float16_ufunc(np.sin),
    good=_good_broadcast_unary_wide,
    grad=_grad_broadcast_unary_wide,
)

# The actual range is [-1, 1] but the numerical gradient is too
# unstable near those values
_grad_broadcast_unary_arcsin = dict(
    normal=(rand_ranged(-0.9, 0.9, (2, 3)),),
)

TestArcsinBroadcast = makeBroadcastTester(
    op=arcsin,
    expected=upcast_float16_ufunc(np.arcsin),
    good=_good_broadcast_unary_arcsin,
    grad=_grad_broadcast_unary_arcsin,
)

TestCosBroadcast = makeBroadcastTester(
    op=cos,
    expected=upcast_float16_ufunc(np.cos),
    good=_good_broadcast_unary_wide,
    grad=_grad_broadcast_unary_wide,
)


def test_py_c_match():
    a = TensorType(dtype="int8", broadcastable=(False,))()
    f = function([a], arccos(a), mode="DebugMode")
    # This can fail in DebugMode
    f(np.asarray([1, 0, -1], dtype="int8"))


TestArccosBroadcast = makeBroadcastTester(
    op=arccos,
    expected=upcast_float16_ufunc(np.arccos),
    good=_good_broadcast_unary_arcsin,
    grad=_grad_broadcast_unary_arcsin,
)

# We do not want to test around the discontinuity.
_grad_broadcast_unary_tan = dict(
    normal=(rand_ranged(-1.5, 1.5, (2, 3)),), shifted=(rand_ranged(1.6, 4.6, (2, 3)),)
)

TestTanBroadcast = makeBroadcastTester(
    op=tan,
    expected=upcast_float16_ufunc(np.tan),
    good=_good_broadcast_unary_tan,
    grad=_grad_broadcast_unary_tan,
)

TestArctanBroadcast = makeBroadcastTester(
    op=arctan,
    expected=upcast_float16_ufunc(np.arctan),
    good=_good_broadcast_unary_wide,
    grad=_grad_broadcast_unary_wide,
)

_grad_broadcast_binary_arctan2 = dict(
    same_shapes=(rand(2, 3), rand(2, 3)),
    scalar=(rand(2, 3), rand(1, 1)),
    row=(rand(2, 3), rand(1, 3)),
    column=(rand(2, 3), rand(2, 1)),
)

TestArctan2Broadcast = makeBroadcastTester(
    op=arctan2,
    expected=upcast_float16_ufunc(np.arctan2),
    good=_good_broadcast_binary_arctan2,
    grad=_grad_broadcast_binary_arctan2,
)

TestCoshBroadcast = makeBroadcastTester(
    op=cosh,
    expected=upcast_float16_ufunc(np.cosh),
    good=dict(
        _good_broadcast_unary_normal,
        int8=[np.arange(-89, 90, dtype="int8")],
        uint8=[np.arange(0, 90, dtype="uint8")],
        uint16=[np.arange(0, 90, dtype="uint16")],
    ),
    grad=_grad_broadcast_unary_normal,
)

_grad_broadcast_unary_arccosh = dict(
    normal=(rand_ranged(1 + _eps, 1000, (2, 3)),),
)

TestArccoshBroadcast = makeBroadcastTester(
    op=arccosh,
    expected=upcast_float16_ufunc(np.arccosh),
    good=_good_broadcast_unary_arccosh,
    grad=_grad_broadcast_unary_arccosh,
)

TestSinhBroadcast = makeBroadcastTester(
    op=sinh,
    expected=upcast_float16_ufunc(np.sinh),
    good=dict(
        _good_broadcast_unary_normal,
        int8=[np.arange(-89, 90, dtype="int8")],
        uint8=[np.arange(0, 90, dtype="uint8")],
        uint16=[np.arange(0, 90, dtype="uint16")],
    ),
    grad=_grad_broadcast_unary_normal,
)

TestArcsinhBroadcast = makeBroadcastTester(
    op=arcsinh,
    expected=upcast_float16_ufunc(np.arcsinh),
    good=_good_broadcast_unary_normal,
    grad=_grad_broadcast_unary_normal,
)

TestTanhBroadcast = makeBroadcastTester(
    op=tanh,
    expected=upcast_float16_ufunc(np.tanh),
    good=_good_broadcast_unary_normal,
    grad=_grad_broadcast_unary_normal,
)

_grad_broadcast_unary_arctanh = dict(
    normal=(rand_ranged(-1 + _eps, 1 - _eps, (2, 3)),),
)

TestArctanhBroadcast = makeBroadcastTester(
    op=arctanh,
    expected=upcast_float16_ufunc(np.arctanh),
    good=_good_broadcast_unary_arctanh,
    grad=_grad_broadcast_unary_arctanh,
)

# Complex operations
_good_complex_from_polar = dict(
    same_shapes=(abs(rand(2, 3)), rand(2, 3)),
    not_same_dimensions=(abs(rand(2, 2)), rand(2)),
    scalar=(abs(rand(2, 3)), rand(1, 1)),
    row=(abs(rand(2, 3)), rand(1, 3)),
    column=(abs(rand(2, 3)), rand(2, 1)),
    integers=(abs(randint(2, 3)), randint(2, 3)),
    empty=(np.asarray([], dtype=config.floatX), np.asarray([1], dtype=config.floatX)),
)
_grad_complex_from_polar = dict(
    same_shapes=(abs(rand(2, 3)), rand(2, 3)),
    scalar=(abs(rand(2, 3)), rand(1, 1)),
    row=(abs(rand(2, 3)), rand(1, 3)),
    column=(abs(rand(2, 3)), rand(2, 1)),
)

TestComplexFromPolarBroadcast = makeBroadcastTester(
    op=complex_from_polar,
    expected=lambda r, theta: r * np.cos(theta) + 1j * r * np.sin(theta),
    good=_good_complex_from_polar,
)

TestConjBroadcast = makeBroadcastTester(
    op=conj, expected=np.conj, good=_good_broadcast_unary_normal
)


TestDenseDot = makeTester(
    name="DenseDotTester",
    op=dense_dot,
    expected=lambda x, y: np.dot(x, y),
    checks={},
    good=dict(
        correct1=(rand(5, 7), rand(7, 5)),
        correct2=(rand(5, 7), rand(7, 9)),
        correct3=(rand(5, 7), rand(7)),
        correct4=(rand(5), rand(5, 7)),
        mixed1=(rand(5).astype("float32"), rand(5, 7)),
        mixed2=(rand(5).astype("float64"), rand(5, 7)),
        complex1=(randcomplex(5, 7), randcomplex(7)),
        complex2=(rand(5, 7), randcomplex(7)),
        complex3=(randcomplex(5, 7), rand(7)),
        empty1=(
            np.asarray([], dtype=config.floatX),
            np.asarray([], dtype=config.floatX),
        ),
        empty2=(rand(5, 0), rand(0, 2)),
        empty3=(rand(0, 5), rand(5, 0)),
    ),
    bad_build=dict(),
    bad_runtime=dict(bad1=(rand(5, 7), rand(5, 7)), bad2=(rand(5, 7), rand(8, 3))),
)


def test_isnan():
    for x in [matrix(), imatrix(), matrix(dtype="bool")]:
        y = isnan(x)
        assert isinstance(y.owner.op, Elemwise) == (x.dtype not in discrete_dtypes)
        assert y.dtype == "bool"

        # Test c code generator even for int type.
        y = isnan_(x)
        assert isinstance(y.owner.op, Elemwise)
        assert y.dtype == "bool"
        f = function([x], y, allow_input_downcast=True)
        f([[0, 1, 2]])


class TestMaxAndArgmax:
    def setup_method(self):
        utt.seed_rng()
        MaxAndArgmax.debug = 0

    def test_basic(self):
        n = as_tensor_variable(5.0)
        v, i = eval_outputs(max_and_argmax(n))
        assert v == 5.0
        assert i == 0
        assert i.dtype == "int64"
        v = eval_outputs(max_and_argmax(n)[0].shape)
        assert len(v) == 0
        v = eval_outputs(max_and_argmax(n)[1].shape)
        assert len(v) == 0

    def test_basic_1(self):
        n = as_tensor_variable([1, 2, 3, 2, -6])
        v, i = eval_outputs(max_and_argmax(n))
        assert v == 3
        assert i == 2
        assert i.dtype == "int64"
        v = eval_outputs(max_and_argmax(n)[0].shape)
        assert len(v) == 0

    def test_basic_2(self):
        data = rand(2, 3)
        n = as_tensor_variable(data)
        for (axis, np_axis) in [
            (-1, -1),
            (0, 0),
            (1, 1),
            (None, None),
            ([0, 1], None),
            ([1, 0], None),
            (NoneConst.clone(), None),
            (constant(0), 0),
        ]:
            v, i = eval_outputs(max_and_argmax(n, axis))
            assert i.dtype == "int64"
            assert np.all(v == np.max(data, np_axis))
            assert np.all(i == np.argmax(data, np_axis))
            v_shape = eval_outputs(max_and_argmax(n, axis)[0].shape)
            assert tuple(v_shape) == np.max(data, np_axis).shape

    def test_basic_2_float16(self):
        # Test negative values and bigger range to make sure numpy don't do the argmax as on uint16
        data = (rand(20, 30).astype("float16") - 0.5) * 20
        n = shared(data)
        for (axis, np_axis) in [
            (-1, -1),
            (0, 0),
            (1, 1),
            (None, None),
            ([0, 1], None),
            ([1, 0], None),
            (NoneConst.clone(), None),
            (constant(0), 0),
        ]:
            v, i = eval_outputs(max_and_argmax(n, axis), (MaxAndArgmax,))
            assert i.dtype == "int64"
            assert np.all(v == np.max(data, np_axis))
            assert np.all(i == np.argmax(data, np_axis))
            v_shape = eval_outputs(max_and_argmax(n, axis)[0].shape)
            assert tuple(v_shape) == np.max(data, np_axis).shape

    def test_basic_2_invalid(self):
        n = as_tensor_variable(rand(2, 3))
        with pytest.raises(ValueError):
            eval_outputs(max_and_argmax(n, 3))

        n = as_tensor_variable(rand(2, 3))
        with pytest.raises(ValueError):
            eval_outputs(max_and_argmax(n, -3))

    def test_basic_2_valid_neg(self):
        n = as_tensor_variable(rand(2, 3))
        v, i = eval_outputs(max_and_argmax(n, -1))
        assert i.dtype == "int64"
        assert v.shape == (2,)
        assert i.shape == (2,)
        assert np.all(v == np.max(n.value, -1))
        assert np.all(i == np.argmax(n.value, -1))
        v, i = eval_outputs(max_and_argmax(n, -2))
        assert i.dtype == "int64"
        assert v.shape == (3,)
        assert i.shape == (3,)
        assert np.all(v == np.max(n.value, -2))
        assert np.all(i == np.argmax(n.value, -2))
        v = eval_outputs(max_and_argmax(n, -1)[0].shape)
        assert v == (2)
        v = eval_outputs(max_and_argmax(n, -2)[0].shape)
        assert v == (3)

    def test_basic_3(self):
        data = rand(2, 3, 4)
        n = as_tensor_variable(data)
        for (axis, np_axis) in [
            (-1, -1),
            (0, 0),
            (1, 1),
            (None, None),
            ([0, 1, 2], None),
            ([1, 2, 0], None),
        ]:
            v, i = eval_outputs(max_and_argmax(n, axis))
            assert i.dtype == "int64"
            assert np.all(v == np.max(data, np_axis))
            assert np.all(i == np.argmax(data, np_axis))
            v = eval_outputs(max_and_argmax(n, axis)[0].shape)
            assert tuple(v) == np.max(data, np_axis).shape

    def test_arg_grad(self):
        # The test checks that the gradient of argmax(x).sum() is 0

        x = matrix()
        cost = argmax(x, axis=0).sum()
        gx = grad(cost, x)
        val = get_scalar_constant_value(gx)
        assert val == 0.0

    def test_grad(self):
        data = rand(2, 3)
        n = as_tensor_variable(data)

        def safe_verify_grad(func, data):
            # Wrapper around 'verify_grad' that picks a proper value for epsilon.
            #
            # This is needed because 'verify_grad' may fail when its epsilon is
            # too large, due to the fact the argmax is not continuous.
            # We make sure epsilon is less than the minimum absolute value found
            # in the matrix of pairwise differences between all elements in the
            # data. This way, the argmax will not change when adding epsilon.

            # 'data' is a one-element list.
            (data_tensor,) = data
            # Flatten it into a 1D vector.
            data_vector = data_tensor.flatten()
            # Compute pairwise absolute differences.
            diff = np.abs(data_vector.reshape((-1, 1)) - data_vector)
            # Alter the diagonal to avoid a zero minimum.
            for i in range(len(diff)):
                diff[i, i] = 1
            # Find an appropriate epsilon.
            eps = builtins.min(numeric_grad.type_eps[config.floatX], diff.min() / 2)
            # Run gradient verification.
            utt.verify_grad(func, data, eps=eps)

        def check_grad_max(data, max_grad_data, axis=None):
            # Why this is needed? verify_grad is not enough?
            # This works only for axis in [0, None].
            assert axis in [0, None]
            z = np.zeros_like(data)
            z = z.flatten()
            argmax = np.argmax(data, axis=axis)
            if argmax.ndim == 0:
                z[argmax] += 1
            else:
                for id, v in enumerate(argmax):
                    z[v * np.prod(data.shape[data.ndim - 1 : axis : -1]) + id] += 1

            z = z.reshape(data.shape)
            assert np.all(max_grad_data == z)

        for axis in (-1, 0, 1, None):
            for j in range(2):
                safe_verify_grad(lambda v: max_and_argmax(v, axis=axis)[j], [data])
                if axis != 1:
                    safe_verify_grad(
                        lambda v: max_and_argmax(v.flatten(), axis=axis)[j], [data]
                    )
            if axis in (0, None):
                check_grad_max(
                    data,
                    eval_outputs(grad(max_and_argmax(n, axis=axis)[0].sum(), n)),
                    axis=axis,
                )
            check_grad_max(data, eval_outputs(grad(max_and_argmax(n.flatten())[0], n)))

        # Test 3d inner dimensions
        data = rand(3, 4, 5)

        for i in [0, 1, 2]:
            safe_verify_grad(lambda v: max_and_argmax(v, axis=[i])[0], [data])
            safe_verify_grad(lambda v: max_and_argmax(v, axis=[i])[1], [data])

        # Test 4d inner dimensions
        data = rand(2, 3, 4, 5)

        for i in [0, 1, 2, 3]:
            safe_verify_grad(lambda v: max_and_argmax(v, axis=[i])[0], [data])
            safe_verify_grad(lambda v: max_and_argmax(v, axis=[i])[1], [data])

        # Test grad with multiple axes
        for i in [[0, 1], [0, 0]]:
            safe_verify_grad(lambda v: max_and_argmax(v, axis=i)[0], [data])
            safe_verify_grad(lambda v: max_and_argmax(v, axis=i)[1], [data])

    def test_preserve_broadcastable(self):
        # Ensure the original broadcastable flags are preserved by Max/Argmax.
        x = matrix().dimshuffle("x", 0, "x", 1, "x")
        y = x.max(axis=1)
        assert y.type.broadcastable == (True, True, False, True)

    def test_multiple_axes(self):
        data = np.arange(24).reshape(3, 2, 4)
        x = as_tensor_variable(data)
        v, i = eval_outputs(max_and_argmax(x, [1, -1]))
        assert np.all(v == np.array([7, 15, 23]))
        assert np.all(i == np.array([7, 7, 7]))

        v = eval_outputs(max_and_argmax(x, [1, -1])[0].shape)
        assert tuple(v) == np.max(data, (1, -1)).shape

    def test_zero_shape(self):
        x = matrix()
        m, i = max_and_argmax(x, axis=1)
        f = function([x], [m, i])
        xv = np.zeros((0, 4), dtype=config.floatX)
        mv, iv = f(xv)
        assert mv.shape == (0,)
        assert iv.shape == (0,)

    def test_numpy_input(self):
        ar = np.array([1, 2, 3])
        max_aet, argmax_aet = max_and_argmax(ar, axis=None)
        assert max_aet.eval(), 3
        assert argmax_aet.eval(), 2


class TestArgminArgmax:
    def setup_method(self):
        utt.seed_rng()
        MaxAndArgmax.debug = 0

    def test_scalar(self):
        for fct in [argmin, argmax]:
            n = as_tensor_variable(5.0)
            i = eval_outputs(fct(n))
            assert i == 0
            v = eval_outputs(fct(n).shape)
            assert len(v) == 0

    def test_list(self):
        n = as_tensor_variable([1, 2, 3, 2, -6])
        i = eval_outputs(argmin(n))
        assert i == 4
        v = eval_outputs(argmin(n).shape)
        assert len(v) == 0

        n = as_tensor_variable([1, 2, 3, 2, -6])
        i = eval_outputs(argmax(n))
        assert i == 2
        v = eval_outputs(argmax(n).shape)
        assert len(v) == 0

    def test2(self):
        data = rand(2, 3)
        n = as_tensor_variable(data)
        for fct, nfct in [(argmax, np.argmax), (argmin, np.argmin)]:
            for (axis, np_axis) in [
                (-1, -1),
                (0, 0),
                (1, 1),
                (None, None),
                ([0, 1], None),
                ([1, 0], None),
            ]:
                v = eval_outputs(fct(n, axis))
                assert np.all(v == nfct(data, np_axis))
                v_shape = eval_outputs(fct(n, axis).shape)
                assert tuple(v_shape) == nfct(data, np_axis).shape

    def test2_float16(self):
        # Test negative values and bigger range to make sure numpy don't do the argmax as on uint16
        data = (rand(20, 30).astype("float16") - 0.5) * 20
        n = shared(data)
        mode = get_default_mode().including("local_max_and_argmax", "uncanonicalize")
        for fct, nfct in [(argmax, np.argmax), (argmin, np.argmin)]:
            for (axis, np_axis) in [
                (-1, -1),
                (0, 0),
                (1, 1),
                (None, None),
                ([0, 1], None),
                ([1, 0], None),
            ]:
                v = eval_outputs(fct(n, axis), (Argmax,), mode=mode)
                assert np.all(v == nfct(data, np_axis))
                v_shape = eval_outputs(fct(n, axis).shape, mode=mode)
                assert tuple(v_shape) == nfct(data, np_axis).shape

    def test2_invalid(self):
        for fct, nfct in [(argmax, np.argmax), (argmin, np.argmin)]:
            n = as_tensor_variable(rand(2, 3))
            with pytest.raises(ValueError):
                eval_outputs(fct(n, 3))
            with pytest.raises(ValueError):
                eval_outputs(fct(n, -3))

    def test2_valid_neg(self):
        for fct, nfct in [(argmax, np.argmax), (argmin, np.argmin)]:
            n = as_tensor_variable(rand(2, 3))
            i = eval_outputs(fct(n, -1))
            assert i.shape == (2,)
            assert np.all(i == nfct(n.value, -1))
            i = eval_outputs(fct(n, -2))
            assert i.shape == (3,)
            assert np.all(i == nfct(n.value, -2))

            v = eval_outputs(fct(n, -1).shape)
            assert v == (2)
            v = eval_outputs(fct(n, -2).shape)
            assert v == (3)

    def test3(self):
        data = rand(2, 3, 4)
        n = as_tensor_variable(data)
        for fct, nfct in [(argmax, np.argmax), (argmin, np.argmin)]:
            for (axis, np_axis) in [
                (-1, -1),
                (0, 0),
                (1, 1),
                (2, 2),
                (None, None),
                ([0, 1, 2], None),
                ([1, 0, 2], None),
            ]:
                v = eval_outputs(fct(n, axis))
                assert np.all(v == nfct(data, np_axis))
                v_shape = eval_outputs(fct(n, axis).shape)
                assert tuple(v_shape) == nfct(data, np_axis).shape

    def test_grad_argmin(self):
        data = rand(2, 3)
        n = as_tensor_variable(data)
        n.name = "n"

        # test grad of argmin
        utt.verify_grad(lambda v: argmin(v, axis=-1), [data])

        utt.verify_grad(lambda v: argmin(v, axis=[0]), [data])

        utt.verify_grad(lambda v: argmin(v, axis=[1]), [data])

        utt.verify_grad(lambda v: argmin(v.flatten()), [data])

        try:
            cost = argmin(n, axis=-1)
            cost.name = None
            grad(cost, n)
            raise Exception("Expected an error")
        except TypeError:
            pass

    def test_grad_argmax(self):
        data = rand(2, 3)
        n = as_tensor_variable(data)

        # test grad of argmax
        utt.verify_grad(lambda v: argmax(v, axis=-1), [data])

        utt.verify_grad(lambda v: argmax(v, axis=[0]), [data])

        utt.verify_grad(lambda v: argmax(v, axis=[1]), [data])

        utt.verify_grad(lambda v: argmax(v.flatten()), [data])

        try:
            grad(argmax(n, axis=-1), n)
            raise Exception("Expected an error")
        except TypeError:
            pass

    def test_uint(self):
        for dtype in ("uint8", "uint16", "uint32", "uint64"):
            itype = np.iinfo(dtype)
            data = np.array([itype.min + 3, itype.min, itype.max - 5, itype.max], dtype)
            n = as_tensor_variable(data)
            i = eval_outputs(argmin(n))
            assert i == 1
            i = eval_outputs(argmax(n))
            assert i == 3

    def test_bool(self):
        data = np.array([True, False], "bool")
        n = as_tensor_variable(data)
        i = eval_outputs(argmin(n))
        assert i == 1
        i = eval_outputs(argmax(n))
        assert i == 0


class TestMinMax:
    def setup_method(self):
        utt.seed_rng()
        MaxAndArgmax.debug = 0

    def test_scalar(self):
        for fct in [max, min]:
            n = as_tensor_variable(5.0)
            v = eval_outputs(fct(n))
            assert v == 5.0

            v = eval_outputs(fct(n).shape)
            assert len(v) == 0

    def test_list(self):
        for fct, nfct in [(max, np.max), (min, np.min)]:
            n = as_tensor_variable([1, 2, 3, 2, -6])
            v = eval_outputs([fct(n)])
            assert v == nfct(n.value)

            v = eval_outputs(fct(n).shape)
            assert len(v) == 0

    def test2(self):
        data = rand(2, 3)
        n = as_tensor_variable(data)
        for fct, nfct in [(max, np.max), (min, np.min)]:
            for (axis, np_axis) in [
                (-1, -1),
                (0, 0),
                (1, 1),
                (None, None),
                ([0, 1], None),
                ([1, 0], None),
            ]:
                v = eval_outputs(fct(n, axis))
                assert np.all(v == nfct(data, np_axis))
                v_shape = eval_outputs(fct(n, axis).shape)
                assert tuple(v_shape) == nfct(data, np_axis).shape

    def test2_invalid(self):
        for fct in [max, min]:
            n = as_tensor_variable(rand(2, 3))
            with pytest.raises(ValueError):
                eval_outputs(fct(n, 3))
            with pytest.raises(ValueError):
                eval_outputs(fct(n, -3))

    def test2_valid_neg(self):
        for fct, nfct in [(max, np.max), (min, np.min)]:
            n = as_tensor_variable(rand(2, 3))
            v = eval_outputs(fct(n, -1))
            assert v.shape == (2,)
            assert np.all(v == nfct(n.value, -1))
            v = eval_outputs(fct(n, -2))
            assert v.shape == (3,)
            assert np.all(v == nfct(n.value, -2))

            v = eval_outputs(fct(n, -1).shape)
            assert v == (2)
            v = eval_outputs(fct(n, -2).shape)
            assert v == (3)

    def test3(self):
        # Test with 1 axis or all axis out of 3 dims
        data = rand(2, 3, 4)
        n = as_tensor_variable(data)
        for fct, nfct in [(max, np.max), (min, np.min)]:
            for (axis, np_axis) in [
                (-1, -1),
                (0, 0),
                (1, 1),
                (2, 2),
                (None, None),
                ([0, 1, 2], None),
                ([1, 0, 2], None),
            ]:
                v = eval_outputs(fct(n, axis))
                assert np.all(v == nfct(data, np_axis))
                v_shape = eval_outputs(fct(n, axis).shape)
                assert tuple(v_shape) == nfct(data, np_axis).shape

    def test3b(self):
        # Test with 2 axis out of 3 dims
        data = rand(2, 3, 4)
        n = as_tensor_variable(data)
        for fct, nfct in [(max, np.max), (min, np.min)]:
            for axis in [[0, 1], [1, 2], [0, 2]]:
                v = eval_outputs(fct(n, axis))
                np_v = nfct(nfct(data, axis[1]), axis[0])
                assert np.all(v == np_v)
                v_shape = eval_outputs(fct(n, axis).shape)
                assert tuple(v_shape) == np_v.shape

    def test_grad_max(self):
        data = rand(2, 3)
        n = as_tensor_variable(data)

        def check_grad_max(data, max_grad_data, axis=None):
            # This work only for axis in [0,None]
            assert axis in [0, None]
            z = np.zeros_like(data)
            z = z.flatten()
            argmax = np.argmax(data, axis=axis)
            if argmax.ndim == 0:
                z[np.argmax(data, axis=axis)] += 1
            else:
                for id, v in enumerate(argmax):
                    z[v * np.prod(data.shape[data.ndim - 1 : axis : -1]) + id] += 1

            z = z.reshape(data.shape)
            assert np.all(max_grad_data == z)

        # test grad of max
        # axis is the last one
        utt.verify_grad(lambda v: max(v, axis=-1), [data])

        utt.verify_grad(lambda v: max(v, axis=[0]), [data])
        check_grad_max(data, eval_outputs(grad(max(n, axis=0).sum(), n)), axis=0)

        utt.verify_grad(lambda v: max(v, axis=[1]), [data])
        # check_grad_max(data,eval_outputs(grad(max(n,axis=1),n)),axis=1)

        utt.verify_grad(lambda v: max(v.flatten()), [data])
        check_grad_max(data, eval_outputs(grad(max(n.flatten()), n)))

    def test_grad_min(self):
        data = rand(2, 3)
        n = as_tensor_variable(data)

        def check_grad_min(data, min_grad_data, axis=None):
            # This work only for axis in [0, None]
            assert axis in [0, None]
            z = np.zeros_like(data)
            z = z.flatten()
            argmin = np.argmin(data, axis=axis)
            if argmin.ndim == 0:
                z[np.argmin(data, axis=axis)] += 1
            else:
                for id, v in enumerate(argmin):
                    z[v * np.prod(data.shape[data.ndim - 1 : axis : -1]) + id] += 1

            z = z.reshape(data.shape)
            assert np.all(min_grad_data == z)

        # test grad of min
        # axis is the last one
        utt.verify_grad(lambda v: min(v, axis=-1), [data])

        utt.verify_grad(lambda v: min(v, axis=[0]), [data])
        check_grad_min(data, eval_outputs(grad(min(n, axis=0).sum(), n)), axis=0)

        utt.verify_grad(lambda v: min(v, axis=[1]), [data])
        # check_grad_min(data,eval_outputs(grad(min(n,axis=1),n)),axis=1)

        utt.verify_grad(lambda v: min(v.flatten()), [data])
        check_grad_min(data, eval_outputs(grad(min(n.flatten()), n)))

    def _grad_list(self):
        # Test the gradient when we have multiple axis at the same time.
        #
        # This not implemented, so we disable the test. See ticket:
        # http://www.assembla.com/spaces/aesara/tickets/511
        data = rand(2, 3)
        for fct in [max_and_argmax, max, min]:
            utt.verify_grad(lambda v: fct(v, axis=[0, 1]), [data])
        # n = as_tensor_variable(data)
        # check_grad_max(data, eval_outputs(grad(max_and_argmax(n,
        # axis=1)[0], n)),axis=1)

    def test_uint(self):
        for dtype in ("uint8", "uint16", "uint32", "uint64"):
            itype = np.iinfo(dtype)
            data = np.array([itype.min + 3, itype.min, itype.max - 5, itype.max], dtype)
            n = as_tensor_variable(data)
            assert min(n).dtype == dtype
            i = eval_outputs(min(n))
            assert i == itype.min
            assert max(n).dtype == dtype
            i = eval_outputs(max(n))
            assert i == itype.max

    def test_bool(self):
        data = np.array([True, False], "bool")
        n = as_tensor_variable(data)
        assert min(n).dtype == "bool"
        i = eval_outputs(min(n))
        assert i.ndim == 0
        assert not np.any(i)
        assert max(n).dtype == "bool"
        i = eval_outputs(max(n))
        assert i.ndim == 0
        assert np.all(i)


TestClip = makeTester(
    name="ClipTester",
    op=clip,
    expected=lambda x, y, z: np.clip(x, y, z),
    good=dict(
        correct1=(
            (5 * rand(5, 5)).astype("float32"),
            np.array(-1, dtype="float32"),
            np.array(1, dtype="float32"),
        ),
        correct2=(
            (5 * rand(5, 5)).astype("float64"),
            np.array(-1, dtype="float64"),
            np.array(1, dtype="float64"),
        ),
        correct3=(
            randint(5, 5).astype("int8"),
            np.array(-1, dtype="int8"),
            np.array(1, dtype="int8"),
        ),
        correct4=(
            randint(5, 5).astype("int16"),
            np.array(-1, dtype="int16"),
            np.array(1, dtype="int16"),
        ),
        correct5=(
            randint(5, 5).astype("int32"),
            np.array(-1, dtype="int32"),
            np.array(1, dtype="int32"),
        ),
        correct6=(
            randint(5, 5).astype("int64"),
            np.array(-1, dtype="int64"),
            np.array(1, dtype="int64"),
        ),
        # min > max case moved below as numpy has changed
        correct8=(
            randint(0, 5).astype("uint8"),
            np.array(2, dtype="uint8"),
            np.array(4, dtype="uint8"),
        ),
        correct9=(
            randint(0, 5).astype("uint16"),
            np.array(2, dtype="uint16"),
            np.array(4, dtype="uint16"),
        ),
    )
    # I can't think of any way to make this fail at runtime
)


# min > max case - numpy.clip has changed but we haven't
# https://github.com/Theano/Theano/issues/6715
TestBackwardsClip = makeTester(
    name="BackwardsClipTester",
    op=clip,
    expected=lambda x, y, z: np.where(x < y, y, np.minimum(x, z)),
    good=dict(
        correct7=(
            (5 * rand(5, 5)).astype("float64"),
            np.array(1, dtype="float64"),
            np.array(-1, dtype="float64"),
        ),
    ),
)


class TestClip:
    def test_complex_value(self):
        for dtype in ["complex64", "complex128"]:
            a = vector(dtype=dtype)
            b = scalar()
            c = scalar()
            with pytest.raises(TypeError):
                clip(a, b, c)

    def test_clip_repeat_grad(self):
        # This is testing for the issue #633
        x, y = vectors("xy")
        a = clip(x, y, x)
        g = grad(a.sum(), x)
        fn = function([x, y], [g])

        # Test the other way around as well
        a2 = clip(x, x, y)
        g2 = grad(a2.sum(), x)
        fn2 = function([x, y], [g2])

        # Test for the equal case too
        a3 = clip(x, x, x)
        g3 = grad(a3.sum(), x)
        fn3 = function([x], [g3])

        rng = np.random.RandomState(utt.fetch_seed())

        nvals = 50
        xval = rng.rand(nvals).astype(config.floatX)
        # To ensure that the min < x
        yval_mn = rng.rand(nvals).astype(config.floatX) - 1.0

        # To ensure that the max > x
        yval_mx = rng.rand(nvals).astype(config.floatX) + 1.0

        (aval,) = fn(xval, yval_mn)
        (aval2,) = fn2(xval, yval_mx)
        (aval3,) = fn3(xval)
        assert np.all(aval == 1.0)
        assert np.all(aval2 == 1.0)
        assert np.all(aval3 == 1.0)

    def test_clip_repeat_verify_grad(self):
        # Additional tests for issue gh-633
        utt.verify_grad(op=lambda x: clip(x, 0, x), pt=[rand_nonzero((3, 7))])

        utt.verify_grad(op=lambda x: clip(x, x, 0), pt=[rand_nonzero((3, 7))])

        utt.verify_grad(op=lambda x: clip(0, x, x), pt=[rand_nonzero((3, 7))])

        utt.verify_grad(op=lambda x: clip(x, x, x), pt=[rand_nonzero((3, 7))])


class TestOuter:
    def test_outer(self):
        for m in range(4):
            for n in range(4):
                x = tensor(dtype="floatX", broadcastable=(False,) * m)
                y = tensor(dtype="floatX", broadcastable=(False,) * n)
                s1 = np.random.randint(1, 10, m)
                s2 = np.random.randint(1, 10, n)
                v1 = np.asarray(np.random.rand(*s1)).astype(config.floatX)
                v2 = np.asarray(np.random.rand(*s2)).astype(config.floatX)
                o = outer(x, y).eval({x: v1, y: v2})
                utt.assert_allclose(o, np.outer(v1, v2))

    def test_grad(self):
        # Test the combined graph of the graph of outer
        # with broadcastable dimensions, just in case.
        for shp0, shp1 in [
            ((1,), (2,)),
            ((3,), (1,)),
            ((1,), (1,)),
            ((3,), (2,)),
            ((3, 2), (1, 1)),
            ((3, 2), (1, 4)),
            ((3, 2), (4, 1)),
            ((3, 2), (4, 5)),
            ((1, 2), (4, 5)),
            ((3, 1), (4, 5)),
            ((1, 1), (4, 5)),
            ((1, 1), (1, 1)),
        ]:
            data0 = np.random.rand(*shp0).astype(config.floatX)
            data1 = np.random.rand(*shp1).astype(config.floatX)
            utt.verify_grad(outer, [data0, data1])


class TestComparison:
    # Test <, >, <=, >=, == and !=
    #
    # Test that we can do the comparison with different
    # combination of tensor(shared and constant variable) with
    # ndarray. ndarray cmp tensor was crashing.  In a NumPy PR (should
    # be in the release 1.8 of NumPy), it will work.  So we assert it
    # work(futur behavior) or raise an error(current NumPy release).
    def setup_method(self):
        utt.seed_rng()
        self.mode = None
        self.shared = shared
        self.dtypes = ["float64", "float32", "complex64", "complex128"]

    def inplace_func(self, inputs, outputs, check_isfinite=None):
        mode = self.mode
        if check_isfinite is False:
            if mode is None:
                mode = get_default_mode()
            mode.check_isfinite = False
        f = inplace_func(inputs, outputs, mode=mode)
        return f

    def test_gt(self):
        for dtype in self.dtypes:
            l = np.asarray([0.0, -1.0, 1.0], dtype=dtype)
            r = np.asarray([0.0, 1.0, -1.0], dtype=dtype)
            for x, y, err in [
                (self.shared(l.astype(dtype)), self.shared(r.astype(dtype)), False),
                (l, self.shared(r.astype(dtype)), True),
                (constant(l), self.shared(r.astype(dtype)), False),
                (self.shared(l.astype(dtype)), r, False),
                (self.shared(l.astype(dtype)), constant(r), False),
            ]:
                try:
                    fn = self.inplace_func([], x > y)
                    v = fn()
                    assert np.all(v == (l > r)), (v, (l > r))
                except TypeError:
                    assert err

    def test_lt(self):
        for dtype in self.dtypes:
            l = np.asarray([0.0, -1.0, 1.0], dtype=dtype)
            r = np.asarray([0.0, 1.0, -1.0], dtype=dtype)
            for x, y, err in [
                (self.shared(l.astype(dtype)), self.shared(r.astype(dtype)), False),
                (l, self.shared(r.astype(dtype)), True),
                (constant(l), self.shared(r.astype(dtype)), False),
                (self.shared(l.astype(dtype)), r, False),
                (self.shared(l.astype(dtype)), constant(r), False),
            ]:
                try:
                    fn = self.inplace_func([], x < y)
                    v = fn()
                    assert np.all(v == (l < r)), (v, (l < r))
                except TypeError:
                    assert err

    def test_le(self):
        for dtype in self.dtypes:
            l = np.asarray([0.0, -1.0, 1.0], dtype=dtype)
            r = np.asarray([0.0, 1.0, -1.0], dtype=dtype)
            for x, y, err in [
                (self.shared(l.astype(dtype)), self.shared(r.astype(dtype)), False),
                (l, self.shared(r.astype(dtype)), True),
                (constant(l), self.shared(r.astype(dtype)), False),
                (self.shared(l.astype(dtype)), r, False),
                (self.shared(l.astype(dtype)), constant(r), False),
            ]:
                try:
                    fn = self.inplace_func([], x <= y)
                    v = fn()
                    assert np.all(v == (l <= r)), (v, (l <= r))
                except TypeError:
                    assert err

    def test_ge(self):
        for dtype in self.dtypes:
            l = np.asarray([0.0, -1.0, 1.0], dtype=dtype)
            r = np.asarray([0.0, 1.0, -1.0], dtype=dtype)
            for x, y, err in [
                (self.shared(l.astype(dtype)), self.shared(r.astype(dtype)), False),
                (l, self.shared(r.astype(dtype)), True),
                (constant(l), self.shared(r.astype(dtype)), False),
                (self.shared(l.astype(dtype)), r, False),
                (self.shared(l.astype(dtype)), constant(r), False),
            ]:
                try:
                    fn = self.inplace_func([], x >= y)
                    v = fn()
                    assert np.all(v == (l >= r)), (v, (l >= r))
                except TypeError:
                    assert err

    def test_eq(self):
        for dtype in self.dtypes:
            l = np.asarray([0.0, -1.0, 1.0], dtype=dtype)
            r = np.asarray([0.0, 1.0, -1.0], dtype=dtype)
            for x, y, err in [
                (self.shared(l.astype(dtype)), self.shared(r.astype(dtype)), False),
                (l, self.shared(r.astype(dtype)), True),
                (constant(l), self.shared(r.astype(dtype)), False),
                (self.shared(l.astype(dtype)), r, False),
                (self.shared(l.astype(dtype)), constant(r), False),
            ]:
                try:
                    fn = self.inplace_func([], eq(x, y))
                    v = fn()
                    assert np.all(v == (l == r)), (v, (l == r))
                except TypeError:
                    assert err

    def test_neq(self):
        for dtype in self.dtypes:
            l = np.asarray([0.0, -1.0, 1.0], dtype=dtype)
            r = np.asarray([0.0, 1.0, -1.0], dtype=dtype)
            for x, y, err in [
                (self.shared(l.astype(dtype)), self.shared(r.astype(dtype)), False),
                (l, self.shared(r.astype(dtype)), True),
                (constant(l), self.shared(r.astype(dtype)), False),
                (self.shared(l.astype(dtype)), r, False),
                (self.shared(l.astype(dtype)), constant(r), False),
            ]:
                try:
                    fn = self.inplace_func([], neq(x, y))
                    v = fn()
                    assert np.all(v == (l != r)), (v, (l != r))
                except TypeError:
                    assert err

    def test_isclose(self):
        for dtype in self.dtypes:
            l = np.asarray(
                [0.0, 1.0, -1.0, 0.0, np.nan, np.inf, -np.inf, np.inf], dtype=dtype
            )
            r = np.asarray(
                [0.0, 1.0001, -1.000000000001, np.nan, np.nan, np.inf, np.inf, 0.0],
                dtype=dtype,
            )
            for x, y, err in [
                (self.shared(l.astype(dtype)), self.shared(r.astype(dtype)), False),
                (l, self.shared(r.astype(dtype)), True),
                (constant(l), self.shared(r.astype(dtype)), False),
                (self.shared(l.astype(dtype)), r, False),
                (self.shared(l.astype(dtype)), constant(r), False),
            ]:
                try:
                    o1 = isclose(x, y, equal_nan=False)
                    fn1 = self.inplace_func([], o1, check_isfinite=False)

                    o2 = isclose(x, y, equal_nan=True)
                    fn2 = self.inplace_func([], o2, check_isfinite=False)

                    v1 = fn1()
                    v2 = fn2()
                    assert np.all(
                        v1
                        == np.asarray(
                            [True, False, True, False, False, True, False, False],
                            dtype="bool",
                        )
                    )
                    assert np.all(
                        v2
                        == np.asarray(
                            [True, False, True, False, True, True, False, False],
                            dtype="bool",
                        )
                    )
                except TypeError:
                    if not dtype.startswith("complex"):
                        raise
                        assert err

    def test_allclose(self):
        # equal_nan argument not in current version of numpy allclose,
        # force it to False.
        for dtype in self.dtypes:
            l = np.asarray(
                [0.0, 1.0, -1.0, 0.0, np.nan, np.inf, -np.inf, np.inf], dtype=dtype
            )
            r = np.asarray(
                [0.0, 1.0001, -1.000000000001, np.nan, np.nan, np.inf, np.inf, 0.0],
                dtype=dtype,
            )
            for x, y, err in [
                (self.shared(l.astype(dtype)), self.shared(r.astype(dtype)), False),
                (l, self.shared(r.astype(dtype)), True),
                (constant(l), self.shared(r.astype(dtype)), False),
                (self.shared(l.astype(dtype)), r, False),
                (self.shared(l.astype(dtype)), constant(r), False),
            ]:
                try:
                    fn = self.inplace_func(
                        [], allclose(x, y, equal_nan=False), check_isfinite=False
                    )
                    v = fn()
                    assert np.all(v == np.allclose(l, r))
                except TypeError:
                    if not dtype.startswith("complex"):
                        assert err


class TestBitwise:
    dtype = [
        "int8",
        "int16",
        "int32",
        "int64",
    ]

    def test_or(self):
        for dtype in self.dtype:
            x, y = vector(dtype=dtype), vector(dtype=dtype)
            fn = inplace_func([x, y], x | y)
            l = _asarray([0, 0, 1, 1], dtype=dtype)
            r = _asarray([0, 1, 0, 1], dtype=dtype)
            v = fn(l, r)
            assert np.all(v == (operator.or_(l, r))), (l, r, v)

    def test_XOR(self):
        for dtype in self.dtype:
            x, y = vector(dtype=dtype), vector(dtype=dtype)
            fn = inplace_func([x, y], x ^ y)
            l = _asarray([0, 0, 1, 1], dtype=dtype)
            r = _asarray([0, 1, 0, 1], dtype=dtype)
            v = fn(l, r)
            assert np.all(v == (operator.xor(l, r))), (l, r, v)

    def test_and(self):
        for dtype in self.dtype:
            x, y = vector(dtype=dtype), vector(dtype=dtype)
            fn = inplace_func([x, y], x & y)
            l = _asarray([0, 0, 1, 1], dtype=dtype)
            r = _asarray([0, 1, 0, 1], dtype=dtype)
            v = fn(l, r)
            assert np.all(v == (operator.and_(l, r))), (l, r, v)

    def test_inv(self):
        for dtype in self.dtype:
            x = vector(dtype=dtype)
            fn = inplace_func([x], ~x)
            for l in [
                [0, 0, 1, 1],
                [0, 1, 0, 1],
                [0, 0, 1, 1],
                [0, 1, 0, 1],
                [-1, 2 ** 16, 2 ** 16 - 1],
            ]:
                l = _asarray([0, 0, 1, 1], dtype=dtype)
                v = fn(l)
                assert np.all(v == (~l)), (l, v)

    def test_eye(self):
        n = iscalar()
        m = iscalar()
        k = iscalar()
        fn = function([m, n, k], eye(m, n, k))
        assert np.all(fn(5, 6, 1) == np.eye(5, 6, 1))


class TestAdd:
    def setup_method(self):
        utt.seed_rng()

    def test_complex_all_ops(self):
        for nbits in (64, 128):
            a = shared(np.ones(3, dtype="complex%i" % nbits) + 0.5j)
            b = shared(np.ones(3, dtype="complex%i" % nbits) + 1.5j)
            tests = (
                ("+", lambda x, y: x + y),
                ("-", lambda x, y: x - y),
                ("*", lambda x, y: x * y),
                ("/", lambda x, y: x / y),
            )
            for s, fn in tests:
                f = inplace_func([], fn(a, b))
                # print 'valid output:', fn(a.data, b.data)
                # print 'Aesara output:', f(a.data, b.data)
                assert a.type.values_eq_approx(fn(a.get_value(), b.get_value()), f())

    def test_grad_scalar_l(self):
        utt.verify_grad(add, [np.asarray([3.0]), rand(3)])

    def test_grad_scalar_r(self):
        utt.verify_grad(add, [rand(3), np.asarray([3.0])])

    def test_grad_row(self):
        utt.verify_grad(add, [rand(3, 5), rand(1, 5)])

    def test_grad_col(self):
        utt.verify_grad(add, [rand(3, 5), rand(3, 1)])


class TestCeil:
    def test_complex(self):
        with pytest.raises(TypeError):
            ceil(zvector())


class TestExp:
    def test_grad_0(self):
        utt.verify_grad(
            exp,
            [
                np.asarray(
                    [
                        [1.5089518, 1.48439076, -4.7820262],
                        [2.04832468, 0.50791564, -1.58892269],
                    ]
                )
            ],
        )

    def test_int(self):
        x = ivector()
        f = function([x], exp(x))
        exp_3 = f([3])
        assert exp_3.dtype == "float64"

    def test_complex(self):
        x = zvector()
        assert exp(x).dtype == "complex128"
        f = function([x], exp(x))
        exp_3 = f([3 + 2j])
        assert np.allclose(exp_3, np.exp(3 + 2j))


class TestDivimpl:
    def test_impls(self):
        i = iscalar()
        ii = lscalar()
        d = dscalar()
        f = fscalar()
        c = cscalar()

        assert np.allclose(function([i, d], i / d)(5, 7.0), (5.0 / 7.0))
        assert np.allclose(function([i, d], d / i)(5, 7.0), (7.0 / 5.0))
        assert np.allclose(function([i, f], i / f)(5, 11.0), (5.0 / 11.0))
        assert np.allclose(function([i, f], f / i)(5, 11.0), (11.0 / 5.0))
        assert np.allclose(function([i, ii], i // ii)(5, 3), (5 // 3))
        assert np.allclose(function([i, ii], ii // i)(5, 3), (3 // 5))
        assert np.allclose(function([i, ii], true_div(i, ii))(5, 3), (5.0 / 3.0))
        assert np.allclose(function([i, ii], true_div(ii, i))(5, 3), (3.0 / 5.0))
        assert np.allclose(
            function([i, c], i / c)(5, np.complex(5, 3)), (5.0 / (5 + 3j))
        )
        assert np.allclose(
            function([i, c], c / i)(5, np.complex(5, 3)), ((5 + 3j) / 5.0)
        )


class TestMean:
    def test_mean_single_element(self):
        res = mean(np.zeros(1))
        assert res.eval() == 0.0

    def test_mean_f16(self):
        x = vector(dtype="float16")
        y = x.mean()
        f = function([x], y)
        utt.assert_allclose(f(np.ones((100000,), dtype="float16")), 1.0)

    def test_basic(self):
        x = vector()
        f = function([x], mean(x))
        data = rand(50)
        assert np.allclose(f(data), np.mean(data))

    def test_list(self):
        ll = [shared(0.0), shared(2.0)]
        assert mean(ll).eval() == 1


def test_dot_numpy_inputs():
    """Test the `Aesara.tensor.dot` interface function with NumPy inputs."""
    a = np.ones(2)
    b = np.ones(2)
    res = dot(a, b)
    assert isinstance(res, Variable)
    assert isinstance(res.owner.op, Dot)


class TestDot:
    def setup_method(self):
        utt.seed_rng()

    def test_Op_dims(self):
        d0 = scalar()
        d1 = vector()
        d2 = matrix()
        d3 = tensor3()

        with pytest.raises(TypeError):
            _dot(d0, d0)
        with pytest.raises(TypeError):
            _dot(d0, d1)
        with pytest.raises(TypeError):
            _dot(d0, d2)
        with pytest.raises(TypeError):
            _dot(d0, d3)
        with pytest.raises(TypeError):
            _dot(d1, d0)
        _dot(d1, d1)
        _dot(d1, d2)
        with pytest.raises(TypeError):
            _dot(d1, d3)
        with pytest.raises(TypeError):
            _dot(d2, d0)
        _dot(d2, d1)
        _dot(d2, d2)
        with pytest.raises(TypeError):
            _dot(d2, d3)
        with pytest.raises(TypeError):
            _dot(d3, d0)
        with pytest.raises(TypeError):
            _dot(d3, d1)
        with pytest.raises(TypeError):
            _dot(d3, d2)
        with pytest.raises(TypeError):
            _dot(d3, d3)

    def test_grad(self):
        utt.verify_grad(dense_dot, [rand(2, 3), rand(3, 2)])
        utt.verify_grad(dense_dot, [rand(2), rand(2, 3)])
        utt.verify_grad(dense_dot, [rand(3, 2), rand(2)])
        utt.verify_grad(dense_dot, [rand(2), rand(2)])
        utt.verify_grad(dense_dot, [rand(), rand()])
        # TODO: What about the broadcastable conditions in `Dot.grad`?

    def test_broadcastable_patterns(self):

        #
        # These examples should all work.  All dimensions of all results have
        # size 1.
        #
        def val_for(r):
            if r.dtype.startswith("complex"):
                # We want to test complex at the same time, so we give a value
                # to the imaginary component.
                # This strange way of doing things is the only way that worked
                # on NumPy 1.4.1.
                if r.ndim == 0:
                    return np.asarray(np.complex(1.1, 2.1), dtype=r.dtype)
                if r.ndim == 1:
                    if r.dtype == "complex64":
                        return np.complex64([np.complex(1.2, 2.2)])
                    elif r.dtype == "complex128":
                        return np.complex128([np.complex(1.2, 2.2)])
                elif r.ndim == 2:
                    if r.dtype == "complex64":
                        return np.complex64([[np.complex(1.3, 2.3)]])
                    elif r.dtype == "complex128":
                        return np.complex128([[np.complex(1.3, 2.3)]])

            if r.ndim == 0:
                return np.asarray(1.1, dtype=r.dtype)
            if r.ndim == 1:
                return np.asarray([1.2], dtype=r.dtype)
            elif r.ndim == 2:
                return np.asarray([[1.3]], dtype=r.dtype)
            raise AssertionError()

        for dtype0 in ("float32", "float64", "complex64"):
            for dtype1 in ("float32", "complex64", "complex128"):
                for bc0 in (
                    (True,),
                    (False,),
                    (True, True),
                    (True, False),
                    (False, True),
                    (False, False),
                ):
                    x = TensorType(dtype=dtype0, broadcastable=bc0)()
                    for bc1 in (
                        (True,),
                        (False,),
                        (True, True),
                        (True, False),
                        (False, True),
                        (False, False),
                    ):

                        y = TensorType(dtype=dtype1, broadcastable=bc1)()
                        z = dense_dot(x, y)

                        if dtype0.startswith("float") and dtype1.startswith("float"):
                            g = grad(z.sum(), x)
                            assert g.broadcastable == x.broadcastable
                            g = grad(z.sum(), y)
                            assert g.broadcastable == y.broadcastable


class TestTensordot:
    def TensorDot(self, axes):
        # Since tensordot is no longer an op, mimic the old op signature
        # to allow easy use of verify_grad.
        return lambda a, b: tensordot(a, b, axes)

    def setup_method(self):
        utt.seed_rng()

    def test_basic(self):

        # Test vector-vector
        avec = vector()
        bvec = vector()
        axes = ((0,), (0,))
        c = tensordot(avec, bvec, axes)
        f1 = inplace_func([avec, bvec], c)
        aval = rand(5)
        bval = rand(5)
        out0 = np.tensordot(aval, bval, axes)
        out1 = f1(aval, bval)
        utt.assert_allclose(out0, out1)
        utt.verify_grad(self.TensorDot(axes), [aval, bval])

        # Test matrix-vector
        bmat = matrix()
        axes = ((0,), (1,))
        c = tensordot(avec, bmat, axes)
        f2 = inplace_func([avec, bmat], c)
        aval = rand(5)
        bval = rand(8, 5)
        utt.assert_allclose(np.tensordot(aval, bval, axes), f2(aval, bval))
        utt.verify_grad(self.TensorDot(axes), [aval, bval])

        # Test matrix-matrix
        amat = matrix()
        for axes, shps in [
            [((0,), (0,)), [(4, 7), (4, 9)]],
            [((0,), (1,)), [(4, 7), (9, 4)]],
            [((1,), (0,)), [(4, 7), (7, 9)]],
            [((1,), (1,)), [(4, 7), (9, 7)]],
            [((0, 1), (0, 1)), [(4, 7), (4, 7)]],
            # [((0, 1), (1, 0)), [(4, 7), (7, 4)]],
            # [((1, 0), (1, 0)), [(4, 7), (4, 7)]],
            # [((1, 0), (0, 1)), [(4, 7), (7, 4)]],
        ]:
            c = tensordot(amat, bmat, axes)
            f3 = inplace_func([amat, bmat], c)
            aval = rand(*shps[0])
            bval = rand(*shps[1])
            utt.assert_allclose(np.tensordot(aval, bval, axes), f3(aval, bval))
            utt.verify_grad(self.TensorDot(axes), [aval, bval])

        # Test ndarray-matrix, sum over one dim of matrix
        for axes, shps in [
            [((2,), (1,)), [(1, 2, 3, 4), (2, 3)]],
            [((0,), (1,)), [(1, 2, 3, 4), (3, 1)]],
            [((0,), (0,)), [(1, 2, 3, 4), (1, 3)]],
            [((3,), (0,)), [(1, 2, 3, 4), (4, 1)]],
            # [((3, 1), (0, 1)), [(1, 2, 3, 4), (4, 2)]],
            # [((0, 1), (1, 0)), [(1, 2, 3, 4), (2, 1)]],
            # [((3, 1), (1, 0)), [(1, 2, 3, 4), (2, 4)]],
        ]:
            atens = tensor4()
            c = tensordot(atens, bmat, axes)
            f4 = inplace_func([atens, bmat], c)
            aval = rand(*shps[0])
            bval = rand(*shps[1])
            utt.assert_allclose(np.tensordot(aval, bval, axes), f4(aval, bval))
            utt.verify_grad(self.TensorDot(axes), [aval, bval])

        # Test ndarray-ndarray
        atens = tensor4()
        btens = tensor3()
        axes = ((1, 3), (0, 2))
        c = tensordot(atens, btens, axes)
        f5 = inplace_func([atens, btens], c)
        aval = rand(4, 3, 5, 2)
        bval = rand(3, 4, 2)
        utt.assert_allclose(np.tensordot(aval, bval, axes), f5(aval, bval))
        utt.verify_grad(self.TensorDot(axes), [aval, bval])

        axes = (axes[1], axes[0])
        c = tensordot(btens, atens, axes)
        f6 = inplace_func([btens, atens], c)
        utt.assert_allclose(np.tensordot(bval, aval, axes), f6(bval, aval))
        utt.verify_grad(self.TensorDot(axes), [bval, aval])

    def test_raise_error(self):
        amat = matrix()
        bmat = matrix()
        bvec = vector()

        # Test invalid length for axes
        with pytest.raises(ValueError):
            tensordot(amat, bmat, (0, 1, 2))

        # Test axes of uneven length
        with pytest.raises(ValueError):
            tensordot(amat, bmat, ((0, 1), (0)))

        # Test invalid len(axes) given inputs are matrices
        with pytest.raises(ValueError):
            tensordot(amat, bmat, ((0, 1, 2), (0, 1, 2)))

        # Test invalid axes[1] given that y is a vector
        with pytest.raises(ValueError):
            tensordot(amat, bvec, (0, 1))

        # Test invalid scalar axes given inputs are matrices
        with pytest.raises(ValueError):
            tensordot(amat, bvec, 2)

    def test_weird_valid_axes(self):
        # Test matrix-matrix
        amat = matrix()
        bmat = matrix()
        for axes in [0, (1, 0), [1, 0], (1, (0,)), ((1,), 0), ([1], [0]), ([], [])]:
            c = tensordot(amat, bmat, axes)
            f3 = inplace_func([amat, bmat], c)
            aval = rand(4, 7)
            bval = rand(7, 9)
            utt.assert_allclose(np.tensordot(aval, bval, axes), f3(aval, bval))
            utt.verify_grad(self.TensorDot(axes), [aval, bval])

    def test_scalar_axes(self):
        # Test matrix-matrix
        amat = fmatrix()
        bmat = dmatrix()
        # We let at float64 to test mix of float32 and float64.
        axes = 1
        aval = rand(4, 5).astype("float32")
        bval = rand(5, 3)
        c = tensordot(amat, bmat, axes)
        f3 = inplace_func([amat, bmat], c)
        assert np.allclose(np.tensordot(aval, bval, axes), f3(aval, bval))
        utt.verify_grad(self.TensorDot(axes), [aval, bval])

        # Test tensor-tensor
        amat = tensor3()
        bmat = tensor3()
        axes = 2
        aval = rand(3, 4, 5)
        bval = rand(4, 5, 3)
        c = tensordot(amat, bmat, axes)
        f3 = inplace_func([amat, bmat], c)
        assert np.allclose(np.tensordot(aval, bval, axes), f3(aval, bval))
        utt.verify_grad(self.TensorDot(axes), [aval, bval])

    def test_scalar0(self):
        # Test tensor-tensor
        amat = matrix()
        bmat = matrix()
        axes = 0
        aval = rand(4, 5)
        bval = rand(5, 4)
        c = tensordot(amat, bmat, axes)
        f3 = inplace_func([amat, bmat], c)
        assert np.allclose(np.tensordot(aval, bval, axes), f3(aval, bval))
        utt.verify_grad(self.TensorDot(axes), [aval, bval])

    def test_broadcastable1(self):
        x = TensorType(dtype=config.floatX, broadcastable=(True, False, False))("x")
        y = tensor3("y")
        z = tensordot(x, y)
        assert z.broadcastable == (True, False)
        f = inplace_func([x, y], z)
        xv = rand(1, 3, 4)
        yv = rand(3, 4, 5)
        zv = f(xv, yv)
        assert np.allclose(np.tensordot(xv, yv), zv)

    def test_broadcastable2(self):
        x = TensorType(dtype=config.floatX, broadcastable=(True, False, False))("x")
        y = tensor3("y")
        axes = [[2, 1], [0, 1]]
        z = tensordot(x, y, axes=axes)
        assert z.broadcastable == (True, False)
        f = inplace_func([x, y], z)
        xv = rand(1, 3, 4)
        yv = rand(4, 3, 5)
        zv = f(xv, yv)
        assert np.allclose(np.tensordot(xv, yv, axes=axes), zv)


def test_smallest():
    x = dvector()
    y = dvector()
    z = dvector()
    f1 = inplace_func([x], smallest(x))
    assert np.all([1, 2, 3] == f1([1, 2, 3]))
    f3 = inplace_func([x, y, z], smallest(x, y, z))
    assert np.all([1, 2, 3] == f3([1, 3, 9], [7, 7, 7], [8, 2, 3]))

    sx, sy = dscalar(), dscalar()

    assert -4 == inplace_func([sx, sy], smallest(sx, sy))(-4.0, -2.0)


def test_var():
    a = TensorType(dtype="float64", broadcastable=[False, False, False])()
    f = function([a], var(a))

    a_val = np.arange(6).reshape(1, 2, 3)
    assert np.allclose(np.var(a_val), f(a_val))

    f = function([a], var(a, axis=0))
    assert np.allclose(np.var(a_val, axis=0), f(a_val))

    f = function([a], var(a, axis=1))
    assert np.allclose(np.var(a_val, axis=1), f(a_val))

    f = function([a], var(a, axis=2))
    assert np.allclose(np.var(a_val, axis=2), f(a_val))

    f = function([a], var(a, axis=0, ddof=0))
    assert np.allclose(np.var(a_val, axis=0, ddof=0), f(a_val))

    f = function([a], var(a, axis=1, ddof=1))
    assert np.allclose(np.var(a_val, axis=1, ddof=1), f(a_val))

    f = function([a], var(a, axis=2, ddof=1))
    assert np.allclose(np.var(a_val, axis=2, ddof=1), f(a_val))

    f = function([a], var(a, ddof=0, corrected=True))
    mean_a = np.mean(a_val)
    centered_a = a_val - mean_a
    v = np.mean(centered_a ** 2)
    error = (np.mean(centered_a)) ** 2
    v = v - error
    assert np.allclose(v, f(a_val))

    f = function([a], var(a, axis=2, ddof=1, corrected=True))
    mean_a = np.mean(a_val, axis=2, keepdims=True)
    centered_a = a_val - mean_a
    v = np.var(a_val, axis=2, ddof=1)
    shp_inp = np.shape(a_val)
    shp = shp_inp - np.array(1)
    error = (np.sum(centered_a, axis=2)) ** 2
    error = np.true_divide(error, shp[1] * shp_inp[1])
    v = v - error
    assert np.allclose(v, f(a_val))

    # Test that we don't upcast float16 computation
    assert vector(dtype="float16").var().dtype == "float16"


class TestSum:
    def test_sum_overflow(self):
        # Ensure that overflow errors are a little bit harder to get
        a = TensorType(dtype="int8", broadcastable=[False])()
        f = function([a], aet_sum(a))
        assert f([1] * 300) == 300

    def test_list(self):
        ll = [shared(0.0), shared(2.0)]
        aet_sum(ll).eval() == 2


class TestArithmeticCast:
    """Test output types of basic arithmeric operations (* / + - //).

    We only test the behavior for `config.cast_policy` set to either 'numpy' or
    'numpy+floatX': the 'custom' behavior is (at least partially) tested in
    `_test_autocast_custom`.

    """

    def test_arithmetic_cast(self):
        dtypes = get_numeric_types(with_complex=True)

        # Here:
        # scalar == scalar stored as a 0d array
        # array == 1d array
        # i_scalar == scalar type used internally by Aesara
        def Aesara_scalar(dtype):
            return scalar(dtype=str(dtype))

        def numpy_scalar(dtype):
            return np.array(1, dtype=dtype)

        def Aesara_array(dtype):
            return vector(dtype=str(dtype))

        def numpy_array(dtype):
            return np.array([1], dtype=dtype)

        def Aesara_i_scalar(dtype):
            return aes.Scalar(str(dtype))()

        def numpy_i_scalar(dtype):
            return numpy_scalar(dtype)

        with warnings.catch_warnings():
            # Avoid deprecation warning during tests.
            warnings.simplefilter("ignore", category=DeprecationWarning)
            for cfg in ("numpy+floatX",):  # Used to test 'numpy' as well.
                with config.change_flags(cast_policy=cfg):
                    for op in (
                        operator.add,
                        operator.sub,
                        operator.mul,
                        operator.truediv,
                        operator.floordiv,
                    ):
                        for a_type in dtypes:
                            for b_type in dtypes:

                                # We will test all meaningful combinations of
                                # scalar and array operations.
                                for combo in (
                                    ("scalar", "scalar"),
                                    ("array", "array"),
                                    ("scalar", "array"),
                                    ("array", "scalar"),
                                    ("i_scalar", "i_scalar"),
                                ):

                                    Aesara_args = list(
                                        map(eval, [f"Aesara_{c}" for c in combo])
                                    )
                                    numpy_args = list(
                                        map(eval, [f"numpy_{c}" for c in combo])
                                    )
                                    Aesara_dtype = op(
                                        Aesara_args[0](a_type),
                                        Aesara_args[1](b_type),
                                    ).type.dtype

                                    # For numpy we have a problem:
                                    #   http://projects.scipy.org/numpy/ticket/1827
                                    # As a result we only consider the highest data
                                    # type that numpy may return.
                                    numpy_dtypes = [
                                        op(
                                            numpy_args[0](a_type), numpy_args[1](b_type)
                                        ).dtype,
                                        op(
                                            numpy_args[1](b_type), numpy_args[0](a_type)
                                        ).dtype,
                                    ]
                                    numpy_dtype = aes.upcast(
                                        *list(map(str, numpy_dtypes))
                                    )
                                    if numpy_dtype == Aesara_dtype:
                                        # Same data type found, all is good!
                                        continue
                                    if (
                                        cfg == "numpy+floatX"
                                        and config.floatX == "float32"
                                        and a_type != "float64"
                                        and b_type != "float64"
                                        and numpy_dtype == "float64"
                                    ):
                                        # We should keep float32.
                                        assert Aesara_dtype == "float32"
                                        continue
                                    if "array" in combo and "scalar" in combo:
                                        # For mixed scalar / array operations,
                                        # Aesara may differ from numpy as it does
                                        # not try to prevent the scalar from
                                        # upcasting the array.
                                        array_type, scalar_type = (
                                            (a_type, b_type)[list(combo).index(arg)]
                                            for arg in ("array", "scalar")
                                        )
                                        up_type = aes.upcast(array_type, scalar_type)
                                        if (
                                            # The two data types are different.
                                            scalar_type != array_type
                                            and
                                            # The array type is not enough to hold
                                            # the scalar type as well.
                                            array_type != up_type
                                            and
                                            # Aesara upcasted the result array.
                                            Aesara_dtype == up_type
                                            and
                                            # But Numpy kept its original type.
                                            array_type == numpy_dtype
                                        ):
                                            # Then we accept this difference in
                                            # behavior.
                                            continue

                                    if (
                                        cfg == "numpy+floatX"
                                        and a_type == "complex128"
                                        and (b_type == "float32" or b_type == "float16")
                                        and combo == ("scalar", "array")
                                        and Aesara_dtype == "complex128"
                                        and numpy_dtype == "complex64"
                                    ):
                                        # In numpy 1.6.x adding a complex128 with
                                        # a float32 may result in a complex64. As
                                        # of 1.9.2. this is still the case so it is
                                        # probably by design
                                        pytest.skip("Known issue with" "numpy see #761")
                                    # In any other situation: something wrong is
                                    # going on!
                                    raise AssertionError()


def test_divmod():
    # Confirm that divmod is equivalent to the python version.
    x, y = fscalars("xy")
    d, r = divmod(x, y)
    fn = DualLinker().accept(FunctionGraph([x, y], [d, r])).make_function()
    for a, b in (
        (0, 1),
        (1, 1),
        (0, -1),
        (1, -1),
        (-1, -1),
        (1, 2),
        (-1, 2),
        (1, -2),
        (-1, -2),
        (5, 3),
        (-5, 3),
        (5, -3),
        (-5, -3),
    ):
        d_v, r_v = fn(a, b)
        d_vp, r_vp = divmod(a, b)
        assert d_v == d_vp and r_v == r_vp, (a,)


def test_mod_compile():
    # This test generate an Elemwise of Composite as:
    #     Elemwise{
    #         Composite{
    #             Composite{
    #                 Composite{
    #                     Composite{mod,EQ},
    #                     Switch},
    #                 mul},
    #             add}}
    #
    # The c_code generated is not compiling as of 30 June 2010. I fix the
    # compilation in the same commit.
    x = vector()
    y = vector()
    out = switch(eq(3 % x.shape[0], 0), y, y[:-1])

    function([x, y], out)


class TestInferShape(utt.InferShapeTester):
    def test_Mean(self):
        adtens3 = dtensor3()
        adtens3_val = rand(3, 4, 5)
        aiscal_val = 2
        self._compile_and_check([adtens3], [Mean(None)(adtens3)], [adtens3_val], Mean)
        self._compile_and_check(
            [adtens3], [Mean(aiscal_val)(adtens3)], [adtens3_val], Mean
        )

    def test_MaxAndArgmax(self):

        adtens3 = dtensor3()
        adtens3_val = rand(4, 5, 3)
        self._compile_and_check(
            [adtens3], max_and_argmax(adtens3, None), [adtens3_val], MaxAndArgmax
        )

        self._compile_and_check(
            [adtens3], max_and_argmax(adtens3, 0), [adtens3_val], MaxAndArgmax
        )

        self._compile_and_check(
            [adtens3], max_and_argmax(adtens3, 1), [adtens3_val], MaxAndArgmax
        )

        self._compile_and_check(
            [adtens3], max_and_argmax(adtens3, 2), [adtens3_val], MaxAndArgmax
        )

        self._compile_and_check(
            [adtens3], max_and_argmax(adtens3, [0, 1, 2]), [adtens3_val], MaxAndArgmax
        )

    def test_Dot(self):
        # Dot

        # vec/vec
        advec = dvector()
        bdvec = dvector()
        advec_val = rand(4)
        bdvec_val = rand(4)
        self._compile_and_check(
            [advec, bdvec],
            [Dot()(advec, bdvec)],
            [advec_val, bdvec_val],
            (Dot, blas.Dot22, blas.Gemv, blas_c.CGemv),
        )

        # mat/mat
        admat = dmatrix()
        bdmat = dmatrix()
        admat_val = rand(4, 5)
        bdmat_val = rand(5, 3)
        self._compile_and_check(
            [admat, bdmat],
            [Dot()(admat, bdmat)],
            [admat_val, bdmat_val],
            (Dot, blas.Dot22),
        )

        # vec/mat
        bdmat_val = rand(4, 5)
        self._compile_and_check(
            [advec, bdmat],
            [Dot()(advec, bdmat)],
            [advec_val, bdmat_val],
            (Dot, blas.Dot22, blas.Gemv, blas_c.CGemv),
        )

        # mat/vec
        admat_val = rand(5, 4)
        self._compile_and_check(
            [admat, bdvec],
            [Dot()(admat, bdvec)],
            [admat_val, bdvec_val],
            (Dot, blas.Dot22, blas.Gemv, blas_c.CGemv),
        )


class TestTensorInstanceMethods:
    def setup_method(self):
        self.vars = matrices("X", "Y")
        self.vals = [m.astype(config.floatX) for m in [rand(2, 2), rand(2, 2)]]

    def test_argmin(self):
        X, _ = self.vars
        x, _ = self.vals
        assert_array_equal(X.argmin().eval({X: x}), x.argmin())

    def test_argmax(self):
        X, _ = self.vars
        x, _ = self.vals
        assert_array_equal(X.argmax().eval({X: x}), x.argmax())

    def test_argsort(self):
        X, _ = self.vars
        x, _ = self.vals
        assert_array_equal(X.argsort().eval({X: x}), x.argsort())
        assert_array_equal(X.argsort(1).eval({X: x}), x.argsort(1))

    def test_clip(self):
        X, Y = self.vars
        x, y = self.vals
        # np.clip gives unexpected values when min > max,
        # so we have to make sure that min <= max in that test,
        # otherwise it randomly fails.
        Z = X.clip(Y - 0.5, Y + 0.5)
        z = x.clip(y - 0.5, y + 0.5)
        assert_array_equal(Z.eval({X: x, Y: y}), z)

    def test_dot(self):
        X, Y = self.vars
        x, y = self.vals
        # Use allclose comparison as a user reported on the mailing
        # list failure otherwise with array that print exactly the same.
        utt.assert_allclose(x.dot(y), X.dot(Y).eval({X: x, Y: y}))
        Z = X.dot(Y)
        z = x.dot(y)
        utt.assert_allclose(x.dot(z), X.dot(Z).eval({X: x, Z: z}))

    def test_real_imag(self):
        X, Y = self.vars
        x, y = self.vals
        Z = X + Y * 1j
        z = x + y * 1j
        assert_array_equal(Z.real.eval({Z: z}), x)
        assert_array_equal(Z.imag.eval({Z: z}), y)

    def test_conj(self):
        X, Y = self.vars
        x, y = self.vals
        Z = X + Y * 1j
        z = x + y * 1j
        assert_array_equal(Z.conj().eval({Z: z}), z.conj())
        assert_array_equal(Z.conjugate().eval({Z: z}), z.conj())

    def test_round(self):
        X, _ = self.vars
        x, _ = self.vals
        assert_array_equal(X.round().eval({X: x}), x.round())

    def test_std(self):
        X, _ = self.vars
        x, _ = self.vals
        # std() is implemented as Aesara tree and does not pass its
        # args directly to numpy. This sometimes results in small
        # difference, so we use allclose test.
        utt.assert_allclose(X.std().eval({X: x}), x.std())

    def test_cumsum(self):
        X, _ = self.vars
        x, _ = self.vals
        assert_array_equal(X.cumsum().eval({X: x}), x.cumsum())

    def test_cumprod(self):
        X, _ = self.vars
        x, _ = self.vals
        assert_array_equal(X.cumprod().eval({X: x}), x.cumprod())


def test_norm():
    x = vector("x")
    n = x.norm(2)
    f = function([x], n)
    assert np.allclose(f([1, 1]), np.sqrt(2))


def test_cov():
    x = matrix("x")
    y = matrix("y")

    for rowvar, bias, ddof in product([True, False], [True, False], [None, 2]):
        c = cov(x, rowvar=rowvar, bias=bias, ddof=ddof)
        f = function([x], c)

        data = np.asarray(np.random.rand(3, 5), dtype=config.floatX)
        assert np.allclose(f(data), np.cov(data, rowvar=rowvar, bias=bias, ddof=ddof))

        c = cov(x, y=y, rowvar=rowvar, bias=bias, ddof=ddof)
        f = function([x, y], c)

        data = np.asarray(np.random.rand(3, 5), dtype=config.floatX)
        y_val = np.asarray(np.random.rand(3, 5), dtype=config.floatX)
        assert np.allclose(
            f(data, y_val), np.cov(data, rowvar=rowvar, y=y_val, bias=bias, ddof=ddof)
        )


def test_ptp():
    # Should return 0 for all scalar
    x = scalar("x")
    p = ptp(x)
    f = function([x], p)

    y = np.asarray(rand() * 20 - 10, dtype=config.floatX)
    result = f(y)
    numpyResult = np.ptp(y)

    assert np.array_equal(result, numpyResult)


class TestPower:
    def test_numpy_compare(self):
        rng = np.random.RandomState(utt.fetch_seed())
        A = matrix("A", dtype=config.floatX)
        Q = power(A, 3)
        fn = function([A], [Q])
        a = rng.rand(4, 4).astype(config.floatX)

        n_p = np.power(a, 3)
        t_p = fn(a)
        assert np.allclose(n_p, t_p)

    def test_multiple_power(self):
        x = vector()
        y = [1, 2, 3]
        z = power(x, y)
        f = function([x], z)
        assert np.allclose(f([1, 2, 3]), [1, 4, 27])

    def test_wrong_shape(self):
        x = vector()
        y = [1, 2, 3]
        z = power(x, y)
        f = function([x], z)
        with pytest.raises(ValueError):
            f([1, 2, 3, 4])


class TestProd:
    def setup_method(self):
        utt.seed_rng()

        # we want to allow nans in the matrices, so we disable this
        # DEBUG_MODE check
        mode = get_default_mode()
        mode = copy(mode)
        mode.check_isfinite = False

        self.mode = mode

    def test_verify_grad(self):

        # including zeros, as the case with zeros is important
        # (and special cases: 1 zero in the row, more than 1 zero in the row)
        x_val = np.asarray(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype="float32"
        )
        # now with verify_grad
        utt.verify_grad(Prod(axis=1), [x_val], mode=self.mode)

        # second time, with some added complexity
        # verify_grad takes the sum of the matrices anyway
        def fn(x2):
            return sqr(Prod(axis=1)(x2))

        utt.verify_grad(fn, [x_val], mode=self.mode)

    def test_verify_grad_with_zeros(self):
        # including zeros, as the case with zeros is important
        # (and special cases: 1 zero in the row, more than 1 zero in the row)
        x_val = np.asarray(
            [[1.0, 2.0, 3.0], [0.0, 5.0, 6.0], [0.0, 0.0, 9.0]], dtype="float32"
        )
        x = dmatrix()

        # sanity check
        p = Prod(axis=1)(x)

        fn3 = function([x], [p], mode=self.mode)
        assert np.allclose(fn3(x_val), [6.0, 0.0, 0.0])

        # now with verify_grad
        utt.verify_grad(Prod(axis=1), [x_val], mode=self.mode)

    def test_prod_no_zeros_in_input(self):
        x = dmatrix()
        x_val = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype="float32")
        pwz = Prod(axis=1, no_zeros_in_input=True)(x)
        fn = function([x], pwz, mode=self.mode)

        assert np.allclose(fn(x_val), [6, 120, 504])

        pwz = Prod(no_zeros_in_input=True)(x)
        g = grad(pwz, x)
        gg = grad(g.sum(), x)
        fn = function([x], g, mode=self.mode)
        assert np.allclose(
            fn(x_val),
            [
                [362880.0, 181440.0, 120960.0],
                [90720.0, 72576.0, 60480.0],
                [51840.0, 45360.0, 40320.0],
            ],
        )
        fn = function([x], gg, mode=self.mode)
        assert np.allclose(
            fn(x_val),
            [
                [663696.0, 422568.0, 301872.0],
                [233964.0, 190800.0, 161016.0],
                [139248.0, 122652.0, 109584.0],
            ],
        )
        utt.verify_grad(Prod(axis=1, no_zeros_in_input=True), [x_val], mode=self.mode)
        utt.verify_grad(Prod(no_zeros_in_input=True), [x_val], mode=self.mode)

    def test_prod_without_zeros(self):
        x = dmatrix()
        x_val = np.array([[1, 2, 3], [0, 5, 6], [0, 0, 9]], dtype="float32")
        pwz = ProdWithoutZeros(axis=1)(x)
        fn = function([x], pwz, mode=self.mode)
        assert np.allclose(fn(x_val), [6, 30, 9])

        pwz_a0 = ProdWithoutZeros(axis=0)(x)
        fn_a0 = function([x], pwz_a0, mode=self.mode)
        assert np.allclose(fn_a0(x_val), [1, 10, 162])

    @pytest.mark.xfail(raises=NullTypeGradError)
    def test_prod_without_zeros_grad(self):
        x = dmatrix()
        pwz_a1 = ProdWithoutZeros(axis=0)(x)
        pwz_grad = grad(aet_sum(pwz_a1), x)
        # FIXME: This is not a real test.
        function([x], pwz_grad, mode=self.mode)

    def test_other_grad_tests(self):
        x = dmatrix()
        x_val1 = np.array([[1, 2, 3], [0, 5, 6], [0, 0, 9]], dtype="float32")
        x_val2 = np.array(
            [[1, 2, 0], [0, 5, 6], [7, 8, 9], [9, 10, 0]], dtype="float32"
        )
        rng = rng = np.random.RandomState(43)

        p = Prod(axis=1)
        grad_p = grad(p(x).sum(), x)
        grad_fn = function([x], grad_p, mode=self.mode)
        assert np.allclose(
            grad_fn(x_val1), [[6.0, 3.0, 2.0], [30.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )
        assert np.allclose(
            grad_fn(x_val2),
            [[0.0, 0.0, 2.0], [30.0, 0.0, 0.0], [72.0, 63.0, 56.0], [0.0, 0.0, 90.0]],
        )

        p_axis0 = Prod(axis=0)
        grad_p_axis0 = grad(p_axis0(x).sum(), x)
        grad_fn_axis0 = function([x], grad_p_axis0, mode=self.mode)
        assert np.allclose(
            grad_fn_axis0(x_val2),
            [
                [0.0, 400.0, 0.0],
                [63.0, 160.0, 0.0],
                [0.0, 100.0, 0.0],
                [0.0, 80.0, 0.0],
            ],
        )

        utt.verify_grad(p, [x_val1], rng=rng, mode=self.mode)

    def test_pickle(self):
        test_prod = Prod()
        prod_pickled = pickle.dumps(test_prod, protocol=-1)
        unpickled_prod = pickle.loads(prod_pickled)
        assert not unpickled_prod.no_zeros_in_input

        prod_pickled = pickle.dumps(test_prod)
        unpickled_prod = pickle.loads(prod_pickled)
        assert not unpickled_prod.no_zeros_in_input


class TestIsInfIsNan:
    def setup_method(self):
        self.test_vals = [
            np.array(x, dtype=config.floatX)
            for x in [
                0,
                1,
                np.nan,
                np.inf,
                -np.inf,
                [np.nan, np.inf, -np.inf, 0, 1, -1],
            ]
        ]
        self.scalar = scalar()
        self.vector = vector()
        self.mode = get_default_mode()
        if isinstance(self.mode, DebugMode):
            # Disable the check preventing usage of NaN / Inf values.
            self.mode = copy(self.mode)
            self.mode.check_isfinite = False

    def run_isfunc(self, aet_func, np_func):
        for args in (self.scalar, self.vector):
            Aesara_isfunc = function([args], aet_func(args), mode=self.mode)
            for x in self.test_vals:
                if (x.ndim == 0 and args is not self.scalar) or (
                    x.ndim == 1 and args is not self.vector
                ):
                    # We only test with the appropriate input type.
                    continue
                t_out = Aesara_isfunc(x)
                n_out = np_func(x)
                assert (t_out == n_out).all(), (t_out, n_out)

    def test_isinf(self):
        self.run_isfunc(isinf, np.isinf)

    def test_isnan(self):
        self.run_isfunc(isnan, np.isnan)


class TestSumProdReduceDtype:
    mode = get_default_mode().excluding("local_cut_useless_reduce")
    op = CAReduce
    axes = [None, 0, 1, [], [0], [1], [0, 1]]
    methods = ["sum", "prod"]
    dtypes = list(map(str, aes.all_types))

    # Test the default dtype of a method().
    def test_reduce_default_dtype(self):
        # We try multiple axis combinations even though axis should not matter.
        for method in self.methods:
            for idx, dtype in enumerate(self.dtypes):
                axis = self.axes[idx % len(self.axes)]
                x = matrix(dtype=dtype)
                s = getattr(x, method)(axis=axis)
                assert (
                    s.dtype
                    == dict(
                        bool="int64",
                        int8="int64",
                        int16="int64",
                        int32="int64",
                        uint8="uint64",
                        uint16="uint64",
                        uint32="uint64",
                    ).get(dtype, dtype)
                )
                f = function([x], s, mode=self.mode)
                topo = f.maker.fgraph.toposort()
                assert [n for n in topo if isinstance(n.op, self.op)], (topo, dtype)
                data = np.random.rand(3, 4) * 10
                data = data.astype(dtype)
                f(data)

    def test_reduce_default_acc_dtype(self):
        # Test the default acc_dtype of a reduce().

        # We try multiple axis combinations even though axis should not matter.
        for method in self.methods:
            for idx, dtype in enumerate(self.dtypes):
                axis = self.axes[idx % len(self.axes)]
                x = matrix(dtype=dtype)
                s = getattr(x, method)(axis=axis)
                assert (
                    s.owner.op.acc_dtype
                    == dict(
                        bool="int64",
                        int8="int64",
                        int16="int64",
                        int32="int64",
                        uint8="uint64",
                        uint16="uint64",
                        uint32="uint64",
                        float16="float32",
                        float32="float64",
                        complex64="complex128",
                    ).get(dtype, dtype)
                )
                f = function([x], s, mode=self.mode)
                topo = f.maker.fgraph.toposort()
                assert [n for n in topo if isinstance(n.op, self.op)], (topo, dtype)
                data = np.random.rand(3, 4) * 10
                data = data.astype(dtype)
                f(data)

    @pytest.mark.slow
    def test_reduce_custom_dtype(self):
        # Test the ability to provide your own output dtype for a reduce.

        # We try multiple axis combinations even though axis should not matter.
        idx = 0
        for method in self.methods:
            for input_dtype in self.dtypes:
                x = matrix(dtype=input_dtype)
                for output_dtype in self.dtypes:
                    # Only tests case where both input and output are complex.
                    icomplex = input_dtype.startswith("complex")
                    ocomplex = output_dtype.startswith("complex")
                    if icomplex != ocomplex:
                        continue

                    axis = self.axes[idx % len(self.axes)]
                    var = getattr(x, method)(dtype=output_dtype, axis=axis)
                    assert var.dtype == output_dtype

                    f = function([x], var, mode=self.mode)
                    topo = f.maker.fgraph.toposort()
                    assert [n for n in topo if isinstance(n.op, self.op)], (
                        topo,
                        output_dtype,
                    )
                    data = np.random.rand(3, 4) * 10
                    data = data.astype(input_dtype)
                    if method == "prod" and output_dtype in [
                        "float16",
                        "int8",
                        "uint8",
                        "int16",
                        "uint16",
                    ]:
                        # We will likely get something infinite,
                        # or the overflow will be different between CPU and GPU,
                        # and DebugMode will complain.
                        data = data[0:1]
                    f(data)
                    if "complex" in input_dtype:
                        continue
                    # Check that we can take the gradient
                    grad(var.sum(), x, disconnected_inputs="ignore")
                    idx += 1

    def test_reduce_custom_acc_dtype(self):
        # Test the ability to provide your own accumulator dtype for a reduce.

        # We try multiple axis combinations even though axis should not matter.
        idx = 0
        for method in self.methods:
            for input_dtype in self.dtypes:
                x = matrix(dtype=input_dtype)
                for acc_dtype in self.dtypes:
                    # If the accumulator is a complex, the gradient of the reduce will
                    # cast the complex to the input dtype. We can't call the normal
                    # cast on a complex to a not complex as this is ambiguous.
                    if not input_dtype.startswith("complex") and acc_dtype.startswith(
                        "complex"
                    ):
                        continue

                    axis = self.axes[idx % len(self.axes)]
                    # If output_dtype would force a downcast, we expect a TypeError
                    # We always allow int/uint inputs with float/complex outputs.
                    upcasted_dtype = aes.upcast(input_dtype, acc_dtype)
                    if acc_dtype == upcasted_dtype or (
                        input_dtype in discrete_dtypes
                        and acc_dtype in continuous_dtypes
                    ):
                        var = getattr(x, method)(acc_dtype=acc_dtype, axis=axis)
                        assert var.owner.op.acc_dtype == acc_dtype

                        if "complex" in input_dtype:
                            continue
                        # Check that we can take the gradient
                        grad(var.sum(), x, disconnected_inputs="ignore")
                    else:
                        with pytest.raises(TypeError):
                            getattr(x(method), acc_dtype=acc_dtype, axis=axis)

                    idx += 1

    def test_reduce_precision(self):
        # Check that the default accumulator precision is sufficient
        for method in self.methods:
            x = shared(np.asarray([1e8, 1, -1e8], dtype="float32"))
            s = getattr(x, method)()
            f = function([], s, mode=self.mode)
            topo = f.maker.fgraph.toposort()
            assert [n for n in topo if isinstance(n.op, self.op)], topo
            s_val = f()
            # Use extra precision in NumPy to compute the good answer.
            ret = getattr(np.asarray([1e8, 1, -1e8], dtype="float64"), method)()
            assert np.allclose(s_val, ret), (s_val, ret)


class TestMeanDtype:
    def test_mean_default_dtype(self):
        # Test the default dtype of a mean().

        # We try multiple axis combinations even though axis should not matter.
        axes = [None, 0, 1, [], [0], [1], [0, 1]]
        for idx, dtype in enumerate(map(str, aes.all_types)):
            axis = axes[idx % len(axes)]
            x = matrix(dtype=dtype)
            m = x.mean(axis=axis)
            if dtype in discrete_dtypes:
                assert m.dtype == "float64"
            else:
                assert m.dtype == dtype, (m, m.dtype, dtype)
            f = function([x], m)
            data = np.random.rand(3, 4) * 10
            data = data.astype(dtype)
            f(data)

    @pytest.mark.slow
    def test_mean_custom_dtype(self):
        # Test the ability to provide your own output dtype for a mean.

        # We try multiple axis combinations even though axis should not matter.
        axes = [None, 0, 1, [], [0], [1], [0, 1]]
        idx = 0
        for input_dtype in map(str, aes.all_types):
            x = matrix(dtype=input_dtype)
            for sum_dtype in map(str, aes.all_types):
                axis = axes[idx % len(axes)]
                # If the inner sum cannot be created, it will raise a
                # TypeError.
                try:
                    mean_var = x.mean(dtype=sum_dtype, axis=axis)
                except TypeError:
                    pass
                else:
                    # Executed if no TypeError was raised
                    if sum_dtype in discrete_dtypes:
                        assert mean_var.dtype == "float64", (mean_var.dtype, sum_dtype)
                    else:
                        assert mean_var.dtype == sum_dtype, (mean_var.dtype, sum_dtype)
                    if (
                        "complex" in input_dtype or "complex" in sum_dtype
                    ) and input_dtype != sum_dtype:
                        continue
                    f = function([x], mean_var)
                    data = np.random.rand(3, 4) * 10
                    data = data.astype(input_dtype)
                    f(data)
                    # Check that we can take the gradient, when implemented
                    if "complex" in mean_var.dtype:
                        continue
                    try:
                        grad(mean_var.sum(), x, disconnected_inputs="ignore")
                    except NotImplementedError:
                        # TrueDiv does not seem to have a gradient when
                        # the numerator is complex.
                        if mean_var.dtype in complex_dtypes:
                            pass
                        else:
                            raise

                idx += 1

    def test_mean_precision(self):
        # Check that the default accumulator precision is sufficient
        x = shared(np.asarray([1e8, 1, -1e8], dtype="float32"))
        m = x.mean()
        f = function([], m)
        m_val = f()
        assert np.allclose(m_val, 1.0 / 3)


class TestProdWithoutZerosDtype:
    def test_prod_without_zeros_default_dtype(self):
        # Test the default dtype of a ProdWithoutZeros().

        # We try multiple axis combinations even though axis should not matter.
        axes = [None, 0, 1, [], [0], [1], [0, 1]]
        for idx, dtype in enumerate(map(str, aes.all_types)):
            axis = axes[idx % len(axes)]
            x = ProdWithoutZeros(axis=axis)(matrix(dtype=dtype))
            assert (
                x.dtype
                == dict(
                    bool="int64",
                    int8="int64",
                    int16="int64",
                    int32="int64",
                    uint8="uint64",
                    uint16="uint64",
                    uint32="uint64",
                ).get(dtype, dtype)
            )

    def test_prod_without_zeros_default_acc_dtype(self):
        # Test the default dtype of a ProdWithoutZeros().

        # We try multiple axis combinations even though axis should not matter.
        axes = [None, 0, 1, [], [0], [1], [0, 1]]
        for idx, dtype in enumerate(map(str, aes.all_types)):
            axis = axes[idx % len(axes)]
            x = matrix(dtype=dtype)
            p = ProdWithoutZeros(axis=axis)(x)
            assert (
                p.owner.op.acc_dtype
                == dict(
                    bool="int64",
                    int8="int64",
                    int16="int64",
                    int32="int64",
                    uint8="uint64",
                    uint16="uint64",
                    uint32="uint64",
                    float16="float32",
                    float32="float64",
                    complex64="complex128",
                ).get(dtype, dtype)
            )

            if "complex" in dtype:
                continue
            f = function([x], p)
            data = np.random.rand(2, 3) * 3
            data = data.astype(dtype)
            f(data)

    @pytest.mark.slow
    def test_prod_without_zeros_custom_dtype(self):
        # Test ability to provide your own output dtype for a ProdWithoutZeros().

        # We try multiple axis combinations even though axis should not matter.
        axes = [None, 0, 1, [], [0], [1], [0, 1]]
        idx = 0
        for input_dtype in map(str, aes.all_types):
            x = matrix(dtype=input_dtype)
            for output_dtype in map(str, aes.all_types):
                axis = axes[idx % len(axes)]
                prod_woz_var = ProdWithoutZeros(axis=axis, dtype=output_dtype)(x)
                assert prod_woz_var.dtype == output_dtype
                idx += 1
                if "complex" in output_dtype or "complex" in input_dtype:
                    continue
                f = function([x], prod_woz_var)
                data = np.random.rand(2, 3) * 3
                data = data.astype(input_dtype)
                f(data)

    @pytest.mark.slow
    def test_prod_without_zeros_custom_acc_dtype(self):
        # Test ability to provide your own acc_dtype for a ProdWithoutZeros().

        # We try multiple axis combinations even though axis should not matter.
        axes = [None, 0, 1, [], [0], [1], [0, 1]]
        idx = 0
        for input_dtype in map(str, aes.all_types):
            x = matrix(dtype=input_dtype)
            for acc_dtype in map(str, aes.all_types):
                axis = axes[idx % len(axes)]
                # If acc_dtype would force a downcast, we expect a TypeError
                # We always allow int/uint inputs with float/complex outputs.
                upcasted_dtype = aes.upcast(input_dtype, acc_dtype)
                if acc_dtype == upcasted_dtype or (
                    input_dtype in discrete_dtypes and acc_dtype in continuous_dtypes
                ):
                    prod_woz_var = ProdWithoutZeros(axis=axis, acc_dtype=acc_dtype)(x)
                    assert prod_woz_var.owner.op.acc_dtype == acc_dtype

                    if acc_dtype.startswith("complex") and input_dtype != acc_dtype:
                        continue
                    f = function([x], prod_woz_var)
                    data = np.random.rand(2, 3) * 3
                    data = data.astype(input_dtype)
                    f(data)
                else:
                    with pytest.raises(TypeError):
                        ProdWithoutZeros(axis=axis, acc_dtype=acc_dtype)(x)

                idx += 1


class TestSumMeanMaxMinArgMaxVarReduceAxes:
    def test_sum_axes(self):
        axes = [None, 0, 1, [0, 1], np.array(1), [np.array(0), np.array(1)]]
        for a in axes:
            x = matrix()
            x.sum(a)

    def test_mean_axes(self):
        axes = [None, 0, 1, [0, 1], np.array(1), [np.array(0), np.array(1)]]
        for a in axes:
            x = matrix()
            x.mean(a)

    def test_max_axes(self):
        axes = [None, 0, 1, [0, 1], np.array(1), [np.array(0), np.array(1)]]
        for a in axes:
            x = matrix()
            x.max(a)

    def test_min_axes(self):
        axes = [None, 0, 1, [0, 1], np.array(1), [np.array(0), np.array(1)]]
        for a in axes:
            x = matrix()
            x.min(a)

    def test_argmax_axes(self):
        axes = [None, 0, 1, [0, 1], np.array(1), [np.array(0), np.array(1)]]
        for a in axes:
            x = matrix()
            x.argmax(a)

    def test_var_axes(self):
        axes = [None, 0, 1, [0, 1], np.array(1), [np.array(0), np.array(1)]]
        for a in axes:
            x = matrix()
            x.var(a)


def reduce_bitwise_and(x, axis=-1, dtype="int8"):
    identity = np.array((-1,), dtype=dtype)[0]

    shape_without_axis = tuple([s for i, s in enumerate(x.shape) if i != axis])
    if 0 in shape_without_axis:
        return np.empty(shape=shape_without_axis, dtype=x.dtype)

    def custom_reduce(a):
        out = identity
        for i in range(a.size):
            out = np.bitwise_and(a[i], out)
        return out

    return np.apply_along_axis(custom_reduce, axis, x)


def test_clip_grad():

    # test the gradient of clip
    def func(x, y, z):
        return clip(x, y, z)

    # use an x value less than y, an x value between y and z, and an x value
    # greater than z
    utt.verify_grad(func, [np.asarray([-1.0, 0.5, 2.0]), 0.0, 1.0])


def test_clip_grad_int():
    # FIXME: This is not a real test.
    # test that integers don't crash clip gradient
    x = iscalar()
    y = iscalar()
    z = iscalar()
    c = clip(x, y, z)
    grad(c, [x, y, z])


def test_grad_useless_sum():
    """
    Test absence of useless sum.

    When an operation (such as `Aesara.tensor.mul`) is done on a broadcastable
    vector and a matrix, the gradient in backward path is computed for the
    broadcasted vector. So a sum reverts the broadcasted vector to a vector. In
    the case of operations on two broadcastable vectors, the sum should not be
    generated.

    This test checks whether there is a useless sum in the gradient
    computations.
    """

    mode = get_default_mode().including("canonicalize")
    mode.check_isfinite = False
    x = TensorType(config.floatX, (True,))("x")
    l = log(1.0 - sigmoid(x))[0]
    g = grad(l, x)

    f = function([x], g, mode=mode)
    test_values = [-100, -1, 0, 1, 100]
    outputs = []
    old_values_eq_approx = staticmethod(TensorType.values_eq_approx)
    TensorType.values_eq_approx = staticmethod(values_eq_approx_remove_nan)
    try:
        for test_value in test_values:
            outputs.append(f(np.array([test_value]).astype("float32")))
    finally:
        TensorType.values_eq_approx = old_values_eq_approx

    assert not any([isinstance(node.op, Sum) for node in applys_between([x], [g])])
    assert np.allclose(
        outputs, [[-3.72007598e-44], [-0.26894142], [-0.5], [-0.73105858], [-1.0]]
    )


def test_tanh_grad_broadcast():
    # FIXME: This is not a real test.
    # This crashed in the past.

    x = tensor(dtype="float32", broadcastable=(True, False, False, False))
    y = tensor(dtype="float32", broadcastable=(True, True, False, False))

    grad(tanh(x).sum(), x)
    grad(tanh(x + y).sum(), y)
    grad(tanh(x + y).sum(), [x, y])
