import copy
import time

import numpy as np
import pytest

import aesara
import aesara.scalar as aes
import aesara.tensor as aet
from aesara import shared
from aesara.assert_op import Assert
from aesara.compile import optdb
from aesara.compile.debugmode import DebugMode
from aesara.compile.function import function
from aesara.compile.mode import Mode, get_default_mode, get_mode
from aesara.compile.ops import DeepCopyOp, deep_copy_op
from aesara.configdefaults import config
from aesara.graph.basic import Apply, Constant
from aesara.graph.fg import FunctionGraph
from aesara.graph.op import Op
from aesara.graph.opt import check_stack_trace, local_optimizer, out2in
from aesara.graph.optdb import Query
from aesara.misc.safe_asarray import _asarray
from aesara.tensor import inplace
from aesara.tensor.basic import (
    Alloc,
    Join,
    MakeVector,
    Rebroadcast,
    ScalarFromTensor,
    Split,
    TensorFromScalar,
    _convert_to_int8,
    as_tensor_variable,
    join,
    make_vector,
    tile,
)
from aesara.tensor.basic_opt import (
    ShapeFeature,
    apply_rebroadcast_opt,
    assert_op,
    local_canonicalize_alloc,
    local_dimshuffle_lift,
    local_merge_alloc,
    local_reshape_to_dimshuffle,
    local_useless_alloc,
    local_useless_dimshuffle_in_reshape,
    local_useless_elemwise,
    local_useless_reshape,
    register_specialize,
)
from aesara.tensor.elemwise import DimShuffle, Elemwise
from aesara.tensor.math import (
    Dot,
    add,
    bitwise_and,
    bitwise_or,
    bitwise_xor,
    cos,
    cosh,
    dot,
    eq,
    exp,
    floor_div,
    ge,
    gt,
    int_div,
    inv,
    invert,
    iround,
    le,
    log,
    log2,
    log10,
    lt,
    maximum,
    minimum,
    mul,
    neg,
    neq,
)
from aesara.tensor.math import pow as aet_pow
from aesara.tensor.math import round as aet_round
from aesara.tensor.math import sin, sinh, softplus, sqr, sqrt, sub
from aesara.tensor.math import sum as aet_sum
from aesara.tensor.math import tan, tanh, true_div, xor
from aesara.tensor.math_opt import local_lift_transpose_through_dot
from aesara.tensor.shape import Reshape, Shape_i, SpecifyShape, reshape, specify_shape
from aesara.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedIncSubtensor1,
    AdvancedSubtensor,
    AdvancedSubtensor1,
    IncSubtensor,
    Subtensor,
    advanced_inc_subtensor,
    advanced_inc_subtensor1,
    inc_subtensor,
    set_subtensor,
)
from aesara.tensor.type import (
    TensorType,
    bmatrix,
    bscalar,
    col,
    dmatrices,
    dmatrix,
    dscalar,
    dvector,
    fmatrix,
    fscalar,
    fvector,
    imatrices,
    int_dtypes,
    iscalar,
    ivector,
    lscalar,
    lscalars,
    lvector,
    matrices,
    matrix,
    row,
    scalar,
    scalars,
    tensor,
    tensor3,
    tensor4,
    values_eq_approx_remove_nan,
    vector,
    vectors,
)
from tests import unittest_tools as utt


mode_opt = config.mode
if mode_opt == "FAST_COMPILE":
    mode_opt = "FAST_RUN"
mode_opt = get_mode(mode_opt)

dimshuffle_lift = out2in(local_dimshuffle_lift)

_optimizer_stabilize = Query(include=["fast_run"])
_optimizer_stabilize.position_cutoff = 1.51
_optimizer_stabilize = optdb.query(_optimizer_stabilize)

_optimizer_specialize = Query(include=["fast_run"])
_optimizer_specialize.position_cutoff = 2.01
_optimizer_specialize = optdb.query(_optimizer_specialize)

_optimizer_fast_run = Query(include=["fast_run"])
_optimizer_fast_run = optdb.query(_optimizer_fast_run)


def ds(x, y):
    return DimShuffle(x.type.broadcastable, y)(x)


def optimize(g, level="fast_run"):
    if level == "fast_run":
        _optimizer_fast_run.optimize(g)
    elif level == "specialize":
        _optimizer_specialize.optimize(g)
    elif level == "stabilize":
        _optimizer_stabilize.optimize(g)
    else:
        raise ValueError(level)
    return g


def inputs(xbc=(0, 0), ybc=(0, 0), zbc=(0, 0)):
    x = TensorType(broadcastable=xbc, dtype="float64")("x")
    y = TensorType(broadcastable=ybc, dtype="float64")("y")
    z = TensorType(broadcastable=zbc, dtype="float64")("z")
    return x, y, z


class TestDimshuffleLift:
    def test_double_transpose(self):
        x, y, z = inputs()
        e = ds(ds(x, (1, 0)), (1, 0))
        g = FunctionGraph([x], [e])
        assert (
            str(g) == "FunctionGraph(InplaceDimShuffle{1,0}(InplaceDimShuffle{1,0}(x)))"
        )
        dimshuffle_lift.optimize(g)
        assert str(g) == "FunctionGraph(x)"
        # no need to check_stack_trace as graph is supposed to be empty

    def test_merge2(self):
        x, y, z = inputs()
        e = ds(ds(x, (1, "x", 0)), (2, 0, "x", 1))
        g = FunctionGraph([x], [e])
        assert (
            str(g)
            == "FunctionGraph(InplaceDimShuffle{2,0,x,1}(InplaceDimShuffle{1,x,0}(x)))"
        ), str(g)
        dimshuffle_lift.optimize(g)
        assert str(g) == "FunctionGraph(InplaceDimShuffle{0,1,x,x}(x))", str(g)
        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(g, ops_to_check="all")

    def test_elim3(self):
        x, y, z = inputs()
        e = ds(ds(ds(x, (0, "x", 1)), (2, 0, "x", 1)), (1, 0))
        g = FunctionGraph([x], [e])
        assert str(g) == (
            "FunctionGraph(InplaceDimShuffle{1,0}(InplaceDimShuffle{2,0,x,1}"
            "(InplaceDimShuffle{0,x,1}(x))))"
        ), str(g)
        dimshuffle_lift.optimize(g)
        assert str(g) == "FunctionGraph(x)", str(g)
        # no need to check_stack_trace as graph is supposed to be empty

    def test_lift(self):
        x, y, z = inputs([False] * 1, [False] * 2, [False] * 3)
        e = x + y + z
        g = FunctionGraph([x, y, z], [e])

        # It does not really matter if the DimShuffles are inplace
        # or not.
        init_str_g_inplace = (
            "FunctionGraph(Elemwise{add,no_inplace}(InplaceDimShuffle{x,0,1}"
            "(Elemwise{add,no_inplace}(InplaceDimShuffle{x,0}(x), y)), z))"
        )
        init_str_g_noinplace = (
            "FunctionGraph(Elemwise{add,no_inplace}(DimShuffle{x,0,1}"
            "(Elemwise{add,no_inplace}(DimShuffle{x,0}(x), y)), z))"
        )
        assert str(g) in (init_str_g_inplace, init_str_g_noinplace), str(g)

        opt_str_g_inplace = (
            "FunctionGraph(Elemwise{add,no_inplace}(Elemwise{add,no_inplace}"
            "(InplaceDimShuffle{x,x,0}(x), InplaceDimShuffle{x,0,1}(y)), z))"
        )
        opt_str_g_noinplace = (
            "FunctionGraph(Elemwise{add,no_inplace}(Elemwise{add,no_inplace}"
            "(DimShuffle{x,x,0}(x), DimShuffle{x,0,1}(y)), z))"
        )
        dimshuffle_lift.optimize(g)
        assert str(g) in (opt_str_g_inplace, opt_str_g_noinplace), str(g)
        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(g, ops_to_check="all")

    def test_recursive_lift(self):
        v = vector(dtype="float64")
        m = matrix(dtype="float64")
        out = ((v + 42) * (m + 84)).T
        g = FunctionGraph([v, m], [out])
        init_str_g = (
            "FunctionGraph(InplaceDimShuffle{1,0}(Elemwise{mul,no_inplace}"
            "(InplaceDimShuffle{x,0}(Elemwise{add,no_inplace}"
            "(<TensorType(float64, vector)>, "
            "InplaceDimShuffle{x}(TensorConstant{42}))), "
            "Elemwise{add,no_inplace}"
            "(<TensorType(float64, matrix)>, "
            "InplaceDimShuffle{x,x}(TensorConstant{84})))))"
        )
        assert str(g) == init_str_g
        new_out = local_dimshuffle_lift.transform(g, g.outputs[0].owner)[0]
        new_g = FunctionGraph(g.inputs, [new_out])
        opt_str_g = (
            "FunctionGraph(Elemwise{mul,no_inplace}(Elemwise{add,no_inplace}"
            "(InplaceDimShuffle{0,x}(<TensorType(float64, vector)>), "
            "InplaceDimShuffle{x,x}(TensorConstant{42})), "
            "Elemwise{add,no_inplace}(InplaceDimShuffle{1,0}"
            "(<TensorType(float64, matrix)>), "
            "InplaceDimShuffle{x,x}(TensorConstant{84}))))"
        )
        assert str(new_g) == opt_str_g
        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(new_g, ops_to_check="all")

    def test_useless_dimshuffle(self):
        x, _, _ = inputs()
        e = ds(x, (0, 1))
        g = FunctionGraph([x], [e])
        assert str(g) == "FunctionGraph(InplaceDimShuffle{0,1}(x))"
        dimshuffle_lift.optimize(g)
        assert str(g) == "FunctionGraph(x)"
        # Check stacktrace was copied over correctly after opt was applied
        assert hasattr(g.outputs[0].tag, "trace")

    def test_dimshuffle_on_broadcastable(self):
        x, y, z = inputs([False, True], [True, False, True], [False, False, True])
        u = aet.constant(1)
        ds_x = ds(x, (0, "x"))  # useless
        ds_y = ds(y, (2, 1, 0))  # useless
        ds_z = ds(z, (2, 1, 0))  # useful
        ds_u = ds(u, ("x"))  # useful
        g = FunctionGraph([x, y, z, u], [ds_x, ds_y, ds_z, ds_u])
        assert (
            str(g)
            == "FunctionGraph(InplaceDimShuffle{0,x}(x), InplaceDimShuffle{2,1,0}(y), InplaceDimShuffle{2,1,0}(z), InplaceDimShuffle{x}(TensorConstant{1}))"
        )
        dimshuffle_lift.optimize(g)
        assert (
            str(g)
            == "FunctionGraph(x, y, InplaceDimShuffle{2,1,0}(z), InplaceDimShuffle{x}(TensorConstant{1}))"
        )
        # Check stacktrace was copied over correctly after opt was applied
        assert hasattr(g.outputs[0].tag, "trace")


def test_local_useless_dimshuffle_in_reshape():
    vec = TensorType(broadcastable=(False,), dtype="float64")("vector")
    mat = TensorType(broadcastable=(False, False), dtype="float64")("mat")
    row = TensorType(broadcastable=(True, False), dtype="float64")("row")
    col = TensorType(broadcastable=(False, True), dtype="float64")("col")

    reshape_dimshuffle_vector = reshape(vec.dimshuffle("x", 0), vec.shape)
    reshape_dimshuffle_mat = reshape(mat.dimshuffle("x", 0, "x", 1), mat.shape)
    reshape_dimshuffle_row = reshape(row.dimshuffle(1, "x"), row.shape)
    reshape_dimshuffle_col = reshape(col.dimshuffle(0), col.shape)

    g = FunctionGraph(
        [vec, mat, row, col],
        [
            reshape_dimshuffle_vector,
            reshape_dimshuffle_mat,
            reshape_dimshuffle_row,
            reshape_dimshuffle_col,
        ],
    )

    assert str(g) == (
        "FunctionGraph(Reshape{1}(InplaceDimShuffle{x,0}(vector), Shape(vector)), "
        "Reshape{2}(InplaceDimShuffle{x,0,x,1}(mat), Shape(mat)), "
        "Reshape{2}(InplaceDimShuffle{1,x}(row), Shape(row)), "
        "Reshape{2}(InplaceDimShuffle{0}(col), Shape(col)))"
    )
    useless_dimshuffle_in_reshape = out2in(local_useless_dimshuffle_in_reshape)
    useless_dimshuffle_in_reshape.optimize(g)
    assert str(g) == (
        "FunctionGraph(Reshape{1}(vector, Shape(vector)), "
        "Reshape{2}(mat, Shape(mat)), "
        "Reshape{2}(row, Shape(row)), "
        "Reshape{2}(col, Shape(col)))"
    )

    # Check stacktrace was copied over correctly after opt was applied
    assert check_stack_trace(g, ops_to_check="all")

    # Check that the optimization does not get applied when the order
    # of dimensions has changed.
    reshape_dimshuffle_mat2 = reshape(mat.dimshuffle("x", 1, "x", 0), mat.shape)
    h = FunctionGraph([mat], [reshape_dimshuffle_mat2])
    str_h = str(h)
    useless_dimshuffle_in_reshape.optimize(h)
    assert str(h) == str_h


class TestFusion:
    opts = Query(
        include=[
            "local_elemwise_fusion",
            "composite_elemwise_fusion",
            "canonicalize",
            "inplace",
        ],
        exclude=["cxx_only", "BlasOpt"],
    )
    mode = Mode(get_default_mode().linker, opts)
    _shared = staticmethod(shared)
    topo_exclude = ()

    def do(self, mode, shared_fn, shp, nb_repeat=1, assert_len_topo=True, slice=None):
        """
        param shared_fn: if None, will use function
        verify that the elemwise fusion work
        Test with and without DimShuffle
        """
        # TODO: disable the canonizer?
        def my_init(shp, dtype="float64", num=0):
            ret = np.zeros(shp, dtype=dtype) + num
            return ret

        fw, fx, fy, fz = [
            tensor(dtype="float32", broadcastable=[False] * len(shp), name=n)
            for n in "wxyz"
        ]
        dw, dx, dy, dz = [
            tensor(dtype="float64", broadcastable=[False] * len(shp), name=n)
            for n in "wxyz"
        ]
        ix, iy, iz = [
            tensor(dtype="int32", broadcastable=[False] * len(shp), name=n)
            for n in "xyz"
        ]
        fv = fvector("v")
        fs = fscalar("s")

        fwv = my_init(shp, "float32", 1)
        fxv = my_init(shp, "float32", 2)
        fyv = my_init(shp, "float32", 3)
        fzv = my_init(shp, "float32", 4)
        fvv = _asarray(np.random.rand(shp[0]), dtype="float32")
        fsv = np.asarray(np.random.rand(), dtype="float32")
        dwv = my_init(shp, "float64", 5)
        ixv = _asarray(my_init(shp, num=60), dtype="int32")
        iyv = _asarray(my_init(shp, num=70), dtype="int32")
        izv = _asarray(my_init(shp, num=70), dtype="int32")
        fwx = fw + fx
        ftanx = tan(fx)
        cases = [
            (
                fx + fy + fz,
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv + fyv + fzv,
                "float32",
            ),  # 0
            (
                fx * fy * fz,
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv * fyv * fzv,
                "float32",
            ),  # 1
            (
                fx + fy * fz,
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv + fyv * fzv,
                "float32",
            ),  # 2
            (
                fx * fy + fz,
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv * fyv + fzv,
                "float32",
            ),  # 3
            (
                fw + fx + fy + fz,
                (fw, fx, fy, fz),
                (fwv, fxv, fyv, fzv),
                1,
                fwv + fxv + fyv + fzv,
                "float32",
            ),
            (
                (fw + fx) + (fy + fz),
                (fw, fx, fy, fz),
                (fwv, fxv, fyv, fzv),
                1,
                fwv + fxv + fyv + fzv,
                "float32",
            ),  # 5
            (
                ((fw + fx) + fy) + fz,
                (fw, fx, fy, fz),
                (fwv, fxv, fyv, fzv),
                1,
                fwv + fxv + fyv + fzv,
                "float32",
            ),
            (
                (fw + (fx + fy)) + fz,
                (fw, fx, fy, fz),
                (fwv, fxv, fyv, fzv),
                1,
                fwv + fxv + fyv + fzv,
                "float32",
            ),
            (
                (fw + (fx + fy) + fz),
                (fw, fx, fy, fz),
                (fwv, fxv, fyv, fzv),
                1,
                fwv + fxv + fyv + fzv,
                "float32",
            ),
            (
                fw + (fx + (fy + fz)),
                (fw, fx, fy, fz),
                (fwv, fxv, fyv, fzv),
                1,
                fwv + fxv + fyv + fzv,
                "float32",
            ),
            (
                (fw + fx) + (fy + fz),
                (fw, fx, fy, fz),
                (fwv, fxv, fyv, fzv),
                1,
                fwv + fxv + fyv + fzv,
                "float32",
            ),  # 10
            (
                fw * fx * fy * fz,
                (fw, fx, fy, fz),
                (fwv, fxv, fyv, fzv),
                1,
                fwv * fxv * fyv * fzv,
                "float32",
            ),
            (
                fw + fx * fy * fz,
                (fw, fx, fy, fz),
                (fwv, fxv, fyv, fzv),
                1,
                fwv + fxv * fyv * fzv,
                "float32",
            ),
            (
                fx + fy * fz * fx,
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv + fyv * fzv * fxv,
                "float32",
            ),
            (
                fx * fy + fz + fy,
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv * fyv + fzv + fyv,
                "float32",
            ),
            (
                fx * fy * fz * fw + fx + fy + fz + fw,
                (fw, fx, fy, fz),
                (fwv, fxv, fyv, fzv),
                1,
                fxv * fyv * fzv * fwv + fxv + fyv + fzv + fwv,
                "float32",
            ),  # 15
            # test with constant
            (
                (fw + fx) + (fy + fz) + 2.0,
                (fw, fx, fy, fz),
                (fwv, fxv, fyv, fzv),
                1,
                fwv + fxv + fyv + fzv + 2,
                "float32",
            ),
            (
                ((fw + fx) + 2.0 + fy) + fz,
                (fw, fx, fy, fz),
                (fwv, fxv, fyv, fzv),
                1,
                fwv + fxv + fyv + fzv + 2,
                "float32",
            ),
            (
                (fw + (fx + 2.0 + fy)) + fz,
                (fw, fx, fy, fz),
                (fwv, fxv, fyv, fzv),
                1,
                fwv + fxv + fyv + fzv + 2,
                "float32",
            ),
            (
                (fw + (fx + fy) + 2 + fz),
                (fw, fx, fy, fz),
                (fwv, fxv, fyv, fzv),
                1,
                fwv + fxv + fyv + fzv + 2,
                "float32",
            ),
            (
                fw + (fx + (fy + fz) + 2.0),
                (fw, fx, fy, fz),
                (fwv, fxv, fyv, fzv),
                1,
                fwv + fxv + fyv + fzv + 2,
                "float32",
            ),  # 20
            (
                2 + (fw + fx) + (fy + fz),
                (fw, fx, fy, fz),
                (fwv, fxv, fyv, fzv),
                1,
                fwv + fxv + fyv + fzv + 2,
                "float32",
            ),
            # mix float32 and float64
            (
                2 + (dw + fx) + (fy + fz),
                (dw, fx, fy, fz),
                (dwv, fxv, fyv, fzv),
                1,
                dwv + fxv + fyv + fzv + 2,
                "float64",
            ),
            (
                2 + (fw + dw) + (fy + fz),
                (fw, dw, fy, fz),
                (fwv, dwv, fyv, fzv),
                1,
                fwv + dwv + fyv + fzv + 2,
                "float64",
            ),
            (
                2 + (fw + fx) + (dw + fz),
                (fw, fx, dw, fz),
                (fwv, fxv, dwv, fzv),
                1,
                fwv + fxv + dwv + fzv + 2,
                "float64",
            ),
            (
                2 + (fw + fx) + (fy + dw),
                (fw, fx, fy, dw),
                (fwv, fxv, fyv, dwv),
                1,
                fwv + fxv + fyv + dwv + 2,
                "float64",
            ),  # 25
            # test when their is other op then elemwise.
            (
                (fwx.sum()) + (fwx) + (fy + fz),
                (fw, fx, fy, fz),
                (fwv, fxv, fyv, fzv),
                4,
                (fwv + fxv).sum() + fwv + fxv + fyv + fzv,
                "float32",
            ),
            # test other elemwise op
            (
                fx + fy + cos(fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv + fyv + np.cos(fzv),
                "float32",
            ),
            (
                fx + fy + cosh(fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv + fyv + np.cosh(fzv),
                "float32",
            ),
            (
                fx + fy + abs(fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv + fyv + np.absolute(fzv),
                "float32",
            ),
            (
                ix + iy + abs(iz),
                (ix, iy, iz),
                (ixv, iyv, izv),
                1,
                ixv + iyv + np.absolute(izv),
                "int32",
            ),  # 30
            (
                fx + fy + log(fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv + fyv + np.log(fzv),
                "float32",
            ),
            (
                fx + fy + log2(fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv + fyv + np.log2(fzv),
                "float32",
            ),
            (
                fx + fy + log10(fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv + fyv + np.log10(fzv),
                "float32",
            ),
            (
                fx + fy ** fz,
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv + fyv ** fzv,
                "float32",
            ),  # pow
            (
                fx + fy + exp(fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv + fyv + np.exp(fzv),
                "float32",
            ),  # 35
            (
                fx - fy - fz,
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - fyv - fzv,
                "float32",
            ),
            (
                fx - (fy / fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - (fyv / fzv),
                "float32",
            ),
            (
                fx - true_div(fy, 2),
                (fx, fy),
                (fxv, fyv),
                1,
                fxv - (fyv / 2),
                "float32",
            ),
            (
                fx - true_div(fy, fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - (fyv / fzv),
                "float32",
            ),
            (
                fx - int_div(ix * 100, iy * 1000),
                (fx, ix, iy),
                (fxv, ixv, iyv),
                1,
                fxv - ((ixv * 100) // (iyv * 1000)),
                {
                    "custom": "float64",
                    "numpy + floatX": config.floatX,
                    "numpy": "float64",
                },
            ),  # 40
            (fx - (fy / 2), (fx, fy), (fxv, fyv), 1, fxv - (fyv / 2), "float32"),
            (
                fx - (fy % fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - (fyv % fzv),
                "float32",
            ),
            (
                fx - (fy > fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - (fyv > fzv),
                "float32",
            ),
            (
                fx - (fy >= fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - (fyv >= fzv),
                "float32",
            ),
            (
                fx - (fy < fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - (fyv < fzv),
                "float32",
            ),  # 45
            (
                fx - (fy <= fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - (fyv <= fzv),
                "float32",
            ),
            (
                fx - eq(fy, fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - (fyv == fzv),
                "float32",
            ),
            (
                fx - neq(fy, fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - (fyv != fzv),
                "float32",
            ),
            (
                fx - fy + tan(fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - fyv + np.tan(fzv),
                "float32",
            ),
            (
                fx - fy + tanh(fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - fyv + np.tanh(fzv),
                "float32",
            ),  # 50
            (
                fx - fy + sin(fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - fyv + np.sin(fzv),
                "float32",
            ),
            (
                fx - fy + sinh(fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - fyv + np.sinh(fzv),
                "float32",
            ),
            (
                fx - fy + sqr(fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - fyv + (fzv * fzv),
                "float32",
            ),
            (
                fx - fy + sqrt(fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - fyv + np.sqrt(fzv),
                "float32",
            ),
            (
                fx - fy + inv(fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - fyv + (1 / fzv),
                "float32",
            ),  # 55
            (
                fx - fy + neg(fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - fyv + (-fzv),
                "float32",
            ),
            (
                fx - fy + aet_round(fz),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                fxv - fyv + np.round(fzv),
                "float32",
            ),
            (
                ix - iy + iround(fz),
                (ix, iy, fz),
                (ixv, iyv, fzv),
                1,
                ixv - iyv + np.round(fzv),
                "int64",
            ),
            # Bit op
            (
                fx - bitwise_or(iy, iz),
                (fx, iy, iz),
                (fxv, iyv, izv),
                1,
                fxv - (iyv | izv),
                {
                    "custom": "float64",
                    "numpy + floatX": config.floatX,
                    "numpy": "float64",
                },
            ),
            (
                fx - xor(iy, iz),
                (fx, iy, iz),
                (fxv, iyv, izv),
                1,
                fxv - (iyv ^ izv),
                {
                    "custom": "float64",
                    "numpy + floatX": config.floatX,
                    "numpy": "float64",
                },
            ),  # 60
            (
                fx - bitwise_and(iy, iz),
                (fx, iy, iz),
                (fxv, iyv, izv),
                1,
                fxv - (iyv & izv),
                {
                    "custom": "float64",
                    "numpy + floatX": config.floatX,
                    "numpy": "float64",
                },
            ),
            (
                fx - invert(iy),
                (fx, iy),
                (fxv, iyv),
                1,
                fxv - (~iyv),
                {
                    "custom": "float64",
                    "numpy + floatX": config.floatX,
                    "numpy": "float64",
                },
            ),
            (
                fx - aet.cast(fy, dtype="float64"),
                (fx, fy),
                (fxv, fyv),
                1,
                fxv - np.asarray(fyv, "float64"),
                "float64",
            ),
            (
                aet_pow(fx * fy + fz, fx * fy),
                (fx, fy, fz),
                (fxv, fyv, fzv),
                1,
                np.power(fxv * fyv + fzv, fxv * fyv),
                "float32",
            ),
            (
                fv + fy ** fz,
                (fv, fy, fz),
                (fvv, fyv, fzv),
                2,
                fvv + fyv ** fzv,
                "float32",
            ),  # fused with a dimshuffle #65
            (
                fv - fy + tanh(fz),
                (fv, fy, fz),
                (fvv, fyv, fzv),
                2,
                fvv - fyv + np.tanh(fzv),
                "float32",
            ),  # fused with a dimshuffle
            # Cases where the same input is reused many times.
            (
                mul(fx, fx, fx, fx),
                (fx,),
                (fxv,),
                1,
                fxv * fxv * fxv * fxv,
                "float32",
            ),
            (
                mul(fx, ftanx, ftanx),
                (fx,),
                (fxv,),
                1,
                fxv * np.tan(fxv) * np.tan(fxv),
                "float32",
            ),
            (
                mul(fx, ftanx, ftanx, fx),
                (fx,),
                (fxv,),
                1,
                fxv * np.tan(fxv) * np.tan(fxv) * fxv,
                "float32",
            ),
            (
                mul(ftanx, ftanx, fx + fy),
                (fx, fy),
                (fxv, fyv),
                1,
                np.tan(fxv) * np.tan(fxv) * (fxv + fyv),
                "float32",
            ),  # 70
            # Cases with different broadcast pattern. They should not
            # be merged as this would duplicate computation
            # The graph should have 2 elemwise and 1 dimshuffle
            (
                fx * sin(fs),
                (fx, fs),
                (fxv, fsv),
                3,
                fxv * np.sin(fsv),
                "float32",
            ),
        ]
        if slice:
            cases = cases[slice]
        times = np.zeros(len(cases))
        fail1 = []
        fail2 = []
        fail3 = []
        fail4 = []
        for (
            id,
            [g, sym_inputs, val_inputs, nb_elemwise, answer, out_dtype],
        ) in enumerate(cases):
            if isinstance(out_dtype, dict):
                out_dtype = out_dtype[config.cast_policy]

            if shared_fn is None:
                f = function(list(sym_inputs), g, mode=mode)
                for x in range(nb_repeat):
                    out = f(*val_inputs)
                t1 = time.time()
            else:
                out = shared_fn(np.zeros(shp, dtype=out_dtype), "out")
                assert out.dtype == g.dtype
                f = function(sym_inputs, [], updates=[(out, g)], mode=mode)
                t0 = time.time()
                for x in range(nb_repeat):
                    f(*val_inputs)
                t1 = time.time()
                out = out.get_value()

            times[id] = t1 - t0
            atol = 1e-8
            if out_dtype == "float32":
                atol = 1e-6
            if not np.allclose(out, answer * nb_repeat, atol=atol):
                fail1.append(id)
            topo = f.maker.fgraph.toposort()
            topo_ = [n for n in topo if not isinstance(n.op, self.topo_exclude)]
            if assert_len_topo:
                if not len(topo_) == nb_elemwise:
                    fail3.append((id, topo_, nb_elemwise))
                if nb_elemwise == 1:
                    # if no variable appears multiple times in the
                    # input of g,
                    # check that the number of input to the Composite
                    # Elemwise is ok
                    if len(set(g.owner.inputs)) == len(g.owner.inputs):
                        expected_len_sym_inputs = np.sum(
                            [not isinstance(x, Constant) for x in topo_[0].inputs]
                        )
                        assert expected_len_sym_inputs == len(sym_inputs)

            if not out_dtype == out.dtype:
                fail4.append((id, out_dtype, out.dtype))

        assert len(fail1 + fail2 + fail3 + fail4) == 0

        return times

    def test_elemwise_fusion(self):
        shp = (5, 5)
        self.do(self.mode, self._shared, shp)

    def test_fusion_35_inputs(self):
        """Make sure we don't fuse too many `Op`s and go past the 31 function arguments limit."""
        inpts = vectors(["i%i" % i for i in range(35)])

        # Make an elemwise graph looking like:
        # sin(i34 + sin(i33 + sin(... i1 + sin(i0) ...)))
        out = sin(inpts[0])
        for idx in range(1, 35):
            out = sin(inpts[idx] + out)

        with config.change_flags(cxx=""):
            f = function(inpts, out, mode=self.mode)

        # Make sure they all weren't fused
        composite_nodes = [
            node
            for node in f.maker.fgraph.toposort()
            if isinstance(getattr(node.op, "scalar_op", None), aes.basic.Composite)
        ]
        assert not any(len(node.inputs) > 31 for node in composite_nodes)

    @pytest.mark.skipif(not config.cxx, reason="No cxx compiler")
    def test_big_fusion(self):
        # In the past, pickle of Composite generated in that case
        # crashed with max recursion limit. So we were not able to
        # generate C code in that case.
        factors = []
        sd = dscalar()
        means = dvector()

        cst_05 = aet.constant(0.5)
        cst_m05 = aet.constant(-0.5)
        cst_2 = aet.constant(2)
        cst_m2 = aet.constant(-2)
        ones = aet.constant(np.ones(10))
        n = 85
        if config.mode in ["DebugMode", "DEBUG_MODE"]:
            n = 10

        for i in range(n):
            f = cst_m05 * sd ** cst_m2 * (ones - means[i]) ** cst_2 + cst_05 * log(
                cst_05 * (sd ** cst_m2) / np.pi
            )
            factors.append(aet_sum(f))

        logp = add(*factors)

        vars = [sd, means]

        # Make sure that C compilation is used
        mode = Mode("cvm", self.opts)
        dlogp = function(vars, [aesara.grad(logp, v) for v in vars], mode=mode)

        # Make sure something was fused
        assert any(
            isinstance(getattr(node.op, "scalar_op", None), aes.basic.Composite)
            for node in dlogp.maker.fgraph.toposort()
        )

    def test_add_mul_fusion_inplace(self):

        opts = Query(
            include=[
                "local_elemwise_fusion",
                "composite_elemwise_fusion",
                "canonicalize",
                "inplace",
            ],
            exclude=["cxx_only", "BlasOpt"],
        )

        mode = Mode(self.mode.linker, opts)

        x, y, z = dmatrices("xyz")
        out = dot(x, y) + x + y + z
        f = function([x, y, z], out, mode=mode)
        topo = [n for n in f.maker.fgraph.toposort()]
        assert len(topo) == 2
        assert topo[-1].op.inplace_pattern

        new_out = f.maker.fgraph.outputs[0]
        assert isinstance(new_out.owner.op, Elemwise)
        assert isinstance(new_out.owner.op.scalar_op, aes.basic.Add)
        assert len(new_out.owner.inputs) == 4

        # TODO: Do we really need to do this?
        _ = f(
            np.random.random((5, 5)), np.random.random((5, 5)), np.random.random((5, 5))
        )

    def speed_fusion(self, s=None):
        """
        param type s: a slice object
        param s: a slice to apply to the case to execute. If None, exec all case.
        """

        shp = (3000, 3000)
        shp = (1000, 1000)
        nb_repeat = 50
        # linker=CLinker
        # linker=OpWiseCLinker

        mode1 = copy.copy(self.mode)
        mode1._optimizer = mode1._optimizer.including("local_elemwise_fusion")
        # TODO:clinker is much faster... but use to much memory
        # Possible cause: as their is do deletion of intermediate value when we don't keep the fct.
        # More plausible cause: we keep a link to the output data?
        # Follow up. Clinker do the same... second cause?
        mode2 = copy.copy(self.mode)
        mode2._optimizer = mode2._optimizer.excluding("local_elemwise_fusion")
        print("test with linker", str(mode1.linker))
        times1 = self.do(
            mode1,
            self._shared,
            shp,
            nb_repeat=nb_repeat,
            assert_len_topo=False,
            slice=s,
        )
        times2 = self.do(
            mode2,
            self._shared,
            shp,
            nb_repeat=nb_repeat,
            assert_len_topo=False,
            slice=s,
        )
        print("times1 with local_elemwise_fusion")
        print(times1, times1.min(), times1.max(), times1.sum())
        print("times2 without local_elemwise_fusion")
        print(times2, times2.min(), times2.max(), times2.sum())
        d = times2 / times1

        print("times2/times1")
        print(d)
        print(
            "min",
            d.min(),
            "argmin",
            d.argmin(),
            "max",
            d.max(),
            "mean",
            d.mean(),
            "std",
            d.std(),
        )

    def speed_log_exp(self):
        s = slice(31, 36)
        print(
            "time",
            self.do(
                self.mode,
                self._shared,
                shp=(1000, 1000),
                assert_len_topo=False,
                slice=s,
                nb_repeat=100,
            ),
        )

    @pytest.mark.skipif(not config.cxx, reason="No cxx compiler")
    def test_no_c_code(self):
        """Make sure we avoid fusions for `Op`s without C code implementations."""

        # This custom `Op` has no `c_code` method
        class NoCCodeOp(aes.basic.UnaryScalarOp):
            def impl(self, x):
                return x * 2

        no_c_code_op = Elemwise(NoCCodeOp(aes.basic.upgrade_to_float))

        mode = Mode(linker="cvm")
        mode._optimizer = mode._optimizer.including(
            "local_elemwise_fusion",
            "composite_elemwise_fusion",
            "canonicalize",
            "inplace",
        )

        x = vector()
        out = x * no_c_code_op(x + 1)
        f = function([x], out, mode=mode)

        assert not any(
            isinstance(getattr(n.op, "scalar_op"), aes.basic.Composite)
            for n in f.maker.fgraph.toposort()
        )


class TimesN(aes.basic.UnaryScalarOp):
    """
    Used in test TestCompositeCodegen

    Must be outside of the class, otherwise, the c cache code can't
    pickle this class and this cause stuff printing during test.
    """

    def __eq__(self, other):
        return super().__eq__(other) and self.n == other.n

    def __hash__(self):
        return super().__hash__() ^ hash(self.n)

    def __init__(self, n, *args, **kwargs):
        self.n = n
        aes.basic.UnaryScalarOp.__init__(self, *args, **kwargs)

    def impl(self, x):
        return x * self.n

    def c_support_code_apply(self, node, nodename):
        n = str(self.n)
        return (
            """
        float %(nodename)s_timesn(float x) { return x * %(n)s; }
        """
            % locals()
        )

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        return f"{z} = {name}_timesn({x});"


class TestCompositeCodegen:
    """
    Test The Composite Ops code generation in a case where there is multiple
    scalar ops with support code.
    """

    def setup_method(self):
        upgrade_to_float = aes.basic.upgrade_to_float

        self.scal_times_2 = TimesN(2, upgrade_to_float, name="times_2")
        self.times_2 = Elemwise(self.scal_times_2, name="times_2")

        self.scal_times_3 = TimesN(3, upgrade_to_float, name="times_3")
        self.times_3 = Elemwise(self.scal_times_3, name="times_3")

        self.x = fvector()

    def test_nested_composite(self):
        y = self.times_2(self.x)
        z = self.times_3(y)
        f = function([self.x], z)
        if config.mode != "FAST_COMPILE":
            assert len(f.maker.fgraph.toposort()) == 1
        fval = f([1, 2, 3])
        assert np.all(fval == [6, 12, 18])

    def test_local_useless_composite(self):
        x = aes.float32()
        c = aes.Composite([x], [x + 1, x - 1])
        X = matrix()
        o = Elemwise(scalar_op=c)(X)
        mode = get_default_mode().including("local_useless_composite")

        f = function([X], o[0], mode=mode)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert len(topo[0].outputs) == 1
        utt.assert_allclose(f([[1.0]]), [[2.0]])

        f = function([X], o[1], mode=mode)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert len(topo[0].outputs) == 1
        utt.assert_allclose(f([[1.0]]), [[0.0]])


def test_local_useless_slice():
    # test a simple matrix
    x = matrix("x")
    mode_unopt = get_default_mode().excluding(
        "local_useless_slice", "local_mul_canonizer"
    )
    mode_opt = (
        get_default_mode()
        .including("local_useless_slice")
        .excluding("local_mul_canonizer")
    )

    # test with and without the useless slice
    o = 2 * x[0, :]
    f_unopt = function([x], o, mode=mode_unopt)
    f_opt = function([x], o, mode=mode_opt)
    test_inp = np.random.randint(-10, 10, (4, 4)).astype("float32")
    assert all(
        f_opt(test_inp) == f_unopt(test_inp)
    ), "The optimization caused a mismatch in the result"
    # test to see if the slice is truly gone
    apply_node = f_opt.maker.fgraph.toposort()[0]
    subtens = apply_node.op
    assert not any(
        isinstance(idx, slice) for idx in subtens.idx_list
    ), "Slice should be gone"

    # Now test that the stack trace is copied over properly,
    # before before and after optimization.
    assert check_stack_trace(f_unopt, ops_to_check="all")
    assert check_stack_trace(f_opt, ops_to_check="all")

    # test a 4d tensor
    z = tensor4("z")
    o2 = z[1, :, :, 1]
    o3 = z[0, :, :, :]
    f_opt_check = function([z], o2, mode=mode_opt)
    f_opt_check_apply = function([z], o3, mode=mode_opt)

    # The optimization shouldn't apply here
    apply_node = f_opt_check.maker.fgraph.toposort()[0]
    subtens = apply_node.op
    assert [isinstance(idx, slice) for idx in subtens.idx_list].count(True) == 2
    # But it should here
    apply_node = f_opt_check_apply.maker.fgraph.toposort()[0]
    subtens = apply_node.op
    assert not any(isinstance(idx, slice) for idx in subtens.idx_list)

    # Finally, test that the stack trace is copied over properly,
    # before before and after optimization.
    assert check_stack_trace(f_opt_check, ops_to_check=Subtensor)
    assert check_stack_trace(f_opt_check_apply, ops_to_check=Subtensor)


def test_local_useless_inc_subtensor():
    x = matrix("x")
    y = matrix("y")
    mode = get_default_mode().including("local_useless_inc_subtensor")
    for s in [slice(None), slice(None, None, -1)]:
        o = set_subtensor(x[::, s], y)
        f = function([x, y], o, mode=mode)
        o_shape = set_subtensor(x[::, s], specify_shape(y, x.shape))
        f_shape = function([x, y], o_shape, mode=mode)

        # Test with shape info
        topo = f_shape.maker.fgraph.toposort()
        assert not any(isinstance(n.op, IncSubtensor) for n in topo)
        out = f_shape([[2, 3]], [[3, 4]])
        assert (out == np.asarray([[3, 4]])[::, s]).all()

        # Test that without shape info, we don't apply the opt.
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert isinstance(topo[0].op, IncSubtensor)
        out = f([[2, 3]], [[3, 4]])
        assert (out == np.asarray([[3, 4]])[::, s]).all()

        # Test that we don't remove shape error
        with pytest.raises(ValueError):
            f([[2, 3]], [[3, 4], [4, 5]])

        # Test that we don't remove broadcastability
        out = f([[2, 3], [3, 4]], [[5, 6]])
        assert (out == np.asarray([[5, 6], [5, 6]])[::, s]).all()

    # Test that we do not optimize others strides even when sub and y
    # have same shapes
    s = x[::, ::2]
    o_shape = set_subtensor(s, specify_shape(y, s.shape))
    f_shape = function([x, y], o_shape)
    topo = f_shape.maker.fgraph.toposort()
    assert any(isinstance(n.op, IncSubtensor) for n in topo)
    out = f_shape([[2, 3, 6, 7]], [[8, 9]])
    assert (out == np.asarray([[8, 3, 9, 7]])).all()


def test_local_useless_subtensor():
    x = matrix("x")

    # Test default
    for dims in [
        (slice(0, None),),
        (slice(0, None), slice(0, None)),
    ]:
        f = function([x], exp(x).__getitem__(dims), mode=mode_opt)
        prog = f.maker.fgraph.toposort()
        assert prog[0].op == exp
        assert len(prog) == 1
        f([[0, 1, 2], [3, 4, 5]])  # let debugmode test something

    x_c = specify_shape(x, (2, 3))
    # Test constant
    for dims, res in [
        ((slice(0, 2),), True),
        ((slice(0, 2), slice(0, None)), True),
        ((slice(0, 2), slice(0, 3)), True),
        ((slice(0, None), slice(0, 3)), True),
        ((slice(0, 3), slice(0, 13)), True),
        ((slice(0, 3), slice(0, 2)), False),
        ((slice(0, 1), slice(0, None)), False),
        ((slice(0, 1), 1), False),
    ]:
        f = function([x], exp(x_c).__getitem__(dims), mode=mode_opt)
        prog = f.maker.fgraph.toposort()
        if res:
            assert isinstance(prog[0].op, SpecifyShape), dims
            assert prog[1].op == exp, (dims, prog)
            assert len(prog) == 2, dims
        else:
            assert any([isinstance(node.op, Subtensor) for node in prog])
        f([[0, 1, 2], [3, 4, 5]])  # let debugmode test something

    # Test Variable
    for idx, (dims, res) in enumerate(
        [
            ((slice(0, x.shape[0]),), True),
            ((slice(0, x.shape[1]),), False),
            (
                (
                    slice(0, x.shape[0]),
                    slice(0, x.shape[1]),
                ),
                True,
            ),
            (
                (
                    slice(0, x.shape[0]),
                    slice(0, x.shape[0]),
                ),
                False,
            ),
            (
                (
                    slice(0, x.shape[1]),
                    slice(0, x.shape[0]),
                ),
                False,
            ),
            (
                (
                    slice(0, x.shape[1]),
                    slice(0, x.shape[1]),
                ),
                False,
            ),
            ((slice(0, x.shape[1]), 2), False),
            (
                (
                    slice(0, x.shape[1]),
                    slice(x.shape[0] - x.shape[0], x.shape[1]),
                ),
                False,
            ),
            ((slice(0, aet.scalar_from_tensor(x.shape[0])),), True),
        ]
    ):
        f = function([x], exp(x).__getitem__(dims), mode=mode_opt)
        prog = f.maker.fgraph.toposort()
        if res:
            assert prog[0].op == exp, dims
            assert len(prog) == 1, dims
        else:
            assert any([isinstance(node.op, Subtensor) for node in prog])
        f([[0, 1, 2], [3, 4, 5]])  # let debugmode test something
    # Test mix Variable and Constant
    # Currently not supported
    for idx, (dims, res) in enumerate(
        [
            ((slice(0, x.shape[0]), slice(0, 3)), False),
            ((slice(0, 3), slice(0, x.shape[1])), False),
        ]
    ):
        f = function([x], exp(x_c).__getitem__(dims), mode=mode_opt)
        prog = f.maker.fgraph.toposort()
        if res:
            assert prog[0].op == exp, dims
            assert len(prog) == 1, dims
        else:
            assert any([isinstance(node.op, Subtensor) for node in prog])
        f([[0, 1, 2], [3, 4, 5]])  # let debugmode test something

    # Test scalar variable
    s = aes.int32("s")
    for idx, (dims, res) in enumerate(
        [
            ((slice(0, s),), False),
        ]
    ):
        f = function([x, s], exp(x).__getitem__(dims), mode=mode_opt)
        prog = f.maker.fgraph.toposort()
        if res:
            assert prog[0].op == exp, dims
            assert len(prog) == 1, dims
        else:
            assert any([isinstance(node.op, Subtensor) for node in prog])
        f([[1, 2, 3], [4, 5, 6]], 1)
        f([[1, 2, 3], [4, 5, 6]], 3)

    # Test AdvancedSubtensor1 case when all rows are selected by a list/vector
    # or ARange op
    for dims, res in (
        ([0, 1], True),
        ([1, 0], False),
        ([0, 0], False),
        ([0, 0, 1], False),
        (aet.arange(2), True),
        (aet.arange(0, 2), True),
        (aet.arange(0, 2, 2), False),
        (aet.arange(0, 2, -1), False),
        (aet.arange(1, 2), False),
    ):
        f = function([x], exp(x_c).__getitem__(dims), mode=mode_opt)
        prog = f.maker.fgraph.toposort()
        if res:
            assert isinstance(prog[0].op, SpecifyShape), dims
            assert prog[1].op == exp, dims
            assert len(prog) == 2, dims
        else:
            assert any([isinstance(node.op, AdvancedSubtensor1) for node in prog])
        f([[0, 1, 2], [3, 4, 5]])  # let debugmode test something


def test_local_subtensor_remove_broadcastable_index():
    # testing local_subtensor_remove_broadcastable_index optimization
    #
    # tests removing broadcastable dimensions with index 0 or -1,
    # otherwise the optimzation should not be applied

    mode = get_default_mode()
    mode = mode.including("local_subtensor_remove_broadcastable_index")
    x = dmatrix("x")
    y1 = x.dimshuffle(0, "x", 1)
    y2 = x.dimshuffle("x", 1, 0, "x")
    y3 = x.dimshuffle("x", 1, "x", 0, "x")

    # testing for cases that the optimzation should be applied
    z1 = y1[:, 0, :]
    z2 = y1[:, -1, :]
    z3 = y2[0, :, :, -1]
    z4 = y2[0, :, :, 0]
    z5 = y2[-1, :, :, -1]
    z6 = y3[-1, :, 0, :, -1]
    z7 = y3[-1, :, -1, :, -1]
    z8 = y3[0, :, 0, :, 0]
    f = function([x], [z1, z2, z3, z4, z5, z6, z7, z8], mode=mode)
    for elem in f.maker.fgraph.toposort():
        assert type(elem.op) not in [
            Subtensor,
            AdvancedSubtensor,
            AdvancedSubtensor1,
            IncSubtensor,
            AdvancedIncSubtensor,
            AdvancedIncSubtensor1,
        ]

    rng = np.random.RandomState(seed=utt.fetch_seed())
    xn = rng.rand(5, 5)
    f(xn)

    # testing for cases that the optimzation should not be applied
    # to verify that other subtensor usage are passed without errors
    w1 = y1[3, 0, :]
    w2 = y1[2:4, -1, :]
    w3 = y2[0, :, 4:, -1]
    w4 = y2[:, :, 0, -1]
    w5 = y2[0, 2:4, :, 0]
    w6 = y2[0, -1, :, -1]
    w7 = y2[-1, 4:, :, -1]
    w8 = y2[-1, :, :3, -1]
    w9 = y2[-1, :, -1, -1]
    w10 = y3[-1, 2, 0, :, -1]
    w11 = y3[-1, 0, -1, :, -1]
    w12 = y3[-1, :, -1, -1, -1]
    w13 = y3[0, 0, 0, :, 0]
    w14 = y3[-1, 2:4, 0, 1:5, -1]
    w15 = y3[-1, 0, -1, 0, -1]
    w16 = y3[0, 2, 0, 4, 0]
    w17 = y3[:, 0, :, 1]
    w18 = y3[0, :, :, 2]
    w19 = y3[:, 2, 0]
    w20 = y3[:, 3]
    f2 = function(
        [x],
        [
            w1,
            w2,
            w3,
            w4,
            w5,
            w6,
            w7,
            w8,
            w9,
            w10,
            w11,
            w12,
            w13,
            w14,
            w15,
            w16,
            w17,
            w18,
            w19,
            w20,
        ],
        mode=mode,
    )
    f2(xn)


class TestSubtensorIncSubtensor:
    @classmethod
    def setup_class(cls):
        cls.mode = get_default_mode().including("local_subtensor_inc_subtensor")

    @pytest.mark.parametrize(
        "val, indices, optype",
        [
            (vector(), (iscalar(),), IncSubtensor),
            (vector(), (ivector(),), AdvancedIncSubtensor1),
            (vector(), (ivector(), ivector()), AdvancedIncSubtensor),
        ],
    )
    def test_inplace(self, val, indices, optype):
        x = matrix("x")
        y = set_subtensor((2 * x)[indices], val, inplace=False)
        assert isinstance(y.owner.op, optype)
        assert y.owner.op.inplace is False
        f = function(
            [x, val] + list(indices), y, mode=get_default_mode().including("inplace")
        )
        assert f.maker.fgraph.outputs[0].owner.op.inplace is True

    def test_basic(self):
        # basic test
        x = matrix("x")
        i = iscalar("i")
        v = vector("v")
        y = set_subtensor(x[i], v)
        z = y[i]
        f = function([x, i, v], z, mode=self.mode)
        prog = f.maker.fgraph.toposort()
        assert len(prog) == 1
        assert isinstance(prog[0].op, DeepCopyOp)
        # basic test, numerical check
        x_ = np.random.uniform(size=[3, 4]).astype(config.floatX)
        v_ = np.random.uniform(
            size=[
                4,
            ]
        ).astype(config.floatX)
        i_ = 1
        assert np.array_equal(f(x_, i_, v_), v_)

    def test_multiple_idx(self):
        # complicated test
        x = tensor4("x")
        i1 = iscalar("i1")
        i2 = iscalar("i2")
        i3 = iscalar("i3")
        i4 = iscalar("i4")
        v = tensor3("v")
        y = set_subtensor(x[i1, :i2, i3:, ::i4], v)
        z = y[i1, :i2, i3:, ::i4]
        f = function([x, i1, i2, i3, i4, v], z, mode=self.mode)
        prog = f.maker.fgraph.toposort()
        assert len(prog) == 1
        assert isinstance(prog[0].op, DeepCopyOp)
        # complicated test, numerical check
        x_ = np.random.uniform(size=[3, 4, 5, 6]).astype(config.floatX)
        v_ = np.random.uniform(size=[2, 2, 2]).astype(config.floatX)
        i1_, i2_, i3_, i4_ = 1, 2, 3, 4
        assert np.array_equal(f(x_, i1_, i2_, i3_, i4_, v_), v_)

    def test_not_applied(self):
        # case not use this optimization
        x = tensor4("x")
        i1 = iscalar("i1")
        i2 = iscalar("i2")
        i3 = iscalar("i3")
        i4 = iscalar("i4")
        v = tensor3("v")
        y = set_subtensor(x[i1, :i2, i3:, ::i4], v)
        z = y[i1, :i3, i2:, ::i4]
        f = function([x, i1, i2, i3, i4, v], z, mode=self.mode)
        prog = f.maker.fgraph.toposort()
        assert len(prog) != 1
        assert any(isinstance(x.op, IncSubtensor) for x in prog)
        assert any(isinstance(x.op, Subtensor) for x in prog)
        # case not use this optimization, numerical check
        x_ = np.random.uniform(size=[3, 4, 5, 6]).astype(config.floatX)
        v_ = np.random.uniform(size=[2, 2, 2]).astype(config.floatX)
        i1_, i2_, i3_, i4_ = 1, 2, 3, 4
        x_[i1_, :i2_, i3_:, ::i4_] = v_
        assert np.array_equal(f(x_, i1_, i2_, i3_, i4_, v_), x_[i1_, :i3_, i2_:, ::i4_])

    def test_fewer_dims(self):
        # case when v has fewer dimensions
        x = matrix("x")
        i1 = iscalar("i")
        i2 = iscalar("i")
        v = vector("v")
        y = set_subtensor(x[:i1, :i2], v)
        z = y[:i1, :i2]
        f = function([x, i1, i2, v], z, mode=self.mode)
        prog = f.maker.fgraph.toposort()
        assert any(isinstance(x.op, Alloc) for x in prog)
        # case when v is broadcastable, numerical check
        x_ = np.random.uniform(size=[3, 4]).astype(config.floatX)
        v_ = np.random.uniform(
            size=[
                2,
            ]
        ).astype(config.floatX)
        i1_, i2_ = 2, 2
        x_[:i1_, :i2_] = v_
        assert np.array_equal(f(x_, i1_, i2_, v_), x_[:i1_, :i2_])

    def test_broadcasted(self):
        # case when v has the same number of dimensions, some broadcastable
        x = matrix("x")
        i1 = iscalar("i")
        i2 = iscalar("i")
        v = col("v")
        y = set_subtensor(x[:i1, :i2], v)
        z = y[:i1, :i2]
        f = function([x, i1, i2, v], z, mode=self.mode)
        prog = f.maker.fgraph.toposort()
        assert any(isinstance(x.op, Alloc) for x in prog)
        # case when v is broadcastable, numerical check
        x_ = np.random.uniform(size=[3, 4]).astype(config.floatX)
        v_ = np.random.uniform(size=[2, 1]).astype(config.floatX)
        i1_, i2_ = 2, 2
        x_[:i1_, :i2_] = v_
        assert np.array_equal(f(x_, i1_, i2_, v_), x_[:i1_, :i2_])

    def test_different_dtypes(self):
        # Case when the dtype differs
        x = bmatrix("x")
        i = iscalar("i")
        v = vector("v")
        y = set_subtensor(x[i], v)
        z = y[i]
        f = function([x, i, v], z, mode=self.mode)
        prog = f.maker.fgraph.toposort()
        assert len(prog) == 1
        assert prog[0].op == _convert_to_int8
        # basic test, numerical check
        x_ = np.random.randint(12, size=[3, 4]).astype("int8")
        v_ = np.random.uniform(
            12,
            size=[
                4,
            ],
        ).astype(config.floatX)
        i_ = 1
        assert np.array_equal(f(x_, i_, v_), v_.astype("int8"))


class TestLocalSubtensorMakeVector:
    def test_scalar_idx(self):
        x, y, z = lscalars("xyz")
        v = make_vector(x, y, z)
        f = function([x, y, z], v[0], mode=mode_opt)

        prog = f.maker.fgraph.toposort()
        assert len(prog) == 1
        assert isinstance(prog[0].op, DeepCopyOp)
        assert f(0, 1, 2) == 0

    def test_slice_idx_stop(self):
        x, y, z = lscalars("xyz")
        v = make_vector(x, y, z)
        f = function([x, y, z], v[:2], mode=mode_opt)

        prog = f.maker.fgraph.toposort()
        assert len(prog) == 1
        assert isinstance(prog[0].op, MakeVector)
        assert len(prog[0].inputs) == 2
        r = f(0, 1, 2)
        assert r[0] == 0 and r[1] == 1

    def test_slice_idx_step(self):
        x, y, z = lscalars("xyz")
        v = make_vector(x, y, z)
        f = function([x, y, z], v[::2], mode=mode_opt)

        prog = f.maker.fgraph.toposort()
        assert len(prog) == 1
        assert isinstance(prog[0].op, MakeVector)
        assert len(prog[0].inputs) == 2
        r = f(0, 1, 2)
        assert r[0] == 0 and r[1] == 2

    def test_AdvancedSubtensor1_idx(self):
        x, y, z = lscalars("xyz")
        v = make_vector(x, y, z)
        f = function([x, y, z], v[[0, 2]], mode=mode_opt)

        prog = f.maker.fgraph.toposort()
        assert len(prog) == 1
        assert isinstance(prog[0].op, MakeVector)
        assert len(prog[0].inputs) == 2
        r = f(0, 1, 2)
        assert r[0] == 0 and r[1] == 2

    @pytest.mark.xfail(
        reason="local_subtensor_make_vector doesn't handle all index cases"
    )
    def test_MakeVector_idx(self):
        x, y, z, q = lscalars("xyzq")
        v = make_vector(x, y, z)
        q = make_vector(0, 2)
        f = function([x, y, z], v[q], mode=mode_opt)

        prog = f.maker.fgraph.toposort()
        assert len(prog) == 1
        assert isinstance(prog[0].op, MakeVector)
        assert len(prog[0].inputs) == 2
        r = f(0, 1, 2)
        assert r[0] == 0 and r[1] == 2

    def test_stack_trace(self):
        x, y, z = lscalars("xyz")
        v = make_vector(x, y, z)

        mode = get_default_mode().including("local_subtensor_make_vector")

        # list of subtensor cases, where local_subtensor_make_vector
        # inserts a new MakeVector node
        v_subtensors = [v[:2], v[::2], v[[0, 2]]]

        for v_subtensor in v_subtensors:
            f = function([x, y, z], v_subtensor, mode=mode)
            assert check_stack_trace(f, ops_to_check="all")


class TestLocalSubtensorLift:
    def test_basic(self):
        # basic test that the Op works
        x = matrix("x")
        f = function([x], exp(x)[0], mode=mode_opt)

        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f, ops_to_check="all")

        prog = f.maker.fgraph.toposort()
        assert isinstance(prog[0].op, Subtensor)  # first subtensor
        assert prog[1].op == exp
        assert len(prog) == 2
        f([[0, 1], [2, 3]])  # let debugmode test something

    def test_basic_1(self):
        # as test0, but we reuse the output of the elemwise
        # So we should not lift the subtensor
        x = matrix("x")
        f = function([x], [exp(x)[0], exp(x)], mode=mode_opt)

        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f, ops_to_check=[Subtensor, Elemwise])

        prog = f.maker.fgraph.toposort()
        assert prog[0].op == exp
        assert isinstance(prog[1].op, Subtensor)  # first subtensor
        assert isinstance(prog[2].op, DeepCopyOp)
        assert len(prog) == 3
        f([[0, 1], [2, 3]])  # let debugmode test something

    def test_basic_2(self):
        # basic test that the optimization work with scalar broadcasted
        x = matrix("x")
        y = scalar("y")
        z = matrix("z")
        f = function([x, y, z], exp(x + y + z)[0], mode=mode_opt)

        prog = f.maker.fgraph.toposort()
        assert isinstance(prog[0].op, Subtensor)
        assert isinstance(prog[1].op, DimShuffle)
        assert isinstance(prog[2].op, Subtensor)
        assert isinstance(prog[3].op.scalar_op, aes.Composite)  # Composite{add,add}
        assert len(prog) == 4

        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f, ops_to_check=[Subtensor])

        # let debugmode test something
        f([[0, 1], [2, 3]], 4, [[4, 5], [6, 7]])

    def test_basic_3(self):
        # as 1, but take a slice
        x = matrix("x")
        y = scalar("y")
        z = matrix("z")
        f = function([x, y, z], exp(x + y + z)[0:2], mode=mode_opt)

        prog = f.maker.fgraph.toposort()
        assert isinstance(prog[0].op, Subtensor)
        assert isinstance(prog[1].op, DimShuffle)
        assert isinstance(prog[2].op, Subtensor)
        assert isinstance(prog[3].op.scalar_op, aes.Composite)  # Composite{add,add}
        assert len(prog) == 4

        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f, ops_to_check=[Subtensor])

        # let debugmode test something
        f([[0, 1], [2, 3]], 4, [[4, 5], [6, 7]])

    def test_basic_4(self):
        # basic test that the optimization does work with broadcasting
        # for unary elemwise.
        y = vector("y")
        f = function([y], exp(y.dimshuffle(0, "x"))[0], mode=mode_opt)

        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f, ops_to_check="all")

        prog = f.maker.fgraph.toposort()
        assert isinstance(prog[0].op, DimShuffle)
        assert isinstance(prog[1].op, Subtensor)
        assert prog[2].op == exp
        assert len(prog) == 3
        f([4, 5])  # let debugmode test something

    @utt.assertFailure_fast
    def test_basic_5(self):
        # basic test that the optimization doesn't work with broadcasting
        # ... It *could* be extended to,
        # ... but right now it doesn't, so it shouldn't try.
        x = matrix("x")
        y = vector("y")
        f = function([x, y], exp(x + y)[0], mode=mode_opt)

        # Opt doesn't apply, so no need for check_stack_trace
        # assert check_stack_trace(f, ops_to_check='all')

        prog = f.maker.fgraph.toposort()
        assert isinstance(prog[0].op, DimShuffle)
        assert prog[1].op == add
        assert isinstance(prog[2].op, Subtensor)  # first subtensor
        assert prog[3].op == inplace.exp_inplace
        assert len(prog) == 4
        f([[0, 1], [2, 3]], [4, 5])  # let debugmode test something

    def test_basic_6(self):
        # test that we don't lift when we reuse the output of the
        # elemwise for other computation.
        x = matrix("x")
        y = vector("y")
        f = function([x, y], [exp(x + y)[0], exp(x + y) + x], mode=mode_opt)

        # Opt doesn't apply, so no need for check_stack_trace
        # assert check_stack_trace(f, ops_to_check=Subtensor)

        prog = f.maker.fgraph.toposort()
        assert isinstance(prog[0].op, DimShuffle)
        assert isinstance(prog[1].op.scalar_op, aes.Composite)  # Composite{add,exp}
        assert prog[2].op == add or prog[3].op == add
        # first subtensor
        assert isinstance(prog[2].op, Subtensor) or isinstance(prog[3].op, Subtensor)
        assert len(prog) == 4
        f([[0, 1], [2, 3]], [4, 5])  # let debugmode test something

    def test_basic_7(self):
        # basic test that the optimization works with a scalar as input,
        # and a scalar as output (no broadcasting of the scalar needed).
        # The optimization used to fail and display an ERROR message.

        x = vector("x")
        y = scalar("y")
        f = function([x, y], exp(x + y)[0], mode=mode_opt)

        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f, ops_to_check=Subtensor)

        prog = f.maker.fgraph.toposort()
        assert isinstance(prog[0].op, Subtensor)
        # Composite{add,exp}
        assert isinstance(prog[1].op.scalar_op, aes.Composite)
        assert len(prog) == 2
        f([1, 2, 3], 4)  # let debugmode test something

    def test_basic_8(self):
        # Test that Subtensor(Rebroadcast(x)) gets optimized into
        # Rebroadcast(Subtensor(x)).

        # test basic case
        x = matrix("x")
        xval = np.random.rand(1, 10).astype(config.floatX)
        assert x.broadcastable == (False, False)
        newx = Rebroadcast((0, True), (1, False))(x)
        assert newx.broadcastable == (True, False)

        f1 = function([x], newx[:2, :5], mode=mode_opt)
        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f1, ops_to_check=[Subtensor, Rebroadcast])
        prog = f1.maker.fgraph.toposort()
        assert isinstance(prog[0].op, Subtensor)
        assert isinstance(prog[1].op, Rebroadcast)
        assert (f1(xval) == xval[:2, :5]).all()

        # corner case 1: rebroadcast changes dims which are dropped through subtensor
        y = tensor4("x")
        yval = np.random.rand(1, 10, 1, 3).astype(config.floatX)
        assert y.broadcastable == (False, False, False, False)
        newy = Rebroadcast((0, True), (2, True))(y)
        assert newy.broadcastable == (True, False, True, False)

        f2 = function([y], newy[:, 3, 0, :], mode=mode_opt)
        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f2, ops_to_check=[Subtensor, Rebroadcast])
        prog = f2.maker.fgraph.toposort()
        assert isinstance(prog[0].op, Subtensor)
        assert isinstance(prog[1].op, Rebroadcast)
        assert (f2(yval) == yval[:, 3, 0, :]).all()

        # corner case 2: subtensor idx_list is shorter than resulting broadcast pattern
        f3 = function([y], newy[:, 3, 0], mode=mode_opt)
        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f3, ops_to_check=[Subtensor, Rebroadcast])
        prog = f3.maker.fgraph.toposort()
        assert isinstance(prog[0].op, Subtensor)
        assert isinstance(prog[1].op, Rebroadcast)
        assert (f3(yval) == yval[:, 3, 0]).all()

        # corner case 3: subtensor idx_list is shorter than rebroadcast.axis
        z = tensor4("x")
        zval = np.random.rand(4, 10, 3, 1).astype(config.floatX)
        assert z.broadcastable == (False, False, False, False)
        newz = Rebroadcast((3, True))(z)
        assert newz.broadcastable == (False, False, False, True)

        f4 = function([z], newz[:, 3, 0], mode=mode_opt)
        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f4, ops_to_check=[Subtensor, Rebroadcast])
        prog = f4.maker.fgraph.toposort()
        assert isinstance(prog[0].op, Subtensor)
        assert isinstance(prog[1].op, Rebroadcast)
        assert (f4(zval) == zval[:, 3, 0]).all()


class TestLocalSubtensorMerge:
    def setup_method(self):
        utt.seed_rng()
        self.x_shapes = [(2, 2), (5, 3), (4, 1), (1, 2), (0, 2), (2, 0), (1, 0), (0, 0)]
        self.rng = np.random.RandomState(seed=utt.fetch_seed())

    def test_const(self):
        # var[const::][-1] -> var[-1]
        x = matrix("x")
        for idx in range(-7, 6):
            f = function([x], x[idx::][-1], mode=mode_opt)
            g = function(
                [x], x[idx::][-1], mode=mode_opt.excluding("local_subtensor_merge")
            )

            # Check stacktrace was copied over correctly after opt was applied
            assert check_stack_trace(f, ops_to_check=Subtensor)

            topo = f.maker.fgraph.toposort()
            assert len([t for t in topo if isinstance(t.op, Subtensor)]) == 1
            assert isinstance(topo[-1].op, DeepCopyOp)

            for x_s in self.x_shapes:
                x_val = self.rng.uniform(size=x_s).astype(config.floatX)

                if idx < x_s[0] and x_s[0] > 0:
                    # The first subtensor is non-empty, so it makes sense
                    f(x_val)  # let debugmode test something
                else:
                    # A non-empty subtensor of an empty one should be
                    # an IndexError
                    with pytest.raises(IndexError):
                        f(x_val)
                    with pytest.raises(IndexError):
                        g(x_val)

    def test_scalar(self):
        # var[int::][-1] -> var[-1]
        x = matrix("x")
        y = iscalar("y")
        f = function([x, y], x[y::][-1], mode=mode_opt)
        g = function(
            [x, y], x[y::][-1], mode=mode_opt.excluding("local_subtensor_merge")
        )

        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f, ops_to_check=Subtensor)

        topo = f.maker.fgraph.toposort()
        assert len([t for t in topo if isinstance(t.op, Subtensor)]) == 1
        assert isinstance(topo[-1].op, DeepCopyOp)

        for x_s in self.x_shapes:
            x_val = self.rng.uniform(size=x_s).astype(config.floatX)

            for idx in range(-9, 8):
                if (idx < x_s[0]) and (x_s[0] > 0):
                    # The first subtensor is non-empty
                    f(x_val, idx)  # let debugmode test something
                else:
                    with pytest.raises(IndexError):
                        f(x_val, idx)
                    with pytest.raises(IndexError):
                        g(x_val, idx)

    @pytest.mark.slow
    def test_const2(self):
        # var[::-1][const] -> var[-1]
        x = matrix("x")
        for idx in range(-8, 7):
            f = function([x], x[::-1][idx], mode=mode_opt)
            g = function(
                [x], x[::-1][idx], mode=mode_opt.excluding("local_subtensor_merge")
            )

            # Check stacktrace was copied over correctly after opt was applied
            assert check_stack_trace(f, ops_to_check=Subtensor)

            topo = f.maker.fgraph.toposort()
            assert len([t for t in topo if isinstance(t.op, Subtensor)]) == 1
            assert isinstance(topo[-1].op, DeepCopyOp)

            for x_s in self.x_shapes:
                x_val = self.rng.uniform(size=x_s).astype(config.floatX)
                if (idx < x_s[0]) and (idx >= -x_s[0]):
                    # The first subtensor is non-empty, so it makes sense
                    f(x_val)  # let debugmode test something
                else:
                    # A non-empty subtensor of an empty one should be
                    # an IndexError
                    with pytest.raises(IndexError):
                        f(x_val)
                    with pytest.raises(IndexError):
                        g(x_val)

    def test_scalar2(self):
        # var[::-1][int] -> var[-1]
        x = matrix("x")
        y = iscalar("y")
        f = function([x, y], x[::-1][y], mode=mode_opt)
        g = function(
            [x, y], x[::-1][y], mode=mode_opt.excluding("local_subtensor_merge")
        )

        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f, ops_to_check=Subtensor)

        topo = f.maker.fgraph.toposort()
        assert len([t for t in topo if isinstance(t.op, Subtensor)]) == 1
        assert isinstance(topo[-1].op, DeepCopyOp)

        for x_s in self.x_shapes:
            x_val = self.rng.uniform(size=x_s).astype(config.floatX)

            for idx in range(-x_s[0], x_s[0]):
                f(x_val, idx)  # let debugmode test something
            for idx in list(range(x_s[0], 9)) + list(range(-9, -x_s[0])):
                with pytest.raises(IndexError):
                    f(x_val, idx)
                with pytest.raises(IndexError):
                    g(x_val, idx)

    def test_const3(self):
        # var[::-1][:const] -> var[-1]
        x = matrix("x")
        for idx in range(-9, 8):
            f = function([x], x[::-1][:idx], mode=mode_opt)

            # Check stacktrace was copied over correctly after opt was applied
            assert check_stack_trace(f, ops_to_check=Subtensor)

            topo = f.maker.fgraph.toposort()
            assert len([t for t in topo if isinstance(t.op, Subtensor)]) == 1
            assert isinstance(topo[-1].op, DeepCopyOp)

            for x_s in self.x_shapes:
                x_val = self.rng.uniform(size=x_s).astype(config.floatX)
                f(x_val)  # let debugmode test something

    def test_scalar3(self):
        # var[::-1][:int] -> var[-1]
        x = matrix("x")
        y = iscalar("y")
        f = function([x, y], x[::-1][:y], mode=mode_opt)

        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f, ops_to_check=Subtensor)

        topo = f.maker.fgraph.toposort()
        assert len([t for t in topo if isinstance(t.op, Subtensor)]) == 1
        assert isinstance(topo[-1].op, DeepCopyOp)

        for x_s in self.x_shapes:
            x_val = self.rng.uniform(size=x_s).astype(config.floatX)
            for idx in range(-7, 7):
                f(x_val, idx)  # let debugmode test something

    def test_const4(self):
        # var[const1::][:const2]
        x = matrix("x")
        for idx1 in range(-7, 7):
            for idx2 in range(-7, 7):
                f = function([x], x[idx1:][:idx2], mode=mode_opt)

                # Check stacktrace was copied over correctly after opt was applied
                assert check_stack_trace(f, ops_to_check=Subtensor)

                topo = f.maker.fgraph.toposort()
                assert len([t for t in topo if isinstance(t.op, Subtensor)]) == 1
                assert isinstance(topo[-1].op, DeepCopyOp)

                for x_s in self.x_shapes:
                    x_val = self.rng.uniform(size=x_s).astype(config.floatX)
                    f(x_val)  # let debugmode test something

    def test_scalar4(self):
        # var[int1:][:int2]
        x = matrix("x")
        y = iscalar("y")
        z = iscalar("y")
        f = function([x, y, z], x[y:][:z], mode=mode_opt)

        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f, ops_to_check=Subtensor)

        topo = f.maker.fgraph.toposort()
        assert len([t for t in topo if isinstance(t.op, Subtensor)]) == 1
        assert isinstance(topo[-1].op, DeepCopyOp)

        for x_s in self.x_shapes:
            x_val = self.rng.uniform(size=x_s).astype(config.floatX)
            for idx1 in range(-11, 11):
                for idx2 in range(-11, 11):
                    f(x_val, idx1, idx2)  # let debugmode test something

    def test_const_general(self):
        # Some cases of merge: shape, (start, stop, step) of first,
        # (start, stop, step) of second subtensor
        cases = [
            ((2, 3), (None, None, None), (None, None, -1)),
            ((12, 1), (None, None, -4), (None, None, 1)),
            ((5, 3), (1, 4, 2), (None, None, -1)),
        ]
        x = matrix("x")

        for shape, sl1, sl2 in cases:
            z = x[slice(*sl1)][slice(*sl2)]
            f = function([x], z, mode=mode_opt)

            # Check stacktrace was copied over correctly after opt was applied
            assert check_stack_trace(f, ops_to_check=Subtensor)

            x_val = self.rng.uniform(size=shape).astype(config.floatX)
            f(x_val)

    def test_scalar5(self):
        # General case with two real slices
        # var[b1:e1:s1][b2:e2:s2]
        x = matrix("x")
        b1 = iscalar("b1")
        e1 = iscalar("e1")
        s1 = iscalar("s1")
        b2 = iscalar("b2")
        e2 = iscalar("e2")
        s2 = iscalar("s2")
        f = function([x, b1, e1, s1, b2, e2, s2], x[b1:e1:s1][b2:e2:s2], mode=mode_opt)

        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f, ops_to_check=Subtensor)

        topo = f.maker.fgraph.toposort()
        assert len([t for t in topo if isinstance(t.op, Subtensor)]) == 1
        assert isinstance(topo[-1].op, DeepCopyOp)

        b1r = self.rng.permutation(list(range(-8, 8)))[:2]
        e1r = self.rng.permutation(list(range(-8, 8)))[:2]
        b2r = self.rng.permutation(list(range(-8, 8)))[:2]
        e2r = self.rng.permutation(list(range(-8, 8)))[:2]

        s1r = self.rng.permutation([-7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7])[
            :2
        ]
        s2r = self.rng.permutation([-7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7])[
            :2
        ]

        for x_s in self.x_shapes:
            x_val = self.rng.uniform(size=x_s).astype(config.floatX)
            for b1 in b1r:
                for e1 in e1r:
                    for s1 in s1r:
                        for b2 in b2r:
                            for e2 in e2r:
                                for s2 in s2r:
                                    f(x_val, b1, e1, s1, b2, e2, s2)

    def test_const5(self):
        # Bug reported by Razvan
        data = np.asarray(np.arange(8), dtype=config.floatX)
        x = vector("x")
        y = x[7:1:-1]
        t = shared(np.int64(0))

        fun = function([x], y[t])

        val = fun(data)
        assert val == data[7:1:-1][0]

    def test_const6(self):
        # Bug reported by Graham
        data = self.rng.uniform(size=(8, 8, 8)).astype(config.floatX)
        x = tensor3("x")

        nops = 1
        if config.mode == "FAST_COMPILE":
            nops = 2

        # test 1)
        y = x[3:6, 2:6, 1:7][1]
        fun = function([x], y)
        val = fun(data)
        assert np.all(val == data[3:6, 2:6, 1:7][1])
        assert (
            len([n for n in fun.maker.fgraph.toposort() if isinstance(n.op, Subtensor)])
            == nops
        )

        # test 2)
        y = x[2, 3][1]
        fun = function([x], y)
        val = fun(data)
        assert np.all(val == data[2, 3][1])
        assert (
            len([n for n in fun.maker.fgraph.toposort() if isinstance(n.op, Subtensor)])
            == nops
        )

        # test 3)
        y = x[3:6, 2, 1:7][1]
        fun = function([x], y)
        val = fun(data)
        assert np.all(val == data[3:6, 2, 1:7][1])
        assert (
            len([n for n in fun.maker.fgraph.toposort() if isinstance(n.op, Subtensor)])
            == nops
        )

    def test_scalar6(self):
        # General case with one slice and one index
        # var[b:e:s][i]
        x = matrix("x")
        b = iscalar("b")
        e = iscalar("e")
        s = iscalar("s")
        i = iscalar("i")
        f = function([x, b, e, s, i], x[b:e:s][i], mode=mode_opt)

        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f, ops_to_check=Subtensor)

        topo = f.maker.fgraph.toposort()
        assert len([t for t in topo if isinstance(t.op, Subtensor)]) == 1
        assert isinstance(topo[-1].op, DeepCopyOp)

        b_r = self.rng.permutation(list(range(-4, 4)))[:3]
        e_r = self.rng.permutation(list(range(-4, 4)))[:3]
        i_r = self.rng.permutation(list(range(-4, 4)))[:3]

        s_r = self.rng.permutation([-3, -2, -1, 1, 2, 3])[:3]

        for x_s in self.x_shapes:
            n_index_err = 0
            n_ok = 0
            x_val = self.rng.uniform(size=x_s).astype(config.floatX)
            for b_v in b_r:
                for e_v in e_r:
                    for s_v in s_r:
                        for i_v in i_r:
                            # The index could be out of bounds
                            # In that case, an Exception should be raised,
                            # otherwise, we let DebugMode check f
                            try:
                                x_val[b_v:e_v:s_v][i_v]
                            except IndexError:
                                n_index_err += 1
                                with pytest.raises(IndexError):
                                    f(x_val, b_v, e_v, s_v, i_v)
                            else:
                                # Executed if the "try" clause did not raise
                                # any exception
                                n_ok += 1
                                f(x_val, b_v, e_v, s_v, i_v)

    @pytest.mark.slow
    def test_none_slice(self):
        # Test case of two slices, var[b1:e1:s1][b2:e2:s2]
        # where any of the b, e, and s can be None
        x = matrix("x")
        b1 = iscalar("b1")
        e1 = iscalar("e1")
        s1 = iscalar("s1")
        b2 = iscalar("b2")
        e2 = iscalar("e2")
        s2 = iscalar("s2")

        # Generate all possible lists of positions for None in those 6 slots
        # A 1 indicates None is present, 0 that there is an Aesara scalar.
        none_positions = np.ndindex(2, 2, 2, 2, 2, 2)

        # Ranges to be used when not None
        b1r = self.rng.permutation(list(range(-4, 4)))[:]
        e1r = self.rng.permutation(list(range(-4, 4)))[:]
        b2r = self.rng.permutation(list(range(-4, 4)))[:]
        e2r = self.rng.permutation(list(range(-4, 4)))[:]
        s1r = self.rng.permutation([-4, -3, -2, -1, 1, 2, 3, 4])[:]
        s2r = self.rng.permutation([-4, -3, -2, -1, 1, 2, 3, 4])[:]

        scalar_vars = [b1, e1, s1, b2, e2, s2]
        scalar_ranges = [b1r, e1r, s1r, b2r, e2r, s2r]

        # For each case, we will build a graph, function, and list of values
        # Then, we test it on each input shape.
        for none_pos in none_positions:
            slice_inputs = []
            input_vars = []
            values = []
            if sum(none_pos) == 0:
                # Those case are already tested in test_scalar4
                continue

            for i, none_i in enumerate(none_pos):
                if none_i:
                    slice_inputs.append(None)
                else:
                    slice_inputs.append(scalar_vars[i])
                    input_vars.append(scalar_vars[i])
                    values.append(scalar_ranges[i])

            slice1 = slice(*slice_inputs[:3])
            slice2 = slice(*slice_inputs[3:])
            sub_x = x[slice1][slice2]
            f = function([x] + input_vars, sub_x, mode=mode_opt)

            # Check stacktrace was copied over correctly after opt was applied
            # for some cases, the optimization may remove all Subtensors,
            # which is why we pass "bug_print='ignore'".
            assert check_stack_trace(f, ops_to_check=Subtensor, bug_print="ignore")

            topo = f.maker.fgraph.toposort()
            assert len([t for t in topo if isinstance(t.op, Subtensor)]) <= 1
            assert isinstance(topo[-1].op, DeepCopyOp)

            for x_s in self.x_shapes:
                x_val = self.rng.uniform(size=x_s).astype(config.floatX)
                for i_val in zip(*values):
                    f(x_val, *i_val)

    def test_none_index(self):
        # Test the general case of indexing into a subvector,
        # like x[b:e:s][i], where any of b, e, and s can be None
        x = matrix("x")
        b = iscalar("b")
        e = iscalar("e")
        s = iscalar("s")
        i = iscalar("i")

        # Generate all possible lists of positions for None in those 6 slots
        # A 1 indicates None is present, 0 that there is an Aesara scalar.
        # The last index (i) is never None
        none_positions = np.ndindex(2, 2, 2, 1)

        # Ranges to be used when not None
        b_r = self.rng.permutation(list(range(-4, 4)))[:]
        e_r = self.rng.permutation(list(range(-4, 4)))[:]
        i_r = self.rng.permutation(list(range(-4, 4)))[:]
        s_r = self.rng.permutation([-4, -3, -2, -1, 1, 2, 3, 4])[:]

        scalar_vars = [b, e, s, i]
        scalar_ranges = [b_r, e_r, s_r, i_r]

        # For each case, we will build a graph, function, and list of values
        # Then, we test it on each input shape.
        for none_pos in none_positions:
            slice_inputs = []
            input_vars = []
            values = []
            if sum(none_pos) == 0:
                # Those case are already tested in test_scalar6
                continue

            for j, none_j in enumerate(none_pos):
                if none_j:
                    slice_inputs.append(None)

                else:
                    slice_inputs.append(scalar_vars[j])
                    input_vars.append(scalar_vars[j])
                    values.append(scalar_ranges[j])

            symbol_slice = slice(*slice_inputs[:3])
            sub_x = x[symbol_slice][i]
            f = function([x] + input_vars, sub_x, mode=mode_opt)

            # Check stacktrace was copied over correctly after opt was applied
            assert check_stack_trace(f, ops_to_check=Subtensor)

            topo = f.maker.fgraph.toposort()
            assert len([t for t in topo if isinstance(t.op, Subtensor)]) <= 1
            assert isinstance(topo[-1].op, DeepCopyOp)

            for x_s in self.x_shapes:
                x_val = self.rng.uniform(size=x_s).astype(config.floatX)
                for i_val in zip(*values):
                    # The index could be out of bounds
                    # In that case, an Exception should be raised,
                    # otherwise, we let DebugMode check f
                    # For that, we need to create a numerical slice.
                    i_val_idx = 0
                    num_slice_inputs = []
                    for none_j in none_pos:
                        if none_j:
                            num_slice_inputs.append(None)
                        else:
                            num_slice_inputs.append(i_val[i_val_idx])
                            i_val_idx += 1
                    num_slice = slice(*num_slice_inputs[:3])
                    num_i = num_slice_inputs[3]

                    try:
                        x_val[num_slice][num_i]
                    except IndexError:
                        with pytest.raises(IndexError):
                            f(x_val, *i_val)
                    else:
                        # Executed if the "try" clause did not raise
                        # any exception
                        f(x_val, *i_val)


class TestLocalAdvSub1AdvIncSub1:
    def setup_method(self):
        utt.seed_rng()
        mode = get_default_mode()
        self.mode = mode.including("local_adv_sub1_adv_inc_sub1").excluding("fusion")
        self.mode_no_assert = self.mode.including("local_remove_all_assert")

    def test_basic(self):
        for dtype1, dtype2 in [
            ("float32", "float32"),
            ("float32", "float64"),
            ("float64", "float32"),
            ("float64", "float64"),
        ]:
            x = matrix(dtype=dtype1)
            y = matrix(dtype=dtype2)
            idx = ivector()

            dx = np.random.rand(4, 5).astype(dtype1)
            dy = np.random.rand(2, 5).astype(dtype2)
            # Duplicate the last row of dy
            dy = np.vstack([dy, dy[-1:]])
            # Use the same index twice, with the same corresponding value.
            # That makes set_subtensor well-defined, and tests
            # duplication for inc_subtensor.
            didx = np.asarray([1, 3, 3], "int32")

            # set_subtensor
            inc = set_subtensor(x[idx], y)
            o = inc[idx]
            f = function([x, y, idx], o, self.mode_no_assert)

            res = f(dx, dy, didx)
            utt.assert_allclose(dy, res)
            topo = f.maker.fgraph.toposort()
            assert len(topo) == 1
            assert isinstance(topo[0].op, (DeepCopyOp, Elemwise))

            # inc_subtensor(data[idx], y)
            inc = inc_subtensor(x[idx], y)
            o = inc[idx]
            f = function([x, y, idx], o, self.mode_no_assert)

            res = f(dx, dy, didx)
            _dx = dx.copy()
            np.add.at(_dx, didx, dy)
            utt.assert_allclose(_dx[didx], res)
            topo = f.maker.fgraph.toposort()
            len(topo) == 2

            # inc_subtensor(0[idx], y)
            inc = inc_subtensor(x.zeros_like()[idx], y)
            o = inc[idx]
            f = function([x, y, idx], o, self.mode_no_assert)

            res = f(dx, dy, didx)
            utt.assert_allclose(np.vstack([dy[0], 2 * dy[1], 2 * dy[2]]), res)

    def test_assert(self):
        x = matrix("x")
        y = matrix("y")
        idx = ivector()

        dx = np.random.rand(4, 5).astype(config.floatX)
        dy = np.random.rand(2, 5).astype(config.floatX)

        # set_subtensor
        inc = set_subtensor(x[idx], y)
        o = inc[idx]
        f = function([x, y, idx], o, self.mode)
        # test wrong index
        for i in [dx.shape[0], -dx.shape[0] - 1]:
            with pytest.raises((AssertionError, IndexError)):
                f(dx, dy, [i, i])
        # test wrong shape
        with pytest.raises((AssertionError, IndexError)):
            f(dx, dy, [1])

    def test_stack_trace(self):
        x = matrix("x")
        # test cases with y.dtype
        # - equal to x.dtype
        # - different from x.dtype (to trigger the cast in
        #   local_adv_sub1_adv_inc_sub1)
        ys = [matrix("y"), dmatrix("y")]
        idx = ivector()

        # set_subtensor and then subtensor with both ys
        incs = [set_subtensor(x[idx], y) for y in ys]
        outs = [inc[idx] for inc in incs]

        for y, out in zip(ys, outs):
            f = function([x, y, idx], out, self.mode)
            assert check_stack_trace(f, ops_to_check=(Assert, aes.Cast))


class TestAllocZero:
    def setup_method(self):
        mode = get_default_mode()
        self.mode = mode.including(
            "local_incsubtensor_of_zeros",
            "local_setsubtensor_of_constants",
            "local_0_dot_x",
        )

    def test_setsubtensor_allocs0(self):
        x = matrix()
        y = matrix()
        x0 = aet.zeros_like(x)
        y0 = aet.zeros_like(y)
        z = set_subtensor(x0[:4], y0)
        f = function([x, y], z, mode=self.mode)
        assert np.all(
            [not isinstance(n.op, IncSubtensor) for n in f.maker.fgraph.toposort()]
        )

    def test_setsubtensor_allocs1(self):
        y = matrix()
        x0 = aet.constant(np.asarray(np.zeros((4, 4)), dtype=config.floatX))
        y0 = aet.zeros_like(y)
        z = set_subtensor(x0[:4], y0)
        f = function([y], z, mode=self.mode)
        assert np.all(
            [not isinstance(n.op, IncSubtensor) for n in f.maker.fgraph.toposort()]
        )

    def test_setsubtensor_allocs1t(self):
        y = matrix()
        x0 = aet.constant(np.asarray(np.zeros((4, 4)), dtype=config.floatX))
        y0 = aet.zeros_like(y)
        z = set_subtensor(x0[:4], y0.T)
        f = function([y], z, mode=mode_opt)
        assert np.all(
            [not isinstance(n.op, IncSubtensor) for n in f.maker.fgraph.toposort()]
        )

    def test_setsubtensor_allocs2(self):
        x = matrix()
        y0 = aet.constant(np.asarray(np.zeros_like((4, 4)), dtype=config.floatX))
        x0 = aet.zeros_like(x)
        z = set_subtensor(x0[:4], y0)
        f = function([x], z, mode=self.mode)
        assert np.all(
            [not isinstance(n.op, IncSubtensor) for n in f.maker.fgraph.toposort()]
        )

    def test_incsubtensor_allocs0(self):
        x = matrix()
        y = matrix()
        y0 = aet.zeros_like(y)
        z = inc_subtensor(x[:4], y0)
        f = function([x, y], z, mode=self.mode)
        assert np.all(
            [not isinstance(n.op, IncSubtensor) for n in f.maker.fgraph.toposort()]
        )

    def test_incsubtensor_allocs0t(self):
        x = matrix()
        y = matrix()
        y0 = aet.zeros_like(y)
        z = inc_subtensor(x[:4], y0.T)
        f = function([x, y], z, mode=mode_opt)
        assert np.all(
            [not isinstance(n.op, IncSubtensor) for n in f.maker.fgraph.toposort()]
        )

    def test_incsubtensor_allocs1(self):
        x = matrix()
        y0 = aet.constant(np.asarray(np.zeros_like((4, 4)), dtype=config.floatX))
        z = inc_subtensor(x[:4], y0)
        f = function([x], z, mode=self.mode)
        assert np.all(
            [not isinstance(n.op, IncSubtensor) for n in f.maker.fgraph.toposort()]
        )

    def test_incsubtensor_x_zeros(self):
        x = aet.constant(np.asarray(np.zeros((4, 4)), dtype=config.floatX))
        y = matrix()
        z = inc_subtensor(x[:4], y)
        f = function([y], z)
        inc_nodes = [
            n for n in f.maker.fgraph.toposort() if isinstance(n.op, IncSubtensor)
        ]

        assert len(inc_nodes) == 1
        node_is_set_instead_of_inc = inc_nodes[0].op.set_instead_of_inc
        mode = config.mode
        assert (mode != "FAST_COMPILE" and node_is_set_instead_of_inc) or (
            mode == "FAST_COMPILE" and not node_is_set_instead_of_inc
        )
        test_X = np.random.random((4, 4)).astype(config.floatX)
        utt.assert_allclose(f(test_X), test_X)

        # also check the flag doesn't get set if first input is not zeros:
        not_all_zeros = np.zeros((4, 4))
        not_all_zeros[1, 0] = 0.001
        x = aet.constant(np.asarray(not_all_zeros, dtype=config.floatX))
        y = matrix()
        z = inc_subtensor(x[:4], y)
        f = function([y], z)
        inc_nodes = [
            n for n in f.maker.fgraph.toposort() if isinstance(n.op, IncSubtensor)
        ]
        assert len(inc_nodes) == 1
        assert inc_nodes[0].op.set_instead_of_inc is False
        test_X = np.random.random((4, 4)).astype(config.floatX)
        utt.assert_allclose(f(test_X), test_X + not_all_zeros)

    def test_advancedincsubtensor1_allocs0(self):
        x = matrix()
        y = matrix()
        y0 = aet.zeros_like(y)
        z = inc_subtensor(x[[0, 1, 2, 3]], y0)
        f = function([x, y], z, mode=self.mode)
        assert np.all(
            [
                not isinstance(n.op, AdvancedIncSubtensor1)
                for n in f.maker.fgraph.toposort()
            ]
        )

    def test_advancedincsubtensor1_allocs0t(self):
        x = matrix()
        y = matrix()
        y0 = aet.zeros_like(y)
        z = inc_subtensor(x[[0, 1, 2, 3]], y0.T)
        f = function([x, y], z, mode=mode_opt)
        assert np.all(
            [
                not isinstance(n.op, AdvancedIncSubtensor1)
                for n in f.maker.fgraph.toposort()
            ]
        )

    def test_advancedincsubtensor1_allocs1(self):
        x = matrix()
        y0 = aet.constant(np.asarray(np.zeros_like((4, 4)), dtype=config.floatX))
        z = inc_subtensor(x[[0, 1, 2, 3]], y0)
        f = function([x], z, mode=self.mode)
        assert np.all(
            [
                not isinstance(n.op, AdvancedIncSubtensor1)
                for n in f.maker.fgraph.toposort()
            ]
        )

    def test_advancedincsubtensor_allocs0(self):
        x = matrix()
        y = matrix()
        y0 = aet.zeros_like(y)
        z = inc_subtensor(x[[[0, 0], [1, 1]], [[0, 1], [0, 1]]], y0)
        f = function([x, y], z, mode=self.mode)
        assert np.all(
            [
                not isinstance(n.op, AdvancedIncSubtensor)
                for n in f.maker.fgraph.toposort()
            ]
        )

    def test_advancedincsubtensor_allocs0t(self):
        x = matrix()
        y = matrix()
        y0 = aet.zeros_like(y)
        z = inc_subtensor(x[[[0, 0], [1, 1]], [[0, 1], [0, 1]]], y0.T)
        f = function([x, y], z, mode=mode_opt)
        assert np.all(
            [
                not isinstance(n.op, AdvancedIncSubtensor)
                for n in f.maker.fgraph.toposort()
            ]
        )

    def test_advancedincsubtensor_allocs1(self):
        x = matrix()
        y0 = aet.constant(np.asarray(np.zeros_like((2, 2)), dtype=config.floatX))
        z = inc_subtensor(x[[[0, 0], [1, 1]], [[0, 1], [0, 1]]], y0)
        f = function([x], z, mode=self.mode)
        assert np.all(
            [
                not isinstance(n.op, AdvancedIncSubtensor)
                for n in f.maker.fgraph.toposort()
            ]
        )

    def test_dot_allocs_0(self):
        v1 = vector("v1")
        v2 = vector("v2")
        m1 = matrix("m1")
        m2 = matrix("m2")
        vv2 = np.asarray([0, 1], dtype=config.floatX)
        vm2 = np.asarray([[1, 2], [4, 5]], dtype=config.floatX)
        vv3 = np.asarray([0, 1, 2], dtype=config.floatX)
        vm3 = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=config.floatX)
        for _e1 in [(v1, vv2, vv3), (m1, vm2, vm3)]:
            for _e2 in [(v2, vv2, vv3), (m2, vm2, vm3)]:
                for p in [0, 1]:
                    if p == 0:
                        e1 = aet.zeros_like(_e1[0])
                        e2 = _e2[0]
                    else:
                        e1 = _e1[0]
                        e2 = aet.zeros_like(_e2[0])
                    o = dot(e1, e2)
                    f = function([_e1[0], _e2[0]], o, mode=self.mode)
                    f(_e1[1], _e2[1])
                    f(_e1[2], _e2[2])
                    assert np.all(
                        [not isinstance(n.op, Dot) for n in f.maker.fgraph.toposort()]
                    )

                    # test that we don't remove shape errors
                    with pytest.raises((ValueError, AssertionError)):
                        f(_e1[1], _e2[2])
                    with pytest.raises((ValueError, AssertionError)):
                        f(_e1[2], _e2[1])


def test_local_IncSubtensor_serialize():
    d = np.random.normal(0, 0.01, size=(100, 100))
    d = d.astype(config.floatX)

    W = shared(d, name="W")
    i = vector("i", dtype="int64")
    j = vector("j", dtype="int64")
    t = scalar("t")
    y = (W[i] + W[j] + W[1] + W[i, j]).sum()
    cost = sqr(t - y)
    dW = aesara.grad(cost, W)
    mode = get_default_mode().excluding("fusion")
    mode = mode.including("local_IncSubtensor_serialize")
    f = function([i, j, t], updates=[(W, W - 0.01 * dW)], mode=mode)
    topo = f.maker.fgraph.toposort()
    adds = [
        n
        for n in topo
        if isinstance(n.op, Elemwise) and isinstance(n.op.scalar_op, aes.Add)
    ]
    for a in adds:
        assert not any(
            [
                inp.owner
                and isinstance(
                    inp.owner.op,
                    (
                        IncSubtensor,
                        AdvancedIncSubtensor,
                        AdvancedIncSubtensor1,
                    ),
                )
                for inp in a.inputs
            ]
        )

    # Now test that the stack trace is copied over properly,
    # if we return the gradients. We need to use same mode as before.
    f = function([i, j, t], dW, mode=mode)
    assert check_stack_trace(
        f,
        ops_to_check=[
            IncSubtensor,
            AdvancedIncSubtensor,
            AdvancedIncSubtensor1,
        ],
    )


def test_local_set_to_inc_subtensor():
    v = fmatrix()
    s = v[[2, 1]]
    g = s + 3
    r = set_subtensor(s, g)
    moder = get_default_mode().excluding("local_set_to_inc_subtensor")
    modet = get_default_mode().including("local_set_to_inc_subtensor")
    f1 = function([v], r, mode=moder)
    f2 = function([v], r, mode=modet)

    advi1 = [
        n for n in f1.maker.fgraph.toposort() if isinstance(n.op, AdvancedIncSubtensor1)
    ]

    advi2 = [
        n for n in f2.maker.fgraph.toposort() if isinstance(n.op, AdvancedIncSubtensor1)
    ]

    # We only have SetSubtensor in f1
    assert all(n.op.set_instead_of_inc for n in advi1)
    # We don't have any SetSubtensor in f2
    assert all(not n.op.set_instead_of_inc for n in advi2)

    val = np.random.randn(3, 2).astype("float32")

    r1 = f1(val)
    r2 = f2(val)

    utt.assert_allclose(r1, r2)

    # Finally, test that the stack trace is copied over properly,
    # before and after optimization.
    assert check_stack_trace(f1, ops_to_check=AdvancedIncSubtensor1)
    assert check_stack_trace(f2, ops_to_check="all")


class TestLocalElemwiseAlloc:
    dtype = config.floatX

    def setup_method(self):
        self.fast_compile_mode = get_mode("FAST_COMPILE")
        self.fast_run_mode = get_mode("FAST_RUN")

        self.vec = vector("vec", dtype=self.dtype)
        self.mat = matrix("mat", dtype=self.dtype)
        self.tens = tensor3("tens", dtype=self.dtype)

        self.alloc_wo_dep = aet.alloc(self.vec, 2, 2)
        self.alloc_wo_dep_broad = aet.alloc(self.vec, 1, 2)
        self.alloc_w_dep = aet.alloc(self.vec, *self.mat.shape)
        self.alloc_w_dep_broad = aet.alloc(self.vec, 1, *self.mat.shape)
        self.alloc_w_dep_broad2 = aet.alloc(
            self.vec, self.mat.shape[0], self.mat.shape[1], 1
        )
        self.alloc_w_dep_tens = aet.alloc(
            self.vec, self.tens.shape[0], self.tens.shape[1]
        )
        self.tv_wo_dep = aet.alloc(self.vec, 5, 5)
        self.tm_wo_dep = aet.alloc(self.mat, 5, 5, 5)
        self.s = iscalar("s")
        self.tv_w_dep = aet.alloc(self.vec, self.s, self.s)
        self.tm_w_dep = aet.alloc(self.mat, 5, 5, 5)
        self.row = row(dtype=self.dtype)
        self.o = aet.alloc(self.row, 5, 5)

    def _verify_alloc_count(self, f, count):
        assert (
            sum(
                [
                    isinstance(elem.op, Alloc)
                    for elem in f.maker.fgraph.toposort()
                    if elem.op is not None
                ]
            )
            == count
        )

    def _verify_assert_count(self, f, count):
        assert (
            sum(
                [
                    isinstance(elem.op, Assert)
                    for elem in f.maker.fgraph.toposort()
                    if elem.op is not None
                ]
            )
            == count
        )

    def test_remove_alloc_wo_dimshuffle(self):
        # Exclude local_useless_alloc, since it does not introduce
        # assert in all the same cases.
        self.fast_run_mode = self.fast_run_mode.excluding(
            "local_useless_alloc", "local_canonicalize_alloc"
        )
        # No optimization on alloc
        func = function(
            [self.vec, self.mat],
            self.alloc_wo_dep + self.mat,
            mode=self.fast_compile_mode,
        )
        self._verify_alloc_count(func, 1)
        self._verify_assert_count(func, 0)
        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(func, ops_to_check="all")

        # Optimization on alloc with assert
        func = function(
            [self.vec, self.mat], self.alloc_wo_dep + self.mat, mode=self.fast_run_mode
        )
        self._verify_alloc_count(func, 0)
        self._verify_assert_count(func, 1)

        # Optimization on alloc with assert and broadcast
        func = function(
            [self.vec, self.mat],
            self.alloc_wo_dep_broad + self.mat,
            mode=self.fast_run_mode,
        )
        self._verify_alloc_count(func, 0)
        self._verify_assert_count(func, 1)

        # No optimization on alloc without assert
        func = function(
            [self.vec, self.mat],
            self.alloc_w_dep + self.mat,
            mode=self.fast_compile_mode,
        )
        self._verify_alloc_count(func, 1)
        self._verify_assert_count(func, 0)

        # Optimization on alloc without assert
        func = function(
            [self.vec, self.mat], self.alloc_w_dep + self.mat, mode=self.fast_run_mode
        )
        self._verify_alloc_count(func, 0)
        self._verify_assert_count(func, 0)

        # Optimization on alloc without assert and with broadcast
        func = function(
            [self.vec, self.mat],
            self.alloc_w_dep_broad + self.mat,
            mode=self.fast_run_mode,
        )
        self._verify_alloc_count(func, 0)
        self._verify_assert_count(func, 0)

        # Not optimized case on alloc and with broadcast
        func = function(
            [self.vec, self.mat],
            self.alloc_w_dep_broad2 + self.mat,
            mode=self.fast_run_mode,
        )
        self._verify_alloc_count(func, 1)
        self._verify_assert_count(func, 0)

    def test_remove_alloc_w_dimshuffle(self):
        # No optimization on dimshuffle with assert
        func = function(
            [self.vec, self.tens],
            self.alloc_wo_dep.dimshuffle(0, 1, "x") + self.tens,
            mode=self.fast_compile_mode,
        )
        self._verify_alloc_count(func, 1)
        self._verify_assert_count(func, 0)

        # Optimization on dimshuffle with assert
        func = function(
            [self.vec, self.tens],
            self.alloc_wo_dep.dimshuffle(0, 1, "x") + self.tens,
            mode=self.fast_run_mode,
        )
        self._verify_alloc_count(func, 0)
        self._verify_assert_count(func, 1)

        # No optimization on dimshuffle without assert
        func = function(
            [self.vec, self.tens],
            self.alloc_w_dep_tens.dimshuffle(0, 1, "x") + self.tens,
            mode=self.fast_compile_mode,
        )
        self._verify_alloc_count(func, 1)
        self._verify_assert_count(func, 0)

        # Optimization on dimshuffle without assert
        func = function(
            [self.vec, self.tens],
            self.alloc_w_dep_tens.dimshuffle(0, 1, "x") + self.tens,
            mode=self.fast_run_mode,
        )
        self._verify_alloc_count(func, 0)
        self._verify_assert_count(func, 0)

    def test_multi_input_single_alloc(self):
        # No optimization on dimshuffle with assert
        func = function(
            [self.vec, self.mat],
            self.tv_wo_dep + self.tm_wo_dep,
            mode=self.fast_compile_mode,
        )
        self._verify_alloc_count(func, 2)
        self._verify_assert_count(func, 0)

        # Optimization on dimshuffle with assert
        func = function(
            [self.vec, self.mat],
            self.tv_wo_dep + self.tm_wo_dep,
            mode=self.fast_run_mode,
        )
        self._verify_alloc_count(func, 1)
        self._verify_assert_count(func, 0)

        # No optimization on dimshuffle without assert
        func = function(
            [self.vec, self.mat, self.s],
            self.tv_w_dep + self.tm_w_dep,
            mode=self.fast_compile_mode,
        )
        self._verify_alloc_count(func, 2)
        self._verify_assert_count(func, 0)

        # Optimization on dimshuffle without assert
        func = function(
            [self.vec, self.mat, self.s],
            self.tv_w_dep + self.tm_w_dep,
            mode=self.fast_run_mode,
        )
        self._verify_alloc_count(func, 1)
        self._verify_assert_count(func, 1)

    def test_error(self):
        t3fft = tensor(dtype=self.dtype, broadcastable=(False, False, True))
        o = self.o.dimshuffle(0, 1, "x") + t3fft
        func = function([t3fft, self.row], o, mode=self.fast_run_mode)
        self._verify_alloc_count(func, 0)
        self._verify_assert_count(func, 1)
        d = np.random.rand(5, 5, 1).astype(self.dtype)
        r = np.random.rand(1, 5).astype(self.dtype)
        func(d, r)


def test_local_subtensor_of_alloc():

    # DebugMode should detect if something goes wrong.
    # test shape combination of odd and event shape.
    for shape in [(3, 5), (4, 6), (3, 8), (4, 7), (1, 5), (5, 1)]:
        x = tensor(dtype=config.floatX, broadcastable=(shape[0] == 1, shape[1] == 1))

        xval = np.zeros(shape, dtype=config.floatX)
        yval = np.arange(shape[1], dtype=config.floatX)

        for y in [shared(yval), aet.constant([1.0])]:

            # The rows of yx are copies of y
            yx = aet.alloc(y, x.shape[0], x.shape[1])

            # Slice of each row
            z_mat = yx[:, 3:]
            assert z_mat.ndim == 2

            # Only one column
            z_vec = yx[:, 3]
            assert z_vec.ndim == 1
            # results are vector
            slicess = []
            if shape[0] != 1:
                slicess.append((2, slice(None)))
            if shape[1] != 1:
                slicess.append((slice(None), 3))

            # results are matrix
            slicess += [
                (slice(None), slice(3, None)),
                (slice(3, None),),
                (slice(3, None), slice(3, None)),
                (slice(1, 3), slice(None, -1)),
                (slice(None, None, 2)),
                (slice(1, None, 2)),
            ]
            for slices in slicess:
                z = yx.__getitem__(slices)
                f = function([x], z)
                if config.mode != "FAST_COMPILE":
                    # Subtensor can be in the input of Alloc
                    assert not isinstance(f.maker.fgraph.toposort()[-1].op, Subtensor)
                val = f(xval)
                assert xval.__getitem__(slices).shape == val.shape


def test_local_fill_useless():
    # Test opt local_fill_useless
    x = dvector()
    y = dvector()
    z = lvector()
    m = dmatrix()

    x_ = np.random.rand(
        5,
    )
    y_ = np.random.rand(
        5,
    )
    z_ = (
        np.random.rand(
            5,
        )
        * 5
    ).astype("int64")
    m_ = np.random.rand(5, 5)

    # basic case
    f = function([x], aet.fill(x, x) * 2, mode=mode_opt)
    assert [node.op for node in f.maker.fgraph.toposort()] == [mul]
    f(x_)

    # basic case
    f = function([x, y], aet.second(y, x) * 2, mode=mode_opt)
    assert [node.op for node in f.maker.fgraph.toposort()] == [mul]
    f(x_, y_)

    # basic case
    f = function([x, y], aet.fill(x, y) * 2, mode=mode_opt)
    assert [node.op for node in f.maker.fgraph.toposort()] == [mul]
    f(x_, y_)

    # now with different type(cast)
    f = function([x, z], aet.fill(z, x) * 2, mode=mode_opt)
    assert [node.op for node in f.maker.fgraph.toposort()] == [mul]
    f(x_, z_)

    # now with different type(cast)
    f = function([x, z], aet.fill(x, z) * 2, mode=mode_opt)
    assert [node.op for node in f.maker.fgraph.toposort()] == [mul]
    f(x_, z_)

    # now cutting out the input ??
    f = function([x, y], aet.fill(x, y) * 2, mode=mode_opt)
    assert [node.op for node in f.maker.fgraph.toposort()] == [mul]
    f(x_, y_)

    # Test with different number of dimensions
    # The fill is not useless, so it should stay
    f = function([m, x], aet.fill(m, x) * 2, mode=mode_opt)
    ops = [node.op.__class__ for node in f.maker.fgraph.toposort()]
    assert Alloc in ops
    f(m_, x_)


class TestLocalCanonicalizeAlloc:
    def setup_method(self):
        self.rng = np.random.RandomState(utt.fetch_seed())

    @config.change_flags(compute_test_value="off")
    def test_basic(self):
        x = shared(self.rng.randn(3, 7))
        a = aet.alloc(x, 6, 7)

        # It is a bad idea to have aet.alloc return x directly,
        # because the shape mismatch cannot be caught.
        assert a.owner and isinstance(a.owner.op, Alloc)

        f = function([], a, mode=mode_opt)
        # The optimization should then be applied, and remove Alloc
        assert [node.op for node in f.maker.fgraph.toposort()] == [deep_copy_op]

        # In DebugMode, the shape mismatch should be detected
        if isinstance(mode_opt, DebugMode):
            with pytest.raises(ValueError):
                f

        # No need to check_stack_trace as the optimization
        # local_canonicalize_alloc only removes nodes.

    def test_basic_1(self):
        # Test that alloc never gets instantiated during optimization
        mode = mode_opt.excluding("local_canonicalize_alloc")

        x = matrix("x")
        xx = aet.fill(x, x)

        # The optimization 'locall_fill_to_alloc' should call aet.alloc,
        # which should return x and not alloc(x, ...)
        f = function([x], [xx], mode=mode)
        op_classes = [node.op.__class__ for node in f.maker.fgraph.toposort()]
        assert Alloc not in op_classes

        # No need to check_stack_trace as the optimization
        # local_canonicalize_alloc only removes nodes.

    def test_basic_2(self):
        # Test that alloc never gets instantiated during optimization
        mode = mode_opt.excluding("local_canonicalize_alloc")

        x = matrix("x")
        y = aet.tile(x, (1,) * 2)

        f = function([x], [y], mode=mode)
        op_classes = [node.op.__class__ for node in f.maker.fgraph.toposort()]

        # We are supposed to test if tensr.Alloc is not in op_classes,
        # but since the proper proper optimization is not currently
        # implemented it will fail. Once the correct optimization is in place,
        # we have to change the following we should not see Alloc
        # in op_classes and we have to change the assert.
        assert Alloc in op_classes
        # The correct opt removes nodes, no need for check_stack_trace

    def test_useless_alloc_with_shape_one(self):
        alloc_lift = out2in(local_canonicalize_alloc)
        x = shared(
            self.rng.randn(
                2,
            )
        )
        y = shared(self.rng.randn())
        z = shared(self.rng.randn(1, 1))
        w = shared(self.rng.randn(1, 1))
        alloc_x = aet.alloc(x, 1, 3, 2)
        alloc_y = aet.alloc(y, 1, 1)
        alloc_z = aet.alloc(z, 1, 1, 2)
        alloc_w = aet.alloc(w, 1, 2)

        g = FunctionGraph([x, y, z, w], [alloc_x, alloc_y, alloc_z, alloc_w])
        assert str(g) == (
            "FunctionGraph(Alloc(<TensorType(float64, vector)>, "
            "TensorConstant{1}, "
            "TensorConstant{3}, "
            "TensorConstant{2}), "
            "Alloc(<TensorType(float64, scalar)>, "
            "TensorConstant{1}, "
            "TensorConstant{1}), "
            "Alloc(<TensorType(float64, matrix)>, "
            "TensorConstant{1}, "
            "TensorConstant{1}, "
            "TensorConstant{2}), "
            "Alloc(<TensorType(float64, matrix)>, "
            "TensorConstant{1}, "
            "TensorConstant{2}))"
        )

        alloc_lift.optimize(g)
        assert str(g) == (
            "FunctionGraph(InplaceDimShuffle{x,0,1}"
            "(Alloc(<TensorType(float64, vector)>, "
            "TensorConstant{3}, "
            "TensorConstant{2})), "
            "InplaceDimShuffle{x,x}"
            "(<TensorType(float64, scalar)>), "
            "InplaceDimShuffle{x,0,1}"
            "(Alloc(<TensorType(float64, matrix)>, "
            "TensorConstant{1}, "
            "TensorConstant{2})), "
            "Alloc(<TensorType(float64, matrix)>, "
            "TensorConstant{1}, "
            "TensorConstant{2}))"
        )

        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(g, ops_to_check="all")


class TestLocalUselessIncSubtensorAlloc:
    opt_name = "local_useless_inc_subtensor_alloc"

    def setup_method(self):
        # The optimization requires the shape feature so we need to compile in
        # FAST_RUN mode.
        mode = config.mode
        if mode == "FAST_COMPILE":
            mode = "FAST_RUN"
        self.mode = get_mode(mode)

    def test_advanced_inc_subtensor(self):
        x = vector("x")
        y = scalar("y")
        i = matrix("i", dtype="int64")
        z = advanced_inc_subtensor(x, aet.alloc(y, *i.shape), i)
        mode1 = self.mode.excluding(self.opt_name)
        mode2 = self.mode.including(self.opt_name)
        f1 = function([x, i, y], z, mode=mode1)
        f2 = function([x, i, y], z, mode=mode2)

        # the alloc op should still be there
        assert (
            len([n for n in f1.maker.fgraph.toposort() if isinstance(n.op, Alloc)]) == 1
        )
        # the alloc op should have been removed
        assert (
            len([n for n in f2.maker.fgraph.toposort() if isinstance(n.op, Alloc)]) == 0
        )

        x_value = np.random.randn(5).astype(config.floatX)
        y_value = np.random.randn()
        i_value = np.random.randint(0, 3, size=(2, 3))

        r1 = f1(x_value, i_value, y_value)
        r2 = f2(x_value, i_value, y_value)

        utt.assert_allclose(r1, r2)

        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f1, ops_to_check=AdvancedIncSubtensor)
        assert check_stack_trace(f2, ops_to_check=AdvancedIncSubtensor)

    def test_advanced_inc_subtensor1(self):
        x = vector("x")
        y = scalar("y")
        i = vector("i", dtype="int64")
        z = advanced_inc_subtensor1(x, aet.alloc(y, *i.shape), i)
        mode1 = self.mode.excluding(self.opt_name)
        mode2 = self.mode.including(self.opt_name)
        f1 = function([x, i, y], z, mode=mode1)
        f2 = function([x, i, y], z, mode=mode2)

        # the alloc op should still be there
        assert (
            len([n for n in f1.maker.fgraph.toposort() if isinstance(n.op, Alloc)]) == 1
        )
        # the alloc op should have been removed
        assert (
            len([n for n in f2.maker.fgraph.toposort() if isinstance(n.op, Alloc)]) == 0
        )

        x_value = np.random.randn(5).astype(config.floatX)
        y_value = np.random.randn()
        i_value = np.random.randint(0, 3, size=2)

        r1 = f1(x_value, i_value, y_value)
        r2 = f2(x_value, i_value, y_value)

        utt.assert_allclose(r1, r2)

        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f1, ops_to_check=AdvancedIncSubtensor1)
        assert check_stack_trace(f2, ops_to_check="all")

    def test_incsubtensor(self):
        x = vector("x")
        y = scalar("y")
        i = scalar("i", dtype="int64")
        z = inc_subtensor(x[:i], aet.alloc(y, i))
        mode1 = self.mode.excluding(self.opt_name)
        mode2 = self.mode.including(self.opt_name)
        f1 = function([x, i, y], z, mode=mode1)
        f2 = function([x, i, y], z, mode=mode2)

        # the alloc op should still be there
        assert (
            len([n for n in f1.maker.fgraph.toposort() if isinstance(n.op, Alloc)]) == 1
        )
        # the alloc op should have been removed
        assert (
            len([n for n in f2.maker.fgraph.toposort() if isinstance(n.op, Alloc)]) == 0
        )

        x_value = np.random.randn(5).astype(config.floatX)
        y_value = np.random.randn()
        i_value = 3

        r1 = f1(x_value, i_value, y_value)
        r2 = f2(x_value, i_value, y_value)

        utt.assert_allclose(r1, r2)

        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(f1, ops_to_check="last")
        assert check_stack_trace(f2, ops_to_check="last")


class TestShapeOptimizer:
    def setup_method(self):
        utt.seed_rng()

    def test_basic(self):
        mode = config.mode
        if mode == "FAST_COMPILE":
            mode = "FAST_RUN"
        v = vector()
        m = matrix()
        f = function([v, m], (v + m).shape, mode=mode)
        for node in f.maker.fgraph.toposort():
            assert node.op != add

    def test_constant(self):
        mode = config.mode
        if mode == "FAST_COMPILE":
            mode = "FAST_RUN"

        v = vector()
        f = function([v], v.dimshuffle("x", "x", 0).shape[1], mode=mode)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert topo[0].op == deep_copy_op

    @staticmethod
    def max_pool_c01b(c01b, pool_shp, pool_stride, img_shp):
        """
        Like max_pool but with input using axes ('c', 0, 1, 'b')
          (Alex Krizhevsky format)

        pool_shp, pool_stride and img_shp are int that represent
        the same shp in x and y.
        """
        mx = None

        # Compute index in pooled space of last needed pool
        # (needed = each input pixel must appear in at least one pool)
        def last_pool(im_shp, p_shp, p_strd):
            rval = int(np.ceil(float(im_shp - p_shp) / p_strd))
            assert p_strd * rval + p_shp >= im_shp
            assert p_strd * (rval - 1) + p_shp < im_shp
            return rval

        # Compute starting row of the last pool
        last_pool_r = last_pool(img_shp, pool_shp, pool_stride) * pool_stride
        # Compute number of rows needed in img for all indexes to work out
        required_r = last_pool_r + pool_shp

        last_pool_c = last_pool(img_shp, pool_shp, pool_stride) * pool_stride
        required_c = last_pool_c + pool_shp

        wide_infinity = aet.alloc(
            -np.inf, c01b.shape[0], required_r, required_c, c01b.shape[3]
        )

        c01b = set_subtensor(wide_infinity[:, 0:img_shp, 0:img_shp, :], c01b)

        for row_within_pool in range(pool_shp):
            row_stop = last_pool_r + row_within_pool + 1
            for col_within_pool in range(pool_shp):
                col_stop = last_pool_c + col_within_pool + 1
                cur = c01b[
                    :,
                    row_within_pool:row_stop:pool_stride,
                    col_within_pool:col_stop:pool_stride,
                    :,
                ]
                if mx is None:
                    mx = cur
                else:
                    mx = maximum(mx, cur)
        return mx

    def test_broadcasted_dims(self):
        # This test a case that caused a crash during optimization
        shp = (1, 1, 1, 1)
        rng = np.random.RandomState(utt.fetch_seed())
        a = shared(rng.rand(*shp).astype(config.floatX))
        out = self.max_pool_c01b(a, 1, 1, 1)

        # max_pool_c01b use -inf and this will trigger DebugMode error.
        mode = copy.copy(get_default_mode())
        mode.check_isfinite = False
        f = function([], out, mode=mode)
        f()

    def test_constant_merge(self):
        # This test the error in gh-1122 that is a caused by the
        # combination of merge optimizer and ShapeFeature.

        x = aet.constant([0, 0])
        y = x[1:]
        x1 = x - aet.join(0, y, y)
        x1.eval()

    def test_local_track_shape_i(self):
        class IdentityNoShape(Op):
            """Op that does not infer the output shape from the input one"""

            def make_node(self, x):
                x = as_tensor_variable(x)
                return Apply(self, [x], [x.type()])

            def perform(self, node, inp, out_):
                (x,) = inp
                (out,) = out_
                out[0] = x.copy()

            # def infer_shape(self, fgraph, node, (xshp,)):
            # return [tuple([self.shape_i(i)(r) for i in range(r.ndim)])]

        identity_noshape = IdentityNoShape()

        class IdentityShape(Op):
            """Op that does infer the output shape from the input one"""

            def make_node(self, x):
                x = as_tensor_variable(x)
                return Apply(self, [x], [x.type()])

            def perform(self, node, inp, out_):
                (x,) = inp
                (out,) = out_
                out[0] = x.copy()

            def infer_shape(self, fgraph, node, xshp_):
                # Could also just return.
                (xshp,) = xshp_
                return (xshp,)

        identity_shape = IdentityShape()

        @local_optimizer([IdentityNoShape])
        def local_identity_noshape_to_identity_shape(fgraph, node):
            """Optimization transforming the first Op into the second"""
            if isinstance(node.op, IdentityNoShape):
                return [identity_shape(node.inputs[0])]

        mode = get_default_mode().including("ShapeOpt", "specialize")
        rng = np.random.RandomState(utt.fetch_seed())
        x = tensor3("x")
        ins_x = identity_noshape(x)

        # Without the optimization
        f = function([x], ins_x.shape, mode=mode)
        xval = rng.randn(3, 4, 7).astype(config.floatX)
        assert np.all(f(xval) == [3, 4, 7])
        f_ops = [node.op for node in f.maker.fgraph.toposort()]
        assert len(f_ops) == 5
        assert identity_noshape in f_ops
        assert identity_shape not in f_ops

        # Register the optimization
        register_specialize(local_identity_noshape_to_identity_shape)

        mode = get_default_mode().including("ShapeOpt", "specialize")
        # With the optimization
        # The identity_shape op should not be needed anymore to compute
        # the shape
        g = function([x], ins_x.shape, mode=mode)
        xval = rng.randn(6, 1, 2).astype(config.floatX)
        assert np.all(g(xval) == [6, 1, 2])
        g_ops = [node.op for node in g.maker.fgraph.toposort()]
        assert len(g_ops) == 4
        assert identity_noshape not in g_ops
        assert identity_shape not in g_ops

        # test multiple level of op without infer_shape
        ins_x3 = identity_noshape(identity_noshape(identity_noshape(x)))
        h = function([x], ins_x3.shape, mode=mode)
        xval = rng.randn(6, 1, 2).astype(config.floatX)
        assert np.all(h(xval) == [6, 1, 2])
        h_ops = [node.op for node in h.maker.fgraph.toposort()]
        assert len(h_ops) == 4
        assert identity_noshape not in h_ops
        assert identity_shape not in h_ops

    def test_no_shapeopt(self):
        # Test that a basic example works even when ShapeOpt is excluded
        X = matrix()
        expr = X.shape[0]

        mode = get_default_mode().excluding("ShapeOpt")
        f = function([X], expr, mode=mode)
        # FIXME: This is not a good test.
        f([[1, 2], [2, 3]])


class TestAssert(utt.InferShapeTester):
    def setup_method(self):
        super().setup_method()

    def test_basic(self):
        x = scalar()
        y = scalar()
        f = function([x, y], assert_op(x, eq(x, y)))
        f(1, 1)
        with pytest.raises(AssertionError):
            f(1, 0)

    def test_local_remove_useless_assert1(self):
        # remove assert that are always true
        mode = config.mode
        if mode == "FAST_COMPILE":
            mode = "FAST_RUN"
        mode = get_mode(mode)

        x = scalar()
        f = function([x], assert_op(x, 1), mode=mode)
        assert f(1) == 1
        assert f(5) == 5
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert topo[0].op == deep_copy_op

    def test_test_local_remove_useless_assert2(self):
        # remove assert condition that are always true
        mode = config.mode
        if mode == "FAST_COMPILE":
            mode = "FAST_RUN"
        mode = get_mode(mode)

        x = scalar()
        y = scalar()
        f = function([x, y], assert_op(x, y, 1), mode=mode)
        assert f(1, 1) == 1
        assert f(5, 1) == 5
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 2
        assert len(topo[0].inputs) == 2
        assert topo[1].op == deep_copy_op

    def test_local_remove_useless_assert3(self):
        # don't remove assert condition that are always false
        mode = config.mode
        if mode == "FAST_COMPILE":
            mode = "FAST_RUN"
        mode = get_mode(mode)

        x = scalar()
        y = scalar()
        f = function([x, y], assert_op(x, y, 0), mode=mode)
        with pytest.raises(AssertionError):
            f(1, 0)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 2
        assert len(topo[0].inputs) == 3
        assert topo[1].op == deep_copy_op

    def test_local_remove_all_assert1(self):
        # remove assert condition that are unknown
        mode = config.mode
        if mode == "FAST_COMPILE":
            mode = "FAST_RUN"
        mode = get_mode(mode).including("local_remove_all_assert")

        x = scalar()
        y = scalar()
        f = function([x, y], assert_op(x, y), mode=mode)
        if isinstance(mode, DebugMode):
            # DebugMode will run the original version with the Assert
            with pytest.raises(AssertionError):
                f(1, 0)
        else:
            f(1, 0)  # Without opt, it should fail.
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1, topo
        assert topo[0].op == deep_copy_op, topo

        mode = get_default_mode()
        a = assert_op(x, eq(x, 0).any())
        f = function([x], a, mode=mode.excluding("unsafe"))
        topo = f.maker.fgraph.toposort()
        a_op = [n for n in topo if isinstance(n.op, Assert)]
        assert len(a_op) == 1

    def test_infer_shape(self):

        adscal = dscalar()
        bdscal = dscalar()
        adscal_val = np.random.rand()
        bdscal_val = np.random.rand() + 1
        out = assert_op(adscal, bdscal)
        self._compile_and_check(
            [adscal, bdscal], [out], [adscal_val, bdscal_val], Assert
        )

        admat = dmatrix()
        admat_val = np.random.rand(3, 4)
        adscal_val += 1
        out = assert_op(admat, adscal, bdscal)
        self._compile_and_check(
            [admat, adscal, bdscal], [out], [admat_val, adscal_val, bdscal_val], Assert
        )


class TestTile:
    def test_local_useless_tile(self):
        v = vector()
        m = matrix()
        mode = None
        if config.mode == "FAST_COMPILE":
            mode = "FAST_RUN"
        for var, data in [(v, [1, 2, 3]), (m, [[1, 2], [3, 4]])]:
            # When len(repeat pattern) <= var.ndim, everything is removed
            # for ndim in range(1, var.ndim):
            for ndim in range(var.ndim + 1):
                f = function([var], tile(var, (1,) * ndim), mode=mode)
                topo = f.maker.fgraph.toposort()
                assert len(topo) == 1
                assert isinstance(topo[0].op, DeepCopyOp)
                f(data)
                # In this case the opt only removes nodes,
                # no need to check_stack_trace
            # When len(repeat pattern) > var.ndim, only a dimshuffle should be
            # left, but there can be a DeepCopy as well
            for ndim in range(var.ndim + 1, var.ndim + 3):
                f = function([var], tile(var, (1,) * ndim), mode=mode)
                topo = f.maker.fgraph.toposort()
                assert len(topo) <= 2
                assert isinstance(topo[0].op, DimShuffle)
                assert check_stack_trace(f, ops_to_check=[DimShuffle])
                f(data)


class TestRebroadcast:
    def test_local_useless_rebroadcast(self):
        mode = get_default_mode().including("canonicalize")
        v1 = vector()
        v2 = vector()
        j = aet.join(0, v1, v2)
        f = function([v1, v2], j, mode=mode)
        f([1, 2], [3, 4, 5])
        e = f.maker.fgraph.toposort()
        assert len([n for n in e if isinstance(n.op, Rebroadcast)]) == 0

        assert check_stack_trace(f, ops_to_check="all")

    def test_rebroadcast_rebroadcast(self):
        mode = get_default_mode().including("canonicalize")
        m = matrix()
        s = aet.addbroadcast(m, 0, 1)
        v = aet.unbroadcast(s, 1)
        f = function([m], v, mode=mode)
        f([[76]])
        e = f.maker.fgraph.toposort()
        rebroadcast_nodes = [n for n in e if isinstance(n.op, Rebroadcast)]
        assert len(rebroadcast_nodes) == 1
        assert rebroadcast_nodes[0].op.axis == {0: True}


class TestUselessElemwise:
    def setup_method(self):
        self.mode = get_default_mode().including("canonicalize", "local_fill_to_alloc")

    def test_eq(self):
        x = dmatrix()
        y = dmatrix()
        f = function([x, y], eq(x, y), mode=self.mode)
        vx = np.random.rand(5, 4)
        vy = np.random.rand(5, 4)
        f(vx, vy)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert isinstance(topo[0].op, Elemwise)
        assert isinstance(topo[0].op.scalar_op, aes.EQ)
        f2 = function([x], eq(x, x), mode=self.mode)
        assert np.all(f2(vx) == np.ones((5, 4)))
        topo2 = f2.maker.fgraph.toposort()
        # Shape_i{1}(<TensorType(float64, matrix)>),
        # Shape_i{0}(<TensorType(float64, matrix)>), Alloc([[1]], Shape_i{0}.0,
        # Shape_i{1}.0
        assert len(topo2) == 3
        assert isinstance(topo2[-1].op, Alloc)

    def test_neq(self):
        x = dmatrix()
        y = dmatrix()
        f = function([x, y], neq(x, y), mode=self.mode)
        vx = np.random.rand(5, 4)
        vy = np.random.rand(5, 4)
        f(vx, vy)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert isinstance(topo[0].op, Elemwise)
        assert isinstance(topo[0].op.scalar_op, aes.NEQ)
        f2 = function([x], neq(x, x), mode=self.mode)
        assert np.all(f2(vx) == np.zeros((5, 4)))
        topo2 = f2.maker.fgraph.toposort()
        assert len(topo2) == 3
        assert isinstance(topo2[-1].op, Alloc)

    def test_mul(self):
        x = dmatrix()
        y = dmatrix()
        f = function([x], mul(x), mode=self.mode)
        vx = np.random.rand(5, 4)
        vy = np.random.rand(5, 4)
        f(vx)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert topo[0].op == deep_copy_op
        f2 = function([x, y], mul(x, y), mode=self.mode)
        assert np.all(f2(vx, vy) == vx * vy)
        topo2 = f2.maker.fgraph.toposort()
        assert len(topo2) == 1
        assert isinstance(topo2[0].op, Elemwise)
        assert isinstance(topo2[0].op.scalar_op, aes.Mul)

    def test_add(self):
        x = dmatrix()
        y = dmatrix()
        f = function([x], add(x), mode=self.mode)
        vx = np.random.rand(5, 4)
        vy = np.random.rand(5, 4)
        f(vx)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert topo[0].op == deep_copy_op
        f2 = function([x, y], add(x, y), mode=self.mode)
        assert np.all(f2(vx, vy) == vx + vy)
        topo2 = f2.maker.fgraph.toposort()
        assert len(topo2) == 1
        assert isinstance(topo2[0].op, Elemwise)
        assert isinstance(topo2[0].op.scalar_op, aes.Add)

    def test_identity(self):
        # aes.identity is used in 2 Elemwise functions:
        # tensor_copy, and view
        x = matrix()
        f = function([x], aet.tensor_copy(x), mode=self.mode)
        vx = np.random.rand(5, 4).astype(config.floatX)
        f(vx)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert topo[0].op == deep_copy_op


class TestCastCast:
    def setup_method(self):
        mode = get_default_mode()
        self.mode = mode.including("local_cast_cast")

    def test_consecutive(self):
        x = fmatrix()
        o = Elemwise(aes.Cast(aes.Scalar("float64")))(x.astype("float64"))
        f = function([x], o, mode=self.mode)
        dx = np.random.rand(5, 4).astype("float32")
        f(dx)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert isinstance(topo[0].op.scalar_op, aes.basic.Cast)

        x = dmatrix()
        o = Elemwise(aes.Cast(aes.Scalar("float32")))(x.astype("float32"))
        f = function([x], o, mode=self.mode)
        dx = np.random.rand(5, 4)
        f(dx)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert isinstance(topo[0].op.scalar_op, aes.basic.Cast)

    def test_upcast(self):
        # Upcast followed by any other cast
        x = fmatrix()
        o = Elemwise(aes.Cast(aes.Scalar("complex128")))(x.astype("complex64"))
        f = function([x], o, mode=self.mode)
        dx = np.random.rand(5, 4).astype("float32")
        f(dx)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert isinstance(topo[0].op.scalar_op, aes.basic.Cast)

        # Upcast followed by a downcast back to the base type
        x = fmatrix()
        o = Elemwise(aes.Cast(aes.Scalar("float32")))(x.astype("float64"))
        f = function([x], o, mode=self.mode)
        dx = np.random.rand(5, 4).astype("float32")
        f(dx)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert isinstance(topo[0].op, DeepCopyOp)

        # Downcast followed by an upcast back to the base type
        # Optimization shouldn't be applied
        x = dmatrix()
        o = Elemwise(aes.Cast(aes.Scalar("float64")))(x.astype("float32"))
        f = function([x], o, mode=self.mode)
        dx = np.random.rand(5, 4)
        f(dx)
        topo = f.maker.fgraph.toposort()
        assert (
            len(topo) == 1 and isinstance(topo[0].op.scalar_op, aes.basic.Composite)
        ) or (len(topo) > 1)


def test_constant_folding():
    # Test that constant folding get registered at fast_compile
    # An error removed that registration during the registration.
    x = dvector()
    mode = get_mode("FAST_COMPILE").excluding("fusion")
    f = function([x], [x * 2, x + x], mode=mode)
    topo = f.maker.fgraph.toposort()
    assert len(topo) == 2

    # Test that we do not crash when constant folding elemwise scalar
    # as they should not generate c code.

    x = aet.constant(3)
    assert x.ndim == 0
    mode = get_mode("FAST_COMPILE").excluding("fusion")
    f = function([], [x * 2, x + x], mode=mode)
    topo = f.maker.fgraph.toposort()
    assert len(topo) == 2
    assert all([isinstance(n.op, DeepCopyOp) for n in topo])


@pytest.mark.xfail(
    reason="Aesara optimizes constant before stabilization. "
    "This breaks stabilization optimizations in some "
    "cases. See #504.",
    raises=AssertionError,
)
def test_constant_get_stabilized():
    # Currently Aesara enables the `constant_folding` optimization before stabilization optimization.
    # This caused some stabilization optimizations to not be activated and that
    # caused inf values to appear when they should not.

    # We can't simply move the `constant_folding` optimization to
    # specialize since this will break other optimizations.  We will need to
    # partially duplicate some canonicalize optimizations to fix this issue.

    x2 = scalar()
    y2 = log(1 + exp(x2))
    mode = get_default_mode()
    mode.check_isfinite = False
    f2 = function([x2], y2, mode=mode)

    assert len(f2.maker.fgraph.toposort()) == 1
    assert f2.maker.fgraph.toposort()[0].op == softplus
    assert f2(800) == 800

    x = aet.as_tensor_variable(800)
    y = log(1 + exp(x))
    f = function([], y, mode=mode)
    # When this error is fixed, the following line should be ok.
    assert f() == 800, f()


class TestLocalSwitchSink:
    def setup_method(self):
        # condition values
        self.condm = np.asarray([[0.1, 0, 1, -1], [0.0, 0.0, 0.0, 0.0], [1, 1, 1, 1]])
        self.condv = np.asarray([0.1, 0, 1, -1])
        self.conds = [0.1, 0, 1, -1]

        # x values
        self.xm = np.ones((3, 4))
        self.xv = np.ones((4,))
        self.xs = 1.0

        # expected results
        self.resm = (
            [np.asarray([[1, 0, 1, 0], [0, 0, 0, 0], [1, 1, 1, 1]])] * 3
            + [np.asarray([[1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0]])]
            + 2 * [np.asarray([[1, 0, 1, 0]])]
            + [[np.ones((3, 4)), np.zeros((3, 4)), np.ones((3, 4)), np.zeros((3, 4))]]
            + [[np.ones((4,)), np.zeros((4,)), np.ones((4,)), np.zeros((4,))]]
            + [[np.asarray(1.0), np.asarray(0.0), np.asarray(1.0), np.asarray(0.0)]]
        )

        self.mode = (
            get_default_mode()
            .including("canonicalize", "fast_run")
            .excluding("gpu", "fusion")
        )
        self.mode = copy.copy(self.mode)
        self.mode.check_isfinite = False

    def function_remove_nan(self, *args, **kwargs):
        """
        Wrapper around function for this test.

        It disables checking for NaN removed by optimizations in DebugMode
        (it has false positives in that case).
        """
        f = function(*args, **kwargs)

        def wrapped_f(*args, **kwargs):
            # This is a bit ugly since it changes the global value of
            # TensorType.values_eq_approx.
            old_values_eq_approx = staticmethod(TensorType.values_eq_approx)
            TensorType.values_eq_approx = staticmethod(values_eq_approx_remove_nan)
            try:
                out = f(*args, **kwargs)
            finally:
                TensorType.values_eq_approx = old_values_eq_approx
            return out

        return wrapped_f

    def test_local_mul_switch_sink(self):
        c = dscalar()
        idx = 0
        for condition in [
            (dmatrix("cond"), self.condm),
            (dvector("cond"), self.condv),
            (dscalar("cond"), self.conds),
        ]:
            for x in [
                (dmatrix("x"), self.xm),
                (dvector("x"), self.xv),
                (dscalar("x"), self.xs),
            ]:
                y = mul(
                    aet.switch(condition[0] > 0, 1.0 * x[0], 0.0 * x[0]),
                    aet.switch(condition[0] > 0, 1.0 * x[0], log(c) * x[0]),
                )
                f = self.function_remove_nan(
                    [condition[0], x[0], c], [y], mode=self.mode
                )
                if type(condition[1]) is list:
                    for i in range(len(condition[1])):
                        res = f(condition[1][i], x[1], -1)
                        assert (
                            res == np.asarray(self.resm[idx][i])
                        ).sum() == self.resm[idx][i].size
                else:
                    res = f(condition[1], x[1], -1)
                    assert (res == np.asarray(self.resm[idx])).sum() == self.resm[
                        idx
                    ].size
                idx += 1

        # This case caused a missed optimization in the past.
        x = dscalar("x")
        y = aet.switch(x < 7, x, sqrt(x - 7))
        f = self.function_remove_nan([x], aesara.gradient.grad(y, x), self.mode)
        assert f(5) == 1, f(5)

    @pytest.mark.slow
    def test_local_div_switch_sink(self):
        c = dscalar()
        idx = 0
        for condition in [
            (dmatrix("cond"), self.condm),
            (dvector("cond"), self.condv),
            (dscalar("cond"), self.conds),
        ]:
            for x in [
                (dmatrix("x"), self.xm),
                (dvector("x"), self.xv),
                (dscalar("x"), self.xs),
            ]:
                y = true_div(
                    aet.switch(condition[0] > 0, 1.0 * x[0], 0.0 * x[0]),
                    aet.switch(condition[0] > 0, 1.0 * x[0], log(c) * x[0]),
                )
                f = self.function_remove_nan(
                    [condition[0], x[0], c], [y], mode=self.mode
                )
                if type(condition[1]) is list:
                    for i in range(len(condition[1])):
                        res = f(condition[1][i], x[1], -1)
                        assert (
                            res == np.asarray(self.resm[idx][i])
                        ).sum() == self.resm[idx][i].size
                else:
                    res = f(condition[1], x[1], -1)
                    assert (res == np.asarray(self.resm[idx])).sum() == self.resm[
                        idx
                    ].size
                idx += 1


class TestLocalUselessSwitch:
    def setup_method(self):
        self.mode = mode_opt.excluding("constant_folding")

    def test_const_0(self):
        for dtype1 in ["int32", "int64"]:
            for dtype2 in ["int32", "int64"]:
                x = matrix("x", dtype=dtype1)
                y = matrix("y", dtype=dtype2)
                z = aet.switch(0, x, y)
                f = function([x, y], z, mode=self.mode)
                assert (
                    len(
                        [
                            node.op
                            for node in f.maker.fgraph.toposort()
                            if (
                                isinstance(node.op, Elemwise)
                                and isinstance(node.op.scalar_op, aes.basic.Switch)
                            )
                        ]
                    )
                    == 0
                )
                vx = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype1)
                vy = np.array([[7, 8, 9], [10, 11, 12]], dtype=dtype2)
                np_res = np.where(0, vx, vy)
                assert np.array_equal(f(vx, vy), np_res)

        res_non_bool_np = np.where(np.ones(10), 0, 1)
        non_bool_graph = aet.switch(np.ones(10), 0, 1)
        non_bool_fn = function([], non_bool_graph, mode=self.mode)
        assert np.array_equal(non_bool_fn(), res_non_bool_np)

    def test_const_1(self):
        for dtype1 in ["int32", "int64"]:
            for dtype2 in ["int32", "int64"]:
                x = matrix("x", dtype=dtype1)
                y = matrix("y", dtype=dtype2)
                z = aet.switch(1, x, y)
                f = function([x, y], z, mode=self.mode)
                assert (
                    len(
                        [
                            node.op
                            for node in f.maker.fgraph.toposort()
                            if (
                                isinstance(node.op, Elemwise)
                                and isinstance(node.op.scalar_op, aes.basic.Switch)
                            )
                        ]
                    )
                    == 0
                )
                vx = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype1)
                vy = np.array([[7, 8, 9], [10, 11, 12]], dtype=dtype2)
                np_res = np.where(1, vx, vy)
                assert np.array_equal(f(vx, vy), np_res)

    def test_left_is_right(self):
        for dtype1 in ["int32", "int64"]:
            x = matrix("x", dtype=dtype1)
            varc = matrix("varc", dtype=dtype1)
            z1 = aet.switch(1, x, x)
            z0 = aet.switch(0, x, x)
            z2 = aet.switch(varc, x, x)
            f1 = function([x], z1, mode=self.mode)
            f0 = function([x], z0, mode=self.mode)
            f2 = function([x, varc], z2, mode=self.mode)

            topo = f1.maker.fgraph.toposort()
            assert len(topo) == 1
            assert topo[0].op == deep_copy_op

            topo = f0.maker.fgraph.toposort()
            assert len(topo) == 1
            assert topo[0].op == deep_copy_op

            topo = f2.maker.fgraph.toposort()
            assert len(topo) == 1
            assert topo[0].op == deep_copy_op

            vx = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype1)
            vc = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype1)
            assert np.array_equal(f1(vx), vx)
            assert np.array_equal(f0(vx), vx)
            assert np.array_equal(f2(vx, vc), vx)

    def test_shape_le_0(self):
        for dtype1 in ["float32", "float64"]:
            x = matrix("x", dtype=dtype1)
            z0 = aet.switch(le(x.shape[0], 0), 0, x.shape[0])
            f0 = function([x], z0, mode=self.mode)
            assert isinstance(f0.maker.fgraph.toposort()[0].op, Shape_i)

            z1 = aet.switch(le(x.shape[1], 0), 0, x.shape[1])
            f1 = function([x], z1, mode=self.mode)
            assert isinstance(f1.maker.fgraph.toposort()[0].op, Shape_i)

            vx = np.random.randn(0, 5).astype(dtype1)
            assert f0(vx) == 0
            assert f1(vx) == 5

    def test_broadcasting_1(self):
        # test switch(cst, matrix, row)
        x = matrix("x", dtype="int32")
        y = vector("y", dtype="int64")

        z = aet.switch(1, x, y)
        f = function([x, y], z, mode=self.mode)

        assert isinstance(f.maker.fgraph.outputs[0].owner.op, Elemwise)
        assert isinstance(f.maker.fgraph.outputs[0].owner.op.scalar_op, aes.basic.Cast)
        assert not any(node.op == aet.switch for node in f.maker.fgraph.toposort())

        vx = np.array([[1, 2, 3], [4, 5, 6]], dtype="int32")
        vy = np.array([10, 11, 12], dtype="int64")
        np_res = np.where(1, vx, vy)
        assert np.array_equal(f(vx, vy), np_res)

        z = aet.switch(0, x, y)
        f = function([x, y], z, mode=self.mode)

        assert isinstance(f.maker.fgraph.outputs[0].owner.op, Alloc)
        assert f.maker.fgraph.inputs[1] == f.maker.fgraph.outputs[0].owner.inputs[0]
        assert not any(node.op == aet.switch for node in f.maker.fgraph.toposort())

        vx = np.array([[1, 2, 3], [4, 5, 6]], dtype="int32")
        vy = np.array([10, 11, 12], dtype="int64")
        np_res = np.where(0, vx, vy)
        assert np.array_equal(f(vx, vy), np_res)

    def test_broadcasting_2(self):
        # test switch(cst, vector, matrix)

        x = vector("x", dtype="int32")
        y = matrix("y", dtype="int64")

        z = aet.switch(1, x, y)
        f = function([x, y], z, mode=self.mode)

        assert isinstance(f.maker.fgraph.outputs[0].owner.op, Alloc)
        assert not any(node.op == aet.switch for node in f.maker.fgraph.toposort())

        vx = np.array([4, 5, 6], dtype="int32")
        vy = np.array([[7, 8, 9], [10, 11, 12]], dtype="int64")
        np_res = np.where(1, vx, vy)
        assert np.array_equal(f(vx, vy), np_res)

        z = aet.switch(0, x, y)
        f = function([x, y], z, mode=self.mode)

        assert isinstance(f.maker.fgraph.outputs[0].owner.op, DeepCopyOp)
        assert not any(node.op == aet.switch for node in f.maker.fgraph.toposort())

        vx = np.array([4, 5, 6], dtype="int32")
        vy = np.array([[7, 8, 9], [10, 11, 12]], dtype="int64")
        np_res = np.where(0, vx, vy)
        assert np.array_equal(f(vx, vy), np_res)

    def test_broadcasting_3(self):
        # test switch(matrix, same_vector, same_vector)

        x = matrix("x", dtype="int32")
        y = vector("y", dtype="int64")
        z = aet.switch(x, y, y)
        f = function([x, y], z, mode=self.mode)
        vx = np.array([[0, 1], [1, 0]], dtype="int32")
        vy = np.array([7, 8], dtype="int64")
        utt.assert_allclose(f(vx, vy), np.where(vx, vy, vy))

        assert isinstance(f.maker.fgraph.outputs[0].owner.op, Alloc)
        assert not any(node.op == aet.switch for node in f.maker.fgraph.toposort())


class TestLocalMergeSwitchSameCond:
    def test_elemwise(self):
        # float Ops
        mats = matrices("cabxy")
        c, a, b, x, y = mats
        s1 = aet.switch(c, a, b)
        s2 = aet.switch(c, x, y)
        for op in (
            add,
            sub,
            mul,
            true_div,
            int_div,
            floor_div,
            minimum,
            maximum,
            gt,
            lt,
            ge,
            le,
            eq,
            neq,
            aet_pow,
        ):
            g = optimize(FunctionGraph(mats, [op(s1, s2)]))
            assert str(g).count("Switch") == 1
        # integer Ops
        mats = imatrices("cabxy")
        c, a, b, x, y = mats
        s1 = aet.switch(c, a, b)
        s2 = aet.switch(c, x, y)
        for op in (
            bitwise_and,
            bitwise_or,
            bitwise_xor,
        ):
            g = optimize(FunctionGraph(mats, [op(s1, s2)]))
            assert str(g).count("Switch") == 1
        # add/mul with more than two inputs
        u, v = matrices("uv")
        s3 = aet.switch(c, u, v)
        for op in (add, mul):
            g = optimize(FunctionGraph(mats + [u, v], [op(s1, s2, s3)]))
            assert str(g).count("Switch") == 1


class TestLocalOptAlloc:
    dtype = "float32"

    def test_sum_upcast(self):
        s = lscalar()
        a = aet.alloc(np.asarray(5, dtype=self.dtype), s, s)
        with config.change_flags(warn_float64="raise"):
            f = function([s], a.sum())
            f(5)

    def test_prod_upcast(self):
        s = lscalar()
        a = aet.alloc(np.asarray(5, dtype=self.dtype), s, s)

        with config.change_flags(warn_float64="raise"):
            f = function([s], a.prod())
            f(5)

    @config.change_flags(on_opt_error="raise")
    def test_sum_bool_upcast(self):
        s = lscalar()
        a = aet.alloc(np.asarray(True, dtype="bool"), s, s)
        f = function([s], a.sum())
        f(5)
        # test with user specified dtype
        f = function([s], a.sum(dtype=self.dtype))
        f(5)
        # test only 1 axis summed
        f = function([s], a.sum(axis=0, dtype=self.dtype))
        f(5)


class TestLocalOptAllocF16(TestLocalOptAlloc):
    dtype = "float16"


class TestMakeVector(utt.InferShapeTester):
    def setup_method(self):
        super().setup_method()

    def test_make_vector(self):
        b = bscalar()
        i = iscalar()
        d = dscalar()

        # TODO: draw random values instead. Not really important.
        val = {b: 2, i: -3, d: 0.7}

        # Should work
        for (dtype, inputs) in [
            ("int8", (b, b)),
            ("int32", (i, b)),
            ("int32", (b, i)),
            ("float64", (b, i)),
            ("float64", (b, d)),
            ("float64", (d, i)),
            ("float64", ()),
            ("int64", ()),
        ]:
            mv = MakeVector(dtype=dtype)(*inputs)
            assert mv.dtype == dtype
            f = function([b, i, d], mv, on_unused_input="ignore")
            f(val[b], val[i], val[d])

            s = mv.sum()
            gb = aesara.gradient.grad(s, b, disconnected_inputs="ignore")
            gi = aesara.gradient.grad(s, i, disconnected_inputs="ignore")
            gd = aesara.gradient.grad(s, d, disconnected_inputs="ignore")

            g = function([b, i, d], [gb, gi, gd])
            g_val = g(val[b], val[i], val[d])

            if dtype in int_dtypes:
                # The gradient should be 0
                utt.assert_allclose(g_val, 0)
            else:
                for var, grval in zip((b, i, d), g_val):
                    float_inputs = []
                    if var.dtype in int_dtypes:
                        pass
                        # Currently we don't do any checks on these variables
                        # verify_grad doesn't support integer inputs yet
                        # however, the gradient on them is *not* defined to
                        # be 0
                    elif var not in inputs:
                        assert grval == 0
                    else:
                        float_inputs.append(var)

                # Build a function that takes float_inputs, use fix values for the
                # other inputs, and returns the MakeVector. Use it for verify_grad.
                if float_inputs:

                    def fun(*fl_inputs):
                        f_inputs = []
                        for var in f_inputs:
                            if var in fl_inputs:
                                # use symbolic variable
                                f_inputs.append(var)
                            else:
                                # use constant value
                                f_inputs.append(val[var])
                        return MakeVector(dtype=dtype)(*f_inputs)

                    utt.verify_grad(fun, [val[ri] for ri in float_inputs])

        # should fail
        for (dtype, inputs) in [
            ("int8", (b, i)),
            ("int8", (i, b)),
            ("int8", (b, d)),
            ("int8", (i, i)),
            ("int32", (d, i)),
            ("int32", (i, d)),
            ("float32", (i, d)),
        ]:
            try:
                MakeVector(dtype=dtype)(*inputs)
                raise Exception("Aesara should have raised an error")
            except AssertionError:
                pass

    def test_infer_shape(self):
        adscal = dscalar()
        bdscal = dscalar()
        aiscal = iscalar()
        biscal = iscalar()
        ciscal = iscalar()
        discal = iscalar()
        adscal_val = np.random.rand()
        bdscal_val = np.random.rand()
        aiscal_val = np.random.randint(10)
        biscal_val = np.random.randint(10)
        ciscal_val = np.random.randint(10)
        discal_val = np.random.randint(10)
        self._compile_and_check(
            [adscal, aiscal],
            [MakeVector("float64")(adscal, aiscal)],
            [adscal_val, aiscal_val],
            MakeVector,
        )

        self._compile_and_check(
            [adscal, bdscal, aiscal],
            [MakeVector("float64")(adscal, bdscal, aiscal)],
            [adscal_val, bdscal_val, aiscal_val],
            MakeVector,
        )

        self._compile_and_check(
            [aiscal, biscal, ciscal, discal],
            [MakeVector("int32")(aiscal, biscal, ciscal, discal)],
            [aiscal_val, biscal_val, ciscal_val, discal_val],
            MakeVector,
        )


def test_local_join_1():
    # test for vector
    a = vector("a")
    s = aet.stack([a])
    f = function([a], s, mode=mode_opt)
    val = f([1])
    assert np.all(val == [1])
    e = f.maker.fgraph.toposort()
    assert len([n for n in e if isinstance(n.op, Join)]) == 0
    assert f.maker.fgraph.outputs[0].dtype == config.floatX

    # test for matrix join(0,a)
    a = matrix("a")
    s = join(0, a)
    f = function([a], s, mode=mode_opt)
    val = f([[1]])
    assert np.all(val == [[1]])
    e = f.maker.fgraph.toposort()
    assert len([n for n in e if isinstance(n.op, Join)]) == 0
    assert f.maker.fgraph.outputs[0].dtype == config.floatX

    # test for matrix join(1,a)
    s = join(1, a)
    f = function([a], s, mode=mode_opt)
    val = f([[1]])
    assert np.all(val == [[1]])
    e = f.maker.fgraph.toposort()
    assert len([n for n in e if isinstance(n.op, Join)]) == 0
    assert f.maker.fgraph.outputs[0].dtype == config.floatX

    # test we don't apply when their is 2 inputs
    s = join(1, a, a)
    f = function([a], s, mode=mode_opt)
    val = f([[1]])
    assert np.all(val == [[1]])
    e = f.maker.fgraph.toposort()
    assert len([n for n in e if isinstance(n.op, Join)]) == 1
    assert f.maker.fgraph.outputs[0].dtype == config.floatX


def test_local_join_empty():
    # test for vector, vector, empty to vector
    empty_vec = np.asarray([], dtype=config.floatX)
    a = vector("a")
    s = aet.join(0, a, a, empty_vec)
    f = function([a], s, mode=mode_opt)
    val = f([1])
    assert np.all(val == [1])
    e = f.maker.fgraph.toposort()
    assert len([n for n in e if isinstance(n.op, Join)]) == 1
    assert all(
        [
            not isinstance(n.op, Join) or len(n.inputs) == 3
            for n in e
            if isinstance(n.op, Join)
        ]
    )
    assert f.maker.fgraph.outputs[0].dtype == config.floatX

    # test for matrix join(1,a)
    empty_mat = np.asarray([[]], dtype=config.floatX)
    m = matrix("m")
    s = join(1, empty_mat, m, m, m)
    f = function([m], s, mode=mode_opt)
    val = f([[1]])
    assert np.all(val == [[1]])
    e = f.maker.fgraph.toposort()
    assert len([n for n in e if isinstance(n.op, Join)]) == 1
    assert all(
        [
            not isinstance(n.op, Join) or len(n.inputs) == 4
            for n in e
            if isinstance(n.op, Join)
        ]
    )
    assert f.maker.fgraph.outputs[0].dtype == config.floatX
    # test for vector, vector, empty to matrix
    # We can't optimize this case.
    s = aet.stack([a, a, empty_vec])
    f = function([a], s, mode=mode_opt)
    val = f([])
    assert np.all(val == [1])
    e = f.maker.fgraph.toposort()
    assert len([n for n in e if isinstance(n.op, Join)]) == 1
    assert all(
        [
            not isinstance(n.op, Join) or len(n.inputs) == 4
            for n in e
            if isinstance(n.op, Join)
        ]
    )
    assert f.maker.fgraph.outputs[0].dtype == config.floatX
    # test for matrix join(0,a)
    # We can't optimize this case.
    s = join(0, m, np.asarray([[2.0]], dtype=config.floatX), m)
    f = function([m], s, mode=mode_opt)
    val = f([[1]])
    assert np.all(val == [[1], [2], [1]])
    e = f.maker.fgraph.toposort()
    assert len([n for n in e if isinstance(n.op, Join)]) == 1
    assert all(
        [
            not isinstance(n.op, Join) or len(n.inputs) == 4
            for n in e
            if isinstance(n.op, Join)
        ]
    )
    assert f.maker.fgraph.outputs[0].dtype == config.floatX


def test_local_join_make_vector():
    a, b, c, d, e = scalars("abcde")
    v = vector("v")
    mv = MakeVector(config.floatX)
    s = aet.join(0, mv(a), v, mv(b, c), mv(d, e))
    f = function([a, b, c, d, e, v], s, mode=mode_opt)
    val = f(1, 2, 3, 4, 6, [7, 8])
    assert np.all(val == [1, 7, 8, 2, 3, 4, 6])
    e = f.maker.fgraph.toposort()
    assert len([n for n in e if isinstance(n.op, Join)]) == 1
    assert all(
        [
            not isinstance(n.op, Join) or len(n.inputs) == 4
            for n in e
            if isinstance(n.op, Join)
        ]
    )
    assert f.maker.fgraph.outputs[0].dtype == config.floatX

    assert check_stack_trace(f, ops_to_check="all")


def test_local_tensor_scalar_tensor():
    dtypes = [
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
        "complex64",
        "complex128",
    ]

    for dtype in dtypes:
        t_type = TensorType(dtype=dtype, broadcastable=())
        t = t_type()
        s = aet.scalar_from_tensor(t)
        t2 = aet.tensor_from_scalar(s)

        f = function([t], t2, mode=mode_opt)
        e = f.maker.fgraph.toposort()
        cast_nodes = [
            n for n in e if isinstance(n.op, (TensorFromScalar, ScalarFromTensor))
        ]
        assert len(cast_nodes) == 0
        f(0)


def test_local_scalar_tensor_scalar():
    dtypes = [
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
        "complex64",
        "complex128",
    ]

    for dtype in dtypes:
        s_type = aes.Scalar(dtype=dtype)
        s = s_type()
        t = aet.tensor_from_scalar(s)
        s2 = aet.scalar_from_tensor(t)

        f = function([s], s2, mode=mode_opt)
        e = f.maker.fgraph.toposort()
        cast_nodes = [
            n for n in e if isinstance(n.op, (TensorFromScalar, ScalarFromTensor))
        ]
        assert len(cast_nodes) == 0
        f(0)


def test_local_useless_split():
    x = matrix("x")
    splits = ivector("splits")
    opt = aet.split(x, splits, n_splits=1)
    nonopt = aet.split(x, splits, n_splits=3)

    mode = get_default_mode().including("local_useless_split")
    f_opt = function([x, splits], opt, mode=mode)
    f_nonopt = function([x, splits], nonopt, mode=mode)

    f_opt(np.random.rand(4, 4).astype(config.floatX), [4])
    f_nonopt(np.random.rand(4, 4).astype(config.floatX), [1, 2, 1])
    graph_opt = f_opt.maker.fgraph.toposort()
    graph_nonopt = f_nonopt.maker.fgraph.toposort()

    assert isinstance(graph_opt[-1].op, DeepCopyOp)
    assert len(graph_nonopt) == 1
    assert isinstance(graph_nonopt[0].op, Split)

    assert check_stack_trace(f_opt, ops_to_check=[Assert])
    assert check_stack_trace(f_nonopt, ops_to_check="all")


def test_local_flatten_lift():
    for i in range(1, 4):
        x = tensor4()
        out = aet.flatten(exp(x), i)
        assert out.ndim == i
        mode = get_default_mode()
        mode = mode.including("local_reshape_lift")
        f = function([x], out, mode=mode)
        x_np = np.random.rand(5, 4, 3, 2).astype(config.floatX)
        out_np = f(x_np)
        topo = f.maker.fgraph.toposort()
        shape_out_np = tuple(x_np.shape[: i - 1]) + (np.prod(x_np.shape[i - 1 :]),)
        assert shape_out_np == out_np.shape

        reshape_nodes = [n for n in topo if isinstance(n.op, Reshape)]
        assert len(reshape_nodes) == 1 and aet.is_flat(
            reshape_nodes[0].outputs[0], ndim=i
        )
        assert isinstance(topo[-1].op, Elemwise)


class TestReshape:
    def setup_method(self):
        self.mode = mode_opt
        self.op = Reshape

    def test_local_reshape(self):
        a = fmatrix()
        b = self.op(3)(a, [2, 3, 4])
        c = self.op(1)(b, [24])
        f = function([a], c, mode=self.mode)
        topo = f.maker.fgraph.toposort()
        assert sum(isinstance(node.op, self.op) for node in topo) == 1

        # Check stack trace
        assert check_stack_trace(f, ops_to_check=[self.op])


class TestLocalUselessReshape:
    def setup_method(self):
        self.rng = np.random.RandomState(utt.fetch_seed())

    def test_0(self):
        mode = get_default_mode().including("local_useless_reshape")
        i = iscalar("i")
        m = aet.mgrid[
            0:i,
        ]
        f = function([i], m, mode=mode)
        topo = f.maker.fgraph.toposort()
        assert not any(isinstance(n.op, Reshape) for n in topo)

    def test_1(self):
        x = matrix("x")
        r = x.reshape(x.shape)

        m0 = get_default_mode()
        m1 = m0.including("local_useless_reshape")
        f1 = function([x], r, mode=m1)
        topo = f1.maker.fgraph.toposort()
        assert not any(isinstance(n.op, Reshape) for n in topo)

        m2 = m1.excluding("ShapeOpt")
        f2 = function([x], r, mode=m2)
        topo = f2.maker.fgraph.toposort()
        assert not any(isinstance(n.op, Reshape) for n in topo)

        # We do not need tests checking that stack traces are copied over,
        # because local_useless_reshape only removes nodes from the graph

    def test_2(self):
        x = matrix("x")
        r = x.reshape([Shape_i(i)(x) for i in range(x.ndim)])

        m0 = get_default_mode()
        m1 = m0.including("local_useless_reshape")
        f1 = function([x], r, mode=m1)
        topo = f1.maker.fgraph.toposort()
        assert not any(isinstance(n.op, Reshape) for n in topo)

        m2 = m1.excluding("ShapeOpt")
        f2 = function([x], r, mode=m2)
        topo = f2.maker.fgraph.toposort()
        assert not any(isinstance(n.op, Reshape) for n in topo)

    def test_m1(self):
        x = matrix("x")
        r = x.reshape((x.shape[0], -1))

        m0 = get_default_mode()
        m1 = m0.including("local_useless_reshape")
        f1 = function([x], r, mode=m1)
        topo = f1.maker.fgraph.toposort()
        assert not any(isinstance(n.op, Reshape) for n in topo)

        m2 = m1.excluding("ShapeOpt")
        f2 = function([x], r, mode=m2)
        topo = f2.maker.fgraph.toposort()
        assert not any(isinstance(n.op, Reshape) for n in topo)


class TestLocalReshapeToDimshuffle:
    def setup_method(self):
        self.rng = np.random.RandomState(utt.fetch_seed())

    def test_1(self):
        reshape_lift = out2in(local_reshape_to_dimshuffle)
        useless_reshape = out2in(local_useless_reshape)
        x = shared(
            self.rng.randn(
                4,
            )
        )
        y = shared(self.rng.randn(5, 6))
        reshape_x = reshape(x, (1, 4))
        reshape_y = reshape(y, (1, 5, 1, 6, 1, 1))

        g = FunctionGraph([x, y], [reshape_x, reshape_y])
        assert str(g) == (
            "FunctionGraph(Reshape{2}"
            "(<TensorType(float64, vector)>, "
            "TensorConstant{[1 4]}), "
            "Reshape{6}"
            "(<TensorType(float64, matrix)>, "
            "TensorConstant{[1 5 1 6 1 1]}))"
        )

        reshape_lift.optimize(g)
        useless_reshape.optimize(g)
        assert str(g) == (
            "FunctionGraph(InplaceDimShuffle{x,0}"
            "(<TensorType(float64, vector)>), "
            "InplaceDimShuffle{x,0,x,1,x,x}"
            "(Reshape{2}(<TensorType(float64, matrix)>, "
            "TensorConstant{[5 6]})))"
        )

        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(g, ops_to_check=(DimShuffle, Reshape))


def test_local_reshape_lift():
    x = tensor4()
    out = exp(x).reshape([x.size])
    assert out.ndim == 1
    mode = get_default_mode()
    mode = mode.including("local_reshape_lift")
    f = function([x], out, mode=mode)
    f(np.random.rand(5, 4, 3, 2).astype(config.floatX))
    topo = f.maker.fgraph.toposort()
    assert isinstance(topo[-2].op, Reshape)
    assert isinstance(topo[-1].op, Elemwise)
    # Check stacktrace was copied over correctly after opt was applied
    assert check_stack_trace(f, ops_to_check="last")


class TestLiftTransposeThroughDot:
    def simple_optimize(self, g):
        out2in(local_useless_elemwise).optimize(g)
        out2in(local_lift_transpose_through_dot).optimize(g)
        out2in(local_useless_elemwise).optimize(g)
        return g

    def test_matrix_matrix(self):
        a, b = matrices("ab")
        g = self.simple_optimize(FunctionGraph([a, b], [dot(a, b).T]))
        sg = "FunctionGraph(dot(InplaceDimShuffle{1,0}(b), InplaceDimShuffle{1,0}(a)))"
        assert str(g) == sg, (str(g), sg)
        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(g, ops_to_check="all")

    def test_row_matrix(self):
        a = vector("a")
        b = matrix("b")
        g = optimize(
            FunctionGraph([a, b], [dot(a.dimshuffle("x", 0), b).T]),
            level="stabilize",
        )
        sg = "FunctionGraph(dot(InplaceDimShuffle{1,0}(b), InplaceDimShuffle{0,x}(a)))"
        assert str(g) == sg, (str(g), sg)
        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(g, ops_to_check="all")

    def test_matrix_col(self):
        a = vector("a")
        b = matrix("b")
        g = optimize(
            FunctionGraph([a, b], [dot(b, a.dimshuffle(0, "x")).T]),
            level="stabilize",
        )
        sg = "FunctionGraph(dot(InplaceDimShuffle{x,0}(a), InplaceDimShuffle{1,0}(b)))"
        assert str(g) == sg, (str(g), sg)
        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(g, ops_to_check="all")


def test_local_upcast_elemwise_constant_inputs():
    s = dvector("s")
    x = aet_sum(log(10 ** s))
    f = function([s], [aesara.gradient.grad(x, s)])
    f([-42, -2.1, -1, -0.5, 0, 0.2, 1, 2, 12])

    # This test a corner where the optimization should not be applied.
    with config.change_flags(floatX="float32"):
        v = lvector()
        function([v], true_div(v, 2))


class TestShapeI(utt.InferShapeTester):
    def setup_method(self):
        super().setup_method()

    def test_perform(self):

        advec = vector()
        advec_val = np.random.rand(3).astype(config.floatX)
        f = function([advec], Shape_i(0)(advec))
        out = f(advec_val)
        utt.assert_allclose(out, advec_val.shape[0])

        admat = matrix()
        admat_val = np.random.rand(4, 3).astype(config.floatX)
        for i in range(2):
            f = function([admat], Shape_i(i)(admat))
            out = f(admat_val)
            utt.assert_allclose(out, admat_val.shape[i])

    def test_infer_shape(self):
        admat = matrix()
        admat_val = np.random.rand(3, 4).astype(config.floatX)
        self._compile_and_check([admat], [Shape_i(0)(admat)], [admat_val], Shape_i)

        self._compile_and_check([admat], [Shape_i(1)(admat)], [admat_val], Shape_i)


class TestShapeFeature:
    def test_scalar(self):
        x = scalar()
        cst = aet.constant(1).clone()
        o = x + cst
        fgraph = FunctionGraph([x], [o], clone=False)
        shape_feature = ShapeFeature()
        fgraph.attach_feature(shape_feature)
        assert shape_feature.same_shape(x, o)

    def test_vector(self):
        x = vector()
        cst = aet.constant(1).clone()
        o = x + cst
        fgraph = FunctionGraph([x], [o], clone=False)
        shape_feature = ShapeFeature()
        fgraph.attach_feature(shape_feature)
        assert shape_feature.same_shape(x, o)

    def test_vector2(self):
        x = vector()
        y = vector()
        o = x + y
        fgraph = FunctionGraph([x, y], [o], clone=False)
        shape_feature = ShapeFeature()
        fgraph.attach_feature(shape_feature)
        assert shape_feature.same_shape(x, o)
        # The following case isn't implemented
        assert not shape_feature.same_shape(y, o)

    def test_vector_dim(self):
        x = vector()
        y = vector()
        o = x + y
        fgraph = FunctionGraph([x, y], [o], clone=False)
        shape_feature = ShapeFeature()
        fgraph.attach_feature(shape_feature)
        assert shape_feature.same_shape(x, o, 0, 0)
        # The following case isn't implemented
        assert not shape_feature.same_shape(y, o, 0, 0)

    def test_vector_dim_err(self):
        x = vector()
        y = vector()
        o = x + y
        fgraph = FunctionGraph([x, y], [o], clone=False)
        shape_feature = ShapeFeature()
        fgraph.attach_feature(shape_feature)
        with pytest.raises(IndexError):
            shape_feature.same_shape(x, o, 1, 0)
        with pytest.raises(IndexError):
            shape_feature.same_shape(x, o, 0, 1)


def test_assert_op_gradient():
    x = vector("x")
    assert_op = Assert()
    cost = aet_sum(assert_op(x, x.size < 2))
    grad = aesara.gradient.grad(cost, x)
    func = function([x], grad)

    x_val = np.ones(shape=(1,), dtype=config.floatX)
    assert func(x_val) == 1


def test_local_merge_alloc():
    # Add this opt to the default mode,
    # otherwise, FAST_COMPILE fails.
    default_mode = get_default_mode()
    opt_mode = default_mode.including("local_merge_alloc")

    x = iscalar("x")
    y = iscalar("y")
    y2 = iscalar("y2")
    z = iscalar("z")
    w = iscalar("w")
    m = fscalar("m")
    # case 1
    # Alloc(Alloc(m, x, 1, 1, 1), x, y, z, w) -> Alloc(m, x, y, z, w)
    output = aet.alloc(aet.alloc(m, 1, y, 1, 1), x, y, z, w)
    f = function([m, x, y, z, w], output, mode=opt_mode)
    topo = f.maker.fgraph.toposort()
    assert len(topo) == 1
    assert isinstance(topo[0].op, Alloc)
    o = f(0.0, 1, 2, 3, 4)
    assert o.shape == (1, 2, 3, 4)

    # case 2
    # Alloc(Alloc(m, y, 1, 1), x, y, z, w) -> Alloc(m, x, y, z, w)
    output = aet.alloc(aet.alloc(m, y, 1, 1), x, y, z, w)
    f = function([m, x, y, z, w], output, mode=opt_mode)
    topo = f.maker.fgraph.toposort()
    assert len(topo) == 1
    assert isinstance(topo[0].op, Alloc)
    o = f(0.0, 1, 2, 3, 4)
    assert o.shape == (1, 2, 3, 4)

    # case 3
    # Alloc(Alloc(m, y1, 1, 1), x, y2, z, w) ->
    #   Alloc(m, x, assert(y1, y1==y2), z, w)
    output = aet.alloc(aet.alloc(m, y, 1, 1), x, y2, z, w)
    f = function([m, x, y, y2, z, w], output, mode=opt_mode)
    topo = f.maker.fgraph.toposort()
    assert len(topo) == 3
    assert isinstance(topo[-2].op, Assert)
    assert isinstance(topo[-1].op, Alloc)
    o = f(0.0, 1, 2, 2, 3, 4)
    assert o.shape == (1, 2, 3, 4)
    with pytest.raises((AssertionError, ValueError)):
        f(0.0, 1, 2, 5, 3, 4)


def test_local_useless_alloc():

    useless_alloc = out2in(local_useless_alloc)
    merge_alloc = out2in(local_merge_alloc)

    x = iscalar("x")
    y = iscalar("y")
    y2 = iscalar("y2")
    z = iscalar("z")
    w = iscalar("w")
    m = fscalar("m")

    # case 1
    # Alloc(Alloc(m, x, 1, 1, 1), x, y, z, w) -> Alloc(m, x, y, z, w)
    output = aet.alloc(aet.alloc(m, 1, y, 1, 1), x, y, z, w)
    g = FunctionGraph([m, x, y, z, w], [output])

    useless_alloc.optimize(g)
    merge_alloc.optimize(g)
    useless_alloc.optimize(g)

    topo = g.toposort()
    assert len(topo) == 1
    assert isinstance(topo[0].op, Alloc)

    # case 2
    # Alloc(Alloc(m, y, 1, 1), x, y, z, w) -> Alloc(m, x, y, z, w)
    output = aet.alloc(aet.alloc(m, y, 1, 1), x, y, z, w)
    g = FunctionGraph([m, x, y, z, w], [output])

    useless_alloc.optimize(g)
    merge_alloc.optimize(g)
    useless_alloc.optimize(g)

    topo = g.toposort()
    assert len(topo) == 1
    assert isinstance(topo[0].op, Alloc)

    # case 3
    # Alloc(Alloc(m, y1, 1, 1), x, y2, z, w) ->
    #   Alloc(m, x, assert(y1, y1==y2), z, w)
    output = aet.alloc(aet.alloc(m, y, 1, 1), x, y2, z, w)
    g = FunctionGraph([m, x, y, y2, z, w], [output])

    useless_alloc.optimize(g)
    merge_alloc.optimize(g)
    useless_alloc.optimize(g)

    topo = g.toposort()
    assert len(topo) == 3
    assert isinstance(topo[-2].op, Assert)
    assert isinstance(topo[-1].op, Alloc)


def test_apply_rebroadcast_opt():
    # Test the `Elemwise` case in `local_rebroadcast_lift` with `fgraph=None`.
    # This is called by in `apply_rebroadcast_opt`.
    a = vector(dtype="float32")
    b = tensor("float64", [True])
    x = b.astype(a.dtype)

    broadcastable = (False,)
    axis = [(i, broadcastable[i]) for i in range(len(broadcastable))]
    rval = Rebroadcast(*axis)(x)

    res = apply_rebroadcast_opt(rval)
    assert res is rval
