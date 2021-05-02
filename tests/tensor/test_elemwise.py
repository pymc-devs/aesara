import math
from copy import copy

import numpy as np
import pytest

import aesara
import aesara.scalar as aes
from aesara.compile.mode import Mode
from aesara.configdefaults import config
from aesara.graph.basic import Variable
from aesara.graph.fg import FunctionGraph
from aesara.link.basic import PerformLinker
from aesara.link.c.basic import CLinker, OpWiseCLinker
from aesara.tensor import as_tensor_variable
from aesara.tensor.basic import second
from aesara.tensor.elemwise import CAReduce, DimShuffle, Elemwise
from aesara.tensor.type import (
    TensorType,
    bmatrix,
    bscalar,
    matrix,
    scalar,
    vector,
    vectors,
)
from tests import unittest_tools


class TestDimShuffle(unittest_tools.InferShapeTester):
    op = DimShuffle
    type = TensorType
    dtype = aesara.config.floatX

    def with_linker(self, linker):
        for xsh, shuffle, zsh in [
            ((2, 3), (1, "x", 0), (3, 1, 2)),
            ((1, 2, 3), (1, 2), (2, 3)),
            ((1, 2, 1, 3), (1, 3), (2, 3)),
            ((2, 3, 4), (2, 1, 0), (4, 3, 2)),
            ((2, 3, 4), ("x", 2, 1, 0, "x"), (1, 4, 3, 2, 1)),
            ((1, 4, 3, 2, 1), (3, 2, 1), (2, 3, 4)),
            ((1, 1, 4), (1, 2), (1, 4)),
            ((1, 1, 1), (), ()),
            ((1,), ("x", "x"), (1, 1)),
        ]:
            ib = [(entry == 1) for entry in xsh]
            x = self.type(self.dtype, ib)("x")
            e = self.op(ib, shuffle)(x)
            f = copy(linker).accept(FunctionGraph([x], [e])).make_function()
            assert f(np.ones(xsh, dtype=self.dtype)).shape == zsh
            # test that DimShuffle.infer_shape work correctly
            x = self.type(self.dtype, ib)("x")
            e = self.op(ib, shuffle)(x)
            f = copy(linker).accept(FunctionGraph([x], [e.shape])).make_function()
            assert all(f(np.ones(xsh, dtype=self.dtype))) == all(zsh)

        # Test when we drop a axis that is not broadcastable
        ib = [False, True, False]
        x = self.type(self.dtype, ib)("x")
        with pytest.raises(ValueError):
            self.op(ib, shuffle)

        # Test when we drop a axis that don't have shape 1
        ib = [True, True, False]
        x = self.type(self.dtype, ib)("x")
        e = self.op(ib, (1, 2))(x)
        f = copy(linker).accept(FunctionGraph([x], [e.shape])).make_function()
        with pytest.raises(TypeError):
            f(np.ones((2, 1, 4)))

        # Test that we can't take a dimensions multiple time
        xsh, shuffle, zsh = ((1, 1, 4), (0, 1, 2, 0), (1, 4))
        ib = [False, True, False]
        x = self.type(self.dtype, ib)("x")
        with pytest.raises(ValueError):
            DimShuffle(ib, shuffle)

    def test_perform(self):
        self.with_linker(PerformLinker())

    def test_c_or_py(self):
        # Shape op don't have C code.
        # But This will test DimShuffle c code
        self.with_linker(OpWiseCLinker())

    def test_infer_shape(self):

        for xsh, shuffle in [
            ((2, 3), (1, "x", 0)),
            ((1, 2, 3), (1, 2)),
            ((1, 2, 1, 3), (1, 3)),
            ((2, 3, 4), (2, 1, 0)),
            ((2, 3, 4), ("x", 2, 1, 0, "x")),
            ((1, 4, 3, 2, 1), (3, 2, 1)),
            ((1, 1, 4), (1, 2)),
            ((1, 1, 1), ()),
            ((1,), ("x", "x")),
        ]:
            ib = [(entry == 1) for entry in xsh]
            adtens = self.type(self.dtype, ib)("x")
            adtens_val = np.ones(xsh, dtype=self.dtype)
            self._compile_and_check(
                [adtens],
                [self.op(ib, shuffle)(adtens)],
                [adtens_val],
                self.op,
                warn=False,
            )

    def test_too_big_rank(self):
        x = self.type(self.dtype, broadcastable=())()
        y = x.dimshuffle(("x",) * (np.MAXDIMS + 1))
        with pytest.raises(ValueError):
            y.eval({x: 0})


class TestBroadcast:
    # this is to allow other types to reuse this class to test their ops
    type = TensorType
    op = Elemwise

    ctype = TensorType
    cop = Elemwise

    openmp_minsize = 2 * config.openmp_elemwise_minsize
    openmp_minsize_sqrt = int(math.ceil(math.sqrt(openmp_minsize)))

    # The order is important if you change them.
    linkers = [PerformLinker, CLinker]

    def rand_val(self, shp):
        return np.asarray(np.random.rand(*shp), dtype=aesara.config.floatX)

    def rand_cval(self, shp):
        return np.asarray(np.random.rand(*shp), dtype=aesara.config.floatX)

    def setup_method(self):
        unittest_tools.seed_rng()

    def with_linker(self, linker, op, type, rand_val):
        for xsh, ysh in [
            ((3, 5), (3, 5)),
            ((3, 5), (1, 5)),
            ((3, 5), (3, 1)),
            ((1, 5), (5, 1)),
            ((1, 1), (1, 1)),
            ((self.openmp_minsize,), (self.openmp_minsize,)),
            (
                (self.openmp_minsize_sqrt, self.openmp_minsize_sqrt),
                (self.openmp_minsize_sqrt, self.openmp_minsize_sqrt),
            ),
            ((2, 3, 4, 5), (2, 3, 4, 5)),
            ((2, 3, 4, 5), (1, 3, 1, 5)),
            ((2, 3, 4, 5), (1, 1, 1, 1)),
            ((), ()),
        ]:
            x = type(aesara.config.floatX, [(entry == 1) for entry in xsh])("x")
            y = type(aesara.config.floatX, [(entry == 1) for entry in ysh])("y")
            e = op(aes.add)(x, y)
            f = copy(linker).accept(FunctionGraph([x, y], [e])).make_function()
            xv = rand_val(xsh)
            yv = rand_val(ysh)
            zv = xv + yv

            unittest_tools.assert_allclose(f(xv, yv), zv)

            # test Elemwise.infer_shape
            # the Shape op don't implement c_code!
            if isinstance(linker, PerformLinker):
                x = type(aesara.config.floatX, [(entry == 1) for entry in xsh])("x")
                y = type(aesara.config.floatX, [(entry == 1) for entry in ysh])("y")
                e = op(aes.add)(x, y)
                f = (
                    copy(linker)
                    .accept(FunctionGraph([x, y], [e.shape]))
                    .make_function()
                )
                assert tuple(f(xv, yv)) == tuple(zv.shape)

    def with_linker_inplace(self, linker, op, type, rand_val):
        for xsh, ysh in [
            ((5, 5), (5, 5)),
            ((5, 5), (1, 5)),
            ((5, 5), (5, 1)),
            ((1, 1), (1, 1)),
            ((2, 3, 4, 5), (2, 3, 4, 5)),
            ((2, 3, 4, 5), (1, 3, 1, 5)),
            ((2, 3, 4, 5), (1, 1, 1, 1)),
            ((), ()),
        ]:
            x = type(aesara.config.floatX, [(entry == 1) for entry in xsh])("x")
            y = type(aesara.config.floatX, [(entry == 1) for entry in ysh])("y")
            e = op(aes.Add(aes.transfer_type(0)), {0: 0})(x, y)
            f = copy(linker).accept(FunctionGraph([x, y], [e])).make_function()
            xv = rand_val(xsh)
            yv = rand_val(ysh)
            zv = xv + yv

            f(xv, yv)

            assert (xv == zv).all()
            # test Elemwise.infer_shape
            # the Shape op don't implement c_code!
            if isinstance(linker, PerformLinker):
                x = type(aesara.config.floatX, [(entry == 1) for entry in xsh])("x")
                y = type(aesara.config.floatX, [(entry == 1) for entry in ysh])("y")
                e = op(aes.Add(aes.transfer_type(0)), {0: 0})(x, y)
                f = (
                    copy(linker)
                    .accept(FunctionGraph([x, y], [e.shape]))
                    .make_function()
                )
                xv = rand_val(xsh)
                yv = rand_val(ysh)
                zv = xv + yv

                f(xv, yv)

                assert xv.shape == zv.shape

    def test_perform(self):
        self.with_linker(PerformLinker(), self.op, self.type, self.rand_val)

    @pytest.mark.skipif(
        not aesara.config.cxx, reason="G++ not available, so we need to skip this test."
    )
    def test_c(self):
        self.with_linker(CLinker(), self.cop, self.ctype, self.rand_cval)

    def test_perform_inplace(self):
        self.with_linker_inplace(PerformLinker(), self.op, self.type, self.rand_val)

    @pytest.mark.skipif(
        not aesara.config.cxx, reason="G++ not available, so we need to skip this test."
    )
    def test_c_inplace(self):
        self.with_linker_inplace(CLinker(), self.cop, self.ctype, self.rand_cval)

    @pytest.mark.skipif(
        not aesara.config.cxx, reason="G++ not available, so we need to skip this test."
    )
    def test_fill(self):
        for linker, op, t, rval in zip(
            self.linkers,
            [self.op, self.cop],
            [self.type, self.ctype],
            [self.rand_val, self.rand_cval],
        ):
            x = t(aesara.config.floatX, [0, 0])("x")
            y = t(aesara.config.floatX, [1, 1])("y")
            e = op(aes.Second(aes.transfer_type(0)), {0: 0})(x, y)
            f = linker().accept(FunctionGraph([x, y], [e])).make_function()
            xv = rval((5, 5))
            yv = rval((1, 1))
            f(xv, yv)
            assert (xv == yv).all()

    def test_fill_var(self):
        x = matrix()
        x.fill(3)

    def test_fill_grad(self):
        # Fix bug reported at
        # https://groups.google.com/d/topic/theano-users/nQshB8gUA6k/discussion
        x = TensorType(config.floatX, [0, 1, 0])("x")
        y = TensorType(config.floatX, [0, 1, 0])("y")
        e = second(x, y)
        aesara.grad(e.sum(), y)

    @pytest.mark.skipif(
        not aesara.config.cxx, reason="G++ not available, so we need to skip this test."
    )
    def test_weird_strides(self):
        for linker, op, t, rval in zip(
            self.linkers,
            [self.op, self.cop],
            [self.type, self.ctype],
            [self.rand_val, self.rand_cval],
        ):
            x = t(aesara.config.floatX, [0, 0, 0, 0, 0])("x")
            y = t(aesara.config.floatX, [0, 0, 0, 0, 0])("y")
            e = op(aes.add)(x, y)
            f = linker().accept(FunctionGraph([x, y], [e])).make_function()
            xv = rval((2, 2, 2, 2, 2))
            yv = rval((2, 2, 2, 2, 2)).transpose(4, 0, 3, 1, 2)
            zv = xv + yv
            assert (f(xv, yv) == zv).all()

    @pytest.mark.skipif(
        not aesara.config.cxx, reason="G++ not available, so we need to skip this test."
    )
    def test_same_inputs(self):
        for linker, op, t, rval in zip(
            self.linkers,
            [self.op, self.cop],
            [self.type, self.ctype],
            [self.rand_val, self.rand_cval],
        ):
            x = t(aesara.config.floatX, [0, 0])("x")
            e = op(aes.add)(x, x)
            f = linker().accept(FunctionGraph([x], [e])).make_function()
            xv = rval((2, 2))
            zv = xv + xv
            assert (f(xv) == zv).all()


class TestCAReduce(unittest_tools.InferShapeTester):
    cases = [
        ((5, 6), (0,), (6,)),
        ((5, 6), None, ()),
        ((5, 6), (), (5, 6)),
        ((5, 6), (-1,), (5,)),
    ]

    def check_out_shape(self, mode, op, dtype):
        if dtype == "floatX":
            dtype = aesara.config.floatX

        for inp_shape, axis, out_shape in self.cases:
            xv = np.asarray(np.random.rand(*inp_shape), dtype=dtype)

            if axis is None:
                broadcastable = [True]
            else:
                broadcastable = [True if len(axis) == 1 else False]
                axis = np.asarray(axis)

            axis_var = TensorType(aesara.config.floatX, broadcastable)("axis")

            x = TensorType(dtype, [(entry == 1) for entry in inp_shape])("x")
            e = as_tensor_variable(CAReduce(op)(x, axis))

            f = aesara.function([x, axis_var], e, mode=mode, on_unused_input="ignore")
            f_xv = f(xv, axis)

            assert f_xv.shape == out_shape

    def test_perform_noopt(self):
        self.check_out_shape(Mode(linker="py", optimizer=None), aes.add, dtype="floatX")

    def test_perform(self):
        self.check_out_shape(Mode(linker="py"), aes.add, dtype="floatX")
        self.check_out_shape(Mode(linker="py"), aes.mul, dtype="floatX")
        self.check_out_shape(Mode(linker="py"), aes.scalar_maximum, dtype="floatX")
        self.check_out_shape(Mode(linker="py"), aes.scalar_minimum, dtype="floatX")

        self.check_out_shape(Mode(linker="py"), aes.and_, dtype="bool")
        self.check_out_shape(Mode(linker="py"), aes.or_, dtype="bool")
        self.check_out_shape(Mode(linker="py"), aes.xor, dtype="bool")

    @pytest.mark.skipif(
        not aesara.config.cxx, reason="G++ not available, so we need to skip this test."
    )
    def test_c_noopt(self):
        # We need to make sure that we cover the corner cases that
        # optimizations normally cover
        self.check_out_shape(Mode(linker="c", optimizer=None), aes.add, dtype="floatX")

    @pytest.mark.skipif(
        not aesara.config.cxx, reason="G++ not available, so we need to skip this test."
    )
    def test_c(self):
        self.check_out_shape(Mode(linker="c"), aes.add, dtype="floatX")
        self.check_out_shape(Mode(linker="c"), aes.mul, dtype="floatX")
        self.check_out_shape(Mode(linker="c"), aes.scalar_maximum, dtype="floatX")
        self.check_out_shape(Mode(linker="c"), aes.scalar_minimum, dtype="floatX")

        self.check_out_shape(Mode(linker="c"), aes.and_, dtype="bool")
        self.check_out_shape(Mode(linker="c"), aes.or_, dtype="bool")
        self.check_out_shape(Mode(linker="c"), aes.xor, dtype="bool")


class TestBitOpReduceGrad:
    def setup_method(self):
        self.rng = np.random.RandomState(unittest_tools.fetch_seed())

    def test_all_grad(self):
        x = bmatrix("x")
        x_all = x.all()
        gx = aesara.grad(x_all, x)
        f = aesara.function([x], gx)
        x_random = self.rng.binomial(n=1, p=0.5, size=(5, 7)).astype("int8")
        for x_val in (x_random, np.zeros_like(x_random), np.ones_like(x_random)):
            gx_val = f(x_val)
            assert gx_val.shape == x_val.shape
            assert np.all(gx_val == 0)

    def test_any_grad(self):
        x = bmatrix("x")
        x_all = x.any()
        gx = aesara.grad(x_all, x)
        f = aesara.function([x], gx)
        x_random = self.rng.binomial(n=1, p=0.5, size=(5, 7)).astype("int8")
        for x_val in (x_random, np.zeros_like(x_random), np.ones_like(x_random)):
            gx_val = f(x_val)
            assert gx_val.shape == x_val.shape
            assert np.all(gx_val == 0)


class TestElemwise(unittest_tools.InferShapeTester):
    def test_elemwise_grad_bool(self):
        x = scalar(dtype="bool")
        y = bscalar()
        z = x * y
        dx, dy = aesara.grad(z, [x, y])

    def test_infer_shape(self):

        for s_left, s_right in [
            ((5, 6), (5, 6)),
            ((5, 6), (5, 1)),
            ((5, 6), (1, 6)),
            ((5, 1), (5, 6)),
            ((1, 6), (5, 6)),
            ((2, 3, 4, 5), (2, 3, 4, 5)),
            ((2, 3, 4, 5), (2, 3, 1, 5)),
            ((2, 3, 4, 5), (1, 3, 4, 5)),
            ((2, 1, 4, 5), (2, 3, 4, 5)),
            ((2, 3, 4, 1), (2, 3, 4, 5)),
        ]:
            dtype = aesara.config.floatX
            t_left = TensorType(dtype, [(entry == 1) for entry in s_left])()
            t_right = TensorType(dtype, [(entry == 1) for entry in s_right])()
            t_left_val = np.zeros(s_left, dtype=dtype)
            t_right_val = np.zeros(s_right, dtype=dtype)
            self._compile_and_check(
                [t_left, t_right],
                [Elemwise(aes.add)(t_left, t_right)],
                [t_left_val, t_right_val],
                Elemwise,
            )

    def test_input_dimensions_overflow(self):
        # Elemwise.perform used to compute the product
        # of input shapes to check if there was a zero in them,
        # it overflowed in this case.
        a, b, c, d, e, f = vectors("abcdef")
        s = a + b + c + d + e + f
        g = aesara.function([a, b, c, d, e, f], s, mode=Mode(linker="py"))
        g(*[np.zeros(2 ** 11, config.floatX) for i in range(6)])

    def check_input_dimensions_match(self, mode):
        """Make sure that our input validation works correctly and doesn't
        throw erroneous broadcast-based errors.
        """
        x_v = matrix("x")
        m_v = vector("m")

        x = np.array([[-1.32720483], [0.23442016]]).astype(config.floatX)
        m = np.array([0.0, 0.0]).astype(config.floatX)

        z_v = x_v - m_v
        f = aesara.function([x_v, m_v], z_v, mode=mode)

        res = f(x, m)

        assert np.array_equal(res, x - m)

    def test_input_dimensions_match_python(self):
        self.check_input_dimensions_match(Mode(linker="py"))

    @pytest.mark.xfail(
        reason="Elemwise C implementation does not broadcast parameters",
        exception=ValueError,
    )
    @pytest.mark.skipif(
        not aesara.config.cxx, reason="G++ not available, so we need to skip this test."
    )
    def test_input_dimensions_match_c(self):
        self.check_input_dimensions_match(Mode(linker="c"))


def test_not_implemented_elemwise_grad():
    # Regression test for unimplemented gradient in an Elemwise Op.

    class TestOp(aes.ScalarOp):
        def __init__(self):
            self.output_types_preference = aes.upgrade_to_float

        def impl(self, n, x):
            return x * n

        def grad(self, inputs, gout):
            (n, x) = inputs
            (gz,) = gout
            dy_dx = n
            return [aesara.gradient.grad_not_implemented(self, 0, n), gz * dy_dx]

    test_op = Elemwise(TestOp())
    x = scalar()
    assert isinstance(aesara.gradient.grad(test_op(2, x), x), Variable)

    # Verify that trying to use the not implemented gradient fails.
    with pytest.raises(aesara.gradient.NullTypeGradError):
        aesara.gradient.grad(test_op(x, 2), x)
