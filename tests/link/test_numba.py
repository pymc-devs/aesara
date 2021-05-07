import contextlib
import inspect
from unittest import mock

import numba
import numpy as np
import pytest

import aesara.scalar as aes
import aesara.scalar.basic as aesb
import aesara.tensor as aet
import aesara.tensor.basic as aetb
from aesara import config
from aesara.compile.function import function
from aesara.compile.mode import Mode
from aesara.compile.ops import ViewOp
from aesara.compile.sharedvalue import SharedVariable
from aesara.graph.basic import Apply, Constant
from aesara.graph.fg import FunctionGraph
from aesara.graph.op import Op
from aesara.graph.optdb import Query
from aesara.graph.type import Type
from aesara.link.numba.dispatch import create_numba_signature, get_numba_type
from aesara.link.numba.linker import NumbaLinker
from aesara.scalar.basic import Composite
from aesara.tensor import elemwise as aet_elemwise
from aesara.tensor import extra_ops
from aesara.tensor import subtensor as aet_subtensor
from aesara.tensor.elemwise import Elemwise
from aesara.tensor.shape import Reshape, Shape, Shape_i, SpecifyShape


class MyType(Type):
    def filter(self, data):
        return data

    def __eq__(self, other):
        return isinstance(other, MyType)

    def __hash__(self):
        return hash(MyType)


class MyOp(Op):
    def perform(self, *args):
        pass


class MySingleOut(Op):
    def make_node(self, a, b):
        return Apply(self, [a, b], [a.type()])

    def perform(self, node, inputs, outputs):
        res = (inputs[0] + inputs[1]).astype(inputs[0][0].dtype)
        outputs[0][0] = res


class MyMultiOut(Op):
    def make_node(self, a, b):
        return Apply(self, [a, b], [a.type(), b.type()])

    def perform(self, node, inputs, outputs):
        res1 = 2 * inputs[0]
        res2 = 2 * inputs[1]
        outputs[0][0] = res1
        outputs[1][0] = res2


opts = Query(include=[None], exclude=["cxx_only", "BlasOpt"])
numba_mode = Mode(NumbaLinker(), opts)
py_mode = Mode("py", opts)


def set_test_value(x, v):
    x.tag.test_value = v
    return x


def compare_shape_dtype(x, y):
    (x,) = x
    (y,) = y
    return x.shape == y.shape and x.dtype == y.dtype


def eval_python_only(fn_inputs, fgraph, inputs):
    """Evaluate the Numba implementation in pure Python for coverage purposes."""

    def py_tuple_setitem(t, i, v):
        ll = list(t)
        ll[i] = v
        return tuple(ll)

    def py_to_scalar(x):
        if isinstance(x, np.ndarray):
            return x.item()
        else:
            return x

    def njit_noop(*args, **kwargs):
        if len(args) == 1:
            return args[0]
        else:
            return lambda x: x

    def vectorize_noop(*args, **kwargs):
        def wrap(fn):
            # `numba.vectorize` allows an `out` positional argument.  We need
            # to account for that
            sig = inspect.signature(fn)
            nparams = len(sig.parameters)

            def inner_vec(*args):
                if len(args) > nparams:
                    out = args[-1]
                    out[:] = fn(*args[:nparams])
                else:
                    return fn(*args)

            return inner_vec

        return wrap

    with mock.patch("aesara.link.numba.dispatch.numba.njit", njit_noop), mock.patch(
        "aesara.link.numba.dispatch.numba.vectorize",
        vectorize_noop,
    ), mock.patch(
        "aesara.link.numba.dispatch.tuple_setitem", py_tuple_setitem
    ), mock.patch(
        "aesara.link.numba.dispatch.direct_cast", lambda x, dtype: x
    ), mock.patch(
        "aesara.link.numba.dispatch.numba.np.numpy_support.from_dtype",
        lambda dtype: dtype,
    ), mock.patch(
        "aesara.link.numba.dispatch.to_scalar", py_to_scalar
    ):
        aesara_numba_fn = function(
            fn_inputs,
            fgraph.outputs,
            mode=numba_mode,
            accept_inplace=True,
        )
        _ = aesara_numba_fn(*inputs)


def compare_numba_and_py(
    fgraph,
    inputs,
    assert_fn=None,
):
    """Function to compare python graph output and Numba compiled output for testing equality

    In the tests below computational graphs are defined in Aesara. These graphs are then passed to
    this function which then compiles the graphs in both Numba and python, runs the calculation
    in both and checks if the results are the same

    Parameters
    ----------
    fgraph: FunctionGraph
        Aesara function Graph object
    inputs: iter
        Inputs for function graph
    assert_fn: func, opt
        Assert function used to check for equality between python and Numba. If not
        provided uses np.testing.assert_allclose

    """
    if assert_fn is None:

        def assert_fn(x, y):
            return np.testing.assert_allclose(x, y, rtol=1e-4) and compare_shape_dtype(
                x, y
            )

    fn_inputs = [i for i in fgraph.inputs if not isinstance(i, SharedVariable)]

    aesara_py_fn = function(
        fn_inputs, fgraph.outputs, mode=py_mode, accept_inplace=True
    )
    py_res = aesara_py_fn(*inputs)

    aesara_numba_fn = function(
        fn_inputs,
        fgraph.outputs,
        mode=numba_mode,
        accept_inplace=True,
    )
    numba_res = aesara_numba_fn(*inputs)

    # Get some coverage
    eval_python_only(fn_inputs, fgraph, inputs)

    if len(fgraph.outputs) > 1:
        for j, p in zip(numba_res, py_res):
            assert_fn(j, p)
    else:
        assert_fn(numba_res, py_res)

    return numba_res


@pytest.mark.parametrize(
    "v, expected, force_scalar, not_implemented",
    [
        (MyType(), None, False, True),
        (aes.float32, numba.types.float32, False, False),
        (aet.fscalar, numba.types.Array(numba.types.float32, 0, "A"), False, False),
        (aet.fscalar, numba.types.float32, True, False),
        (aet.lvector, numba.types.int64[:], False, False),
        (aet.dmatrix, numba.types.float64[:, :], False, False),
        (aet.dmatrix, numba.types.float64, True, False),
    ],
)
def test_get_numba_type(v, expected, force_scalar, not_implemented):
    cm = (
        contextlib.suppress()
        if not not_implemented
        else pytest.raises(NotImplementedError)
    )
    with cm:
        res = get_numba_type(v, force_scalar=force_scalar)
        assert res == expected


@pytest.mark.parametrize(
    "v, expected, force_scalar",
    [
        (Apply(MyOp(), [], []), numba.types.void(), False),
        (Apply(MyOp(), [], []), numba.types.void(), True),
        (
            Apply(MyOp(), [aet.lvector()], []),
            numba.types.void(numba.types.int64[:]),
            False,
        ),
        (Apply(MyOp(), [aet.lvector()], []), numba.types.void(numba.types.int64), True),
        (
            Apply(MyOp(), [aet.dmatrix(), aes.float32()], [aet.dmatrix()]),
            numba.types.float64[:, :](numba.types.float64[:, :], numba.types.float32),
            False,
        ),
        (
            Apply(MyOp(), [aet.dmatrix(), aes.float32()], [aet.dmatrix()]),
            numba.types.float64(numba.types.float64, numba.types.float32),
            True,
        ),
        (
            Apply(MyOp(), [aet.dmatrix(), aes.float32()], [aet.dmatrix(), aes.int32()]),
            numba.types.Tuple([numba.types.float64[:, :], numba.types.int32])(
                numba.types.float64[:, :], numba.types.float32
            ),
            False,
        ),
        (
            Apply(MyOp(), [aet.dmatrix(), aes.float32()], [aet.dmatrix(), aes.int32()]),
            numba.types.Tuple([numba.types.float64, numba.types.int32])(
                numba.types.float64, numba.types.float32
            ),
            True,
        ),
    ],
)
def test_create_numba_signature(v, expected, force_scalar):
    res = create_numba_signature(v, force_scalar=force_scalar)
    assert res == expected


@pytest.mark.parametrize(
    "inputs, input_vals, output_fn",
    [
        (
            [aet.vector()],
            [np.random.randn(100).astype(config.floatX)],
            lambda x: aet.sigmoid(x),
        ),
        (
            [aet.vector() for i in range(4)],
            [np.random.randn(100).astype(config.floatX) for i in range(4)],
            lambda x, y, x1, y1: (x + y) * (x1 + y1) * y,
        ),
    ],
)
def test_Elemwise(inputs, input_vals, output_fn):
    out_fg = FunctionGraph(outputs=[output_fn(*inputs)])
    compare_numba_and_py(out_fg, input_vals)


@pytest.mark.parametrize(
    "inputs, input_values, scalar_fn",
    [
        (
            [aet.scalar("x"), aet.scalar("y")],
            [
                np.array(10, dtype=config.floatX),
                np.array(20, dtype=config.floatX),
            ],
            lambda x, y: x + y * 2 + aes.exp(x - y),
        ),
    ],
)
def test_numba_Composite(inputs, input_values, scalar_fn):
    x_s = aes.float64("x")
    y_s = aes.float64("y")
    comp_op = Elemwise(Composite([x_s, y_s], [scalar_fn(x_s, y_s)]))
    out_fg = FunctionGraph(inputs, [comp_op(*inputs)])
    compare_numba_and_py(out_fg, input_values)


@pytest.mark.parametrize(
    "x, indices",
    [
        (aet.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))), (1,)),
        (
            aet.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            (slice(None)),
        ),
        (aet.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))), (1, 2, 0)),
        (
            aet.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            (slice(1, 2), 1, slice(None)),
        ),
    ],
)
def test_Subtensor(x, indices):
    """Test NumPy's basic indexing."""
    out_aet = x[indices]
    assert isinstance(out_aet.owner.op, aet_subtensor.Subtensor)
    out_fg = FunctionGraph([], [out_aet])
    compare_numba_and_py(out_fg, [])


@pytest.mark.parametrize(
    "x, indices",
    [
        (aet.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))), ([1, 2],)),
        (
            aet.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            ([1, 2], slice(None)),
        ),
    ],
)
def test_AdvancedSubtensor1(x, indices):
    """Test NumPy's advanced indexing in one dimension."""
    out_aet = x[[1, 2]]
    assert isinstance(out_aet.owner.op, aet_subtensor.AdvancedSubtensor1)
    out_fg = FunctionGraph([], [out_aet])
    compare_numba_and_py(out_fg, [])


@pytest.mark.parametrize(
    "x, indices",
    [
        (aet.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))), ([1, 2], [2, 3])),
        (
            aet.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            ([1, 2], slice(None), [3, 4]),
        ),
    ],
)
def test_AdvancedSubtensor(x, indices):
    """Test NumPy's advanced indexing in more than one dimension."""
    out_aet = x[indices]
    assert isinstance(out_aet.owner.op, aet_subtensor.AdvancedSubtensor)
    out_fg = FunctionGraph([], [out_aet])
    compare_numba_and_py(out_fg, [])


@pytest.mark.parametrize(
    "x, y, indices",
    [
        (
            aet.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            aet.as_tensor(np.array(10)),
            (1,),
        ),
        (
            aet.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            aet.as_tensor(np.random.poisson(size=(4, 5))),
            (slice(None)),
        ),
        (
            aet.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            aet.as_tensor(np.array(10)),
            (1, 2, 0),
        ),
        (
            aet.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            aet.as_tensor(np.random.poisson(size=(1, 5))),
            (slice(1, 2), 1, slice(None)),
        ),
    ],
)
def test_IncSubtensor(x, y, indices):
    out_aet = aet.set_subtensor(x[indices], y)
    assert isinstance(out_aet.owner.op, aet_subtensor.IncSubtensor)
    out_fg = FunctionGraph([], [out_aet])
    compare_numba_and_py(out_fg, [])

    out_aet = aet.inc_subtensor(x[indices], y)
    assert isinstance(out_aet.owner.op, aet_subtensor.IncSubtensor)
    out_fg = FunctionGraph([], [out_aet])
    compare_numba_and_py(out_fg, [])

    x_at = x.type()
    out_aet = aet.set_subtensor(x_at[indices], y, inplace=True)
    assert isinstance(out_aet.owner.op, aet_subtensor.IncSubtensor)
    out_fg = FunctionGraph([x_at], [out_aet])
    compare_numba_and_py(out_fg, [x.data])


@pytest.mark.parametrize(
    "x, y, indices",
    [
        (
            aet.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            aet.as_tensor(np.random.poisson(size=(2, 4, 5))),
            ([1, 2],),
        ),
        (
            aet.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            aet.as_tensor(np.random.poisson(size=(2, 4, 5))),
            ([1, 2], slice(None)),
        ),
    ],
)
def test_AdvancedIncSubtensor1(x, y, indices):
    out_aet = aet.set_subtensor(x[indices], y)
    assert isinstance(out_aet.owner.op, aet_subtensor.AdvancedIncSubtensor1)
    out_fg = FunctionGraph([], [out_aet])
    compare_numba_and_py(out_fg, [])

    out_aet = aet.inc_subtensor(x[indices], y)
    assert isinstance(out_aet.owner.op, aet_subtensor.AdvancedIncSubtensor1)
    out_fg = FunctionGraph([], [out_aet])
    compare_numba_and_py(out_fg, [])

    x_at = x.type()
    out_aet = aet.set_subtensor(x_at[indices], y, inplace=True)
    assert isinstance(out_aet.owner.op, aet_subtensor.AdvancedIncSubtensor1)
    out_fg = FunctionGraph([x_at], [out_aet])
    compare_numba_and_py(out_fg, [x.data])


@pytest.mark.parametrize(
    "x, y, indices",
    [
        (
            aet.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            aet.as_tensor(np.random.poisson(size=(2, 5))),
            ([1, 2], [2, 3]),
        ),
        (
            aet.as_tensor(np.arange(3 * 4 * 5).reshape((3, 4, 5))),
            aet.as_tensor(np.random.poisson(size=(2, 4))),
            ([1, 2], slice(None), [3, 4]),
        ),
    ],
)
def test_AdvancedIncSubtensor(x, y, indices):
    out_aet = aet.set_subtensor(x[indices], y)
    assert isinstance(out_aet.owner.op, aet_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([], [out_aet])
    compare_numba_and_py(out_fg, [])

    out_aet = aet.inc_subtensor(x[indices], y)
    assert isinstance(out_aet.owner.op, aet_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([], [out_aet])
    compare_numba_and_py(out_fg, [])

    x_at = x.type()
    out_aet = aet.set_subtensor(x_at[indices], y)
    # Inplace isn't really implemented for `AdvancedIncSubtensor`, so we just
    # hack it on here
    out_aet.owner.op.inplace = True
    assert isinstance(out_aet.owner.op, aet_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([x_at], [out_aet])
    compare_numba_and_py(out_fg, [x.data])


@pytest.mark.parametrize(
    "x, i",
    [
        (np.zeros((20, 3)), 1),
    ],
)
def test_Shape(x, i):
    g = Shape()(aet.as_tensor_variable(x))
    g_fg = FunctionGraph([], [g])

    compare_numba_and_py(g_fg, [])

    g = Shape_i(i)(aet.as_tensor_variable(x))
    g_fg = FunctionGraph([], [g])

    compare_numba_and_py(g_fg, [])


@pytest.mark.parametrize(
    "v, shape",
    [
        (0.0, (2, 3)),
        (1.1, (2, 3)),
        (set_test_value(aet.scalar("a"), np.array(10.0, dtype=config.floatX)), (20,)),
        (set_test_value(aet.vector("a"), np.ones(10, dtype=config.floatX)), (20, 10)),
    ],
)
def test_Alloc(v, shape):
    g = aet.alloc(v, *shape)
    g_fg = FunctionGraph(outputs=[g])

    (numba_res,) = compare_numba_and_py(
        g_fg,
        [
            i.tag.test_value
            for i in g_fg.inputs
            if not isinstance(i, (SharedVariable, Constant))
        ],
    )

    assert numba_res.shape == shape


def test_AllocEmpty():

    x = aet.empty((2, 3), dtype="float32")
    x_fg = FunctionGraph([], [x])

    # We need cannot compare the values in the arrays, only the shapes and
    # dtypes
    compare_numba_and_py(x_fg, [], assert_fn=compare_shape_dtype)


@pytest.mark.parametrize(
    "v, offset",
    [
        (set_test_value(aet.vector(), np.arange(10, dtype=config.floatX)), 0),
        (set_test_value(aet.vector(), np.arange(10, dtype=config.floatX)), 1),
        (set_test_value(aet.vector(), np.arange(10, dtype=config.floatX)), -1),
    ],
)
def test_AllocDiag(v, offset):
    g = aetb.AllocDiag(offset=offset)(v)
    g_fg = FunctionGraph(outputs=[g])

    compare_numba_and_py(
        g_fg,
        [
            i.tag.test_value
            for i in g_fg.inputs
            if not isinstance(i, (SharedVariable, Constant))
        ],
    )


@pytest.mark.parametrize(
    "v, new_order, inplace",
    [
        # `{'drop': [], 'shuffle': [], 'augment': [0, 1]}`
        (
            set_test_value(
                aet.lscalar(name="a"),
                np.array(1, dtype=np.int64),
            ),
            ("x", "x"),
            True,
        ),
        # I.e. `a_aet.T`
        # `{'drop': [], 'shuffle': [1, 0], 'augment': []}`
        (
            set_test_value(
                aet.matrix("a"), np.array([[1.0, 2.0], [3.0, 4.0]], dtype=config.floatX)
            ),
            (1, 0),
            True,
        ),
        # `{'drop': [], 'shuffle': [0, 1], 'augment': [2]}`
        (
            set_test_value(
                aet.matrix("a"), np.array([[1.0, 2.0], [3.0, 4.0]], dtype=config.floatX)
            ),
            (1, 0, "x"),
            True,
        ),
        # `{'drop': [1], 'shuffle': [2, 0], 'augment': [0, 2, 4]}`
        (
            set_test_value(
                aet.tensor(config.floatX, [False, True, False], name="a"),
                np.array([[[1.0, 2.0]], [[3.0, 4.0]]], dtype=config.floatX),
            ),
            ("x", 2, "x", 0, "x"),
            True,
        ),
        # I.e. `a_aet.dimshuffle((0,))`
        # `{'drop': [1], 'shuffle': [0], 'augment': []}`
        (
            set_test_value(
                aet.tensor(config.floatX, [False, True], name="a"),
                np.array([[1.0], [2.0], [3.0], [4.0]], dtype=config.floatX),
            ),
            (0,),
            True,
        ),
        (
            set_test_value(
                aet.tensor(config.floatX, [False, True], name="a"),
                np.array([[1.0], [2.0], [3.0], [4.0]], dtype=config.floatX),
            ),
            (0,),
            True,
        ),
    ],
)
def test_Dimshuffle(v, new_order, inplace):
    g = aet_elemwise.DimShuffle(v.broadcastable, new_order, inplace=inplace)(v)
    g_fg = FunctionGraph(outputs=[g])
    compare_numba_and_py(
        g_fg,
        [
            i.tag.test_value
            for i in g_fg.inputs
            if not isinstance(i, (SharedVariable, Constant))
        ],
    )


@pytest.mark.parametrize(
    "v", [set_test_value(aes.float64(), np.array(1.0, dtype="float64"))]
)
def test_TensorFromScalar(v):
    g = aetb.TensorFromScalar()(v)
    g_fg = FunctionGraph(outputs=[g])
    compare_numba_and_py(
        g_fg,
        [
            i.tag.test_value
            for i in g_fg.inputs
            if not isinstance(i, (SharedVariable, Constant))
        ],
    )


@pytest.mark.parametrize(
    "v",
    [
        set_test_value(aet.scalar(), np.array(1.0, dtype=config.floatX)),
    ],
)
def test_ScalarFromTensor(v):
    g = aetb.ScalarFromTensor()(v)
    g_fg = FunctionGraph(outputs=[g])
    compare_numba_and_py(
        g_fg,
        [
            i.tag.test_value
            for i in g_fg.inputs
            if not isinstance(i, (SharedVariable, Constant))
        ],
    )


@pytest.mark.parametrize(
    "v, axis, fails",
    [
        (
            set_test_value(aet.matrix(), np.array([[1.0]], dtype=config.floatX)),
            [(0, True), (1, True)],
            False,
        ),
        (
            set_test_value(aet.matrix(), np.array([[1.0, 2.0]], dtype=config.floatX)),
            [(0, True), (1, False)],
            False,
        ),
        (
            set_test_value(aet.matrix(), np.array([[1.0, 2.0]], dtype=config.floatX)),
            [(0, True), (1, True)],
            True,
        ),
    ],
)
def test_Rebroadcast(v, axis, fails):
    g = aetb.Rebroadcast(*axis)(v)
    g_fg = FunctionGraph(outputs=[g])
    cm = contextlib.suppress() if not fails else pytest.raises(ValueError)
    with cm:
        compare_numba_and_py(
            g_fg,
            [
                i.tag.test_value
                for i in g_fg.inputs
                if not isinstance(i, (SharedVariable, Constant))
            ],
        )


@pytest.mark.parametrize(
    "v, dtype",
    [
        (set_test_value(aet.fscalar(), np.array(1.0, dtype="float32")), aesb.float64),
        (set_test_value(aet.dscalar(), np.array(1.0, dtype="float64")), aesb.float32),
    ],
)
def test_Cast(v, dtype):
    g = aesb.Cast(dtype)(v)
    g_fg = FunctionGraph(outputs=[g])
    compare_numba_and_py(
        g_fg,
        [
            i.tag.test_value
            for i in g_fg.inputs
            if not isinstance(i, (SharedVariable, Constant))
        ],
    )


@pytest.mark.parametrize(
    "v, shape, ndim",
    [
        (set_test_value(aet.vector(), np.arange(4, dtype=config.floatX)), (2, 2), 2),
        (
            set_test_value(aet.vector(), np.arange(4, dtype=config.floatX)),
            set_test_value(aet.lvector(), np.array([2, 2], dtype="int64")),
            2,
        ),
    ],
)
def test_Reshape(v, shape, ndim):
    g = Reshape(ndim)(v, shape)
    g_fg = FunctionGraph(outputs=[g])
    compare_numba_and_py(
        g_fg,
        [
            i.tag.test_value
            for i in g_fg.inputs
            if not isinstance(i, (SharedVariable, Constant))
        ],
    )


@pytest.mark.parametrize(
    "v, shape, fails",
    [
        (
            set_test_value(aet.matrix(), np.array([[1.0]], dtype=config.floatX)),
            (1, 1),
            False,
        ),
        (
            set_test_value(aet.matrix(), np.array([[1.0, 2.0]], dtype=config.floatX)),
            (1, 1),
            True,
        ),
    ],
)
def test_SpecifyShape(v, shape, fails):
    g = SpecifyShape()(v, shape)
    g_fg = FunctionGraph(outputs=[g])
    cm = contextlib.suppress() if not fails else pytest.raises(AssertionError)
    with cm:
        compare_numba_and_py(
            g_fg,
            [
                i.tag.test_value
                for i in g_fg.inputs
                if not isinstance(i, (SharedVariable, Constant))
            ],
        )


@pytest.mark.parametrize(
    "v",
    [
        set_test_value(aet.vector(), np.arange(4, dtype=config.floatX)),
    ],
)
def test_ViewOp(v):
    g = ViewOp()(v)
    g_fg = FunctionGraph(outputs=[g])
    compare_numba_and_py(
        g_fg,
        [
            i.tag.test_value
            for i in g_fg.inputs
            if not isinstance(i, (SharedVariable, Constant))
        ],
    )


@pytest.mark.parametrize(
    "x, y",
    [
        (
            set_test_value(aet.lvector(), np.arange(4, dtype="int64")),
            set_test_value(aet.dvector(), np.arange(4, dtype="float64")),
        ),
        (
            set_test_value(
                aet.dmatrix(), np.arange(4, dtype="float64").reshape((2, 2))
            ),
            set_test_value(aet.lscalar(), np.array(4, dtype="int64")),
        ),
    ],
)
def test_Second(x, y):
    # We use the `Elemwise`-wrapped version of `Second`
    g = aet.second(x, y)
    g_fg = FunctionGraph(outputs=[g])
    compare_numba_and_py(
        g_fg,
        [
            i.tag.test_value
            for i in g_fg.inputs
            if not isinstance(i, (SharedVariable, Constant))
        ],
    )


@pytest.mark.parametrize(
    "v, min, max",
    [
        (set_test_value(aet.scalar(), np.array(10, dtype=config.floatX)), 3.0, 7.0),
        (set_test_value(aet.scalar(), np.array(1, dtype=config.floatX)), 3.0, 7.0),
        (set_test_value(aet.scalar(), np.array(10, dtype=config.floatX)), 7.0, 3.0),
    ],
)
def test_Clip(v, min, max):
    g = aes.clip(v, min, max)
    g_fg = FunctionGraph(outputs=[g])

    compare_numba_and_py(
        g_fg,
        [
            i.tag.test_value
            for i in g_fg.inputs
            if not isinstance(i, (SharedVariable, Constant))
        ],
    )


@pytest.mark.parametrize(
    "vals, dtype",
    [
        (
            (
                set_test_value(aet.scalar(), np.array(1, dtype=config.floatX)),
                set_test_value(aet.scalar(), np.array(2, dtype=config.floatX)),
                set_test_value(aet.scalar(), np.array(3, dtype=config.floatX)),
            ),
            config.floatX,
        ),
    ],
)
def test_MakeVector(vals, dtype):
    g = aetb.MakeVector(dtype)(*vals)
    g_fg = FunctionGraph(outputs=[g])

    compare_numba_and_py(
        g_fg,
        [
            i.tag.test_value
            for i in g_fg.inputs
            if not isinstance(i, (SharedVariable, Constant))
        ],
    )


@pytest.mark.parametrize(
    "start, stop, step, dtype",
    [
        (
            set_test_value(aet.lscalar(), np.array(1)),
            set_test_value(aet.lscalar(), np.array(10)),
            set_test_value(aet.lscalar(), np.array(3)),
            config.floatX,
        ),
    ],
)
def test_ARange(start, stop, step, dtype):
    g = aetb.ARange(dtype)(start, stop, step)
    g_fg = FunctionGraph(outputs=[g])

    compare_numba_and_py(
        g_fg,
        [
            i.tag.test_value
            for i in g_fg.inputs
            if not isinstance(i, (SharedVariable, Constant))
        ],
    )


@pytest.mark.parametrize(
    "careduce_fn, axis, v",
    [
        (aet.sum, 0, set_test_value(aet.vector(), np.arange(3, dtype=config.floatX))),
        (aet.all, 0, set_test_value(aet.vector(), np.arange(3, dtype=config.floatX))),
        (
            aet.sum,
            0,
            set_test_value(
                aet.matrix(), np.arange(3 * 2, dtype=config.floatX).reshape((3, 2))
            ),
        ),
        (
            aet.sum,
            (0, 1),
            set_test_value(
                aet.matrix(), np.arange(3 * 2, dtype=config.floatX).reshape((3, 2))
            ),
        ),
        (
            aet.sum,
            (1, 0),
            set_test_value(
                aet.matrix(), np.arange(3 * 2, dtype=config.floatX).reshape((3, 2))
            ),
        ),
        (
            aet.sum,
            None,
            set_test_value(
                aet.matrix(), np.arange(3 * 2, dtype=config.floatX).reshape((3, 2))
            ),
        ),
        (
            aet.sum,
            1,
            set_test_value(
                aet.matrix(), np.arange(3 * 2, dtype=config.floatX).reshape((3, 2))
            ),
        ),
        (aet.prod, 0, set_test_value(aet.vector(), np.arange(3, dtype=config.floatX))),
        (
            aet.prod,
            0,
            set_test_value(
                aet.matrix(), np.arange(3 * 2, dtype=config.floatX).reshape((3, 2))
            ),
        ),
        (
            aet.prod,
            1,
            set_test_value(
                aet.matrix(), np.arange(3 * 2, dtype=config.floatX).reshape((3, 2))
            ),
        ),
    ],
)
def test_CAReduce(careduce_fn, axis, v):
    g = careduce_fn(v, axis=axis)
    g_fg = FunctionGraph(outputs=[g])

    compare_numba_and_py(
        g_fg,
        [
            i.tag.test_value
            for i in g_fg.inputs
            if not isinstance(i, (SharedVariable, Constant))
        ],
    )


@pytest.mark.parametrize(
    "vals, axis",
    [
        (
            (
                set_test_value(
                    aet.matrix(), np.random.normal(size=(1, 2)).astype(config.floatX)
                ),
                set_test_value(
                    aet.matrix(), np.random.normal(size=(1, 2)).astype(config.floatX)
                ),
            ),
            0,
        ),
        (
            (
                set_test_value(
                    aet.matrix(), np.random.normal(size=(2, 1)).astype(config.floatX)
                ),
                set_test_value(
                    aet.matrix(), np.random.normal(size=(3, 1)).astype(config.floatX)
                ),
            ),
            0,
        ),
        (
            (
                set_test_value(
                    aet.matrix(), np.random.normal(size=(1, 2)).astype(config.floatX)
                ),
                set_test_value(
                    aet.matrix(), np.random.normal(size=(1, 2)).astype(config.floatX)
                ),
            ),
            1,
        ),
        (
            (
                set_test_value(
                    aet.matrix(), np.random.normal(size=(2, 2)).astype(config.floatX)
                ),
                set_test_value(
                    aet.matrix(), np.random.normal(size=(2, 1)).astype(config.floatX)
                ),
            ),
            1,
        ),
    ],
)
def test_Join(vals, axis):
    g = aet.join(axis, *vals)
    g_fg = FunctionGraph(outputs=[g])

    compare_numba_and_py(
        g_fg,
        [
            i.tag.test_value
            for i in g_fg.inputs
            if not isinstance(i, (SharedVariable, Constant))
        ],
    )


def test_Join_view():
    vals = (
        set_test_value(
            aet.matrix(), np.random.normal(size=(2, 2)).astype(config.floatX)
        ),
        set_test_value(
            aet.matrix(), np.random.normal(size=(2, 2)).astype(config.floatX)
        ),
    )
    g = aetb.Join(view=1)(1, *vals)
    g_fg = FunctionGraph(outputs=[g])

    with pytest.raises(NotImplementedError):
        compare_numba_and_py(
            g_fg,
            [
                i.tag.test_value
                for i in g_fg.inputs
                if not isinstance(i, (SharedVariable, Constant))
            ],
        )


@pytest.mark.parametrize(
    "val, offset",
    [
        (
            set_test_value(
                aet.matrix(), np.arange(10 * 10, dtype=config.floatX).reshape((10, 10))
            ),
            0,
        ),
        (
            set_test_value(aet.vector(), np.arange(10, dtype=config.floatX)),
            0,
        ),
    ],
)
def test_ExtractDiag(val, offset):
    g = aet.diag(val, offset)
    g_fg = FunctionGraph(outputs=[g])

    compare_numba_and_py(
        g_fg,
        [
            i.tag.test_value
            for i in g_fg.inputs
            if not isinstance(i, (SharedVariable, Constant))
        ],
    )


@pytest.mark.parametrize(
    "n, m, k, dtype",
    [
        (set_test_value(aet.lscalar(), np.array(1, dtype=np.int64)), None, 0, None),
        (
            set_test_value(aet.lscalar(), np.array(1, dtype=np.int64)),
            set_test_value(aet.lscalar(), np.array(2, dtype=np.int64)),
            0,
            "float32",
        ),
        (
            set_test_value(aet.lscalar(), np.array(1, dtype=np.int64)),
            set_test_value(aet.lscalar(), np.array(2, dtype=np.int64)),
            1,
            "int64",
        ),
    ],
)
def test_Eye(n, m, k, dtype):
    g = aet.eye(n, m, k, dtype=dtype)
    g_fg = FunctionGraph(outputs=[g])

    compare_numba_and_py(
        g_fg,
        [
            i.tag.test_value
            for i in g_fg.inputs
            if not isinstance(i, (SharedVariable, Constant))
        ],
    )


@pytest.mark.parametrize(
    "inputs, op, exc",
    [
        (
            [
                set_test_value(
                    aet.matrix(), np.random.random(size=(2, 3)).astype(config.floatX)
                ),
                set_test_value(aet.lmatrix(), np.random.poisson(size=(2, 3))),
            ],
            MySingleOut,
            UserWarning,
        ),
        (
            [
                set_test_value(
                    aet.matrix(), np.random.random(size=(2, 3)).astype(config.floatX)
                ),
                set_test_value(aet.lmatrix(), np.random.poisson(size=(2, 3))),
            ],
            MyMultiOut,
            UserWarning,
        ),
    ],
)
def test_perform(inputs, op, exc):

    g = op()(*inputs)

    if isinstance(g, list):
        g_fg = FunctionGraph(outputs=g)
    else:
        g_fg = FunctionGraph(outputs=[g])

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            g_fg,
            [
                i.tag.test_value
                for i in g_fg.inputs
                if not isinstance(i, (SharedVariable, Constant))
            ],
        )


@pytest.mark.parametrize(
    "val",
    [
        set_test_value(aet.lscalar(), np.array(6, dtype="int64")),
    ],
)
def test_Bartlett(val):
    g = extra_ops.bartlett(val)
    g_fg = FunctionGraph(outputs=[g])

    compare_numba_and_py(
        g_fg,
        [
            i.tag.test_value
            for i in g_fg.inputs
            if not isinstance(i, (SharedVariable, Constant))
        ],
    )


@pytest.mark.parametrize(
    "val, axis, mode",
    [
        (
            set_test_value(
                aet.matrix(), np.arange(3, dtype=config.floatX).reshape((3, 1))
            ),
            1,
            "add",
        ),
        (
            set_test_value(
                aet.matrix(), np.arange(6, dtype=config.floatX).reshape((3, 2))
            ),
            0,
            "add",
        ),
        (
            set_test_value(
                aet.matrix(), np.arange(6, dtype=config.floatX).reshape((3, 2))
            ),
            1,
            "add",
        ),
        (
            set_test_value(
                aet.matrix(), np.arange(6, dtype=config.floatX).reshape((3, 2))
            ),
            0,
            "mul",
        ),
        (
            set_test_value(
                aet.matrix(), np.arange(6, dtype=config.floatX).reshape((3, 2))
            ),
            1,
            "mul",
        ),
    ],
)
def test_CumOp(val, axis, mode):
    g = extra_ops.CumOp(axis=axis, mode=mode)(val)
    g_fg = FunctionGraph(outputs=[g])

    compare_numba_and_py(
        g_fg,
        [
            i.tag.test_value
            for i in g_fg.inputs
            if not isinstance(i, (SharedVariable, Constant))
        ],
    )


@pytest.mark.parametrize(
    "val, n, axis",
    [
        (
            set_test_value(
                aet.matrix(), np.random.normal(size=(3, 2)).astype(config.floatX)
            ),
            0,
            0,
        ),
        (
            set_test_value(
                aet.matrix(), np.random.normal(size=(3, 2)).astype(config.floatX)
            ),
            0,
            1,
        ),
        (
            set_test_value(
                aet.matrix(), np.random.normal(size=(3, 2)).astype(config.floatX)
            ),
            1,
            0,
        ),
        (
            set_test_value(
                aet.matrix(), np.random.normal(size=(3, 2)).astype(config.floatX)
            ),
            1,
            1,
        ),
        (
            set_test_value(aet.lmatrix(), np.random.poisson(size=(3, 2))),
            0,
            0,
        ),
    ],
)
def test_DiffOp(val, axis, n):
    g = extra_ops.DiffOp(n=n, axis=axis)(val)
    g_fg = FunctionGraph(outputs=[g])

    (res,) = compare_numba_and_py(
        g_fg,
        [
            i.tag.test_value
            for i in g_fg.inputs
            if not isinstance(i, (SharedVariable, Constant))
        ],
    )


@pytest.mark.parametrize(
    "a, val",
    [
        (
            set_test_value(aet.lmatrix(), np.zeros((10, 2), dtype="int64")),
            set_test_value(aet.lscalar(), np.array(1, dtype="int64")),
        )
    ],
)
def test_FillDiagonal(a, val):
    g = extra_ops.FillDiagonal()(a, val)
    g_fg = FunctionGraph(outputs=[g])

    compare_numba_and_py(
        g_fg,
        [
            i.tag.test_value
            for i in g_fg.inputs
            if not isinstance(i, (SharedVariable, Constant))
        ],
    )


@pytest.mark.parametrize(
    "a, val, offset",
    [
        (
            set_test_value(aet.lmatrix(), np.zeros((10, 2), dtype="int64")),
            set_test_value(aet.lscalar(), np.array(1, dtype="int64")),
            set_test_value(aet.lscalar(), np.array(-1, dtype="int64")),
        ),
        (
            set_test_value(aet.lmatrix(), np.zeros((10, 2), dtype="int64")),
            set_test_value(aet.lscalar(), np.array(1, dtype="int64")),
            set_test_value(aet.lscalar(), np.array(0, dtype="int64")),
        ),
        (
            set_test_value(aet.lmatrix(), np.zeros((10, 3), dtype="int64")),
            set_test_value(aet.lscalar(), np.array(1, dtype="int64")),
            set_test_value(aet.lscalar(), np.array(1, dtype="int64")),
        ),
    ],
)
def test_FillDiagonalOffset(a, val, offset):
    g = extra_ops.FillDiagonalOffset()(a, val, offset)
    g_fg = FunctionGraph(outputs=[g])

    compare_numba_and_py(
        g_fg,
        [
            i.tag.test_value
            for i in g_fg.inputs
            if not isinstance(i, (SharedVariable, Constant))
        ],
    )


@pytest.mark.parametrize(
    "arr, shape, mode, order, exc",
    [
        (
            tuple(set_test_value(aet.lscalar(), v) for v in np.array([0])),
            set_test_value(aet.lvector(), np.array([2])),
            "raise",
            "C",
            None,
        ),
        (
            tuple(set_test_value(aet.lscalar(), v) for v in np.array([0, 0, 3])),
            set_test_value(aet.lvector(), np.array([2, 3, 4])),
            "raise",
            "C",
            None,
        ),
        (
            tuple(
                set_test_value(aet.lvector(), v)
                for v in np.array([[0, 1], [2, 0], [1, 3]])
            ),
            set_test_value(aet.lvector(), np.array([2, 3, 4])),
            "raise",
            "C",
            None,
        ),
        (
            tuple(
                set_test_value(aet.lvector(), v)
                for v in np.array([[0, 1], [2, 0], [1, 3]])
            ),
            set_test_value(aet.lvector(), np.array([2, 3, 4])),
            "raise",
            "F",
            NotImplementedError,
        ),
        (
            tuple(
                set_test_value(aet.lvector(), v)
                for v in np.array([[0, 1, 2], [2, 0, 3], [1, 3, 5]])
            ),
            set_test_value(aet.lvector(), np.array([2, 3, 4])),
            "raise",
            "C",
            ValueError,
        ),
        (
            tuple(
                set_test_value(aet.lvector(), v)
                for v in np.array([[0, 1, 2], [2, 0, 3], [1, 3, 5]])
            ),
            set_test_value(aet.lvector(), np.array([2, 3, 4])),
            "wrap",
            "C",
            None,
        ),
        (
            tuple(
                set_test_value(aet.lvector(), v)
                for v in np.array([[0, 1, 2], [2, 0, 3], [1, 3, 5]])
            ),
            set_test_value(aet.lvector(), np.array([2, 3, 4])),
            "clip",
            "C",
            None,
        ),
    ],
)
def test_RavelMultiIndex(arr, shape, mode, order, exc):
    g = extra_ops.RavelMultiIndex(mode, order)(*(arr + (shape,)))
    g_fg = FunctionGraph(outputs=[g])

    cm = contextlib.suppress() if exc is None else pytest.raises(exc)
    with cm:
        compare_numba_and_py(
            g_fg,
            [
                i.tag.test_value
                for i in g_fg.inputs
                if not isinstance(i, (SharedVariable, Constant))
            ],
        )


@pytest.mark.parametrize(
    "x, repeats, axis, exc",
    [
        (
            set_test_value(aet.lscalar(), np.array(1, dtype="int64")),
            set_test_value(aet.lscalar(), np.array(0, dtype="int64")),
            None,
            None,
        ),
        (
            set_test_value(aet.lmatrix(), np.zeros((2, 2), dtype="int64")),
            set_test_value(aet.lscalar(), np.array(1, dtype="int64")),
            None,
            None,
        ),
        (
            set_test_value(aet.lvector(), np.arange(2, dtype="int64")),
            set_test_value(aet.lvector(), np.array([1, 1], dtype="int64")),
            None,
            None,
        ),
        (
            set_test_value(aet.lmatrix(), np.zeros((2, 2), dtype="int64")),
            set_test_value(aet.lscalar(), np.array(1, dtype="int64")),
            0,
            UserWarning,
        ),
    ],
)
def test_Repeat(x, repeats, axis, exc):
    g = extra_ops.Repeat(axis)(x, repeats)
    g_fg = FunctionGraph(outputs=[g])

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            g_fg,
            [
                i.tag.test_value
                for i in g_fg.inputs
                if not isinstance(i, (SharedVariable, Constant))
            ],
        )


@pytest.mark.parametrize(
    "x, axis, return_index, return_inverse, return_counts, exc",
    [
        (
            set_test_value(aet.lscalar(), np.array(1, dtype="int64")),
            None,
            False,
            False,
            False,
            None,
        ),
        (
            set_test_value(aet.lvector(), np.array([1, 1, 2], dtype="int64")),
            None,
            False,
            False,
            False,
            None,
        ),
        (
            set_test_value(aet.lmatrix(), np.array([[1, 1], [2, 2]], dtype="int64")),
            None,
            False,
            False,
            False,
            None,
        ),
        (
            set_test_value(
                aet.lmatrix(), np.array([[1, 1], [1, 1], [2, 2]], dtype="int64")
            ),
            0,
            False,
            False,
            False,
            UserWarning,
        ),
        (
            set_test_value(
                aet.lmatrix(), np.array([[1, 1], [1, 1], [2, 2]], dtype="int64")
            ),
            0,
            True,
            True,
            True,
            UserWarning,
        ),
    ],
)
def test_Unique(x, axis, return_index, return_inverse, return_counts, exc):
    g = extra_ops.Unique(return_index, return_inverse, return_counts, axis)(x)

    if isinstance(g, list):
        g_fg = FunctionGraph(outputs=g)
    else:
        g_fg = FunctionGraph(outputs=[g])

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            g_fg,
            [
                i.tag.test_value
                for i in g_fg.inputs
                if not isinstance(i, (SharedVariable, Constant))
            ],
        )


@pytest.mark.parametrize(
    "arr, shape, order, exc",
    [
        (
            set_test_value(aet.lvector(), np.array([9, 15, 1], dtype="int64")),
            aet.as_tensor([2, 3, 4]),
            "C",
            None,
        ),
        (
            set_test_value(aet.lvector(), np.array([1, 0], dtype="int64")),
            aet.as_tensor([2]),
            "C",
            None,
        ),
        (
            set_test_value(aet.lvector(), np.array([9, 15, 1], dtype="int64")),
            aet.as_tensor([2, 3, 4]),
            "F",
            NotImplementedError,
        ),
    ],
)
def test_UnravelIndex(arr, shape, order, exc):
    g = extra_ops.UnravelIndex(order)(arr, shape)

    if isinstance(g, list):
        g_fg = FunctionGraph(outputs=g)
    else:
        g_fg = FunctionGraph(outputs=[g])

    cm = contextlib.suppress() if exc is None else pytest.raises(exc)
    with cm:
        compare_numba_and_py(
            g_fg,
            [
                i.tag.test_value
                for i in g_fg.inputs
                if not isinstance(i, (SharedVariable, Constant))
            ],
        )


@pytest.mark.parametrize(
    "a, v, side, sorter, exc",
    [
        (
            set_test_value(
                aet.vector(), np.array([1.0, 2.0, 3.0], dtype=config.floatX)
            ),
            set_test_value(
                aet.matrix(), np.random.random((3, 2)).astype(config.floatX)
            ),
            "left",
            None,
            None,
        ),
        pytest.param(
            set_test_value(
                aet.vector(),
                np.array([0.29769574, 0.71649186, 0.20475563]).astype(config.floatX),
            ),
            set_test_value(
                aet.matrix(),
                np.array(
                    [
                        [0.18847123, 0.39659508],
                        [0.56220006, 0.57428752],
                        [0.86720994, 0.44522637],
                    ]
                ).astype(config.floatX),
            ),
            "left",
            None,
            None,
            marks=pytest.mark.xfail(
                reason="This won't work until https://github.com/numba/numba/pull/7005 is merged"
            ),
        ),
        (
            set_test_value(
                aet.vector(), np.array([1.0, 2.0, 3.0], dtype=config.floatX)
            ),
            set_test_value(
                aet.matrix(), np.random.random((3, 2)).astype(config.floatX)
            ),
            "right",
            set_test_value(aet.lvector(), np.array([0, 2, 1])),
            UserWarning,
        ),
    ],
)
def test_Searchsorted(a, v, side, sorter, exc):
    g = extra_ops.SearchsortedOp(side)(a, v, sorter)
    g_fg = FunctionGraph(outputs=[g])

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            g_fg,
            [
                i.tag.test_value
                for i in g_fg.inputs
                if not isinstance(i, (SharedVariable, Constant))
            ],
        )


@pytest.mark.parametrize(
    "x, shape, exc",
    [
        (
            set_test_value(
                aet.vector(), np.random.random(size=(2,)).astype(config.floatX)
            ),
            [set_test_value(aet.lscalar(), np.array(v)) for v in [3, 2]],
            UserWarning,
        ),
    ],
)
def test_BroadcastTo(x, shape, exc):
    g = extra_ops.BroadcastTo()(x, shape)
    g_fg = FunctionGraph(outputs=[g])

    cm = contextlib.suppress() if exc is None else pytest.warns(exc)
    with cm:
        compare_numba_and_py(
            g_fg,
            [
                i.tag.test_value
                for i in g_fg.inputs
                if not isinstance(i, (SharedVariable, Constant))
            ],
        )
