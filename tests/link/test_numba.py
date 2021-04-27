from functools import partial

import numpy as np
import pytest

import aesara.scalar as aes
import aesara.tensor as aet
from aesara import config
from aesara.compile.function import function
from aesara.compile.mode import Mode
from aesara.compile.sharedvalue import SharedVariable
from aesara.graph.fg import FunctionGraph
from aesara.graph.optdb import Query
from aesara.link.numba.linker import NumbaLinker
from aesara.scalar.basic import Composite, scalar_minimum
from aesara.scalar.basic_scipy import gammainc
from aesara.tensor import subtensor as aet_subtensor
from aesara.tensor.elemwise import Elemwise
from aesara.tensor.type import scalar


opts = Query(include=[None], exclude=["cxx_only", "BlasOpt"])
numba_mode = Mode(NumbaLinker(), opts)
py_mode = Mode("py", opts)


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
        assert_fn = partial(np.testing.assert_allclose, rtol=1e-4)

    fn_inputs = [i for i in fgraph.inputs if not isinstance(i, SharedVariable)]
    aesara_numba_fn = function(
        fn_inputs,
        fgraph.outputs,
        mode=numba_mode,
        accept_inplace=True,
    )
    numba_res = aesara_numba_fn(*inputs)

    aesara_py_fn = function(
        fn_inputs, fgraph.outputs, mode=py_mode, accept_inplace=True
    )
    py_res = aesara_py_fn(*inputs)

    if len(fgraph.outputs) > 1:
        for j, p in zip(numba_res, py_res):
            assert_fn(j, p)
    else:
        assert_fn(numba_res, py_res)

    return numba_res


@pytest.mark.parametrize(
    "inputs, input_vals, output_fn",
    [
        (
            [aet.vector() for i in range(4)],
            [np.random.randn(100).astype(config.floatX) for i in range(4)],
            lambda x, y, x1, y1: (x + y) * (x1 + y1) * y,
        )
    ],
)
def test_Elemwise(inputs, input_vals, output_fn):
    out_fg = FunctionGraph(inputs, [output_fn(*inputs)])
    compare_numba_and_py(out_fg, input_vals)


@pytest.mark.parametrize(
    "scalar_op, input_values",
    [
        (
            lambda x, y: x > y,
            [np.array(10).astype(config.floatX), np.array(20).astype(config.floatX)],
        ),  # LogicalComparison
        (
            lambda x, y: scalar_minimum(x, y),
            [np.array(10).astype(config.floatX), np.array(20).astype(config.floatX)],
        ),  # ScalarOp from numpy
        # SciPy Ops are not supported until numba-scipy is fixed. See numba/numba-scipy#54
        pytest.param(
            gammainc,
            [np.array(10).astype(config.floatX), np.array(20).astype(config.floatX)],
            marks=pytest.mark.xfail,
        ),  # ScalarOp from scipy.special
    ],
)
def test_ScalarOp(scalar_op, input_values):
    x_s = aes.float64("x")
    y_s = aes.float64("y")
    curr_op = scalar_op(x_s, y_s)
    out_fg = FunctionGraph([x_s, y_s], [curr_op])
    compare_numba_and_py(out_fg, [*input_values])


@pytest.mark.parametrize(
    "inputs, input_values",
    [
        (
            [scalar("x"), scalar("y")],
            [np.array(10).astype(config.floatX), np.array(20).astype(config.floatX)],
        ),
    ],
)
def test_numba_Composite(inputs, input_values):
    x_s = aes.float64("x")
    y_s = aes.float64("y")
    comp_op = Elemwise(Composite([x_s, y_s], [x_s + y_s * 2 + aes.exp(x_s - y_s)]))
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
