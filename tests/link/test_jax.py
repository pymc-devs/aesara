from functools import partial

import numpy as np
import pytest
from packaging.version import parse as version_parse

import aesara.scalar.basic as aes
from aesara.compile.function import function
from aesara.compile.mode import Mode
from aesara.compile.ops import DeepCopyOp, ViewOp
from aesara.compile.sharedvalue import SharedVariable, shared
from aesara.configdefaults import config
from aesara.graph.basic import Apply
from aesara.graph.fg import FunctionGraph
from aesara.graph.op import Op, get_test_value
from aesara.graph.optdb import Query
from aesara.ifelse import ifelse
from aesara.link.jax import JAXLinker
from aesara.scalar.basic import Composite
from aesara.scan.basic import scan
from aesara.tensor import basic as aet
from aesara.tensor import blas as aet_blas
from aesara.tensor import elemwise as aet_elemwise
from aesara.tensor import extra_ops as aet_extra_ops
from aesara.tensor import nlinalg as aet_nlinalg
from aesara.tensor import nnet as aet_nnet
from aesara.tensor import slinalg as aet_slinalg
from aesara.tensor import subtensor as aet_subtensor
from aesara.tensor.elemwise import Elemwise
from aesara.tensor.math import MaxAndArgmax
from aesara.tensor.math import all as aet_all
from aesara.tensor.math import clip, cosh, gammaln, log
from aesara.tensor.math import max as aet_max
from aesara.tensor.math import maximum, prod, sigmoid, softplus
from aesara.tensor.math import sum as aet_sum
from aesara.tensor.random.basic import RandomVariable, normal
from aesara.tensor.random.utils import RandomStream
from aesara.tensor.shape import Shape, Shape_i, SpecifyShape, reshape
from aesara.tensor.type import (
    dscalar,
    dvector,
    iscalar,
    ivector,
    lscalar,
    matrix,
    scalar,
    tensor,
    tensor3,
    vector,
)


jax = pytest.importorskip("jax")

opts = Query(include=[None], exclude=["cxx_only", "BlasOpt"])
jax_mode = Mode(JAXLinker(), opts)
py_mode = Mode("py", opts)


@pytest.fixture(scope="module", autouse=True)
def set_aesara_flags():
    with config.change_flags(cxx="", compute_test_value="ignore"):
        yield


def compare_jax_and_py(
    fgraph,
    inputs,
    assert_fn=None,
    must_be_device_array=True,
):
    """Function to compare python graph output and jax compiled output for testing equality

    In the tests below computational graphs are defined in Aesara. These graphs are then passed to
    this function which then compiles the graphs in both jax and python, runs the calculation
    in both and checks if the results are the same

    Parameters
    ----------
    fgraph: FunctionGraph
        Aesara function Graph object
    inputs: iter
        Inputs for function graph
    assert_fn: func, opt
        Assert function used to check for equality between python and jax. If not
        provided uses np.testing.assert_allclose
    must_be_device_array: Bool
        Checks for instance of jax.interpreters.xla.DeviceArray. For testing purposes
        if this device array is found it indicates if the result was computed by jax

    Returns
    -------
    jax_res

    """
    if assert_fn is None:
        assert_fn = partial(np.testing.assert_allclose, rtol=1e-4)

    fn_inputs = [i for i in fgraph.inputs if not isinstance(i, SharedVariable)]
    aesara_jax_fn = function(fn_inputs, fgraph.outputs, mode=jax_mode)
    jax_res = aesara_jax_fn(*inputs)

    if must_be_device_array:
        if isinstance(jax_res, list):
            assert all(
                isinstance(res, jax.interpreters.xla.DeviceArray) for res in jax_res
            )
        else:
            assert isinstance(jax_res, jax.interpreters.xla.DeviceArray)

    aesara_py_fn = function(fn_inputs, fgraph.outputs, mode=py_mode)
    py_res = aesara_py_fn(*inputs)

    if len(fgraph.outputs) > 1:
        for j, p in zip(jax_res, py_res):
            assert_fn(j, p)
    else:
        assert_fn(jax_res, py_res)

    return jax_res


def test_jax_Alloc():
    x = aet.alloc(0.0, 2, 3)
    x_fg = FunctionGraph([], [x])

    (jax_res,) = compare_jax_and_py(x_fg, [])

    assert jax_res.shape == (2, 3)

    x = aet.alloc(1.1, 2, 3)
    x_fg = FunctionGraph([], [x])

    compare_jax_and_py(x_fg, [])

    x = aet.AllocEmpty("float32")(2, 3)
    x_fg = FunctionGraph([], [x])

    def compare_shape_dtype(x, y):
        (x,) = x
        (y,) = y
        return x.shape == y.shape and x.dtype == y.dtype

    compare_jax_and_py(x_fg, [], assert_fn=compare_shape_dtype)

    a = scalar("a")
    x = aet.alloc(a, 20)
    x_fg = FunctionGraph([a], [x])

    compare_jax_and_py(x_fg, [10.0])

    a = vector("a")
    x = aet.alloc(a, 20, 10)
    x_fg = FunctionGraph([a], [x])

    compare_jax_and_py(x_fg, [np.ones(10, dtype=config.floatX)])


def test_jax_shape_ops():
    x_np = np.zeros((20, 3))
    x = Shape()(aet.as_tensor_variable(x_np))
    x_fg = FunctionGraph([], [x])

    compare_jax_and_py(x_fg, [], must_be_device_array=False)

    x = Shape_i(1)(aet.as_tensor_variable(x_np))
    x_fg = FunctionGraph([], [x])

    compare_jax_and_py(x_fg, [], must_be_device_array=False)


@pytest.mark.xfail(
    version_parse(jax.__version__) >= version_parse("0.2.12"),
    reason="Omnistaging cannot be disabled",
)
def test_jax_specify_shape():
    x_np = np.zeros((20, 3))
    x = SpecifyShape()(aet.as_tensor_variable(x_np), (20, 3))
    x_fg = FunctionGraph([], [x])

    compare_jax_and_py(x_fg, [])

    with config.change_flags(compute_test_value="off"):

        x = SpecifyShape()(aet.as_tensor_variable(x_np), (2, 3))
        x_fg = FunctionGraph([], [x])

        with pytest.raises(AssertionError):
            compare_jax_and_py(x_fg, [])


def test_jax_compile_ops():

    x = DeepCopyOp()(aet.as_tensor_variable(1.1))
    x_fg = FunctionGraph([], [x])

    compare_jax_and_py(x_fg, [])

    x_np = np.zeros((20, 1, 1))
    x = aet.Rebroadcast((0, False), (1, True), (2, False))(aet.as_tensor_variable(x_np))
    x_fg = FunctionGraph([], [x])

    compare_jax_and_py(x_fg, [])

    with config.change_flags(compute_test_value="off"):
        x = aet.Rebroadcast((0, True), (1, False), (2, False))(
            aet.as_tensor_variable(x_np)
        )
        x_fg = FunctionGraph([], [x])

        with pytest.raises(ValueError):
            compare_jax_and_py(x_fg, [])

    x = ViewOp()(aet.as_tensor_variable(x_np))
    x_fg = FunctionGraph([], [x])

    compare_jax_and_py(x_fg, [])


def test_jax_basic():
    x = matrix("x")
    y = matrix("y")
    b = vector("b")

    # `ScalarOp`
    z = cosh(x ** 2 + y / 3.0)

    # `[Inc]Subtensor`
    out = aet_subtensor.set_subtensor(z[0], -10.0)
    out = aet_subtensor.inc_subtensor(out[0, 1], 2.0)
    out = out[:5, :3]

    out_fg = FunctionGraph([x, y], [out])

    test_input_vals = [
        np.tile(np.arange(10), (10, 1)).astype(config.floatX),
        np.tile(np.arange(10, 20), (10, 1)).astype(config.floatX),
    ]
    (jax_res,) = compare_jax_and_py(out_fg, test_input_vals)

    # Confirm that the `Subtensor` slice operations are correct
    assert jax_res.shape == (5, 3)

    # Confirm that the `IncSubtensor` operations are correct
    assert jax_res[0, 0] == -10.0
    assert jax_res[0, 1] == -8.0

    out = clip(x, y, 5)
    out_fg = FunctionGraph([x, y], [out])
    compare_jax_and_py(out_fg, test_input_vals)

    out = aet.diagonal(x, 0)
    out_fg = FunctionGraph([x], [out])
    compare_jax_and_py(
        out_fg, [np.arange(10 * 10).reshape((10, 10)).astype(config.floatX)]
    )

    out = aet_slinalg.cholesky(x)
    out_fg = FunctionGraph([x], [out])
    compare_jax_and_py(
        out_fg,
        [(np.eye(10) + np.random.randn(10, 10) * 0.01).astype(config.floatX)],
    )

    # not sure why this isn't working yet with lower=False
    out = aet_slinalg.Cholesky(lower=False)(x)
    out_fg = FunctionGraph([x], [out])
    compare_jax_and_py(
        out_fg,
        [(np.eye(10) + np.random.randn(10, 10) * 0.01).astype(config.floatX)],
    )

    out = aet_slinalg.solve(x, b)
    out_fg = FunctionGraph([x, b], [out])
    compare_jax_and_py(
        out_fg,
        [
            np.eye(10).astype(config.floatX),
            np.arange(10).astype(config.floatX),
        ],
    )

    out = aet.diag(b)
    out_fg = FunctionGraph([b], [out])
    compare_jax_and_py(out_fg, [np.arange(10).astype(config.floatX)])

    out = aet_nlinalg.det(x)
    out_fg = FunctionGraph([x], [out])
    compare_jax_and_py(
        out_fg, [np.arange(10 * 10).reshape((10, 10)).astype(config.floatX)]
    )

    out = aet_nlinalg.matrix_inverse(x)
    out_fg = FunctionGraph([x], [out])
    compare_jax_and_py(
        out_fg,
        [(np.eye(10) + np.random.randn(10, 10) * 0.01).astype(config.floatX)],
    )


@pytest.mark.parametrize(
    "x, y, x_val, y_val",
    [
        (scalar("x"), scalar("y"), np.array(10), np.array(20)),
        (scalar("x"), vector("y"), np.array(10), np.arange(10, 20)),
        (
            matrix("x"),
            vector("y"),
            np.arange(10 * 20).reshape((20, 10)),
            np.arange(10, 20),
        ),
    ],
)
def test_jax_Composite(x, y, x_val, y_val):
    x_s = aes.float64("x")
    y_s = aes.float64("y")

    comp_op = Elemwise(Composite([x_s, y_s], [x_s + y_s * 2 + aes.exp(x_s - y_s)]))

    out = comp_op(x, y)

    out_fg = FunctionGraph([x, y], [out])

    test_input_vals = [
        x_val.astype(config.floatX),
        y_val.astype(config.floatX),
    ]
    _ = compare_jax_and_py(out_fg, test_input_vals)


def test_jax_FunctionGraph_names():
    import inspect

    from aesara.link.jax.dispatch import jax_funcify

    x = scalar("1x")
    y = scalar("_")
    z = scalar()
    q = scalar("def")

    out_fg = FunctionGraph([x, y, z, q], [x, y, z, q], clone=False)
    out_jx = jax_funcify(out_fg)
    sig = inspect.signature(out_jx)
    assert (x.auto_name, "_", z.auto_name, q.auto_name) == tuple(sig.parameters.keys())
    assert (1, 2, 3, 4) == out_jx(1, 2, 3, 4)


def test_jax_FunctionGraph_once():
    """Make sure that an output is only computed once when it's referenced multiple times."""
    from aesara.link.jax.dispatch import jax_funcify

    x = vector("x")
    y = vector("y")

    class TestOp(Op):
        def __init__(self):
            self.called = 0

        def make_node(self, *args):
            return Apply(self, list(args), [x.type() for x in args])

        def perform(self, inputs, outputs):
            for i, inp in enumerate(inputs):
                outputs[i][0] = inp[0]

    @jax_funcify.register(TestOp)
    def jax_funcify_TestOp(op, **kwargs):
        def func(*args, op=op):
            op.called += 1
            return list(args)

        return func

    op1 = TestOp()
    op2 = TestOp()

    q, r = op1(x, y)
    outs = op2(q + r, q + r)

    out_fg = FunctionGraph([x, y], outs, clone=False)
    assert len(out_fg.outputs) == 2

    out_jx = jax_funcify(out_fg)

    x_val = np.r_[1, 2].astype(config.floatX)
    y_val = np.r_[2, 3].astype(config.floatX)

    res = out_jx(x_val, y_val)
    assert len(res) == 2
    assert op1.called == 1
    assert op2.called == 1

    res = out_jx(x_val, y_val)
    assert len(res) == 2
    assert op1.called == 2
    assert op2.called == 2


def test_jax_eye():
    """Tests jaxification of the Eye operator"""
    out = aet.eye(3)
    out_fg = FunctionGraph([], [out])

    compare_jax_and_py(out_fg, [])


def test_jax_basic_multiout():

    np.random.seed(213234)
    M = np.random.normal(size=(3, 3))
    X = M.dot(M.T)

    x = matrix("x")

    outs = aet_nlinalg.eig(x)
    out_fg = FunctionGraph([x], outs)

    def assert_fn(x, y):
        np.testing.assert_allclose(x.astype(config.floatX), y, rtol=1e-3)

    compare_jax_and_py(out_fg, [X.astype(config.floatX)], assert_fn=assert_fn)

    outs = aet_nlinalg.eigh(x)
    out_fg = FunctionGraph([x], outs)
    compare_jax_and_py(out_fg, [X.astype(config.floatX)], assert_fn=assert_fn)

    outs = aet_nlinalg.qr(x, mode="full")
    out_fg = FunctionGraph([x], outs)
    compare_jax_and_py(out_fg, [X.astype(config.floatX)], assert_fn=assert_fn)

    outs = aet_nlinalg.qr(x, mode="reduced")
    out_fg = FunctionGraph([x], outs)
    compare_jax_and_py(out_fg, [X.astype(config.floatX)], assert_fn=assert_fn)

    outs = aet_nlinalg.svd(x)
    out_fg = FunctionGraph([x], outs)
    compare_jax_and_py(out_fg, [X.astype(config.floatX)], assert_fn=assert_fn)


@pytest.mark.xfail(
    version_parse(jax.__version__) >= version_parse("0.2.12"),
    reason="Omnistaging cannot be disabled",
)
def test_jax_basic_multiout_omni():
    # Test that a single output of a multi-output `Op` can be used as input to
    # another `Op`
    x = dvector()
    mx, amx = MaxAndArgmax([0])(x)
    out = mx * amx
    out_fg = FunctionGraph([x], [out])
    compare_jax_and_py(out_fg, [np.r_[1, 2]])


@pytest.mark.xfail(
    version_parse(jax.__version__) >= version_parse("0.2.12"),
    reason="Omnistaging cannot be disabled",
)
def test_jax_scan_multiple_output():
    """Test a scan implementation of a SEIR model.

    SEIR model definition:
    S[t+1] = S[t] - B[t]
    E[t+1] = E[t] +B[t] - C[t]
    I[t+1] = I[t+1] + C[t] - D[t]

    B[t] ~ Binom(S[t], beta)
    C[t] ~ Binom(E[t], gamma)
    D[t] ~ Binom(I[t], delta)
    """

    def binomln(n, k):
        return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)

    def binom_log_prob(n, p, value):
        return binomln(n, value) + value * log(p) + (n - value) * log(1 - p)

    # sequences
    aet_C = ivector("C_t")
    aet_D = ivector("D_t")
    # outputs_info (initial conditions)
    st0 = lscalar("s_t0")
    et0 = lscalar("e_t0")
    it0 = lscalar("i_t0")
    logp_c = scalar("logp_c")
    logp_d = scalar("logp_d")
    # non_sequences
    beta = scalar("beta")
    gamma = scalar("gamma")
    delta = scalar("delta")

    # TODO: Use random streams when their JAX conversions are implemented.
    # trng = aesara.tensor.random.RandomStream(1234)

    def seir_one_step(ct0, dt0, st0, et0, it0, logp_c, logp_d, beta, gamma, delta):
        # bt0 = trng.binomial(n=st0, p=beta)
        bt0 = st0 * beta
        bt0 = bt0.astype(st0.dtype)

        logp_c1 = binom_log_prob(et0, gamma, ct0).astype(logp_c.dtype)
        logp_d1 = binom_log_prob(it0, delta, dt0).astype(logp_d.dtype)

        st1 = st0 - bt0
        et1 = et0 + bt0 - ct0
        it1 = it0 + ct0 - dt0
        return st1, et1, it1, logp_c1, logp_d1

    (st, et, it, logp_c_all, logp_d_all), _ = scan(
        fn=seir_one_step,
        sequences=[aet_C, aet_D],
        outputs_info=[st0, et0, it0, logp_c, logp_d],
        non_sequences=[beta, gamma, delta],
    )
    st.name = "S_t"
    et.name = "E_t"
    it.name = "I_t"
    logp_c_all.name = "C_t_logp"
    logp_d_all.name = "D_t_logp"

    out_fg = FunctionGraph(
        [aet_C, aet_D, st0, et0, it0, logp_c, logp_d, beta, gamma, delta],
        [st, et, it, logp_c_all, logp_d_all],
    )

    s0, e0, i0 = 100, 50, 25
    logp_c0 = np.array(0.0, dtype=config.floatX)
    logp_d0 = np.array(0.0, dtype=config.floatX)
    beta_val, gamma_val, delta_val = [
        np.array(val, dtype=config.floatX) for val in [0.277792, 0.135330, 0.108753]
    ]
    C = np.array([3, 5, 8, 13, 21, 26, 10, 3], dtype=np.int32)
    D = np.array([1, 2, 3, 7, 9, 11, 5, 1], dtype=np.int32)

    test_input_vals = [
        C,
        D,
        s0,
        e0,
        i0,
        logp_c0,
        logp_d0,
        beta_val,
        gamma_val,
        delta_val,
    ]
    compare_jax_and_py(out_fg, test_input_vals)


@pytest.mark.xfail(
    version_parse(jax.__version__) >= version_parse("0.2.12"),
    reason="Omnistaging cannot be disabled",
)
def test_jax_scan_tap_output():

    a_aet = scalar("a")

    def input_step_fn(y_tm1, y_tm3, a):
        y_tm1.name = "y_tm1"
        y_tm3.name = "y_tm3"
        res = (y_tm1 + y_tm3) * a
        res.name = "y_t"
        return res

    y_scan_aet, _ = scan(
        fn=input_step_fn,
        outputs_info=[
            {
                "initial": aet.as_tensor_variable(
                    np.r_[-1.0, 1.3, 0.0].astype(config.floatX)
                ),
                "taps": [-1, -3],
            },
        ],
        non_sequences=[a_aet],
        n_steps=10,
        name="y_scan",
    )
    y_scan_aet.name = "y"
    y_scan_aet.owner.inputs[0].name = "y_all"

    out_fg = FunctionGraph([a_aet], [y_scan_aet])

    test_input_vals = [np.array(10.0).astype(config.floatX)]
    compare_jax_and_py(out_fg, test_input_vals)


def test_jax_Subtensors():
    # Basic indices
    x_aet = aet.arange(3 * 4 * 5).reshape((3, 4, 5))
    out_aet = x_aet[1, 2, 0]
    assert isinstance(out_aet.owner.op, aet_subtensor.Subtensor)
    out_fg = FunctionGraph([], [out_aet])
    compare_jax_and_py(out_fg, [])

    out_aet = x_aet[1:2, 1, :]
    assert isinstance(out_aet.owner.op, aet_subtensor.Subtensor)
    out_fg = FunctionGraph([], [out_aet])
    compare_jax_and_py(out_fg, [])

    # Advanced indexing
    out_aet = x_aet[[1, 2]]
    assert isinstance(out_aet.owner.op, aet_subtensor.AdvancedSubtensor1)
    out_fg = FunctionGraph([], [out_aet])
    compare_jax_and_py(out_fg, [])

    out_aet = x_aet[[1, 2], [2, 3]]
    assert isinstance(out_aet.owner.op, aet_subtensor.AdvancedSubtensor)
    out_fg = FunctionGraph([], [out_aet])
    compare_jax_and_py(out_fg, [])

    # Advanced and basic indexing
    out_aet = x_aet[[1, 2], :]
    assert isinstance(out_aet.owner.op, aet_subtensor.AdvancedSubtensor1)
    out_fg = FunctionGraph([], [out_aet])
    compare_jax_and_py(out_fg, [])

    out_aet = x_aet[[1, 2], :, [3, 4]]
    assert isinstance(out_aet.owner.op, aet_subtensor.AdvancedSubtensor)
    out_fg = FunctionGraph([], [out_aet])
    compare_jax_and_py(out_fg, [])


@pytest.mark.xfail(
    version_parse(jax.__version__) >= version_parse("0.2.12"),
    reason="Omnistaging cannot be disabled",
)
def test_jax_Subtensors_omni():
    x_aet = aet.arange(3 * 4 * 5).reshape((3, 4, 5))

    # Boolean indices
    out_aet = x_aet[x_aet < 0]
    assert isinstance(out_aet.owner.op, aet_subtensor.AdvancedSubtensor)
    out_fg = FunctionGraph([], [out_aet])
    compare_jax_and_py(out_fg, [])


@pytest.mark.xfail(
    version_parse(jax.__version__) >= version_parse("0.2.12"),
    reason="Omnistaging cannot be disabled",
)
def test_jax_IncSubtensor():
    x_np = np.random.uniform(-1, 1, size=(3, 4, 5)).astype(config.floatX)
    x_aet = aet.arange(3 * 4 * 5).reshape((3, 4, 5)).astype(config.floatX)

    # "Set" basic indices
    st_aet = aet.as_tensor_variable(np.array(-10.0, dtype=config.floatX))
    out_aet = aet_subtensor.set_subtensor(x_aet[1, 2, 3], st_aet)
    assert isinstance(out_aet.owner.op, aet_subtensor.IncSubtensor)
    out_fg = FunctionGraph([], [out_aet])
    compare_jax_and_py(out_fg, [])

    st_aet = aet.as_tensor_variable(np.r_[-1.0, 0.0].astype(config.floatX))
    out_aet = aet_subtensor.set_subtensor(x_aet[:2, 0, 0], st_aet)
    assert isinstance(out_aet.owner.op, aet_subtensor.IncSubtensor)
    out_fg = FunctionGraph([], [out_aet])
    compare_jax_and_py(out_fg, [])

    out_aet = aet_subtensor.set_subtensor(x_aet[0, 1:3, 0], st_aet)
    assert isinstance(out_aet.owner.op, aet_subtensor.IncSubtensor)
    out_fg = FunctionGraph([], [out_aet])
    compare_jax_and_py(out_fg, [])

    # "Set" advanced indices
    st_aet = aet.as_tensor_variable(
        np.random.uniform(-1, 1, size=(2, 4, 5)).astype(config.floatX)
    )
    out_aet = aet_subtensor.set_subtensor(x_aet[np.r_[0, 2]], st_aet)
    assert isinstance(out_aet.owner.op, aet_subtensor.AdvancedIncSubtensor1)
    out_fg = FunctionGraph([], [out_aet])
    compare_jax_and_py(out_fg, [])

    st_aet = aet.as_tensor_variable(np.r_[-1.0, 0.0].astype(config.floatX))
    out_aet = aet_subtensor.set_subtensor(x_aet[[0, 2], 0, 0], st_aet)
    assert isinstance(out_aet.owner.op, aet_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([], [out_aet])
    compare_jax_and_py(out_fg, [])

    st_aet = aet.as_tensor_variable(x_np[[0, 2], 0, :3])
    out_aet = aet_subtensor.set_subtensor(x_aet[[0, 2], 0, :3], st_aet)
    assert isinstance(out_aet.owner.op, aet_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([], [out_aet])
    compare_jax_and_py(out_fg, [])

    # "Set" boolean indices
    mask_aet = aet.as_tensor_variable(x_np) > 0
    out_aet = aet_subtensor.set_subtensor(x_aet[mask_aet], 0.0)
    assert isinstance(out_aet.owner.op, aet_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([], [out_aet])
    compare_jax_and_py(out_fg, [])

    # "Increment" basic indices
    st_aet = aet.as_tensor_variable(np.array(-10.0, dtype=config.floatX))
    out_aet = aet_subtensor.inc_subtensor(x_aet[1, 2, 3], st_aet)
    assert isinstance(out_aet.owner.op, aet_subtensor.IncSubtensor)
    out_fg = FunctionGraph([], [out_aet])
    compare_jax_and_py(out_fg, [])

    st_aet = aet.as_tensor_variable(np.r_[-1.0, 0.0].astype(config.floatX))
    out_aet = aet_subtensor.inc_subtensor(x_aet[:2, 0, 0], st_aet)
    assert isinstance(out_aet.owner.op, aet_subtensor.IncSubtensor)
    out_fg = FunctionGraph([], [out_aet])
    compare_jax_and_py(out_fg, [])

    out_aet = aet_subtensor.set_subtensor(x_aet[0, 1:3, 0], st_aet)
    assert isinstance(out_aet.owner.op, aet_subtensor.IncSubtensor)
    out_fg = FunctionGraph([], [out_aet])
    compare_jax_and_py(out_fg, [])

    # "Increment" advanced indices
    st_aet = aet.as_tensor_variable(
        np.random.uniform(-1, 1, size=(2, 4, 5)).astype(config.floatX)
    )
    out_aet = aet_subtensor.inc_subtensor(x_aet[np.r_[0, 2]], st_aet)
    assert isinstance(out_aet.owner.op, aet_subtensor.AdvancedIncSubtensor1)
    out_fg = FunctionGraph([], [out_aet])
    compare_jax_and_py(out_fg, [])

    st_aet = aet.as_tensor_variable(np.r_[-1.0, 0.0].astype(config.floatX))
    out_aet = aet_subtensor.inc_subtensor(x_aet[[0, 2], 0, 0], st_aet)
    assert isinstance(out_aet.owner.op, aet_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([], [out_aet])
    compare_jax_and_py(out_fg, [])

    st_aet = aet.as_tensor_variable(x_np[[0, 2], 0, :3])
    out_aet = aet_subtensor.inc_subtensor(x_aet[[0, 2], 0, :3], st_aet)
    assert isinstance(out_aet.owner.op, aet_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([], [out_aet])
    compare_jax_and_py(out_fg, [])

    # "Increment" boolean indices
    mask_aet = aet.as_tensor_variable(x_np) > 0
    out_aet = aet_subtensor.set_subtensor(x_aet[mask_aet], 1.0)
    assert isinstance(out_aet.owner.op, aet_subtensor.AdvancedIncSubtensor)
    out_fg = FunctionGraph([], [out_aet])
    compare_jax_and_py(out_fg, [])


def test_jax_ifelse():

    true_vals = np.r_[1, 2, 3]
    false_vals = np.r_[-1, -2, -3]

    x = ifelse(np.array(True), true_vals, false_vals)
    x_fg = FunctionGraph([], [x])

    compare_jax_and_py(x_fg, [])

    a = dscalar("a")
    a.tag.test_value = np.array(0.2, dtype=config.floatX)
    x = ifelse(a < 0.5, true_vals, false_vals)
    x_fg = FunctionGraph([a], [x])  # I.e. False

    compare_jax_and_py(x_fg, [get_test_value(i) for i in x_fg.inputs])


def test_jax_CAReduce():
    a_aet = vector("a")
    a_aet.tag.test_value = np.r_[1, 2, 3].astype(config.floatX)

    x = aet_sum(a_aet, axis=None)
    x_fg = FunctionGraph([a_aet], [x])

    compare_jax_and_py(x_fg, [np.r_[1, 2, 3].astype(config.floatX)])

    a_aet = matrix("a")
    a_aet.tag.test_value = np.c_[[1, 2, 3], [1, 2, 3]].astype(config.floatX)

    x = aet_sum(a_aet, axis=0)
    x_fg = FunctionGraph([a_aet], [x])

    compare_jax_and_py(x_fg, [np.c_[[1, 2, 3], [1, 2, 3]].astype(config.floatX)])

    x = aet_sum(a_aet, axis=1)
    x_fg = FunctionGraph([a_aet], [x])

    compare_jax_and_py(x_fg, [np.c_[[1, 2, 3], [1, 2, 3]].astype(config.floatX)])

    a_aet = matrix("a")
    a_aet.tag.test_value = np.c_[[1, 2, 3], [1, 2, 3]].astype(config.floatX)

    x = prod(a_aet, axis=0)
    x_fg = FunctionGraph([a_aet], [x])

    compare_jax_and_py(x_fg, [np.c_[[1, 2, 3], [1, 2, 3]].astype(config.floatX)])

    x = aet_all(a_aet)
    x_fg = FunctionGraph([a_aet], [x])

    compare_jax_and_py(x_fg, [np.c_[[1, 2, 3], [1, 2, 3]].astype(config.floatX)])


def test_jax_MakeVector():
    x = aet.make_vector(1, 2, 3)
    x_fg = FunctionGraph([], [x])

    compare_jax_and_py(x_fg, [])


@pytest.mark.xfail(
    version_parse(jax.__version__) >= version_parse("0.2.12"),
    reason="Omnistaging cannot be disabled",
)
def test_jax_Reshape():
    a = vector("a")
    x = reshape(a, (2, 2))
    x_fg = FunctionGraph([a], [x])
    compare_jax_and_py(x_fg, [np.r_[1.0, 2.0, 3.0, 4.0].astype(config.floatX)])

    # Test breaking "omnistaging" changes in JAX.
    # See https://github.com/tensorflow/probability/commit/782d0c64eb774b9aac54a1c8488e4f1f96fbbc68
    x = reshape(a, (a.shape[0] // 2, a.shape[0] // 2))
    x_fg = FunctionGraph([a], [x])
    compare_jax_and_py(x_fg, [np.r_[1.0, 2.0, 3.0, 4.0].astype(config.floatX)])


@pytest.mark.xfail(reason="jax.numpy.arange requires concrete inputs")
def test_jax_Reshape_nonconcrete():
    a = vector("a")
    b = iscalar("b")
    x = reshape(a, (b, b))
    x_fg = FunctionGraph([a, b], [x])
    compare_jax_and_py(x_fg, [np.r_[1.0, 2.0, 3.0, 4.0].astype(config.floatX), 2])


def test_jax_Dimshuffle():
    a_aet = matrix("a")

    x = a_aet.T
    x_fg = FunctionGraph([a_aet], [x])
    compare_jax_and_py(x_fg, [np.c_[[1.0, 2.0], [3.0, 4.0]].astype(config.floatX)])

    x = a_aet.dimshuffle([0, 1, "x"])
    x_fg = FunctionGraph([a_aet], [x])
    compare_jax_and_py(x_fg, [np.c_[[1.0, 2.0], [3.0, 4.0]].astype(config.floatX)])

    a_aet = tensor(dtype=config.floatX, broadcastable=[False, True])
    x = a_aet.dimshuffle((0,))
    x_fg = FunctionGraph([a_aet], [x])
    compare_jax_and_py(x_fg, [np.c_[[1.0, 2.0, 3.0, 4.0]].astype(config.floatX)])

    a_aet = tensor(dtype=config.floatX, broadcastable=[False, True])
    x = aet_elemwise.DimShuffle([False, True], (0,), inplace=True)(a_aet)
    x_fg = FunctionGraph([a_aet], [x])
    compare_jax_and_py(x_fg, [np.c_[[1.0, 2.0, 3.0, 4.0]].astype(config.floatX)])


@pytest.mark.xfail(
    version_parse(jax.__version__) >= version_parse("0.2.12"),
    reason="Omnistaging cannot be disabled",
)
def test_jax_Join():
    a = matrix("a")
    b = matrix("b")

    x = aet.join(0, a, b)
    x_fg = FunctionGraph([a, b], [x])
    compare_jax_and_py(
        x_fg,
        [
            np.c_[[1.0, 2.0, 3.0]].astype(config.floatX),
            np.c_[[4.0, 5.0, 6.0]].astype(config.floatX),
        ],
    )
    compare_jax_and_py(
        x_fg,
        [
            np.c_[[1.0, 2.0, 3.0]].astype(config.floatX),
            np.c_[[4.0, 5.0]].astype(config.floatX),
        ],
    )

    x = aet.join(1, a, b)
    x_fg = FunctionGraph([a, b], [x])
    compare_jax_and_py(
        x_fg,
        [
            np.c_[[1.0, 2.0, 3.0]].astype(config.floatX),
            np.c_[[4.0, 5.0, 6.0]].astype(config.floatX),
        ],
    )
    compare_jax_and_py(
        x_fg,
        [
            np.c_[[1.0, 2.0], [3.0, 4.0]].astype(config.floatX),
            np.c_[[5.0, 6.0]].astype(config.floatX),
        ],
    )


def test_jax_variadic_Scalar():
    mu = vector("mu", dtype=config.floatX)
    mu.tag.test_value = np.r_[0.1, 1.1].astype(config.floatX)
    tau = vector("tau", dtype=config.floatX)
    tau.tag.test_value = np.r_[1.0, 2.0].astype(config.floatX)

    res = -tau * mu

    fgraph = FunctionGraph([mu, tau], [res])

    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    res = -tau * (tau - mu) ** 2

    fgraph = FunctionGraph([mu, tau], [res])

    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])


def test_jax_logp():

    mu = vector("mu")
    mu.tag.test_value = np.r_[0.0, 0.0].astype(config.floatX)
    tau = vector("tau")
    tau.tag.test_value = np.r_[1.0, 1.0].astype(config.floatX)
    sigma = vector("sigma")
    sigma.tag.test_value = (1.0 / get_test_value(tau)).astype(config.floatX)
    value = vector("value")
    value.tag.test_value = np.r_[0.1, -10].astype(config.floatX)

    logp = (-tau * (value - mu) ** 2 + log(tau / np.pi / 2.0)) / 2.0
    conditions = [sigma > 0]
    alltrue = aet_all([aet_all(1 * val) for val in conditions])
    normal_logp = aet.switch(alltrue, logp, -np.inf)

    fgraph = FunctionGraph([mu, tau, sigma, value], [normal_logp])

    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])


def test_jax_multioutput():
    x = vector("x")
    x.tag.test_value = np.r_[1.0, 2.0].astype(config.floatX)
    y = vector("y")
    y.tag.test_value = np.r_[3.0, 4.0].astype(config.floatX)

    w = cosh(x ** 2 + y / 3.0)
    v = cosh(x / 3.0 + y ** 2)

    fgraph = FunctionGraph([x, y], [w, v])

    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])


def test_nnet():
    x = vector("x")
    x.tag.test_value = np.r_[1.0, 2.0].astype(config.floatX)

    out = sigmoid(x)
    fgraph = FunctionGraph([x], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    out = aet_nnet.ultra_fast_sigmoid(x)
    fgraph = FunctionGraph([x], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    out = softplus(x)
    fgraph = FunctionGraph([x], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    out = aet_nnet.softmax(x)
    fgraph = FunctionGraph([x], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    out = aet_nnet.logsoftmax(x)
    fgraph = FunctionGraph([x], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])


@pytest.mark.xfail(
    version_parse(jax.__version__) >= version_parse("0.2.12"),
    reason="Omnistaging cannot be disabled",
)
def test_tensor_basics():
    y = vector("y")
    y.tag.test_value = np.r_[1.0, 2.0].astype(config.floatX)
    x = vector("x")
    x.tag.test_value = np.r_[3.0, 4.0].astype(config.floatX)
    A = matrix("A")
    A.tag.test_value = np.empty((2, 2), dtype=config.floatX)
    alpha = scalar("alpha")
    alpha.tag.test_value = np.array(3.0, dtype=config.floatX)
    beta = scalar("beta")
    beta.tag.test_value = np.array(5.0, dtype=config.floatX)

    # This should be converted into a `Gemv` `Op` when the non-JAX compatible
    # optimizations are turned on; however, when using JAX mode, it should
    # leave the expression alone.
    out = y.dot(alpha * A).dot(x) + beta * y
    fgraph = FunctionGraph([y, x, A, alpha, beta], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    out = maximum(y, x)
    fgraph = FunctionGraph([y, x], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    out = aet_max(y)
    fgraph = FunctionGraph([y], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])


@pytest.mark.xfail(reason="jax.numpy.arange requires concrete inputs")
def test_arange_nonconcrete():

    a = scalar("a")
    a.tag.test_value = 10

    out = aet.arange(a)
    fgraph = FunctionGraph([a], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])


@pytest.mark.xfail(reason="jax.numpy.arange requires concrete inputs")
def test_unique_nonconcrete():
    a = matrix("a")
    a.tag.test_value = np.arange(6, dtype=config.floatX).reshape((3, 2))

    out = aet_extra_ops.Unique()(a)
    fgraph = FunctionGraph([a], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])


def test_identity():
    a = scalar("a")
    a.tag.test_value = 10

    out = aes.identity(a)
    fgraph = FunctionGraph([a], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])


def test_second():
    a0 = scalar("a0")
    b = scalar("b")

    out = aes.second(a0, b)
    fgraph = FunctionGraph([a0, b], [out])
    compare_jax_and_py(fgraph, [10.0, 5.0])

    a1 = vector("a1")
    out = aet.second(a1, b)
    fgraph = FunctionGraph([a1, b], [out])
    compare_jax_and_py(fgraph, [np.zeros([5], dtype=config.floatX), 5.0])


def test_jax_BatchedDot():
    # tensor3 . tensor3
    a = tensor3("a")
    a.tag.test_value = (
        np.linspace(-1, 1, 10 * 5 * 3).astype(config.floatX).reshape((10, 5, 3))
    )
    b = tensor3("b")
    b.tag.test_value = (
        np.linspace(1, -1, 10 * 3 * 2).astype(config.floatX).reshape((10, 3, 2))
    )
    out = aet_blas.BatchedDot()(a, b)
    fgraph = FunctionGraph([a, b], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    # A dimension mismatch should raise a TypeError for compatibility
    inputs = [get_test_value(a)[:-1], get_test_value(b)]
    opts = Query(include=[None], exclude=["cxx_only", "BlasOpt"])
    jax_mode = Mode(JAXLinker(), opts)
    aesara_jax_fn = function(fgraph.inputs, fgraph.outputs, mode=jax_mode)
    with pytest.raises(TypeError):
        aesara_jax_fn(*inputs)

    # matrix . matrix
    a = matrix("a")
    a.tag.test_value = np.linspace(-1, 1, 5 * 3).astype(config.floatX).reshape((5, 3))
    b = matrix("b")
    b.tag.test_value = np.linspace(1, -1, 5 * 3).astype(config.floatX).reshape((5, 3))
    out = aet_blas.BatchedDot()(a, b)
    fgraph = FunctionGraph([a, b], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])


def test_shared():
    a = shared(np.array([1, 2, 3], dtype=config.floatX))

    aesara_jax_fn = function([], a, mode="JAX")
    jax_res = aesara_jax_fn()

    assert isinstance(jax_res, jax.interpreters.xla.DeviceArray)
    np.testing.assert_allclose(jax_res, a.get_value())

    aesara_jax_fn = function([], a * 2, mode="JAX")
    jax_res = aesara_jax_fn()

    assert isinstance(jax_res, jax.interpreters.xla.DeviceArray)
    np.testing.assert_allclose(jax_res, a.get_value() * 2)

    # Changed the shared value and make sure that the JAX-compiled
    # function also changes.
    new_a_value = np.array([3, 4, 5], dtype=config.floatX)
    a.set_value(new_a_value)

    jax_res = aesara_jax_fn()
    assert isinstance(jax_res, jax.interpreters.xla.DeviceArray)
    np.testing.assert_allclose(jax_res, new_a_value * 2)


def test_extra_ops():
    a = matrix("a")
    a.tag.test_value = np.arange(6, dtype=config.floatX).reshape((3, 2))

    out = aet_extra_ops.cumsum(a, axis=0)
    fgraph = FunctionGraph([a], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    out = aet_extra_ops.cumprod(a, axis=1)
    fgraph = FunctionGraph([a], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    out = aet_extra_ops.diff(a, n=2, axis=1)
    fgraph = FunctionGraph([a], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    out = aet_extra_ops.repeat(a, (3, 3), axis=1)
    fgraph = FunctionGraph([a], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    c = aet.as_tensor(5)

    with pytest.raises(NotImplementedError):
        out = aet_extra_ops.fill_diagonal(a, c)
        fgraph = FunctionGraph([a], [out])
        compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    with pytest.raises(NotImplementedError):
        out = aet_extra_ops.fill_diagonal_offset(a, c, c)
        fgraph = FunctionGraph([a], [out])
        compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    with pytest.raises(NotImplementedError):
        out = aet_extra_ops.Unique(axis=1)(a)
        fgraph = FunctionGraph([a], [out])
        compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    indices = np.arange(np.product((3, 4)))
    out = aet_extra_ops.unravel_index(indices, (3, 4), order="C")
    fgraph = FunctionGraph([], out)
    compare_jax_and_py(
        fgraph, [get_test_value(i) for i in fgraph.inputs], must_be_device_array=False
    )


@pytest.mark.xfail(
    version_parse(jax.__version__) >= version_parse("0.2.12"),
    reason="Omnistaging cannot be disabled",
)
def test_extra_ops_omni():
    a = matrix("a")
    a.tag.test_value = np.arange(6, dtype=config.floatX).reshape((3, 2))

    # This function also cannot take symbolic input.
    c = aet.as_tensor(5)
    out = aet_extra_ops.bartlett(c)
    fgraph = FunctionGraph([], [out])
    compare_jax_and_py(fgraph, [get_test_value(i) for i in fgraph.inputs])

    multi_index = np.unravel_index(np.arange(np.product((3, 4))), (3, 4))
    out = aet_extra_ops.ravel_multi_index(multi_index, (3, 4))
    fgraph = FunctionGraph([], [out])
    compare_jax_and_py(
        fgraph, [get_test_value(i) for i in fgraph.inputs], must_be_device_array=False
    )

    # The inputs are "concrete", yet it still has problems?
    out = aet_extra_ops.Unique()(
        aet.as_tensor(np.arange(6, dtype=config.floatX).reshape((3, 2)))
    )
    fgraph = FunctionGraph([], [out])
    compare_jax_and_py(fgraph, [])


@pytest.mark.xfail(reason="The RNG states are not 1:1", raises=AssertionError)
def test_random():
    rng = shared(np.random.RandomState(123))
    out = normal(rng=rng)
    fgraph = FunctionGraph([out.owner.inputs[0]], [out], clone=False)
    compare_jax_and_py(fgraph, [])


def test_random_unimplemented():
    class NonExistentRV(RandomVariable):
        name = "non-existent"
        ndim_supp = 0
        ndims_params = []
        dtype = "floatX"

        def __call__(self, size=None, **kwargs):
            return super().__call__(size=size, **kwargs)

        def rng_fn(cls, rng, size):
            return 0

    nonexistentrv = NonExistentRV()
    rng = shared(np.random.RandomState(123))
    out = nonexistentrv(rng=rng)
    fgraph = FunctionGraph([out.owner.inputs[0]], [out], clone=False)

    with pytest.raises(NotImplementedError):
        compare_jax_and_py(fgraph, [])


def test_RandomStream():
    srng = RandomStream(seed=123)
    out = srng.normal() - srng.normal()

    fn = function([], out, mode=jax_mode)
    jax_res_1 = fn()
    jax_res_2 = fn()

    assert np.array_equal(jax_res_1, jax_res_2)
