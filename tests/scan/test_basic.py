"""
  Questions and notes about scan that should be answered :

   * Scan seems to do copies of every input variable. Is that needed?
   answer : probably not, but it doesn't hurt also ( what we copy is
   aesara variables, which just cary information about the type / dimension
   of the data)

   * There is some of scan functionality that is not well documented
"""

import os
import pickle
import shutil
import sys
import time
from collections import OrderedDict
from tempfile import mkdtemp

import numpy as np
import pytest

from aesara.assert_op import assert_op
from aesara.compile.debugmode import DebugMode
from aesara.compile.function import function
from aesara.compile.function.pfunc import rebuild_collect_shared
from aesara.compile.io import In
from aesara.compile.mode import FAST_RUN, Mode, get_default_mode, get_mode
from aesara.compile.monitormode import MonitorMode
from aesara.compile.sharedvalue import shared
from aesara.configdefaults import config
from aesara.gradient import (
    NullTypeGradError,
    Rop,
    disconnected_grad,
    grad,
    hessian,
    jacobian,
)
from aesara.graph.basic import clone_replace, graph_inputs
from aesara.graph.fg import MissingInputError
from aesara.misc.safe_asarray import _asarray
from aesara.scan.basic import scan
from aesara.scan.op import Scan
from aesara.scan.opt import ScanMerge
from aesara.scan.utils import until
from aesara.scan.views import foldl, foldr
from aesara.scan.views import map as aet_map
from aesara.scan.views import reduce as aet_reduce
from aesara.tensor import basic as aet
from aesara.tensor.blas import Dot22
from aesara.tensor.elemwise import Elemwise
from aesara.tensor.math import Dot
from aesara.tensor.math import all as aet_all
from aesara.tensor.math import dot, mean, sigmoid
from aesara.tensor.math import sum as aet_sum
from aesara.tensor.math import tanh
from aesara.tensor.nnet import categorical_crossentropy, softmax_graph
from aesara.tensor.random.utils import RandomStream
from aesara.tensor.shape import Shape_i, reshape, shape, specify_shape
from aesara.tensor.sharedvar import SharedVariable
from aesara.tensor.subtensor import Subtensor, inc_subtensor
from aesara.tensor.type import (
    dcol,
    dmatrix,
    dscalar,
    dvector,
    fmatrix,
    fscalar,
    ftensor3,
    fvector,
    imatrix,
    iscalar,
    ivector,
    lscalar,
    lvector,
    matrix,
    scalar,
    tensor3,
    vector,
)
from tests import unittest_tools as utt


if config.mode == "FAST_COMPILE":
    mode_with_opt = get_mode("FAST_RUN")
else:
    mode_with_opt = get_default_mode()
if config.mode in ("DEBUG_MODE", "DebugMode"):
    mode_nodebug = get_mode("FAST_RUN")
else:
    mode_nodebug = mode_with_opt


type_eps = {"float64": 1e-7, "float32": 3e-3}


class multiple_outputs_numeric_grad:
    """WRITEME"""

    def __init__(self, f, pt, ndarray_mask=None, eps=None):
        """
        Return the gradient of f at pt.

        This function computes the gradient by a one-sided finite differences
        of a fixed step size (eps).

        It is assumed that f(...) will return a scalar.
        :param eps: the stepsize for the finite differencing. None means
        input dtype-dependent. See `type_eps`.
        """

        def prod(inputs):
            rval = 1
            for i in inputs:
                rval *= i
            return rval

        if not isinstance(pt, (list, tuple)):
            pt = [pt]

        # This mask tells us if we are dealing with an ndarray input or
        # something else ( a random state ? ) with which we shouldn't really
        # mess up
        if not ndarray_mask:
            ndarray_mask = [True for x in pt]

        dtype_eps = type_eps["float64"]

        for i, p in enumerate(pt):
            if ndarray_mask[i]:
                pt[i] = np.array(p)
                _eps = type_eps[str(pt[i].dtype)]
                if _eps > dtype_eps:
                    dtype_eps = _eps

        self.ndarray_mask = ndarray_mask
        # '''
        # Compute clean output:
        f_x = f(*pt)
        gx = []
        # now iterate over the elements of x and call f on those + delta x
        for i in range(len(pt)):
            if ndarray_mask[i]:
                # It is a ndarray that we can tweak
                if eps:
                    _eps = eps
                else:
                    _eps = dtype_eps
                if pt[i].ndim:
                    _g = []
                    # it has several dimensions:
                    for pos in range(prod(pt[i].shape)):
                        t = pt[i].copy()
                        t = t.flatten()
                        t[pos] += _eps
                        t = t.reshape(pt[i].shape)
                        f_eps = f(*(pt[:i] + [t] + pt[i + 1 :]))
                        _g.append(np.asarray((f_eps - f_x) / _eps))
                    gx.append(np.asarray(_g).reshape(pt[i].shape))
                else:
                    t = np.array(pt[i] + _eps)
                    f_eps = f(*(pt[:i] + [t] + pt[i + 1 :]))
                    gx.append(np.asarray((f_eps - f_x) / _eps))
        self.gx = gx

    @staticmethod
    def abs_rel_err(a, b, eps=1.0e-10):
        """
        Return a small number when a and b are close, relative to how big they are
        """
        return abs(a - b) / (abs(a) + abs(b) + eps)

    def max_err(self, _g_pt):
        """
        Return the biggest relative error between g_pt and self.gx
        """
        g_pt = []
        for i in range(len(_g_pt)):
            if self.ndarray_mask[i]:
                g_pt.append(_g_pt[i])
            elif isinstance(_g_pt[i], np.ndarray):
                assert np.all(_g_pt[i] == 0)
        if len(g_pt) != len(self.gx):
            raise ValueError("argument has wrong number of elements", len(g_pt))
        errs = []

        for i, (a, b) in enumerate(zip(g_pt, self.gx)):
            if a.shape != b.shape:
                raise ValueError(
                    "argument element %i has wrong shape %s"
                    % (i, str((a.shape, b.shape)))
                )
            errs.append(np.max(multiple_outputs_numeric_grad.abs_rel_err(a, b)))
        if np.all(np.isfinite(errs)):
            return np.max(errs), np.argmax(errs)
        else:
            return np.inf, 0


# TODO: Test this function, and if it works,
# use it with the normal verify_grad rather than the
# copy-and-pasted one above.
# Also - add a reference to this technique in the
# verify_grad method so that other ops with multiple outputs can be tested.
# DONE - rp
def scan_project_sum(*args, **kwargs):
    rng = RandomStream(123)
    scan_outputs, updates = scan(*args, **kwargs)
    if type(scan_outputs) not in [list, tuple]:
        scan_outputs = [scan_outputs]
    # we should ignore the random-state updates so that
    # the uniform numbers are the same every evaluation and on every call
    rng.add_default_updates = False
    factors = [rng.uniform(0.1, 0.9, size=s.shape) for s in scan_outputs]
    # Random values (?)
    return (sum([(s * f).sum() for s, f in zip(scan_outputs, factors)]), updates)


def asarrayX(value):
    return _asarray(value, dtype=config.floatX)


def clone_optimized_graph(f):
    maker_ins = [x for x in f.maker.fgraph.inputs if not isinstance(x, SharedVariable)]
    inps, outs, _ = rebuild_collect_shared(
        f.maker.fgraph.outputs, maker_ins, copy_inputs_over=False
    )
    ins = [x for x in inps if not isinstance(x, SharedVariable)]
    return (ins, outs)


def grab_scan_node(output):
    if output.owner is None:
        return None
    if output.owner.op.__class__.__name__ == "Scan":
        return [output.owner]
    rval = []
    for i in output.owner.inputs:
        ri = grab_scan_node(i)
        if ri is not None:
            rval += ri
    if rval is []:
        return None
    else:
        return rval


def scan_nodes_from_fct(fct):
    nodes = fct.maker.fgraph.toposort()
    scan_nodes = [n for n in nodes if isinstance(n.op, Scan)]
    return scan_nodes


class TestScan:
    def setup_method(self):
        utt.seed_rng()

    # generator network, only one output , type scalar ; no sequence or
    # non sequence arguments
    @pytest.mark.skipif(
        isinstance(get_default_mode(), DebugMode),
        reason="This test fails in DebugMode, because it is not yet picklable.",
    )
    def test_pickling(self):
        def f_pow2(x_tm1):
            return 2 * x_tm1

        state = scalar("state")
        n_steps = iscalar("nsteps")
        output, updates = scan(
            f_pow2,
            [],
            state,
            [],
            n_steps=n_steps,
            truncate_gradient=-1,
            go_backwards=False,
        )
        _my_f = function(
            [state, n_steps], output, updates=updates, allow_input_downcast=True
        )

        # TESTING PICKLE-ing this function
        origdir = os.getcwd()
        tmpdir = None
        try:
            tmpdir = mkdtemp()
            os.chdir(tmpdir)

            with open("tmp_scan_test_pickle.pkl", "wb") as f_out:
                pickle.dump(_my_f, f_out, protocol=-1)
            with open("tmp_scan_test_pickle.pkl", "rb") as f_in:
                my_f = pickle.load(f_in)
        finally:
            # Get back to the original dir, and delete the temporary one.
            os.chdir(origdir)
            if tmpdir is not None:
                shutil.rmtree(tmpdir)

        rng = np.random.RandomState(utt.fetch_seed())
        state = rng.uniform()
        steps = 5

        numpy_values = np.array([state * (2 ** (k + 1)) for k in range(steps)])
        aesara_values = my_f(state, steps)
        utt.assert_allclose(numpy_values, aesara_values)

    # Test that the inner input_storage and output_storage are
    # properly cleared
    def test_inner_storage_leak(self):
        def f_pow2(x_tm1):
            return 2 * x_tm1

        state = scalar("state")
        n_steps = iscalar("nsteps")
        output, updates = scan(f_pow2, [], state, [], n_steps=n_steps)

        f = function(
            [state, n_steps], output, updates=updates, allow_input_downcast=True
        )

        scan_node = [
            node for node in f.maker.fgraph.toposort() if isinstance(node.op, Scan)
        ]

        assert len(scan_node) == 1
        scan_node = scan_node[0]

        # Make sure they start out as None
        assert all(i.value is None for i in scan_node.op.fn.input_storage)
        assert all(o.value is None for o in scan_node.op.fn.output_storage)

        rng = np.random.RandomState(utt.fetch_seed())
        state = rng.uniform()
        steps = 5

        f(state, steps)

        # And that they stay that way
        assert all(i.value is None for i in scan_node.op.fn.input_storage)
        assert all(o.value is None for o in scan_node.op.fn.output_storage)

    # generator network, only one output , type scalar ; no sequence or
    # non sequence arguments
    def test_generator_one_output_scalar(self):
        def f_pow2(x_tm1):
            return 2 * x_tm1

        state = scalar("state")
        n_steps = iscalar("nsteps")
        # Test return_list at the same time.
        output, updates = scan(
            f_pow2,
            [],
            state,
            [],
            n_steps=n_steps,
            truncate_gradient=-1,
            return_list=True,
            go_backwards=False,
        )
        my_f = function(
            [state, n_steps], output, updates=updates, allow_input_downcast=True
        )

        rng = np.random.RandomState(utt.fetch_seed())
        state = rng.uniform()
        steps = 5

        numpy_values = np.array([state * (2 ** (k + 1)) for k in range(steps)])
        aesara_values = my_f(state, steps)
        utt.assert_allclose(numpy_values, aesara_values[0])

    def test_subtensor_multiple_slices(self):
        # This addresses a bug reported by Matthias Zoehrer
        # the bug happens when you have multiple subtensors on the output of
        # scan (the bug requires the reshape to be produced, and it has
        # which has something to do with how the subtensors overlap
        def f_pow2(x_tm1):
            return 2 * x_tm1

        state = vector("state")
        n_steps = iscalar("nsteps")
        output, updates = scan(
            f_pow2,
            [],
            state,
            [],
            n_steps=n_steps,
            truncate_gradient=-1,
            go_backwards=False,
        )
        nw_shape = ivector("nw_shape")
        # Note that the output is reshaped to 3 dimensional tensor, and
        my_f = function(
            [state, n_steps, nw_shape],
            [reshape(output, nw_shape, ndim=3)[:-2], output[:-4]],
            updates=updates,
            allow_input_downcast=True,
        )
        nodes = [x for x in my_f.maker.fgraph.toposort() if isinstance(x.op, Scan)]
        # This assertion fails if savemem optimization failed on scan
        if config.mode != "FAST_COMPILE":
            assert nodes[0].op._scan_savemem_visited
        rng = np.random.RandomState(utt.fetch_seed())
        my_f(rng.uniform(size=(3,)), 4, np.int64([2, 2, 3]))

    @pytest.mark.slow
    def test_only_nonseq_inputs(self):
        # Compile the Aesara function
        n_steps = 2
        inp = matrix()
        broadcasted_inp, _ = scan(lambda x: x, non_sequences=[inp], n_steps=n_steps)
        out = broadcasted_inp.sum()
        gr = grad(out, inp)
        fun = function([inp], [broadcasted_inp, gr])

        # Execute the Aesara function and compare outputs to the expected outputs
        inputs = np.array([[1, 2], [3, 4]], dtype=config.floatX)
        expected_out1 = np.repeat(inputs[None], n_steps, axis=0)
        expected_out2 = np.ones(inputs.shape, dtype="int8") * n_steps

        out1, out2 = fun(inputs)
        utt.assert_allclose(out1, expected_out1)
        utt.assert_allclose(out2, expected_out2)

    # simple rnn, one input, one state, weights for each; input/state
    # are vectors, weights are scalars
    def test_one_sequence_one_output_weights(self):
        def f_rnn(u_t, x_tm1, W_in, W):
            return u_t * W_in + x_tm1 * W

        u = vector("u")
        x0 = scalar("x0")
        W_in = scalar("win")
        W = scalar("w")

        output, updates = scan(
            f_rnn,
            u,
            x0,
            [W_in, W],
            n_steps=None,
            truncate_gradient=-1,
            go_backwards=False,
        )

        f2 = function(
            [u, x0, W_in, W], output, updates=updates, allow_input_downcast=True
        )
        # get random initial values
        rng = np.random.RandomState(utt.fetch_seed())
        v_u = rng.uniform(-5.0, 5.0, size=(4,))
        v_x0 = rng.uniform()
        W = rng.uniform()
        W_in = rng.uniform()

        # compute the output in numpy
        v_out = np.zeros((4,))
        v_out[0] = v_u[0] * W_in + v_x0 * W
        for step in range(1, 4):
            v_out[step] = v_u[step] * W_in + v_out[step - 1] * W
        aesara_values = f2(v_u, v_x0, W_in, W)
        utt.assert_allclose(aesara_values, v_out)

    # simple rnn, one input, one state, weights for each; input/state
    # are vectors, weights are scalars; using shared variables
    def test_one_sequence_one_output_weights_shared(self):
        rng = np.random.RandomState(utt.fetch_seed())
        u = vector("u")
        x0 = scalar("x0")
        W_in = shared(asarrayX(rng.uniform()), name="w_in")
        W = shared(asarrayX(rng.uniform()), name="w")

        def f_rnn_shared(u_t, x_tm1, tmp_W_in, tmp_W):
            return u_t * tmp_W_in + x_tm1 * tmp_W

        output, updates = scan(
            f_rnn_shared,
            u,
            x0,
            [W_in, W],
            n_steps=None,
            truncate_gradient=-1,
            go_backwards=False,
        )
        f3 = function([u, x0], output, updates=updates, allow_input_downcast=True)
        # get random initial values

        v_u = rng.uniform(-5.0, 5.0, size=(4,))
        v_x0 = rng.uniform()
        # compute the output i numpy
        v_out = np.zeros((4,))
        v_out[0] = v_u[0] * W_in.get_value() + v_x0 * W.get_value()
        for step in range(1, 4):
            v_out[step] = v_u[step] * W_in.get_value() + v_out[step - 1] * W.get_value()

        aesara_values = f3(v_u, v_x0)
        assert np.allclose(aesara_values, v_out)

    # some rnn with multiple outputs and multiple inputs; other
    # dimension instead of scalars/vectors
    def test_multiple_inputs_multiple_outputs(self):
        rng = np.random.RandomState(utt.fetch_seed())
        vW_in2 = asarrayX(rng.uniform(-5.0, 5.0, size=(2,)))
        vW = asarrayX(rng.uniform(-5.0, 5.0, size=(2, 2)))
        vWout = asarrayX(rng.uniform(-5.0, 5.0, size=(2,)))
        vW_in1 = asarrayX(rng.uniform(-5.0, 5.0, size=(2, 2)))
        v_u1 = asarrayX(rng.uniform(-5.0, 5.0, size=(3, 2)))
        v_u2 = asarrayX(rng.uniform(-5.0, 5.0, size=(3,)))
        v_x0 = asarrayX(rng.uniform(-5.0, 5.0, size=(2,)))
        v_y0 = asarrayX(rng.uniform())

        W_in2 = shared(vW_in2, name="win2")
        W = shared(vW, name="w")
        W_out = shared(vWout, name="wout")
        W_in1 = matrix("win")
        u1 = matrix("u1")
        u2 = vector("u2")
        x0 = vector("x0")
        y0 = scalar("y0")

        def f_rnn_cmpl(u1_t, u2_t, x_tm1, y_tm1, W_in1):
            return [
                dot(u1_t, W_in1) + u2_t * W_in2 + dot(x_tm1, W),
                dot(x_tm1, W_out),
            ]

        outputs, updates = scan(
            f_rnn_cmpl,
            [u1, u2],
            [x0, y0],
            W_in1,
            n_steps=None,
            truncate_gradient=-1,
            go_backwards=False,
        )

        f4 = function(
            [u1, u2, x0, y0, W_in1], outputs, updates=updates, allow_input_downcast=True
        )

        # compute the values in numpy
        v_x = np.zeros((3, 2), dtype=config.floatX)
        v_y = np.zeros((3,), dtype=config.floatX)
        v_x[0] = np.dot(v_u1[0], vW_in1) + v_u2[0] * vW_in2 + np.dot(v_x0, vW)
        v_y[0] = np.dot(v_x0, vWout)
        for i in range(1, 3):
            v_x[i] = np.dot(v_u1[i], vW_in1) + v_u2[i] * vW_in2 + np.dot(v_x[i - 1], vW)
            v_y[i] = np.dot(v_x[i - 1], vWout)

        (aesara_x, aesara_y) = f4(v_u1, v_u2, v_x0, v_y0, vW_in1)
        utt.assert_allclose(aesara_x, v_x)
        utt.assert_allclose(aesara_y, v_y)

    def test_multiple_outs_taps(self):
        l = 5
        rng = np.random.RandomState(utt.fetch_seed())

        vW_in2 = asarrayX(rng.uniform(-2.0, 2.0, size=(2,)))
        vW = asarrayX(rng.uniform(-2.0, 2.0, size=(2, 2)))
        vWout = asarrayX(rng.uniform(-2.0, 2.0, size=(2,)))
        vW_in1 = asarrayX(rng.uniform(-2.0, 2.0, size=(2, 2)))
        v_u1 = asarrayX(rng.uniform(-2.0, 2.0, size=(l, 2)))
        v_u2 = asarrayX(rng.uniform(-2.0, 2.0, size=(l + 2, 2)))
        v_x0 = asarrayX(rng.uniform(-2.0, 2.0, size=(2,)))
        v_y0 = asarrayX(rng.uniform(size=(3,)))

        W_in2 = shared(vW_in2, name="win2")
        W = shared(vW, name="w")
        W_out = shared(vWout, name="wout")
        W_in1 = matrix("win")
        u1 = matrix("u1")
        u2 = matrix("u2")
        x0 = vector("x0")
        y0 = vector("y0")

        def f_rnn_cmpl(u1_t, u2_tm1, u2_t, u2_tp1, x_tm1, y_tm1, y_tm3, W_in1):
            return [
                dot(u1_t, W_in1) + (u2_t + u2_tm1 * u2_tp1) * W_in2 + dot(x_tm1, W),
                (y_tm1 + y_tm3) * dot(x_tm1, W_out),
                dot(u1_t, W_in1),
            ]

        outputs, updates = scan(
            f_rnn_cmpl,
            [u1, dict(input=u2, taps=[-1, 0, 1])],
            [x0, dict(initial=y0, taps=[-1, -3]), None],
            W_in1,
            n_steps=None,
            truncate_gradient=-1,
            go_backwards=False,
        )

        f = function(
            [u1, u2, x0, y0, W_in1], outputs, updates=updates, allow_input_downcast=True
        )

        f(v_u1, v_u2, v_x0, v_y0, vW_in1)

        ny0 = np.zeros((5, 2))
        ny1 = np.zeros((5,))
        ny2 = np.zeros((5, 2))
        ny0[0] = (
            np.dot(v_u1[0], vW_in1)
            + (v_u2[1] + v_u2[0] * v_u2[2]) * vW_in2
            + np.dot(v_x0, vW)
        )

        ny1[0] = (v_y0[2] + v_y0[0]) * np.dot(v_x0, vWout)
        ny2[0] = np.dot(v_u1[0], vW_in1)

        ny0[1] = (
            np.dot(v_u1[1], vW_in1)
            + (v_u2[2] + v_u2[1] * v_u2[3]) * vW_in2
            + np.dot(ny0[0], vW)
        )

        ny1[1] = (ny1[0] + v_y0[1]) * np.dot(ny0[0], vWout)
        ny2[1] = np.dot(v_u1[1], vW_in1)

        ny0[2] = (
            np.dot(v_u1[2], vW_in1)
            + (v_u2[3] + v_u2[2] * v_u2[4]) * vW_in2
            + np.dot(ny0[1], vW)
        )
        ny1[2] = (ny1[1] + v_y0[2]) * np.dot(ny0[1], vWout)
        ny2[2] = np.dot(v_u1[2], vW_in1)

        ny0[3] = (
            np.dot(v_u1[3], vW_in1)
            + (v_u2[4] + v_u2[3] * v_u2[5]) * vW_in2
            + np.dot(ny0[2], vW)
        )

        ny1[3] = (ny1[2] + ny1[0]) * np.dot(ny0[2], vWout)
        ny2[3] = np.dot(v_u1[3], vW_in1)

        ny0[4] = (
            np.dot(v_u1[4], vW_in1)
            + (v_u2[5] + v_u2[4] * v_u2[6]) * vW_in2
            + np.dot(ny0[3], vW)
        )

        ny1[4] = (ny1[3] + ny1[1]) * np.dot(ny0[3], vWout)
        ny2[4] = np.dot(v_u1[4], vW_in1)

    def test_using_taps_sequence(self):
        # this test refers to a bug reported by Nicolas
        # Boulanger-Lewandowski June 6th
        x = dvector()
        y, updates = scan(
            lambda x: [x], sequences=dict(input=x, taps=[-1]), outputs_info=[None]
        )
        inp = np.arange(5).astype("float64")
        rval = function([x], y, updates=updates)(inp)
        assert np.all(rval == inp[:-1])

    def test_using_negative_taps_sequence(self):
        # This test refers to a bug reported on github on May 22 2015 by
        # user june-qijun
        def lp(x, x2):
            return x

        x = fvector("x")
        res, upd = scan(lp, sequences=dict(input=x, taps=[-2, -1]))
        f = function([x], res, updates=upd)

        output = f([1, 2, 3, 4, 5])
        expected_output = np.array([1, 2, 3], dtype="float32")
        utt.assert_allclose(output, expected_output)

    def test_connection_pattern(self):
        # Test connection_pattern() in the presence of recurrent outputs
        # with multiple taps.
        #
        # This test refers to a bug signaled on the theano-users mailing list
        # on March 10 2015 by David Schneider-Joseph.

        def fn(a_m2, a_m1, b_m2, b_m1):
            return a_m1, b_m1

        a0 = shared(np.arange(2))
        b0 = shared(np.arange(2))

        (a, b), _ = scan(
            fn,
            outputs_info=[
                {"initial": a0, "taps": [-2, -1]},
                {"initial": b0, "taps": [-2, -1]},
            ],
            n_steps=2,
        )

        grad(a[-1], a0)

        # Also validate that the mappings outer_inp_from_outer_out and
        # outer_inp_from_inner_inp produce the correct results
        scan_node = a.owner.inputs[0].owner

        result = scan_node.op.var_mappings["outer_inp_from_outer_out"]
        expected_result = {0: 1, 1: 2}
        assert result == expected_result

        result = scan_node.op.var_mappings["outer_inp_from_inner_inp"]
        expected_result = {0: 1, 1: 1, 2: 2, 3: 2}
        assert result == expected_result

    def test_connection_pattern2(self):
        # This tests for a crash in connection_pattern() when a scan node
        # has more than one mitmot (multiple input taps as well as
        # multiple output taps) output

        x = matrix()
        seq = vector()

        def inner_fct(seq, state_old, state_current):
            state_next = state_old * 2 + state_current + seq
            return state_next

        out, _ = scan(
            inner_fct, sequences=seq, outputs_info={"initial": x, "taps": [-2, -1]}
        )

        g_out = grad(out.sum(), [seq, x])

        scan_node = g_out[0].owner.inputs[1].owner.inputs[1].owner.inputs[0].owner
        scan_node.op.connection_pattern(scan_node)

        # Also validate that the mappings outer_inp_from_outer_out and
        # outer_inp_from_inner_inp produce the correct results
        scan_node = out.owner.inputs[0].owner

        result = scan_node.op.var_mappings["outer_inp_from_outer_out"]
        expected_result = {0: 2}
        assert result == expected_result

        result = scan_node.op.var_mappings["outer_inp_from_inner_inp"]
        expected_result = {0: 1, 1: 2, 2: 2}
        assert result == expected_result

    def test_grad_grad_mitsot_sitsot(self):
        # Test for an index error when taking the second derivative
        # through a Scan node with one sitsot and one mitsot.

        def inner_fct(mitsot_m2, mitsot_m1, sitsot):
            total = mitsot_m2 + mitsot_m1 + sitsot
            output = total ** 1.05
            return output, output

        inputs = [matrix(), vector()]
        outputs_info = [dict(initial=inputs[0], taps=[-2, -1]), inputs[1]]

        scan_outputs, updates = scan(fn=inner_fct, outputs_info=outputs_info, n_steps=5)

        # Take the gradient of each output wrt its corresponding initial state
        gradients = [
            grad(scan_outputs[0].sum(), inputs[0]),
            grad(scan_outputs[1].sum(), inputs[1]),
        ]

        # Take the gradient of the sum of gradients wrt the inputs
        sum_of_grads = sum([g.sum() for g in gradients])
        grad(sum_of_grads, inputs[0])

    def test_verify_second_grad_sitsot(self):
        def get_sum_of_grad(inp):

            scan_outputs, updates = scan(
                fn=lambda x: x * 2, outputs_info=[inp], n_steps=5
            )

            # Take the gradient of each output wrt its corresponding initial
            # state
            return grad(scan_outputs.sum(), inp).sum()

        # Call verify_grad to ensure the correctness of the second gradients
        floatX = config.floatX
        inputs_test_values = [np.random.random(3).astype(floatX)]
        utt.verify_grad(get_sum_of_grad, inputs_test_values)

    def test_verify_second_grad_mitsot1(self):
        def inner_fct(mitsot_m2, sitsot):
            total = mitsot_m2 + sitsot
            output = total ** 1.02
            return output, output

        def get_sum_of_grad(input0, input1):
            outputs_info = [dict(initial=input0, taps=[-2]), input1]

            scan_outputs, updates = scan(
                fn=inner_fct, outputs_info=outputs_info, n_steps=3
            )

            # Take the gradient of each output wrt its corresponding initial
            # state
            gradients = [
                grad(scan_outputs[0].sum(), input0),
                grad(scan_outputs[1].sum(), input1),
            ]

            return gradients[0].sum() + gradients[1].sum()

        # Call verify_grad to ensure the correctness of the second gradients
        floatX = config.floatX
        inputs_test_values = [
            np.random.random((2, 3)).astype(floatX),
            np.random.random(3).astype(floatX),
        ]
        utt.verify_grad(get_sum_of_grad, inputs_test_values)

    def test_grad_two_scans(self):

        # data input & output
        x = tensor3("x")
        t = imatrix("t")

        # forward pass
        W = shared(np.random.randn(2, 2).astype("float32"), name="W", borrow=True)

        def forward_scanner(x_t):
            a2_t = dot(x_t, W)
            y_t = softmax_graph(a2_t)
            return y_t

        y, _ = scan(fn=forward_scanner, sequences=x, outputs_info=[None])

        # loss function
        def error_scanner(y_t, t_t):
            return mean(categorical_crossentropy(y_t, t_t))

        L, _ = scan(fn=error_scanner, sequences=[y, t], outputs_info=[None])
        L = mean(L)

        # backward pass
        grad(L, [W])

    # simple rnn, one input, one state, weights for each; input/state are
    # vectors, weights are scalars; using shared variables and past
    # taps (sequences and outputs)
    def test_using_taps_input_output(self):
        rng = np.random.RandomState(utt.fetch_seed())
        vW = asarrayX(rng.uniform())
        vW_in = asarrayX(rng.uniform())
        vu = asarrayX(rng.uniform(-5.0, 5.0, size=(4,)))
        vx0 = asarrayX(rng.uniform(-5.0, 5.0, size=(2,)))

        u = vector("u")
        x0 = vector("x0")
        W_in = shared(vW_in, name="w_in")
        W = shared(vW, name="w")

        def f_rnn_shared(u_tm2, x_tm1, x_tm2):
            return u_tm2 * W_in + x_tm1 * W + x_tm2

        outputs, updates = scan(
            f_rnn_shared,
            dict(input=u, taps=-2),
            dict(initial=x0, taps=[-1, -2]),
            [],
            n_steps=None,
            truncate_gradient=-1,
            go_backwards=False,
        )

        f7 = function([u, x0], outputs, updates=updates, allow_input_downcast=True)
        aesara_out = f7(vu, vx0)

        # compute output in numpy
        # a bit of explaining:
        # due to the definition of sequences taps in scan, v_0[0] is
        # actually v_0[-2], and v_0[1] is v_0[-1]. The values v_0[2]
        # and v_0[3] do not get uesd ( because you do not use v_0[t]
        # in scan) which might seem strange, but then again why not use
        # v_0[t] instead of v_0[t-2] in a real application ??
        # also vx0[0] corresponds to vx0[-2], vx0[1] to vx0[-1]
        numpy_out = np.zeros((2,))
        numpy_out[0] = vu[0] * vW_in + vx0[1] * vW + vx0[0]
        numpy_out[1] = vu[1] * vW_in + numpy_out[0] * vW + vx0[1]
        utt.assert_allclose(numpy_out, aesara_out)

    # simple rnn, one input, one state, weights for each; input/state are
    # vectors, weights are scalars; using shared variables and past
    # taps (sequences and outputs) and future taps for sequences
    def test_past_future_taps_shared(self):
        rng = np.random.RandomState(utt.fetch_seed())
        vW = asarrayX(rng.uniform())
        vW_in = asarrayX(rng.uniform())
        vu = asarrayX(rng.uniform(-5.0, 5.0, size=(6,)))
        vx0 = asarrayX(rng.uniform(-5.0, 5.0, size=(2,)))

        u = vector("u")
        x0 = vector("x0")
        W_in = shared(vW_in, name="w_in")
        W = shared(vW, name="w")

        def f_rnn_shared(u_tm2, u_tp2, x_tm1, x_tm2):
            return (u_tm2 + u_tp2) * W_in + x_tm1 * W + x_tm2

        output, updates = scan(
            f_rnn_shared,
            dict(input=u, taps=[-2, 2]),
            dict(initial=x0, taps=[-1, -2]),
            [],
            n_steps=None,
            truncate_gradient=-1,
            go_backwards=False,
        )

        f8 = function([u, x0], output, updates=updates, allow_input_downcast=True)
        aesara_out = f8(vu, vx0)
        # compute output in numpy
        numpy_out = np.zeros(2)
        # think of vu[0] as vu[-2], vu[4] as vu[2]
        # and vx0[0] as vx0[-2], vx0[1] as vx0[-1]
        numpy_out[0] = (vu[0] + vu[4]) * vW_in + vx0[1] * vW + vx0[0]
        numpy_out[1] = (vu[1] + vu[5]) * vW_in + numpy_out[0] * vW + vx0[1]
        utt.assert_allclose(numpy_out, aesara_out)

    # simple rnn ; compute inplace version 1
    @utt.assertFailure_fast
    def test_inplace1(self):
        rng = np.random.RandomState(utt.fetch_seed())
        vW = asarrayX(np.random.uniform())
        vW_in = asarrayX(np.random.uniform())
        vu0 = asarrayX(rng.uniform(-5.0, 5.0, size=(3,)))
        vu1 = asarrayX(rng.uniform(-5.0, 5.0, size=(3,)))
        vu2 = asarrayX(rng.uniform(-5.0, 5.0, size=(3,)))
        vx0 = asarrayX(rng.uniform())
        vx1 = asarrayX(rng.uniform())

        u0 = vector("u0")
        u1 = vector("u1")
        u2 = vector("u2")
        mu0 = In(u0, mutable=False)
        mu1 = In(u1, mutable=True)
        mu2 = In(u2, mutable=True)
        x0 = scalar("x0")
        x1 = scalar("y0")
        W_in = shared(vW_in, "Win")
        W = shared(vW, "W")
        mode = get_mode(None).including("inplace")

        def f_rnn_shared(u0_t, u1_t, u2_t, x0_tm1, x1_tm1):
            return [
                u0_t * W_in + x0_tm1 * W + u1_t * u2_t,
                u0_t * W_in + x1_tm1 * W + u1_t + u2_t,
            ]

        outputs, updates = scan(
            f_rnn_shared,
            [u0, u1, u2],
            [dict(initial=x0, inplace=u2), dict(initial=x1, inplace=u1)],
            [],
            n_steps=None,
            truncate_gradient=-1,
            go_backwards=False,
            mode=mode,
        )

        f9 = function(
            [mu0, mu1, mu2, x0, x1],
            outputs,
            updates=updates,
            mode=mode,
            allow_input_downcast=True,
        )
        scan_node = [x for x in f9.maker.fgraph.toposort() if isinstance(x.op, Scan)]
        assert 0 in scan_node[0].op.destroy_map.keys()
        assert 1 in scan_node[0].op.destroy_map.keys()
        # compute output in numpy
        numpy_x0 = np.zeros((3,))
        numpy_x1 = np.zeros((3,))
        numpy_x0[0] = vu0[0] * vW_in + vx0 * vW + vu1[0] * vu2[0]
        numpy_x1[0] = vu0[0] * vW_in + vx1 * vW + vu1[0] + vu2[0]
        for i in range(1, 3):
            numpy_x0[i] = vu0[i] * vW_in + numpy_x0[i - 1] * vW + vu1[i] * vu2[i]
            numpy_x1[i] = vu0[i] * vW_in + numpy_x1[i - 1] * vW + vu1[i] + vu2[i]

        # note aesara computes inplace, so call function after numpy
        # equivalent is done
        (aesara_x0, aesara_x1) = f9(vu0, vu1, vu2, vx0, vx1)
        # assert that aesara does what it should
        utt.assert_allclose(aesara_x0, numpy_x0)
        utt.assert_allclose(aesara_x1, numpy_x1)

    # simple rnn ; compute inplace version 2
    @utt.assertFailure_fast
    def test_inplace2(self):
        rng = np.random.RandomState(utt.fetch_seed())
        vW = asarrayX(np.random.uniform())
        vW_in = asarrayX(np.random.uniform())
        vu0 = asarrayX(rng.uniform(-5.0, 5.0, size=(3,)))
        vu1 = asarrayX(rng.uniform(-5.0, 5.0, size=(4,)))
        vu2 = asarrayX(rng.uniform(-5.0, 5.0, size=(5,)))
        vx0 = asarrayX(rng.uniform())
        vx1 = asarrayX(rng.uniform())

        u0 = vector("u0")
        u1 = vector("u1")
        u2 = vector("u2")
        mu0 = In(u0, mutable=True)
        mu1 = In(u1, mutable=True)
        mu2 = In(u2, mutable=True)
        x0 = scalar("x0")
        x1 = scalar("y0")
        W_in = shared(vW_in, "Win")
        W = shared(vW, "W")
        mode = get_mode(None).including("inplace")

        def f_rnn_shared(u0_t, u1_t, u1_tp1, u2_tm1, u2_t, u2_tp1, x0_tm1, x1_tm1):
            return [
                u0_t * W_in + x0_tm1 * W + u1_t * u1_tp1,
                u0_t * W_in + x1_tm1 * W + u2_tm1 + u2_t + u2_tp1,
            ]

        outputs, updates = scan(
            f_rnn_shared,
            [u0, dict(input=u1, taps=[0, 1]), dict(input=u2, taps=[-1, 0, +1])],
            [dict(initial=x0), dict(initial=x1)],
            [],
            n_steps=None,
            truncate_gradient=-1,
            go_backwards=False,
            mode=mode,
        )
        f9 = function(
            [mu0, mu1, mu2, x0, x1],
            outputs,
            updates=updates,
            mode=mode,
            allow_input_downcast=True,
        )

        scan_node = [x for x in f9.maker.fgraph.toposort() if isinstance(x.op, Scan)]
        assert 0 in scan_node[0].op.destroy_map.keys()
        assert 1 in scan_node[0].op.destroy_map.keys()
        # compute output in numpy
        numpy_x0 = np.zeros((3,))
        numpy_x1 = np.zeros((3,))
        numpy_x0[0] = vu0[0] * vW_in + vx0 * vW + vu1[0] * vu1[1]
        numpy_x1[0] = vu0[0] * vW_in + vx1 * vW + vu2[0] + vu2[1] + vu2[2]
        for i in range(1, 3):
            numpy_x0[i] = vu0[i] * vW_in + numpy_x0[i - 1] * vW + vu1[i] * vu1[i + 1]
            numpy_x1[i] = (
                vu0[i] * vW_in + numpy_x1[i - 1] * vW + vu2[i] + vu2[i + 1] + vu2[i + 2]
            )

        # note aesara computes inplace, so call function after numpy
        # equivalent is done
        (aesara_x0, aesara_x1) = f9(vu0, vu1, vu2, vx0, vx1)
        # assert that aesara does what it should
        utt.assert_allclose(aesara_x0, numpy_x0)
        utt.assert_allclose(aesara_x1, numpy_x1)

    @utt.assertFailure_fast
    def test_inplace3(self):
        rng = np.random.RandomState(utt.fetch_seed())

        vx0 = asarrayX(rng.uniform())
        vx1 = asarrayX(rng.uniform())
        x0 = shared(vx0)
        x1 = shared(vx1)
        outputs, updates = scan(
            lambda x, y: (x + asarrayX(1), y + asarrayX(1)), [], [x0, x1], n_steps=3
        )
        x0 = asarrayX(np.zeros((3,)))
        x0[0] = vx0
        x0 = aet.constant(x0)
        to_replace = outputs[0].owner.inputs[0].owner.inputs[1]
        outputs = clone_replace(outputs, replace=[(to_replace, x0)])
        mode = get_mode(None).including("inplace")
        f9 = function([], outputs, updates=updates, mode=mode)
        scan_node = [x for x in f9.maker.fgraph.toposort() if isinstance(x.op, Scan)]
        assert 0 not in scan_node[0].op.destroy_map.keys()
        assert 1 in scan_node[0].op.destroy_map.keys()

    # Shared variable with updates
    def test_shared_arguments_with_updates(self):
        rng = np.random.RandomState(utt.fetch_seed())

        vW1 = asarrayX(rng.rand(2, 3))
        vW2 = asarrayX(rng.rand(3, 2))
        vu1 = asarrayX(rng.rand(3, 2))
        vu2 = asarrayX(rng.rand(3, 3))
        vy0 = asarrayX(rng.rand(3, 2))
        vy1 = asarrayX(rng.rand(2))
        vu1 = asarrayX(rng.rand(3, 2))

        W1 = shared(vW1, "W1")
        W2 = shared(vW2, "W2")
        u1 = shared(vu1, "u1")
        y1 = shared(vy1, "y1")

        def f(u1_t, u2_t, y0_tm3, y0_tm2, y0_tm1, y1_tm1):
            y0_t = dot(dot(u1_t, W1), W2) + 0.1 * y0_tm1 + 0.33 * y0_tm2 + 0.17 * y0_tm3
            y1_t = dot(u2_t, W2) + y1_tm1
            y2_t = dot(u1_t, W1)
            nwW1 = W1 + 0.1
            nwW2 = W2 + 0.05
            # return outputs followed by a list of updates
            return ([y0_t, y1_t, y2_t], [(W1, nwW1), (W2, nwW2)])

        u2 = matrix("u2")
        y0 = matrix("y0")
        outputs, updates = scan(
            f,
            [u1, u2],
            [dict(initial=y0, taps=[-3, -2, -1]), y1, None],
            [],
            n_steps=None,
            go_backwards=False,
            truncate_gradient=-1,
        )

        f10 = function([u2, y0], outputs, updates=updates, allow_input_downcast=True)
        allstuff = f10(vu2, vy0)
        aesara_y0, aesara_y1, aesara_y2 = allstuff

        # do things in numpy
        numpy_y0 = np.zeros((6, 2))
        numpy_y1 = np.zeros((4, 2))
        numpy_y2 = np.zeros((3, 3))
        numpy_y0[:3] = vy0
        numpy_y1[0] = vy1
        numpy_W1 = vW1.copy()
        numpy_W2 = vW2.copy()
        for idx in range(3):
            numpy_y0[idx + 3] = (
                np.dot(np.dot(vu1[idx, :], numpy_W1), numpy_W2)
                + 0.1 * numpy_y0[idx + 2]
                + 0.33 * numpy_y0[idx + 1]
                + 0.17 * numpy_y0[idx]
            )
            numpy_y1[idx + 1] = np.dot(vu2[idx, :], numpy_W2) + numpy_y1[idx]
            numpy_y2[idx] = np.dot(vu1[idx, :], numpy_W1)
            numpy_W1 = numpy_W1 + 0.1
            numpy_W2 = numpy_W2 + 0.05

        utt.assert_allclose(aesara_y0, numpy_y0[3:])
        utt.assert_allclose(aesara_y1, numpy_y1[1:])
        utt.assert_allclose(aesara_y2, numpy_y2)
        utt.assert_allclose(W1.get_value(), numpy_W1)
        utt.assert_allclose(W2.get_value(), numpy_W2)

    def test_grad_dtype_change(self):
        x = fscalar("x")
        y = fscalar("y")
        c = iscalar("c")

        def inner_fn(cond, x, y):
            new_cond = aet.cast(aet.switch(cond, x, y), "int32")
            new_x = aet.switch(cond, sigmoid(y * x), x)
            new_y = aet.switch(cond, y, sigmoid(x))
            return new_cond, new_x, new_y

        values, _ = scan(
            inner_fn,
            outputs_info=[c, x, y],
            n_steps=10,
            truncate_gradient=-1,
            go_backwards=False,
        )
        gX, gY = grad(values[1].sum(), [x, y])
        f = function([c, x, y], [gX, gY], allow_input_downcast=True)
        # Check for runtime errors
        f(np.int32(0), np.float32(1.0), np.float32(0.5))

    def test_simple_shared_random(self):
        aesara_rng = RandomStream(utt.fetch_seed())

        values, updates = scan(
            lambda: aesara_rng.uniform(-1, 1, size=(2,)),
            [],
            [],
            [],
            n_steps=5,
            truncate_gradient=-1,
            go_backwards=False,
        )
        my_f = function([], values, updates=updates, allow_input_downcast=True)

        rng_seed = np.random.RandomState(utt.fetch_seed()).randint(2 ** 30)
        rng = np.random.RandomState(int(rng_seed))  # int() is for 32bit

        numpy_v = np.zeros((10, 2))
        for i in range(10):
            numpy_v[i] = rng.uniform(-1, 1, size=(2,))

        aesara_v = my_f()
        utt.assert_allclose(aesara_v, numpy_v[:5, :])
        aesara_v = my_f()
        utt.assert_allclose(aesara_v, numpy_v[5:, :])

    def test_gibbs_chain(self):
        rng = np.random.RandomState(utt.fetch_seed())
        v_W = np.array(rng.rand(20, 30) - 0.5, dtype="float32")
        v_vsample = np.array(
            rng.binomial(
                1,
                0.5,
                size=(3, 20),
            ),
            dtype="float32",
        )
        v_bvis = np.array(rng.rand(20) - 0.5, dtype="float32")
        v_bhid = np.array(rng.rand(30) - 0.5, dtype="float32")
        W = shared(v_W, "vW")
        bhid = shared(v_bhid, "vbhid")
        bvis = shared(v_bvis, "vbvis")
        vsample = matrix(dtype="float32")
        trng = RandomStream(utt.fetch_seed())

        def f(vsample_tm1):
            hmean_t = sigmoid(dot(vsample_tm1, W) + bhid)
            hsample_t = aet.cast(
                trng.binomial(1, hmean_t, size=hmean_t.shape), dtype="float32"
            )
            vmean_t = sigmoid(dot(hsample_t, W.T) + bvis)
            return aet.cast(
                trng.binomial(1, vmean_t, size=vmean_t.shape), dtype="float32"
            )

        aesara_vsamples, updates = scan(
            f, [], vsample, [], n_steps=10, truncate_gradient=-1, go_backwards=False
        )

        my_f = function(
            [vsample], aesara_vsamples[-1], updates=updates, allow_input_downcast=True
        )

        _rng = np.random.RandomState(utt.fetch_seed())
        rng_seed = _rng.randint(2 ** 30)
        nrng1 = np.random.RandomState(int(rng_seed))  # int() is for 32bit

        rng_seed = _rng.randint(2 ** 30)
        nrng2 = np.random.RandomState(int(rng_seed))  # int() is for 32bit

        def numpy_implementation(vsample):
            for idx in range(10):
                hmean = 1.0 / (1.0 + np.exp(-(np.dot(vsample, v_W) + v_bhid)))
                hsample = np.array(
                    nrng1.binomial(1, hmean, size=hmean.shape), dtype="float32"
                )
                vmean = 1.0 / (1.0 + np.exp(-(np.dot(hsample, v_W.T) + v_bvis)))
                vsample = np.array(
                    nrng2.binomial(1, vmean, size=vmean.shape), dtype="float32"
                )

            return vsample

        t_result = my_f(v_vsample)
        n_result = numpy_implementation(v_vsample)
        utt.assert_allclose(t_result, n_result)

    def test_only_shared_no_input_no_output(self):
        rng = np.random.RandomState(utt.fetch_seed())
        v_state = asarrayX(rng.uniform())
        state = shared(v_state, "vstate")

        def f_2():
            return OrderedDict([(state, 2 * state)])

        n_steps = iscalar("nstep")
        output, updates = scan(
            f_2, [], [], [], n_steps=n_steps, truncate_gradient=-1, go_backwards=False
        )
        this_f = function([n_steps], output, updates=updates, allow_input_downcast=True)
        n_steps = 3
        this_f(n_steps)
        numpy_state = v_state * (2 ** (n_steps))
        utt.assert_allclose(state.get_value(), numpy_state)

    def test_map_functionality(self):
        def f_rnn(u_t):
            return u_t + 3

        u = vector("u")

        outputs, updates = scan(
            f_rnn, u, [], [], n_steps=None, truncate_gradient=-1, go_backwards=False
        )

        f2 = function([u], outputs, updates=updates, allow_input_downcast=True)
        rng = np.random.RandomState(utt.fetch_seed())

        v_u = rng.uniform(-5.0, 5.0, size=(5,))
        numpy_result = v_u + 3
        aesara_result = f2(v_u)
        utt.assert_allclose(aesara_result, numpy_result)

    def test_map(self):
        v = vector("v")
        abs_expr, abs_updates = aet_map(
            lambda x: abs(x), v, [], truncate_gradient=-1, go_backwards=False
        )

        f = function([v], abs_expr, updates=abs_updates, allow_input_downcast=True)

        rng = np.random.RandomState(utt.fetch_seed())
        vals = rng.uniform(-5.0, 5.0, size=(10,))
        abs_vals = abs(vals)
        aesara_vals = f(vals)
        utt.assert_allclose(abs_vals, aesara_vals)

    def test_backwards(self):
        def f_rnn(u_t, x_tm1, W_in, W):
            return u_t * W_in + x_tm1 * W

        u = vector("u")
        x0 = scalar("x0")
        W_in = scalar("win")
        W = scalar("w")

        output, updates = scan(
            f_rnn,
            u,
            x0,
            [W_in, W],
            n_steps=None,
            truncate_gradient=-1,
            go_backwards=True,
        )

        f2 = function(
            [u, x0, W_in, W], output, updates=updates, allow_input_downcast=True
        )
        # get random initial values
        rng = np.random.RandomState(utt.fetch_seed())
        v_u = rng.uniform(-5.0, 5.0, size=(4,))
        v_x0 = rng.uniform()
        W = rng.uniform()
        W_in = rng.uniform()

        # compute the output in numpy
        v_out = np.zeros((4,))
        v_out[0] = v_u[3] * W_in + v_x0 * W
        for step in range(1, 4):
            v_out[step] = v_u[3 - step] * W_in + v_out[step - 1] * W

        aesara_values = f2(v_u, v_x0, W_in, W)
        utt.assert_allclose(aesara_values, v_out)

    def test_reduce(self):
        v = vector("v")
        s = scalar("s")
        result, updates = aet_reduce(lambda x, y: x + y, v, s)

        f = function([v, s], result, updates=updates, allow_input_downcast=True)
        rng = np.random.RandomState(utt.fetch_seed())
        v_v = rng.uniform(-5.0, 5.0, size=(5,))
        assert abs(np.sum(v_v) - f(v_v, 0.0)) < 1e-3

    def test_grad_one_output(self):
        def f_rnn(u_t, x_tm1, W_in, W):
            return u_t * W_in + x_tm1 * W

        u = vector("u")
        x0 = scalar("x0")
        W_in = scalar("W_in")
        W = scalar("W")

        cost, updates = scan_project_sum(
            f_rnn,
            u,
            x0,
            [W_in, W],
            n_steps=None,
            truncate_gradient=-1,
            go_backwards=False,
        )
        gu, gx0, gW_in, gW = grad(cost, [u, x0, W_in, W])
        grad_fn = function(
            [u, x0, W_in, W],
            [gu, gx0, gW_in, gW],
            updates=updates,
            no_default_updates=True,
            allow_input_downcast=True,
        )
        cost_fn = function(
            [u, x0, W_in, W],
            cost,
            updates=updates,
            no_default_updates=True,
            allow_input_downcast=True,
        )

        # get random initial values
        rng = np.random.RandomState(utt.fetch_seed())
        v_u = np.array(rng.uniform(-0.5, 0.5, size=(10,)), dtype=config.floatX)
        v_x0 = np.array(rng.uniform(), dtype=config.floatX)
        W = np.array(rng.uniform(), dtype=config.floatX)
        W_in = np.array(rng.uniform(), dtype=config.floatX)

        analytic_grad = grad_fn(v_u, v_x0, W_in, W)

        num_grad = multiple_outputs_numeric_grad(cost_fn, [v_u, v_x0, W_in, W])
        max_err, max_err_pos = num_grad.max_err(analytic_grad)
        assert max_err <= 1e-2

    def test_grad_multiple_outs(self):
        rng = np.random.RandomState(utt.fetch_seed())
        vW_in2 = asarrayX(rng.uniform(-0.1, 0.1, size=(2,)))
        vW = asarrayX(rng.uniform(-0.1, 0.1, size=(2, 2)))
        vWout = asarrayX(rng.uniform(-0.1, 0.1, size=(2,)))
        vW_in1 = asarrayX(rng.uniform(-0.1, 0.1, size=(2, 2)))
        v_u1 = asarrayX(rng.uniform(-0.1, 0.1, size=(7, 2)))
        v_u2 = asarrayX(rng.uniform(-0.1, 0.1, size=(7,)))
        v_x0 = asarrayX(rng.uniform(0.1, 0.1, size=(2,)))
        v_y0 = asarrayX(rng.uniform())

        W_in2 = shared(vW_in2, name="win2")
        W = shared(vW, name="w")
        W_out = shared(vWout, name="wout")
        W_in1 = matrix("win")
        u1 = matrix("u1")
        u2 = vector("u2")
        x0 = vector("x0")
        y0 = scalar("y0")

        def f_rnn_cmpl(u1_t, u2_t, x_tm1, y_tm1, W_in1):
            return [
                dot(u1_t, W_in1) + u2_t * W_in2 + dot(x_tm1, W),
                dot(x_tm1, W_out),
            ]

        cost, updates = scan_project_sum(
            f_rnn_cmpl,
            [u1, u2],
            [x0, y0],
            W_in1,
            n_steps=None,
            truncate_gradient=-1,
            go_backwards=False,
        )
        # y0 is actually not used in the computation of the cost
        params = [u1, u2, x0, y0, W_in1]
        gparams = grad(cost, params, disconnected_inputs="ignore")

        grad_fn = function(
            [u1, u2, x0, y0, W_in1],
            gparams,
            updates=updates,
            no_default_updates=True,
            allow_input_downcast=True,
        )
        cost_fn = function(
            [u1, u2, x0, y0, W_in1],
            cost,
            updates=updates,
            no_default_updates=True,
            allow_input_downcast=True,
        )

        num_grad = multiple_outputs_numeric_grad(
            cost_fn, [v_u1, v_u2, v_x0, v_y0, vW_in1]
        )
        analytic_grad = grad_fn(v_u1, v_u2, v_x0, v_y0, vW_in1)
        max_err, max_err_pos = num_grad.max_err(analytic_grad)
        assert max_err <= 1e-2

    @pytest.mark.slow
    def test_grad_multiple_outs_taps(self):
        n = 5
        rng = np.random.RandomState(utt.fetch_seed())

        vW_in2 = asarrayX(rng.uniform(-0.2, 0.2, size=(2,)))
        vW = asarrayX(rng.uniform(-0.2, 0.2, size=(2, 2)))
        vWout = asarrayX(rng.uniform(-0.2, 0.2, size=(2,)))
        vW_in1 = asarrayX(rng.uniform(-0.2, 0.2, size=(2, 2)))
        v_u1 = asarrayX(rng.uniform(-0.2, 0.2, size=(n, 2)))
        v_u2 = asarrayX(rng.uniform(-0.2, 0.2, size=(n + 2, 2)))
        v_x0 = asarrayX(rng.uniform(0.2, 0.2, size=(2,)))

        v_y0 = asarrayX(rng.uniform(size=(3,)))

        W_in2 = shared(vW_in2, name="win2")
        W = shared(vW, name="w")
        W_out = shared(vWout, name="wout")
        W_in1 = matrix("win")
        u1 = matrix("u1")
        u2 = matrix("u2")
        x0 = vector("x0")
        y0 = vector("y0")

        W_in1.tag.test_value = vW_in1
        u1.tag.test_value = v_u1
        u2.tag.test_value = v_u2
        x0.tag.test_value = v_x0
        y0.tag.test_value = v_y0

        def f_rnn_cmpl(u1_t, u2_tm1, u2_t, u2_tp1, x_tm1, y_tm1, y_tm3, W_in1):
            return [
                dot(u1_t, W_in1) + (u2_t + u2_tm1 * u2_tp1) * W_in2 + dot(x_tm1, W),
                (y_tm1 + y_tm3) * dot(x_tm1, W_out),
                dot(u1_t, W_in1),
            ]

        # We change the compute_test_value[_opt] flag to run the
        # assert in Scan.grad() of the new scan input sequence related
        # to outer_mitsot_outs, outer_sitsot_outs and
        # outer_nitsot_outs. This allow to test an old Scan bug.
        with config.change_flags(
            compute_test_value="raise", compute_test_value_opt="raise"
        ):
            cost, updates = scan_project_sum(
                f_rnn_cmpl,
                [u1, dict(input=u2, taps=[-1, 0, 1])],
                [x0, dict(initial=y0, taps=[-1, -3]), None],
                W_in1,
                n_steps=None,
                truncate_gradient=-1,
                go_backwards=False,
            )
            params = [u1, u2, x0, y0, W_in1]
            gparams = grad(cost, params)

            cost_fn = function(
                [u1, u2, x0, y0, W_in1],
                cost,
                updates=updates,
                no_default_updates=True,
                allow_input_downcast=True,
            )
            grad_fn = function(
                [u1, u2, x0, y0, W_in1],
                gparams,
                updates=updates,
                no_default_updates=True,
                allow_input_downcast=True,
            )

        num_grad = multiple_outputs_numeric_grad(
            cost_fn, [v_u1, v_u2, v_x0, v_y0, vW_in1]
        )

        analytic_grad = grad_fn(v_u1, v_u2, v_x0, v_y0, vW_in1)
        max_err, max_err_pos = num_grad.max_err(analytic_grad)
        assert max_err <= 1e-2

    @pytest.mark.slow
    def test_grad_multiple_outs_taps_backwards(self):
        n = 5
        rng = np.random.RandomState(utt.fetch_seed())
        vW_in2 = asarrayX(rng.uniform(-0.2, 0.2, size=(2,)))
        vW = asarrayX(rng.uniform(-0.2, 0.2, size=(2, 2)))
        vWout = asarrayX(rng.uniform(-0.2, 0.2, size=(2,)))
        vW_in1 = asarrayX(rng.uniform(-0.2, 0.2, size=(2, 2)))
        v_u1 = asarrayX(rng.uniform(-0.2, 0.2, size=(n, 2)))
        v_u2 = asarrayX(rng.uniform(-0.2, 0.2, size=(n + 2, 2)))
        v_x0 = asarrayX(rng.uniform(-0.2, 0.2, size=(2,)))
        v_y0 = asarrayX(rng.uniform(size=(3,)))

        W_in2 = shared(vW_in2, name="win2")
        W = shared(vW, name="w")
        W_out = shared(vWout, name="wout")
        W_in1 = matrix("win")
        u1 = matrix("u1")
        u2 = matrix("u2")
        x0 = vector("x0")
        y0 = vector("y0")

        def f_rnn_cmpl(u1_t, u2_tm1, u2_t, u2_tp1, x_tm1, y_tm1, y_tm3, W_in1):
            return [
                dot(u1_t, W_in1) + (u2_t + u2_tm1 * u2_tp1) * W_in2 + dot(x_tm1, W),
                (y_tm1 + y_tm3) * dot(x_tm1, W_out),
            ]

        cost, updates = scan_project_sum(
            f_rnn_cmpl,
            [u1, dict(input=u2, taps=[-1, 0, 1])],
            [x0, dict(initial=y0, taps=[-1, -3])],
            W_in1,
            n_steps=None,
            truncate_gradient=-1,
            go_backwards=True,
        )
        params = [u1, u2, x0, y0, W_in1]
        gparams = grad(cost, params)
        grad_fn = function(
            [u1, u2, x0, y0, W_in1],
            gparams,
            updates=updates,
            no_default_updates=True,
            allow_input_downcast=True,
        )
        cost_fn = function(
            [u1, u2, x0, y0, W_in1],
            cost,
            updates=updates,
            no_default_updates=True,
            allow_input_downcast=True,
        )

        num_grad = multiple_outputs_numeric_grad(
            cost_fn, [v_u1, v_u2, v_x0, v_y0, vW_in1]
        )

        analytic_grad = grad_fn(v_u1, v_u2, v_x0, v_y0, vW_in1)
        max_err, max_err_pos = num_grad.max_err(analytic_grad)
        assert max_err <= 1e-2

    def test_grad_multiple_outs_some_uncomputable(self):
        rng = np.random.RandomState(utt.fetch_seed())
        vW_in = asarrayX(rng.uniform(-3.0, 3.0, size=(2, 2)))
        v_u = asarrayX(rng.uniform(-3.0, 3.0, size=(5, 2)))
        v_u2 = np.array([1, 3, 4, 6, 8], dtype="int32")
        v_x0 = asarrayX(rng.uniform(-3.0, 3.0, size=(2,)))

        W_in = matrix("win")
        u = matrix("u1")
        u2 = ivector("u2")
        x0 = vector("x0", dtype=config.floatX)

        def f_rnn_cmpl(u_t, u2_t, x_tm1, W_in):
            trng1 = RandomStream(123)
            x_t = (
                aet.cast(u2_t, config.floatX)
                + dot(u_t, W_in)
                + x_tm1
                + trng1.uniform(low=-1.1, high=1.1, dtype=config.floatX)
            )
            return x_t, 2 * u2_t

        cost, updates = scan_project_sum(
            f_rnn_cmpl,
            [u, u2],
            [x0, None],
            W_in,
            n_steps=None,
            truncate_gradient=-1,
            go_backwards=False,
        )
        params = [u, u2, x0, W_in]
        gparams = grad(cost, params)
        grad_fn = function(
            [u, u2, x0, W_in],
            gparams,
            updates=updates,
            no_default_updates=True,
            allow_input_downcast=True,
        )
        cost_fn = function(
            [u, u2, x0, W_in],
            cost,
            updates=updates,
            no_default_updates=True,
            allow_input_downcast=True,
        )

        def reset_rng_fn(fn, *args):
            for idx, arg in enumerate(fn.maker.expanded_inputs):
                if arg.value and isinstance(arg.value.data, np.random.RandomState):
                    obj = fn.maker.expanded_inputs[idx].value
                    obj.data = np.random.RandomState(123)
                    fn.maker.expanded_inputs[idx].value = obj
            return fn(*args)

        def reset_rng_cost_fn(*args):
            return reset_rng_fn(cost_fn, *args)

        def reset_rng_grad_fn(*args):
            return reset_rng_fn(grad_fn, *args)

        num_grad = multiple_outputs_numeric_grad(
            reset_rng_cost_fn,
            [v_u, v_u2, v_x0, vW_in],
            ndarray_mask=[True, False, True, True],
        )
        analytic_grad = reset_rng_grad_fn(v_u, v_u2, v_x0, vW_in)
        max_err, max_err_pos = num_grad.max_err(analytic_grad)
        assert max_err <= 1e-2

        # Also validate that the mappings outer_inp_from_outer_out and
        # outer_inp_from_inner_inp produce the correct results
        scan_node = list(updates.values())[0].owner

        result = scan_node.op.var_mappings["outer_inp_from_outer_out"]
        expected_result = {0: 3, 1: 5, 2: 4}
        assert result == expected_result

        result = scan_node.op.var_mappings["outer_inp_from_inner_inp"]
        expected_result = {0: 1, 1: 2, 2: 3, 3: 4, 4: 6}
        assert result == expected_result

    def test_grad_multiple_outs_some_truncate(self):
        rng = np.random.RandomState(utt.fetch_seed())
        vW_in = asarrayX(rng.uniform(-0.1, 0.1, size=(2, 2)))
        v_u = asarrayX(rng.uniform(-0.1, 0.1, size=(5, 2)))
        v_x0 = asarrayX(rng.uniform(-0.1, 0.1, size=(2,)))

        W_in = matrix("win")
        u = matrix("u1")
        x0 = vector("x0")
        # trng  = RandomStream(
        #                                               utt.fetch_seed())

        def f_rnn_cmpl(u_t, x_tm1, W_in):
            trng1 = RandomStream(123)
            rnd_nb = trng1.uniform(-0.1, 0.1)
            x_t = dot(u_t, W_in) + x_tm1 + rnd_nb
            x_t = aet.cast(x_t, dtype=config.floatX)
            return x_t

        cost, updates = scan_project_sum(
            f_rnn_cmpl,
            u,
            x0,
            W_in,
            n_steps=None,
            truncate_gradient=3,
            go_backwards=False,
        )
        params = [u, x0, W_in]
        gparams = grad(cost, params)

        grad_fn = function(
            [u, x0, W_in],
            gparams,
            updates=updates,
            no_default_updates=True,
            allow_input_downcast=True,
        )
        cost_fn = function(
            [u, x0, W_in],
            cost,
            updates=updates,
            no_default_updates=True,
            allow_input_downcast=True,
        )

        def reset_rng_fn(fn, *args):
            for idx, arg in enumerate(fn.maker.expanded_inputs):
                if arg.value and isinstance(arg.value.data, np.random.RandomState):
                    obj = fn.maker.expanded_inputs[idx].value
                    obj.data = np.random.RandomState(123)
                    fn.maker.expanded_inputs[idx].value = obj
            out = fn(*args)
            return out

        def reset_rng_cost_fn(*args):
            return reset_rng_fn(cost_fn, *args)

        def reset_rng_grad_fn(*args):
            return reset_rng_fn(grad_fn, *args)

        multiple_outputs_numeric_grad(reset_rng_cost_fn, [v_u, v_x0, vW_in])

        analytic_grad = reset_rng_grad_fn(v_u, v_x0, vW_in)
        utt.assert_allclose(analytic_grad[0][:2], np.zeros((2, 2)))

    def test_grad_multiple_outs_some_disconnected(self):
        final_cost = self._grad_mout_helper(100, mode_nodebug)
        assert final_cost < 0.02, final_cost

    def test_grad_multiple_outs_some_disconnected_2(self):
        # This is to try the network in DEBUG_MODE, but not fully
        # train it since that would take 3 hours
        self._grad_mout_helper(1, None)

    def _grad_mout_helper(self, n_iters, mode):
        # Created on Tue Oct 07 13:28:51 2014
        # @author: vaneetke
        rng = np.random.RandomState(utt.fetch_seed())
        n_hid = 3
        n_in = 1
        n_out = 1

        W_hh_v = asarrayX(rng.uniform(-1, 1, size=(n_hid, n_hid)))
        h0_v = asarrayX(rng.uniform(-1, 1, size=(2, n_hid)))
        b_h_v = asarrayX(rng.uniform(-0.01, 0.01, size=(n_hid)))
        W_ih_v = asarrayX(rng.uniform(-1, 1, size=(n_in, n_hid)))
        W_ho_v = asarrayX(rng.uniform(-1, 1, size=(n_hid, n_out)))
        b_o_v = asarrayX(rng.uniform(-0.01, 0.01, size=(n_out)))

        # parameters of the rnn
        b_h = shared(b_h_v, name="b_h")
        h0 = shared(h0_v, name="h0")
        W_ih = shared(W_ih_v, name="W_ih")
        W_hh = shared(W_hh_v, name="W_hh")
        W_ho = shared(W_ho_v, name="W_ho")
        b_o = shared(b_o_v, name="b_o")
        params = [W_ih, W_hh, b_h, W_ho, b_o, h0]

        # first dimension is time
        x = matrix()

        # sequences: x_t
        # prior results: h_tm2, h_tm1
        # non-sequences: W_ih, W_hh, W_ho, b_h
        def one_step(x_t, h_tm2, h_tm1, W_ih, W_hh, b_h, W_ho, b_o):
            h_t = tanh(dot(x_t, W_ih) + dot(h_tm2, W_hh) + b_h)
            y_t = dot(h_t, W_ho) + b_o
            return [h_t, y_t]

        # hidden and outputs of the entire sequence
        [h, y], _ = scan(
            fn=one_step,
            sequences=dict(input=x),
            # corresponds to the return type of one_step
            outputs_info=[dict(initial=h0, taps=[-2, -1]), None],
            non_sequences=[W_ih, W_hh, b_h, W_ho, b_o],
            mode=mode,
        )

        # target values
        t = matrix()

        # learning rate
        lr = asarrayX(0.1)
        learning_rate = shared(lr)

        cost = (0.5 * ((y - t) ** 2.0).mean()) + (0.5 * (y.std() - t.std()) ** 2.0)

        gparams = grad(cost, params)
        updates = [
            (param, param - gparam * learning_rate)
            for param, gparam in zip(params, gparams)
        ]
        learn_rnn_fn = function(inputs=[x, t], outputs=cost, updates=updates, mode=mode)
        function(inputs=[x], outputs=y, mode=mode)

        # artificial data
        x_v = np.arange(0.0, 10.49, 0.21, dtype=config.floatX)
        x_v = x_v.reshape(len(x_v), 1)
        s_v = np.sin(x_v)
        t_v = np.roll(s_v, -1)[:-1]
        s_v = s_v[:-1]
        for i in range(n_iters):
            cost = learn_rnn_fn(s_v, t_v)
        # pred = eval_rnn_fn(s_v)
        return cost

    def test_draw_as_input_to_scan(self):
        trng = RandomStream(123)

        x = matrix("x")
        y = trng.binomial(1, x, size=x.shape)
        z, updates = scan(lambda a: a, non_sequences=y, n_steps=2)

        f = function([x], [y, z], updates=updates, allow_input_downcast=True)

        rng = np.random.RandomState(utt.fetch_seed())
        nx = rng.uniform(size=(10, 10))
        ny1, nz1 = f(nx)
        ny2, nz2 = f(nx)

        utt.assert_allclose([ny1, ny1], nz1)
        utt.assert_allclose([ny2, ny2], nz2)
        assert not np.allclose(ny1, ny2)

    def test_grad_of_shared(self):
        x1 = shared(3.0)
        x1.name = "x1"
        x2 = vector("x2")
        y, updates = scan(lambda v: aet.cast(v * x1, config.floatX), sequences=x2)
        m = grad(y.sum(), x1)

        f = function([x2], m, allow_input_downcast=True)
        utt.assert_allclose(f([2, 3]), 5)

    def test_computing_gradient(self):
        x1 = scalar("x1")
        x2 = shared(np.array([1, 2, 3, 4, 5]), name="x2")
        K = x2 * x1

        out, updates = scan(
            lambda i, v: grad(K[i], v),
            sequences=aet.arange(K.shape[0]),
            non_sequences=x1,
        )
        f = function([x1], out, allow_input_downcast=True)

        assert np.all(f(3.0) != 0.0)

    def test_shared_updates(self):
        X = shared(np.array(1))

        out, updates = scan(
            lambda: OrderedDict([(X, (X + 1))]),
            outputs_info=[],
            non_sequences=[],
            sequences=[],
            n_steps=10,
        )

        f = function([], [], updates=updates)
        f()
        assert X.get_value() == 11

    def test_memory_aliasing_updates(self):
        x = shared(np.array(1))
        y = shared(np.array(1))

        out, updates = scan(
            lambda: OrderedDict([(x, x + 1), (y, x)]),
            outputs_info=[],
            non_sequences=[],
            sequences=[],
            n_steps=10,
        )

        f = function([], [], updates=updates)
        f()
        assert not np.may_share_memory(x.container.storage[0], y.container.storage[0])

        assert x.get_value() != y.get_value()

    def test_scan_output_padding(self):
        # Scan outputs are usually lists, whose entries correspond to the
        # intermediate result. When n_steps=1, some extra machinery is
        # required in order to mimic this interface. Scan thus calls
        # tensor.shape_padleft on the inner function outputs.
        #
        # However, this is not the proper behavior for shared variables,
        # they should not be padded in any way
        #
        # This unit test addresses the bug fix of changeset ba7157e95cb1.

        a = vector()
        init_a = vector()
        b = shared(np.random.rand(5, 4))

        def inner_func(a):
            return a + 1, OrderedDict([(b, 2 * b)])

        out, updates = scan(
            inner_func, outputs_info=[OrderedDict([("initial", init_a)])], n_steps=1
        )
        out = out[-1]
        assert out.type.ndim == a.type.ndim
        assert updates[b].type.ndim == b.type.ndim

        out, updates = scan(inner_func, outputs_info=[init_a], n_steps=1)
        assert out.type.ndim == a.type.ndim + 1
        assert updates[b].type.ndim == b.type.ndim

    def test_scan_extra_inputs_hessian(self):
        x = vector("x")
        A = matrix("A")
        fc1 = shared(0.5, name="fc1")
        fc2 = shared(0.9, name="fc2")
        y = fc1 * dot(x * x, dot(A, x))
        y.name = "y"
        gy = grad(y, x)
        gy.name = "gy"
        hy, updates = scan(
            lambda i, gy, x: grad(gy[i] * fc2, x),
            sequences=aet.arange(gy.shape[0]),
            non_sequences=[gy, x],
        )

        f = function([x, A], hy, allow_input_downcast=True)
        vx = np.array([1.0, 1.0], dtype=config.floatX)
        vA = np.array([[1.0, 1.0], [1.0, 0.0]], dtype=config.floatX)
        vR = np.array([[3.6, 1.8], [1.8, 0.9]], dtype=config.floatX)
        out = f(vx, vA)

        utt.assert_allclose(out, vR)

    def test_cloning_no_replace_strict_copy_inputs(self):
        # This has nothing to do with scan, but it refers to the clone
        # function that scan uses internally and that pfunc uses now and
        # that users might want to use
        x = vector("x")
        y = vector("y")
        z = shared(0.25)

        f1 = z * (x + y) ** 2 + 5
        f2 = clone_replace(f1, replace=None, strict=True, share_inputs=True)
        f2_inp = graph_inputs([f2])

        assert z in f2_inp
        assert x in f2_inp
        assert y in f2_inp

    def test_cloning_no_replace_strict_not_copy_inputs(self):
        # This has nothing to do with scan, but it refers to the clone
        # function that scan uses internally and that pfunc uses now and
        # that users might want to use
        x = vector("x")
        y = vector("y")
        z = shared(0.25)

        f1 = z * (x + y) ** 2 + 5
        f2 = clone_replace(f1, replace=None, strict=True, share_inputs=False)
        f2_inp = graph_inputs([f2])

        assert z not in f2_inp
        assert x not in f2_inp
        assert y not in f2_inp

    def test_cloning_replace_strict_copy_inputs(self):
        # This has nothing to do with scan, but it refers to the clone
        # function that scan uses internally and that pfunc uses now and
        # that users might want to use
        x = vector("x")
        y = vector("y")
        y2 = vector("y2")
        z = shared(0.25)

        f1 = z * (x + y) ** 2 + 5
        f2 = clone_replace(
            f1, replace=OrderedDict([(y, y2)]), strict=True, share_inputs=True
        )
        f2_inp = graph_inputs([f2])
        assert z in f2_inp
        assert x in f2_inp
        assert y2 in f2_inp

    def test_cloning_replace_not_strict_copy_inputs(self):
        # This has nothing to do with scan, but it refers to the clone
        # function that scan uses internally and that pfunc uses now and
        # that users might want to use
        x = vector("x")
        y = fvector("y")
        y2 = dvector("y2")
        z = shared(0.25)

        f1 = z * (x + y) ** 2 + 5
        f2 = clone_replace(
            f1, replace=OrderedDict([(y, y2)]), strict=False, share_inputs=True
        )
        f2_inp = graph_inputs([f2])
        assert z in f2_inp
        assert x in f2_inp
        assert y2 in f2_inp

    def test_cloning_replace_strict_not_copy_inputs(self):
        # This has nothing to do with scan, but it refers to the clone
        # function that scan uses internally and that pfunc uses now and
        # that users might want to use
        x = vector("x")
        y = vector("y")
        y2 = vector("y2")
        z = shared(0.25)

        f1 = z * (x + y) ** 2 + 5
        f2 = clone_replace(f1, replace=[(y, y2)], strict=True, share_inputs=False)
        f2_inp = graph_inputs([f2])
        assert z not in f2_inp
        assert x not in f2_inp
        assert y2 not in f2_inp

    def test_cloning_replace_not_strict_not_copy_inputs(self):
        # This has nothing to do with scan, but it refers to the clone
        # function that scan uses internally and that pfunc uses now and
        # that users might want to use
        x = vector("x")
        y = fvector("y")
        y2 = dvector("y2")
        z = shared(0.25)

        f1 = z * (x + y) ** 2 + 5
        f2 = clone_replace(f1, replace=[(y, y2)], strict=False, share_inputs=False)
        f2_inp = graph_inputs([f2])
        assert z not in f2_inp
        assert x not in f2_inp
        assert y2 not in f2_inp

    # TEST RE-ordering of inputs
    # some rnn with multiple outputs and multiple inputs; other
    # dimension instead of scalars/vectors
    def test_reordering(self):
        rng = np.random.RandomState(utt.fetch_seed())

        vW_in2 = asarrayX(rng.uniform(-0.5, 0.5, size=(2,)))
        vW = asarrayX(rng.uniform(-0.5, 0.5, size=(2, 2)))
        vWout = asarrayX(rng.uniform(-0.5, 0.5, size=(2,)))
        vW_in1 = asarrayX(rng.uniform(-0.5, 0.5, size=(2, 2)))
        v_u1 = asarrayX(rng.uniform(-0.5, 0.5, size=(3, 2)))
        v_u2 = asarrayX(rng.uniform(-0.5, 0.5, size=(3,)))
        v_x0 = asarrayX(rng.uniform(-0.5, 0.5, size=(2,)))
        v_y0 = asarrayX(rng.uniform(size=(3,)))

        W_in2 = shared(vW_in2, name="win2")
        W = shared(vW, name="w")
        W_out = shared(vWout, name="wout")
        W_in1 = matrix("win")
        u1 = matrix("u1")
        u2 = vector("u2")
        x0 = vector("x0")
        y0 = vector("y0")

        def f_rnn_cmpl(u1_t, u2_t, x_tm1, y_tm1, y_tm3, W_in1):
            return [
                y_tm3 + 1,
                y_tm3 + 2,
                dot(u1_t, W_in1) + u2_t * W_in2 + dot(x_tm1, W),
                y_tm1 + dot(x_tm1, W_out),
            ]

        outputs, updates = scan(
            f_rnn_cmpl,
            [u1, u2],
            [None, None, x0, dict(initial=y0, taps=[-1, -3])],
            W_in1,
            n_steps=None,
            truncate_gradient=-1,
            go_backwards=False,
        )

        f4 = function(
            [u1, u2, x0, y0, W_in1], outputs, updates=updates, allow_input_downcast=True
        )

        # compute the values in numpy
        v_x = np.zeros((3, 2), dtype=config.floatX)
        v_y = np.zeros((3,), dtype=config.floatX)
        v_x[0] = np.dot(v_u1[0], vW_in1) + v_u2[0] * vW_in2 + np.dot(v_x0, vW)
        v_y[0] = np.dot(v_x0, vWout) + v_y0[2]
        for i in range(1, 3):
            v_x[i] = np.dot(v_u1[i], vW_in1) + v_u2[i] * vW_in2 + np.dot(v_x[i - 1], vW)
            v_y[i] = np.dot(v_x[i - 1], vWout) + v_y[i - 1]

        (aesara_dump1, aesara_dump2, aesara_x, aesara_y) = f4(
            v_u1, v_u2, v_x0, v_y0, vW_in1
        )

        utt.assert_allclose(aesara_x, v_x)
        utt.assert_allclose(aesara_y, v_y)

    def test_scan_as_tensor_on_gradients(self):
        # Bug reported by cityhall on scan when computing the gradients

        to_scan = dvector("to_scan")
        seq = dmatrix("seq")
        f1 = dscalar("f1")

        def scanStep(prev, seq, f1):
            return prev + f1 * seq

        scanned, _ = scan(
            fn=scanStep, sequences=[seq], outputs_info=[to_scan], non_sequences=[f1]
        )
        function(inputs=[to_scan, seq, f1], outputs=scanned, allow_input_downcast=True)

        t_grad = grad(scanned.sum(), wrt=[to_scan, f1], consider_constant=[seq])
        function(inputs=[to_scan, seq, f1], outputs=t_grad, allow_input_downcast=True)

    def test_save_mem(self):
        rng = np.random.RandomState(utt.fetch_seed())

        vW_in2 = asarrayX(rng.uniform(-0.5, 0.5, size=(2,)))
        vW = asarrayX(rng.uniform(-0.5, 0.5, size=(2, 2)))
        vWout = asarrayX(rng.uniform(-0.5, 0.5, size=(2,)))
        vW_in1 = asarrayX(rng.uniform(-0.5, 0.5, size=(2, 2)))
        v_u1 = asarrayX(rng.uniform(-0.5, 0.5, size=(8, 2)))
        v_u2 = asarrayX(rng.uniform(-0.5, 0.5, size=(8,)))
        v_x0 = asarrayX(rng.uniform(-0.5, 0.5, size=(2,)))
        v_y0 = asarrayX(rng.uniform(size=(3,)))

        W_in2 = shared(vW_in2, name="win2")
        W = shared(vW, name="w")
        W_out = shared(vWout, name="wout")
        W_in1 = matrix("win")
        u1 = matrix("u1")
        u2 = vector("u2")
        x0 = vector("x0")
        y0 = vector("y0")

        def f_rnn_cmpl(u1_t, u2_t, x_tm1, y_tm1, y_tm3, W_in1):
            return [
                y_tm3 + 1,
                dot(u1_t, W_in1) + u2_t * W_in2 + dot(x_tm1, W),
                y_tm1 + dot(x_tm1, W_out),
            ]

        _outputs, updates = scan(
            f_rnn_cmpl,
            [u1, u2],
            [None, dict(initial=x0), dict(initial=y0, taps=[-1, -3])],
            W_in1,
            n_steps=None,
            truncate_gradient=-1,
            go_backwards=False,
        )
        outputs = [_outputs[0][-1], _outputs[1][-1], _outputs[2][-1]]
        f4 = function(
            [u1, u2, x0, y0, W_in1], outputs, updates=updates, allow_input_downcast=True
        )

        # compute the values in numpy
        v_x = np.zeros((8, 2), dtype=config.floatX)
        v_y = np.zeros((8,), dtype=config.floatX)
        v_x[0] = np.dot(v_u1[0], vW_in1) + v_u2[0] * vW_in2 + np.dot(v_x0, vW)
        v_y[0] = np.dot(v_x0, vWout) + v_y0[2]

        for i in range(1, 8):
            v_x[i] = np.dot(v_u1[i], vW_in1) + v_u2[i] * vW_in2 + np.dot(v_x[i - 1], vW)
            v_y[i] = np.dot(v_x[i - 1], vWout) + v_y[i - 1]

        (aesara_dump, aesara_x, aesara_y) = f4(v_u1, v_u2, v_x0, v_y0, vW_in1)

        utt.assert_allclose(aesara_x, v_x[-1:])
        utt.assert_allclose(aesara_y, v_y[-1:])

    def caching_nsteps_by_scan_op(self):
        W = matrix("weights")
        initial = vector("initial")
        inpt = matrix("inpt")

        def one_step(x_t, h_tm1, W):
            expr = dot(h_tm1, W) + x_t
            return expr

        expr, _ = scan(
            fn=one_step, sequences=[inpt], outputs_info=[initial], non_sequences=[W]
        )

        v1 = shared(np.ones(5, dtype=config.floatX))
        v2 = shared(np.ones((5, 5), dtype=config.floatX))
        shapef = function([W], expr, givens=OrderedDict([(initial, v1), (inpt, v2)]))
        # First execution to cache n_steps
        shapef(np.ones((5, 5), dtype=config.floatX))

        cost = expr.sum()
        d_cost_wrt_W = grad(cost, [W])
        f = function(
            [W, inpt],
            d_cost_wrt_W,
            givens=OrderedDict([(initial, shared(np.zeros(5)))]),
        )

        rval = np.asarray([[5187989] * 5] * 5, dtype=config.floatX)
        arg1 = np.ones((5, 5), dtype=config.floatX)
        arg2 = np.ones((10, 5), dtype=config.floatX)
        utt.assert_allclose(f(arg1, arg2), rval)

    def test_save_mem_reduced_number_of_steps(self):
        def f_rnn(u_t):
            return (
                u_t + 1.0,
                u_t + 2.0,
                u_t + 3.0,
                u_t + 4.0,
                u_t + 5.0,
                u_t + 6.0,
                u_t + 7.0,
            )

        u = vector("u")
        idx = iscalar("idx")
        jdx = iscalar("jdx")
        [x1, x2, x3, x4, x5, x6, x7], updates = scan(
            f_rnn, u, n_steps=None, truncate_gradient=-1, go_backwards=False
        )

        f2 = function(
            [u, idx, jdx],
            [x1[:2], x2[4], x3[idx], x4[:idx], x5[-10], x6[-jdx], x7[:-jdx]],
            updates=updates,
            allow_input_downcast=True,
        )
        # get random initial values
        rng = np.random.RandomState(utt.fetch_seed())
        v_u = rng.uniform(-5.0, 5.0, size=(20,))

        # compute the output in numpy
        tx1, tx2, tx3, tx4, tx5, tx6, tx7 = f2(v_u, 3, 15)

        utt.assert_allclose(tx1, v_u[:2] + 1.0)
        utt.assert_allclose(tx2, v_u[4] + 2.0)
        utt.assert_allclose(tx3, v_u[3] + 3.0)
        utt.assert_allclose(tx4, v_u[:3] + 4.0)
        utt.assert_allclose(tx5, v_u[-10] + 5.0)
        utt.assert_allclose(tx6, v_u[-15] + 6.0)
        utt.assert_allclose(tx7, v_u[:-15] + 7.0)

        # Maybe ugly, way to check if the optimization had
        # been applied

    def test_save_mem_store_steps(self):
        def f_rnn(u_t, x1_tm1, x1_tm3, x2_tm1, x3tm2, x3_tm1, x4_tm1):
            return (
                u_t + 1.0,
                u_t + 2.0,
                u_t + 3.0,
                u_t + 4.0,
                u_t + 5.0,
                u_t + 6.0,
                u_t + 7.0,
            )

        u = vector("u")
        x10 = vector("x10")
        x20 = scalar("x20")
        x30 = vector("x30")
        x40 = scalar("x40")
        [x1, x2, x3, x4, x5, x6, x7], updates = scan(
            f_rnn,
            u,
            [
                None,
                None,
                None,
                dict(initial=x10, taps=[-1, -2]),
                x20,
                dict(initial=x30, taps=[-1, -2]),
                x40,
            ],
            n_steps=None,
            truncate_gradient=-1,
            go_backwards=False,
        )

        f2 = function(
            [u, x10, x20, x30, x40],
            [x1[-7], x2[-3:-1], x3[-6:], x4[-1], x5[-1]],
            updates=updates,
            allow_input_downcast=True,
        )

        # get random initial values
        rng = np.random.RandomState(utt.fetch_seed())
        v_u = rng.uniform(-5.0, 5.0, size=(20,))

        # compute the output in numpy
        tx1, tx2, tx3, tx4, tx5 = f2(v_u, [0, 0], 0, [0, 0], 0)

        utt.assert_allclose(tx1, v_u[-7] + 1.0)
        utt.assert_allclose(tx2, v_u[-3:-1] + 2.0)
        utt.assert_allclose(tx3, v_u[-6:] + 3.0)
        utt.assert_allclose(tx4, v_u[-1] + 4.0)
        utt.assert_allclose(tx5, v_u[-1] + 5.0)

    def test_use_scan_direct_output(self):
        # This test looks for a crash that happened when directly using the
        # recurrent output of a scan node instead of taking the result
        # returned by the scan() function

        # Obtain a compilation mode that will cause the test to fail if an
        # exception occurs in the optimization process
        with config.change_flags(on_opt_error="raise"):
            mode = get_default_mode()

        x = scalar()
        seq = vector()
        outputs_info = [x, aet.zeros_like(x)]
        (out1, out2), updates = scan(
            lambda a, b, c: (a + b, b + c),
            sequences=seq,
            outputs_info=outputs_info,
            mode=mode,
        )

        # Obtain a reference to the scan outputs before the subtensor and
        # compile a function with them as outputs
        assert isinstance(out1.owner.op, Subtensor)
        assert isinstance(out2.owner.op, Subtensor)
        out1_direct = out1.owner.inputs[0]
        out2_direct = out2.owner.inputs[0]
        fct = function([x, seq], [out1_direct[:-1], out2_direct[:-1]], mode=mode)

        # Test the function to ensure valid outputs
        init_value = 5.0
        seq_value = np.arange(4, dtype=config.floatX)
        output1, output2 = fct(init_value, seq_value)

        expected_output1 = [init_value]
        expected_output2 = [0]
        for i in seq_value[:-1]:
            expected_output2.append(expected_output1[-1] + expected_output2[-1])
            expected_output1.append(expected_output1[-1] + i)

        utt.assert_allclose(output1, expected_output1)
        utt.assert_allclose(output2, expected_output2)

    def test_use_scan_direct_output2(self):
        # This test looks for a crash that happened when directly using the
        # recurrent output of a scan node associated with a state with a
        # state with broadcastable dimensions

        x = dcol()
        seq = dcol()
        outputs_info = [x, aet.zeros_like(x)]
        (out1, out2), updates = scan(
            lambda a, b, c: (a + b, a + c), sequences=seq, outputs_info=outputs_info
        )

        # Obtain a reference to the scan outputs before the subtensor and
        # compile a function with them as outputs
        assert isinstance(out1.owner.op, Subtensor)
        assert isinstance(out2.owner.op, Subtensor)
        out1_direct = out1.owner.inputs[0]
        out2_direct = out2.owner.inputs[0]
        fct = function([x, seq], [out1_direct, out2_direct])

        # Test that the function returns valid outputs
        x_val = np.arange(0, 4)[:, None]
        seq_val = np.arange(4, 8)[:, None]

        out1, out2 = fct(x_val, seq_val)

        expected_out1 = np.zeros((5, 4, 1))
        expected_out2 = np.zeros((5, 4, 1))
        for i in range(4):
            expected_out2[i + 1] = expected_out2[i] + seq_val[i]
        for i in range(5):
            expected_out1[i] = expected_out2[i] + x_val

        utt.assert_allclose(out1, expected_out1)
        utt.assert_allclose(out2, expected_out2)

    def test_infer_shape(self):
        # Test for a crash in scan.infer_shape when using both
        # an until condition and random sampling in the inner function.

        x = scalar()
        srng = RandomStream(0)

        def inner_fct(previous_val):
            new_val = previous_val + srng.uniform()
            condition = until(previous_val > 5)
            return new_val, condition

        out, updates = scan(inner_fct, outputs_info=x, n_steps=10)

        g_out = grad(out.sum(), x)
        fct = function([x], [out, g_out])

        for i in range(-5, 5):
            output, g_output = fct(i)
            assert len(output) == g_output

    def test_infer_shape2(self):
        # Ensure that the shape inference can remove the Scan node in the
        # case of a complicated inner graph involving sequences and recurrent
        # states

        seq = lvector()
        sitsot_init = lscalar()
        mitsot_init = lvector()

        def step(seq1, sitsot_m1, mitsot_m2, mitsot_m1):
            # Every iteration, the sitsot state decreases and the mitsot state
            # increases such that their total value remains identical. This
            # is because this value will be used as the shape of a nitsot
            # output and the outputs of every iteration need to have the same
            # shape
            diff = mitsot_m1 + seq1
            next_mitsot_val = mitsot_m2 + diff
            next_sitsot_val = sitsot_m1 - diff
            nitsot_out = aet.alloc(
                np.asarray(0.0, "float32"), next_mitsot_val + next_sitsot_val
            )
            return next_sitsot_val, next_mitsot_val, nitsot_out

        out, updates = scan(
            fn=step,
            sequences=seq,
            outputs_info=[
                sitsot_init,
                {"initial": mitsot_init, "taps": [-2, -1]},
                None,
            ],
            n_steps=5,
        )

        f = function([seq, sitsot_init, mitsot_init], out[2].shape, mode="FAST_RUN")
        # When Scan.infer_shape will cover more case, there will no scan left.
        assert len(scan_nodes_from_fct(f)) == 1

        # This generate a scan crash during execution.
        # output_shape = f(np.arange(5), 5, [1, 2])
        # assert(all(output_shape == (5, 6)))

    # The following test will fail in DebugMode if there are
    # some problems in Scan.infer_shape
    def test_remove_stuff(self):
        x = vector("x")

        def lm(m):
            trng = RandomStream(utt.fetch_seed())
            return [
                2 * m + trng.uniform(-1.1, 1.1, dtype=config.floatX),
                m + trng.uniform(size=[3]),
            ]

        [o1, o2], updates = scan(
            lm,
            sequences=x,
            n_steps=None,
            truncate_gradient=-1,
            name="forward",
            go_backwards=False,
        )
        go1 = grad(o1.mean(), wrt=x)
        f = function(
            [x], go1, updates=updates, allow_input_downcast=True, mode=mode_with_opt
        )
        assert np.allclose(f([1, 2, 3]), 2.0 / 3)

        topo = f.maker.fgraph.toposort()
        # this new assert is here to test if scan_merging works ..
        nb_scan = len([n for n in topo if isinstance(n.op, Scan)])
        assert nb_scan == 1
        nb_shape_i = len([n for n in topo if isinstance(n.op, Shape_i)])
        if config.mode != "FAST_COMPILE":
            assert nb_shape_i == 1

    def test_merge(self):
        x = vector()
        y = vector()

        def sum(s):
            return s + 1

        sx, upx = scan(sum, sequences=[x])
        sy, upy = scan(sum, sequences=[y])

        f = function(
            [x, y], [sx, sy], mode=mode_with_opt.excluding("scanOp_pushout_seqs_ops")
        )
        topo = f.maker.fgraph.toposort()
        scans = [n for n in topo if isinstance(n.op, Scan)]
        assert len(scans) == 2

        sx, upx = scan(sum, sequences=[x], n_steps=2)
        sy, upy = scan(sum, sequences=[y], n_steps=3)

        f = function(
            [x, y], [sx, sy], mode=mode_with_opt.excluding("scanOp_pushout_seqs_ops")
        )
        topo = f.maker.fgraph.toposort()
        scans = [n for n in topo if isinstance(n.op, Scan)]
        assert len(scans) == 2

        sx, upx = scan(sum, sequences=[x], n_steps=4)
        sy, upy = scan(sum, sequences=[y], n_steps=4)

        f = function(
            [x, y], [sx, sy], mode=mode_with_opt.excluding("scanOp_pushout_seqs_ops")
        )
        topo = f.maker.fgraph.toposort()
        scans = [n for n in topo if isinstance(n.op, Scan)]
        assert len(scans) == 1

        sx, upx = scan(sum, sequences=[x])
        sy, upy = scan(sum, sequences=[x])

        f = function(
            [x], [sx, sy], mode=mode_with_opt.excluding("scanOp_pushout_seqs_ops")
        )
        topo = f.maker.fgraph.toposort()
        scans = [n for n in topo if isinstance(n.op, Scan)]
        assert len(scans) == 1

        sx, upx = scan(sum, sequences=[x])
        sy, upy = scan(sum, sequences=[x], mode="FAST_COMPILE")

        f = function(
            [x], [sx, sy], mode=mode_with_opt.excluding("scanOp_pushout_seqs_ops")
        )
        topo = f.maker.fgraph.toposort()
        scans = [n for n in topo if isinstance(n.op, Scan)]
        assert len(scans) == 1

        sx, upx = scan(sum, sequences=[x])
        sy, upy = scan(sum, sequences=[x], truncate_gradient=1)

        f = function(
            [x], [sx, sy], mode=mode_with_opt.excluding("scanOp_pushout_seqs_ops")
        )
        topo = f.maker.fgraph.toposort()
        scans = [n for n in topo if isinstance(n.op, Scan)]
        assert len(scans) == 2

    def test_merge_3scans(self):
        # This test checks a case where we have 3 scans, two of them
        # cannot be merged together, but the third one can be merged with
        # either.
        x = vector()
        y = vector()

        def sum(s):
            return s + 1

        sx, upx = scan(sum, sequences=[x], n_steps=4, name="X")
        # We need to use an expression of y rather than y so the toposort
        # comes up with the 'Y' scan last.
        sy, upy = scan(sum, sequences=[2 * y + 2], n_steps=4, name="Y")
        sz, upz = scan(sum, sequences=[sx], n_steps=4, name="Z")

        f = function(
            [x, y], [sy, sz], mode=mode_with_opt.excluding("scanOp_pushout_seqs_ops")
        )
        topo = f.maker.fgraph.toposort()
        scans = [n for n in topo if isinstance(n.op, Scan)]
        assert len(scans) == 2

        rng = np.random.RandomState(utt.fetch_seed())
        x_val = rng.uniform(size=(4,)).astype(config.floatX)
        y_val = rng.uniform(size=(4,)).astype(config.floatX)
        # Run it so DebugMode can detect optimization problems.
        f(x_val, y_val)

    def test_pushout_seqs(self):
        def init_predictive_output(inputs, targets, hyp, x_star, s_star):
            E = hyp.shape[0]

            def init_K(i, X, Y):
                XX = X.sum(1).reshape((X.shape[0], 1))
                K = XX + XX.T
                return K.sum()

            beta, K_updts = scan(
                init_K, sequences=aet.arange(E), non_sequences=[inputs, targets]
            )

            # mean
            def predict_mean_i(i, x_star, s_star, X, beta, h):
                n, D = shape(X)
                # rescale every dimension by the corresponding inverse lengthscale
                iL = aet.diag(h[i, :D])
                inp = (X - x_star).dot(iL)

                # compute the mean
                B = iL.dot(s_star).dot(iL)
                t = inp.dot(B)

                lb = (inp * t).sum() + beta.sum()

                Mi = aet_sum(lb) * h[i, D]
                return Mi

            (M), M_updts = scan(
                predict_mean_i,
                sequences=aet.arange(E),
                non_sequences=[x_star, s_star, inputs, beta, hyp],
            )
            return M

        # some initializations
        hypx = np.log(np.tile([1, 1, 1, 1, 1, 1, 0.01], (3, 1)))

        # variables used in the following expressions
        hyp = shared(hypx)
        inputs = dmatrix("X")
        targets = dmatrix("Y")
        x_star = dvector("x_star")
        s_star = dmatrix("s_star")

        M = init_predictive_output(inputs, targets, hyp, x_star, s_star)

        X = np.random.random((10, 4))
        Y = np.random.random((10, 3))
        test_m = np.random.random((4,))
        test_s = np.eye(4)

        # Compute expected outputs (jacobian of M wrt x_star)
        dfdm = function(
            [inputs, targets, x_star, s_star],
            [
                grad(M[0], x_star),
                grad(M[1], x_star),
                grad(M[2], x_star),
            ],
        )
        expected_output = dfdm(X, Y, test_m, test_s)

        # equivalent code for the jacobian using scan
        dMdm, dMdm_updts = scan(
            lambda i, M, x: grad(M[i], x),
            sequences=aet.arange(M.shape[0]),
            non_sequences=[M, x_star],
        )
        dfdm = function([inputs, targets, x_star, s_star], [dMdm[0], dMdm[1], dMdm[2]])
        scan_output = dfdm(X, Y, test_m, test_s)

        dMdm_j = jacobian(M, x_star)
        dfdm_j = function(
            [inputs, targets, x_star, s_star], [dMdm_j[0], dMdm_j[1], dMdm_j[2]]
        )
        jacobian_outputs = dfdm_j(X, Y, test_m, test_s)

        utt.assert_allclose(expected_output, scan_output)
        utt.assert_allclose(expected_output, jacobian_outputs)

    @config.change_flags(on_opt_error="raise")
    def test_pushout_seqs2(self):
        # This test for a bug with PushOutSeqScan that was reported on the
        # theano-user mailing list where the optimization raised an exception
        # when applied on this graph.
        x = matrix()
        outputs, updates = scan(
            lambda x: [x * x, aet.constant(0).copy().copy()],
            n_steps=2,
            sequences=[],
            non_sequences=[],
            outputs_info=[x, None],
        )

        # Compile an Aesara function where any optimization error will lead to
        # an exception being raised
        function([x], outputs, updates=updates)

    @config.change_flags(on_opt_error="raise")
    def test_pushout_nonseq(self):
        # Test case originally reported by Daniel Renshaw. The crashed occurred
        # during the optimization PushOutNonSeqScan when it attempted to
        # a scan node with two outputs but only providing a replacement for
        # one of those outputs. This led the optimization to raise an
        # exception.

        outputs, _ = scan(lambda x: (x * x, x), non_sequences=[2], n_steps=2)
        f = function(inputs=[], outputs=outputs)

        outs = f()
        expected_outs = [[4, 4], [2, 2]]
        utt.assert_allclose(outs, expected_outs)

    def test_sequence_dict(self):
        # Test that we can specify sequences as a dictionary with
        # only the 'input' key
        def incr(s):
            return s + 1

        x = vector()
        sx, upx = scan(fn=incr, sequences=[{"input": x}])
        function([x], sx)

    def test_hash(self):
        x = vector()
        y = vector()
        scan1, updates = scan(lambda _x: _x + 1, x)
        scan2, updates = scan(lambda _x: _x + 1, y)
        assert scan1.owner.op == scan2.owner.op
        assert hash(scan1.owner.op) == hash(scan2.owner.op)

    def test_same(self):
        # This test is checking a bug discovered by Arnaud and it is based
        # on his code
        x = fmatrix("x")

        mem_val = np.zeros((2,), dtype="float32")
        memory = shared(mem_val)
        W = shared(np.random.random((5, 2)).astype("float32"))

        def f(inp, mem):
            i = aet.join(0, inp, mem)
            d = dot(i, W)
            return d, d

        outs, updts = scan(
            f, sequences=[x], non_sequences=[], outputs_info=[None, memory]
        )

        f = function([x], outs[0])
        f2 = function([x], outs[1])

        x_val = np.random.random((4, 3)).astype("float32")

        f_vals = f(x_val)
        memory.set_value(mem_val)
        f2_vals = f2(x_val)
        utt.assert_allclose(f_vals, f2_vals)

    def test_reduce_memory_consumption(self):

        x = shared(np.asarray(np.random.uniform(size=(10,)), dtype=config.floatX))
        o, _ = aet_reduce(
            lambda v, acc: acc + v,
            x,
            aet.constant(np.asarray(0.0, dtype=config.floatX)),
        )
        mode = FAST_RUN
        mode = mode.excluding("inplace")
        f1 = function([], o, mode=mode)
        inputs, outputs = clone_optimized_graph(f1)

        scan_nodes = grab_scan_node(outputs[0])
        assert scan_nodes is not None
        scan_node = scan_nodes[0]
        f1 = function(inputs, scan_node.inputs[2])

        # Originally, the shape would have been 1 due to the SaveMem
        # optimization reducing the size to the number of taps (in this case
        # 1) provided to the inner function. Now, because of the memory-reuse
        # feature in Scan it can be 2 because SaveMem needs to keep a
        # larger buffer to avoid aliasing between the inputs and the outputs.
        if config.scan__allow_output_prealloc:
            assert f1().shape[0] == 2
        else:
            assert f1().shape[0] == 1

        gx = grad(o, x)
        f2 = function([], gx)
        utt.assert_allclose(f2(), np.ones((10,)))

    def test_foldl_memory_consumption(self):
        x = shared(np.asarray(np.random.uniform(size=(10,)), dtype=config.floatX))
        o, _ = foldl(
            lambda v, acc: acc + v,
            x,
            aet.constant(np.asarray(0.0, dtype=config.floatX)),
        )

        mode = FAST_RUN
        mode = mode.excluding("inplace")
        f0 = function([], o, mode=mode)
        inputs, outputs = clone_optimized_graph(f0)

        scan_nodes = grab_scan_node(outputs[0])
        assert scan_nodes is not None
        scan_node = scan_nodes[0]
        f1 = function(inputs, scan_node.inputs[2])

        # Originally, the shape would have been 1 due to the SaveMem
        # optimization reducing the size to the number of taps (in this case
        # 1) provided to the inner function. Now, because of the memory-reuse
        # feature in Scan it can be 2 because SaveMem needs to keep a
        # larger buffer to avoid aliasing between the inputs and the outputs.
        if config.scan__allow_output_prealloc:
            assert f1().shape[0] == 2
        else:
            assert f1().shape[0] == 1

        gx = grad(o, x)
        f2 = function([], gx)
        utt.assert_allclose(f2(), np.ones((10,)))

    def test_foldr_memory_consumption(self):

        x = shared(np.asarray(np.random.uniform(size=(10,)), dtype=config.floatX))
        o, _ = foldr(
            lambda v, acc: acc + v,
            x,
            aet.constant(np.asarray(0.0, dtype=config.floatX)),
        )

        mode = FAST_RUN
        mode = mode.excluding("inplace")
        f1 = function([], o, mode=mode)
        inputs, outputs = clone_optimized_graph(f1)

        scan_nodes = grab_scan_node(outputs[0])
        assert scan_nodes is not None
        scan_node = scan_nodes[0]
        f1 = function(inputs, scan_node.inputs[2])

        # Originally, the shape would have been 1 due to the SaveMem
        # optimization reducing the size to the number of taps (in this case
        # 1) provided to the inner function. Now, because of the memory-reuse
        # feature in Scan it can be 2 because SaveMem needs to keep a
        # larger buffer to avoid aliasing between the inputs and the outputs.
        if config.scan__allow_output_prealloc:
            assert f1().shape[0] == 2
        else:
            assert f1().shape[0] == 1

        gx = grad(o, x)
        f2 = function([], gx)
        utt.assert_allclose(f2(), np.ones((10,)))

    @pytest.mark.slow
    def test_rop2(self):
        seed = utt.fetch_seed()
        rng = np.random.RandomState(seed)
        floatX = config.floatX
        v_u = np.array(rng.uniform(size=(3, 5)) - 0.5, dtype=floatX)
        v_W = np.array(rng.uniform(size=(5, 5)) - 0.5, dtype=floatX)
        v_h0 = np.array(rng.uniform(size=(5,)) - 0.5, dtype=floatX)

        v_eu = np.array(rng.uniform(size=(3, 5)) - 0.5, dtype=floatX)
        v_eW = np.array(rng.uniform(size=(5, 5)) - 0.5, dtype=floatX)
        v_eh0 = np.array(rng.uniform(size=(5,)) - 0.5, dtype=floatX)

        def rnn_fn(_u, _y, _W):

            srng = RandomStream(seed)
            tmp_val = (
                _u + _y + srng.uniform(size=v_h0.shape) * np.asarray(1e-6, dtype=floatX)
            )
            sl_o = tanh(dot(_W, tmp_val))
            return sl_o, tmp_val

        u = matrix("U")
        h0 = vector("h0")
        W = matrix("W")

        _u = specify_shape(u, v_u.shape)
        _u.name = "_U"
        _h0 = specify_shape(h0, v_h0.shape)
        _h0.name = "_h0"
        _W = specify_shape(W, v_W.shape)
        _W.name = "_W"

        [o, _], _ = scan(
            rnn_fn,
            sequences=_u,
            outputs_info=[_h0, None],
            non_sequences=_W,
            name="rnn_fn",
        )
        o = o[-1]
        eu = matrix("eu")
        eh0 = vector("eh0")
        eW = matrix("eW")

        nwo_u = Rop(o, _u, eu)
        nwo_h0 = Rop(o, _h0, eh0)
        nwo_W = Rop(o, _W, eW)
        fn_rop = function(
            [u, h0, W, eu, eh0, eW], [nwo_u, nwo_h0, nwo_W, o], on_unused_input="ignore"
        )
        vnu, vnh0, vnW, vno = fn_rop(v_u, v_h0, v_W, v_eu, v_eh0, v_eW)

        n2o_u, _ = scan(
            lambda i, o, u, h0, W, eu: (grad(o[i], u) * eu).sum(),
            sequences=aet.arange(o.shape[0]),
            non_sequences=[o, u, h0, W, eu],
            name="jacobU",
        )

        n2o_h0, _ = scan(
            lambda i, o, u, h0, W, eh0: (grad(o[i], h0) * eh0).sum(),
            sequences=aet.arange(o.shape[0]),
            non_sequences=[o, u, h0, W, eh0],
            name="jacobh",
        )

        n2o_W, _ = scan(
            lambda i, o, u, h0, W, eW: (grad(o[i], W) * eW).sum(),
            sequences=aet.arange(o.shape[0]),
            non_sequences=[o, u, h0, W, eW],
            name="jacobW",
        )

        fn_test = function(
            [u, h0, W, eu, eh0, eW], [n2o_u, n2o_h0, n2o_W, o], on_unused_input="ignore"
        )

        tnu, tnh0, tnW, tno = fn_test(v_u, v_h0, v_W, v_eu, v_eh0, v_eW)
        utt.assert_allclose(vnu, tnu, atol=1e-6)
        utt.assert_allclose(vnh0, tnh0, atol=1e-6)
        utt.assert_allclose(vnW, tnW, atol=2e-6)

    def test_rop(self):
        seed = utt.fetch_seed()
        rng = np.random.RandomState(seed)
        floatX = config.floatX
        v_u = np.array(rng.uniform(size=(20, 5)), dtype=floatX)
        v_W = np.array(rng.uniform(size=(5, 5)), dtype=floatX)
        v_h0 = np.array(rng.uniform(size=(5,)), dtype=floatX)

        v_eu = np.array(rng.uniform(size=(20, 5)), dtype=floatX)
        v_eW = np.array(rng.uniform(size=(5, 5)), dtype=floatX)
        v_eh0 = np.array(rng.uniform(size=(5,)), dtype=floatX)

        def rnn_fn(_u, _y, _W):
            sl_o = tanh(dot(_W, (_u + _y)))
            return sl_o

        u = matrix("U")
        h0 = vector("h0")
        W = matrix("W")

        _u = specify_shape(u, v_u.shape)
        _u.name = "_U"
        _h0 = specify_shape(h0, v_h0.shape)
        _h0.name = "_h0"
        _W = specify_shape(W, v_W.shape)
        _W.name = "_W"

        o, _ = scan(
            rnn_fn, sequences=_u, outputs_info=_h0, non_sequences=_W, name="rnn_fn"
        )
        o = o[-1]
        eu = matrix("eu")
        eh0 = vector("eh0")
        eW = matrix("eW")

        nwo_u = Rop(o, _u, eu)
        nwo_h0 = Rop(o, _h0, eh0)
        nwo_W = Rop(o, _W, eW)
        fn_rop = function(
            [u, h0, W, eu, eh0, eW], [nwo_u, nwo_h0, nwo_W], on_unused_input="ignore"
        )

        n2o_u, _ = scan(
            lambda i, o, u, h0, W, eu: (grad(o[i], u) * eu).sum(),
            sequences=aet.arange(o.shape[0]),
            non_sequences=[o, u, h0, W, eu],
            name="jacobU",
        )

        n2o_h0, _ = scan(
            lambda i, o, u, h0, W, eh0: (grad(o[i], h0) * eh0).sum(),
            sequences=aet.arange(o.shape[0]),
            non_sequences=[o, u, h0, W, eh0],
            name="jacobh",
        )

        n2o_W, _ = scan(
            lambda i, o, u, h0, W, eW: (grad(o[i], W) * eW).sum(),
            sequences=aet.arange(o.shape[0]),
            non_sequences=[o, u, h0, W, eW],
            name="jacobW",
        )

        fn_test = function(
            [u, h0, W, eu, eh0, eW], [n2o_u, n2o_h0, n2o_W], on_unused_input="ignore"
        )

        vnu, vnh0, vnW = fn_rop(v_u, v_h0, v_W, v_eu, v_eh0, v_eW)
        tnu, tnh0, tnW = fn_test(v_u, v_h0, v_W, v_eu, v_eh0, v_eW)

        utt.assert_allclose(vnu, tnu, atol=1e-6)
        utt.assert_allclose(vnh0, tnh0, atol=1e-6)
        utt.assert_allclose(vnW, tnW, atol=1e-6)

    def test_pushout_dot(self):
        W = matrix("W")
        h = matrix("h")

        o, _ = scan(
            lambda hi, him1, W: (hi, dot(hi + him1, W)),
            outputs_info=[aet.zeros([h.shape[1]]), None],
            sequences=[h],
            non_sequences=[W],
        )

        f = function([W, h], o, mode=mode_with_opt)

        scan_nodes = [x for x in f.maker.fgraph.toposort() if isinstance(x.op, Scan)]
        assert len(scan_nodes) == 1
        scan_op = scan_nodes[0].op
        assert not any(
            isinstance(n.op, Dot) for n in scan_op.fn.maker.fgraph.apply_nodes
        )

    def test_pushout_all(self):
        W1 = matrix("W1")
        W2 = matrix("W2")
        h0 = vector("h0")

        def lambda_fn(h, W1, W2):
            return dot(h, W1 + W2)

        o, _ = scan(lambda_fn, non_sequences=[h0, W1, W2], n_steps=5)

        f = function([h0, W1, W2], o, mode=mode_with_opt)

        scan_nodes = [x for x in f.maker.fgraph.toposort() if isinstance(x.op, Scan)]
        assert len(scan_nodes) == 0

        seed = utt.fetch_seed()
        rng = np.random.RandomState(seed)
        floatX = config.floatX
        v_h = np.array(rng.uniform(size=(2,)), dtype=floatX)
        v_W1 = np.array(rng.uniform(size=(2, 2)), dtype=floatX)
        v_W2 = np.array(rng.uniform(size=(2, 2)), dtype=floatX)

        v_out = np.dot(v_h, v_W1 + v_W2)
        sol = np.zeros((5, 2))
        # This line is here to make sol have the same shape as the output of
        # aesara. Note that what we ask aesara to do is to repeat the 2
        # elements vector v_out 5 times
        sol[:, :] = v_out
        utt.assert_allclose(sol, f(v_h, v_W1, v_W2))

    def test_pushout_while(self):
        # Ensure that the optimizations for Scan that push computation out of
        # the Scan don't alter the result for 'as_while' scans.

        W1 = matrix("W1")
        W2 = matrix("W2")
        step_indices = vector("step_indices")

        def lambda_fn(step_idx, W1, W2):
            until_condition = until(step_idx > 2)
            return dot(W1, W2), until_condition

        # Compile a function with the optimization
        o, _ = scan(
            lambda_fn, sequences=[step_indices, W1], non_sequences=[W2], n_steps=5
        )

        f = function([W1, W2, step_indices], o, mode=mode_with_opt)

        # Compule an aesara function without the optimization
        o, _ = scan(
            lambda_fn,
            sequences=[step_indices, W1],
            non_sequences=[W2],
            n_steps=5,
            mode="FAST_COMPILE",
        )

        f_ref = function([W1, W2, step_indices], o, mode="FAST_COMPILE")

        # Compare the results of the two implementations
        input_values = [
            np.random.random((5, 5)).astype("float32"),
            np.random.random((5, 5)).astype("float32"),
            np.arange(5).astype("float32"),
        ]

        out = f(*input_values)
        out_ref = f_ref(*input_values)
        utt.assert_allclose(out, out_ref)

    def test_pushout(self):
        W1 = matrix("W1")
        W2 = matrix("W2")
        h0 = vector("h0")

        def lambda_fn(h, W1, W2):
            return dot(h, W1 + W2)

        o, _ = scan(lambda_fn, outputs_info=h0, non_sequences=[W1, W2], n_steps=5)

        f = function([h0, W1, W2], o, mode=mode_with_opt)

        scan_node = [x for x in f.maker.fgraph.toposort() if isinstance(x.op, Scan)][0]
        assert (
            len(
                [
                    x
                    for x in scan_node.op.fn.maker.fgraph.toposort()
                    if isinstance(x.op, Elemwise)
                ]
            )
            == 0
        )

    def test_pushout_nomodif(self):
        inp = matrix("inp")

        def fn(i, i_tm1):
            return i + 10, i_tm1

        ([i_t, i_tm1], _) = scan(
            fn,
            sequences=[inp],
            outputs_info=[np.asarray([0.0, 0.0], config.floatX), None],
        )
        f = function([inp], [i_t, i_tm1])
        val = np.arange(10).reshape(5, 2).astype(config.floatX)
        ret = f(val)
        utt.assert_allclose(ret[0], val + 10)
        utt.assert_allclose(
            ret[1], [[0.0, 0.0], [10.0, 11.0], [12.0, 13.0], [14.0, 15.0], [16.0, 17.0]]
        )

    def test_alloc_inputs1(self):
        W1 = matrix("W1")
        W2 = matrix("W2")
        h0 = vector("h0")

        def lambda_fn(h, W1, W2):
            return dot(h, W1 * W2)

        o, _ = scan(
            lambda_fn,
            outputs_info=h0,
            non_sequences=[W1, aet.zeros_like(W2)],
            n_steps=5,
        )

        f = function([h0, W1, W2], o, mode=mode_with_opt)
        scan_node = [x for x in f.maker.fgraph.toposort() if isinstance(x.op, Scan)][0]
        assert (
            len(
                [
                    x
                    for x in scan_node.op.fn.maker.fgraph.toposort()
                    if isinstance(x.op, Elemwise)
                ]
            )
            == 0
        )

    @pytest.mark.skip(
        reason="This tests depends on an optimization for "
        "scan that has not been implemented yet."
    )
    def test_alloc_inputs2(self):
        W1 = matrix()
        W2 = matrix()
        h0 = vector()

        def lambda_fn(W1, h, W2):
            return W1 * dot(h, W2)

        o, _ = scan(
            lambda_fn,
            sequences=aet.zeros_like(W1),
            outputs_info=h0,
            non_sequences=[aet.zeros_like(W2)],
            n_steps=5,
        )

        f = function([h0, W1, W2], o, mode=mode_with_opt)
        scan_node = [x for x in f.maker.fgraph.toposort() if isinstance(x.op, Scan)][0]

        assert (
            len(
                [
                    x
                    for x in scan_node.op.fn.maker.fgraph.toposort()
                    if isinstance(x.op, Elemwise)
                ]
            )
            == 0
        )

    def test_alloc_inputs3(self):
        _W1 = matrix()
        _W2 = matrix()
        _h0 = vector()

        W1 = specify_shape(_W1, (3, 3))
        W2 = specify_shape(_W2, (3, 3))
        h0 = specify_shape(_h0, (3,))

        def lambda_fn(W1, h, W2):
            return W1 * dot(h, W2)

        o, _ = scan(
            lambda_fn,
            sequences=aet.zeros_like(W1),
            outputs_info=h0,
            non_sequences=[aet.zeros_like(W2)],
            n_steps=5,
        )

        f = function([_h0, _W1, _W2], o, mode=mode_with_opt)
        scan_node = [x for x in f.maker.fgraph.toposort() if isinstance(x.op, Scan)][0]

        assert len(scan_node.op.inputs) == 1

    def test_while0(self):
        x = vector("x")

        def lambda_fn(x_t):
            return x_t + 1, until(x_t > 3)

        o, _ = scan(lambda_fn, x)
        f = function([x], o)
        vx = np.zeros((50,), dtype=config.floatX)
        vx[23] = 4
        out = f(vx)
        assert len(out) == 24

    def test_while1(self):
        x = vector("x")

        def lambda_fn(x_t):
            return x_t + 1, until(x_t > 3)

        o, _ = scan(lambda_fn, x)
        o2, _ = scan(lambda x_t: x_t + 2, x)

        f = function([x], [o, o2], mode=mode_with_opt)
        vx = np.zeros((50,), dtype=config.floatX)
        vx[23] = 4
        out, out2 = f(vx)
        assert len(out) == 24
        assert np.all(out2 == vx + 2)
        lssc = [x for x in f.maker.fgraph.toposort() if isinstance(x.op, Scan)]
        # One scan node gets optimnized out
        assert len(lssc) == 1

    @pytest.mark.skip(
        reason="This test fails because not typed outputs_info "
        "are always gived the smallest dtype. There is "
        "no upcast of outputs_info in scan for now.",
    )
    def test_outputs_info_not_typed(self):
        # This was ticket 766

        coefficients = vector("coefficients")
        x = scalar("x")
        max_coefficients_supported = 10000

        # Generate the components of the polynomial
        full_range = aet.arange(max_coefficients_supported)
        components, updates = scan(
            fn=lambda coeff, power, free_var: coeff * (free_var ** power),
            sequences=[coefficients, full_range],
            non_sequences=x,
        )
        polynomial1 = components.sum()
        polynomial2, updates = scan(
            fn=lambda coeff, power, prev, free_var: prev + coeff * (free_var ** power),
            outputs_info=aet.constant(0, dtype="floatX"),
            sequences=[coefficients, full_range],
            non_sequences=x,
        )

        # python int
        polynomial3, updates = scan(
            fn=lambda coeff, power, prev, free_var: prev + coeff * (free_var ** power),
            outputs_info=0,
            sequences=[coefficients, full_range],
            non_sequences=x,
        )

        # python float
        polynomial4, updates = scan(
            fn=lambda coeff, power, prev, free_var: prev + coeff * (free_var ** power),
            outputs_info=0.0,
            sequences=[coefficients, full_range],
            non_sequences=x,
        )

        calculate_polynomial = function(
            inputs=[coefficients, x],
            outputs=[polynomial1, polynomial2[-1], polynomial3[-1], polynomial4[-1]],
        )

        test_coeff = np.asarray([1, 0, 2], dtype=config.floatX)
        # This will be tested by DEBUG_MODE
        out = calculate_polynomial(test_coeff, 3)
        assert out[0] == 19
        assert out[1] == 19
        assert out[2] == 19
        assert out[4] == 19
        # 19.0

    def test_crash_nonseq_grad(self):
        # Test case was originally reported by Bitton Tenessi. It crashed
        # during the grad operation and this tests validates that it now
        # raises a NullTypeGradError instead because the gradient relies on
        # the intermediary states of the random number generators used in the
        # test. The test case was modified from the original for simplicity

        rand_stream = RandomStream()
        inp = matrix()
        norm_inp = inp / aet_sum(inp, axis=0)

        def unit_dropout(out_idx):
            def stochastic_pooling(in_idx):
                # sample the input matrix for each column according to the
                # column values
                pvals = norm_inp.T
                sample = rand_stream.multinomial(1, pvals)
                return inp + sample

            pooled, updates_inner = scan(
                fn=stochastic_pooling, sequences=aet.arange(inp.shape[0])
            )

            # randomly add stuff to units
            rand_nums = rand_stream.binomial(1, 0.5, size=pooled.shape)
            return pooled + rand_nums, updates_inner

        out, updates_outer = scan(unit_dropout, sequences=[aet.arange(inp.shape[0])])

        with pytest.raises(NullTypeGradError):
            grad(out.sum(), inp)

    def test_bugFunctioProvidesIntermediateNodesAsInputs(self):
        # This is a bug recently reported by Ilya
        # made it CPU friendly
        V = ftensor3("INPUT")
        orig = fmatrix("PARAM")
        # = gpu_from_host(orig)  # <-- this doesn't work
        W = orig + 2  # <-- has same effect but it works on CPU as well
        # W = T.fmatrix('PARAM') # <-- this line works

        def one_step(v, W):
            o = v + 1 + W.sum()  # <-- this doesn't work
            # o = v + 1  # <-- this line works
            return o

        OS, updates = scan(
            fn=one_step, sequences=V, outputs_info=[None], non_sequences=[W]
        )

        O = OS.sum() + W.sum()

        # This bug manifests itself by not allowing the function to compile,
        # so if it compiles it means the test pass
        function([V, W], O)

    def test_while2(self):
        x = vector("x")

        def lambda_fn(x_t):
            return x_t + 1, until(x_t > 3)

        o, _ = scan(lambda_fn, x)
        o2, _ = scan(lambda x_t: (x_t + 2, until(x_t > 3)), x)

        f = function([x], [o, o2], mode=mode_with_opt)
        vx = np.zeros((50,), dtype=config.floatX)
        vx[23] = 4
        out, out2 = f(vx)
        assert len(out) == 24
        assert len(out2) == 24
        lssc = [x for x in f.maker.fgraph.toposort() if isinstance(x.op, Scan)]
        assert len(lssc) == 1

    def test_while_infershape(self):
        x = vector("x")

        def lambda_fn(x_t):
            return x_t + 1, until(x_t > 3)

        o, _ = scan(lambda_fn, x)

        f = function([x], o.shape[0], mode=mode_with_opt)
        vx = np.zeros((50,), dtype=config.floatX)
        vx[23] = 4
        out = f(vx)
        assert out == 24

    @pytest.mark.skip(
        reason="This is a generic problem with "
        "infershape that has to be discussed "
        "and figured out"
    )
    def test_infershape_seq_shorter_nsteps(self):
        x = vector("x")
        [o1, o2], _ = scan(
            lambda x, y: (x + 1, y + x),
            sequences=x,
            outputs_info=[None, x[0]],
            n_steps=20,
        )

        f = function([x], [o1.shape[0], o2.shape[0]], mode=mode_with_opt)

        vx = np.ones((10,), dtype=config.floatX)
        out1, out2 = f(vx)
        assert out1 == 10
        assert out2 == 10
        lssc = [x for x in f.maker.fgraph.toposort() if isinstance(x.op, Scan)]
        assert len(lssc) == 0

    def test_infershape_nsteps_smaller_seq_length(self):
        x = vector("x")
        [o1, o2], _ = scan(
            lambda x, y: (x + 1, y + x),
            sequences=x,
            outputs_info=[None, x[0]],
            n_steps=20,
        )

        f = function([x], [o1.shape[0], o2.shape[0]], mode=mode_with_opt)

        vx = np.ones((30,), dtype=config.floatX)
        o1, o2 = f(vx)
        assert o1 == 20
        assert o2 == 20
        lssc = [x for x in f.maker.fgraph.toposort() if isinstance(x.op, Scan)]
        assert len(lssc) == 0

    def test_oinp_iinp_iout_oout_mappings(self):
        # Test the mapping produces by
        # ScanOp.get_oinp_iinp_iout_oout_mappings()

        rng = RandomStream(123)

        def inner_fct(seq, mitsot, sitsot, nitsot, nseq):
            random_scalar = rng.uniform((1,))[0]
            total = seq + mitsot + sitsot + nitsot + nseq + random_scalar
            return total, total, total

        # Assemble a scan with one sequence, one mitsot, one sitsot, one nitsot
        # a non-sequence and a random state to test the mappings.
        seq = [vector()]
        non_seq = [scalar()]
        outputs_info = [
            dict(initial=vector(), taps=[-3, -1]),
            scalar(),
            None,
        ]

        scan_outputs, _ = scan(
            fn=inner_fct,
            sequences=seq,
            outputs_info=outputs_info,
            non_sequences=non_seq,
        )

        # Compare the mappings with the expected values
        scan_node = scan_outputs[0].owner.inputs[0].owner
        mappings = scan_node.op.var_mappings

        assert mappings["inner_inp_from_outer_inp"] == {
            0: [],
            1: [0],
            2: [1, 2],
            3: [3],
            4: [4],
            5: [],
            6: [5],
        }
        assert mappings["inner_out_from_outer_inp"] == {
            0: [],
            1: [],
            2: [0],
            3: [1],
            4: [3],
            5: [2],
            6: [],
        }
        assert mappings["outer_out_from_outer_inp"] == {
            0: -1,
            1: -1,
            2: 0,
            3: 1,
            4: 3,
            5: 2,
            6: -1,
        }

        assert mappings["outer_inp_from_inner_inp"] == {
            0: 1,
            1: 2,
            2: 2,
            3: 3,
            4: 4,
            5: 6,
        }
        assert mappings["inner_out_from_inner_inp"] == {
            0: [],
            1: [0],
            2: [0],
            3: [1],
            4: [3],
            5: [],
        }
        assert mappings["outer_out_from_inner_inp"] == {
            0: -1,
            1: 0,
            2: 0,
            3: 1,
            4: 3,
            5: -1,
        }

        assert mappings["outer_inp_from_inner_out"] == {0: 2, 1: 3, 2: 5, 3: 4}
        assert mappings["inner_inp_from_inner_out"] == {
            0: [1, 2],
            1: [3],
            2: [],
            3: [4],
        }
        assert mappings["outer_out_from_inner_out"] == {0: 0, 1: 1, 2: 2, 3: 3}

        assert mappings["outer_inp_from_outer_out"] == {0: 2, 1: 3, 2: 5, 3: 4}
        assert mappings["inner_inp_from_outer_out"] == {
            0: [1, 2],
            1: [3],
            2: [],
            3: [4],
        }
        assert mappings["inner_out_from_outer_out"] == {0: [0], 1: [1], 2: [2], 3: [3]}

    def test_grad_duplicate_outputs(self):
        # This test validates that taking the gradient of a scan, in which
        # multiple outputs are the same aesara variable, works.

        def inner_fct(inp1, inp2, inp3):
            total = inp1 + inp2 + inp3
            return total, total

        # Assemble the scan
        seq = matrix()
        out_init = matrix()
        non_seq = vector()

        outputs_info = [None, dict(initial=out_init, taps=[-3])]

        scan_outputs, _ = scan(
            fn=inner_fct,
            sequences=seq,
            outputs_info=outputs_info,
            non_sequences=non_seq,
        )

        # Attempt to take various gradients
        g_output0 = grad(scan_outputs[0].sum(), [seq, out_init, non_seq])
        g_output1 = grad(scan_outputs[1].sum(), [seq, out_init, non_seq])

        # Compile the function
        fct = function([seq, out_init, non_seq], g_output0 + g_output1)

        # Run the function and validate the outputs
        dtype = config.floatX
        seq_value = np.random.random((10, 3)).astype(dtype)
        out_init_value = np.random.random((3, 3)).astype(dtype)
        non_seq_value = np.random.random(3).astype(dtype)

        outputs = fct(seq_value, out_init_value, non_seq_value)

        expected_g_seq = np.array(
            [
                [4, 4, 4],
                [3, 3, 3],
                [3, 3, 3],
                [3, 3, 3],
                [2, 2, 2],
                [2, 2, 2],
                [2, 2, 2],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
            ]
        )
        expected_g_out_init = expected_g_seq[:3]
        expected_g_non_seq = np.array([22, 22, 22])

        utt.assert_allclose(outputs[0], expected_g_seq)
        utt.assert_allclose(outputs[1], expected_g_out_init)
        utt.assert_allclose(outputs[2], expected_g_non_seq)
        utt.assert_allclose(outputs[3], expected_g_seq)
        utt.assert_allclose(outputs[4], expected_g_out_init)
        utt.assert_allclose(outputs[5], expected_g_non_seq)

    def test_grad_duplicate_outputs_connection_pattern(self):
        # This test checks for a crash in scan.connection_pattern when taking
        # the grad of a scan with certain combinations of outputs.

        def inner_fct(inp1, inp2, inp3, inp4, inp5, inp6):
            total = inp1 + inp2 + inp3 + inp4 + inp5 + inp6
            return total, total, total, total, total, total

        # Assemble the scan
        out_init = [vector(), vector(), matrix(), matrix()]

        outputs_info = [
            None,
            None,
            out_init[0],
            out_init[1],
            dict(initial=out_init[2], taps=[-2, -1]),
            dict(initial=out_init[3], taps=[-2, -1]),
        ]

        scan_outputs, _ = scan(fn=inner_fct, outputs_info=outputs_info, n_steps=10)

        grad(scan_outputs[0].sum(), out_init[1])

        # Validate the connnection pattern is as it should be
        node = scan_outputs[0].owner
        connection_pattern = node.op.connection_pattern(node)
        expected_connection_pattern = [
            [(j in [1, 2, 3, 4]) for i in range(6)] for j in range(7)
        ]

        assert connection_pattern == expected_connection_pattern

    def test_grad_multiple_seqs_different_nsteps(self):
        # Example provided Michael Forbes
        # This test assures that we clip the sequences to n_steps before
        # computing the gradient (so that when we reverse them we actually
        # get the right values in
        c = vector("c")
        x = scalar("x")
        _max_coefficients_supported = 1000
        full_range = aet.arange(_max_coefficients_supported)
        components, updates = scan(
            fn=lambda coeff, power, free_var: coeff * (free_var ** power),
            outputs_info=None,
            sequences=[c, full_range],
            non_sequences=x,
        )
        P = components.sum()
        dP = grad(P, x)
        tf = function([c, x], dP)
        assert tf([1.0, 2.0, -3.0, 4.0], 2.0) == 38

    def test_grad_of_grad_of_state(self):
        # Example provided Michael Forbes
        # This tests ensures that we can compute gradients through cost
        # defines in terms of gradients of scan
        c = vector("c")
        x = scalar("x")
        _max_coefficients_supported = 1000
        full_range = aet.arange(_max_coefficients_supported)
        components, updates = scan(
            fn=lambda coeff, power, free_var: coeff * (free_var ** power),
            outputs_info=None,
            sequences=[c, full_range],
            non_sequences=x,
        )
        P = components.sum()
        dP = grad(P, x).sum()
        ddP = grad(dP, x)
        tf = function([c, x], ddP)
        assert tf([1.0, 2.0, -3.0, 4.0], 2.0) == 42

    def test_return_steps(self):
        rng = np.random.RandomState(utt.fetch_seed())

        vW_in2 = asarrayX(rng.uniform(-0.5, 0.5, size=(2,)))
        vW = asarrayX(rng.uniform(-0.5, 0.5, size=(2, 2)))
        vWout = asarrayX(rng.uniform(-0.5, 0.5, size=(2,)))
        vW_in1 = asarrayX(rng.uniform(-0.5, 0.5, size=(2, 2)))
        v_u1 = asarrayX(rng.uniform(-0.5, 0.5, size=(8, 2)))
        v_u2 = asarrayX(rng.uniform(-0.5, 0.5, size=(8,)))
        v_x0 = asarrayX(rng.uniform(-0.5, 0.5, size=(2,)))
        v_y0 = asarrayX(rng.uniform(size=(3,)))

        W_in2 = shared(vW_in2, name="win2")
        W = shared(vW, name="w")
        W_out = shared(vWout, name="wout")
        W_in1 = matrix("win")
        u1 = matrix("u1")
        u2 = vector("u2")
        x0 = vector("x0")
        y0 = vector("y0")

        def f_rnn_cmpl(u1_t, u2_t, x_tm1, y_tm1, y_tm3, W_in1):
            return [
                y_tm3 + 1,
                dot(u1_t, W_in1) + u2_t * W_in2 + dot(x_tm1, W),
                y_tm1 + dot(x_tm1, W_out),
            ]

        rval, updates = scan(
            f_rnn_cmpl,
            [u1, u2],
            [None, dict(initial=x0), dict(initial=y0, taps=[-1, -3])],
            W_in1,
            n_steps=None,
            truncate_gradient=-1,
            go_backwards=False,
        )

        outputs = []
        outputs += [rval[0][-3:]]
        outputs += [rval[1][-2:]]
        outputs += [rval[2][-4:]]
        f4 = function(
            [u1, u2, x0, y0, W_in1], outputs, updates=updates, allow_input_downcast=True
        )

        # compute the values in numpy
        v_x = np.zeros((8, 2), dtype=config.floatX)
        v_y = np.zeros((8,), dtype=config.floatX)
        v_x[0] = np.dot(v_u1[0], vW_in1) + v_u2[0] * vW_in2 + np.dot(v_x0, vW)
        v_y[0] = np.dot(v_x0, vWout) + v_y0[2]

        for i in range(1, 8):
            v_x[i] = np.dot(v_u1[i], vW_in1) + v_u2[i] * vW_in2 + np.dot(v_x[i - 1], vW)
            v_y[i] = np.dot(v_x[i - 1], vWout) + v_y[i - 1]

        (aesara_dump, aesara_x, aesara_y) = f4(v_u1, v_u2, v_x0, v_y0, vW_in1)

        utt.assert_allclose(aesara_x, v_x[-2:])
        utt.assert_allclose(aesara_y, v_y[-4:])

    def test_opt_order(self):
        # Verify that scan optimizations are applied before blas
        # optimizations.
        # This is needed as otherwise, the dot won't become a dot22
        # so it will be slower and won't get transferred to the gpu.

        x = matrix("x")
        A = matrix("A")

        z, updates = scan(dot, sequences=[], non_sequences=[x, A], n_steps=2)
        f = function([x, A], z)
        topo = f.maker.fgraph.toposort()
        if config.mode != "FAST_COMPILE":
            assert any([isinstance(node.op, Dot22) for node in topo])

        vx = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=config.floatX)
        vA = np.array([[1.0, 1.0], [1.0, 0.0]], dtype=config.floatX)
        vR = np.array([[[2, 1], [4, 2]], [[2, 1], [4, 2]]], dtype=config.floatX)
        utt.assert_allclose(f(vx, vA), vR)

    def test_savemem_opt(self):
        y0 = shared(np.ones((2, 10)))
        [y1, y2], updates = scan(
            lambda y: [y, y],
            outputs_info=[dict(initial=y0, taps=[-2]), None],
            n_steps=5,
        )
        function([], y2.sum())()

    def test_savemem_opt_0_step(self):
        # Test a case where the savemem optimization has the opportunity to
        # lower the number of steps of a Scan to 0. It tests that the
        # optimization doesn't do so since Scan nodes with 0
        # steps are not currently supported and doing so would result in a
        # crash during the function execution.

        def inner_scan_step(x_t_t, h_tm1, w):
            return dot(h_tm1, w) + x_t_t

        def outer_scan_step(x_t, w):
            h, _ = scan(
                inner_scan_step,
                sequences=[x_t[1:]],
                outputs_info=[x_t[0]],
                non_sequences=[w],
                strict=True,
                name="the_inner_scan",
            )
            return h

        def get_outputs(x, w):
            features, _ = scan(
                outer_scan_step,
                sequences=[x],
                non_sequences=[w],
                strict=True,
                name="the_outer_scan",
            )

            return_val = grad(features.sum(), w)
            return return_val

        # Compile the aesara function
        x = tensor3("x")
        w = matrix("w")
        f = function(inputs=[x, w], outputs=get_outputs(x, w))

        # Test the function to ensure it returns valid results
        x_value = np.random.random((2, 2, 3)).astype(config.floatX)
        w_value = np.random.random((3, 3)).astype(config.floatX)
        expected_output = np.tile(x_value[:, 0].sum(0), (3, 1)).transpose()

        output = f(x_value, w_value)
        utt.assert_allclose(output, expected_output)

    def test_grad_multiple_taps_state(self):
        # The test is based on the code provided by Timothy Lillicrap

        def onestep(xdl, xprev, w):
            xnew = w + xprev
            return xnew

        xinit = tensor3("xinit")
        w = matrix("w")
        (xseq, updates) = scan(
            n_steps=10,
            fn=onestep,
            outputs_info=[dict(initial=xinit, taps=[-4, -1])],
            non_sequences=w,
        )
        loss = (xseq[-1] ** 2).sum()
        cost_fn = function(
            [xinit, w], loss, no_default_updates=True, allow_input_downcast=True
        )

        gw, gx = grad(loss, [w, xinit])
        grad_fn = function([xinit, w], [gx, gw], allow_input_downcast=True)
        rng = np.random.RandomState(utt.fetch_seed())
        # If numbers are small, the gradients with respect to x are small
        # and the numeric differentiation becomes unstable.
        # To fix this issue I ensure we are sampling numbers larger in
        # absolute value than 1.
        v_x = np.array(rng.uniform(1.0, 3.0, size=(5, 2, 2)), dtype=config.floatX)
        # Making some entries to be negative.
        pos = rng.uniform(0.0, 1, size=(5, 2, 2)) < 0.5
        v_x[pos] = -1 * v_x[pos]
        v_w = np.array(rng.uniform(1.0, 3.0, size=(2, 2)), dtype=config.floatX)
        pos = rng.uniform(0.0, 1.0, size=(2, 2)) < 0.5
        v_w[pos] = -1 * v_w[pos]
        analytic_grad = grad_fn(v_x, v_w)
        num_grad = multiple_outputs_numeric_grad(cost_fn, [v_x, v_w])
        max_err, max_err_pos = num_grad.max_err(analytic_grad)
        assert max_err <= 1e-2

    def test_grad_numeric_shared(self):
        shared_var = shared(np.float32(1.0))

        def inner_fn():
            return [], OrderedDict([(shared_var, shared_var + np.float32(1.0))])

        _, updates = scan(
            inner_fn, n_steps=10, truncate_gradient=-1, go_backwards=False
        )
        cost = list(updates.values())[0]
        g_sh = grad(cost, shared_var)
        fgrad = function([], g_sh)
        assert fgrad() == 1

    def test_rop_mitmot(self):
        # this test is a copy paste from the script given by Justin Bayer to
        # reproduce this bug
        # We have 2 parameter groups with the following shapes.
        W1shape = (1, 3)
        W2shape = (3, 3)

        n_pars = 1 * 3 + 3 * 3

        # Allocate big parameter array.
        pars = shared(np.empty(n_pars))

        # Assign slices.
        W1 = pars[:3].reshape(W1shape)
        W2 = pars[3:].reshape(W2shape)

        # Define recurrent model. We are using a model where each input is a
        # tensor
        # of shape (T, B, D) where T is the number of timesteps, B is the
        # number of
        # sequences iterated over in parallel and D is the dimensionality of
        # each
        # item at a timestep.

        inpt = tensor3("inpt")
        target = tensor3("target")

        # Make these flat in order to be able to use dot products instead of
        # tensordot,
        # which is slower.
        inpt_flat = inpt.reshape((inpt.shape[0] * inpt.shape[1], inpt.shape[2]))
        hidden_flat = dot(inpt_flat, W1)
        hidden = hidden_flat.reshape((inpt.shape[0], inpt.shape[1], 3))

        transfer = sigmoid

        hidden_rec, _ = scan(
            lambda x, h_tm1: transfer(dot(h_tm1, W2) + x),
            sequences=hidden,
            outputs_info=[aet.zeros_like(hidden[0])],
        )

        hidden_rec.reshape(
            (hidden_rec.shape[0] * hidden_rec.shape[1], hidden_rec.shape[2])
        )

        cost = ((hidden_rec - target) ** 2).mean()
        d_cost_wrt_pars = grad(cost, pars)

        p = dvector()
        Rop(d_cost_wrt_pars, pars, p)

    def test_seq_tap_bug_jeremiah(self):
        inp = np.arange(10).reshape(-1, 1).astype(config.floatX)
        exp_out = np.zeros((10, 1)).astype(config.floatX)
        exp_out[4:] = inp[:-4]

        def onestep(x, x_tm4):
            return x, x_tm4

        seq = matrix()
        initial_value = shared(np.zeros((4, 1), dtype=config.floatX))
        outputs_info = [OrderedDict([("initial", initial_value), ("taps", [-4])]), None]
        results, updates = scan(fn=onestep, sequences=seq, outputs_info=outputs_info)

        f = function([seq], results[1])
        assert np.all(exp_out == f(inp))

    def test_borrow_bug_jeremiah(self):
        # This tests two things. The first is a bug occurring when scan wrongly
        # used the borrow flag. The second thing it that Scan's infer_shape()
        # method will be able to remove the Scan node from the graph in this
        # case.

        inp = np.arange(10).reshape(-1, 1).astype(config.floatX)
        exp_out = np.zeros((10, 1)).astype(config.floatX)
        exp_out[4:] = inp[:-4]

        def onestep(x, x_tm4):
            return x, x_tm4

        seq = matrix()
        initial_value = shared(np.zeros((4, 1), dtype=config.floatX))
        outputs_info = [OrderedDict([("initial", initial_value), ("taps", [-4])]), None]
        results, _ = scan(fn=onestep, sequences=seq, outputs_info=outputs_info)
        sharedvar = shared(np.zeros((1, 1), dtype=config.floatX))
        updates = OrderedDict([(sharedvar, results[0][-1:])])

        f = function([seq], results[1], updates=updates)

        # This fails if scan uses wrongly the borrow flag
        assert np.all(exp_out == f(inp))

        # This fails if Scan's infer_shape() is unable to remove the Scan
        # node from the graph.
        f_infershape = function([seq], results[1].shape, mode="FAST_RUN")
        scan_nodes_infershape = scan_nodes_from_fct(f_infershape)
        assert len(scan_nodes_infershape) == 0

    def test_memory_reuse_with_outputs_as_inputs(self):
        # Test the memory pre-allocation feature in scan for the following
        # cases :
        #  - An output of the inner graph is also an input of the inner graph
        #  - An output of the inner graph is not an input in the unoptimized
        #    graph but it could becomes the case in the optimized graph due to
        #    the optimizations.
        #  - An output of the inner graph is obtained through a view op on an
        #    input of the inner graph and the view op is removed by the
        #    optimization process
        #  - An output of the inner graph is obtained through a view op on an
        #    input of the inner graph and the view op is NOT removed by the
        #    optimization process
        #  - An output of the inner graph is not obtained through any of the
        #    previously mentioned cases (standard case)

        def inner_fn(tap_m3, tap_m2, tap_m1):
            return (
                tap_m2,
                (tap_m1 * 1),
                disconnected_grad(tap_m2),
                assert_op(tap_m2, 1),
                tap_m3 + tap_m2 + tap_m1,
            )

        init = matrix()
        outputs_info = [None, None, None, None, dict(initial=init, taps=[-3, -2, -1])]

        out, _ = scan(inner_fn, outputs_info=outputs_info, n_steps=3)
        fct = function([init], out)

        # Compare obtained outputs with expected outputs
        outputs = fct(np.arange(9, dtype=config.floatX).reshape(3, 3))

        states = np.array(
            [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 12, 15], [18, 23, 28], [33, 42, 51]],
            dtype=config.floatX,
        )
        expected_outputs = [
            states[1:4],
            states[2:5],
            states[1:4],
            states[1:4],
            states[3:6],
        ]

        utt.assert_allclose(outputs, expected_outputs)

    def test_grad_connectivity_matrix(self):
        def inner_fn(x_tm1, y_tm1, z_tm1):
            x_tm1.name = "x"
            y_tm1.name = "y"
            z_tm1.name = "z"
            return x_tm1 ** 2, y_tm1, x_tm1 + 1

        x0 = vector("X")
        y0 = vector("y0")
        z0 = vector("Z")
        [x, y, z], _ = scan(inner_fn, outputs_info=[x0, y0, z0], n_steps=10)
        cost = (x + y + z).sum()

        grad(cost, x0)  # defined
        grad(cost, y0)  # defined

        with pytest.raises(ValueError):
            grad(cost, z0)
        cost = x.sum()
        with pytest.raises(ValueError):
            grad(cost, y0)

    def test_disconnected_gradient(self):
        v = vector("v")
        m = matrix("m")
        u0 = aet.zeros((7,))

        [u, m2], _ = scan(lambda _, u: [u, v], sequences=m, outputs_info=[u0, None])
        # This used to raise an exception with older versions because for a
        # disconnected gradient a non disconnected type was returned
        grad((m * m2).sum(), v)

    def test_disconnected_gradient2(self):
        v = vector("v")
        m = matrix("m")
        u0 = aet.zeros((7,))

        [u, m2], _ = scan(
            lambda x, u: [x + u, u + v], sequences=m, outputs_info=[u0, None]
        )
        # This used to raise an exception with older versions because
        # scan could not detect the connection between `m2` and `x`
        grad(m2.sum(), m)

    def test_disconnected_gradient3(self):
        # This tests for a crash that would occur sometimes when taking the
        # gradient through a scan with a non-recurrent output which would
        # receive a disconnected gradient

        v = dvector("v")

        def step(seq):
            out1 = seq + 1
            out2 = out1 + 1
            return out1, out2

        [out1, out2], _ = scan(step, sequences=v)
        gv = grad(out2.sum(), [v])
        f = function([v], gv)

        # Ensure the output of the function is valid
        output = f(np.random.random(5))
        utt.assert_allclose(output, np.ones(5))

    def test_dot_optimization(self):
        A = matrix("A")
        B = matrix("B")
        S, _ = scan(
            lambda x1, x2, u: u + dot(x1, x2),
            sequences=[A.dimshuffle(0, 1, "x"), B.dimshuffle(0, "x", 1)],
            outputs_info=[aet.zeros_like(A)],
        )
        f = function([A, B], S.owner.inputs[0][-1])
        rng = np.random.RandomState(utt.fetch_seed())
        vA = rng.uniform(size=(5, 5)).astype(config.floatX)
        vB = rng.uniform(size=(5, 5)).astype(config.floatX)
        utt.assert_allclose(f(vA, vB), np.dot(vA.T, vB))

    def test_pregreedy_optimizer(self):
        W = aet.zeros((5, 4))
        bv = aet.zeros((5,))
        bh = aet.zeros((4,))
        v = matrix("v")
        (bv_t, bh_t), _ = scan(
            lambda _: [bv, bh], sequences=v, outputs_info=[None, None]
        )
        chain, _ = scan(
            lambda x: dot(dot(x, W) + bh_t, W.T) + bv_t,
            outputs_info=v,
            n_steps=2,
        )
        function([v], chain)(np.zeros((3, 5), dtype=config.floatX))

    def test_savemem_does_not_duplicate_number_of_scan_nodes(self):
        var = aet.ones(())
        values, _ = scan(
            lambda x: ([x], (), until(x)),
            outputs_info=[var],
            n_steps=2,
        )

        tmp_fn = function([var], values)
        scan_nodes = [
            x for x in tmp_fn.maker.fgraph.toposort() if isinstance(x.op, Scan)
        ]
        assert len(scan_nodes) == 1

    def test_eliminate_seqs(self):
        U = vector("U")
        sh = shared(asarrayX(2.0))
        x1 = vector("x1")
        x2 = scalar("x2")

        def rec_fn(*args):
            u_t = args[0]
            return [
                (u_t + 1, u_t + 2, u_t + 3),  # mitsot  # sitsot  # nitsot
                {sh: u_t + 4},
            ]  # shared

        [X1, X2, X3], updates = scan(
            rec_fn,
            U,
            [dict(initial=x1, taps=[-1, -3]), x2, None],
            n_steps=None,
            truncate_gradient=-1,
            go_backwards=False,
        )
        f = function(
            [U, x1, x2],
            [X1, X2, X3],
            updates=updates,
            mode=Mode(linker="py"),
            allow_input_downcast=True,
        )
        rng = np.random.RandomState(utt.fetch_seed())
        v_u = asarrayX(rng.uniform(size=(5,)))
        outs = f(v_u, [0, 0, 0], 0)
        utt.assert_allclose(outs[0], v_u + 1)
        utt.assert_allclose(outs[1], v_u + 2)
        utt.assert_allclose(outs[2], v_u + 3)
        utt.assert_allclose(sh.get_value(), v_u[-1] + 4)

    def test_eliminate_nonseqs(self):
        W = scalar("W")
        sh = shared(asarrayX(2.0))
        x1 = vector("x1")
        x2 = scalar("x2")

        def rec_fn(*args):
            w = args[-1]
            return [
                (w + 1.0, w + 2.0, w + 3.0),  # mitsot  # sitsot  # nitsot
                {sh: w + 4.0},
            ]  # shared

        [X1, X2, X3], updates = scan(
            rec_fn,
            [],
            [dict(initial=x1, taps=[-1, -3]), x2, None],
            W,
            n_steps=5,
            truncate_gradient=-1,
            go_backwards=False,
        )
        f = function(
            [W, x1, x2],
            [X1, X2, X3],
            updates=updates,
            mode=Mode(linker="py"),
            allow_input_downcast=True,
        )
        rng = np.random.RandomState(utt.fetch_seed())
        v_w = asarrayX(rng.uniform())
        outs = f(v_w, [0, 0, 0], 0)
        utt.assert_allclose(outs[0], v_w + 1)
        utt.assert_allclose(outs[1], v_w + 2)
        utt.assert_allclose(outs[2], v_w + 3)
        utt.assert_allclose(sh.get_value(), v_w + 4)

    def test_grad_bug_disconnected_input(self):
        W = shared(np.zeros((3, 3)), name="W")
        v = ivector(name="v")
        y, _ = scan(lambda i, W: W[i], sequences=v, outputs_info=None, non_sequences=W)

        # This used to raise an exception
        f = function([v], grad(y.sum(), W))
        utt.assert_allclose(f([1, 2]), [[0, 0, 0], [1, 1, 1], [1, 1, 1]])

    def test_clone(self):
        def test(x, y, mention_y):
            if mention_y:
                d = 0.1 + 0 * y
            else:
                d = 0.1
            out = clone_replace(y, replace={x: x + d})
            return function([], out)()

        x = shared(np.asarray(0.0, dtype=config.floatX))
        utt.assert_allclose(
            test(x, aet_sum((x + 1) ** 2), mention_y=False), 1.21000003815
        )
        utt.assert_allclose(
            test(x, aet_sum((x + 1) ** 2), mention_y=True), 1.21000003815
        )

    def test_grad_find_input(self):
        w = shared(np.array(0, dtype="float32"), name="w")
        init = fscalar("init")

        out, _ = scan(
            fn=lambda prev: w,
            outputs_info=init,
            n_steps=2,
        )
        grad(out[-1], w)

    def test_scan_merge_nodes(self):
        inps = vector()
        state = scalar()
        y1, _ = scan(lambda x, y: x * y, sequences=inps, outputs_info=state, n_steps=5)

        y2, _ = scan(
            lambda x, y: (x + y, until(x > 0)),
            sequences=inps,
            outputs_info=state,
            n_steps=5,
        )
        scan_node1 = y1.owner.inputs[0].owner
        assert isinstance(scan_node1.op, Scan)
        scan_node2 = y2.owner.inputs[0].owner
        assert isinstance(scan_node2.op, Scan)
        opt_obj = ScanMerge()
        # Test the method belongs_to of this class. Specifically see if it
        # detects the two scan_nodes as not being similar
        assert not opt_obj.belongs_to_set(scan_node1, [scan_node2])
        assert not opt_obj.belongs_to_set(scan_node2, [scan_node1])

    def test_remove_constants_and_unused_inputs_scan_non_seqs(self):
        # Test the opt remove_constants_and_unused_inputs_scan for
        # non sequences.
        W = matrix(name="W")
        v = ivector(name="v")
        y1, _ = scan(
            lambda i, W: W[i], sequences=v, outputs_info=None, non_sequences=[W]
        )
        y2, _ = scan(
            lambda i, _, W: W[i],
            sequences=v,
            outputs_info=None,
            non_sequences=[W[0], W],
        )
        y3, _ = scan(
            lambda i, W, _: W[i],
            sequences=v,
            outputs_info=None,
            non_sequences=[W, W[0]],
        )
        y4, _ = scan(
            lambda i, _, _2, W: W[i],
            sequences=v,
            outputs_info=None,
            non_sequences=[W[0], W[0], W],
        )
        y5, _ = scan(
            lambda i, _, W, _2: W[i],
            sequences=v,
            outputs_info=None,
            non_sequences=[W[0], W, W[0]],
        )
        y6, _ = scan(
            lambda i, W, _, _2: W[i],
            sequences=v,
            outputs_info=None,
            non_sequences=[W, W[0], W[0]],
        )
        # TODO: y7 have problem during run time. I think it should
        # raise an error during the scan construction.
        # y7, _ = scan(lambda i, W, _, _2: W[i], sequences=v,
        #                    outputs_info=None, non_sequences=[v, W[0], W])
        for out in [y1, y2, y3, y4, y5, y6]:
            # This used to raise an exception
            f = function([W, v], out, mode=mode_with_opt)
            f(np.zeros((3, 3), dtype=config.floatX), [1, 2])

            scan_nodes = scan_nodes_from_fct(f)
            assert len(scan_nodes) == 1
            scan_node = scan_nodes[0]

            # The first input is the number of iteration.
            assert len(scan_node.inputs[1:]) == len(set(scan_node.inputs[1:]))
            inp = scan_node.op.inner_non_seqs(scan_node.op.inputs)
            assert len(inp) == 1
            assert len(inp) == len(set(inp))
            inp = scan_node.op.outer_non_seqs(scan_node)
            assert len(inp) == 1
            assert len(inp) == len(set(inp))

    def test_remove_constants_and_unused_inputs_scan_seqs(self):
        # Test the opt remove_constants_and_unused_inputs_scan for sequences.
        W = matrix(name="W")
        v = ivector(name="v")
        vv = matrix(name="vv")
        y1, _ = scan(
            lambda i, W: W[i], sequences=v, outputs_info=None, non_sequences=[W]
        )
        y2, _ = scan(
            lambda i, _, W: W[i], sequences=[v, v], outputs_info=None, non_sequences=W
        )
        y3, _ = scan(
            lambda i, _, W: W[i],
            sequences=[v, vv[0]],
            outputs_info=None,
            non_sequences=W,
        )
        y4, _ = scan(
            lambda _, i, W: W[i],
            sequences=[vv[0], v],
            outputs_info=None,
            non_sequences=W,
        )
        y5, _ = scan(
            lambda _, i, _2, W: W[i],
            sequences=[vv, v, vv[0]],
            outputs_info=None,
            non_sequences=W,
        )
        y6, _ = scan(
            lambda _, _2, i, W: W[i],
            sequences=[vv[0], vv, v],
            outputs_info=None,
            non_sequences=W,
        )
        y7, _ = scan(
            lambda i, _, _2, W: W[i],
            sequences=[v, vv[0], vv[0]],
            outputs_info=None,
            non_sequences=W,
        )
        y8, _ = scan(
            lambda _, i, W, _2, _3: W[i],
            sequences=[vv[0], v],
            outputs_info=None,
            non_sequences=[W, W[0], W[0]],
        )
        for out in [y1, y2, y3, y4, y5, y6, y7, y8]:
            # This used to raise an exception
            f = function([W, v, vv], out, on_unused_input="ignore", mode=mode_with_opt)
            f(
                np.zeros((3, 3), config.floatX),
                [1, 2],
                np.zeros((3, 3), config.floatX),
            )

            scan_nodes = scan_nodes_from_fct(f)
            assert len(scan_nodes) == 1
            scan_node = scan_nodes[0]

            # The first input is the number of iteration.
            assert len(scan_node.inputs[1:]) == len(set(scan_node.inputs[1:]))
            inp = scan_node.op.inner_seqs(scan_node.op.inputs)
            assert len(inp) == 1
            inp = scan_node.op.outer_seqs(scan_node)
            assert len(inp) == 1
            inp = scan_node.op.inner_non_seqs(scan_node.op.inputs)
            assert len(inp) == 1
            inp = scan_node.op.outer_non_seqs(scan_node)
            assert len(inp) == 1

    @pytest.mark.slow
    def test_hessian_bug_grad_grad_two_scans(self):
        # Bug reported by Bitton Tenessi
        # NOTE : The test to reproduce the bug reported by Bitton Tenessi
        # was modified from its original version to be faster to run.

        W = fvector(name="W")
        n_steps = iscalar(name="Nb_steps")

        def loss_outer(sum_outer, W):
            def loss_inner(sum_inner, W):

                return sum_inner + (W ** 2).sum()

            result_inner, _ = scan(
                fn=loss_inner,
                outputs_info=aet.as_tensor_variable(np.asarray(0, dtype=np.float32)),
                non_sequences=[W],
                n_steps=1,
            )
            return sum_outer + result_inner[-1]

        # Also test return_list for that case.
        result_outer, _ = scan(
            fn=loss_outer,
            outputs_info=aet.as_tensor_variable(np.asarray(0, dtype=np.float32)),
            non_sequences=[W],
            n_steps=n_steps,
            return_list=True,
        )

        cost = result_outer[0][-1]
        H = hessian(cost, W)
        print(".", file=sys.stderr)
        f = function([W, n_steps], H)
        f(np.ones((8,), dtype="float32"), 1)

    def test_strict_mode(self):
        n = 10

        w = np.array([[-1, 2], [3, -4]]).astype(config.floatX)
        w_ = shared(w)
        x0 = np.array([1, 2]).astype(config.floatX)
        x0_ = vector(name="x0", dtype=config.floatX)

        def _scan_loose(x):
            return dot(x, w_)

        def _scan_strict(x, w_ns):
            return dot(x, w_ns)

        ret_loose = scan(
            _scan_loose, sequences=[], outputs_info=[x0_], n_steps=n, strict=False
        )
        f_loose = function([x0_], ret_loose[0][-1])

        ret_strict = scan(
            _scan_strict,
            sequences=[],
            outputs_info=[x0_],
            non_sequences=[w_],
            n_steps=n,
            strict=True,
        )
        f_strict = function([x0_], ret_strict[0][-1])

        result_loose = f_loose(x0)
        result_strict = f_strict(x0)

        diff = (abs(result_loose - result_strict)).mean()

        assert diff <= type_eps[config.floatX]

    def test_strict_mode_ex(self):
        n = 10

        w = np.array([[-1, 2], [3, -4]]).astype(config.floatX)
        w_ = shared(w)
        x0_ = vector(name="x0", dtype=config.floatX)

        def _scan_loose(x):
            return dot(x, w_)

        with pytest.raises(MissingInputError):
            scan(_scan_loose, sequences=[], outputs_info=[x0_], n_steps=n, strict=True)

    def test_monitor_mode(self):
        # Test that it is possible to pass an instance of MonitorMode
        # to the inner function
        k = iscalar("k")
        A = vector("A")

        # Build a MonitorMode that counts how many values are greater than 10
        def detect_large_outputs(fgraph, i, node, fn):
            for output in fn.outputs:
                if isinstance(output[0], np.ndarray):
                    detect_large_outputs.large_count += (output[0] > 10).sum()

        detect_large_outputs.large_count = 0

        mode = MonitorMode(post_func=detect_large_outputs)

        # Symbolic description of the result
        result, updates = scan(
            fn=lambda prior_result, A: prior_result * A,
            outputs_info=aet.ones_like(A),
            non_sequences=A,
            n_steps=k,
            mode=mode,
        )

        final_result = result[-1]

        f = function(inputs=[A, k], outputs=final_result, updates=updates)
        f(np.asarray([2, 3, 0.1, 0, 1], dtype=config.floatX), 4)

        # There should be 3 outputs greater than 10: prior_result[0] at step 3,
        # and prior_result[1] at steps 2 and 3.
        if config.mode in ["DEBUG_MODE", "DebugMode"]:
            # DebugMode will run all the intermediate nodes, so we
            # should expect a multiple of 3, not exactly 3.
            assert detect_large_outputs.large_count % 3 == 0

        else:
            assert detect_large_outputs.large_count == 3


@pytest.mark.skipif(
    not config.cxx, reason="G++ not available, so we need to skip this test."
)
def test_speed():
    from timeit import timeit

    n_timeit = 50

    # We need the CVM for this speed test
    r = np.arange(10000).astype(config.floatX).reshape(1000, 10)

    def f_py():
        for i in range(1, 1000):
            r[i] += r[i - 1]

    python_duration = timeit(lambda: f_py(), number=n_timeit)

    r = np.arange(10000).astype(config.floatX).reshape(1000, 10)

    def f_py_iter():
        r_i = iter(r[1:])
        r_ii = iter(r[:-1])
        while True:
            try:
                tmp = next(r_i)
                tmp += next(r_ii)
            except StopIteration:
                break

    python_iter_duration = timeit(lambda: f_py_iter(), number=n_timeit)

    # r = np.arange(10000).astype(config.floatX).reshape(1000, 10)
    # s_r = matrix()
    # s_y, updates = scan(
    #     fn=lambda ri, rii: ri + rii,
    #     sequences=[s_r[1:]],
    #     outputs_info=aet.constant(r[0]),
    #     mode=Mode(linker="cvm"),
    # )
    # assert not updates
    #
    # f_cvm = function([s_r], s_y)
    #
    # cvm_duration = timeit(lambda: f_cvm(r), number=n_timeit)

    # XXX: Why does this take so much longer than Python?!
    # assert cvm_duration - python_duration < python_duration * 0.15

    r = np.arange(10000).astype(config.floatX).reshape(-1, 10)
    shared_r = shared(r)
    s_i = shared(np.array(1))
    s_rinc = inc_subtensor(
        shared_r[s_i], shared_r[s_i - 1], tolerate_inplace_aliasing=True
    )

    f_cvm_shared = function(
        [],
        [],
        updates=OrderedDict([(s_i, s_i + 1), (shared_r, s_rinc)]),
        mode=Mode(linker="cvm"),
    )
    f_cvm_shared._check_for_aliased_inputs = False

    cvm_shared_duration = timeit(lambda: f_cvm_shared(), number=n_timeit)

    assert cvm_shared_duration < python_duration
    assert cvm_shared_duration < python_iter_duration


@pytest.mark.skipif(
    not config.cxx, reason="G++ not available, so we need to skip this test."
)
def test_speed_rnn():
    from timeit import timeit

    n_timeit = 50
    L = 10000
    N = 50

    np.random.seed(2523452)
    r = np.arange(L * N).astype(config.floatX).reshape(L, N)
    w = np.random.randn(N, N).astype(config.floatX)

    def f_py():
        for i in range(1, L):
            r[i] = np.tanh(np.dot(r[i - 1], w))

    python_duration = timeit(lambda: f_py(), number=n_timeit)

    # r = np.arange(L * N).astype(config.floatX).reshape(L, N)
    # s_r = matrix()
    # s_y, updates = scan(
    #     fn=lambda ri, rii: tanh(dot(rii, w)),
    #     sequences=[s_r[1:]],
    #     outputs_info=aet.constant(r[0]),
    #     mode=Mode(linker="cvm"),
    # )
    # assert not updates
    #
    # f_cvm = function([s_r], s_y, mode=Mode(linker="cvm"))
    #
    # cvm_duration = timeit(lambda: f_cvm(r), number=n_timeit)

    # XXX: Why does this take so much longer than Python?!
    # assert cvm_duration - python_duration < python_duration * 0.15

    r = np.arange(L * N).astype(config.floatX).reshape(L, N)
    shared_r = shared(r)
    s_i = shared(1)
    s_rinc = inc_subtensor(
        shared_r[s_i],
        tanh(dot(shared_r[s_i - 1], w)),
        tolerate_inplace_aliasing=True,
    )
    f_cvm_shared = function(
        [],
        [],
        updates=OrderedDict([(s_i, s_i + 1), (shared_r, s_rinc)]),
        mode=Mode(linker="cvm"),
    )

    cvm_shared_duration = timeit(lambda: f_cvm_shared(), number=n_timeit)

    assert cvm_shared_duration < python_duration


@pytest.mark.skipif(
    not config.cxx, reason="G++ not available, so we need to skip this test."
)
def test_speed_batchrnn():
    """
    This function prints out the speed of recurrent neural network
    calculations implemented in various ways.

    We force the mode to Mode(linker='cvm'). If you manually
    change this code to use DebugMode this will test the correctness
    of the optimizations applied, but generally correctness-testing
    is not the goal of this test.

    The computation being tested here is a repeated tanh of a matrix-vector
    multiplication - the heart of an ESN or RNN.
    """
    L = 100
    B = 50
    N = 400

    np.random.seed(2523452)
    r = np.arange(B * L * N).astype(config.floatX).reshape(L, B, N)
    w = np.random.randn(N, N).astype(config.floatX)

    t0 = time.time()
    for i in range(1, L):
        r[i] = np.tanh(np.dot(r[i - 1], w))
    t1 = time.time()
    python_duration = t1 - t0

    r = np.arange(B * L * N).astype(config.floatX).reshape(L, B, N)
    shared_r = shared(r)
    s_i = shared(1)
    s_rinc = inc_subtensor(
        shared_r[s_i],
        tanh(dot(shared_r[s_i - 1], w)),
        tolerate_inplace_aliasing=True,
    )
    f = function(
        [],
        [],
        updates=[(s_i, s_i + 1), (shared_r, s_rinc)],
        mode=Mode(linker="cvm"),
    )
    f_fn = f.fn
    t2 = time.time()
    f_fn(n_calls=L - 2)
    f()  # 999 to update the profiling timers
    t3 = time.time()
    cvm_duration = t3 - t2

    assert cvm_duration < python_duration


def test_compute_test_value():
    # Verify that test values can be used with scan.
    with config.change_flags(compute_test_value="raise"):
        x = vector("x")
        xv = np.ones(3, dtype=config.floatX)
        x.tag.test_value = xv
        y = shared(np.arange(3, dtype=config.floatX), name="y")
        z, updates = scan(fn=lambda u, v: u + v, sequences=[x, y])
        assert not updates
        z.name = "z"
        # The gradient computation used to crash before 6af465e.
        grad(z.sum(), x)


def test_compute_test_value_nonseq():
    # Verify that test values can be used for non_sequences with scan.
    with config.change_flags(compute_test_value="raise"):
        x = vector("x")
        xv = np.ones(3, dtype=config.floatX)
        x.tag.test_value = xv
        y = shared(np.arange(9, dtype=config.floatX).reshape(3, 3), name="y")
        z, updates = scan(fn=lambda u, v: u + v, sequences=[x], non_sequences=[y])
        assert not updates
        z.name = "z"
        # The gradient computation used to crash before 6af465e.
        grad(z.sum(), x)


def test_compute_test_value_grad():
    # Test case originally reported by Bitton Tenessi
    # https://groups.google.com/d/msg/theano-users/fAP3i2CbskQ/3OgBf4yjqiQJ
    WEIGHT = np.array([1, 2, 1, 3, 4, 1, 5, 6, 1, 7, 8, 1], dtype="float32")

    with config.change_flags(compute_test_value="raise", exception_verbosity="high"):
        W_flat = fvector(name="W")
        W_flat.tag.test_value = WEIGHT
        W = W_flat.reshape((2, 2, 3))

        outputs_mi = aet.as_tensor_variable(np.asarray(0, dtype="float32"))
        outputs_mi.tag.test_value = np.asarray(0, dtype="float32")

        def loss_mi(mi, sum_mi, W):
            outputs_ti = aet.as_tensor_variable(np.asarray(0, dtype="float32"))
            outputs_ti.tag.test_value = np.asarray(0, dtype="float32")

            def loss_ti(ti, sum_ti, mi, W):
                return W.sum().sum().sum() + sum_ti

            result_ti, _ = scan(
                fn=loss_ti,
                outputs_info=outputs_ti,
                sequences=aet.arange(W.shape[1], dtype="int32"),
                non_sequences=[mi, W],
            )
            lossmi = result_ti[-1]
            return sum_mi + lossmi

        result_mi, _ = scan(
            fn=loss_mi,
            outputs_info=outputs_mi,
            sequences=aet.arange(W.shape[0], dtype="int32"),
            non_sequences=[W],
        )

        loss = result_mi[-1]
        grad(loss, W_flat)


def test_compute_test_value_grad_cast():
    # Test for test values when variables have to be casted
    # Reported by Daniel Renshaw at
    # https://groups.google.com/d/topic/theano-users/o4jK9xDe5WI/discussion
    with config.change_flags(compute_test_value="raise"):
        h = matrix("h")
        h.tag.test_value = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=config.floatX)

        w = shared(np.random.randn(4, 3).astype(config.floatX), name="w")

        outputs, _ = scan(
            lambda i, h, w: (dot(h[i], w), i),
            outputs_info=[None, 0],
            non_sequences=[h, w],
            n_steps=3,
        )

        grad(outputs[0].sum(), w)


def test_constant_folding_n_steps():
    # The following code used to crash at revision 2060b8f, in the constant
    # folding optimization step.
    res, _ = scan(
        lambda x: x * 2,
        outputs_info=aet.ones(()),
        # The constant `n_steps` was causing the crash.
        n_steps=10,
    )
    with config.change_flags(on_opt_error="raise"):
        function([], res)()


def test_outputs_taps_check():
    # Checks that errors are raised with bad output_info taps.
    x = fvector("x")
    y = fvector("y")

    def f(x, y):
        return [x]

    outputs_info = {"initial": y, "taps": [0]}
    with pytest.raises(ValueError):
        scan(f, x, outputs_info)
    outputs_info = {"initial": y, "taps": [-1, -1]}
    with pytest.raises(ValueError):
        scan(f, x, outputs_info)


def test_default_value_broadcasted():
    def floatx(X):
        return np.asarray(X, dtype=config.floatX)

    def init_weights(shape, name):
        return shared(floatx(np.random.randn(*shape) * 0.1), name)

    X = matrix("X")
    in_size = 2
    out_size = 4
    W_x = init_weights((in_size, out_size), "W_x")

    def _active(x, pre_h):
        x = reshape(x, (1, in_size))
        pre_h = dot(x, W_x)
        return pre_h

    value, scan_updates = scan(
        _active,
        sequences=X,
        outputs_info=[aet.alloc(floatx(0.0), 1, out_size)],
    )
    cost = mean(value)
    gW_x = grad(cost, W_x)
    updates = [(W_x, W_x - 0.1 * gW_x)]
    f = function([X], outputs=cost, updates=updates)
    f(np.random.rand(10, in_size).astype(X.dtype))


class TestInconsistentBroadcast:
    def test_raise_error(self):
        x = tensor3()
        initial_x = aet.constant(np.zeros((1, 10)))
        y, updates = scan(
            fn=lambda x, prev_x: x + prev_x,
            sequences=x,
            outputs_info=[dict(initial=initial_x)],
        )
        # Error, because the broadcast patterns are inconsistent.
        with pytest.raises(TypeError):
            grad(y.sum(), x)

        # No error here, because the broadcast patterns are consistent.
        initial_x = aet.unbroadcast(initial_x, 0, 1)
        y, updates = scan(
            fn=lambda x, prev_x: x + prev_x,
            sequences=x,
            outputs_info=[dict(initial=initial_x)],
        )
        grad(y.sum(), x)


class TestMissingInputError:
    @pytest.mark.xfail(raises=MissingInputError)
    def test_raise_error(self):
        c = shared(0.0)
        inc = scalar("inc")

        def count_up():
            return aet.zeros(()), {c: c + inc}

        _, updates = scan(count_up, n_steps=20)
        function(inputs=[inc], outputs=[], updates=updates)


class TestGradUntil:
    def setup_method(self):
        self.x = vector(name="x")
        self.threshold = scalar(name="threshold", dtype="int64")
        self.seq = np.arange(15, dtype=config.floatX)
        self.numpy_output = self.seq[:7] ** 2
        z = np.zeros(8, dtype=config.floatX)
        self.numpy_gradient = 2 * np.concatenate([self.seq[:7], z], axis=0)

    def test_grad_until(self):
        r, _ = scan(
            lambda x, u: (x * x, until(x > u)),
            sequences=self.x,
            non_sequences=[self.threshold],
        )
        g = grad(r.sum(), self.x)
        f = function([self.x, self.threshold], [r, g])
        aesara_output, aesara_gradient = f(self.seq, 5)

        utt.assert_allclose(aesara_output, self.numpy_output)
        utt.assert_allclose(aesara_gradient, self.numpy_gradient)

    def test_grad_until_ndim_greater_one(self):
        def tile_array(inp):
            n_cols = 5
            return np.tile(inp.reshape((-1, 1)), (1, n_cols))

        X = matrix(name="x")
        arr = tile_array(self.seq)
        r, _ = scan(
            lambda x, u: (x * x, until(aet_all(x > u))),
            sequences=X,
            non_sequences=[self.threshold],
        )
        g = grad(r.sum(), X)
        f = function([X, self.threshold], [r, g])
        aesara_output, aesara_gradient = f(arr, 5)

        utt.assert_allclose(aesara_output, tile_array(self.numpy_output))
        utt.assert_allclose(aesara_gradient, tile_array(self.numpy_gradient))

    def test_grad_until_and_truncate(self):
        n = 3
        r, _ = scan(
            lambda x, u: (x * x, until(x > u)),
            sequences=self.x,
            non_sequences=[self.threshold],
            truncate_gradient=n,
        )
        g = grad(r.sum(), self.x)
        f = function([self.x, self.threshold], [r, g])
        aesara_output, aesara_gradient = f(self.seq, 5)

        self.numpy_gradient[: 7 - n] = 0
        utt.assert_allclose(aesara_output, self.numpy_output)
        utt.assert_allclose(aesara_gradient, self.numpy_gradient)

    def test_grad_until_and_truncate_sequence_taps(self):
        n = 3
        r, _ = scan(
            lambda x, y, u: (x * y, until(y > u)),
            sequences=dict(input=self.x, taps=[-2, 0]),
            non_sequences=[self.threshold],
            truncate_gradient=n,
        )
        g = grad(r.sum(), self.x)
        f = function([self.x, self.threshold], [r, g])
        aesara_output, aesara_gradient = f(self.seq, 6)

        # Gradient computed by hand:
        numpy_grad = np.array([0, 0, 0, 5, 6, 10, 4, 5, 0, 0, 0, 0, 0, 0, 0])
        numpy_grad = numpy_grad.astype(config.floatX)
        utt.assert_allclose(aesara_gradient, numpy_grad)


def test_condition_hidden_inp():
    max_value = scalar("max_value")
    n_steps = iscalar("n_steps")

    def accum(prev_value, step):
        new_value = prev_value + step
        new_step = step + 1
        condition = until(new_value > max_value)
        return [new_value, new_step], condition

    rs, updates = scan(fn=accum, outputs_info=[0, 0], n_steps=n_steps)

    f = function(inputs=[max_value, n_steps], outputs=rs)

    _sum, total_steps = f(100, 100)


def test_mintap_onestep():
    seq = ivector("seq")
    seq_info = dict(input=seq, taps=[2])

    def accum(seq_t, prev_sum):
        new_sum = prev_sum + seq_t
        return new_sum

    rs, updates = scan(fn=accum, sequences=seq_info, outputs_info=0, n_steps=1)

    f = function(inputs=[seq], outputs=rs)
    _seq = np.arange(20).astype("int32")
    _sum = f(_seq)
    print("sum %f" % _sum)
    assert _sum == 2
