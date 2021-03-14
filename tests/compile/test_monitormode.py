import sys
from io import StringIO

import numpy as np

from aesara.compile import MonitorMode
from aesara.compile.function import function
from aesara.printing import debugprint
from aesara.tensor import log, outer
from aesara.tensor.type import dscalar, vector


def test_detect_nan():
    # Test the code snippet example that detects NaN values.

    nan_detected = [False]

    def detect_nan(fgraph, i, node, fn):
        for output in fn.outputs:
            if np.isnan(output[0]).any():
                print("*** NaN detected ***")
                debugprint(node)
                print("Inputs : %s" % [input[0] for input in fn.inputs])
                print("Outputs: %s" % [output[0] for output in fn.outputs])
                nan_detected[0] = True
                break

    x = dscalar("x")
    f = function(
        [x],
        [log(x) * x],
        mode=MonitorMode(post_func=detect_nan),
    )
    try:
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        f(0)  # log(0) * 0 = -inf * 0 = NaN
    finally:
        sys.stdout = old_stdout
    assert nan_detected[0]


def test_optimizer():
    # Test that we can remove optimizer

    nan_detected = [False]

    def detect_nan(fgraph, i, node, fn):
        for output in fn.outputs:
            if np.isnan(output[0]).any():
                print("*** NaN detected ***")
                debugprint(node)
                print("Inputs : %s" % [input[0] for input in fn.inputs])
                print("Outputs: %s" % [output[0] for output in fn.outputs])
                nan_detected[0] = True
                break

    x = dscalar("x")
    mode = MonitorMode(post_func=detect_nan)
    mode = mode.excluding("fusion")
    f = function([x], [log(x) * x], mode=mode)
    # Test that the fusion wasn't done
    assert len(f.maker.fgraph.apply_nodes) == 2
    try:
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        f(0)  # log(0) * 0 = -inf * 0 = NaN
    finally:
        sys.stdout = old_stdout

    # Test that we still detect the nan
    assert nan_detected[0]


def test_not_inplace():
    # Test that we can remove optimizers including inplace optimizers

    nan_detected = [False]

    def detect_nan(fgraph, i, node, fn):
        for output in fn.outputs:
            if np.isnan(output[0]).any():
                print("*** NaN detected ***")
                debugprint(node)
                print("Inputs : %s" % [input[0] for input in fn.inputs])
                print("Outputs: %s" % [output[0] for output in fn.outputs])
                nan_detected[0] = True
                break

    x = vector("x")
    mode = MonitorMode(post_func=detect_nan)
    # mode = mode.excluding('fusion', 'inplace')
    mode = mode.excluding("local_elemwise_fusion", "inplace_elemwise_optimizer")
    o = outer(x, x)
    out = log(o) * o
    f = function([x], [out], mode=mode)

    # Test that the fusion wasn't done
    assert len(f.maker.fgraph.apply_nodes) == 5
    assert not f.maker.fgraph.toposort()[-1].op.destroy_map
    try:
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        f([0, 0])  # log(0) * 0 = -inf * 0 = NaN
    finally:
        sys.stdout = old_stdout

    # Test that we still detect the nan
    assert nan_detected[0]
