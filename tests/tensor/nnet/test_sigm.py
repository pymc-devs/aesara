import numpy as np

import aesara
from aesara.configdefaults import config
from aesara.graph.opt import check_stack_trace
from aesara.tensor.math import clip, sigmoid
from aesara.tensor.nnet.sigm import hard_sigmoid, ultra_fast_sigmoid
from aesara.tensor.type import matrix
from tests.tensor.utils import (
    _good_broadcast_unary_normal_no_complex,
    check_floatX,
    copymod,
    makeBroadcastTester,
    upcast_int8_nfunc,
)


TestUltraFastSigmoidBroadcast = makeBroadcastTester(
    op=ultra_fast_sigmoid,
    expected=upcast_int8_nfunc(
        lambda inputs: check_floatX(inputs, 1 / (1 + np.exp(-inputs)))
    ),
    good=copymod(
        _good_broadcast_unary_normal_no_complex, without=["uint16"]
    ),  # numpy fucnting overflows with uint16.
    # grad=_grad_broadcast_unary_normal,
    name="UltraFastSigmoidTester",
    # This is an approx of the sigmoid. That is why we raise eps
    eps=5e-2,
)

TestHardSigmoidBroadcast = makeBroadcastTester(
    op=hard_sigmoid,
    expected=upcast_int8_nfunc(
        lambda inputs: check_floatX(inputs, 1 / (1 + np.exp(-inputs)))
    ),
    good=copymod(
        _good_broadcast_unary_normal_no_complex, without=["uint16"]
    ),  # numpy fucnting overflows with uint16.
    # grad=_grad_broadcast_unary_normal,
    name="HardSigmoidTester",
    # This is an approx of the sigmoid. That is why we raise eps
    eps=1e-1,
)


class TestSpecialSigmoidOpts:
    def get_mode(self, excluding=None):
        """
        Return appropriate mode for the tests.

        :param excluding: List of optimizations to exclude.

        :return: The current default mode unless the `config.mode` option is
        set to 'FAST_COMPILE' (in which case it is replaced by the 'FAST_RUN'
        mode), without the optimizations specified in `excluding`.
        """
        if excluding is None:
            excluding = []
        m = config.mode
        if m == "FAST_COMPILE":
            mode = aesara.compile.mode.get_mode("FAST_RUN")
        else:
            mode = aesara.compile.mode.get_default_mode()
        if excluding:
            return mode.excluding(*excluding)
        else:
            return mode

    def test_local_ultra_fast_sigmoid(self):
        x = matrix("x")
        s = sigmoid(x)

        mode = self.get_mode("local_ultra_fast_sigmoid")
        f = aesara.function([x], s, mode=mode)
        assert check_stack_trace(f, ops_to_check=sigmoid)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert topo[0].op == sigmoid

        mode = self.get_mode().including("local_ultra_fast_sigmoid")
        f = aesara.function([x], s, mode=mode)
        assert check_stack_trace(f, ops_to_check=ultra_fast_sigmoid)
        topo = f.maker.fgraph.toposort()
        assert topo[0].op == ultra_fast_sigmoid
        assert len(topo) == 1
        f([[-50, -10, -4, -1, 0, 1, 4, 10, 50]])

    def test_local_hard_sigmoid(self):
        x = matrix("x")
        s = sigmoid(x)

        mode = self.get_mode("local_hard_sigmoid")
        f = aesara.function([x], s, mode=mode)
        assert check_stack_trace(f, ops_to_check=sigmoid)
        topo = f.maker.fgraph.toposort()
        assert topo[0].op == sigmoid
        assert len(topo) == 1

        mode = self.get_mode().including("local_hard_sigmoid")
        f = aesara.function([x], s, mode=mode)
        topo = f.maker.fgraph.toposort()
        assert not any([n.op == sigmoid for n in topo])
        f([[-50, -10, -4, -1, 0, 1, 4, 10, 50]])

        mode2 = mode.excluding("fusion").excluding("inplace")
        f2 = aesara.function([x], s, mode=mode2)
        assert check_stack_trace(f2, ops_to_check=clip)
