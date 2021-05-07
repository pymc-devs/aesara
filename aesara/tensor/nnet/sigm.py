"""
Ops and optimizations: sigmoid, softplus.

These functions implement special cases of exp and log to improve numerical
stability.

"""

import aesara
from aesara import printing
from aesara import scalar as aes
from aesara.graph.opt import copy_stack_trace, local_optimizer
from aesara.printing import pprint
from aesara.scalar import sigmoid as scalar_sigmoid
from aesara.tensor.basic import constant
from aesara.tensor.elemwise import Elemwise
from aesara.tensor.math import clip, sigmoid
from aesara.tensor.type import TensorType


class UltraFastScalarSigmoid(aes.UnaryScalarOp):
    """
    This is just speed opt. Not for stability.

    """

    nfunc_spec = ("scipy.special.expit", 1, 1)

    @staticmethod
    def st_impl(x):
        x = 0.5 * x
        # The if is a tanh approximate.
        if x >= 0:
            if x < 1.7:
                z = 1.5 * x / (1 + x)
            elif x < 3:
                z = 0.935409070603099 + 0.0458812946797165 * (x - 1.7)
            else:
                z = 0.99505475368673
        else:
            xx = -x
            if xx < 1.7:
                z = 1.5 * xx / (1 + xx)
            elif xx < 3:
                z = 0.935409070603099 + 0.0458812946797165 * (xx - 1.7)
            else:
                z = 0.99505475368673
            z = -z

        return 0.5 * (z + 1.0)

    def impl(self, x):
        return UltraFastScalarSigmoid.st_impl(x)

    def c_code(self, node, name, inp, out, sub):
        (x,) = inp
        (z,) = out
        dtype = node.outputs[0].type.dtype_specs()[1]

        return (
            """
        %(dtype)s x = 0.5 * %(x)s;
   // The if is a tanh approximate.
   if(x>=0) {
        %(z)s = (x<1.7 ? (1.5*x/(1+x)) :
                         (x<3 ? (0.935409070603099 + 0.0458812946797165*(x-1.7)):
                         0.99505475368673));
    } else {
        %(dtype)s xx = -x;
        %(z)s = -(xx<1.7 ? (1.5*xx/(1+xx)) :
                           (xx<3 ? (0.935409070603099 + 0.0458812946797165*(xx-1.7)):
                                   0.99505475368673));
    }

        //%(z)s = 0.5*(ultrafasttanh(0.5*x)+1.);
        %(z)s = 0.5*(%(z)s+1.);
        """
            % locals()
        )


ultra_fast_scalar_sigmoid = UltraFastScalarSigmoid(
    aes.upgrade_to_float, name="ultra_fast_scalar_sigmoid"
)
ultra_fast_sigmoid = Elemwise(ultra_fast_scalar_sigmoid, name="ultra_fast_sigmoid")

ultra_fast_sigmoid_inplace = Elemwise(
    UltraFastScalarSigmoid(aes.transfer_type(0)),
    inplace_pattern={0: 0},
    name="ultra_fast_sigmoid_inplace",
)

pprint.assign(ultra_fast_sigmoid, printing.FunctionPrinter("ultra_fast_sigmoid"))


# @opt.register_uncanonicalize
@local_optimizer([sigmoid])
def local_ultra_fast_sigmoid(fgraph, node):
    """
    When enabled, change all sigmoid to ultra_fast_sigmoid.

    For example do mode.including('local_ultra_fast_sigmoid')
    or use the Aesara flag optimizer_including=local_ultra_fast_sigmoid.

    This speeds up the sigmoid op by using an approximation.

    This is done after the stabilization and specialize phases
    to avoid interacting with them.

    """
    if isinstance(node.op, Elemwise) and node.op.scalar_op == scalar_sigmoid:
        out = ultra_fast_sigmoid(node.inputs[0])
        copy_stack_trace(node.outputs[0], out)

        def values_eq_approx_remove_low_prec(a, b):
            # atol is found by trial/error.
            # Other test could fail without good reason.
            return TensorType.values_eq_approx(a, b, atol=0.02)

        # Let DebugMode know that there this opt approx the values.
        out.tag.values_eq_approx = values_eq_approx_remove_low_prec
        return [out]


aesara.compile.optdb["uncanonicalize"].register(
    "local_ultra_fast_sigmoid", local_ultra_fast_sigmoid
)


def hard_sigmoid(x):
    """
    An approximation of sigmoid.

    More approximate and faster than ultra_fast_sigmoid.

    Approx in 3 parts: 0, scaled linear, 1.

    Removing the slope and shift does not make it faster.

    """
    # Use the same dtype as determined by "upgrade_to_float",
    # and perform computation in that dtype.
    out_dtype = aes.upgrade_to_float(aes.Scalar(dtype=x.dtype))[0].dtype
    slope = constant(0.2, dtype=out_dtype)
    shift = constant(0.5, dtype=out_dtype)
    x = (x * slope) + shift
    x = clip(x, 0, 1)
    return x


# @opt.register_uncanonicalize
@local_optimizer([sigmoid])
def local_hard_sigmoid(fgraph, node):
    if isinstance(node.op, Elemwise) and node.op.scalar_op == scalar_sigmoid:
        out = hard_sigmoid(node.inputs[0])
        copy_stack_trace(node.outputs[0], out)

        def values_eq_approx_remove_low_prec(a, b):
            # atol is found by trial/error.
            # Other test could fail without good reason.
            return TensorType.values_eq_approx(a, b, atol=0.1)

        # Let DebugMode know that there this opt approx the values.
        out.tag.values_eq_approx = values_eq_approx_remove_low_prec
        return [out]


aesara.compile.optdb["uncanonicalize"].register(
    "local_hard_sigmoid", local_hard_sigmoid
)
