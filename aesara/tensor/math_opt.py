""" Tensor optimizations addressing the ops in math.py."""
# TODO: intelligent merge for mul/add
# TODO: 0*x -> 0

import itertools
import logging
import operator
import warnings
from functools import reduce

import numpy as np

import aesara.scalar.basic as aes
from aesara import compile
from aesara.assert_op import assert_op
from aesara.configdefaults import config
from aesara.graph.basic import Constant, Variable
from aesara.graph.opt import (
    LocalOptGroup,
    LocalOptimizer,
    PatternSub,
    copy_stack_trace,
    in2out,
    local_optimizer,
)
from aesara.misc.safe_asarray import _asarray
from aesara.tensor.basic import (
    Alloc,
    Join,
    MakeVector,
    alloc,
    as_tensor_variable,
    cast,
    constant,
    extract_constant,
    fill,
    get_scalar_constant_value,
    ones_like,
    switch,
    zeros_like,
)
from aesara.tensor.basic_opt import (
    FusionOptimizer,
    _fill_chain,
    broadcast_like,
    encompasses_broadcastable,
    fuse_seqopt,
    local_fill_sink,
    register_canonicalize,
    register_specialize,
    register_specialize_device,
    register_stabilize,
    register_uncanonicalize,
    register_useless,
    scalarconsts_rest,
)
from aesara.tensor.elemwise import CAReduce, DimShuffle, Elemwise
from aesara.tensor.exceptions import NotScalarConstantError
from aesara.tensor.math import (
    All,
    Any,
    Dot,
    NonZeroCAReduce,
    Prod,
    ProdWithoutZeros,
    Sum,
    abs_,
    add,
    dot,
    eq,
    erf,
    erfc,
    exp,
    expm1,
    ge,
    int_div,
    inv,
    log,
    log1p,
    makeKeepDims,
)
from aesara.tensor.math import max as aet_max
from aesara.tensor.math import maximum, mul, neg
from aesara.tensor.math import pow as aet_pow
from aesara.tensor.math import prod, sgn, sigmoid, softplus, sqr, sqrt, sub
from aesara.tensor.math import sum as aet_sum
from aesara.tensor.math import true_div
from aesara.tensor.shape import Shape, Shape_i
from aesara.tensor.subtensor import Subtensor
from aesara.tensor.type import (
    uint_dtypes,
    values_eq_approx_remove_inf,
    values_eq_approx_remove_inf_nan,
    values_eq_approx_remove_nan,
)
from aesara.tensor.var import TensorConstant
from aesara.utils import NoDuplicateOptWarningFilter


_logger = logging.getLogger("aesara.tensor.math_opt")
_logger.addFilter(NoDuplicateOptWarningFilter())


@register_canonicalize
@register_stabilize
@local_optimizer([Dot])
def local_0_dot_x(fgraph, node):
    if not isinstance(node.op, Dot):
        return False

    x = node.inputs[0]
    y = node.inputs[1]
    replace = False
    try:
        if get_scalar_constant_value(x, only_process_constants=True) == 0:
            replace = True
    except NotScalarConstantError:
        pass

    try:
        if get_scalar_constant_value(y, only_process_constants=True) == 0:
            replace = True
    except NotScalarConstantError:
        pass

    if replace:
        constant_zero = constant(0, dtype=node.outputs[0].type.dtype)
        if x.ndim == 2 and y.ndim == 2:
            constant_zero = assert_op(constant_zero, eq(x.shape[1], y.shape[0]))
            return [alloc(constant_zero, x.shape[0], y.shape[1])]
        elif x.ndim == 1 and y.ndim == 2:
            constant_zero = assert_op(constant_zero, eq(x.shape[0], y.shape[0]))
            return [alloc(constant_zero, y.shape[1])]
        elif x.ndim == 2 and y.ndim == 1:
            constant_zero = assert_op(constant_zero, eq(x.shape[1], y.shape[0]))
            return [alloc(constant_zero, x.shape[0])]
        elif x.ndim == 1 and y.ndim == 1:
            constant_zero = assert_op(constant_zero, eq(x.shape[0], y.shape[0]))
            return [constant_zero]
        else:
            _logger.warning(
                "Optimization Warning: "
                "Optimization aesara/opt.py:local_0_dot_x Found "
                "that it could apply, but was not implemented "
                "for dot product with these input types:\n"
                f"({x.type}, {y.type})"
            )


@register_canonicalize
@local_optimizer([DimShuffle])
def local_lift_transpose_through_dot(fgraph, node):
    """Perform the rewrite ``dot(x,y).T -> dot(y.T, x.T)``

    These optimizations "lift" (propagate towards the inputs) DimShuffle
    through dot product.  It allows to put the graph in a more standard shape,
    and to later merge consecutive DimShuffles.

    The transformation should be apply whether or not the transpose is
    inplace.  The newly-introduced transpositions are not inplace, this will
    be taken care of in a later optimization phase.

    """
    if not (isinstance(node.op, DimShuffle) and node.op.new_order == (1, 0)):
        return False
    if not (node.inputs[0].owner and isinstance(node.inputs[0].owner.op, Dot)):
        return False
    x, y = node.inputs[0].owner.inputs

    if x.ndim == y.ndim == 2:
        # Output is dot product of transposed inputs in reverse order
        ret = [dot(y.T, x.T)]

        # Copy over stack trace to output from result of dot-product
        copy_stack_trace(node.inputs[0], ret)
        return ret


def is_inverse_pair(node_op, prev_op, inv_pair):
    """
    Given two consecutive operations, check if they are the
    provided pair of inverse functions.

    """
    node_is_op0 = isinstance(node_op, inv_pair[0])
    node_is_op1 = isinstance(node_op, inv_pair[1])
    prev_is_op0 = isinstance(prev_op, inv_pair[0])
    prev_is_op1 = isinstance(prev_op, inv_pair[1])

    return (node_is_op0 and prev_is_op1) or (node_is_op1 and prev_is_op0)


@register_canonicalize
@register_specialize
@local_optimizer([Elemwise])
def local_func_inv(fgraph, node):
    """
    Check for two consecutive operations that are functional inverses
    and remove them from the function graph.

    """
    inv_pairs = (
        (aes.Deg2Rad, aes.Rad2Deg),
        (aes.Cosh, aes.ArcCosh),
        (aes.Tanh, aes.ArcTanh),
        (aes.Sinh, aes.ArcSinh),
        (aes.Conj, aes.Conj),
        (aes.Neg, aes.Neg),
        (aes.Inv, aes.Inv),
    )
    x = node.inputs[0]

    if not isinstance(node.op, Elemwise):
        return
    if not x.owner or not isinstance(x.owner.op, Elemwise):
        return

    prev_op = x.owner.op.scalar_op
    node_op = node.op.scalar_op

    for inv_pair in inv_pairs:
        if is_inverse_pair(node_op, prev_op, inv_pair):
            # We don't need to copy stack trace, because the optimization
            # is trivial and maintains the earlier stack trace
            return x.owner.inputs

    return


@register_canonicalize
@register_specialize
@local_optimizer([Elemwise])
def local_exp_log(fgraph, node):
    x = node.inputs[0]

    if not isinstance(node.op, Elemwise):
        return
    if not x.owner or not isinstance(x.owner.op, Elemwise):
        return

    prev_op = x.owner.op.scalar_op
    node_op = node.op.scalar_op

    # Case for log(exp(x))
    if isinstance(prev_op, aes.Exp) and isinstance(node_op, aes.Log):
        return x.owner.inputs

    # Case for exp(log(x))
    if isinstance(prev_op, aes.Log) and isinstance(node_op, aes.Exp):
        x = x.owner.inputs[0]
        old_out = node.outputs[0]
        new_out = switch(ge(x, 0), x, np.asarray(np.nan, old_out.dtype))
        if new_out.type != old_out.type:
            return
        return [new_out]

    # Case for exp(log1p(x))
    if isinstance(prev_op, aes.Log1p) and isinstance(node_op, aes.Exp):
        x = x.owner.inputs[0]
        old_out = node.outputs[0]
        new_out = switch(ge(x, -1), add(1, x), np.asarray(np.nan, old_out.dtype))
        if new_out.type != old_out.type:
            return
        return [new_out]


@register_canonicalize
@register_specialize
@local_optimizer([Sum])
def local_sumsqr2dot(fgraph, node):
    """
    This optimization detects
    ``aet.sqr(W.dimshuffle("x", 0, 1) * G.dimshuffle(0, "x", 1) ).sum(axis=(1, 2))``
    and converts it to ``aet.dot(aet.sqr(G), aet.sqr(W).sum(axis=0))``.
    """
    if (
        isinstance(node.op, Sum)
        and isinstance(node.op.scalar_op, aes.Add)
        and node.op.axis == (1, 2)
    ):
        in1 = node.inputs[0]
        out = node.outputs[0]

        if (
            in1.owner
            and isinstance(in1.owner.op, Elemwise)
            and isinstance(in1.owner.op.scalar_op, aes.Sqr)
        ):
            in_sqr = in1.owner.inputs[0]
            if (
                in_sqr.owner
                and isinstance(in_sqr.owner.op, Elemwise)
                and isinstance(in_sqr.owner.op.scalar_op, aes.Mul)
                and len(in_sqr.owner.inputs) == 2
            ):
                in_mul1, in_mul2 = in_sqr.owner.inputs

                if (
                    isinstance(in_mul1.owner.op, DimShuffle)
                    and in_mul1.owner.op.new_order == ("x", 0, 1)
                    and isinstance(in_mul2.owner.op, DimShuffle)
                    and in_mul2.owner.op.new_order == (0, "x", 1)
                ):
                    W = in_mul1.owner.inputs[0]
                    G = in_mul2.owner.inputs[0]

                    new_out = dot(sqr(G), sqr(W).sum(axis=0))
                    if new_out.dtype != out.dtype:
                        new_out = cast(new_out, dtype=out.dtype)
                    return [new_out]


@register_stabilize
@register_specialize
@register_canonicalize
@local_optimizer([Elemwise])
def local_expm1(fgraph, node):
    """
    This optimization detects exp(a)-1 and converts this to expm1(a).
    """
    if isinstance(node.op, Elemwise) and isinstance(node.op.scalar_op, aes.Sub):
        in1, in2 = node.inputs
        out = node.outputs[0]

        if (
            in1.owner
            and isinstance(in1.owner.op, Elemwise)
            and isinstance(in1.owner.op.scalar_op, aes.Exp)
            and extract_constant(in2, only_process_constants=False) == 1
        ):
            in11 = in1.owner.inputs[0]
            new_out = expm1(in11)

            if new_out.dtype != out.dtype:
                new_out = cast(new_out, dtype=out.dtype)
            if new_out.type != out.type:
                return
            return [new_out]


@register_specialize
@register_canonicalize
@local_optimizer([mul])
def local_mul_switch_sink(fgraph, node):
    """
    This optimization makes the following changes in the graph:
    ``aet.mul(A, aet.switch(cond, 0, iff), B)`` -> ``aet.switch(cond, 0, aet.mul(A, B, iff))``
    ``aet.mul(A, aet.switch(cond, ift, 0), B)`` -> ``aet.switch(cond, aet.mul(A, B, ift), 0)``
    ``A`` and ``B`` being several (or none) symbolic variables.
    This is useful because ``A`` and ``B`` may not be numerically stable and give
    NaN or inf values for cases where the switch returns 0.
    With this optimization ``aet.grad(aet.switch(...))`` has the right behavior.

    Examples
    --------

        x -> f(x)
        x -> g(x)
        y = aet.switch(cond, f(x), g(x))

    without the optimization:

        aet.grad(y, x) -> grad(f(x), x) * grad(y, f(x)) + grad(g(x), x) * grad(y, g(x))

    with the optimization

        aet.grad(y, x) -> switch(cond, grad(f(x), x), 0) + switch(cond, 0, grad(g(x), x))

    This will be particularly useful for the lazy ``if`` because we skip an entire
    part of the graph.

    """
    if node.op != mul:
        return False
    for idx, i in enumerate(node.inputs):
        if i.owner and i.owner.op == switch:
            switch_node = i.owner
            try:
                if (
                    get_scalar_constant_value(
                        switch_node.inputs[1], only_process_constants=True
                    )
                    == 0.0
                ):
                    listmul = node.inputs[:idx] + node.inputs[idx + 1 :]
                    fmul = mul(*(listmul + [switch_node.inputs[2]]))

                    # Copy over stacktrace for elementwise multiplication op
                    # from previous elementwise multiplication op.
                    # An error in the multiplication (e.g. errors due to
                    # inconsistent shapes), will point to the
                    # multiplication op.
                    copy_stack_trace(node.outputs, fmul)

                    fct = [switch(switch_node.inputs[0], 0, fmul)]
                    fct[0].tag.values_eq_approx = values_eq_approx_remove_nan

                    # Copy over stacktrace for switch op from both previous
                    #  elementwise multiplication op and previous switch op,
                    # because an error in this part can be caused by either
                    # of the two previous ops.
                    copy_stack_trace(node.outputs + switch_node.outputs, fct)
                    return fct
            except NotScalarConstantError:
                pass
            try:
                if (
                    get_scalar_constant_value(
                        switch_node.inputs[2], only_process_constants=True
                    )
                    == 0.0
                ):
                    listmul = node.inputs[:idx] + node.inputs[idx + 1 :]
                    fmul = mul(*(listmul + [switch_node.inputs[1]]))
                    # Copy over stacktrace for elementwise multiplication op
                    # from previous elementwise multiplication op.
                    # An error in the multiplication (e.g. errors due to
                    # inconsistent shapes), will point to the
                    # multiplication op.
                    copy_stack_trace(node.outputs, fmul)

                    fct = [switch(switch_node.inputs[0], fmul, 0)]
                    fct[0].tag.values_eq_approx = values_eq_approx_remove_nan

                    # Copy over stacktrace for switch op from both previous
                    # elementwise multiplication op and previous switch op,
                    # because an error in this part can be caused by either
                    # of the two previous ops.
                    copy_stack_trace(node.outputs + switch_node.outputs, fct)
                    return fct
            except NotScalarConstantError:
                pass
    return False


@register_canonicalize
@local_optimizer([true_div, int_div])
def local_div_switch_sink(fgraph, node):
    """
    This optimization makes the following changes in the graph:

    ``aet.div(aet.switch(cond, 0, iff), A)`` -> ``aet.switch(cond, 0, aet.div(iff, A))``
    ``aet.div(aet.switch(cond, ift, 0), A)`` -> ``aet.switch(cond, aet.div(ift, A), 0)``

    where ``A`` is a symbolic variable.

    This is useful because ``A`` may not be numerically stable and give
    ``nan`` or ``inf`` values for cases where the switch returns 0.

    See `local_mul_switch_sink` for more details.

    """
    if node.op != true_div and node.op != int_div:
        return False
    op = node.op
    if node.inputs[0].owner and node.inputs[0].owner.op == switch:
        switch_node = node.inputs[0].owner
        try:
            if (
                get_scalar_constant_value(
                    switch_node.inputs[1], only_process_constants=True
                )
                == 0.0
            ):
                fdiv = op(switch_node.inputs[2], node.inputs[1])
                # Copy over stacktrace for elementwise division op
                # from previous elementwise multiplication op.
                # An error in the division (e.g. errors due to
                # inconsistent shapes or division by zero),
                # will point to the new division op.
                copy_stack_trace(node.outputs, fdiv)

                fct = [switch(switch_node.inputs[0], 0, fdiv)]
                fct[0].tag.values_eq_approx = values_eq_approx_remove_nan

                # Copy over stacktrace for switch op from both previous
                # elementwise division op and previous switch op,
                # because an error in this part can be caused by either
                # of the two previous ops.
                copy_stack_trace(node.outputs + switch_node.outputs, fct)
                return fct
        except NotScalarConstantError:
            pass
        try:
            if (
                get_scalar_constant_value(
                    switch_node.inputs[2], only_process_constants=True
                )
                == 0.0
            ):
                fdiv = op(switch_node.inputs[1], node.inputs[1])
                # Copy over stacktrace for elementwise division op
                # from previous elementwise multiplication op.
                # An error in the division (e.g. errors due to
                # inconsistent shapes or division by zero),
                # will point to the new division op.
                copy_stack_trace(node.outputs, fdiv)

                fct = [switch(switch_node.inputs[0], fdiv, 0)]
                fct[0].tag.values_eq_approx = values_eq_approx_remove_nan

                # Copy over stacktrace for switch op from both previous
                # elementwise division op and previous switch op,
                # because an error in this part can be caused by either
                # of the two previous ops.
                copy_stack_trace(node.outputs + switch_node.outputs, fct)
                return fct
        except NotScalarConstantError:
            pass
    return False


class AlgebraicCanonizer(LocalOptimizer):
    r"""
    Simplification tool. The variable is a local_optimizer. It is best used
    with a TopoOptimizer in in_to_out order.

    Usage: AlgebraicCanonizer(main, inverse, reciprocal, calculate)

    Parameters
    ----------
    main
        A suitable Op class that is commutative, associative and
        takes one to an arbitrary number of inputs, e.g. add or
        mul
    inverse
        An Op class such that inverse(main(x, y), y) == x
        e.g. sub or true_div
    reciprocal
        A function such that main(x, reciprocal(y)) == inverse(x, y)
        e.g. neg or inv
    calculate
        Function that takes a list of numpy.ndarray instances
        for the numerator, another list for the denumerator,
        and calculates inverse(main(\*num), main(\*denum)). It
        takes a keyword argument, aslist. If True, the value
        should be returned as a list of one element, unless
        the value is such that value = main(). In that case,
        the return value should be an empty list.

    Examples
    --------
    >>> import aesara.tensor as aet
    >>> from aesara.tensor.math_opt import AlgebraicCanonizer
    >>> add_canonizer = AlgebraicCanonizer(add, sub, neg, \\
    ...                                    lambda n, d: sum(n) - sum(d))
    >>> mul_canonizer = AlgebraicCanonizer(mul, true_div, inv, \\
    ...                                    lambda n, d: prod(n) / prod(d))

    Examples of optimizations mul_canonizer can perform:

    | x / x -> 1
    | (x * y) / x -> y
    | x / y / x -> 1 / y
    | x / y / z -> x / (y * z)
    | x / (y / z) -> (x * z) / y
    | (a / b) * (b / c) * (c / d) -> a / d
    | (2.0 * x) / (4.0 * y) -> (0.5 * x) / y
    | 2 * x / 2 -> x
    | x * y * z -> Elemwise(mul){x,y,z} #only one pass over the memory.
    |           !-> Elemwise(mul){x,Elemwise(mul){y,z}}

    """

    def __init__(self, main, inverse, reciprocal, calculate, use_reciprocal=True):
        self.main = main
        self.inverse = inverse
        self.reciprocal = reciprocal
        self.calculate = calculate
        self.use_reciprocal = use_reciprocal

        self.external_simplifiers = []

    def add_simplifier(self, simplifier, reason):
        self.external_simplifiers.append((reason, simplifier))

    def tracks(self):
        return [self.main, self.inverse, self.reciprocal]

    def get_num_denum(self, input):
        r"""
        This extract two lists, num and denum, such that the input is:
        self.inverse(self.main(\*num), self.main(\*denum)). It returns
        the two lists in a (num, denum) pair.

        For example, for main, inverse and reciprocal = \*, / and inv(),

        | input -> returned value (num, denum)

        | x*y -> ([x, y], [])
        | inv(x) -> ([], [x])
        | inv(x) * inv(y) -> ([], [x, y])
        | x*y/z -> ([x, y], [z])
        | log(x) / y * (z + x) / y -> ([log(x), z + x], [y, y])
        | (((a / b) * c) / d) -> ([a, c], [b, d])
        | a / (b / c) -> ([a, c], [b])
        | log(x) -> ([log(x)], [])
        | x**y -> ([x**y], [])
        | x * y * z -> ([x, y, z], [])

        """
        # This function is recursive.  The idea is that there is a
        # get_num_denum recursion in which the internal ops are all
        # one of (main, inverse, reciprocal, DimShuffle) and the
        # internal data nodes all have the dtype of the 'input'
        # argument. The leaf-Variables of the graph covered by the
        # recursion may be of any Variable type.

        if input.owner is None or input.owner.op not in [
            self.main,
            self.inverse,
            self.reciprocal,
        ]:
            if input.owner and isinstance(input.owner.op, DimShuffle):
                # If input is a DimShuffle of some input which does
                # something like this:

                # * change a vector of length N into a 1xN row matrix
                # * change a scalar into a 1x1x1 tensor
                # * in general, complete the shape of a tensor
                #   with broadcastable 1s to the *left*
                # Then we will simply discard the DimShuffle and return
                # the num/denum of its input
                dsn = input.owner  # dimshuffle node
                dsop = dsn.op  # dimshuffle op

                # the first input of the dimshuffle i.e. the ndarray to redim
                dsi0 = dsn.inputs[0]

                # The compatible order is a DimShuffle "new_order" of the form:
                # ('x', ..., 'x', 0, 1, 2, ..., dimshuffle_input.type.ndim)

                # That kind of DimShuffle only adds broadcastable
                # dimensions on the left, without discarding any
                # existing broadcastable dimension and is inserted
                # automatically by Elemwise when the inputs have
                # different numbers of dimensions (hence why we can
                # discard its information - we know we can retrieve it
                # later on).
                compatible_order = ("x",) * (input.type.ndim - dsi0.type.ndim) + tuple(
                    range(dsi0.type.ndim)
                )
                if dsop.new_order == compatible_order:
                    # If the "new_order" is the one we recognize,
                    # we return the num_denum of the dimshuffled input.
                    return self.get_num_denum(input.owner.inputs[0])
                else:
                    # This is when the input isn't produced by main,
                    # inverse or reciprocal.
                    return [input], []
            else:
                return [input], []
        num = []
        denum = []
        parent = input.owner

        # We get the (num, denum) pairs for each input
        # pairs = [self.get_num_denum(input2) if input2.type.dtype ==
        # input.type.dtype else ([input2], []) for input2 in
        # parent.inputs]
        pairs = [self.get_num_denum(input2) for input2 in parent.inputs]

        if parent.op == self.main:
            # If we have main(x, y, ...), numx, denumx, numy, denumy, ...
            # then num is concat(numx, numy, num...) and denum is
            # concat(denumx, denumy, denum...) note that main() can have any
            # number of arguments >= 0 concat is list concatenation
            num = reduce(list.__iadd__, map(operator.itemgetter(0), pairs))
            denum = reduce(list.__iadd__, map(operator.itemgetter(1), pairs))
        elif parent.op == self.inverse:
            # If we have inverse(x, y), numx, denumx, numy and denumy
            # then num is concat(numx, denumy) and denum is
            # concat(denumx, numy) note that inverse() is binary
            num = pairs[0][0] + pairs[1][1]
            denum = pairs[0][1] + pairs[1][0]
        elif parent.op == self.reciprocal:
            # If we have reciprocal(x), numx, denumx
            # then num is denumx and denum is numx
            # note that reciprocal() is unary
            num = pairs[0][1]
            denum = pairs[0][0]
        return num, denum

    def merge_num_denum(self, num, denum):
        r"""
        Utility function which takes two lists, num and denum, and
        returns something which is equivalent to inverse(main(\*num),
        main(\*denum)), but depends on the length of num and the length
        of denum (in order to minimize the number of operations).

        Let n = len(num) and d = len(denum):

        | n=0, d=0: neutral element (given by self.calculate([], []))
        |           (for example, this would be 0 if main is addition
        |           and 1 if main is multiplication)
        | n=1, d=0: num[0]
        | n=0, d=1: reciprocal(denum[0])
        | n=1, d=1: inverse(num[0], denum[0])
        | n=0, d>1: reciprocal(main(\*denum))
        | n>1, d=0: main(\*num)
        | n=1, d>1: inverse(num[0], main(\*denum))
        | n>1, d=1: inverse(main(\*num), denum[0])
        | n>1, d>1: inverse(main(\*num), main(\*denum))

        Given the values of n and d to which they are associated, all
        of the above are equivalent to:
        inverse(main(\*num), main(\*denum))

        """

        ln, ld = len(num), len(denum)
        if not ln and not ld:
            return as_tensor_variable(self.calculate([], []))
        if not ln:
            if self.use_reciprocal:
                return self.reciprocal(self.merge_num_denum(denum, []))
            else:
                ln = [self.calculate([], [], aslist=False)]
        if not ld:
            if ln == 1:
                # num[0] should always be a variable
                assert isinstance(num[0], Variable)
                return num[0]
            else:
                return self.main(*num)
        return self.inverse(
            self.merge_num_denum(num, []), self.merge_num_denum(denum, [])
        )

    @staticmethod
    def get_constant(v):
        """

        Returns
        -------
        object
            A numeric constant if v is a Constant or, well, a
            numeric constant. If v is a plain Variable, returns None.

        """
        if isinstance(v, Constant):
            if getattr(v.tag, "unique_value", None) is not None:
                data = v.tag.unique_value
            else:
                data = v.data
            if data.ndim == 0:
                return data
            else:
                return None
        elif isinstance(v, Variable):
            return None
        else:
            return v

    def simplify(self, num, denum, out_type):
        """
        Shorthand for:

        .. code-block:: python

            self.simplify_constants(*self.simplify_factors(num, denum))

        """
        rval = self.simplify_constants(
            *self.simplify_factors(num, denum), out_type=out_type
        )
        for reason, simplifier in self.external_simplifiers:
            # TODO: document that 'reason' is associated with this
            #       simplification to help auditing when things go
            #       wrong
            rval = simplifier(*rval)
        return rval

    def simplify_factors(self, num, denum):
        """
        For any Variable r which is both in num and denum, removes it
        from both lists. Modifies the lists inplace. Returns the
        modified lists. For example:

        | [x], [x] -> [], []
        | [x, y], [x] -> [y], []
        | [a, b], [c, d] -> [a, b], [c, d]

        """
        ln = len(num)
        ld = len(denum)
        if ld > 2 and ln > 2:
            # Faster version for "big" inputs.
            while True:
                s = set(num)
                # Inputs can appear multiple times
                redo = len(s) != len(num)
                inter = s.intersection(denum)
                for v in inter:
                    num.remove(v)
                    denum.remove(v)
                if not redo or not inter:
                    break
        else:
            for v in list(num):
                if v in denum:
                    num.remove(v)
                    denum.remove(v)
        return num, denum

    def simplify_constants(self, orig_num, orig_denum, out_type=None):
        """
        Find all constants and put them together into a single constant.

        Finds all constants in orig_num and orig_denum (using
        get_constant) and puts them together into a single
        constant. The constant is inserted as the first element of the
        numerator. If the constant is the neutral element, it is
        removed from the numerator.

        Examples
        --------
        Let main be multiplication:

        | [2, 3, x], [] -> [6, x], []
        | [x, y, 2], [4, z] -> [0.5, x, y], [z]
        | [x, 2, y], [z, 2] -> [x, y], [z]

        """
        # Lists representing the numerator and denumerator
        num, denum = [], []

        # Lists representing the *constant* elements of num and denum
        numct, denumct = [], []

        for v in orig_num:
            ct = self.get_constant(v)
            if ct is not None:
                # We found a constant in the numerator!
                # We add it to numct
                numct.append(ct)
            else:
                num.append(v)
        for v in orig_denum:
            ct = self.get_constant(v)
            if ct is not None:
                denumct.append(ct)
            else:
                denum.append(v)

        if self.use_reciprocal or num:
            # This will calculate either:
            # [inverse(main(*numct), main(*denumct))]
            # [] - if inverse(main(*numct), main(*denumct)) is the
            # neutral element
            ct = self.calculate(numct, denumct, aslist=True, out_type=out_type)
        else:
            # This happens if we don't allow the reciprocal and the
            # numerator is empty. That means we will need to represent
            # reciprocal(x) like inverse(neutral_element, x) so
            # we can't allow ct == []
            # TODO: why is this branch needed when merge_num_denum
            # does it for us?
            ct = [self.calculate(numct, denumct, aslist=False, out_type=out_type)]

        # Wrapping ct in a Constant with the right dtype
        ct = [constant(c, dtype=out_type.dtype) for c in ct]

        if orig_num and len(numct) == 1 and len(denumct) == 0 and ct:
            # In that case we should only have one constant in `ct`.
            assert len(ct) == 1
            first_num_ct = self.get_constant(orig_num[0])
            if first_num_ct is not None and ct[0].type.values_eq(
                ct[0].data, first_num_ct
            ):
                # This is an important trick :( if it so happens that:
                # * there's exactly one constant on the numerator and none on
                #   the denominator
                # * it's not the neutral element (ct is an empty list in that
                #   case)
                # * the constant is the same as the first argument in the
                #   numerator (we only check the first argument because the
                #   canonizer puts the computed constants first)
                # -> then we return very exactly the original num/denum.
                # If we don't do that the optimizer will just loop
                # infinitely because it will not catch on that there are
                # no changes to be made and every time it will want to
                # replace something by the same thing...
                # Note that it is important to use `values_eq` instead of
                # the == operator, to handle NaN values correctly.
                return orig_num, orig_denum

        return ct + num, denum

    def transform(self, fgraph, node):
        op = node.op
        if op not in [self.main, self.inverse, self.reciprocal]:
            return False

        assert len(node.outputs) == 1
        out = node.outputs[0]

        out_clients = fgraph.clients.get(out)

        if not out_clients:
            return False

        # check if any of the clients of this node would be part of
        # this canonized graph...  if so, we do nothing and wait for
        # them to be transformed.
        for c, c_idx in out_clients:
            if c == "output":
                continue
            while (
                isinstance(getattr(c, "op", None), DimShuffle)
                and len(fgraph.clients[c.outputs[0]]) <= 1
            ):
                c = fgraph.clients[c.outputs[0]][0][0]
            if getattr(c, "op", "") in [self.main, self.inverse, self.reciprocal]:
                return False

        # Here we make the canonical version of the graph around this node
        # See the documentation of get_num_denum and simplify
        orig_num, orig_denum = self.get_num_denum(node.outputs[0])
        num, denum = self.simplify(list(orig_num), list(orig_denum), out.type)

        def same(x, y):
            return len(x) == len(y) and all(np.all(xe == ye) for xe, ye in zip(x, y))

        if (
            same(orig_num, num)
            and same(orig_denum, denum)
            and
            # Check to see if we've collapsed some nested ops.
            not (
                len(orig_denum) == 0
                and
                # Make sure this change would increase the number of vector
                # arguments--decreasing the number of unnecessary `self.main`
                # nodes.
                len(node.inputs) < len(orig_num)
            )
            and
            # Do a similar check for the reciprocal op.
            not (
                self.use_reciprocal
                and node.op == self.reciprocal
                and len(orig_num) == 0
                and node.inputs[0].owner
                and len(node.inputs[0].owner.inputs) < len(orig_denum)
            )
        ):
            return False

        new = self.merge_num_denum(num, denum)
        if new.type.dtype != out.type.dtype:
            new = cast(new, out.type.dtype)

        assert (new.type == out.type) == (not (new.type != out.type))

        if not (new.type == out.type):
            new = _fill_chain(new, node.inputs)[0]

        if new.type == out.type:
            # This happen with test
            # aesara/tensor/tests/test_opt.py:T_local_switch_sink
            new.tag.values_eq_approx = values_eq_approx_remove_inf_nan

            # We need to implement the copy over of the stacktrace.
            # See issue #5104.
            return [new]
        else:
            _logger.warning(
                " ".join(
                    (
                        "CANONIZE FAILED: new, out = ",
                        new,
                        ",",
                        out,
                        "types",
                        new.type,
                        ",",
                        out.type,
                    )
                )
            )
            return False

    def __str__(self):
        return getattr(
            self,
            "name",
            f"AlgebraicCanonizer({self.main}, {self.inverse}, {self.reciprocal})",
        )


def mul_calculate(num, denum, aslist=False, out_type=None):
    if not num and not denum:
        # Smallest 1 possible.
        if aslist:
            return []
        else:
            return np.int8(1)

    # Make sure we do not accidentally upcast data types.
    if out_type is None:
        out_dtype = aes.upcast(*[v.dtype for v in (num + denum)])
    else:
        out_dtype = out_type.dtype
    one = _asarray(1, dtype=out_dtype)

    v = reduce(np.multiply, num, one) / reduce(np.multiply, denum, one)
    if aslist:
        if np.all(v == 1):
            return []
        else:
            return [v]
    return v


local_mul_canonizer = AlgebraicCanonizer(mul, true_div, inv, mul_calculate, False)
register_canonicalize(local_mul_canonizer, name="local_mul_canonizer")


@local_optimizer([neg])
def local_neg_to_mul(fgraph, node):
    if node.op == neg:
        return [mul(np.array(-1, dtype=node.inputs[0].dtype), node.inputs[0])]


register_canonicalize(local_neg_to_mul)


@register_specialize
@local_optimizer([Sum, Prod])
def local_sum_prod_mul_by_scalar(fgraph, node):
    """
    sum(scalar * smth) -> scalar * sum(smth)
    sum(-smth) -> -sum(smth)

    or

    prod(scalar * smth) -> scalar ** size(smth) * prod(smth)
    prod(-smth) -> -1 ** size(smth) * prod(smth)

    """
    # TODO: if the the thing inside the Sum is a division,
    # we should get at the numerator....
    if isinstance(node.op, (Sum, Prod)):
        (node_inps,) = node.inputs
        if node_inps.owner and node_inps.owner.op == mul:
            terms = node_inps.owner.inputs
            scalars = [t.dimshuffle() for t in terms if np.all(t.type.broadcastable)]

            if len(scalars) == 0:
                # Nothing to optimize here
                return

            non_scalars = [t for t in terms if not np.all(t.broadcastable)]

            # Perform the op only on the non-scalar inputs, if applicable
            if len(non_scalars) == 0:
                new_op_input_nb_elements = 1
                new_op_output = 1
            elif len(non_scalars) == 1:
                new_op_input_nb_elements = non_scalars[0].size
                new_op_output = node.op(non_scalars[0])
            else:
                new_op_input = mul(*non_scalars)
                # We assume that errors always come from the prod/mul op in the
                # original computational graph, and therefore need to only
                # copy over its output stacktrace.
                copy_stack_trace(node.outputs, new_op_input)

                new_op_input_nb_elements = new_op_input.size
                new_op_output = node.op(new_op_input)

            if not len(non_scalars) == 0:
                # Copy over stacktrace from previous output to new mul op,
                # for same reason as above.
                copy_stack_trace(node.outputs, new_op_output)

            # If `node.op` is a `Prod`, then the scalars need to be raised to
            # the power of the number of elements in the input to the `Prod`
            if isinstance(node.op, Prod) and new_op_input_nb_elements != 1:

                scalars = [s ** new_op_input_nb_elements for s in scalars]

            # Scale the output of the op by the scalars and return as
            # replacement for the original output
            mul_inputs = scalars
            if new_op_input_nb_elements != 1:
                mul_inputs.append(new_op_output)

            if len(mul_inputs) == 1:
                # Copy over stacktrace from previous output to new mul op,
                # for same reason as above.
                copy_stack_trace(node.outputs, mul_inputs)

                return mul_inputs
            else:
                ret = mul(*mul_inputs)
                # Copy over stacktrace from previous output to new mul op,
                # for same reason as above.
                copy_stack_trace(node.outputs, [ret] + mul_inputs)

                return [ret]

        if isinstance(node.op, Sum) and node_inps.owner and node_inps.owner.op == neg:
            s = node.op(node_inps.owner.inputs[0])
            ret = neg(s)
            # There are never errors in the negative op, thus
            # we need only to copy over stacktrace from previous output node to
            # the two new ops.
            copy_stack_trace(node.outputs, [s, ret])

            return [ret]


@register_specialize
@local_optimizer([Elemwise])
def local_elemwise_sub_zeros(fgraph, node):
    """
    Elemwise{sub}(X,X) -> zeros_like(X)
    """
    if (
        isinstance(node.op, Elemwise)
        and node.op.scalar_op.nin == 2
        and node.op.scalar_op == aes.sub
        and node.inputs[0] == node.inputs[1]
    ):
        res = zeros_like(node.inputs[0])
        # Copy over stacktrace from previous output.
        # This could help for failures due to out-of-memory.
        copy_stack_trace(node.outputs, res)
        return [res]


@register_useless
@register_specialize
@register_stabilize
@register_canonicalize
@local_optimizer([Elemwise])
def local_useless_elemwise_comparison(fgraph, node):
    """...

    :note: These cases appear in the graph generated by scan.
           These optimizations will make the graph easier to read.
    # Comparing to itself is constant
    Elemwise[{LT,GT}](X, X) -> Elemwise[zeros](X)
    Elemwise[{LE,GE}](X, X) -> Elemwise[ones](X)
    Elemwise[{minimum,maximum}](X, X) -> X

    # Comparing shape to 0 can be constant
    Elemwise[LT](X.shape[i], 0) -> Elemwise[zeros](X)
    Elemwise[GE](X.shape[i], 0) -> Elemwise[ones](X)
    Elemwise[maximum](X.shape[i], 0) -> X.shape[i]
    Elemwise[maximum](0, X.shape[i]) -> X.shape[i]
    Elemwise[minimum](X.shape[i], 0) -> 0
    Elemwise[minimum](0, X.shape[i]) -> 0

    # The shape can be replaced with sum of shapes
    Elemwise[LT](add([anything that is shapes]), 0) -> Elemwise[zeros](X)
    Elemwise[GE](add([anything that is shapes]), 0) -> Elemwise[ones](X)

    # Shapes are never negative
    # Needed by Reshape.infer_shape
    Elemwise[EQ](Subtensor(Shape(x)), -N) -> Elemwise[zeros](X)

    """
    if not isinstance(node.op, Elemwise):
        return
    if node.op.scalar_op.nin != 2:
        return

    # We call zeros_like and one_like with opt=True to generate a
    # cleaner graph.
    dtype = node.outputs[0].dtype

    # Elemwise[{LT,GT}](X, X) -> Elemwise[zeros](X)
    if (
        isinstance(node.op.scalar_op, (aes.LT, aes.GT))
        and node.inputs[0] is node.inputs[1]
    ):
        res = zeros_like(node.inputs[0], dtype=dtype, opt=True)
        # Copy over stacktrace from previous output.
        copy_stack_trace(node.outputs, res)
        return [res]
    # Elemwise[{LE,GE}](X, X) -> Elemwise[ones](X)
    if (
        isinstance(node.op.scalar_op, (aes.LE, aes.GE))
        and node.inputs[0] is node.inputs[1]
    ):
        res = ones_like(node.inputs[0], dtype=dtype, opt=True)

        # Copy over stacktrace from previous output.
        copy_stack_trace(node.outputs, res)
        return [res]
    # Elemwise[{minimum,maximum}](X, X) -> X
    if (
        isinstance(node.op.scalar_op, (aes.ScalarMinimum, aes.ScalarMaximum))
        and node.inputs[0] is node.inputs[1]
    ):
        res = node.inputs[0]
        # Copy over stacktrace from previous output.
        copy_stack_trace(node.outputs, res)
        return [res]

    # Elemwise[LT](X.shape[i], 0) -> Elemwise[zeros](X)
    if (
        isinstance(node.op.scalar_op, aes.LT)
        and node.inputs[0].owner
        and isinstance(node.inputs[0].owner.op, Shape_i)
        and extract_constant(node.inputs[1], only_process_constants=True) == 0
    ):
        res = zeros_like(node.inputs[0], dtype=dtype, opt=True)
        # Copy over stacktrace from previous output.
        copy_stack_trace(node.outputs, res)
        return [res]
    # Elemwise[GE](X.shape[i], 0) -> Elemwise[ones](X)
    if (
        isinstance(node.op.scalar_op, aes.GE)
        and node.inputs[0].owner
        and isinstance(node.inputs[0].owner.op, Shape_i)
        and extract_constant(node.inputs[1], only_process_constants=True) == 0
    ):
        res = ones_like(node.inputs[0], dtype=dtype, opt=True)
        # Copy over stacktrace from previous output.
        copy_stack_trace(node.outputs, res)
        return [res]
    # Elemwise[maximum](X.shape[i], 0) -> X.shape[i]
    if (
        isinstance(node.op.scalar_op, aes.ScalarMaximum)
        and node.inputs[0].owner
        and isinstance(node.inputs[0].owner.op, Shape_i)
        and extract_constant(node.inputs[1], only_process_constants=True) == 0
    ):
        # No need to copy over stacktrace.
        return [node.inputs[0]]
    # Elemwise[maximum](0, X.shape[i]) -> X.shape[i]
    if (
        isinstance(node.op.scalar_op, aes.ScalarMaximum)
        and extract_constant(node.inputs[0], only_process_constants=True) == 0
        and node.inputs[1].owner
        and isinstance(node.inputs[1].owner.op, Shape_i)
    ):
        # No need to copy over stacktrace.
        return [node.inputs[1]]
    # Elemwise[minimum](X.shape[i], 0) -> 0
    if (
        isinstance(node.op.scalar_op, aes.ScalarMinimum)
        and node.inputs[0].owner
        and isinstance(node.inputs[0].owner.op, Shape_i)
        and extract_constant(node.inputs[1], only_process_constants=True) == 0
    ):
        res = zeros_like(node.inputs[0], dtype=dtype, opt=True)
        # Copy over stacktrace from previous output.
        copy_stack_trace(node.outputs, res)
        return [res]

    # Elemwise[minimum](0, X.shape[i]) -> 0
    if (
        isinstance(node.op.scalar_op, aes.ScalarMinimum)
        and extract_constant(node.inputs[0], only_process_constants=True) == 0
        and node.inputs[1].owner
        and isinstance(node.inputs[1].owner.op, Shape_i)
    ):
        res = zeros_like(node.inputs[1], dtype=dtype, opt=True)
        # Copy over stacktrace from previous output.
        copy_stack_trace(node.outputs, res)
        return [res]

    # Elemwise[LT](add([anything that is shapes]), 0) -> Elemwise[zeros](X)
    if (
        isinstance(node.op.scalar_op, aes.LT)
        and node.inputs[0].owner
        and isinstance(node.inputs[0].owner.op, Elemwise)
        and isinstance(node.inputs[0].owner.op.scalar_op, aes.Add)
        and all(
            [
                isinstance(var.owner and var.owner.op, Shape_i)
                for var in node.inputs[0].owner.inputs
            ]
        )
        and extract_constant(node.inputs[1], only_process_constants=True) == 0
    ):
        res = zeros_like(node.inputs[0], dtype=dtype, opt=True)
        # Copy over stacktrace from previous output.
        copy_stack_trace(node.outputs, res)
        return [res]
    # Elemwise[GE](add([anything that is shapes]), 0) -> Elemwise[ones](X)
    if (
        isinstance(node.op.scalar_op, aes.GE)
        and node.inputs[0].owner
        and isinstance(node.inputs[0].owner.op, Elemwise)
        and isinstance(node.inputs[0].owner.op.scalar_op, aes.Add)
        and all(
            [
                isinstance(var.owner and var.owner.op, Shape_i)
                for var in node.inputs[0].owner.inputs
            ]
        )
        and extract_constant(node.inputs[1], only_process_constants=True) == 0
    ):
        res = ones_like(node.inputs[0], dtype=dtype, opt=True)

        # Copy over stacktrace from previous output.
        copy_stack_trace(node.outputs, res)
        return [res]

        # Elemwise[EQ](Subtensor(Shape(x)), -N)
        # Elemwise[EQ](somegraph that only depend of shape, -N)
        # TODO: handle the case where the -N is on either side
        """
 |Elemwise{eq,no_inplace} [id B] ''
 | |Subtensor{int64} [id C] ''
 | | |Join [id D] ''
 | | | |TensorConstant{0} [id E]
 | | | |Subtensor{int64:int64:} [id F] ''
 | | | | |Shape [id G] ''
        """

    def investigate(node):
        " Return True if values will be shapes, so >= 0"
        if isinstance(node.op, (Shape, Shape_i)):
            return True
        elif isinstance(node.op, Subtensor) and node.inputs[0].owner:
            return investigate(node.inputs[0].owner)
        elif isinstance(node.op, Join):
            return all(v.owner and investigate(v.owner) for v in node.inputs[1:])
        elif isinstance(node.op, MakeVector):
            return all(v.owner and investigate(v.owner) for v in node.inputs)

    if (
        isinstance(node.op.scalar_op, aes.EQ)
        and node.inputs[0].owner
        and investigate(node.inputs[0].owner)
    ):
        try:
            cst = get_scalar_constant_value(node.inputs[1], only_process_constants=True)

            res = zeros_like(node.inputs[0], dtype=dtype, opt=True)

            if cst < 0:
                # Copy over stacktrace from previous output.
                copy_stack_trace(node.outputs, res)

                return [res]

        except NotScalarConstantError:
            pass
    return


@register_canonicalize
@register_specialize
@local_optimizer([Sum, Prod])
def local_sum_prod_div_dimshuffle(fgraph, node):
    """
    sum(a / dimshuffle{...}(b), axis=l) -> sum(a, axis={...}) / b,
    if dimension l of the DimShuffle is 'x'

    or

    prod(a / dimshuffle{...}(b), axis=l) ->
    prod(a, axis={...}) / b ** a.shape[l],
    if dimension l of the DimShuffle is 'x'
    """

    # It does not make much sense now to extend it to the case where the
    # dimshuffle is in the numerator, since elemwise inversion of the
    # denominator would still be needed before the summation or production.

    if isinstance(node.op, (Sum, Prod)):
        axis = node.op.axis
        if axis is None:
            axis = list(range(node.inputs[0].ndim))
        node_input = node.inputs[0]
        if node_input.owner and node_input.owner.op == true_div:
            numerator, denominator = node_input.owner.inputs

            # Old, bugged logic, reproduced here only to warn users
            if (
                config.warn__sum_div_dimshuffle_bug
                and isinstance(node.op, Sum)
                and numerator.owner
                and isinstance(numerator.owner.op, DimShuffle)
            ):
                # Check compatibility
                new_order = numerator.owner.op.new_order
                compatible_dims = True
                for ax in axis:
                    if len(new_order) <= ax or new_order[ax] != "x":
                        compatible_dims = False
                        break

                if compatible_dims:
                    _logger.warning(
                        "Your current code is fine, but"
                        " Aesara versions between "
                        "rev. 3bd9b789f5e8 (2010-06-16) and"
                        " cfc6322e5ad4 (2010-08-03) would "
                        "have given an incorrect result. "
                        "To disable this warning, set the Aesara"
                        " flag warn__sum_div_dimshuffle_bug to"
                        " False."
                    )

            if denominator.owner and isinstance(denominator.owner.op, DimShuffle):
                dimshuffle_input = denominator.owner.inputs[0]
                dimshuffle_order = denominator.owner.op.new_order

                compatible_dims = []
                incompatible_dims = []
                for ax in axis:
                    if ax < len(dimshuffle_order) and dimshuffle_order[ax] == "x":
                        compatible_dims.append(ax)
                    else:
                        incompatible_dims.append(ax)
                reordered_incompatible_dims = []
                for ic_ax in incompatible_dims:
                    reordered_incompatible_dims.append(
                        ic_ax - sum([1 for c_ax in compatible_dims if c_ax < ic_ax])
                    )

                if len(compatible_dims) > 0:
                    optimized_dimshuffle_order = list(
                        ax
                        for i, ax in enumerate(dimshuffle_order)
                        if (i not in axis) or (ax != "x")
                    )

                    # Removing leading 'x' (since it will be done automatically)
                    while (
                        len(optimized_dimshuffle_order) > 0
                        and optimized_dimshuffle_order[0] == "x"
                    ):
                        del optimized_dimshuffle_order[0]

                    # if optimized_dimshuffle_order is sorted with
                    # not 'x', then dimshuffle is useless.
                    if all(i == e for i, e in enumerate(optimized_dimshuffle_order)):
                        optimized_dimshuffle = dimshuffle_input
                    else:
                        optimized_dimshuffle = DimShuffle(
                            dimshuffle_input.type.broadcastable,
                            optimized_dimshuffle_order,
                        )(dimshuffle_input)

                        if config.warn__sum_div_dimshuffle_bug and isinstance(
                            node.op, Sum
                        ):
                            _logger.warning(
                                "Your current code is fine,"
                                " but Aesara versions between "
                                "rev. 3bd9b789f5e8 (2010-06-16) and"
                                " cfc6322e5ad4 (2010-08-03) would "
                                "have given an incorrect result. "
                                "To disable this warning, set the"
                                " Aesara flag "
                                "warn__sum_div_dimshuffle_bug"
                                " to False."
                            )

                    if isinstance(node.op, Sum):
                        op_on_compatible_dims = aet_sum(numerator, axis=compatible_dims)
                        rval = true_div(op_on_compatible_dims, optimized_dimshuffle)
                        if len(reordered_incompatible_dims) > 0:
                            rval = aet_sum(rval, axis=reordered_incompatible_dims)
                    elif isinstance(node.op, Prod):
                        op_on_compatible_dims = prod(numerator, axis=compatible_dims)
                        dtype = numerator.dtype
                        rval = true_div(
                            op_on_compatible_dims,
                            (
                                optimized_dimshuffle
                                ** prod(
                                    [
                                        numerator.shape[ax].astype(dtype)
                                        for ax in compatible_dims
                                    ]
                                )
                            ),
                        )
                        if len(reordered_incompatible_dims) > 0:
                            rval = prod(rval, axis=reordered_incompatible_dims)
                    return [rval]


@register_canonicalize
@local_optimizer([Sum, Prod])
def local_sum_prod_all_to_none(fgraph, node):
    """
    Sum{0,1,...N} -> Sum{} or
    Prod{0,1,...N} -> Prod{}

    """
    if isinstance(node.op, Sum) or isinstance(node.op, Prod):
        opt_type = Sum if isinstance(node.op, Sum) else Prod
        # if all the axes are named, then use None as a shorthand
        # this permits more merging
        if node.op.axis is None:
            return
        if set(node.op.axis) == set(range(node.inputs[0].type.ndim)):
            return [opt_type(axis=None, dtype=node.op.dtype)(node.inputs[0])]


@register_canonicalize
@local_optimizer([Sum, Prod])
def local_op_of_op(fgraph, node):
    """
    Prod(Prod()) -> single Prod()
    or
    Sum(Sum()) -> single Sum()

    """
    if isinstance(node.op, Prod) or isinstance(node.op, Sum):
        opt_type = Sum if isinstance(node.op, Sum) else Prod
        (node_inps,) = node.inputs
        out_dtype = node.op.dtype
        # We manipulate the graph so this is done to make sure the opt
        # doesn't affect other computations.
        if len(fgraph.clients[node_inps]) == 1:
            if node_inps.owner and (isinstance(node_inps.owner.op, node.op.__class__)):

                # check to see either the inner or outer prod is doing a
                # product over all axis, in which case we can remove it
                if node_inps.owner.op.axis is None or node.op.axis is None:
                    return [opt_type(None, dtype=out_dtype)(node_inps.owner.inputs[0])]

                # figure out which axes were in the original sum
                newaxis = list(tuple(node_inps.owner.op.axis))
                for i in node.op.axis:
                    new_i = i
                    for ii in node_inps.owner.op.axis:
                        if new_i >= ii:
                            new_i += 1
                    assert new_i not in newaxis
                    newaxis.append(new_i)

                assert len(newaxis) == len(
                    list(node_inps.owner.op.axis) + list(node.op.axis)
                )

                # The old bugged logic. We keep it there to generate a warning
                # when we generated bad code.
                alldims = list(range(node_inps.owner.inputs[0].type.ndim))
                alldims = [
                    d for i, d in enumerate(alldims) if i in node_inps.owner.op.axis
                ]
                alldims = [d for i, d in enumerate(alldims) if i in node.op.axis]
                newaxis_old = [
                    i
                    for i in range(node_inps.owner.inputs[0].type.ndim)
                    if i not in alldims
                ]

                if (
                    config.warn__sum_sum_bug
                    and newaxis != newaxis_old
                    and len(newaxis) == len(newaxis_old)
                ):
                    _logger.warning(
                        "(YOUR CURRENT CODE IS FINE): Aesara "
                        "versions between version 9923a40c7b7a and August "
                        "2nd, 2010 generated bugged code in this case. "
                        "This happens when there are two consecutive sums "
                        "in the graph and the intermediate sum is not "
                        "used elsewhere in the code. Some safeguard "
                        "removed some bad code, but not in all cases. You "
                        "are in one such case. To disable this warning "
                        "(that you can safely ignore since this bug has "
                        "been fixed) set the aesara flag "
                        "`warn__sum_sum_bug` to False."
                    )

                combined = opt_type(newaxis, dtype=out_dtype)
                return [combined(node_inps.owner.inputs[0])]


ALL_REDUCE = (
    [
        CAReduce,
        All,
        Any,
        Sum,
        Prod,
        ProdWithoutZeros,
    ]
    + CAReduce.__subclasses__()
    + NonZeroCAReduce.__subclasses__()
)


@register_canonicalize
@register_uncanonicalize  # Needed for MaxAndArgmax -> CAReduce
@local_optimizer(ALL_REDUCE)
def local_reduce_join(fgraph, node):
    """
    Reduce{scalar.op}(Join(axis=0, a, b), axis=0) -> Elemwise{scalar.op}(a, b)

    Notes
    -----
    Supported scalar.op are Maximum, Mimimum in some cases and Add and Mul in
    all cases.

    Currently we must reduce on axis 0. It is probably extensible to the case
    where we join and reduce on the same set of axis.

    """
    if (
        isinstance(node.op, CAReduce)
        and node.inputs[0].owner
        and isinstance(node.inputs[0].owner.op, Join)
    ):
        join_node = node.inputs[0].owner
        if extract_constant(join_node.inputs[0], only_process_constants=True) != 0:
            return

        if isinstance(node.op.scalar_op, (aes.ScalarMaximum, aes.ScalarMinimum)):
            # Support only 2 inputs for now
            if len(join_node.inputs) != 3:
                return
        elif not isinstance(node.op.scalar_op, (aes.Add, aes.Mul)):
            return
        elif len(join_node.inputs) <= 2:
            # This is a useless join, that will get removed by another opt.
            return

        new_inp = []
        for inp in join_node.inputs[1:]:
            inp = inp.owner
            if not inp:
                return
            if not isinstance(inp.op, DimShuffle) or inp.op.new_order != ("x",) + tuple(
                range(inp.inputs[0].ndim)
            ):
                return
            new_inp.append(inp.inputs[0])
        ret = Elemwise(node.op.scalar_op)(*new_inp)

        if ret.dtype != node.outputs[0].dtype:
            # The reduction do something about the dtype.
            return

        reduce_axis = node.op.axis
        if reduce_axis is None:
            reduce_axis = tuple(range(node.inputs[0].ndim))

        # I put this warning late to don't add extra warning.
        if len(reduce_axis) != 1 or 0 not in reduce_axis:
            if config.warn__reduce_join:
                warnings.warning(
                    "Your current code is fine, but Aesara versions "
                    "prior to 0.7 (or this development version Sept 2014) "
                    "might have given an incorrect result for this code. "
                    "To disable this warning, set the Aesara flag "
                    "warn__reduce_join to False. The problem was an "
                    "optimization, that modified the pattern "
                    '"Reduce{scalar.op}(Join(axis=0, a, b), axis=0)", '
                    "did not check the reduction axis. So if the "
                    "reduction axis was not 0, you got a wrong answer."
                )
            return

        # We add the new check late to don't add extra warning.
        try:
            join_axis = get_scalar_constant_value(
                join_node.inputs[0], only_process_constants=True
            )

            if join_axis != reduce_axis[0]:
                return
        except NotScalarConstantError:
            return

        return [ret]


@register_canonicalize("fast_compile", "local_cut_useless_reduce")
@register_useless("local_cut_useless_reduce")
@local_optimizer(ALL_REDUCE)
def local_useless_reduce(fgraph, node):
    """Sum(a, axis=[]) -> a  """
    if isinstance(node.op, CAReduce):
        (summed,) = node.inputs
        # if reduce were doing anything, the output ndim would be reduced
        if summed.type == node.outputs[0].type:
            return [summed]


@register_canonicalize
@register_uncanonicalize
@register_specialize
@local_optimizer(ALL_REDUCE)
def local_reduce_broadcastable(fgraph, node):
    """Remove reduction over broadcastable dimensions."""
    if isinstance(node.op, CAReduce):
        (reduced,) = node.inputs
        odtype = node.outputs[0].dtype
        if node.op.axis is None:
            if all(reduced.broadcastable):
                return [reduced.dimshuffle().astype(odtype)]
        else:
            axis = list(node.op.axis)
            cuttable = [a for a in axis if reduced.broadcastable[a]]
            if cuttable:
                # -- we can remove some axes of summation,
                #    which simplifies the codegen for sum, especially on GPU
                new_axis = []
                pattern = []
                ii = 0
                for p in range(reduced.ndim):
                    if p not in cuttable:
                        if p in axis:
                            new_axis.append(ii)
                        pattern.append(p)
                        ii += 1
                new_reduced = reduced.dimshuffle(*pattern)
                if new_axis:
                    if type(node.op) == CAReduce:
                        # This happen for aet_max(), aet_min()
                        new_op = node.op.__class__(node.op.scalar_op, axis=new_axis)
                    else:
                        new_op = node.op.__class__(axis=new_axis)
                    return [new_op(new_reduced)]
                else:
                    # -- in this case we can remove the reduction completely
                    return [new_reduced.astype(odtype)]


@register_specialize
@local_optimizer([Sum, Prod])
def local_opt_alloc(fgraph, node):
    """
    sum(alloc(constant,shapes...)) => constant*prod(shapes)
    or
    prod(alloc(constant,shapes...)) => constant**prod(shapes)

    """
    if isinstance(node.op, Sum) or isinstance(node.op, Prod):
        (node_inps,) = node.inputs
        if node_inps.owner and isinstance(node_inps.owner.op, Alloc):
            input = node_inps.owner.inputs[0]
            shapes = node_inps.owner.inputs[1:]
            try:
                val = get_scalar_constant_value(input, only_process_constants=True)
                assert val.size == 1
                val = val.reshape(1)[0]
                # check which type of op
                size = mul(*shapes)
                if input.dtype in ["float16", "float32"]:
                    # shapes are ints and normally int64.
                    # We don't want to have a float64 upcast
                    # We don't want to downcast to float16
                    # as we fear it could loose too much precision
                    # that will be amplified by the mul/pow below.
                    size = size.astype("float32")
                if node.op.axis is None or node.op.axis == tuple(range(input.ndim)):
                    if isinstance(node.op, Sum):
                        val = val * size
                    else:
                        val = val ** size
                    # Sum can change the input dtype (upcast or bool
                    # -> float32) by default or by user request.
                    # We can ignore the acc_dtype, as there is only 1
                    # elemwise we will do and not a sequence, so there is no
                    # accumulation of errors.
                    # So mostly, we just need to cast the output to the old
                    # dtype.
                    val = val.astype(node.outputs[0].dtype)
                    return [val]
                to_prod = [shapes[i] for i in range(len(shapes)) if i in node.op.axis]
                if to_prod:
                    size = mul(*to_prod)
                    if isinstance(node.op, Sum):
                        val *= size
                    else:
                        val = val ** size
                # See comments above.
                val = val.astype(node.outputs[0].dtype)
                return [
                    alloc(
                        val,
                        *[
                            shapes[i]
                            for i in range(len(shapes))
                            if i not in node.op.axis
                        ],
                    )
                ]
            except NotScalarConstantError:
                pass


@register_specialize
@local_optimizer([neg])
def local_neg_neg(fgraph, node):
    # other specializations shouldn't put this in,
    # but sometimes they do
    if node.op == neg:
        if node.inputs[0].owner and node.inputs[0].owner.op == neg:
            return [node.inputs[0].owner.inputs[0]]


@register_specialize
@local_optimizer([neg])
def local_neg_div_neg(fgraph, node):
    """
    - (-a / b) -> a / b

    Also performs - (c / b) -> ((-c) / b) when c is a scalar constant.

    """
    if node.op == neg:
        if node.inputs[0].owner and node.inputs[0].owner.op == true_div:
            frac = node.inputs[0]
            num, denom = frac.owner.inputs
            if num.owner and num.owner.op == neg:
                if len(fgraph.clients[frac]) == 1:
                    # No other clients of the original division
                    new_num = num.owner.inputs[0]
                    return [true_div(new_num, denom)]
            elif np.all(num.broadcastable) and isinstance(num, Constant):
                if len(fgraph.clients[frac]) == 1:
                    new_num = -num.data
                    return [true_div(new_num, denom)]


@local_optimizer([mul])
def local_mul_zero(fgraph, node):
    """
    As part of canonicalization, we replace multiplication by zero
    with zero.

    """
    if node.op == mul:
        otype = node.outputs[0].type

        for i in node.inputs:
            try:
                value = get_scalar_constant_value(i)
            except NotScalarConstantError:
                continue
            # print 'MUL by value', value, node.inputs
            if value == 0:
                # print '... returning zeros'
                return _fill_chain(_asarray(0, dtype=otype.dtype), node.inputs)


register_canonicalize(local_mul_zero)


@local_optimizer([true_div])
def local_div_to_inv(fgraph, node):
    if node.op == true_div and np.all(
        local_mul_canonizer.get_constant(node.inputs[0]) == 1.0
    ):
        out = node.outputs[0]
        new_out = inv(local_mul_canonizer.merge_num_denum(node.inputs[1:], []))
        # The ones could have forced upcasting
        if new_out.dtype != out.dtype:
            new_out = cast(new_out, dtype=out.dtype)
        # The ones could have forced a specific length
        if new_out.type != out.type:
            new_out = broadcast_like(new_out, out, fgraph)
        return [new_out]
    else:
        return False


register_specialize(local_div_to_inv)


@local_optimizer([inv])
def local_inv_canon(fgraph, node):
    if node.op == inv:
        return [aet_pow(node.inputs[0], -1.0)]
    else:
        return False


register_canonicalize(local_inv_canon)


@local_optimizer([aet_pow])
def local_pow_canonicalize(fgraph, node):
    if node.op == aet_pow:
        cst = local_mul_canonizer.get_constant(node.inputs[1])
        if cst == 0:
            return [broadcast_like(1, node.outputs[0], fgraph)]
        if cst == 1:
            return [broadcast_like(node.inputs[0], node.outputs[0], fgraph)]
    else:
        return False


register_canonicalize(local_pow_canonicalize)


@register_specialize
@local_optimizer([mul])
def local_mul_to_sqr(fgraph, node):
    """
    x*x -> sqr(x)

    This is faster on the GPU when memory fetching is a big part of
    the computation time.

    """
    if node.op == mul:
        if len(node.inputs) == 2:
            if node.inputs[0] is node.inputs[1]:
                return [sqr(node.inputs[0])]


@register_canonicalize
@local_optimizer([int_div])
def local_intdiv_by_one(fgraph, node):
    """x // 1 -> x"""
    if node.op in [int_div]:
        if isinstance(node.inputs[1], TensorConstant) and np.all(
            node.inputs[1].value == 1
        ):
            return [node.inputs[0].astype(node.outputs[0].dtype)]


@register_canonicalize
@register_specialize
@local_optimizer([int_div, true_div])
def local_zero_div(fgraph, node):
    """0 / x -> 0"""
    if isinstance(node.op, Elemwise) and isinstance(
        node.op.scalar_op, (aes.IntDiv, aes.TrueDiv)
    ):
        if local_mul_canonizer.get_constant(node.inputs[0]) == 0:
            ret = broadcast_like(0, node.outputs[0], fgraph)
            ret.tag.values_eq_approx = values_eq_approx_remove_nan
            return [ret]


@local_optimizer([aet_pow])
def local_pow_specialize(fgraph, node):
    # here, we are past the point of canonicalization, so we don't want
    # to put in un-necessary fills.
    if node.op == aet_pow:
        # the idea here is that we have pow(x, y)
        odtype = node.outputs[0].dtype
        xsym = node.inputs[0]
        ysym = node.inputs[1]
        y = local_mul_canonizer.get_constant(ysym)
        if (y is not None) and encompasses_broadcastable(
            xsym.type.broadcastable, ysym.type.broadcastable
        ):
            rval = None

            if np.all(y == 2):
                rval = [sqr(xsym)]
            if np.all(y == 1):
                rval = [xsym]
            if np.all(y == 0):
                rval = [fill(xsym, np.asarray(1, dtype=odtype))]
            if np.all(y == 0.5):
                rval = [sqrt(xsym)]
            if np.all(y == -0.5):
                rval = [inv(sqrt(xsym))]
            if np.all(y == -1):
                rval = [inv(xsym)]
            if np.all(y == -2):
                rval = [inv(sqr(xsym))]
            if rval:
                rval[0] = cast(rval[0], odtype)
                assert rval[0].type == node.outputs[0].type, (rval, node.outputs)
                return rval
    else:
        return False


register_specialize(local_pow_specialize)


@register_specialize_device
@local_optimizer([aet_pow])
def local_pow_specialize_device(fgraph, node):
    """
    This optimization is not the same on all device. We do it only on cpu here.
    """
    if node.op == aet_pow:
        # the idea here is that we have pow(x, y)
        odtype = node.outputs[0].dtype
        xsym = node.inputs[0]
        ysym = node.inputs[1]
        y = local_mul_canonizer.get_constant(ysym)

        # the next line is needed to fix a strange case that I don't
        # know how to make a separate test.
        # That happen in the test_opt.py:test_log_erfc test.
        # y is a ndarray with dtype int8 and value 2,4 or 6. This make
        # the abs(y) <= 512 fail!
        # taking the value outside ndarray solve the problem.
        # it could be that in that case, numpy make the comparaison
        # into the wrong type(do in int8 that overflow.)
        if isinstance(y, np.ndarray):
            assert y.size == 1
            try:
                y = y[0]
            except IndexError:
                pass
        if (y is not None) and encompasses_broadcastable(
            xsym.type.broadcastable, ysym.type.broadcastable
        ):
            rval = None
            # 512 is too small for the cpu and too big for some gpu!
            if abs(y) == int(abs(y)) and abs(y) <= 512:
                pow2 = [xsym]
                pow2_scal = [aes.get_scalar_type(xsym.dtype)()]
                y_to_do = abs(y)
                for i in range(int(np.log2(y_to_do))):
                    pow2.append(sqr(pow2[i]))
                    pow2_scal.append(aes.sqr(pow2_scal[i]))
                rval1 = None
                rval1_scal = None
                while y_to_do > 0:
                    log_to_do = int(np.log2(y_to_do))
                    if rval1:
                        rval1 *= pow2[log_to_do]
                        rval1_scal *= pow2_scal[log_to_do]
                    else:
                        rval1 = pow2[log_to_do]
                        rval1_scal = pow2_scal[log_to_do]
                    y_to_do -= 2 ** log_to_do

                if abs(y) > 2:
                    # We fuse all the pow together here to make
                    # compilation faster
                    rval1 = Elemwise(
                        aes.Composite([pow2_scal[0]], [rval1_scal])
                    ).make_node(xsym)
                if y < 0:
                    rval = [inv(rval1)]
                else:
                    rval = [rval1]
            if rval:
                rval[0] = cast(rval[0], odtype)
                assert rval[0].type == node.outputs[0].type, (rval, node.outputs)
                return rval


@local_optimizer([mul])
def local_mul_specialize(fgraph, node):
    """
    Remove special-case constants from mul arguments and useless neg in inputs.

    mul(-1, x) -> neg(x)
    mul(1, x, y) -> mul(x, y)
    mul(0, ...) -> alloc(0, shapes...)

    This is not done if we would add more nodes in the graph, like with:

    mul(-1, x, y) -/-> neg(mul(x, y))

    """
    # here, we are past the point of canonicalization, so we don't
    # want to put in un-necessary fills.
    #
    # at this point [post canonicalize], mul() may have many inputs.
    if node.op == mul:
        # the idea here is that we have pow(x, y)
        has_neg = False
        new_inputs = []
        nb_neg_node = 0
        nb_cst = 0
        for input in node.inputs:
            # remove any neg arguments
            while input.owner and input.owner.op == neg:
                has_neg ^= True
                input = input.owner.inputs[0]
                nb_neg_node += 1

            # remove special case arguments of 1, -1 or 0
            y = local_mul_canonizer.get_constant(input)
            if y == 1.0:
                nb_cst += 1
            elif y == -1.0:
                nb_cst += 1
                has_neg ^= True  # toggles
            elif y == 0.0:
                # if we find any zero, we just return right away
                return [broadcast_like(0, node.outputs[0], fgraph)]
            else:
                new_inputs.append(input)

        if new_inputs != node.inputs:
            if new_inputs:
                if len(new_inputs) == 1:
                    if has_neg:
                        if new_inputs[0].dtype in (uint_dtypes + ["bool"]):
                            return
                        else:
                            rval = -new_inputs[0]
                    else:
                        rval = new_inputs[0]
                else:
                    # The next case would cause a replace by an equivalent case.
                    if has_neg and nb_neg_node == 0 and nb_cst == 1:
                        return
                    elif has_neg:
                        # Don't add an extra neg node as we can't
                        # fully replace this mul by a neg.
                        m1 = np.asarray(-1, dtype=node.outputs[0].dtype)
                        new_inputs = [m1] + new_inputs
                    rval = mul(*new_inputs)

                return [broadcast_like(rval, node.outputs[0], fgraph)]
            else:
                # there are no variable inputs to mul
                # N.B. this could have been constant-folded...
                if has_neg:
                    return [broadcast_like(-1, node.outputs[0], fgraph)]
                else:
                    return [broadcast_like(1, node.outputs[0], fgraph)]


register_specialize(local_mul_specialize)


@local_optimizer([add])
def local_add_specialize(fgraph, node):
    def fill_chain(v):
        out = _fill_chain(v, node.inputs)
        return out

    # here, we are past the point of canonicalization, so we don't want
    # to put in un-necessary fills.
    if node.op == add:
        new_inputs = []
        for input in node.inputs:
            try:
                y = get_scalar_constant_value(input)
            except NotScalarConstantError:
                y = input
            if np.all(y == 0.0):
                continue
            new_inputs.append(input)

        if len(new_inputs) < len(node.inputs):
            dtype = node.outputs[0].type.dtype
            if len(new_inputs) == 0:
                # we got rid of the entire expression!
                ndim = node.outputs[0].type.ndim
                # Reuse call to constant for cache()
                cst = constant(np.zeros((1,) * ndim, dtype=dtype))
                assert cst.type.broadcastable == (True,) * ndim
                return fill_chain(cst)

            if len(new_inputs) == 1:
                ret = fill_chain(new_inputs[0])
            else:
                ret = fill_chain(add(*new_inputs))
            # The dtype should not be changed. It can happen if the input
            # that was forcing upcasting was equal to 0.
            if ret[0].dtype != dtype:
                ret = [cast(ret[0], dtype)]
            return ret
    else:
        return False


register_specialize(local_add_specialize)

mul_canonizer = in2out(
    LocalOptGroup(local_mul_canonizer, local_fill_sink, apply_all_opts=True),
    name="mul_canonizer_groups",
)


def check_for_x_over_absX(numerators, denominators):
    """Convert x/abs(x) into sign(x). """
    # TODO: this function should dig/search through dimshuffles
    # This won't catch a dimshuffled absolute value
    for den in list(denominators):
        if den.owner and den.owner.op == abs_ and den.owner.inputs[0] in numerators:
            if den.owner.inputs[0].type.dtype.startswith("complex"):
                # TODO: Make an Op that projects a complex number to
                #      have unit length but projects 0 to 0.  That
                #      would be a weird Op, but consistent with the
                #      special case below.  I heard there's some
                #      convention in Matlab that is similar to
                #      this... but not sure.
                pass
            else:
                denominators.remove(den)
                numerators.remove(den.owner.inputs[0])
                numerators.append(sgn(den.owner.inputs[0]))
    return numerators, denominators


local_mul_canonizer.add_simplifier(check_for_x_over_absX, "X_over_absX")


@register_canonicalize
@local_optimizer([abs_])
def local_abs_lift(fgraph, node):
    """
    Move the abs toward the input.

    This is needed for check_for_x_over_absX to apply in more case.

    """
    if node.op == abs_ and node.inputs[0].owner:
        assert node.nin == 1
        if node.inputs[0].owner.op == mul:
            return [mul(*[abs_(i) for i in node.inputs[0].owner.inputs])]
        if node.inputs[0].owner.op == true_div:
            i = node.inputs[0].owner.inputs
            return [true_div(abs_(i[0]), abs_(i[1]))]


@register_specialize
@local_optimizer([mul, true_div])
def local_abs_merge(fgraph, node):
    """
    Merge abs generated by local_abs_lift when the canonizer don't
    need it anymore

    """
    if node.op == mul and sum([i.owner.op == abs_ for i in node.inputs if i.owner]) > 1:
        inputs = []
        for i in node.inputs:
            if i.owner and i.owner.op == abs_:
                inputs.append(i.owner.inputs[0])
            elif isinstance(i, Constant):
                try:
                    const = get_scalar_constant_value(i, only_process_constants=True)
                except NotScalarConstantError:
                    return False
                if not (const >= 0).all():
                    return False
                inputs.append(i)
            else:
                return False
        return [abs_(mul(*inputs))]
    if (
        node.op == true_div
        and sum([i.owner.op == abs_ for i in node.inputs if i.owner]) == 2
    ):
        return [
            abs_(
                true_div(node.inputs[0].owner.inputs[0], node.inputs[1].owner.inputs[0])
            )
        ]


@register_stabilize
@register_specialize
@local_optimizer([log])
def local_log1p(fgraph, node):
    # log(1+x) -> log1p(x)
    # log(1-x) -> log1p(-x)
    if node.op == log:
        (log_arg,) = node.inputs
        if log_arg.owner and log_arg.owner.op == add:
            scalars, scalar_inputs, nonconsts = scalarconsts_rest(
                log_arg.owner.inputs, only_process_constants=True
            )
            # scalar_inputs are potentially dimshuffled and fill'd scalars
            if scalars and np.allclose(np.sum(scalars), 1):
                if nonconsts:
                    if len(nonconsts) > 1:
                        ninp = add(*nonconsts)
                    else:
                        ninp = nonconsts[0]
                    if ninp.dtype != log_arg.type.dtype:
                        ninp = ninp.astype(node.outputs[0].dtype)
                    return _fill_chain(log1p(ninp), scalar_inputs)

        elif log_arg.owner and log_arg.owner.op == sub:
            one = extract_constant(log_arg.owner.inputs[0], only_process_constants=True)
            if one != 1:
                return
            other = log_arg.owner.inputs[1]
            if other.dtype != log_arg.dtype:
                other = other.astype(log_arg.dtype)
            return [log1p(neg(other))]


# TODO: in canonicalize, change log10 and log2 -> log
@register_stabilize
@register_specialize
@local_optimizer([log])
def local_log_add(fgraph, node):
    # log(exp(x)+exp(y))
    #
    # Suppose x >= y
    # log(exp(x) + exp(y))
    # log(exp(x) * (1 + exp(y)/exp(x)))
    # x + log(1 + exp(y)/exp(x))
    # x + log1p(exp(y)/exp(x))
    # x + log1p(exp(y-x))
    if node.op == log:
        z = node.inputs[0]
        if z.owner and z.owner.op == add:
            zi = z.owner.inputs
            if len(zi) != 2:
                # -- upgrading Maximum to handle multiple inputs wasn't trivial
                #    TODO
                # raise NotImplementedError()
                return
            pre_exp = [x.owner.inputs[0] for x in zi if x.owner and x.owner.op == exp]
            if len(pre_exp) == len(zi):
                # all arguments to add are exp(<something>)
                max_pre = maximum(*pre_exp)

                ret = max_pre + log1p(exp(add(*[p - max_pre for p in pre_exp])))
                ret.tag.values_eq_approx = values_eq_approx_remove_inf
                return [ret]


@local_optimizer([log])
def local_log_sum_exp(fgraph, node):
    # log(sum_i(exp(x_i))) = x_max + log(sum_i(exp(x_i - x_max)))

    if node.op != log:
        return

    sum_node = node.inputs[0].owner
    # If the sum has keepdims=True, there might be a dimshuffle
    if sum_node and isinstance(sum_node.op, DimShuffle):
        dimshuffle_op = sum_node.op
        sum_node = sum_node.inputs[0].owner
    else:
        dimshuffle_op = None

    if not sum_node or not isinstance(sum_node.op, Sum):
        return

    exp_node, axis = sum_node.inputs[0].owner, sum_node.op.axis
    if not exp_node or not (
        isinstance(exp_node.op, Elemwise) and isinstance(exp_node.op.scalar_op, aes.Exp)
    ):
        return

    pre_exp = exp_node.inputs[0]
    max_pre_exp = aet_max(pre_exp, axis=axis)
    max_pre_exp_keepdims = makeKeepDims(pre_exp, max_pre_exp, axis)

    ret = max_pre_exp + log(aet_sum(exp(pre_exp - max_pre_exp_keepdims), axis=axis))

    # Restore the dimshuffle op, if any.
    if dimshuffle_op:
        ret = dimshuffle_op(ret)

    return [ret]


compile.optdb.register(
    "local_log_sum_exp",
    in2out(local_log_sum_exp, ignore_newtrees=True),
    1.6,
    "fast_run",
)


def add_calculate(num, denum, aslist=False, out_type=None):
    # TODO: make sure that this function and mul_calculate are similar
    if out_type is None:
        zero = 0.0
    else:
        zero = _asarray(0, dtype=out_type.dtype)
    # zero = 0.0 if out_type is None else _asarray(0,
    # dtype=out_type.dtype)
    if out_type and out_type.dtype == "bool":
        if len(denum) == 0:
            # NumPy 1.14 do not accept to do "bool - bool"
            v = reduce(np.add, num, zero)
        else:
            raise Exception(
                "bool subtraction not supported. This should not happen as"
                " an earlier error should have been raised"
            )
    else:
        v = reduce(np.add, num, zero) - reduce(np.add, denum, zero)
    if aslist:
        if np.all(v == 0):
            return []
        else:
            return [v]
    return v


local_add_canonizer = AlgebraicCanonizer(add, sub, neg, add_calculate)
add_canonizer = in2out(
    LocalOptGroup(local_add_canonizer, local_fill_sink, apply_all_opts=True),
    name="add_canonizer_group",
)


register_canonicalize(local_add_canonizer, name="local_add_canonizer")


def distribute_greedy(pos_pairs, neg_pairs, num, denum, out_type, minscore=0):
    # each pair in pos_pairs and neg_pairs is a num/denum pair. this
    # function attempts to add num and denum to the corresponding parts
    # of each pair, and counts how many multiplications/divisions can
    # be saved in that way.

    # each division is counted like div_cost multiplications
    # (typically, division costs more so we are willing to multiply more
    # in order to divide less)
    # 1.5 was obtained through an informal test and may very well be
    # platform dependent
    div_cost = 1.5

    # score is number of operations saved, higher is better
    score = len(num) + div_cost * len(denum)
    new_pos_pairs = list(
        itertools.starmap(
            local_mul_canonizer.simplify,
            [(n + num, d + denum, out_type) for (n, d) in pos_pairs],
        )
    )
    new_neg_pairs = list(
        itertools.starmap(
            local_mul_canonizer.simplify,
            [(n + num, d + denum, out_type) for (n, d) in neg_pairs],
        )
    )
    for (n, d), (nn, dd) in zip(pos_pairs + neg_pairs, new_pos_pairs + new_neg_pairs):
        # We calculate how many operations we are saving with the new
        # num and denum
        score += len(n) + div_cost * len(d) - len(nn) - div_cost * len(dd)
    if score <= minscore:
        # the change is not applied because it adds too many operations
        return False, pos_pairs, neg_pairs
    return True, new_pos_pairs, new_neg_pairs


def attempt_distribution(factor, num, denum, out_type):
    """Try to insert each `num` and each `denum` in the factor?

    Returns
    -------
    changes?, new_factor, new_num, new_denum
        If there are changes, `new_num` and `new_denum` contain all the
        numerators and denominators that could not be distributed in the factor

    """
    pos_terms, neg_terms = local_add_canonizer.get_num_denum(factor)
    if len(pos_terms) == 1 and not neg_terms:
        return False, factor, num, denum
    pos_pairs = list(map(local_mul_canonizer.get_num_denum, pos_terms))
    neg_pairs = list(map(local_mul_canonizer.get_num_denum, neg_terms))
    change = False
    for n in list(num):
        success, pos_pairs, neg_pairs = distribute_greedy(
            pos_pairs, neg_pairs, [n], [], out_type
        )
        if success:
            change = True
            num.remove(n)
    for d in list(denum):
        success, pos_pairs, neg_pairs = distribute_greedy(
            pos_pairs, neg_pairs, [], [d], out_type
        )
        if success:
            change = True
            denum.remove(d)
    if not change:
        return change, factor, num, denum
    else:
        return (
            change,
            local_add_canonizer.merge_num_denum(
                list(itertools.starmap(local_mul_canonizer.merge_num_denum, pos_pairs)),
                list(itertools.starmap(local_mul_canonizer.merge_num_denum, neg_pairs)),
            ),
            num,
            denum,
        )


@register_canonicalize
@register_stabilize
@local_optimizer([mul, true_div, inv])
def local_greedy_distributor(fgraph, node):
    """
    Optimize by reducing the number of multiplications and/or divisions.

    This optimization tries to apply distributivity of multiplication
    to addition in order to reduce the number of multiplications
    and/or divisions that must be done. The algorithm weighs division
    more than multiplication to account for the former's slightly
    greater computational cost.

    The following expressions are simplified:
    1. ((a/x + b/y) * x * y) --> a*y + b*x
    2. ((a/x + b) * x) --> a + b*x
    3. There are other forms too where node is a true_div.

    The following expressions are not simplified:
    4. ((a + b) * x) -/-> a*x + b*x

    This optimization aims to reduce computational cost. It may also
    increase numerical stability, e.g. when x and/or y tend to 0 in
    example 1.

    """

    out = node.outputs[0]
    num, denum = local_mul_canonizer.get_num_denum(out)
    if len(num) == 1 and not denum:
        return False

    new_num, new_denum = [], []

    change = False

    out_type = out.type
    for candidate in list(num):
        if candidate not in num:
            continue
        num.remove(candidate)
        _change, candidate, num, denum = attempt_distribution(
            candidate,
            num,
            denum,
            out_type,
        )

        change |= _change
        new_num.append(candidate)

    for candidate in list(denum):
        if candidate not in denum:
            continue
        denum.remove(candidate)
        _change, candidate, denum, num = attempt_distribution(
            candidate, denum, num, out_type
        )
        change |= _change
        new_denum.append(candidate)
    if not change:
        return False

    new_num += num
    new_denum += denum

    rval = local_mul_canonizer.merge_num_denum(new_num, new_denum)

    if not (rval.type == out.type):
        # WHY DOES THIS HAPPEN?
        return False

    return [rval]


def get_clients(fgraph, node):
    """
    Used by erf/erfc opt to track less frequent op.

    """
    return [c for c, i in fgraph.clients[node.outputs[0]] if c != "output"]


def get_clients2(fgraph, node):
    """
    Used by erf/erfc opt to track less frequent op.

    """
    l = []
    for c, i in fgraph.clients[node.outputs[0]]:
        if c != "output":
            for var in c.outputs:
                l.extend([cc for cc, ii in fgraph.clients[var] if cc != "output"])
    return l


# 1+erf(x)=>erfc(-x)
local_one_plus_erf = PatternSub(
    (add, 1, (erf, "x")),
    (erfc, (neg, "x")),
    allow_multiple_clients=True,
    name="local_one_plus_erf",
    tracks=[erf],
    get_nodes=get_clients,
)
register_canonicalize(local_one_plus_erf)
register_stabilize(local_one_plus_erf)
register_specialize(local_one_plus_erf)

# 1-erf(x)=>erfc(x)
local_one_minus_erf = PatternSub(
    (sub, 1, (erf, "x")),
    (erfc, "x"),
    allow_multiple_clients=True,
    name="local_one_minus_erf",
)
register_canonicalize(local_one_minus_erf)
register_stabilize(local_one_minus_erf)
register_specialize(local_one_minus_erf)

local_one_minus_erf2 = PatternSub(
    (add, 1, (mul, -1, (erf, "x"))),
    (erfc, "x"),
    allow_multiple_clients=True,
    name="local_one_minus_erf2",
)
register_canonicalize(local_one_minus_erf2)
register_stabilize(local_one_minus_erf2)
register_specialize(local_one_minus_erf2)

# 1+(-erf(x))=>erfc(x) This is a different graph then the previous as
# the canonicalize don't work completly
local_one_plus_neg_erf = PatternSub(
    (add, 1, (neg, (erf, "x"))),
    (erfc, "x"),
    allow_multiple_clients=True,
    name="local_one_plus_neg_erf",
    tracks=[erf],
    get_nodes=get_clients2,
)
register_canonicalize(local_one_plus_neg_erf)
register_stabilize(local_one_plus_neg_erf)
register_specialize(local_one_plus_neg_erf)

# (-1)+erf(x) => -erfc(x) don't need erf(x)+(-1) as the canonicalize
# will put the -1 as the first argument.
local_erf_minus_one = PatternSub(
    (add, -1, (erf, "x")),
    (neg, (erfc, "x")),
    allow_multiple_clients=True,
    name="local_erf_minus_one",
    tracks=[erf],
    get_nodes=get_clients,
)
register_canonicalize(local_erf_minus_one)
register_stabilize(local_erf_minus_one)
register_specialize(local_erf_minus_one)

# 1-erfc(x) => erf(x)
local_one_minus_erfc = PatternSub(
    (sub, 1, (erfc, "x")),
    (erf, "x"),
    allow_multiple_clients=True,
    name="local_one_minus_erfc",
    tracks=[erfc],
    get_nodes=get_clients,
)
register_canonicalize(local_one_minus_erfc)
register_stabilize(local_one_minus_erfc)
register_specialize(local_one_minus_erfc)

local_one_minus_erfc2 = PatternSub(
    (add, 1, (neg, (erfc, "x"))),
    (erf, "x"),
    allow_multiple_clients=True,
    name="local_one_minus_erfc2",
    tracks=[erfc],
    get_nodes=get_clients2,
)
register_canonicalize(local_one_minus_erfc2)
register_stabilize(local_one_minus_erfc2)
register_specialize(local_one_minus_erfc2)

local_one_minus_erfc3 = PatternSub(
    (add, 1, (mul, -1, (erfc, "x"))),
    (erf, "x"),
    allow_multiple_clients=True,
    name="local_one_minus_erfc3",
    tracks=[erfc],
    get_nodes=get_clients2,
)
register_canonicalize(local_one_minus_erfc3)
register_stabilize(local_one_minus_erfc3)
register_specialize(local_one_minus_erfc3)

# 1+(-erfc(x)) => erf(x) This is a different graph then the previous as
# the canonicalize don't work completly
local_one_add_neg_erfc = PatternSub(
    (add, 1, (neg, (erfc, "x"))),
    (erf, "x"),
    allow_multiple_clients=True,
    name="local_one_add_neg_erfc",
    tracks=[erfc],
    get_nodes=get_clients2,
)

register_canonicalize(local_one_add_neg_erfc)
register_stabilize(local_one_add_neg_erfc)
register_specialize(local_one_add_neg_erfc)

# (-1)+erfc(-x)=>erf(x)
local_erf_neg_minus_one = PatternSub(
    (add, -1, (erfc, (neg, "x"))),
    (erf, "x"),
    allow_multiple_clients=True,
    name="local_erf_neg_minus_one",
    tracks=[erfc],
    get_nodes=get_clients,
)
register_canonicalize(local_erf_neg_minus_one)
register_stabilize(local_erf_neg_minus_one)
register_specialize(local_erf_neg_minus_one)

# (-1)+erfc(-1*x)=>erf(x)
local_erf_neg_minus_one2 = PatternSub(
    (add, -1, (erfc, (mul, -1, "x"))),
    (erf, "x"),
    allow_multiple_clients=True,
    name="local_erf_neg_minus_one2",
    tracks=[erfc],
    get_nodes=get_clients,
)
register_canonicalize(local_erf_neg_minus_one2)
register_stabilize(local_erf_neg_minus_one2)
register_specialize(local_erf_neg_minus_one2)


@register_stabilize
@register_specialize
@local_optimizer([log])
def local_log_erfc(fgraph, node):
    """Stability optimization for `log(erfc(x))`.

    log(erfc(x)) => when x>threshold,
                -x**2-log(x)-.5*log(pi)+log(1-1/(2*x**2)+3/(4*x**4)-15/(8*x**6))
    for float64: threshold=26.641747557 was choosed with:
    [(i,numpy.log(scipy.special.erfc(numpy.asarray([i],dtype='float64'))))
    for i in numpy.arange(26.641747557,26.6417475571,.00000000001)]
    for float32: threshold=10.0541949, [(i,numpy.log(scipy.special.erfc(
        numpy.asarray([i],dtype='float32')))) for i in numpy.arange(
        10.0541948,10.0541951,.0000001)]
    """
    if node.op != log:
        return False
    if not node.inputs[0].owner or node.inputs[0].owner.op != erfc:
        return False

    if hasattr(node.tag, "local_log_erfc_applied"):
        # We use that flag to don't apply the optimization recursively
        return False
    node.tag.local_log_erfc_applied = True

    x = node.inputs[0].owner.inputs[0]
    stab_value = (
        -(x ** 2)
        - log(x)
        - 0.5 * log(np.pi)
        + log(1 - 1 / (2 * x ** 2) + 3 / (4 * x ** 4) - 15 / (8 * x ** 6))
    )

    if node.outputs[0].dtype == "float32" or node.outputs[0].dtype == "float16":
        threshold = 10.0541949
    elif node.outputs[0].dtype == "float64":
        threshold = 26.641747557

    ret = switch(x < threshold, node.outputs[0], stab_value)
    ret.tag.values_eq_approx = values_eq_approx_remove_inf
    return [ret]


@register_stabilize
@register_specialize
@local_optimizer([true_div])
def local_grad_log_erfc_neg(fgraph, node):
    """Stability optimization for the grad of `log(erfc(x))`.

    ([y*]exp(-(x**2)))/erfc(x) # The y* is optional
    ([y*]exp(x**2))/erfc(-x) => [y*](when x > threshold,
                            sqrt(pi)*-x/(1-1/(2*x**2)+3/(4*x**4)-15/(8*x**6)))

    for float64: threshold=26.63 see at the end of the fct for the explanation
    for float32: threshold=9.3 see at the end of the fct for the explanation

    TODO: remove the contraint that there are only 2 inputs to exp(x**2)
        is the second.
    TODO: at the test point 10 in float32, there is instability in the original
        value. The original gives -30.0, the stab -20.1 and in float64 -18.1.
        Make it so that the test does not generate an error in that case!

    """
    if node.op != true_div:
        return False
    if not node.inputs[1].owner or node.inputs[1].owner.op != erfc:
        return False

    erfc_in = node.inputs[1]
    erfc_x = erfc_in.owner.inputs[0]

    if not node.inputs[0].owner:
        return False

    # TODO: All of this should be replaced with a single, simple unification
    # The mul is optional.
    if node.inputs[0].owner.op != mul:
        mul_in = None
        y = []
        if not node.inputs[0].owner or node.inputs[0].owner.op != exp:
            return False
        exp_in = node.inputs[0]
    else:
        mul_in = node.inputs[0]
        exp_in = None
        for idx, inp in enumerate(mul_in.owner.inputs):
            if inp.owner and inp.owner.op == exp:
                exp_in = inp
                break
        else:
            return False

        if len(mul_in.owner.inputs) == 2:
            y = [mul_in.owner.inputs[1 - idx]]
        else:
            y = mul_in.owner.inputs[:]
            del y[idx]

    if not exp_in.owner.inputs[0].owner:
        return False

    if exp_in.owner.inputs[0].owner.op == neg:
        neg_in = exp_in.owner.inputs[0]
        if not neg_in.owner.inputs[0].owner or neg_in.owner.inputs[0].owner.op != sqr:
            return False
        sqr_in = neg_in.owner.inputs[0]
        x = sqr_in.owner.inputs[0]
    elif exp_in.owner.inputs[0].owner.op == mul:
        # We should compare that -(erfc_x**2) is equivalent to mul_neg.
        # There is currently no easy way to do this in the general case,
        # so we implement some common case for now.

        # In many cases the neg are replaced by mul in the graph.
        # This also allows to stabilize log(erfc(cst*x)).
        mul_neg = exp_in.owner.inputs[0]

        # In case that multiple mul are not fused together, we do it here.
        def check_input(inputs):
            new_inputs = []
            for i in inputs:
                if i.owner and i.owner.op == mul:
                    new_inputs.extend(check_input(i.owner.inputs))
                else:
                    new_inputs.append(i)
            return new_inputs

        mul_inputs = check_input(mul_neg.owner.inputs)

        # Put the constant first.
        for i in range(len(mul_inputs)):
            if isinstance(i, Constant):
                if i == 0:
                    break
                else:
                    tmp = mul_inputs[0]
                    mul_inputs[0] = mul_inputs[i]
                    mul_inputs[i] = tmp
                    break
        mul_neg = mul(*mul_inputs)

        try:
            cst2 = get_scalar_constant_value(
                mul_neg.owner.inputs[0], only_process_constants=True
            )
        except NotScalarConstantError:
            return False

        if len(mul_neg.owner.inputs) == 2:
            if (
                not mul_neg.owner.inputs[1].owner
                or mul_neg.owner.inputs[1].owner.op != sqr
            ):
                return False
            sqr_in = mul_neg.owner.inputs[1]
            x = sqr_in.owner.inputs[0]
        elif len(mul_neg.owner.inputs) == 3:
            if mul_neg.owner.inputs[1] is not mul_neg.owner.inputs[2]:
                return False
            x = mul_neg.owner.inputs[1]
        else:
            return False

        if cst2 != -1:
            if (
                not erfc_x.owner
                or erfc_x.owner.op != mul
                or len(erfc_x.owner.inputs) != 2
            ):
                # todo implement that case
                return False
            if erfc_x.owner.inputs[1] is not mul_neg.owner.inputs[1]:
                return False

            x = erfc_x
            try:
                cst = get_scalar_constant_value(
                    erfc_x.owner.inputs[0], only_process_constants=True
                )
            except NotScalarConstantError:
                return False
            if cst2 != -cst * 2:
                return False

            # The constant is valid. Must check that the
        elif erfc_x is not x:
            return False

    else:
        return False

    if hasattr(node.tag, "local_grad_log_erfc_neg"):
        # We use that flag to don't apply the optimization recursively
        return False

    if erfc_x is not x:
        return None

    # we move the y outside the div.
    true_div_no_mul = true_div(exp_in, erfc_in)
    true_div_no_mul.owner.tag.local_grad_log_erfc_neg = True

    # aaron value
    stab_value = (
        x
        * aet_pow(1 - 1 / (2 * (x ** 2)) + 3 / (4 * (x ** 4)) - 15 / (8 * (x ** 6)), -1)
        * cast(sqrt(np.pi), dtype=x.dtype)
    )

    if x.dtype == "float32" or x.dtype == "float16":
        threshold = 9.3
        # threshold = 10.1
    elif x.dtype == "float64":
        threshold = 26.641747557

    ret = switch(x < threshold, true_div_no_mul, stab_value)

    if y:
        ret = mul(ret, *y)

    ret.tag.values_eq_approx = values_eq_approx_remove_inf_nan

    return [ret]


def local_add_mul_fusion(fgraph, node):
    """Fuse consecutive add or mul in one such node with more inputs.

    It is better to fuse add/mul that way then in a Composite node as
    this make the inner graph of the Composite smaller. This allow to
    put more computation in a Composite before hitting the max
    recusion limit when pickling Composite.

    """
    if not isinstance(node.op, Elemwise) or not isinstance(
        node.op.scalar_op, (aes.Add, aes.Mul)
    ):
        return False

    s_op = node.op.scalar_op.__class__
    new_inp = []
    fused = False
    nb_inputs = len(node.inputs)
    max_inputs = float("inf")
    if hasattr(node.op, "max_inputs"):
        max_inputs = node.op.max_inputs(node)
    for inp in node.inputs:
        if (
            inp.owner
            and isinstance(inp.owner.op, Elemwise)
            and isinstance(inp.owner.op.scalar_op, s_op)
            and
            # Do not duplicate the operation.
            len(fgraph.clients[inp]) == 1
            and (nb_inputs + len(inp.owner.inputs) - 1) <= max_inputs
        ):
            new_inp.extend(inp.owner.inputs)
            fused = True
        else:
            new_inp.append(inp)

    # We can not compare the number of inputs as Mul and Add could have
    # 0 or 1 inputs in some corner cases.
    if fused:
        output = node.op(*new_inp)
        copy_stack_trace(node.outputs[0], output)

        # Do the recursion here to help lower the number of
        # FusionOptimizer iteration.
        if output.owner:
            output2 = local_add_mul_fusion(fgraph, output.owner)
            if output2:
                return output2
        return [output]


fuse_seqopt.register(
    "local_add_mul_fusion",
    FusionOptimizer(local_add_mul_fusion),
    0,
    "fast_run",
    "fusion",
)


def _skip_mul_1(r):
    if r.owner and r.owner.op == mul:
        not_is_1 = [i for i in r.owner.inputs if not _is_1(i)]
        if len(not_is_1) == 1:
            return not_is_1[0]


def _is_1(expr):
    """

    Returns
    -------
    bool
        True iff expr is a constant close to 1.

    """
    try:
        v = get_scalar_constant_value(expr)
        return np.allclose(v, 1)
    except NotScalarConstantError:
        return False


logsigm_to_softplus = PatternSub(
    (log, (sigmoid, "x")),
    (neg, (softplus, (neg, "x"))),
    allow_multiple_clients=True,
    values_eq_approx=values_eq_approx_remove_inf,
    skip_identities_fn=_skip_mul_1,
)

log1msigm_to_softplus = PatternSub(
    (log, (sub, dict(pattern="y", constraint=_is_1), (sigmoid, "x"))),
    (neg, (softplus, "x")),
    allow_multiple_clients=True,
    values_eq_approx=values_eq_approx_remove_inf,
    skip_identities_fn=_skip_mul_1,
)
log1pexp_to_softplus = PatternSub(
    (log1p, (exp, "x")),
    (softplus, "x"),
    values_eq_approx=values_eq_approx_remove_inf,
    allow_multiple_clients=True,
)
log1p_neg_sigmoid = PatternSub(
    (log1p, (neg, (sigmoid, "x"))),
    (neg, (softplus, "x")),
    values_eq_approx=values_eq_approx_remove_inf,
    allow_multiple_clients=True,
)

register_stabilize(logsigm_to_softplus, name="logsigm_to_softplus")
register_stabilize(log1msigm_to_softplus, name="log1msigm_to_softplus")
register_stabilize(log1pexp_to_softplus, name="log1pexp_to_softplus")
register_stabilize(log1p_neg_sigmoid, name="log1p_neg_sigmoid,")


def is_1pexp(t, only_process_constants=True):
    """

    Returns
    -------
    object
        If 't' is of the form (1+exp(x)), return (False, x).
        Else return None.

    """
    if t.owner and t.owner.op == add:
        scalars, scalar_inputs, nonconsts = scalarconsts_rest(
            t.owner.inputs, only_process_constants=only_process_constants
        )
        # scalar_inputs are potentially dimshuffled and filled with scalars
        if len(nonconsts) == 1:
            maybe_exp = nonconsts[0]
            if maybe_exp.owner and maybe_exp.owner.op == exp:
                # Verify that the constant terms sum to 1.
                if scalars:
                    scal_sum = scalars[0]
                    for s in scalars[1:]:
                        scal_sum = scal_sum + s
                    if np.allclose(scal_sum, 1):
                        return False, maybe_exp.owner.inputs[0]
                # Before 7987b51 there used to be a bug where *any* constant
                # was considered as if it was equal to 1, and thus this
                # function would incorrectly identify it as (1 + exp(x)).
                if config.warn__identify_1pexp_bug:
                    warnings.warn(
                        "Although your current code is fine, please note that "
                        "Aesara versions prior to 0.5 (more specifically, "
                        "prior to commit 7987b51 on 2011-12-18) may have "
                        "yielded an incorrect result. To remove this warning, "
                        "either set the `warn__identify_1pexp_bug` config "
                        "option to False, or `warn__ignore_bug_before` to at "
                        "least '0.4.1'."
                    )
    return None


def is_exp(var):
    """
    Match a variable with either of the `exp(x)` or `-exp(x)` patterns.

    Parameters
    ----------
    var
        The Variable to analyze.

    Returns
    -------
    tuple
        A pair (b, x) with `b` a boolean set to True if `var` is of the
        form `-exp(x)` and False if `var` is of the form `exp(x)`. If `var`
        cannot be cast into either form, then return `None`.

    """
    _neg = False
    neg_info = is_neg(var)
    if neg_info is not None:
        _neg = True
        var = neg_info
    if var.owner and var.owner.op == exp:
        return _neg, var.owner.inputs[0]


def is_mul(var):
    """
    Match a variable with `x * y * z * ...`.

    Parameters
    ----------
    var
        The Variable to analyze.

    Returns
    -------
    object
        A list [x, y, z, ...] if `var` is of the form `x * y * z * ...`,
        or None if `var` cannot be cast into this form.

    """
    if var.owner and var.owner.op == mul:
        return var.owner.inputs
    else:
        return None


def partition_num_or_denom(r, f):
    if r.owner and r.owner.op == mul:
        a = r.owner.inputs
    else:
        a = [r]

    # ugly 2.4-compatible thing
    f_terms = []
    _neg = False
    rest = []
    for t in a:
        f_t = f(t)
        if f_t is None:
            rest.append(t)
        else:
            neg_t, f_t = f_t
            f_terms.append(f_t)
            _neg ^= neg_t  # bit flip if neg_t is true
    return f_terms, rest, _neg


def is_neg(var):
    """
    Match a variable with the `-x` pattern.

    Parameters
    ----------
    var
        The Variable to analyze.

    Returns
    -------
    object
        `x` if `var` is of the form `-x`, or None otherwise.

    """
    var_node = var.owner
    if not var_node:
        return None
    # First match against `neg`.
    if var_node.op == neg:
        return var_node.inputs[0]
    # Then match against a multiplication by -1.
    if var_node.op == mul and len(var_node.inputs) >= 2:
        for idx, mul_input in enumerate(var_node.inputs):
            try:
                constant = get_scalar_constant_value(mul_input)
                is_minus_1 = np.allclose(constant, -1)
            except NotScalarConstantError:
                is_minus_1 = False
            if is_minus_1:
                # Found a multiplication by -1.
                if len(var_node.inputs) == 2:
                    # Only return the other input.
                    return var_node.inputs[1 - idx]
                else:
                    # Return the multiplication of all other inputs.
                    return mul(*(var_node.inputs[0:idx] + var_node.inputs[idx + 1 :]))
    # No match.
    return None


@register_stabilize
@local_optimizer([true_div])
def local_exp_over_1_plus_exp(fgraph, node):
    """
    exp(x)/(1+exp(x)) -> sigm(x)
    c/(1+exp(x)) -> c*sigm(-x)

    """
    # this optimization should be done for numerical stability
    # so we don't care to check client counts
    if node.op == true_div:

        # find all the exp() terms in the numerator
        num, denom = node.inputs
        num_exp_x, num_rest, num_neg = partition_num_or_denom(num, is_exp)
        denom_1pexp, denom_rest, denom_neg = partition_num_or_denom(denom, is_1pexp)

        sigmoids = []
        for t in denom_1pexp:
            if t in num_exp_x:
                # case: exp(x) /(1+exp(x))
                sigmoids.append(sigmoid(t))
                del num_exp_x[num_exp_x.index(t)]
            else:
                # case: 1/(1+exp(x))
                sigmoids.append(sigmoid(-t))
            copy_stack_trace(node.outputs[0], sigmoids[-1])

        if not sigmoids:  # we didn't find any.  abort
            return
        # put the new numerator together
        new_num = sigmoids + [exp(t) for t in num_exp_x] + num_rest
        if len(new_num) == 1:
            new_num = new_num[0]
        else:
            new_num = mul(*new_num)

        if num_neg ^ denom_neg:
            new_num = -new_num

        copy_stack_trace(num, new_num)

        if len(denom_rest) == 0:
            return [new_num]
        elif len(denom_rest) == 1:
            out = new_num / denom_rest[0]
        else:
            out = new_num / mul(*denom_rest)

        copy_stack_trace(node.outputs[0], out)
        return [out]


def parse_mul_tree(root):
    """
    Parse a tree of multiplications starting at the given root.

    Parameters
    ----------
    root
        The variable at the root of the tree.

    Returns
    -------
    object
        A tree where each non-leaf node corresponds to a multiplication
        in the computation of `root`, represented by the list of its inputs.
        Each input is a pair [n, x] with `n` a boolean value indicating whether
        sub-tree `x` should be negated.

    Examples
    --------
        x * y               -> [False, [[False, x], [False, y]]]
        -(x * y)            -> [True, [[False, x], [False, y]]]
        -x * y              -> [False, [[True, x], [False, y]]]
        -x                  -> [True, x]
        (x * y) * -z        -> [False, [[False, [[False, x], [False, y]]],
                                        [True, z]]]

    """
    # Is it a multiplication?
    mul_info = is_mul(root)
    if mul_info is None:
        # Is it a negation?
        neg_info = is_neg(root)
        if neg_info is None:
            # Keep the root "as is".
            return [False, root]
        else:
            # Recurse, inverting the negation.
            neg, sub_tree = parse_mul_tree(neg_info)
            return [not neg, sub_tree]
    else:
        # Recurse into inputs.
        return [False, list(map(parse_mul_tree, mul_info))]


def replace_leaf(arg, leaves, new_leaves, op, neg):
    """
    Attempt to replace a leaf of a multiplication tree.

    We search for a leaf in `leaves` whose argument is `arg`, and if we find
    one, we remove it from `leaves` and add to `new_leaves` a leaf with
    argument `arg` and variable `op(arg)`.

    Parameters
    ----------
    arg
        The argument of the leaf we are looking for.
    leaves
        List of leaves to look into. Each leaf should be a pair
        (x, l) with `x` the argument of the Op found in the leaf, and `l` the
        actual leaf as found in a multiplication tree output by `parse_mul_tree`
        (i.e. a pair [boolean, variable]).
    new_leaves
        If a replacement occurred, then the leaf is removed from `leaves`
        and added to the list `new_leaves` (after being modified by `op`).
    op
        A function that, when applied to `arg`, returns the Variable
        we want to replace the original leaf variable with.
    neg : bool
        If True, then the boolean value associated to the leaf should
        be swapped. If False, then this value should remain unchanged.

    Returns
    -------
    bool
        True if a replacement occurred, or False otherwise.

    """
    for idx, x in enumerate(leaves):
        if x[0] == arg:
            x[1][0] ^= neg
            x[1][1] = op(arg)
            leaves.pop(idx)
            new_leaves.append(x)
            return True
    return False


def simplify_mul(tree):
    """
    Simplify a multiplication tree.

    Parameters
    ----------
    tree
        A multiplication tree (as output by `parse_mul_tree`).

    Returns
    -------
    object
        A multiplication tree computing the same output as `tree` but without
        useless multiplications by 1 nor -1 (identified by leaves of the form
        [False, None] or [True, None] respectively). Useless multiplications
        (with less than two inputs) are also removed from the tree.

    """
    neg, inputs = tree
    if isinstance(inputs, list):
        # Recurse through inputs.
        s_inputs = []
        for s_i in map(simplify_mul, inputs):
            if s_i[1] is None:
                # Multiplication by +/-1.
                neg ^= s_i[0]
            else:
                s_inputs.append(s_i)
        if not s_inputs:
            # The multiplication is empty.
            rval = [neg, None]
        elif len(s_inputs) == 1:
            # The multiplication has a single input.
            s_inputs[0][0] ^= neg
            rval = s_inputs[0]
        else:
            rval = [neg, s_inputs]
    else:
        rval = tree
    # print 'simplify_mul: %s -> %s' % (tree, rval)
    return rval


def compute_mul(tree):
    """
    Compute the Variable that is the output of a multiplication tree.

    This is the inverse of the operation performed by `parse_mul_tree`, i.e.
    compute_mul(parse_mul_tree(tree)) == tree.

    Parameters
    ----------
    tree
        A multiplication tree (as output by `parse_mul_tree`).

    Returns
    -------
    object
        A Variable that computes the multiplication represented by the tree.

    """
    neg, inputs = tree
    if inputs is None:
        raise AssertionError(
            "Function `compute_mul` found a missing leaf, did you forget to "
            "call `simplify_mul` on the tree first?"
        )
    elif isinstance(inputs, list):
        # Recurse through inputs.
        rval = mul(*list(map(compute_mul, inputs)))
    else:
        rval = inputs
    if neg:
        rval = -rval
    return rval


def perform_sigm_times_exp(
    tree,
    exp_x=None,
    exp_minus_x=None,
    sigm_x=None,
    sigm_minus_x=None,
    parent=None,
    child_idx=None,
    full_tree=None,
):
    """
    Core processing of the `local_sigm_times_exp` optimization.

    This recursive function operates on a multiplication tree as output by
    `parse_mul_tree`. It walks through the tree and modifies it in-place
    by replacing matching pairs (exp, sigmoid) with the desired optimized
    version.

    Parameters
    ----------
    tree
        The sub-tree to operate on.
    exp_x
        List of arguments x so that `exp(x)` exists somewhere in the whole
        multiplication tree. Each argument is a pair (x, leaf) with `x` the
        argument of the exponential, and `leaf` the corresponding leaf in the
        multiplication tree (of the form [n, exp(x)] -- see `parse_mul_tree`).
        If None, this argument is initialized to an empty list.
    exp_minus_x
        Similar to `exp_x`, but for `exp(-x)`.
    sigm_x
        Similar to `exp_x`, but for `sigmoid(x)`.
    sigm_minus_x
        Similar to `exp_x`, but for `sigmoid(-x)`.
    parent
        Parent of `tree` (None if `tree` is the global root).
    child_idx
        Index of `tree` in its parent's inputs (None if `tree` is the global
        root).
    full_tree
        The global multiplication tree (should not be set except by recursive
        calls to this function). Used for debugging only.

    Returns
    -------
    bool
        True if a modification was performed somewhere in the whole multiplication
        tree, or False otherwise.

    """
    if exp_x is None:
        exp_x = []
    if exp_minus_x is None:
        exp_minus_x = []
    if sigm_x is None:
        sigm_x = []
    if sigm_minus_x is None:
        sigm_minus_x = []
    if full_tree is None:
        full_tree = tree
    if False:  # Debug code.
        print("<perform_sigm_times_exp>")
        print(f"  full_tree   = {full_tree}")
        print(f"  tree        = {tree}")
        print(f"  exp_x       = {exp_x}")
        print(f"  exp_minus_x = {exp_minus_x}")
        print(f"  sigm_x      = {sigm_x}")
        print(f"  sigm_minus_x= {sigm_minus_x}")
    neg, inputs = tree
    if isinstance(inputs, list):
        # Recurse through inputs of the multiplication.
        rval = False
        for sub_idx, sub_tree in enumerate(inputs):
            rval |= perform_sigm_times_exp(
                tree=sub_tree,
                parent=tree,
                child_idx=sub_idx,
                exp_x=exp_x,
                exp_minus_x=exp_minus_x,
                sigm_x=sigm_x,
                sigm_minus_x=sigm_minus_x,
                full_tree=full_tree,
            )
        return rval
    else:
        # Reached a leaf: if it is an exponential or a sigmoid, then we
        # first attempt to find a match in leaves already visited.
        # If there is such a match, we modify the already-visited leaf
        # accordingly: for instance if we visited a leaf sigmoid(x), then
        # find later a -exp(-x), we replace the previous leaf by
        # -sigmoid(-x) and remove the -exp(-x) from the tree.
        # If no match is found, then we register this leaf so that it can
        # be found later while walking the tree.
        var = inputs
        keep_it = False
        exp_info = is_exp(var)
        if exp_info is not None:
            exp_neg, exp_arg = exp_info
            neg ^= exp_neg
            neg_arg = is_neg(exp_arg)
            if neg_arg is None:
                if not replace_leaf(exp_arg, sigm_minus_x, sigm_x, sigmoid, neg):
                    exp_x.append((exp_arg, tree))
                    keep_it = True
            else:
                if not replace_leaf(
                    neg_arg, sigm_x, sigm_minus_x, lambda x: sigmoid(-x), neg
                ):
                    exp_minus_x.append((neg_arg, tree))
                    keep_it = True
        elif var.owner and var.owner.op == sigmoid:
            sigm_arg = var.owner.inputs[0]
            neg_arg = is_neg(sigm_arg)
            if neg_arg is None:
                if not replace_leaf(
                    sigm_arg, exp_minus_x, sigm_minus_x, lambda x: sigmoid(-x), neg
                ):
                    sigm_x.append((sigm_arg, tree))
                    keep_it = True
            else:
                if not replace_leaf(neg_arg, exp_x, sigm_x, sigmoid, neg):
                    sigm_minus_x.append((neg_arg, tree))
                    keep_it = True
        else:
            # It is not an exponential nor a sigmoid.
            keep_it = True
        if not keep_it:
            # Delete this leaf, i.e. replace it by [False, None] (corresponding
            # to a multiplication by 1).
            assert parent is not None
            parent[1][child_idx] = [False, None]
        return not keep_it


@register_stabilize
@local_optimizer([mul])
def local_sigm_times_exp(fgraph, node):
    """
    exp(x) * sigm(-x) -> sigm(x)
    exp(-x) * sigm(x) -> sigm(-x)

    todo: add stack traces to the intermediate variables
    """
    # Bail early if it is not a multiplication.
    if node.op != mul:
        return None
    # Obtain tree of multiplications starting at this node.
    mul_tree = parse_mul_tree(node.outputs[0])
    # Perform core optimization.
    did_something = perform_sigm_times_exp(mul_tree)
    if not did_something:
        # No change.
        return None
    # The optimization may have introduced multiplications by 1 in the tree:
    # get rid of them.
    mul_tree = simplify_mul(mul_tree)
    # Recompute final output based on the updated tree.
    out = compute_mul(mul_tree)
    # keep the stack trace
    copy_stack_trace(node.outputs[0], out)
    return [out]


@register_stabilize
@local_optimizer([inv])
def local_inv_1_plus_exp(fgraph, node):
    """
    1/(1+exp(x)) -> sigm(-x)

    """
    # this optimization should be done for numerical stability
    # so we don't care to check client counts
    if node.op == inv:
        inv_arg = node.inputs[0]
        if inv_arg.owner and inv_arg.owner.op == add:
            scalars_, scalar_inputs, nonconsts = scalarconsts_rest(
                inv_arg.owner.inputs, only_process_constants=True
            )
            # scalar_inputs are potentially dimshuffled and fill'd scalars
            if len(nonconsts) == 1:
                if nonconsts[0].owner and nonconsts[0].owner.op == exp:
                    if scalars_ and np.allclose(np.sum(scalars_), 1):
                        out = _fill_chain(
                            sigmoid(neg(nonconsts[0].owner.inputs[0])),
                            scalar_inputs,
                        )
                        # keep combined stack traces of
                        #     exp(x):           nonconsts[0],
                        #     1 + exp(x):       inv_arg,
                        #     1 / (1 + exp(x)): node.outputs[0]
                        copy_stack_trace([nonconsts[0], inv_arg, node.outputs[0]], out)
                        return out


# Registration is below, and conditional.
@local_optimizer([sub])
def local_1msigmoid(fgraph, node):
    """
    1-sigm(x) -> sigm(-x)

    """
    if node.op == sub:
        sub_l, sub_r = node.inputs
        if len(fgraph.clients[sub_r]) > 1:
            return  # graph is using both sigm and 1-sigm
        if sub_r.owner and sub_r.owner.op == sigmoid:
            try:
                val_l = get_scalar_constant_value(sub_l)
            except NotScalarConstantError:
                return
            if np.allclose(np.sum(val_l), 1):
                out = sigmoid(-sub_r.owner.inputs[0])
                copy_stack_trace([sub_r, node.outputs[0]], out)
                return [out]


register_local_1msigmoid = False
# This is False because the Stabilize pattern above
# is looking for 1-sigm.  Also AlgebraicCanonizer turns neg into *(-1) and so
# this optimization might set off an unwanted chain of things.
# OTH - this transformation can be seen as pushing normal arithmetic either  below or above the
# sigmoidal nonlinearity... so if the canonicalized form had anything to say about that then it
# would be a consideration... anyway leaving False for now.

if register_local_1msigmoid:
    register_canonicalize(local_1msigmoid)
