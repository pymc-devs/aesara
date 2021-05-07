""" Tensor optimizations addressing the ops in basic.py."""
# TODO: intelligent merge for mul/add
# TODO: 0*x -> 0

import logging
import sys
import time
import traceback
import warnings
from collections import defaultdict
from io import StringIO

import numpy as np

import aesara
import aesara.scalar.basic as aes
from aesara import compile
from aesara.assert_op import Assert, assert_op
from aesara.compile.ops import ViewOp
from aesara.configdefaults import config
from aesara.graph import toolbox
from aesara.graph.basic import (
    Constant,
    Variable,
    ancestors,
    equal_computations,
    io_toposort,
)
from aesara.graph.fg import FunctionGraph, InconsistencyError
from aesara.graph.op import get_test_value
from aesara.graph.opt import (
    GlobalOptimizer,
    OpRemove,
    TopoOptimizer,
    check_chain,
    copy_stack_trace,
    in2out,
    local_optimizer,
)
from aesara.graph.optdb import SequenceDB
from aesara.graph.utils import (
    MethodNotDefined,
    TestValueError,
    get_variable_trace_string,
)
from aesara.printing import pprint
from aesara.tensor.basic import (
    Alloc,
    AllocEmpty,
    ARange,
    Flatten,
    Join,
    MakeVector,
    Rebroadcast,
    ScalarFromTensor,
    Split,
    TensorFromScalar,
    Tile,
    alloc,
    as_tensor_variable,
    cast,
    constant,
    extract_constant,
    fill,
    get_scalar_constant_value,
    get_vector_length,
    join,
    make_vector,
    ones_like,
    patternbroadcast,
    switch,
    tensor_copy,
    unbroadcast,
    zeros,
    zeros_like,
)
from aesara.tensor.elemwise import DimShuffle, Elemwise
from aesara.tensor.exceptions import NotScalarConstantError, ShapeError
from aesara.tensor.extra_ops import broadcast_shape
from aesara.tensor.math import Dot, add
from aesara.tensor.math import all as aet_all
from aesara.tensor.math import (
    and_,
    ceil_intdiv,
    dot,
    eq,
    ge,
    gt,
    le,
    lt,
    maximum,
    minimum,
    or_,
)
from aesara.tensor.shape import Reshape, Shape, Shape_i, shape, shape_padleft
from aesara.tensor.sort import TopKOp
from aesara.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedIncSubtensor1,
    AdvancedSubtensor1,
    IncSubtensor,
    Subtensor,
    advanced_inc_subtensor1,
    advanced_subtensor,
    advanced_subtensor1,
    as_index_constant,
    get_canonical_form_slice,
    get_idx_list,
)
from aesara.tensor.type import TensorType, discrete_dtypes, integer_dtypes, lscalar
from aesara.tensor.var import TensorConstant
from aesara.utils import NoDuplicateOptWarningFilter


_logger = logging.getLogger("aesara.tensor.basic_opt")
_logger.addFilter(NoDuplicateOptWarningFilter())


def _fill_chain(new_out, orig_inputs):
    for i in orig_inputs:
        new_out = fill(i, new_out)
    return [new_out]


def encompasses_broadcastable(b1, b2):
    """

    Parameters
    ----------
    b1
        The broadcastable attribute of a tensor type.
    b2
        The broadcastable attribute of a tensor type.

    Returns
    -------
    bool
        True if the broadcastable patterns b1 and b2 are such that b2 is
        broadcasted to b1's shape and not the opposite.

    """
    if len(b1) < len(b2):
        return False
    b1 = b1[-len(b2) :]
    return not any(v1 and not v2 for v1, v2 in zip(b1, b2))


def merge_broadcastables(broadcastables):
    return [all(bcast) for bcast in zip(*broadcastables)]


def scalarconsts_rest(inputs, elemwise=True, only_process_constants=False):
    """Partition a list of variables into two kinds:
    scalar constants, and the rest."""
    consts = []
    origconsts = []
    nonconsts = []
    for i in inputs:
        try:
            v = get_scalar_constant_value(
                i, elemwise=elemwise, only_process_constants=only_process_constants
            )
            consts.append(v)
            origconsts.append(i)
        except NotScalarConstantError:
            nonconsts.append(i)
    return consts, origconsts, nonconsts


def broadcast_like(value, template, fgraph, dtype=None):
    """
    Return a Variable with the same shape and dtype as the template,
    filled by broadcasting value through it. `value` will be cast as
    necessary.

    """
    value = as_tensor_variable(value)
    if value.type == template.type:
        return value
    if template not in fgraph.variables:
        raise NotImplementedError(
            "broadcast_like currently requires the "
            "template Variable to be in the fgraph already"
        )
    if dtype is None:
        dtype = template.dtype
    value = cast(value, dtype)
    if value.type == template.type:
        return value
    if hasattr(fgraph, "shape_feature"):
        new_shape = fgraph.shape_feature.shape_of[template]
    else:
        new_shape = template.shape
    rval = alloc(value, *new_shape)
    # the template may have 1s in its shape without being broadcastable
    if rval.broadcastable != template.broadcastable:
        rval = unbroadcast(
            rval,
            *[
                i
                for i in range(rval.ndim)
                if rval.broadcastable[i] and not template.broadcastable[i]
            ],
        )
    assert rval.type.dtype == dtype

    if rval.type.broadcastable != template.broadcastable:
        raise AssertionError(
            "rval.type.broadcastable is "
            + str(rval.type.broadcastable)
            + " but template.broadcastable is"
            + str(template.broadcastable)
        )

    return rval


class InplaceElemwiseOptimizer(GlobalOptimizer):
    """
    We parametrise it to make it work for Elemwise and GpuElemwise op.
    """

    def __init__(self, OP):
        self.op = OP

    def add_requirements(self, fgraph):
        from aesara.graph.destroyhandler import DestroyHandler

        fgraph.attach_feature(DestroyHandler())

    @staticmethod
    def print_profile(stream, prof, level=0):
        blanc = "    " * level
        print(blanc, "InplaceElemwiseOptimizer ", prof["opt"].op, file=stream)
        for k in [
            "node_before",
            "nb_call_replace",
            "nb_call_validate",
            "nb_inconsistent",
        ]:
            print(blanc, k, prof[k], file=stream)
        ndim = prof["ndim"]
        if ndim:
            print(blanc, "ndim", "nb", file=stream)
            for n in sorted(ndim.keys()):
                print(blanc, n, ndim[n], file=stream)

    def apply(self, fgraph):
        """
        Usage: InplaceElemwiseOptimizer(op).optimize(fgraph)

        Attempts to replace all Broadcast ops by versions of them
        that operate inplace. It operates greedily: for each Broadcast
        Op that is encountered, for each output, tries each input to
        see if it can operate inplace on that input. If so, makes the
        change and go to the next output or Broadcast Op.

        Examples
        --------

            `x + y + z -> x += y += z`

            `(x + y) * (x * y) -> (x += y) *= (x * y) or (x + y) *= (x *= y)`

        """
        # We should not validate too often as this takes too much time to
        # execute!
        # It is the _dfs_toposort() fct in aesara/graph/destroyhandler.py
        # that takes so much time.
        # Should we try to use another lib that does toposort?
        #   igraph: http://igraph.sourceforge.net/
        #   networkx: https://networkx.lanl.gov/
        # Should we try to use cython?
        #   Compiling only that fct is not enough, should we try to add the
        #   deque class too?
        #   And init the deque and other list to an upper bound number of
        #   elements?
        # Maybe Aesara should do online toposort as in
        #   http://code.google.com/p/acyclic
        #
        # The next longest optimizer is the canonizer phase.
        # Then I think it is the [io_?]toposort (need to validate) so check if
        # the solution is also applicable there.

        # We execute `validate` after this number of change.
        prof = {
            "opt": self,
            "node_before": len(fgraph.apply_nodes),
            "nb_call_replace": 0,
            "nb_call_validate": 0,
            "nb_inconsistent": 0,
            "ndim": defaultdict(lambda: 0),
        }

        check_each_change = config.tensor__insert_inplace_optimizer_validate_nb
        if check_each_change == -1:
            if len(fgraph.apply_nodes) > 500:
                check_each_change = 10
            else:
                check_each_change = 1

        nb_change_no_validate = 0
        chk = fgraph.checkpoint()

        if fgraph.update_mapping:
            update_outs = [fgraph.outputs[i] for i in fgraph.update_mapping]
        else:
            update_outs = []

        protected_inputs = [
            f.protected
            for f in fgraph._features
            if isinstance(f, aesara.compile.function.types.Supervisor)
        ]
        protected_inputs = sum(protected_inputs, [])  # flatten the list
        protected_inputs.extend(fgraph.outputs)
        for node in list(io_toposort(fgraph.inputs, fgraph.outputs)):
            op = node.op
            # gpuarray GpuElemwise inherit from Elemwise
            if not type(op) == self.op:
                continue
            # If big graph and the outputs are scalar, do not make it
            # inplace.
            if (
                check_each_change != 1
                and
                # If multiple outputs, they must all have the same size,
                # so only check the first.
                getattr(node.outputs[0].type, "ndim", -1) == 0
            ):
                continue

            if op.inplace_pattern:
                # Maybe this isn't needed anymore, but I don't want to
                # rish regression now. This case only happen if the
                # original node add already some inplace patter and we
                # still try to add more pattern.

                baseline = op.inplace_pattern
                candidate_outputs = [
                    i for i in range(len(node.outputs)) if i not in baseline
                ]
                # node inputs that are Constant, already destroyed,
                # or fgraph protected inputs and fgraph outputs can't be used as
                # inplace target.
                # Remove here as faster.
                candidate_inputs = [
                    i
                    for i in range(len(node.inputs))
                    if i not in baseline.values()
                    and not isinstance(node.inputs[i], Constant)
                    and
                    # the next line should not be costly most of the time.
                    not fgraph.has_destroyers([node.inputs[i]])
                    and node.inputs[i] not in protected_inputs
                ]
            else:
                baseline = []
                candidate_outputs = list(range(len(node.outputs)))
                # node inputs that are Constant, already destroyed,
                # fgraph protected inputs and fgraph outputs can't be used as inplace
                # target.
                # Remove here as faster.
                candidate_inputs = [
                    i
                    for i in range(len(node.inputs))
                    if not isinstance(node.inputs[i], Constant)
                    and not fgraph.has_destroyers([node.inputs[i]])
                    and node.inputs[i] not in protected_inputs
                ]

            verbose = False

            raised_warning = not verbose

            for candidate_output in candidate_outputs:

                # If the output of the node can be established as an update
                # output of the fgraph, visit the candidate_inputs in an order
                # that will improve the chances of making the node operate
                # inplace on the input it's meant to update
                candidate_out_var = node.outputs[candidate_output]
                sorted_candidate_inputs = candidate_inputs

                if candidate_out_var in update_outs:

                    # The candidate output is an update. Sort the
                    # variables in candidate_inputs in the following order:
                    # - Vars corresponding to the actual updated input
                    #   (best case scenario is for the node that procudes
                    #   an update to operate inplace on the variable to
                    #   update)
                    # - Vars computed inplace on the updates input (second
                    #   best scenario if for the node to work inplace on
                    #   a variable obtained by a chain of inplace on the
                    #   variable to update. In some cases, this will be
                    #   equivalent to operating inplace on the variable to
                    #   update)
                    # - Remaining variables
                    updated_inputs = []
                    for i, f_out in enumerate(fgraph.outputs):
                        if f_out is candidate_out_var and i in fgraph.update_mapping:
                            updated_inp_idx = fgraph.update_mapping[i]
                            updated_inputs.append(fgraph.inputs[updated_inp_idx])

                    updated_vars = []
                    vars_from_inplace = []
                    other_vars = []
                    for inp_idx in candidate_inputs:
                        inp = node.inputs[inp_idx]
                        if inp in updated_inputs:
                            # the candidate input is the actual updated input
                            updated_vars.append(inp_idx)
                        elif (
                            hasattr(fgraph, "destroy_handler")
                            and inp.owner
                            and any(
                                [
                                    fgraph.destroy_handler.root_destroyer.get(
                                        up_inp, None
                                    )
                                    is inp.owner
                                    for up_inp in updated_inputs
                                ]
                            )
                        ):

                            # the candidate input is a variable computed
                            # inplace on the updated input via a sequence of
                            # one or more inplace operations
                            vars_from_inplace.append(inp_idx)
                        else:
                            other_vars.append(inp_idx)

                    sorted_candidate_inputs = (
                        updated_vars + vars_from_inplace + other_vars
                    )

                for candidate_input in sorted_candidate_inputs:
                    # remove inputs that don't have the same dtype as the output
                    if (
                        node.inputs[candidate_input].type
                        != node.outputs[candidate_output].type
                    ):
                        continue

                    inplace_pattern = dict(baseline)
                    inplace_pattern[candidate_output] = candidate_input
                    try:
                        if hasattr(op.scalar_op, "make_new_inplace"):
                            new_scal = op.scalar_op.make_new_inplace(
                                aes.transfer_type(
                                    *[
                                        inplace_pattern.get(i, o.dtype)
                                        for i, o in enumerate(node.outputs)
                                    ]
                                )
                            )
                        else:
                            new_scal = op.scalar_op.__class__(
                                aes.transfer_type(
                                    *[
                                        inplace_pattern.get(i, None)
                                        for i in range(len(node.outputs))
                                    ]
                                )
                            )
                        new_outputs = self.op(new_scal, inplace_pattern)(
                            *node.inputs, **dict(return_list=True)
                        )
                        new_node = new_outputs[0].owner

                        for r, new_r in zip(node.outputs, new_outputs):
                            prof["nb_call_replace"] += 1
                            fgraph.replace(
                                r, new_r, reason="inplace_elemwise_optimizer"
                            )
                        nb_change_no_validate += 1
                        prof["ndim"][candidate_out_var.ndim] += 1
                        if nb_change_no_validate >= check_each_change:
                            prof["nb_call_validate"] += 1
                            fgraph.validate()
                            chk = fgraph.checkpoint()
                            nb_change_no_validate = 0
                    except (ValueError, InconsistencyError) as e:
                        prof["nb_inconsistent"] += 1
                        if check_each_change != 1 and not raised_warning:
                            print(
                                (
                                    "Some inplace optimization was not "
                                    "performed due to unexpected error:"
                                ),
                                file=sys.stderr,
                            )
                            print(e, file=sys.stderr)
                            raised_warning = True
                        fgraph.revert(chk)
                        continue
                    candidate_inputs.remove(candidate_input)
                    node = new_node
                    baseline = inplace_pattern
                    break

        if nb_change_no_validate > 0:
            try:
                fgraph.validate()
            except Exception:
                if not raised_warning:
                    print(
                        (
                            "Some inplace optimization was not "
                            "performed due to unexpected error"
                        ),
                        file=sys.stderr,
                    )
                fgraph.revert(chk)
        return prof

    def print_summary(self, stream=sys.stdout, level=0, depth=-1):
        print(
            f"{' ' * level}{self.__class__.__name__} ({self.op})",
            file=stream,
        )
        return inplace_elemwise_optimizer


inplace_elemwise_optimizer = InplaceElemwiseOptimizer(Elemwise)
compile.optdb.register(
    "inplace_elemwise_opt",
    inplace_elemwise_optimizer,
    75,
    "inplace_opt",  # for historic reason
    "inplace_elemwise_optimizer",
    "fast_run",
    "inplace",
)


def register_useless(lopt, *tags, **kwargs):
    if type(lopt) == str:

        def register(inner_lopt):
            return register_useless(inner_lopt, lopt, *tags, **kwargs)

        return register
    else:
        name = kwargs.pop("name", None) or lopt.__name__

        compile.mode.local_useless.register(
            name, lopt, "last", "fast_run", *tags, **kwargs
        )
        return lopt


def register_canonicalize(lopt, *tags, **kwargs):
    if type(lopt) == str:

        def register(inner_lopt):
            return register_canonicalize(inner_lopt, lopt, *tags, **kwargs)

        return register
    else:
        name = kwargs.pop("name", None) or lopt.__name__
        compile.optdb["canonicalize"].register(name, lopt, "fast_run", *tags, **kwargs)
        return lopt


def register_stabilize(lopt, *tags, **kwargs):
    if type(lopt) == str:

        def register(inner_lopt):
            return register_stabilize(inner_lopt, lopt, *tags, **kwargs)

        return register
    else:
        name = kwargs.pop("name", None) or lopt.__name__
        compile.optdb["stabilize"].register(name, lopt, "fast_run", *tags, **kwargs)
        return lopt


def register_specialize(lopt, *tags, **kwargs):
    if type(lopt) == str:

        def register(inner_lopt):
            return register_specialize(inner_lopt, lopt, *tags, **kwargs)

        return register
    else:
        name = kwargs.pop("name", None) or lopt.__name__
        compile.optdb["specialize"].register(name, lopt, "fast_run", *tags, **kwargs)
        return lopt


def register_uncanonicalize(lopt, *tags, **kwargs):
    if type(lopt) == str:

        def register(inner_lopt):
            return register_uncanonicalize(inner_lopt, lopt, *tags, **kwargs)

        return register
    else:
        name = (kwargs and kwargs.pop("name", None)) or lopt.__name__
        compile.optdb["uncanonicalize"].register(
            name, lopt, "fast_run", *tags, **kwargs
        )
        return lopt


def register_specialize_device(lopt, *tags, **kwargs):
    if type(lopt) == str:

        def register(inner_lopt):
            return register_specialize_device(inner_lopt, lopt, *tags, **kwargs)

        return register
    else:
        name = (kwargs and kwargs.pop("name", None)) or lopt.__name__
        compile.optdb["specialize_device"].register(
            name, lopt, "fast_run", *tags, **kwargs
        )
        return lopt


def apply_local_dimshuffle_lift(fgraph, var):
    # return var
    # lift recursively
    if not var.owner:
        return var
    new = local_dimshuffle_lift.transform(fgraph, var.owner)
    if new:
        return new[0]
    return var


# Checks for two types of useless dimshuffles:
#   1 - dimshuffle all dimensions in order.
#   2 - dimshuffle a broadcastable dimension.
def is_dimshuffle_useless(new_order, input):
    is_useless = True
    if len(new_order) == input.type.ndim:
        all_broadcastable_dims = [
            i
            for (i, is_broadcastable) in enumerate(input.type.broadcastable)
            if is_broadcastable
        ] + ["x"]
        for i in range(input.type.ndim):
            if new_order[i] == i or (
                i in all_broadcastable_dims and new_order[i] in all_broadcastable_dims
            ):
                is_useless = True
            else:
                is_useless = False
                break
    else:
        is_useless = False
    return is_useless


@local_optimizer([DimShuffle])
def local_dimshuffle_lift(fgraph, node):
    """
    "Lifts" DimShuffle through Elemwise operations and merges
    consecutive DimShuffles. Basically, applies the following
    transformations on the whole graph:

    DimShuffle(Elemwise(x, y)) => Elemwise(DimShuffle(x), DimShuffle(y))
    DimShuffle(DimShuffle(x)) => DimShuffle(x)
    DimShuffle{0,1,...}(x) => x (when the dimshuffle do nothing)

    After this transform, clusters of Elemwise operations are
    void of DimShuffle operations.

    """
    op = node.op
    if not isinstance(op, DimShuffle):
        return False

    input = node.inputs[0]
    inode = input.owner
    new_order = op.new_order
    if inode and isinstance(inode.op, Elemwise) and (len(fgraph.clients[input]) == 1):
        # Don't use make_node to have tag.test_value set.
        new_inputs = []
        for inp in inode.inputs:
            new_inp = op.__class__(inp.type.broadcastable, op.new_order)(inp)
            new_inputs.append(apply_local_dimshuffle_lift(fgraph, new_inp))
        copy_stack_trace(node.outputs[0], new_inputs)
        ret = inode.op(*new_inputs, **dict(return_list=True))
        return ret
    if inode and isinstance(inode.op, DimShuffle):
        new_order = [x == "x" and "x" or inode.op.new_order[x] for x in new_order]
        input = inode.inputs[0]

    if is_dimshuffle_useless(new_order, input):
        return [input]
    elif inode and isinstance(inode.op, DimShuffle):
        ret = op.__class__(input.type.broadcastable, new_order)(input)
        ret = apply_local_dimshuffle_lift(fgraph, ret)
        copy_stack_trace(node.outputs[0], ret)
        return [ret]


@register_canonicalize
@local_optimizer([Reshape])
def local_useless_dimshuffle_in_reshape(fgraph, node):
    """
    Removes useless DimShuffle operation inside Reshape:

      reshape(vector.dimshuffle('x', 0), shp) => reshape(vector, shp)
      reshape(matrix.dimshuffle('x', 0, 'x', 1), shp) => reshape(matrix, shp)
      reshape(row.dimshuffle(1, 'x'), shp) => reshape(row, shp)
      reshape(col.dimshuffle(0), shp) => reshape(col, shp)

    """
    op = node.op
    if not isinstance(op, Reshape):
        return False
    if not (
        node.inputs[0].owner is not None
        and isinstance(node.inputs[0].owner.op, DimShuffle)
    ):
        return False

    new_order = node.inputs[0].owner.op.new_order
    input = node.inputs[0].owner.inputs[0]
    broadcastables = node.inputs[0].broadcastable
    new_order_of_nonbroadcast = []
    for i, bd in zip(new_order, broadcastables):
        if not bd:
            new_order_of_nonbroadcast.append(i)
    no_change_in_order = all(
        new_order_of_nonbroadcast[i] <= new_order_of_nonbroadcast[i + 1]
        for i in range(len(new_order_of_nonbroadcast) - 1)
    )
    if no_change_in_order:
        shape = node.inputs[1]
        ret = op.__class__(node.outputs[0].ndim)(input, shape)
        copy_stack_trace(node.outputs[0], ret)
        return [ret]


register_canonicalize(local_dimshuffle_lift)
register_specialize(local_dimshuffle_lift)

######################
# Casting operations #
######################


@register_canonicalize
@register_specialize
@local_optimizer([TensorFromScalar])
def local_tensor_scalar_tensor(fgraph, node):
    """tensor_from_scalar(scalar_from_tensor(x)) -> x"""
    if isinstance(node.op, TensorFromScalar):
        s = node.inputs[0]
        if s.owner and isinstance(s.owner.op, ScalarFromTensor):
            t = s.owner.inputs[0]

            # We don't need to copy over any stack traces here
            return [t]


@register_canonicalize
@register_specialize
@local_optimizer([ScalarFromTensor])
def local_scalar_tensor_scalar(fgraph, node):
    """scalar_from_tensor(tensor_from_scalar(x)) -> x"""
    if isinstance(node.op, ScalarFromTensor):
        t = node.inputs[0]
        if t.owner and isinstance(t.owner.op, TensorFromScalar):
            s = t.owner.inputs[0]

            # We don't need to copy over any stack traces here
            return [s]


#####################################
# ShapeFeature, Shape optimizations
#####################################
class MakeVectorPrinter:
    def process(self, r, pstate):
        if r.owner is None:
            raise TypeError("Can only print make_vector.")
        elif isinstance(r.owner.op, MakeVector):
            old_precedence = getattr(pstate, "precedence", None)
            try:
                pstate.precedence = 1000
                s = [pstate.pprinter.process(input) for input in r.owner.inputs]
            finally:
                pstate.precedence = old_precedence
            return f"[{', '.join(s)}]"
        else:
            raise TypeError("Can only print make_vector.")


pprint.assign(MakeVector, MakeVectorPrinter())


class ShapeFeature(toolbox.Feature):
    """Graph optimizer for removing all calls to shape().

    This optimizer replaces all Shapes and Subtensors of Shapes with
    Shape_i and MakeVector Ops.

    This optimizer has several goals:

    1. to 'lift' Shapes to as close to the inputs as possible.

    2. to infer the shape of every node in the graph in terms of the
       input shapes.

    3. remove all fills ``(aet.second, aet.fill)`` from the graph

    Lifting shapes as close to the inputs as possible is important for
    canonicalization because it is very bad form to have to compute
    something just to know how big it will be.  Firstly, it is a waste
    of time to compute such outputs.  But it is important to get rid
    of these outputs as early as possible in the compilation process
    because the extra computations make it appear as if many internal
    graph nodes have multiple clients.  Many optimizations refuse to
    work on nodes with multiple clients.

    Lifting is done by using an `<Op>.infer_shape` function if one is
    present, or else using a conservative default.  An Op that
    supports shape-lifting should define a infer_shape(self, fgraph, node,
    input_shapes) function.  The argument input_shapes is a tuple of
    tuples... there is an interior tuple for each input to the node.
    The tuple has as many elements as dimensions.  The element in
    position i of tuple j represents the i'th shape component of the
    j'th input.  The function should return a tuple of tuples.  One
    output tuple for each node.output.  Again, the i'th element of the
    j'th output tuple represents the output[j].shape[i] of the
    function.  If an output is not a TensorType, then None should be
    returned instead of a tuple for that output.

    For example the infer_shape for a matrix-matrix product would accept
    input_shapes=((x0,x1), (y0,y1)) and return ((x0, y1),).

    Inferring the shape of internal nodes in the graph is important
    for doing size-driven optimizations.  If we know how big various
    intermediate results will be, we can estimate the cost of many Ops
    accurately, and generate c-code that is specific [e.g. unrolled]
    to particular sizes.

    In cases where you cannot figure out the shape, raise a ShapeError.

    Notes
    -----
    Right now there is only the ConvOp that could really take
    advantage of this shape inference, but it is worth it even
    just for the ConvOp.  All that's necessary to do shape
    inference is 1) to mark shared inputs as having a particular
    shape, either via a .tag or some similar hacking; and 2) to
    add an optional In() argument to promise that inputs will
    have a certain shape (or even to have certain shapes in
    certain dimensions). We can't automatically infer the shape of
    shared variables as they can change of shape during the
    execution by default.  (NOT IMPLEMENTED YET, BUT IS IN TRAC)


    **Using Shape information in Optimizations**

    To use this shape information in OPTIMIZATIONS, use the
    ``shape_of`` dictionary.

    For example:

    .. code-block:: python

        try:
            shape_of = fgraph.shape_feature.shape_of
        except AttributeError:
            # This can happen when the mode doesn't include the ShapeFeature.
            return

        shape_of_output_zero = shape_of[node.output[0]]

    The ``shape_of_output_zero`` symbol will contain a tuple, whose
    elements are either integers or symbolic integers.

    TODO: check to see if the symbols are necessarily
    non-constant... or are integer literals sometimes Aesara
    constants?? That would be confusing.

    """

    def get_node_infer_shape(self, node):
        try:
            shape_infer = node.op.infer_shape
        except AttributeError:
            shape_infer = self.default_infer_shape

        try:
            o_shapes = shape_infer(
                self.fgraph, node, [self.shape_of[r] for r in node.inputs]
            )
        except ShapeError:
            o_shapes = self.default_infer_shape(
                self.fgraph, node, [self.shape_of[r] for r in node.inputs]
            )
        except NotImplementedError as e:
            raise NotImplementedError(
                "Code called by infer_shape failed raising a "
                "NotImplementedError. Raising NotImplementedError to "
                "indicate that a shape cannot be computed is no longer "
                "supported, and one should now use ShapeError "
                f"instead. The original exception message is: {e}"
            ).with_traceback(e.__traceback__)
        except Exception as e:
            msg = (
                f"Failed to infer_shape from Op {node.op}.\nInput shapes: "
                f"{[self.shape_of[r] for r in node.inputs]}\nException encountered during infer_shape: "
                f"{type(e)}\nException message: {str(e)}\nTraceback: {traceback.format_exc()}"
            )
            if config.on_shape_error == "raise":
                raise Exception(msg).with_traceback(e.__traceback__)
            else:
                _logger.warning(msg)
            o_shapes = self.default_infer_shape(
                self.fgraph, node, [self.shape_of[r] for r in node.inputs]
            )

        return o_shapes

    def get_shape(self, var, idx):
        """Optimization can call this to get the current shape_i

        It is better to call this then use directly shape_of[var][idx]
        as this method should update shape_of if needed.

        TODO: Up to now, we don't update it in all cases. Update in all cases.
        """
        r = self.shape_of[var][idx]
        if (
            r.owner
            and isinstance(r.owner.op, Shape_i)
            and r.owner.inputs[0] not in self.fgraph.variables
        ):
            assert var.owner
            node = var.owner
            # recur on inputs
            for i in node.inputs:
                if getattr(i, "ndim", None) > 0:
                    self.get_shape(i, 0)
            o_shapes = self.get_node_infer_shape(node)
            assert len(o_shapes) == len(node.outputs)

            # Only change the variables and dimensions that would introduce
            # extra computation
            for new_shps, out in zip(o_shapes, node.outputs):
                if not hasattr(out, "ndim"):
                    continue

                merged_shps = list(self.shape_of[out])
                changed = False
                for i in range(out.ndim):
                    n_r = merged_shps[i]
                    if (
                        n_r.owner
                        and isinstance(n_r.owner.op, Shape_i)
                        and n_r.owner.inputs[0] not in self.fgraph.variables
                    ):
                        changed = True
                        merged_shps[i] = new_shps[i]
                if changed:
                    self.set_shape(out, merged_shps, override=True)
            r = self.shape_of[var][idx]
        return r

    def shape_ir(self, i, r):
        """Return symbolic r.shape[i] for tensor variable r, int i."""
        if hasattr(r.type, "broadcastable") and r.type.broadcastable[i]:
            return self.lscalar_one
        else:
            # Do not call make_node for test_value
            s = Shape_i(i)(r)
            try:
                s = get_scalar_constant_value(s)
            except NotScalarConstantError:
                pass
            return s

    def shape_tuple(self, r):
        """Return a tuple of symbolic shape vars for tensor variable r."""
        if not hasattr(r, "ndim"):
            # This happen for NoneConst.
            return None
        return tuple([self.shape_ir(i, r) for i in range(r.ndim)])

    def default_infer_shape(self, fgraph, node, i_shapes):
        """Return a list of shape tuple or None for the outputs of node.

        This function is used for Ops that don't implement infer_shape.
        Ops that do implement infer_shape should use the i_shapes parameter,
        but this default implementation ignores it.

        """
        rval = []
        for r in node.outputs:
            try:
                rval.append(self.shape_tuple(r))
            except AttributeError:
                rval.append(None)
        return rval

    def unpack(self, s_i, var):
        """Return a symbolic integer scalar for the shape element s_i.

        The s_i argument was produced by the infer_shape() of an Op subclass.

        var: the variable that correspond to s_i. This is just for
        error reporting.

        """
        # unpack the s_i that the Op returned
        assert s_i is not None
        if s_i == 1:
            # don't make the optimizer merge a zillion ones together
            # by always returning the same object to represent 1
            return self.lscalar_one
        if isinstance(s_i, float) and int(s_i) == s_i:
            s_i = int(s_i)
        if isinstance(s_i, (np.integer, int)) or (
            isinstance(s_i, np.ndarray) and s_i.ndim == 0
        ):
            # this shape is a constant
            if s_i < 0:
                msg = "There is a negative shape in the graph!"
                msg += get_variable_trace_string(var)
                # The rest of the pipeline don't handle correctly this
                # case.  So we have 2 choices, stop compilation or
                # consider the shape as unknow.  As we have more
                # chance to give the stack trace here then later, I
                # choose that options as it would give better error
                # message.
                raise AssertionError(msg)
            return constant(s_i, dtype="int64")
        if isinstance(s_i, (tuple, list)):
            # this dimension is the same as many of the inputs
            # which tells us that if one of the inputs is known,
            # the others all become known.
            # TODO: should be implemented in Elemwise, and Dot
            #
            # worst case, we loop over shape_of and replace things
            raise NotImplementedError(s_i)

        # s_i is x.shape[i] for some x, we change it to shape_of[x][i]
        if (
            s_i.owner
            and isinstance(s_i.owner.op, Subtensor)
            and s_i.owner.inputs[0].owner
            and isinstance(s_i.owner.inputs[0].owner.op, Shape)
        ):
            assert s_i.ndim == 0
            assert len(s_i.owner.op.idx_list) == 1

            # The current Subtensor always put constant index in the graph.
            # This was not True in the past. So call the Subtensor function
            # that will return the right index.
            idx = get_idx_list(s_i.owner.inputs, s_i.owner.op.idx_list)
            assert len(idx) == 1
            idx = idx[0]
            try:
                i = get_scalar_constant_value(idx)
            except NotScalarConstantError:
                pass
            else:
                # Executed only if no exception was raised
                x = s_i.owner.inputs[0].owner.inputs[0]
                # x should already have been imported, and should be in shape_of.
                s_i = self.shape_of[x][i]

        if s_i.type.dtype in integer_dtypes:
            if getattr(s_i.type, "ndim", 0):
                raise TypeError("Shape element must be scalar", s_i)
            return s_i
        else:
            raise TypeError(
                "Unsupported shape element", s_i, type(s_i), getattr(s_i, "type", None)
            )

    def set_shape(self, r, s, override=False):
        """Assign the shape `s` to previously un-shaped variable `r`.

        Parameters
        ----------
        r : a variable
        s : None or a tuple of symbolic integers
        override : If False, it mean r is a new object in the fgraph.
            If True, it mean r is already in the fgraph and we want to
            override its shape.

        """
        if not override:
            assert r not in self.shape_of, "r already in shape_of"
        if s is None:
            self.shape_of[r] = s
        else:
            if not isinstance(s, (tuple, list)):
                raise TypeError("shapes must be tuple/list", (r, s))

            if r.ndim != len(s):
                sio = StringIO()
                aesara.printing.debugprint(r, file=sio, print_type=True)
                raise AssertionError(
                    f"Something inferred a shape with {len(s)} dimensions "
                    f"for a variable with {int(r.ndim)} dimensions"
                    f" for the variable:\n{sio.getvalue()}"
                )

            shape_vars = []
            for i in range(r.ndim):
                if hasattr(r.type, "broadcastable") and r.type.broadcastable[i]:
                    shape_vars.append(self.lscalar_one)
                else:
                    shape_vars.append(self.unpack(s[i], r))
            assert all(
                [
                    not hasattr(r.type, "broadcastable")
                    or not r.type.broadcastable[i]
                    or
                    # The two following comparison are a speed optimization
                    # But we never timed this speed optimization!
                    self.lscalar_one.equals(shape_vars[i])
                    or self.lscalar_one.equals(extract_constant(shape_vars[i]))
                    for i in range(r.ndim)
                ]
            )
            self.shape_of[r] = tuple(shape_vars)
            for sv in shape_vars:
                self.shape_of_reverse_index.setdefault(sv, set()).add(r)

    def update_shape(self, r, other_r):
        """Replace shape of r by shape of other_r.

        If, on some dimensions, the shape of other_r is not informative,
        keep the shape of r on those dimensions.

        """
        # other_r should already have a shape
        assert other_r in self.shape_of, ("other_r not in shape_of", other_r)
        other_shape = self.shape_of[other_r]

        # If other_shape has no information, call is pointless.
        if other_shape is None:
            return

        if r in self.shape_of:
            r_shape = self.shape_of[r]
        else:
            # If no info is known on r's shape, use other_shape
            self.set_shape(r, other_shape)
            return
        if (
            other_r.owner
            and r.owner
            and other_r.owner.inputs == r.owner.inputs
            and other_r.owner.op == r.owner.op
        ):
            # We are doing a merge. So the 2 shapes graph will be the
            # same.  This is only a speed optimization to call
            # ancestors() less frequently.
            return

        # Merge other_shape with r_shape, giving the priority to other_shape
        merged_shape = []
        for i, ps in enumerate(other_shape):
            if r_shape is None and other_shape:
                merged_shape.append(other_shape[i])
            elif (
                ps.owner
                and isinstance(getattr(ps.owner, "op", None), Shape_i)
                and ps.owner.op.i == i
                and ps.owner.inputs[0] in (r, other_r)
            ):
                # If other_shape[i] is uninformative, use r_shape[i].
                # For now, we consider 2 cases of uninformative other_shape[i]:
                #  - Shape_i(i)(other_r);
                #  - Shape_i(i)(r).
                merged_shape.append(r_shape[i])
            elif isinstance(r_shape[i], (Constant, int)):
                # We do this to call less often ancestors and make
                # sure we have the simplest shape possible.
                merged_shape.append(r_shape[i])
            elif isinstance(other_shape[i], (Constant, int)):
                # We do this to call less often ancestors and make
                # sure we have the simplest shape possible.
                merged_shape.append(other_shape[i])
            elif other_shape[i] == r_shape[i]:
                # This mean the shape is equivalent
                # We do not want to do the ancestor check in those cases
                merged_shape.append(r_shape[i])
            elif r_shape[i] in ancestors([other_shape[i]]):
                # Another case where we want to use r_shape[i] is when
                # other_shape[i] actually depends on r_shape[i]. In that case,
                # we do not want to substitute an expression with another that
                # is strictly more complex. Such a substitution could also lead
                # to cycles: if (in the future) r_shape[i] gets replaced by an
                # expression of other_shape[i], other_shape[i] may end up
                # depending on itself.
                merged_shape.append(r_shape[i])
            else:
                merged_shape.append(other_shape[i])
        assert all(
            [
                (
                    not hasattr(r.type, "broadcastable")
                    or not r.type.broadcastable[i]
                    and not other_r.type.broadcastable[i]
                )
                or
                # The two following comparison are a speed optimization
                # But we never timed this speed optimization!
                self.lscalar_one.equals(merged_shape[i])
                or self.lscalar_one.equals(
                    extract_constant(merged_shape[i], only_process_constants=True)
                )
                for i in range(r.ndim)
            ]
        )
        self.shape_of[r] = tuple(merged_shape)
        for sv in self.shape_of[r]:
            self.shape_of_reverse_index.setdefault(sv, set()).add(r)

    def set_shape_i(self, r, i, s_i):
        """Replace element i of shape_of[r] by s_i"""
        assert r in self.shape_of
        prev_shape = self.shape_of[r]
        # prev_shape is a tuple, so we cannot change it inplace,
        # so we build another one.
        new_shape = []
        for j, s_j in enumerate(prev_shape):
            if j == i:
                new_shape.append(self.unpack(s_i, r))
            else:
                new_shape.append(s_j)
        assert all(
            [
                not hasattr(r.type, "broadcastable") or not r.type.broadcastable[idx] or
                # The two following comparison are a speed optimization
                # But we never timed this speed optimization!
                self.lscalar_one.equals(new_shape[idx])
                or self.lscalar_one.equals(extract_constant(new_shape[idx]))
                for idx in range(r.ndim)
            ]
        )
        self.shape_of[r] = tuple(new_shape)
        for sv in self.shape_of[r]:
            self.shape_of_reverse_index.setdefault(sv, set()).add(r)

    def init_r(self, r):
        """Register r's shape in the shape_of dictionary."""
        if r not in self.shape_of:
            self.set_shape(r, self.shape_tuple(r))

    def make_vector_shape(self, r):
        return make_vector(*self.shape_of[r])

    def on_attach(self, fgraph):

        if getattr(self, "fgraph", None):
            raise ValueError("This ShapeFeature is already attached to a graph")

        self.fgraph = fgraph

        if hasattr(fgraph, "shape_feature"):
            raise ValueError("This FunctionGraph already has a ShapeFeature")

        fgraph.shape_feature = self
        # Must be local to the object as otherwise we reuse the same
        # variable for multiple fgraph!
        self.lscalar_one = constant(1, dtype="int64")
        assert self.lscalar_one.type == lscalar

        self.fgraph = fgraph
        # Variable -> tuple(scalars) or None  (All tensor vars map to tuple)
        self.shape_of = {}
        # Variable ->
        self.scheduled = {}
        # shape var -> graph v
        self.shape_of_reverse_index = {}

        for node in fgraph.toposort():
            self.on_import(fgraph, node, reason="on_attach")

    def on_detach(self, fgraph):
        self.shape_of = {}
        self.scheduled = {}
        self.shape_of_reverse_index = {}
        self.fgraph = None
        del fgraph.shape_feature

    def on_import(self, fgraph, node, reason):
        if node.outputs[0] in self.shape_of:
            # this is a revert, not really an import
            for r in node.outputs + node.inputs:
                assert r in self.shape_of
            return

        for i, r in enumerate(node.inputs):
            # make sure we have shapes for the inputs
            self.init_r(r)

        o_shapes = self.get_node_infer_shape(node)

        # this is packed information
        # an element of o_shapes is either None or a tuple
        #   elements of the tuple can be either strings, or ints
        if len(o_shapes) != len(node.outputs):
            raise Exception(
                (
                    f'The infer_shape method for the Op "{node.op}" returned a list '
                    f"with the wrong number of element: len(o_shapes) = {len(o_shapes)} "
                    f" != len(node.outputs) = {len(node.outputs)}"
                )
            )

        # Ensure shapes are in 'int64'. This is to make sure the assert
        # found in the `local_useless_subtensor` optimization does not fail.
        for sh_idx, sh in enumerate(o_shapes):
            if sh is None:
                continue
            if not isinstance(sh, (list, tuple)):
                raise ValueError(
                    f"infer_shape of {node} didn't return a list of"
                    f" list. It returned '{o_shapes}'"
                )
            new_shape = []
            for i, d in enumerate(sh):
                # Note: we ignore any shape element that is not typed (i.e.,
                # does not have a 'dtype' attribute). This means there may
                # still remain int elements that are int32 on 32-bit platforms,
                # but this works with `local_useless_subtensor`, so for now we
                # keep it this way. See #266 for a better long-term fix.
                if getattr(d, "dtype", "int64") != "int64":
                    assert d.dtype in discrete_dtypes, (node, d.dtype)
                    assert str(d.dtype) != "uint64", node
                    new_shape += sh[len(new_shape) : i + 1]
                    if isinstance(d, Constant):
                        casted_d = constant(d.data, dtype="int64")
                    else:
                        casted_d = cast(d, "int64")
                    new_shape[i] = casted_d
            if new_shape:
                # We replace the shape with wrong dtype by the one with
                # 'int64'.
                new_shape += sh[len(new_shape) :]
                o_shapes[sh_idx] = tuple(new_shape)

        for r, s in zip(node.outputs, o_shapes):
            self.set_shape(r, s)

    def on_change_input(self, fgraph, node, i, r, new_r, reason):
        if new_r not in self.shape_of:
            # It happen that the fgraph didn't called on_import for some
            # new_r.  This happen when new_r don't have an
            # owner(i.e. it is a constant or an input of the graph)
            # update_shape suppose that r and new_r are in shape_of.
            self.init_r(new_r)

        # This tells us that r and new_r must have the same shape if
        # we didn't know that the shapes are related, now we do.
        self.update_shape(new_r, r)

        # change_input happens in two cases:
        # 1) we are trying to get rid of r, or
        # 2) we are putting things back after a failed transaction.

        # In case 1, if r has a shape_i client, we will want to
        # replace the shape_i of r with the shape of new_r.  Say that
        # r is *scheduled*.
        # At that point, node is no longer a client of r, but of new_r
        for (shpnode, idx) in fgraph.clients[r] + [(node, i)]:
            if isinstance(getattr(shpnode, "op", None), Shape_i):
                idx = shpnode.op.i
                repl = self.shape_of[new_r][idx]
                if repl.owner is shpnode:
                    # This mean the replacement shape object is
                    # exactly the same as the current shape object. So
                    # no need for replacement. This happen for example
                    # with the InputToGpuOptimizer optimizer.
                    continue
                if (
                    repl.owner
                    and repl.owner.inputs[0] is shpnode.inputs[0]
                    and isinstance(repl.owner.op, Shape_i)
                    and repl.owner.op.i == shpnode.op.i
                ):
                    # The replacement is a shape_i of the same
                    # input. So no need to do this equivalent
                    # replacement.
                    continue

                if shpnode.outputs[0] in ancestors([repl]):
                    raise InconsistencyError(
                        "This substitution would insert a cycle in the graph:"
                        f"node: {node}, i: {i}, r: {r}, new_r: {new_r}"
                    )

                self.scheduled[shpnode] = new_r
        # In case 2, if r is a variable that we've scheduled for shape update,
        # then we should cancel it.
        unscheduled = [k for k, v in self.scheduled.items() if v == r]
        for k in unscheduled:
            del self.scheduled[k]

        # In either case, r could be in shape_of.values(), that is, r itself
        # is the shape of  something. In that case, we want to update
        # the value in shape_of, to keep it up-to-date.
        for v in self.shape_of_reverse_index.get(r, []):
            # The reverse index is only approximate. It is not updated on
            # deletion of variables, or on change_input so it might be the
            # case that there are a few extra `v`'s in it that no longer have
            # a shape of r or possibly have been deleted from shape_of
            # entirely. The important thing is that it permits to recall
            # all variables with r in their shape.
            for ii, svi in enumerate(self.shape_of.get(v, [])):
                if svi == r:
                    self.set_shape_i(v, ii, new_r)
        self.shape_of_reverse_index[r] = set()

    def same_shape(self, x, y, dim_x=None, dim_y=None):
        """Return True if we are able to assert that x and y have the
        same shape.

        dim_x and dim_y are optional. If used, they should be an index
        to compare only 1 dimension of x and y.

        """
        sx = self.shape_of[x]
        sy = self.shape_of[y]
        if sx is None or sy is None:
            return False
        if dim_x is not None:
            sx = [sx[dim_x]]
        if dim_y is not None:
            sy = [sy[dim_y]]
        assert len(sx) == len(sy)

        # We look on each dimensions we want to compare.
        # If any of them can't be asserted to be equal, return False.
        # Otherwise, we return True at the end.
        for dx, dy in zip(sx, sy):
            if dx is dy:
                continue
            # Need to try to find that they are the same shape. We
            # need to compare the full graph. It could be slow. So I
            # just implement for now the case of Shape_i.
            if not dx.owner or not dy.owner:
                return False
            if not isinstance(dx.owner.op, Shape_i) or not isinstance(
                dy.owner.op, Shape_i
            ):
                return False
            opx = dx.owner.op
            opy = dy.owner.op
            if not (opx.i == opy.i):
                return False
            # FB I'm not sure if this handle correctly constants.
            if dx.owner.inputs[0] == dy.owner.inputs[0]:
                continue
            # To be sure to cover all case, call equal_computation.
            # Can't use aesara.graph.basic.is_same_graph(dx, dy)
            # As it currently expect that dx and dy aren't in a FunctionGraph
            if not equal_computations([dx], [dy]):
                return False
        return True


class ShapeOptimizer(GlobalOptimizer):
    """Optimizer that serves to add ShapeFeature as an fgraph feature."""

    def add_requirements(self, fgraph):
        fgraph.attach_feature(ShapeFeature())

    def apply(self, fgraph):
        pass


class UnShapeOptimizer(GlobalOptimizer):
    """Optimizer remove ShapeFeature as an fgraph feature."""

    def apply(self, fgraph):
        for feature in fgraph._features:
            if isinstance(feature, ShapeFeature):
                fgraph.remove_feature(feature)


# Register it after merge1 optimization at 0. We don't want to track
# the shape of merged node.
aesara.compile.mode.optdb.register(
    "ShapeOpt", ShapeOptimizer(), 0.1, "fast_run", "fast_compile"
)
# Not enabled by default for now. Some crossentropy opt use the
# shape_feature.  They are at step 2.01. uncanonicalize is at step
# 3. After it goes to 48.5 that move to the gpu. So 10 seem resonable.
aesara.compile.mode.optdb.register("UnShapeOpt", UnShapeOptimizer(), 10)


def local_elemwise_alloc_op(ElemwiseOP, AllocOP, DimShuffleOP):
    def local_elemwise_alloc(fgraph, node):
        """
        elemwise(alloc(x, shp), ..., y.TensorType(BROADCAST CONDITION))
          -> elemwise(x, y.TensorType(BROADCAST CONDITION))

        elemwise(dimshuffle(alloc(x, shp)),... ,y.TensorType(BROADCAST CONDITION))
          -> elemwise(x.dimshuffle(...), y.TensorType(BROADCAST CONDITION))

        BROADCAST CONDITION: the condition is that the one input that are
        not to be optimized to have the same broadcast pattern as the
        output.

        We can change the alloc by a dimshuffle as the elemwise
        already have the shape info.  The dimshuffle will be faster
        to exec.

        """
        if not isinstance(node.op, ElemwiseOP):
            return False

        if len(node.outputs) > 1:
            # Ensure all outputs have the same broadcast pattern
            # This is a supposition that I'm not sure is always true.
            assert all(
                [
                    o.type.broadcastable == node.outputs[0].type.broadcastable
                    for o in node.outputs[1:]
                ]
            )

        # The broadcast pattern of the ouptut must match the broadcast
        # pattern of at least one of the inputs.
        if not any(
            [
                i.type.broadcastable == node.outputs[0].type.broadcastable
                for i in node.inputs
            ]
        ):
            return False

        def dimshuffled_alloc(i):
            return (
                isinstance(i.owner.op, DimShuffleOP)
                and i.owner.inputs[0].owner
                and isinstance(i.owner.inputs[0].owner.op, AllocOP)
            )

        # At least one input must have an owner that is either a AllocOP or a
        # DimShuffleOP with an owner that is a AllocOP -- otherwise there is
        # nothing to optimize.
        if not any(
            [
                i.owner and (isinstance(i.owner.op, AllocOP) or dimshuffled_alloc(i))
                for i in node.inputs
            ]
        ):
            return False

        # Search for input that we can use as a baseline for the dimensions.
        assert_op_idx = -1
        for idx, i in enumerate(node.inputs):
            if i.type.broadcastable == node.outputs[0].type.broadcastable:
                # Prefer an input that is not a AllocOP nor a DimShuffleOP of a
                # AllocOP so that all allocs can be optimized.
                if not (
                    i.owner
                    and (isinstance(i.owner.op, AllocOP) or dimshuffled_alloc(i))
                ):
                    assert_op_idx = idx
                    break

        # It may be the case that only AllocOP and DimShuffleOP of AllocOP exist.
        if assert_op_idx < 0:
            # We want to optimize as many allocs as possible. When
            # there is more than one then do all but one.  number of
            # inputs with alloc or dimshuffle alloc
            l2 = [
                i
                for i in node.inputs
                if (
                    i.owner
                    and (isinstance(i.owner.op, AllocOP) or dimshuffled_alloc(i))
                )
            ]
            # If only 1 alloc or dimshuffle alloc, it is the one we
            # will use for the shape. So no alloc would be removed.
            if len(l2) > 1:
                # l containt inputs with alloc or dimshuffle alloc
                # only.  Its length will always be at least one, as we
                # checked that before
                l = [
                    idx
                    for idx, i in enumerate(node.inputs)
                    if i.broadcastable == node.outputs[0].broadcastable
                ]
                assert_op_idx = l[0]  # The first one is as good as any to use.
            else:
                # Nothing would be optimized!
                return False

        assert_op_in = node.inputs[assert_op_idx]
        cmp_op = assert_op_in
        new_i = []
        same_shape = fgraph.shape_feature.same_shape
        for i in node.inputs:
            # Remove alloc
            if (
                i.owner
                and isinstance(i.owner.op, AllocOP)
                and i.owner.inputs[0].type != i.owner.outputs[0].type
            ):
                # when i.owner.inputs[0].type == i.owner.outputs[0].type we
                # will remove that alloc later
                assert i.type.ndim == cmp_op.ndim
                if config.experimental__local_alloc_elemwise_assert:
                    get_shape = fgraph.shape_feature.get_shape
                    cond = []
                    for idx in range(i.type.ndim):
                        if not i.type.broadcastable[idx] and not same_shape(
                            i, cmp_op, idx, idx
                        ):
                            i_shp = get_shape(i, idx)
                            cmp_shp = get_shape(cmp_op, idx)
                            cond.append(eq(i_shp, cmp_shp))
                    if cond:
                        assert_op_in = assert_op(assert_op_in, *cond)
                new_i.append(i.owner.inputs[0])

            # Remove Alloc in DimShuffle
            elif i.owner and dimshuffled_alloc(i):
                assert i.type.ndim == cmp_op.type.ndim
                if config.experimental__local_alloc_elemwise_assert:
                    assert_cond = [
                        eq(i.shape[idx], cmp_op.shape[idx])
                        for idx in range(i.type.ndim)
                        if not i.type.broadcastable[idx]
                        and not same_shape(i, cmp_op, idx, idx)
                    ]
                    if assert_cond:
                        assert_op_in = assert_op(assert_op_in, *assert_cond)
                alloc_input = i.owner.inputs[0].owner.inputs[0]
                if alloc_input.ndim != i.owner.inputs[0].ndim:
                    # The alloc can add dimension to the value
                    # We add a dimshuffle to add them.
                    # We let later optimization merge the multiple dimshuffle
                    nb_dim_to_add = i.owner.inputs[0].ndim - alloc_input.ndim
                    alloc_input = alloc_input.dimshuffle(
                        ["x"] * nb_dim_to_add + list(range(alloc_input.ndim))
                    )

                # We need to keep the dimshuffle. It could swap axes or
                # add dimensions anywhere.
                r_i = i.owner.op(alloc_input)

                # Copy stack trace from i to new_i
                copy_stack_trace(i, r_i)
                new_i.append(r_i)
            else:
                new_i.append(i)
        new_i[assert_op_idx] = assert_op_in

        ret = node.op(*new_i, return_list=True)

        # Copy over stack trace from previous outputs to new outputs.
        copy_stack_trace(node.outputs, ret)
        return ret

    return local_elemwise_alloc


# TODO, global optimizer that lift the assert to the beginning of the graph.
# TODO, optimize all inputs when possible -- currently when all inputs have
# an alloc all but one is optimized.

local_elemwise_alloc = register_specialize(
    local_optimizer([Elemwise])(local_elemwise_alloc_op(Elemwise, Alloc, DimShuffle)),
    "local_alloc_elemwise",
)


@local_optimizer([Elemwise])
def local_fill_sink(fgraph, node):
    """
    f(fill(a, b), fill(c, d), e) -> fill(c, fill(a, f(b, d, e)))
    f need to be an elemwise that isn't a fill.
    """
    if not hasattr(node, "op") or not isinstance(node.op, Elemwise) or node.op == fill:
        return False
    models = []
    inputs = []
    for input in node.inputs:
        if input.owner and input.owner.op == fill:
            models.append(input.owner.inputs[0])
            inputs.append(input.owner.inputs[1])
        else:
            inputs.append(input)
    if not models:
        return False
    c = node.op(*inputs)
    for model in models:
        if model.type != c.type:
            c = fill(model, c)

    # The newly created node c doesn't has 'clients',
    # so this iteration is took place with node.outputs[0]
    replacements = {node.outputs[0]: c}
    for client, cl_idx in fgraph.clients[node.outputs[0]]:
        if (
            hasattr(client, "op")
            and isinstance(client.op, Elemwise)
            and not client.op == fill
        ):
            client_inputs = client.inputs[:]
            client_inputs[cl_idx] = c
            new_client = client.op(*client_inputs)

            # Add clients to new_client
            fgraph.clients[new_client.owner.outputs[0]] = fgraph.clients[
                client.outputs[0]
            ]
            r = local_fill_sink.transform(fgraph, new_client.owner)
            if not r:
                continue
            replacements.update(r)
    return replacements


register_canonicalize(local_fill_sink)


@register_specialize
@register_stabilize
# @register_canonicalize  # We make full pass after the canonizer phase.
@local_optimizer([fill])
def local_fill_to_alloc(fgraph, node):
    """fill(s,v) -> alloc(v, shape(s))

    This is an important optimization because with the shape_to_shape_i
    optimization, the dependency on 's' is often removed.

    """
    if node.op == fill:
        r, v = node.inputs
        if v.type == node.outputs[0].type:
            # this is a useless fill, erase it.
            rval = [v]
        elif v.type.broadcastable == node.outputs[0].type.broadcastable:
            # this is a cast
            rval = [cast(v, node.outputs[0].type.dtype)]
        elif r.type.broadcastable == node.outputs[0].type.broadcastable:
            # we are broadcasting v somehow, but not r
            o = broadcast_like(v, r, fgraph, dtype=v.dtype)
            copy_stack_trace(node.outputs[0], o)
            rval = [o]
        else:
            # we are broadcasting both v and r,
            # the output shape must be computed
            #
            # TODO: implement this case (including a test!)
            #
            #  I think the strategy should be to extend the shorter
            #  shape vector with 1s (how?) and then take the
            #  elementwise max of the two.  - how to flag an error of
            #  shape mismatch where broadcasting should be illegal?
            return
            # TODO: cut out un-necessary dimshuffles of v

        assert rval[0].type == node.outputs[0].type, (
            "rval",
            rval[0].type,
            "orig",
            node.outputs[0].type,
            "node",
            node,
        )  # aesara.printing.debugprint(node.outputs[0], file='str'))
        return rval


# Register this after stabilize at 1.5 to make sure stabilize don't
# get affected by less canonicalized graph due to alloc.
compile.optdb.register(
    "local_fill_to_alloc", in2out(local_fill_to_alloc), 1.51, "fast_run"
)
# Needed to clean some extra alloc added by local_fill_to_alloc
compile.optdb.register(
    "local_elemwise_alloc", in2out(local_elemwise_alloc), 1.52, "fast_run"
)


@register_canonicalize("fast_compile")
@register_useless
@local_optimizer([fill])
def local_useless_fill(fgraph, node):
    """fill(s,v) -> v

    This optimization is only needed in FAST_COMPILE to make the code
    more readable. Normally, it is done by the local_fill_to_alloc
    opt.

    """
    if node.op == fill:
        r, v = node.inputs
        if v.type == node.outputs[0].type:
            # this is a useless fill, erase it.
            # also, we don't need to copy over any stack traces here
            return [v]


@register_specialize
@register_stabilize
@register_canonicalize
@register_useless
@local_optimizer([alloc])
def local_useless_alloc(fgraph, node):
    """
    If the input type is the same as the output type (dtype and broadcast)
    there is no change in the shape of the input. So this is just a simple copy
    of the input. This is not needed.

    """
    op = node.op
    if not isinstance(op, Alloc):
        return False

    input = node.inputs[0]
    output = node.outputs[0]

    # Check if dtype and broadcast remain the same.
    if input.type == output.type:
        # We don't need to copy over any stack traces here
        return [input]


@register_specialize
@register_stabilize
@register_canonicalize
@local_optimizer([alloc])
def local_canonicalize_alloc(fgraph, node):
    """If the input type is the same as the output type (dtype and broadcast)
    there is no change in the shape of the input. So this is just a simple copy
    of the input. This is not needed. (as local_useless_alloc)

    Also, it will canonicalize alloc by creating Dimshuffle after the
    alloc to introduce the dimensions of constant size 1.

    See https://github.com/Theano/Theano/issues/4072 to know why this
    is needed.

    """
    op = node.op
    if not isinstance(op, Alloc):
        return False

    input = node.inputs[0]
    output = node.outputs[0]

    # Check if dtype and broadcast remain the same.
    if input.type == output.type:
        # We don't need to copy over any stack traces here
        return [input]

    # Allow local_merge_alloc to do its work first
    clients = fgraph.clients[output]
    for client, i in clients:
        if client != "output" and isinstance(client.op, Alloc):
            return

    # Check if alloc adds a broadcastable dimension with shape 1.

    output_shape = node.inputs[1:]
    num_dims_with_size_1_added_to_left = 0
    for i in range(len(output_shape) - input.ndim):
        if extract_constant(output_shape[i], only_process_constants=True) == 1:
            num_dims_with_size_1_added_to_left += 1
        else:
            break
    new_output_shape = output_shape[num_dims_with_size_1_added_to_left:]
    if num_dims_with_size_1_added_to_left > 0 and len(new_output_shape) >= input.ndim:
        if (
            output.broadcastable[num_dims_with_size_1_added_to_left:]
            == input.broadcastable
        ):
            inner = input
        else:
            inner = op(*([input] + new_output_shape))
        dimshuffle_new_order = ["x"] * num_dims_with_size_1_added_to_left + list(
            range(len(new_output_shape))
        )
        return [DimShuffle(inner.type.broadcastable, dimshuffle_new_order)(inner)]


# Don't register by default.
@local_optimizer([AllocEmpty])
def local_alloc_empty_to_zeros(fgraph, node):
    """This convert AllocEmpty to Alloc of 0.

    This help investigate NaN with NanGuardMode.  Not registered by
    default. To activate it, use the Aesara flag
    optimizer_including=alloc_empty_to_zeros. This also enable
    the GPU version of this optimizations.

    """
    if isinstance(node.op, AllocEmpty):
        return [zeros(node.inputs, dtype=node.outputs[0].dtype)]


compile.optdb.register(
    "local_alloc_empty_to_zeros",
    in2out(local_alloc_empty_to_zeros),
    # After move to gpu and merge2, before inplace.
    49.3,
    "alloc_empty_to_zeros",
)


@register_specialize
@register_canonicalize
@local_optimizer([Shape])
def local_shape_to_shape_i(fgraph, node):
    if node.op == shape:
        # This optimization needs ShapeOpt and fgraph.shape_feature
        if not hasattr(fgraph, "shape_feature"):
            return
        shape_feature = fgraph.shape_feature
        ret = shape_feature.make_vector_shape(node.inputs[0])

        # We need to copy over stack trace from input to output
        copy_stack_trace(node.outputs[0], ret)
        return [ret]


# TODO: Not sure what type of node we are expecting here
@register_specialize
@register_canonicalize
@local_optimizer(None)
def local_track_shape_i(fgraph, node):
    try:
        shape_feature = fgraph.shape_feature
    except AttributeError:
        return
    if node in shape_feature.scheduled:
        # Don't unschedule node as it could be reinserted in the
        # fgraph as we don't change it in the shapefeature internal
        # structure.
        assert isinstance(node.op, Shape_i)
        replacement = shape_feature.scheduled[node]
        return [shape_feature.shape_of[replacement][node.op.i]]


@register_specialize
@register_canonicalize
@local_optimizer([Subtensor])
def local_subtensor_inc_subtensor(fgraph, node):
    """
    Subtensor(SetSubtensor(x, y, idx), idx) -> y

    """
    if isinstance(node.op, Subtensor):
        x = node.inputs[0]
        if not x.owner or not isinstance(x.owner.op, IncSubtensor):
            return
        if not x.owner.op.set_instead_of_inc:
            return

        if x.owner.inputs[2:] == node.inputs[1:] and tuple(
            x.owner.op.idx_list
        ) == tuple(node.op.idx_list):
            out = node.outputs[0]
            y = x.owner.inputs[1]
            # If the dtypes differ, cast y into x.dtype
            if x.dtype != y.dtype:
                y = y.astype(x.dtype)
            if out.type == y.type:
                # if x[idx] and y have the same type, directly return y
                return [y]
            else:
                # The difference is related to broadcasting pattern
                assert out.broadcastable != y.broadcastable
                # We have to alloc y to the shape of x[idx]
                x_subtensor = node.op(x.owner.inputs[0], *x.owner.inputs[2:])
                return [alloc(y, *x_subtensor.shape)]
        else:
            return


@register_specialize
@register_canonicalize
@local_optimizer([Subtensor])
def local_subtensor_remove_broadcastable_index(fgraph, node):
    """
    Remove broadcastable dimension with index 0 or -1
    a[:,:,:,0] -> a.dimshuffle(0,1,2), when
        a.broadcastable = (False, False, False, True)
    a[0,:,-1,:] -> a.dimshuffle(1,3), when
        a.broadcastable = (True, False, True, False)

    """
    if isinstance(node.op, Subtensor):
        idx = node.op.idx_list
    else:
        return

    remove_dim = []
    node_inputs_idx = 1
    for dim, elem in enumerate(idx):
        if isinstance(elem, (aes.Scalar)):
            # The idx is a Scalar, ie a Type. This means the actual index
            # is contained in node.inputs[1]
            dim_index = node.inputs[node_inputs_idx]
            if type(dim_index) == aes.ScalarConstant:
                dim_index = dim_index.value
            if dim_index in [0, -1] and node.inputs[0].broadcastable[dim]:
                remove_dim.append(dim)
                node_inputs_idx += 1
            else:
                return
        elif isinstance(elem, slice):
            if elem != slice(None):
                return
        elif isinstance(elem, (int, np.integer)):
            if elem in [0, -1] and node.inputs[0].broadcastable[dim]:
                remove_dim.append(dim)
        else:
            raise TypeError("case not expected")

    if len(remove_dim) == 0:
        return
    else:
        all_dim = range(node.inputs[0].ndim)
        remain_dim = [x for x in all_dim if x not in remove_dim]
        return [node.inputs[0].dimshuffle(tuple(remain_dim))]


@register_specialize
@register_canonicalize("fast_compile_gpu")
@register_useless
@local_optimizer([Subtensor, AdvancedSubtensor1])
def local_subtensor_make_vector(fgraph, node):
    """
    Replace all subtensor(make_vector) like:
    [a,b,c][0] -> a
    [a,b,c][0:2] -> [a,b]

    Replace all AdvancedSubtensor1(make_vector) like:
    [a,b,c][[0,2]] -> [a,c]

    We can do this for constant indexes.

    """
    x = node.inputs[0]
    if not x.owner or x.owner.op != make_vector:
        return

    if isinstance(node.op, Subtensor):
        # This optimization needs ShapeOpt and fgraph.shape_feature
        try:
            (idx,) = node.op.idx_list
        except Exception:
            # 'how can you have multiple indexes into a shape?'
            raise

        if isinstance(idx, (aes.Scalar, TensorType)):
            # The idx is a Scalar, ie a Type. This means the actual index
            # is contained in node.inputs[1]
            old_idx, idx = idx, node.inputs[1]
            assert idx.type == old_idx
    elif isinstance(node.op, AdvancedSubtensor1):
        idx = node.inputs[1]
    else:
        return

    if isinstance(idx, (int, np.integer)):
        # We don't need to copy over any stack traces here
        return [x.owner.inputs[idx]]
    elif isinstance(idx, Variable):
        if idx.ndim == 0:
            # if it is a constant we can do something with it
            try:
                v = get_scalar_constant_value(idx, only_process_constants=True)
                if isinstance(v, np.integer):
                    # Python 2.4 wants to index only with Python integers
                    v = int(v)
                # We don't need to copy over any stack traces here
                try:
                    ret = [x.owner.inputs[v]]
                except IndexError:
                    raise NotScalarConstantError("Bad user graph!")
                return ret
            except NotScalarConstantError:
                pass
        elif idx.ndim == 1 and isinstance(idx, Constant):
            values = list(map(int, list(idx.value)))
            ret = make_vector(*[x.owner.inputs[v] for v in values])

            # Copy over stack trace from previous output to new output
            copy_stack_trace(node.outputs[0], ret)
            ret = patternbroadcast(ret, node.outputs[0].broadcastable)
            return [ret]
        else:
            raise TypeError("case not expected")
    elif isinstance(idx, slice):
        # it is a slice of ints and/or Variables
        # check subtensor to see if it can contain constant variables, and if
        # it can, then try to unpack them.
        try:
            const_slice = node.op.get_constant_idx(node.inputs, allow_partial=False)[0]
            ret = make_vector(*x.owner.inputs[const_slice])
            # Copy over stack trace from previous outputs to new output
            copy_stack_trace(node.outputs, ret)
            ret = patternbroadcast(ret, node.outputs[0].broadcastable)
            return [ret]
        except NotScalarConstantError:
            pass
    else:
        raise TypeError("case not expected")


# TODO: the other optimization for and, or, xor, le and ge see ticket #496.


@register_useless
@register_canonicalize("fast_compile")
@register_specialize
@local_optimizer([Elemwise])
def local_useless_elemwise(fgraph, node):
    """
    eq(x, x) -> 1
    neq(x, x) -> 0
    mul(x) -> x
    add(x) -> x
    identity(x) -> x
    and(x, 1) -> x  (if x.dtype == 'bool')
    and(x, 0) -> zeros_like(x)
    or(x, 0) -> x
    or(x, 1) -> ones_like(x)  (if x.dtype == 'bool')
    xor(x, x) -> zeros_like(x)

    """
    if isinstance(node.op, Elemwise):
        # We call zeros_like and one_like with opt=True to generate a
        # cleaner graph.
        dtype = node.outputs[0].dtype

        if node.op.scalar_op == aes.eq and len(node.inputs) == 2:
            if node.inputs[0] == node.inputs[1]:
                # it is the same var in the graph. That will always be true
                ret = ones_like(node.inputs[0], dtype=dtype, opt=True)

                # Copy stack trace from input to constant output
                copy_stack_trace(node.outputs[0], ret)
                return [ret]
        elif node.op.scalar_op == aes.neq and len(node.inputs) == 2:
            if node.inputs[0] == node.inputs[1]:
                # it is the same var in the graph. That will always be false
                ret = zeros_like(node.inputs[0], dtype=dtype, opt=True)

                # Copy stack trace from input to constant output
                copy_stack_trace(node.outputs[0], ret)
                return [ret]

        elif node.op.scalar_op == aes.mul and len(node.inputs) == 1:
            # No need to copy over any stack trace
            return [node.inputs[0]]

        elif node.op.scalar_op == aes.add and len(node.inputs) == 1:
            # No need to copy over any stack trace
            return [node.inputs[0]]
        elif node.op.scalar_op == aes.identity and len(node.inputs) == 1:
            return [node.inputs[0]]

        elif isinstance(node.op.scalar_op, aes.AND) and len(node.inputs) == 2:

            if isinstance(node.inputs[0], TensorConstant):
                const_val = extract_constant(
                    node.inputs[0], only_process_constants=True
                )
                if not isinstance(const_val, Variable):
                    if const_val == 0:
                        return [zeros_like(node.inputs[1], dtype=dtype, opt=True)]
                    elif node.outputs[0].dtype == "bool":
                        # If the output is not Boolean, it is the bitwise AND,
                        # and this optimization would be wrong
                        return [node.inputs[1].astype(node.outputs[0].dtype)]

            if isinstance(node.inputs[1], TensorConstant):
                const_val = extract_constant(
                    node.inputs[1], only_process_constants=True
                )
                if not isinstance(const_val, Variable):
                    if const_val == 0:
                        return [zeros_like(node.inputs[0], dtype=dtype, opt=True)]
                    elif node.outputs[0].dtype == "bool":
                        # If the output is not Boolean, it is the bitwise AND,
                        # and this optimization would be wrong
                        return [node.inputs[0].astype(node.outputs[0].dtype)]

        elif isinstance(node.op.scalar_op, aes.OR) and len(node.inputs) == 2:

            if isinstance(node.inputs[0], TensorConstant):
                const_val = extract_constant(
                    node.inputs[0], only_process_constants=True
                )
                if not isinstance(const_val, Variable):
                    if const_val == 0:
                        return [node.inputs[1].astype(node.outputs[0].dtype)]
                    elif node.outputs[0].dtype == "bool":
                        # If the output is not Boolean, it is the bitwise OR,
                        # and this optimization would be wrong
                        return [ones_like(node.inputs[1], dtype=dtype, opt=True)]

            if isinstance(node.inputs[1], TensorConstant):
                const_val = extract_constant(
                    node.inputs[1], only_process_constants=True
                )
                if not isinstance(const_val, Variable):
                    if const_val == 0:
                        return [node.inputs[0].astype(node.outputs[0].dtype)]
                    elif node.outputs[0].dtype == "bool":
                        # If the output is not Boolean, it is the bitwise OR,
                        # and this optimization would be wrong
                        return [ones_like(node.inputs[0], dtype=dtype, opt=True)]

        elif isinstance(node.op.scalar_op, aes.XOR) and len(node.inputs) == 2:
            if node.inputs[0] is node.inputs[1]:
                return [zeros_like(node.inputs[0], dtype=dtype, opt=True)]


@register_specialize
@local_optimizer([Elemwise])
def local_alloc_unary(fgraph, node):
    """unary(alloc(x, shp)) -> alloc(unary(x), shp)"""
    if isinstance(node.op, Elemwise) and len(node.inputs) == 1:
        a = node.inputs[0]
        if a.owner and isinstance(a.owner.op, Alloc):
            x = a.owner.inputs[0]
            shp = a.owner.inputs[1:]
            v = node.op(x)
            # aet.alloc does not preserve the stacktrace of v,
            # so we need to copy it over from x.
            copy_stack_trace(node.outputs[0], v)
            ret = alloc(cast(v, node.outputs[0].dtype), *shp)

            # aet.cast does not preserve the stacktrace of x,
            # so we need to copy it over to the output.
            copy_stack_trace([node.outputs[0], a], ret)
            return [ret]


@register_canonicalize
@register_specialize
@local_optimizer([Elemwise])
def local_cast_cast(fgraph, node):
    """cast(cast(x, dtype1), dtype2)

    when those contrain:
    dtype1 == dtype2
    OR the base dtype is the same (int, uint, float, complex)
          and the first cast cause an upcast.

    """
    if not isinstance(node.op, Elemwise) or not isinstance(node.op.scalar_op, aes.Cast):
        return
    x = node.inputs[0]
    if (
        not x.owner
        or not isinstance(x.owner.op, Elemwise)
        or not isinstance(x.owner.op.scalar_op, aes.Cast)
    ):
        return

    type1 = x.owner.op.scalar_op.o_type
    type2 = node.op.scalar_op.o_type
    base = x.owner.inputs[0]

    if type1 == type2:
        # We don't need to copy over any stack traces here
        return [x]

    if is_an_upcast(base.dtype, type1.dtype):
        # Checking for further redundancy. Eg: int8 -> int32 -> int8
        if type2.dtype == base.dtype:
            return x.owner.inputs
        else:
            # Apply the second cast only
            v = node.op(base)
            # Copy stack trace from the output of the original cast
            copy_stack_trace(node.outputs[0], v)
            return [v]


def is_an_upcast(type1, type2):
    """Given two data types (as strings), check if converting to
    type2 from type1 constitutes an upcast.
    Differs from aesara.scalar.upcast

    """
    category = {
        # The first number in the pair is the dtype (bool, uint, int, float,
        # complex). Conversion from higher to lower is never an upcast.
        # The second number roughly indicates the precision. Again, conversion
        # from higher to lower is never an upcast.
        "bool": (0, 0),
        "uint8": (1, 1),
        "uint16": (1, 2),
        "uint32": (1, 3),
        "uint64": (1, 4),
        "int8": (2, 1),
        "int16": (2, 2),
        "int32": (2, 3),
        "int64": (2, 4),
        "float16": (3, 1.5),
        "float32": (3, 2.5),
        "float64": (3, 3.5),
        "complex64": (4, 3),
        "complex128": (4, 4),
    }

    cat1 = category[type1]
    cat2 = category[type2]

    if cat2[0] >= cat1[0] and cat2[1] > cat1[1]:
        return True
    else:
        return False


@register_specialize
@local_optimizer([Assert])
def local_remove_useless_assert(fgraph, node):
    if isinstance(node.op, Assert):
        cond = []
        for c in node.inputs[1:]:
            try:
                const = get_scalar_constant_value(c)

                if 0 != const.ndim or const == 0:
                    # Should we raise an error here? How to be sure it
                    # is not catched?
                    cond.append(c)
            except NotScalarConstantError:
                cond.append(c)

        if len(cond) == 0:
            # We don't need to copy over any stack traces here
            return [node.inputs[0]]
        if len(cond) != len(node.inputs) - 1:
            ret = assert_op(node.inputs[0], *cond)

            # We copy over stack trace from the output of the original assert
            copy_stack_trace(node.outputs[0], ret)
            return [ret]


@local_optimizer([Assert])
def local_remove_all_assert(fgraph, node):
    """An optimization disabled by default that removes all asserts from
    the graph.

    Notes
    -----
    See the :ref:`unsafe` section to know how to enable it.

    """
    if not isinstance(node.op, Assert):
        return

    # We don't need to copy over any stack traces here
    return [node.inputs[0]]


# Disabled by default
compile.optdb["canonicalize"].register(
    "local_remove_all_assert",
    local_remove_all_assert,
    "unsafe",
    use_db_name_as_tag=False,
)
compile.optdb["stabilize"].register(
    "local_remove_all_assert",
    local_remove_all_assert,
    "unsafe",
    use_db_name_as_tag=False,
)
compile.optdb["specialize"].register(
    "local_remove_all_assert",
    local_remove_all_assert,
    "unsafe",
    use_db_name_as_tag=False,
)
compile.optdb["useless"].register(
    "local_remove_all_assert",
    local_remove_all_assert,
    "unsafe",
    use_db_name_as_tag=False,
)


@register_canonicalize
@local_optimizer([Elemwise])
def local_upcast_elemwise_constant_inputs(fgraph, node):
    """This explicitly upcasts constant inputs to elemwise Ops, when
    those Ops do implicit upcasting anyway.

    Rationale: it helps merge things like (1-x) and (1.0 - x).

    """
    if len(node.outputs) > 1:
        return
    try:
        shape_i = fgraph.shape_feature.shape_i
    except AttributeError:
        shape_i = None
    if isinstance(node.op, Elemwise):
        scalar_op = node.op.scalar_op
        # print "aa", scalar_op.output_types_preference
        if getattr(scalar_op, "output_types_preference", None) in (
            aes.upgrade_to_float,
            aes.upcast_out,
        ):
            # this is the kind of op that we can screw with the input
            # dtypes by upcasting explicitly
            output_dtype = node.outputs[0].type.dtype
            new_inputs = []
            for i in node.inputs:
                if i.type.dtype == output_dtype:
                    new_inputs.append(i)
                else:
                    try:
                        # works only for scalars
                        cval_i = get_scalar_constant_value(
                            i, only_process_constants=True
                        )
                        if all(i.broadcastable):
                            new_inputs.append(
                                shape_padleft(cast(cval_i, output_dtype), i.ndim)
                            )
                        else:
                            if shape_i is None:
                                return
                            new_inputs.append(
                                alloc(
                                    cast(cval_i, output_dtype),
                                    *[shape_i(d)(i) for d in range(i.ndim)],
                                )
                            )
                            # print >> sys.stderr, "AAA",
                            # *[Shape_i(d)(i) for d in range(i.ndim)]
                    except NotScalarConstantError:
                        # for the case of a non-scalar
                        if isinstance(i, TensorConstant):
                            new_inputs.append(cast(i, output_dtype))
                        else:
                            new_inputs.append(i)

            if new_inputs != node.inputs:
                rval = [node.op(*new_inputs)]
                if rval[0].type != node.outputs[0].type:
                    # This can happen for example when floatX=float32
                    # and we do the true division between and int64
                    # and a constant that will get typed as int8.

                    # As this is just to allow merging more case, if
                    # the upcast don't work, we can just skip it.
                    return

                # Copy over output stacktrace from before upcasting
                copy_stack_trace(node.outputs[0], rval)
                return rval


##################
# Subtensor opts #
##################


@register_useless
@register_canonicalize
@register_specialize
@local_optimizer([IncSubtensor])
def local_useless_inc_subtensor(fgraph, node):
    """
    Remove IncSubtensor, when we overwrite the full inputs with the
    new value.

    """
    if not isinstance(node.op, IncSubtensor):
        return
    if node.op.set_instead_of_inc is False:
        # This is an IncSubtensor, so the init value must be zeros
        try:
            c = get_scalar_constant_value(node.inputs[0], only_process_constants=True)
            if c != 0:
                return
        except NotScalarConstantError:
            return
    if (
        node.inputs[0].ndim != node.inputs[1].ndim
        or node.inputs[0].broadcastable != node.inputs[1].broadcastable
    ):
        # FB: I didn't check if this case can happen, but this opt
        # don't support it.
        return
    # We have a SetSubtensor or an IncSubtensor on zeros
    # If is this IncSubtensor useful?

    # Check that we keep all the original data.
    # Put the constant inputs in the slice.
    idx_cst = get_idx_list(node.inputs[1:], node.op.idx_list)
    if all(
        isinstance(e, slice)
        and e.start is None
        and e.stop is None
        and (
            e.step is None
            or extract_constant(e.step, only_process_constants=True) == -1
        )
        for e in idx_cst
    ):
        # IncSubtensor broadcast node.inputs[1] on node.inputs[0]
        # based on run time shapes, so we must check they are the same.
        if not hasattr(fgraph, "shape_feature"):
            return
        if not fgraph.shape_feature.same_shape(node.inputs[0], node.inputs[1]):
            return
        # There is no reverse, so we don't need a replacement.
        if all(e.step is None for e in node.op.idx_list):
            # They are the same shape, so we can remore this IncSubtensor
            return [node.inputs[1]]
        ret = Subtensor(node.op.idx_list)(*node.inputs[1:])
        # Copy over previous output stacktrace
        copy_stack_trace(node.outputs, ret)
        return [ret]


@register_canonicalize
@local_optimizer([AdvancedIncSubtensor1])
def local_set_to_inc_subtensor(fgraph, node):
    """
    AdvancedIncSubtensor1(x, x[ilist]+other, ilist, set_instead_of_inc=True) ->
    AdvancedIncSubtensor1(x, other, ilist, set_instead_of_inc=False)

    """
    if (
        isinstance(node.op, AdvancedIncSubtensor1)
        and node.op.set_instead_of_inc
        and node.inputs[1].owner
        and isinstance(node.inputs[1].owner.op, Elemwise)
        and isinstance(node.inputs[1].owner.op.scalar_op, aes.Add)
    ):
        addn = node.inputs[1].owner
        subn = None
        other = None

        if addn.inputs[0].owner and isinstance(
            addn.inputs[0].owner.op, AdvancedSubtensor1
        ):
            subn = addn.inputs[0].owner
            other = addn.inputs[1]
        elif addn.inputs[1].owner and isinstance(
            addn.inputs[1].owner.op, AdvancedSubtensor1
        ):
            subn = addn.inputs[1].owner
            other = addn.inputs[0]
        else:
            return
        if subn.inputs[1] != node.inputs[2] or subn.inputs[0] != node.inputs[0]:
            return
        ret = advanced_inc_subtensor1(node.inputs[0], other, node.inputs[2])
        # Copy over previous output stacktrace
        # Julian: I'm not sure about this at all...
        copy_stack_trace(node.outputs, ret)
        return [ret]


@register_useless
@register_canonicalize
@register_specialize
@local_optimizer([Subtensor])
def local_useless_slice(fgraph, node):
    """
    Remove Subtensor of the form X[0, :] -> X[0]
    """
    if isinstance(node.op, Subtensor):
        slices = get_idx_list(node.inputs, node.op.idx_list)
        last_slice = len(slices)
        for s in slices[::-1]:
            # check if slice and then check slice indices
            if (
                isinstance(s, slice)
                and s.start is None
                and s.stop is None
                and (
                    s.step is None
                    or extract_constant(s.step, only_process_constants=True) == 1
                )
            ):
                last_slice -= 1
            else:
                break
        # check if we removed something
        if last_slice < len(slices):
            subtens = Subtensor(slices[:last_slice])
            sl_ins = Subtensor.collapse(
                slices[:last_slice], lambda x: isinstance(x, Variable)
            )
            out = subtens(node.inputs[0], *sl_ins)
            # Copy over previous output stacktrace
            copy_stack_trace(node.outputs, out)
            return [out]


@register_canonicalize
@register_specialize
@local_optimizer([Subtensor, AdvancedSubtensor1])
def local_useless_subtensor(fgraph, node):
    """
    Remove Subtensor/AdvancedSubtensor1 if it takes the full input. In the
    AdvancedSubtensor1 case, the full input is taken when the indices are
    equivalent to `arange(0, input.shape[0], 1)` using either an explicit
    list/vector or the ARange op.

    """
    # This optimization needs ShapeOpt and fgraph.shape_feature
    if not hasattr(fgraph, "shape_feature"):
        return

    shape_of = fgraph.shape_feature.shape_of

    if isinstance(node.op, Subtensor):
        cdata = node.op.get_constant_idx(
            node.inputs, allow_partial=True, only_process_constants=True
        )
        for pos, idx in enumerate(cdata):
            if not isinstance(idx, slice):
                # If idx is not a slice, this means we remove this dimension
                # from the output, so the subtensor is not useless
                return False
            if idx.start is not None and idx.start != 0:
                # If the start of the slice is different from 0, or is a
                # variable, then we assume the subtensor is not useless
                return False
            if idx.step is not None and idx.step != 1:
                # If we are going backwards, or skipping elements, then this
                # is not a useless subtensor
                return False

        for pos, idx in enumerate(cdata):

            length_pos = shape_of[node.inputs[0]][pos]

            if isinstance(idx.stop, (int, np.integer)):
                length_pos_data = sys.maxsize
                try:
                    length_pos_data = get_scalar_constant_value(
                        length_pos, only_process_constants=True
                    )
                except NotScalarConstantError:
                    pass

                if idx.stop < length_pos_data:
                    return False
            elif isinstance(idx.stop, Variable):
                length_pos_shape_i = idx.stop
                # length_pos is a tensor variable, but length_pos_shape_i
                # is a scalar variable. We try to see if they represent
                # the same underlying variable.
                if length_pos_shape_i.owner and isinstance(
                    length_pos_shape_i.owner.op, ScalarFromTensor
                ):
                    length_pos_shape_i = length_pos_shape_i.owner.inputs[0]
                elif length_pos.owner and isinstance(
                    length_pos.owner.op, TensorFromScalar
                ):
                    length_pos = length_pos.owner.inputs[0]
                else:
                    # We did not find underlying variables of the same type
                    return False

                # The type can be different: int32 vs int64. length_pos
                # should always be int64 as that is what the shape
                # tracker keep. Subtensor accept any scalar int{8,16,32,64}
                # as index type.
                assert str(length_pos.type.dtype) == "int64"
                assert str(length_pos_shape_i.type.dtype) in [
                    "int8",
                    "int16",
                    "int32",
                    "int64",
                ]

                # length_pos_shape_i cannot be None
                if length_pos_shape_i != length_pos:
                    return False
            elif idx.stop is None:
                pass
            else:
                return False
    elif isinstance(node.op, AdvancedSubtensor1):
        # get length of the indexed tensor along the first axis
        try:
            length = get_scalar_constant_value(
                shape_of[node.inputs[0]][0], only_process_constants=True
            )
        except NotScalarConstantError:
            return False

        # get index (which must be a vector by definition)
        idx = node.inputs[1]

        # `idx` must be equivalent to [0,1,...,shape[0] - 1] to qualify for
        # this optimization
        if isinstance(idx, Constant):
            idx = idx.value
            if len(idx) != length:
                return False
            if np.any(idx != np.arange(length)):
                return False
        elif idx.owner is not None and isinstance(idx.owner.op, ARange):
            try:
                start, stop, step = map(
                    lambda x: get_scalar_constant_value(x, only_process_constants=True),
                    idx.owner.inputs,
                )
            except NotScalarConstantError:
                return False

            if start != 0:
                return False
            if stop != length:
                return False
            if step != 1:
                return False
        else:
            return False
    else:
        return False

    # We don't need to copy over any stacktrace here,
    # because previous stacktrace should suffice.
    return [node.inputs[0]]


# fast_compile to allow opt subtensor(cast{float32}(make_vector))
@register_canonicalize("fast_compile")
@local_optimizer([Subtensor])
def local_subtensor_lift(fgraph, node):
    """
    unary(x)[idx] -> unary(x[idx])#any broadcast pattern.

    Handles the following unary ops:
    elemwise(x,...)[idx] -> elemwise(x[idx],...)
      when x,... are broadcasted scalar or not broadcasted at all
    rebroadcast(x)[idx] => rebroadcast(x[idx])

    """
    if isinstance(node.op, Subtensor):
        u = node.inputs[0]
        if not u.owner or len(fgraph.clients[u]) > 1:
            return False

        if isinstance(u.owner.op, Elemwise) and len(u.owner.inputs) == 1:
            idx = node.inputs[1:]
            x_idx = node.op(u.owner.inputs[0], *idx)
            # Copy over previous output stacktrace
            copy_stack_trace(node.outputs, x_idx)
            ret = u.owner.op(x_idx)
            # Copy over previous output stacktrace
            # and stacktrace from previous unary operation
            copy_stack_trace([node.outputs[0], node.inputs[0]], ret)
            return [ret]

        if isinstance(u.owner.op, Elemwise):
            new_inputs = []
            if all([sum(i.type.broadcastable) == 0 for i in u.owner.inputs]):
                # There is no broadcastable in the inputs
                idx = node.inputs[1:]
                new_inputs = [node.op(i, *idx) for i in u.owner.inputs]
                # Copy over previous output stacktrace
                copy_stack_trace(node.outputs[0], new_inputs)

                ret = u.owner.op(*new_inputs)
                # Copy over previous output stacktrace
                # and stacktrace from previous unary operation
                copy_stack_trace([node.outputs[0], node.inputs[0]], ret)
                return [ret]
            elif all(
                [sum(i.type.broadcastable) in [i.ndim, 0] for i in u.owner.inputs]
            ):
                # There is no broadcastable in the inputs or it is scalar
                idx = node.inputs[1:]
                new_inputs = []
                for i in u.owner.inputs:
                    if sum(i.type.broadcastable) == 0:
                        new_inputs.append(node.op(i, *idx))
                    else:
                        # If the subtensor remove some dims, we must
                        # lower the number of dimensions of this scalar.
                        if node.outputs[0].ndim == i.ndim:
                            new_inputs.append(i)
                        else:
                            new_inputs.append(
                                i.dimshuffle(["x"] * node.outputs[0].ndim)
                            )

                # Copy over previous output stacktrace
                copy_stack_trace(node.outputs[0], new_inputs)

                ret = u.owner.op(*new_inputs)
                # Copy over previous output stacktrace
                # and stacktrace from previous unary operation
                copy_stack_trace([node.outputs[0], node.inputs[0]], ret)
                return [ret]

        if isinstance(u.owner.op, Rebroadcast):
            # make sure that Rebroadcast has only 1 input
            assert len(u.owner.inputs) == 1

            # Subtensor might reduce dim., adapt broadcast pattern accordingly
            new_axis = []

            # loop through indices being subtensor-ed
            # i indexes broadcastable pattern before subtensor
            # j indexes broadcastable pattern after subtensor
            j = 0
            for (i, x) in enumerate(node.op.idx_list):
                # if its not a slice, it will reduce the dimension, should
                # not appear in the broascastable dimensions
                if isinstance(x, slice):
                    new_axis += [(j, u.broadcastable[i])]
                    j += 1
            # now keep the broadcastable pattern of all
            # items not appearing in subtensor list
            for i in range(len(node.op.idx_list), len(u.broadcastable)):
                new_axis += [(j, u.broadcastable[i])]
                j += 1

            subt_x = node.op(u.owner.inputs[0], *node.inputs[1:])
            # Copy over previous output stacktrace
            copy_stack_trace(node.outputs[0], subt_x)

            rbcast_subt_x = Rebroadcast(*new_axis)(subt_x)
            # Copy over previous output stacktrace
            # and stacktrace from previous unary operation
            copy_stack_trace([node.outputs[0], node.inputs[0]], rbcast_subt_x)

            return [rbcast_subt_x]


def merge_two_slices(fgraph, slice1, len1, slice2, len2):
    """
     This function merges two slices into a single slice. The code works on
     the assumption that:

     a) slice1 is actually a slice and not an index, while slice2
        can be just an index.

     b) the two slices **have been applied consecutively** on the same
        tensor

    The output slice is **not** in canonical form, but actually just a slice
    that can be applied to a tensor to produce the same output as applying
    the two consecutive slices.
    ``len1`` is the length of the tensor **before** applying the first slice,
    while ``len2`` is the length **after** applying the first slice.
    """

    if not isinstance(slice1, slice):
        raise ValueError(
            (
                "First provided slice should actually be of type"
                "slice and not an index !"
            ),
            slice1,
        )
    sl1, reverse1 = get_canonical_form_slice(slice1, len1)
    sl2, reverse2 = get_canonical_form_slice(slice2, len2)

    if not isinstance(sl2, slice):
        if reverse1 is None:
            # The first slice is not in reverse, which makes things a lot
            # more clear.
            # In this case we need to take care only of the special cases:
            # len2 <=0    -> throw index error regardless of sl2
            # sl2 > len2  -> throw index error
            # sl2 < -len2 -> throw index error
            # To get a index error we simply use len1+1 to indicate we are
            # out of bounds, because passing this index through the formula
            # of getting the mixed slice is not guaranteed to result in an
            # index error. The **issue though** if that the error will
            # complain about accessing element len1+1 which is probably not
            # too intuitive for the user
            val = sl1.start + sl2 * sl1.step
            val = switch(le(len2, 0), len1 + 1, val)
            val = switch(ge(sl2, len2), len1 + 1, val)
            val = switch(lt(sl2, 0), -len1 - 1, val)
            if sl1.step:
                val = switch(eq(sl1.step, 0), len1 + 1, val)
            return val
        else:
            # We are in the more complex case when we do not actually know
            # if the first slice was in reverse or not.
            # in case it was not in reverse:
            p_val = sl1.start + sl2 * sl1.step
            # case it was in reverse we need to realize that we do not want
            # the k-th element from sl.start but the k-th element from
            # sl.stop backwards
            n_val = sl1.stop - 1 - sl2 * sl1.step
            if config.warn__subtensor_merge_bug:
                warnings.warning(
                    "Your current code is fine, but Aesara versions "
                    "prior to 0.5rc2 might have given an incorrect result. "
                    "To disable this warning, set the Aesara flag "
                    "warn__subtensor_merge_bug to False."
                )
            # we need to pick either n_val or p_val and then follow same
            # steps as above for covering the index error cases
            val = switch(lt(reverse1, 0), n_val, p_val)
            val = switch(le(len2, 0), len1 + 1, val)
            val = switch(ge(sl2, len2), len1 + 1, val)
            val = switch(lt(sl2, 0), -len1 - 1, val)
            if sl1.step:
                val = switch(eq(sl1.step, 0), len1 + 1, val)
            return val
    else:
        # We are deleaing with two slices that need to be put together
        # according to the two steps we have 4 different combinations of
        # positive/negative. I will denote the case I'm looking at by
        # suffixes to the variables (nn,np,pn,pp):
        flen = sl2.stop - sl2.start
        p_step = sl1.step * sl2.step
        n_step = sl1.step * sl2.step * -1

        pp_start = minimum(sl1.start + sl2.start * sl1.step, sl1.stop)
        pp_stop = minimum(sl1.start + sl2.stop * sl1.step, sl1.stop)

        pn_stop = sl1.start + (sl2.start - 1) * sl1.step
        pn_stop = switch(
            and_(lt(pn_stop, 0), gt(flen, 0)),
            -len1 - 1,
            minimum(pn_stop, sl1.stop),
        )
        pn_start = sl1.start + (sl2.stop - 1) * sl1.step
        pn_start = minimum(pn_start, sl1.stop)
        pn_start = maximum(pn_start, 0)

        np_stop = sl1.stop - sl2.stop * sl1.step - 1
        np_stop = switch(
            and_(lt(np_stop, 0), gt(flen, 0)),
            -len1 - 1,
            maximum(sl1.start - 1, np_stop),
        )
        np_start = maximum(sl1.start, sl1.stop - sl2.start * sl1.step - 1)

        nn_start = maximum(sl1.start, (sl1.stop - 1) - (sl2.stop - 1) * sl1.step)
        nn_stop = maximum(sl1.start, sl1.stop - sl2.start * sl1.step)

        start = switch(
            lt(reverse2 * reverse1, 0),
            switch(lt(reverse1, 0), np_start, pn_start),
            switch(lt(reverse1, 0), nn_start, pp_start),
        )

        stop = switch(
            lt(reverse2 * reverse1, 0),
            switch(lt(reverse1, 0), np_stop, pn_stop),
            switch(lt(reverse1, 0), nn_stop, pp_stop),
        )

        step = switch(lt(reverse2 * reverse1, 0), n_step, p_step)
        start = switch(le(flen, 0), 0, start)
        stop = switch(le(flen, 0), 0, stop)

        return slice(start, stop, step)


@register_canonicalize
@register_specialize
@local_optimizer([Subtensor])
def local_subtensor_merge(fgraph, node):
    """
    Refactored optimization to deal with all cases of tensor merging.
    Given a subgraph of the form Subtensor(Subtensor(u)), the optimization
    expresses all slices in a canonical form, and then merges them together.

    """

    if isinstance(node.op, Subtensor):
        u = node.inputs[0]
        if u.owner and isinstance(u.owner.op, Subtensor):
            # We can merge :)
            # x actual tensor on which we are picking slices
            x = u.owner.inputs[0]
            # slices of the first applied subtensor
            slices1 = get_idx_list(u.owner.inputs, u.owner.op.idx_list)
            slices2 = get_idx_list(node.inputs, node.op.idx_list)
            # Get the shapes of the vectors !
            try:
                # try not to introduce new shape into the graph
                xshape = fgraph.shape_feature.shape_of[x]
                ushape = fgraph.shape_feature.shape_of[u]
            except AttributeError:
                # Following the suggested use of shape_feature which should
                # consider the case when the compilation mode doesn't
                # include the ShapeFeature
                xshape = x.shape
                ushape = u.shape

            merged_slices = []
            pos_2 = 0
            pos_1 = 0
            while (pos_1 < len(slices1)) and (pos_2 < len(slices2)):
                slice1 = slices1[pos_1]
                if isinstance(slice1, slice):
                    merged_slices.append(
                        merge_two_slices(
                            fgraph, slice1, xshape[pos_1], slices2[pos_2], ushape[pos_2]
                        )
                    )
                    pos_2 += 1
                else:
                    merged_slices.append(slice1)
                pos_1 += 1

            if pos_2 < len(slices2):
                merged_slices += slices2[pos_2:]
            else:
                merged_slices += slices1[pos_1:]

            merged_slices = tuple(as_index_constant(s) for s in merged_slices)
            subtens = Subtensor(merged_slices)

            sl_ins = Subtensor.collapse(
                merged_slices, lambda x: isinstance(x, Variable)
            )
            # Do not call make_node for test_value
            out = subtens(x, *sl_ins)

            # Copy over previous output stacktrace
            # and stacktrace from previous slicing operation.
            # Why? Because, the merged slicing operation could have failed
            # because of either of the two original slicing operations
            orig_out = node.outputs[0]
            copy_stack_trace([orig_out, node.inputs[0]], out)

            # Restore original broadcastable dimensions that `subtens()` may
            # have been unable to infer again
            if out.type != orig_out.type:
                assert out.dtype == orig_out.dtype
                assert out.ndim == orig_out.ndim
                out = patternbroadcast(out, orig_out.broadcastable)
                copy_stack_trace([orig_out, node.inputs[0]], out)
            return [out]


@register_useless
@register_canonicalize
@register_specialize
@local_optimizer([Subtensor])
def local_subtensor_of_alloc(fgraph, node):
    """

    alloc(val)[x:y] -> alloc(val[...])
    alloc(val)[x:y] -> alloc(val)
    This can be seen as a lift, but it also reduce the number of computation/memory.

    """
    if not isinstance(node.op, Subtensor):
        return False
    u = node.inputs[0]
    if u.owner is None:
        return False
    if not isinstance(u.owner.op, Alloc):
        return False
    slices = get_idx_list(node.inputs, node.op.idx_list)
    val = u.owner.inputs[0]
    dims = u.owner.inputs[1:]
    assert len(slices) <= len(dims)

    # Number of dimensions added to val
    n_added_dims = u.ndim - val.ndim
    # Dimensions of the returned alloc
    nw_dims = []
    # Slices to take from val
    val_slices = []

    for i, (sl, dim) in enumerate(zip(slices, dims)):
        # If val was not copied over that dim,
        # we need to take the appropriate subtensor on it.
        if i >= n_added_dims:
            # We check that the corresponding val dimensions was
            # not a broadcasted dimensions.
            if (
                val.type.ndim > (i - n_added_dims)
                and val.type.broadcastable[i - n_added_dims]
            ):
                val_slices.append(slice(None))
            else:
                val_slices.append(sl)

        csl, _ = get_canonical_form_slice(sl, dim)
        if type(csl) is not slice:
            # That dimension is removed.
            pass
        else:
            nw_dim = csl.stop - csl.start

            if csl.step != 1:
                # Do not add the ceil_intdiv() graphs in the graphs
                # when this is not needed as it prevent detecting the
                # correct broadcast pattern.
                nw_dim = ceil_intdiv(nw_dim, csl.step)
            nw_dims += [nw_dim]

    nw_val = val[tuple(val_slices)]
    nw_dims += dims[len(slices) :]
    if nw_val.ndim > len(nw_dims):
        return False
    rval = alloc(nw_val, *nw_dims)
    if type(rval) not in (list, tuple):
        rval = [rval]
    if rval[0].type != node.outputs[0].type:
        # It happen that the make_node() isn't able to infer the same pattern.
        # We know it is safe, so fix that.
        rval[0] = patternbroadcast(rval[0], node.outputs[0].broadcastable)

    return rval


@register_canonicalize
@register_stabilize
@register_specialize
@local_optimizer([Subtensor])
def local_subtensor_of_dot(fgraph, node):
    """Rewrite ``aet.dot(A, B)[idxs]`` into ``aet.dot(A[idxs_a], B[idxs_b])``.

    ``idxs_a`` is the first ``A.ndim-1`` entries of ``idxs``, and ``idxs_b`` is
    the remaining entries of ``idxs`` (if any), modified to skip the
    second-to-last dimension of ``B`` (because dot sums over this dimension).

    """
    if not isinstance(node.op, Subtensor):
        return
    if not node.inputs[0].owner or not isinstance(node.inputs[0].owner.op, Dot):
        return
    # If there is other node that use the outputs of the dot
    # We don't want to compute twice the sub part.
    if len(fgraph.clients[node.inputs[0]]) > 1:
        return

    a = node.inputs[0].owner.inputs[0]
    b = node.inputs[0].owner.inputs[1]

    idx_list = get_idx_list(node.inputs, node.op.idx_list)

    num_a_indices = min(a.ndim - 1, len(idx_list))
    a_indices = idx_list[:num_a_indices]
    b_indices = idx_list[num_a_indices:]

    # This is necessary because np.dot sums the last index of a with the second to last of b
    # so we want to skip the second-to-last index into b.
    # This wasn't necessary for a, because we just omitted the last index.
    # We skip this if b.ndim = 1, since then we just want b_sub = b, not b_sub = b[:]
    # (dot also handles b.ndim < 2 as a special case)
    if b.ndim > 1 and len(b_indices) >= b.ndim - 1:
        b_indices = (
            b_indices[: b.ndim - 2]
            + (slice(None, None, None),)
            + b_indices[b.ndim - 2 :]
        )

    a_sub = a.__getitem__(tuple(a_indices))
    b_sub = b.__getitem__(tuple(b_indices)) if b_indices else b

    # Copy over previous output stacktrace to a_sub and b_sub,
    # because an error in the subtensor operation (e.g. an index error)
    # on either a or b must correspond to an error in the
    # subtensor operation on their dot product.
    copy_stack_trace(node.outputs[0], [a_sub, b_sub])

    # Copy over previous output stacktrace and previous dot product stacktrace,
    # because an error here may correspond to an either in either the original
    # dot product, or in the dot product after the subtensor operation.
    r = dot(a_sub, b_sub)
    copy_stack_trace([node.outputs[0], node.inputs[0]], r)

    return [r]


@register_canonicalize
@local_optimizer([add])
def local_IncSubtensor_serialize(fgraph, node):
    """
    When using Subtensor, gradient graphs can be ugly.

    If we ask for grad(f(a[0]), a), we are going to get something like

        IncSubtensor(Elemwise{second}(a, 0), g(f(a[0])), [0])

    This might be ugly, but at least it's as fast as you could want.
    If we ask for grad(f(a[0], a[1], a[2]), a), it's much worse...

        Elemwise{Add}
            IncSubtensor(Elemwise{second}(a, 0), g(f(a[0])), [0])
            IncSubtensor(Elemwise{second}(a, 0), g(f(a[1])), [1])
            IncSubtensor(Elemwise{second}(a, 0), g(f(a[2])), [2])

    This is much worse because this time we have to produce 3 matrices
    the size of 'a', just so we can add them together.

    This Op rearranges IncSubtensor's that all work on the same
    initial argument (here, Elemwise{second}(a,0)) into a chain.  The
    advantage of the chain structure is that each one can be optimized
    later in the pipeline to operate inplace.

    Ideally, the op will do something like this:

    #
    #  add(x, incsubtensor(b, c), incsubtensor(b, d))
    #  -> incsubtensor(incsubtensor(add(x,b,b), c), d)

    """

    def movable(i):
        # Return True iff this is a incsubtensor that we can move
        return (
            i.owner
            and isinstance(
                i.owner.op,
                (
                    IncSubtensor,
                    AdvancedIncSubtensor1,
                    AdvancedIncSubtensor,
                ),
            )
            and i.type == o_type
            and len(fgraph.clients[i]) == 1
            and not i.owner.op.set_instead_of_inc
        )

    if node.op == add:
        o_type = node.outputs[0].type

        movable_inputs = [i for i in node.inputs if movable(i)]

        if movable_inputs:
            new_inputs = [i for i in node.inputs if not movable(i)] + [
                mi.owner.inputs[0] for mi in movable_inputs
            ]
            if len(new_inputs) == 0:
                new_add = new_inputs[0]
            else:
                new_add = add(*new_inputs)

                # Copy over stacktrace from original output, as an error
                # (e.g. an index error) in this add operation should
                # correspond to an error in the original add operation.
                copy_stack_trace(node.outputs[0], new_add)

            # stack up the new incsubtensors
            tip = new_add
            for mi in movable_inputs:
                assert tip.type == o_type
                assert tip.type == mi.owner.inputs[0].type
                tip = mi.owner.op(tip, *mi.owner.inputs[1:])
                # Copy over stacktrace from outputs of the original
                # "movable" operation to the new operation.
                copy_stack_trace(node.outputs + mi.owner.outputs, tip)

            return [tip]

        # print incsub_inputs, [id(i.owner.inputs[0]) for i in incsub_inputs]


# We register it in a TopoOptimizer inside the canonizer EQ optimizer.
# Otherwise in some cases it was making the EQ optimizer use 45. In
# the TopoOptimizer, the EQ only use 5 passes.
compile.optdb.register(
    "pre_local_IncSubtensor_serialize",
    in2out(local_IncSubtensor_serialize),
    # Just before canonizer
    0.99,
    "fast_run",
)


# after priority 50 Destructive inplace operations
# gemm is the first one now, at priority 70


@local_optimizer([IncSubtensor], inplace=True)
def local_inplace_setsubtensor(fgraph, node):
    if isinstance(node.op, IncSubtensor) and not node.op.inplace:
        dta = node.op.destroyhandler_tolerate_aliased
        new_op = node.op.__class__(
            node.op.idx_list,
            inplace=True,
            set_instead_of_inc=node.op.set_instead_of_inc,
            destroyhandler_tolerate_aliased=dta,
        )
        new_node = new_op(*node.inputs)
        val = getattr(node.outputs[0].tag, "nan_guard_mode_check", True)
        new_node.tag.nan_guard_mode_check = val

        # Copy stacktrace from original outputs to new outputs.
        # This is sensible, because the new operation is the
        # same as the old one, but now with different attributes.
        copy_stack_trace(node.outputs, new_node)
        return [new_node]
    return False


compile.optdb.register(
    "local_inplace_setsubtensor",
    TopoOptimizer(
        local_inplace_setsubtensor, failure_callback=TopoOptimizer.warn_inplace
    ),
    60,
    "fast_run",
    "inplace",
)


@local_optimizer([AdvancedIncSubtensor1], inplace=True)
def local_inplace_AdvancedIncSubtensor1(fgraph, node):
    if isinstance(node.op, AdvancedIncSubtensor1) and not node.op.inplace:
        new_op = node.op.clone_inplace()
        new_node = new_op(*node.inputs)
        copy_stack_trace(node.outputs, new_node)
        return [new_node]
    return False


compile.optdb.register(
    "local_inplace_AdvancedIncSubtensor1",
    TopoOptimizer(
        local_inplace_AdvancedIncSubtensor1, failure_callback=TopoOptimizer.warn_inplace
    ),
    60,
    "fast_run",
    "inplace",
)


@local_optimizer([AdvancedIncSubtensor], inplace=True)
def local_inplace_AdvancedIncSubtensor(fgraph, node):
    if isinstance(node.op, AdvancedIncSubtensor) and not node.op.inplace:
        new_op = type(node.op)(
            inplace=True, set_instead_of_inc=node.op.set_instead_of_inc
        )
        new_node = new_op(*node.inputs)
        copy_stack_trace(node.outputs, new_node)
        return [new_node]
    return False


compile.optdb.register(
    "local_inplace_AdvancedIncSubtensor",
    TopoOptimizer(
        local_inplace_AdvancedIncSubtensor, failure_callback=TopoOptimizer.warn_inplace
    ),
    60,
    "fast_run",
    "inplace",
)


# Register old name
@register_canonicalize("local_incsubtensor_of_allocs")
@register_stabilize("local_incsubtensor_of_allocs")
@local_optimizer([IncSubtensor, AdvancedIncSubtensor, AdvancedIncSubtensor1])
def local_incsubtensor_of_zeros(fgraph, node):
    """
    IncSubtensor(x, zeros, idx) -> x

    """
    if (
        isinstance(node.op, (IncSubtensor, AdvancedIncSubtensor, AdvancedIncSubtensor1))
        and not node.op.set_instead_of_inc
    ):
        x = node.inputs[0]
        y = node.inputs[1]
        try:
            # Don't use only_process_constants=True. We need to
            # investigate Alloc of 0s but with non constant shape.
            if get_scalar_constant_value(y, elemwise=False) == 0:
                # No need to copy over the stacktrace,
                # because x should already have a stacktrace
                return [x]
        except NotScalarConstantError:
            return


@register_canonicalize
@register_specialize
@local_optimizer([IncSubtensor])
def local_incsubtensor_of_zeros_to_setsubtensor(fgraph, node):
    """
    IncSubtensor(zeros, x, ...) -> SetSubtensor(zeros, x, ...)
    """
    if isinstance(node.op, (IncSubtensor)) and not node.op.set_instead_of_inc:
        x = node.inputs[0]

        if isinstance(x, Constant) and not np.any(x.data):
            return [
                IncSubtensor(
                    node.op.idx_list,
                    node.op.inplace,
                    set_instead_of_inc=True,
                    destroyhandler_tolerate_aliased=node.op.destroyhandler_tolerate_aliased,
                )(*node.inputs)
            ]


@register_canonicalize("local_setsubtensor_of_allocs")
@register_stabilize("local_setsubtensor_of_allocs")
@local_optimizer([IncSubtensor])
def local_setsubtensor_of_constants(fgraph, node):
    """
    SetSubtensor(x, x[idx], idx) -> x

    when x is constant or alloc.

    """
    if isinstance(node.op, IncSubtensor) and node.op.set_instead_of_inc:
        x = node.inputs[0]
        y = node.inputs[1]

        # Don't use only_process_constants=True. We need to
        # investigate Alloc of 0s but with non constant shape.
        try:
            replace_x = get_scalar_constant_value(x, elemwise=False)
        except NotScalarConstantError:
            return

        try:
            replace_y = get_scalar_constant_value(y, elemwise=False)
        except NotScalarConstantError:
            return

        if replace_x == replace_y:

            # No need to copy over the stacktrace,
            # because x should already have a stacktrace
            return [x]
        else:
            return False


@register_canonicalize
@register_stabilize
@local_optimizer([AdvancedSubtensor1])
def local_adv_sub1_adv_inc_sub1(fgraph, node):
    """Optimize the possible AdvSub1(AdvSetSub1(...), ...).

    AdvancedSubtensor1(AdvancedSetSubtensor1(x, y, idx), idx) -> y

    Notes
    -----
    This opt add AssertOp. Otherwise, it would remove shape and
    index error. If you want to get rid of them, see the
    :ref:`unsafe_optimization` section.

    WARNING:
    A previous version of this optimization also matched
    AdvancedSubtensor1(AdvancedIncSubtensor1(0s, y, idx), idx) -> y
    This is incorrect when there are duplicate indices.
    The current version warns the user about potential past issues.

    """
    if not isinstance(node.op, AdvancedSubtensor1):
        return
    inp = node.inputs[0]
    if not inp.owner or not isinstance(inp.owner.op, AdvancedIncSubtensor1):
        return
    idx = node.inputs[1]
    idx2 = inp.owner.inputs[2]
    x = inp.owner.inputs[0]
    y = inp.owner.inputs[1]
    if idx is not idx2:
        return
    if (
        not inp.owner.op.set_instead_of_inc
        and
        # Don't use only_process_constants=True. We need to
        # investigate Alloc of 0s but with non constant shape.
        extract_constant(x, elemwise=False) != 0
    ):
        return

    if not inp.owner.op.set_instead_of_inc:
        if config.warn__inc_subtensor1_opt:
            warnings.warning(
                "Your current code is fine, but Aesara versions "
                "between 0.7rc1 and 0.10 (or development versions "
                "between Nov. 2014 and May 2017) "
                "might have given incorrect results. This graph has "
                "following pattern: inc_subtensor(zeros[idx], x)[idx], "
                "where idx is an array of integers. This used to be "
                'optimized to "x", which is incorrect if there are '
                "duplicated indices in idx. "
                "To disable this warning, set the Aesara flag "
                "warn__inc_subtensor1_opt to False."
            )
        return

    cond = [aet_all(and_(lt(idx, x.shape[0]), ge(idx, -x.shape[0])))]
    if not fgraph.shape_feature.same_shape(idx, y, 0, 0):
        cond.append(eq(idx.shape[0], y.shape[0]))
    r = Assert(
        "Bad indexing or shapes in a AdvancedIncSubtensor1 " "that was optimized away"
    )(y, *cond)
    copy_stack_trace(y, r)

    if r.dtype == node.outputs[0].dtype:
        return [r]
    # It is possible that y is upcast or downcast to x.dtype.
    # In all case, as we set or add with 0, we can just cast y.
    r2 = cast(r, node.outputs[0].dtype)

    # Copy over stacktrace from before casting, since
    # we don't expect problems in the casting operation,
    # and any problems in the indexing would have been spotted above.
    copy_stack_trace(r, r2)
    return [r2]


@register_specialize
@register_stabilize
@register_canonicalize
@register_useless
@local_optimizer([IncSubtensor, AdvancedIncSubtensor, AdvancedIncSubtensor1])
def local_useless_inc_subtensor_alloc(fgraph, node):
    """
    Replaces an [Advanced]IncSubtensor[1], whose increment is an `alloc` of
    a fully or partially broadcastable variable, by one that skips the
    intermediate `alloc` where possible.

    """
    if isinstance(node.op, (IncSubtensor, AdvancedIncSubtensor, AdvancedIncSubtensor1)):
        x = node.inputs[0]
        y = node.inputs[1]
        i = node.inputs[2:]

        if y.owner is not None and isinstance(y.owner.op, Alloc):
            # `z` is the input of the Alloc op, i.e. aet.alloc(z, <shape>)
            z = y.owner.inputs[0]

            try:
                shape_feature = fgraph.shape_feature
            except AttributeError:
                # The shape feature may not be available in some mode, but we
                # need it for this optimization, so don't continue.
                return False

            shape_of = shape_feature.shape_of
            same_shape = shape_feature.same_shape

            # Get the subtensor of `x` indexed by `i` in order to compare
            # shapes later.
            if isinstance(node.op, IncSubtensor):
                xi = Subtensor(node.op.idx_list)(x, *i)
            elif isinstance(node.op, AdvancedIncSubtensor):
                xi = advanced_subtensor(x, *i)
            elif isinstance(node.op, AdvancedIncSubtensor1):
                xi = advanced_subtensor1(x, *i)
            else:
                raise Exception("Should never happen!")

            reason = "local_useless_incsubtensor_alloc"

            # Add `xi` to the shape feature `fgraph`. This is important for
            # shape inference later because the variable must be part of the
            # function graph in order to call `same_shape` on it.
            if xi not in shape_of:
                shape_feature.on_import(fgraph, xi.owner, f"{reason}: add `xi`")

            # `xi` may have more dimensions than `y` since the subtensor ops
            # do automatic broadcasting of the increment internally. Thus, we
            # need to make the leading implicitly broadcasted dimensions
            # explicit for shape comparison later.
            if xi.ndim > y.ndim:
                y = shape_padleft(y, xi.ndim - y.ndim)
                if y not in shape_of:
                    shape_feature.on_import(fgraph, y.owner, f"{reason}: add `y`")

            # Build `z_broad` explicitly to include extra implicit dimensions.
            z_broad = (True,) * (xi.ndim - z.ndim) + z.broadcastable

            cond = [
                # The shapes of `y` and `xi` must either agree or `y` may
                # also have shape equal to 1 which may be treated as a
                # broadcastable dimension by the subtensor op.
                or_(eq(y.shape[k], 1), eq(y.shape[k], xi.shape[k]))
                # Loop over all dimensions.
                for k in range(xi.ndim)
                # We need to check the above shapes, if
                # * the pre-alloc increment `z` is broadcastable in
                # dimension `k` (if it isn't, then the shapes of `z` and
                # `y` are the same by the definition of the `Alloc` op in
                # this dimension and replacing `y` by `z` will not hide a
                # shape error), and
                # * `xi` and `y` do not have the same shape in dimension
                # `k` or we cannot infer the shape statically (if the
                # shapes of `xi` and `y` are not the same, then replacing
                # `y` by `z` will hide the shape error of `y`), and
                # * the shape of `y` is not equal to 1 or we cannot infer
                # the shape statically (if the shape of `y` is equal to
                # 1, then `y` is broadcasted by the inc_subtensor op
                # internally, so the shapes of `xi` and `y` do not need
                # to match in dimension `k`; else we need to check at
                # runtime that the shape of `y` is either 1 or the same
                # as `xi` or otherwise replacing `y` by `z` will hide a
                # shape error).
                if (
                    z_broad[k]
                    and not same_shape(xi, y, dim_x=k, dim_y=k)
                    and shape_of[y][k] != 1
                )
            ]

            if len(cond) > 0:
                msg = "`x[i]` and `y` do not have the same shape."
                z = Assert(msg)(z, *cond)

            r = node.op(x, z, *i)
            # Copy over stacktrace from previous output, since
            # we don't expect problems when removing the intermediate
            # alloc operation and so we still want to point at the line
            # of the inc_subtensor operation.
            copy_stack_trace(node.outputs, r)

            return [r]


@register_useless
@register_canonicalize
@register_specialize
@local_optimizer([Rebroadcast])
def local_useless_rebroadcast(fgraph, node):
    """
    Remove Rebroadcast if id does not actually change the broadcasting pattern.

    """
    if isinstance(node.op, Rebroadcast):
        x = node.inputs[0]
        if np.all(x.broadcastable == node.outputs[0].broadcastable):
            # No broadcastable flag was modified
            # No need to copy over stack trace,
            # because x should already have a stack trace.
            return [x]
        else:
            # Keep the flags that modify something
            new_axis = {}
            for dim, bc in list(node.op.axis.items()):
                if x.broadcastable[dim] != bc:
                    new_axis[dim] = bc
            if new_axis == node.op.axis:
                # All flags are useful
                return
            else:
                r = Rebroadcast(*list(new_axis.items()))(x)
                # Copy over stacktrace from previous output
                copy_stack_trace(node.outputs, r)
                return [r]


@register_canonicalize
@register_specialize
@local_optimizer([Rebroadcast])
def local_rebroadcast_lift(fgraph, node):
    """
    Lifts Rebroadcast through unary Elemwise operations,
    and merges consecutive Rebroadcasts.

    Rebroadcast(Elemwise(x)) => Elemwise(Rebroadcast(x))
    Rebroadcast(Rebroadcast(x)) => Rebroadcast(x)

    """
    op = node.op
    if not isinstance(op, Rebroadcast):
        return False

    input = node.inputs[0]
    inode = input.owner
    if inode and isinstance(inode.op, Elemwise) and len(inode.inputs) == 1:
        # It may happen that `input` has no client because this optimization
        # is called from `apply_rebroadcast_opt`, which in particular is used
        # by the `unbroadcast` function before we are in the actual function
        # compilation phase.
        if len(fgraph.clients.get(input, ())) == 1:
            rebroadcasted = Rebroadcast(*list(op.axis.items()))(inode.inputs[0])
            # Copy over stacktrace from previous output (after rebroadcasting)
            # to new output, because an error in the new graph right after
            # rebroadcasting must have been caused by the previous rebroadcasting.
            copy_stack_trace(node.outputs, rebroadcasted)

            rval = inode.op.make_node(rebroadcasted).outputs

            # Copy over stacktrace from previous output (after rebroadcasting)
            # and input (after elemwise operation) to new output, because an
            # error in the new graph could have been caused by either of the
            # two ops.
            copy_stack_trace(node.outputs + node.inputs, rval)

            return rval
    if inode and isinstance(inode.op, Rebroadcast):
        # the "axis" specification in the outer Rebroadcast overrides
        # the axis of the inner one
        axis = inode.op.axis.copy()
        axis.update(op.axis)
        iinput = inode.inputs[0]

        rval = [Rebroadcast(*list(axis.items()))(iinput)]

        # Copy over stacktrace from previous output (after second rebroadcast)
        # and from previous input (after first rebroadcast op) because an error in
        # the new graph could have been caused by either of the two
        # rebroadcast ops.
        copy_stack_trace(node.outputs + node.inputs, rval)
        return rval


def apply_rebroadcast_opt(rval):
    """
    Apply as many times as required the optimization local_useless_rebroadcast
    and local_rebroadcast_lift.

    Parameters
    ----------
    rval: a Variable

    Returns
    -------
    A Variable (the same if no optimization can be applied)

    """

    fg = FunctionGraph([], [])
    changed = True
    while changed and rval.owner:
        changed = False
        rval2 = local_useless_rebroadcast.transform(fg, rval.owner)
        if rval2:
            assert len(rval2) == 1
            rval = rval2[0]
            changed = True
        if rval.owner:
            rval2 = local_rebroadcast_lift.transform(fg, rval.owner)
            if rval2:
                assert len(rval2) == 1
                rval = rval2[0]
                changed = True
    return rval


@register_specialize
@register_canonicalize
@register_useless
@local_optimizer([Join])
def local_join_1(fgraph, node):
    """Join(i, x) => x

    Remove Join() when only one element is joined.

    """
    if not isinstance(node.op, Join):
        return
    tensors = node.inputs[1:]
    if len(tensors) == 1:
        # We don't need to copy over any stacktrace here, because the
        # input variable should already have its own stacktrace.
        return [tensors[0]]


# TODO: merge in local_useless_join
@register_useless
@register_specialize
@register_canonicalize
@local_optimizer([Join])
def local_join_empty(fgraph, node):
    """Join(i, x, y, empty) => Join(i, x, y)

    Remove empty inputs to joins. The empty inputs can be anywhere.

    """
    if not isinstance(node.op, Join):
        return
    new_inputs = []
    try:
        join_idx = get_scalar_constant_value(
            node.inputs[0], only_process_constants=True
        )
    except NotScalarConstantError:
        return
    for idx in range(1, len(node.inputs)):
        inp = node.inputs[idx]
        # We can not use size == 0,, as this can change shape from 3,0
        # to 2,0.  This trigger DebugMode error. This happen with
        # stack(...,[]) as this add a dimshuffle on [], that add a
        # dimensions with shape 1.
        if isinstance(inp, Constant) and inp.data.shape[join_idx] == 0:
            continue
        new_inputs.append(inp)
    if len(new_inputs) < len(node.inputs) - 1:
        if len(new_inputs) == 0:
            # aet.join do not work in that case.
            # constant folding will take care of this case.
            return
        ret = join(node.inputs[0], *new_inputs)
        o = node.outputs[0]
        if ret.dtype != o.dtype:
            # Join can upcast some inputs
            return

        # Copy over stacktrace from previous output (after join op)
        # to new output, because an error in the new op must be caused
        # by an error in the old join op.
        copy_stack_trace(node.outputs, ret)

        if ret.type != o.type:
            assert ret.dtype == o.dtype
            assert ret.ndim == o.ndim
            ret = patternbroadcast(ret, node.outputs[0].broadcastable)

        # Copy over stacktrace from previous output
        # (after patternbroadcast op) for same reasons as before.
        copy_stack_trace(node.outputs, ret)

        return [ret]


@register_specialize
@register_canonicalize
@register_useless
@local_optimizer([Join])
def local_join_make_vector(fgraph, node):
    """Join(0, make_vector1, make_vector2, ...) => Join(0, make_vector12, ...)

    Merge MakeVector inputs to Join. This can make the join completly
    disapear with the local_join_1 opt.

    """
    if not isinstance(node.op, Join) or node.outputs[0].ndim != 1:
        return
    new_inputs = [node.inputs[1]]
    for idx in range(2, len(node.inputs)):
        inp = node.inputs[idx]
        if (
            inp.owner
            and isinstance(inp.owner.op, MakeVector)
            and new_inputs[-1].owner
            and isinstance(new_inputs[-1].owner.op, MakeVector)
            and
            # MakeVector have a dtype parameter
            inp.owner.op == new_inputs[-1].owner.op
        ):
            inps = new_inputs[-1].owner.inputs + inp.owner.inputs
            new_inputs[-1] = inp.owner.op(*inps)

            # Copy over stacktrace from previous output (after join op)
            # to new intermediate output, because an error in the intermediate
            # op must be caused by an error in the old join op.
            copy_stack_trace(node.outputs, new_inputs[-1])
        else:
            new_inputs.append(inp)
    if len(new_inputs) < len(node.inputs) - 1:
        ret = join(node.inputs[0], *new_inputs)

        # Copy over stacktrace from previous output (after join op)
        # to new output, because an error in the new op must be caused
        # by an error in the old join op.
        copy_stack_trace(node.outputs, ret)
        return [ret]


@register_useless("local_remove_switch_const_cond")
@register_canonicalize("fast_compile", "local_remove_switch_const_cond")
@register_specialize
@local_optimizer([Elemwise])
def local_useless_switch(fgraph, node):
    """
    This optimization makes the following changes in the graph:

    ``aet.switch(cond, left, right)`` ->
            ``if cond is constant and cond == 0``: right
            ``if cond is constant and cond != 0``: left
            ``if left is right`` -> ``left``

    and

    ``aet.switch(le(shape_i{id}(X), 0), 0, shape_i{id}(X))`` -> ``shape_i{id}(X)``

    """
    if isinstance(node.op, Elemwise) and isinstance(node.op.scalar_op, aes.Switch):

        cond = extract_constant(node.inputs[0], only_process_constants=True)

        if (isinstance(cond, np.ndarray) and cond.ndim == 0) or isinstance(
            cond, np.number
        ):
            if cond == 0:
                correct_out = node.inputs[2]
            else:
                correct_out = node.inputs[1]

            if correct_out.dtype != node.outputs[0].dtype:
                out = cast(correct_out, node.outputs[0].dtype)
            else:
                out = correct_out

            out_shape = broadcast_shape(*node.inputs)
            out = alloc(out, *out_shape)

            # Copy over stacktrace from selected output to new output
            copy_stack_trace(node.outputs + correct_out, out)
            return [out]

        # if left is right -> left
        if node.inputs[1] is node.inputs[2]:
            # Note: No need to copy over stacktrace, because the input node
            # already has its own stacktrace
            if cond.type == node.inputs[1].type:
                return [node.inputs[1]]

            ret = fill(cond, node.inputs[1])

            # Copy over stacktrace from switch output and correct branch
            copy_stack_trace(node.outputs + node.inputs[1], ret)
            return [ret]

        # This case happens with scan.
        # Elemwise{switch}(le(shape_i{id}(X), 0), 0, shape_i{id}(X)) -> shape_i{id}(X)
        left = node.inputs[1]
        right = node.inputs[2]
        cond_var = node.inputs[0]
        if (
            cond_var.owner
            and isinstance(cond_var.owner.op, Elemwise)
            and isinstance(cond_var.owner.op.scalar_op, aes.LE)
            and cond_var.owner.inputs[0].owner
            and isinstance(cond_var.owner.inputs[0].owner.op, Shape_i)
            and extract_constant(cond_var.owner.inputs[1], only_process_constants=True)
            == 0
            and extract_constant(left, only_process_constants=True) == 0
            and right is cond_var.owner.inputs[0]
        ):
            assert right.type == node.outputs[0].type
            # No need to copy over stacktrace, because the right input node
            # already has its own stacktrace
            return [right]
        return False
    return False


# Merge add/sub/mul/div/minimum/maximum/... of switches sharing the same
# condition, to enable further simplification of their branches
# Example: switch(c, a, b) + switch(c, x, y) -> switch(c, a+x, b+y)
@register_canonicalize
@local_optimizer([Elemwise])
def local_merge_switch_same_cond(fgraph, node):
    # node must be binary elemwise or add or mul
    if not isinstance(node.op, Elemwise) or not isinstance(
        node.op.scalar_op, (aes.BinaryScalarOp, aes.Add, aes.Mul)
    ):
        return
    # all inputs must be switch
    if not all(
        s.owner
        and isinstance(s.owner.op, Elemwise)
        and isinstance(s.owner.op.scalar_op, aes.Switch)
        for s in node.inputs
    ):
        return
    # all switch conditions must be the same
    cond = node.inputs[0].owner.inputs[0]
    if not all(s.owner.inputs[0] is cond for s in node.inputs[1:]):
        return
    # pull out switch
    return [
        switch(
            cond,
            node.op(*[s.owner.inputs[1] for s in node.inputs]),
            node.op(*[s.owner.inputs[2] for s in node.inputs]),
        )
    ]


#############
# Tile Opts #
#############
@register_useless
@register_canonicalize
@register_stabilize
@local_optimizer([Tile])
def local_useless_tile(fgraph, node):
    """Tile(x, (1,)*N) -> x

    This is useless tile. (1,)*N, just mean a vector with all element
    being 1.

    """
    if isinstance(node.op, Tile):
        try:
            a = get_scalar_constant_value(node.inputs[1], only_process_constants=True)
            if a == 1:
                try:
                    l = get_vector_length(node.inputs[1])
                    if l == node.inputs[0].ndim:
                        # No need to copy over any stacktrace as previous
                        # input variable already has a stacktrace
                        return [node.inputs[0]]
                    elif l < node.inputs[0].ndim:
                        # The Op don't support that case, so we can't
                        # implement the opt and test it.
                        return
                        return [node.inputs[0]]
                    else:
                        # The Op don't support that case, so we can't
                        # implement the opt and test it.
                        return
                        x_nd = node.inputs[0].ndim
                        broad = ["x"] * (l - x_nd) + range(x_nd)
                        ret = node.inputs[0].dimshuffle(broad)
                        # Copy over stacktrace from previous output node,
                        # and from node before tiling operation.
                        copy_stack_trace(node.outputs + node.inputs[0], ret)
                        return [ret]
                except ValueError:
                    return
        except NotScalarConstantError:
            return


##############
# Split Opts #
##############
@register_useless
@register_canonicalize
@register_specialize
@local_optimizer([Split])
def local_useless_split(fgraph, node):
    """Split{n_splits=1}(x, y) -> x

    Remove Split with only 1 split.

    """
    if isinstance(node.op, Split):
        if node.op.len_splits == 1:
            x, axis, splits = node.inputs
            out = assert_op(x, eq(splits.shape[0], 1))
            # Copy over stacktrace from previous output node.
            copy_stack_trace(node.outputs, out)
            out2 = assert_op(out, eq(x.shape[axis], splits[0]))
            # Copy over stacktrace from previous output node.
            copy_stack_trace(out, out2)

            return [out2]


################
# Flatten Opts #
################
@register_canonicalize
@register_stabilize
@local_optimizer([Flatten])
def local_flatten_lift(fgraph, node):
    """
    Flatten(UnaryElemwise(x)) -> UnaryElemwise(Flatten(x))

    This optimization is needed by optimization
    log1msigm_to_softplus to get applied when there is a flatten.

    """
    if (
        isinstance(node.op, Flatten)
        and node.inputs[0].owner
        and isinstance(node.inputs[0].owner.op, Elemwise)
        and len(node.inputs[0].owner.inputs) == 1
    ):
        f = node.op(node.inputs[0].owner.inputs[0])

        # Copy over stacktrace from previous output node (flatten op),
        # since this is the op which may cause an error for f.
        copy_stack_trace(node.outputs, f)

        e = node.inputs[0].owner.op(f)

        # Copy over stacktrace from previous output node and from unary
        # elementwise output node since if there was an error, it would
        # probably have come from that operation.
        copy_stack_trace(node.outputs + [node.inputs[0]], e)

        return [e]


##################
# Reshape opts   #
##################


def local_reshape_chain(op):
    @local_optimizer([op])
    def f(fgraph, node):
        """
        Reshape(Reshape(shape1),shape2) -> Reshape(shape2)

        """
        if not check_chain(node, op, op):
            return False

        # TODO: this can permit a failing program to run by eliminating
        #       the lower reshape
        rval = node.op(node.inputs[0].owner.inputs[0], node.inputs[1])

        # Copy over stacktrace from previous output node, as any error
        # in new computational graph would have been caused by last op
        # in the old computational graph.
        copy_stack_trace(node.outputs, rval)

        # It might happen that the desired output of this node has a
        # broadcastable pattern that does not match that of 'rval'. This is
        # when originally, we were able to figure out that one of the
        # dimensions of the reshape is one, but some other transformation
        # replaced the shape by one for which this cannot be guessed.
        # We should try to figure out why we lost the information about this
        # constant value... but in the meantime, better not apply this
        # optimization.
        if rval.broadcastable == node.outputs[0].broadcastable:
            return [rval]
        else:
            return False

    return f


register_canonicalize(local_reshape_chain(Reshape), name="local_reshape_chain")


@register_useless
@register_canonicalize
@register_stabilize
@local_optimizer([Reshape])
def local_useless_reshape(fgraph, node):
    """
    Remove two kinds of useless reshape.

    Remove Reshape when both the input and output have a single dimension.
    Remove Reshape when reshaping to the shape of the input.

    """
    op = node.op
    if not isinstance(op, Reshape):
        return False

    input = node.inputs[0]
    output = node.outputs[0]
    output_shape = node.inputs[1]

    if input.ndim != output.ndim:
        return False

    # Simple case: both input and output have a single dimension.
    # This could hide errors if the user provides inconsistent shapes.
    if (
        input.ndim == 1
        and output.ndim == 1
        and input.broadcastable == output.broadcastable
    ):
        return [input]

    # Second case: all the shapes match the input shape
    # Match Reshape(x, x.shape)
    if output_shape.owner and isinstance(output_shape.owner.op, Shape):
        shape_input = output_shape.owner.inputs[0]
        if shape_input == input:
            return [input]

    # Match Reshape(x, [x.shape[0], ..., x.shape[-1]]), accounting for
    # broadcastable and constant dimensions
    if output_shape.owner and isinstance(output_shape.owner.op, MakeVector):
        output_shape_is = output_shape.owner.inputs

        shape_feature = getattr(fgraph, "shape_feature", None)

        nb_m1 = 0
        shape_match = [False] * input.ndim
        for dim in range(input.ndim):
            outshp_i = output_shape_is[dim]
            # Match Shape_i{dim}(input)
            if (
                outshp_i.owner
                and isinstance(outshp_i.owner.op, Shape_i)
                and outshp_i.owner.op.i == dim
                and outshp_i.owner.inputs[0] == input
            ):
                shape_match[dim] = True
                continue

            # Match Shape(input)[dim]
            if (
                outshp_i.owner
                and isinstance(outshp_i.owner.op, Subtensor)
                and len(outshp_i.owner.inputs) == 2
                and extract_constant(outshp_i.owner.inputs[1]) == dim
            ):
                subtensor_inp = outshp_i.owner.inputs[0]
                if subtensor_inp.owner and isinstance(subtensor_inp.owner.op, Shape):
                    shape_input_i = subtensor_inp.owner.inputs[0]
                    if shape_input_i == input:
                        shape_match[dim] = True
                        continue

            # Match 1 if input.broadcastable[dim] is True
            cst_outshp_i = extract_constant(outshp_i, only_process_constants=1)
            if input.broadcastable[dim] and cst_outshp_i == 1:
                shape_match[dim] = True
                continue

            # Match -1
            if cst_outshp_i == -1:
                shape_match[dim] = True
                nb_m1 += 1
                continue

            # Match shape_of[input][dim] or its constant equivalent
            if shape_feature:
                inpshp_i = shape_feature.get_shape(input, dim)
                if inpshp_i == outshp_i or (
                    extract_constant(inpshp_i, only_process_constants=1)
                    == extract_constant(outshp_i, only_process_constants=1)
                ):
                    shape_match[dim] = True
                    continue

        if all(shape_match) and nb_m1 <= 1:
            return [input]

        # TODO later: if all the shapes except one match, we may want to
        # consider it useless as well, like we do in the 1-dim case.


@register_canonicalize
@local_optimizer([Reshape])
def local_reshape_to_dimshuffle(fgraph, node):
    """
    Broadcastable dimensions in Reshape are replaced with dimshuffle.

    The goal is to avoid using reshape to add or remove broadcastable
    dimensions, but use dimshuffle instead, so dimshuffles can cancel out
    or be removed later on.

    For example:
        - reshape(x, (1, n)) --> dimshuffle{x,0}(reshape(x, (n,))
        - reshape(x, (1, m, 1, n, 1, 1))
          --> dimshuffle{x,0,x,1,x,x}(reshape(x, (m, n)))
    """
    op = node.op
    if not isinstance(op, Reshape):
        return False

    input = node.inputs[0]
    output = node.outputs[0]
    output_shape = node.inputs[1]

    dimshuffle_new_order = []
    new_output_shape = []
    index = 0  # index over the output of the new reshape
    for i in range(output.ndim):
        # Since output_shape is a symbolic vector, we trust extract_constant
        # to go through however it is formed to see if its i-th element is 1.
        # We need only_process_constants=False for that.
        dim = extract_constant(
            output_shape[i], only_process_constants=False, elemwise=False
        )
        if dim == 1:
            dimshuffle_new_order.append("x")
        else:
            dimshuffle_new_order.append(index)
            new_output_shape.append(dim)
            index = index + 1
    if index != output.ndim:
        inner = op.__class__(len(new_output_shape))(input, new_output_shape)
        copy_stack_trace(output, inner)
        new_node = [DimShuffle(inner.type.broadcastable, dimshuffle_new_order)(inner)]
        copy_stack_trace(output, new_node)
        return new_node


@register_canonicalize
@register_stabilize
@local_optimizer([Reshape])
def local_reshape_lift(fgraph, node):
    """
    Reshape(UnaryElemwise(x)) -> UnaryElemwise(Reshape(x))

    This optimization is needed by optimization
    log1msigm_to_softplus to get applied when there is a reshape.

    """
    if (
        isinstance(node.op, Reshape)
        and node.inputs[0].owner
        and isinstance(node.inputs[0].owner.op, Elemwise)
        and len(node.inputs[0].owner.inputs) == 1
    ):
        r = node.op(node.inputs[0].owner.inputs[0], node.inputs[1])
        # Copy stacktrace from previous Reshape op, as an error in new
        # Reshape op could only have been caused by old one.
        copy_stack_trace(node.outputs, r)

        e = node.inputs[0].owner.op(r)
        # Copy stacktrace from both previous Reshape and UnaryElemwise op
        # because an error in new cg could have been caused by either ops.
        copy_stack_trace(node.outputs + node.inputs, e)

        # In rare case the original broadcast was (False, True), but
        # the new one is (False, False). So don't crash in that case.
        if e.type != node.outputs[0].type:
            re = patternbroadcast(e, node.outputs[0].broadcastable)

            # Copy over stack trace.
            # If the graph fails it is usually due to the fact that a dimension
            # that should be broadcastable does not actually have length 1,
            copy_stack_trace(e, re)
        else:
            re = e

        return [re]


##################
# Middleman cuts #
##################

register_canonicalize(OpRemove(tensor_copy), name="remove_tensor_copy")


@local_optimizer(None)
def constant_folding(fgraph, node):
    for input in node.inputs:
        if not isinstance(input, Constant):
            return False
    # condition:  all inputs are constant
    if not node.op.do_constant_folding(fgraph, node):
        # The op asks not to be constant folded.
        return False

    storage_map = {i: [i.data] for i in node.inputs}
    compute_map = {i: [True] for i in node.inputs}
    for o in node.outputs:
        storage_map[o] = [None]
        compute_map[o] = [False]

    thunk = node.op.make_thunk(node, storage_map, compute_map, no_recycling=[])
    required = thunk()
    # A node whose inputs are all provided should always return successfully
    assert not required

    rval = []
    for output in node.outputs:
        assert compute_map[output][0], (output, storage_map[output][0])
        try:
            constant = output.type.Constant
        except AttributeError:
            constant = Constant

        v = constant(output.type, storage_map[output][0])
        copy_stack_trace(output, v)

        rval.append(v)
    return rval


topo_constant_folding = in2out(
    constant_folding, ignore_newtrees=True, name="topo_constant_folding"
)
register_canonicalize(topo_constant_folding, "fast_compile", final_opt=True)
register_uncanonicalize(topo_constant_folding, "fast_compile", final_opt=True)
register_stabilize(topo_constant_folding, "fast_compile", final_opt=True)
register_specialize(topo_constant_folding, "fast_compile", final_opt=True)


def local_elemwise_fusion_op(op_class, max_input_fct=lambda node: 32, maker=None):
    """Create a recursive function that fuses `Elemwise` `Op`s.

    The basic idea is that we loop through an `Elemwise` node's inputs, find
    other `Elemwise` nodes, determine the scalars input types for all of the
    `Elemwise` `Op`s, construct a new scalar `Op` using the scalar input types
    and each `Elemwise`'s scalar `Op`, and use the composite scalar `Op` in a
    new "fused" `Elemwise`.

    It's parameterized in order to work for `Elemwise` and `GpuElemwise` `Op`s.

    Parameters
    ----------
    op_class : type
        `GpuElemwise` or `Elemwise` class (the one that we want to fuse)
    max_input_fct : callable
        A function that returns the maximum number of inputs that this `Elemwise`
        can take (useful for `GpuElemwise`).  The GPU kernel currently has a
        limit of 256 bytes for the size of all parameters passed to it. As
        currently we pass a lot of information only by parameter, we must limit how
        many `Op`s we fuse together to avoid busting that 256 limit.

        On the CPU we limit to 32 input variables since that is the maximum
        NumPy support.

    maker: callable
        A function with the signature `(node, *args)` that constructs an
        `op_class` instance (e.g. `op_class(*args)`).

    """
    if maker is None:

        def maker(node, scalar_op):
            return op_class(scalar_op)

    def local_fuse(fgraph, node):
        """Fuse `Elemwise` `Op`s in a node.

        As part of specialization, we fuse two consecutive elemwise `Op`s of the
        same shape.

        For mixed dtype, we let the `Composite` `Op` do the cast. It lets the C
        compiler do the cast.

        The number of dimensions is validated at call time by Aesara itself.

        """
        # META TODO:  PUT THESE THINGS IN TRAC, NOT TODO NOTES!!
        # TODO: use broadcast flag?

        # TODO: don't do this optimization as a localOptimizer.
        # Analyze the graph in terms of elemwise subgraphs, and then
        # replace each subgraph with a Composite version.

        # TODO: use malloc and copy to transfer arguments that don't
        # fit within the parameter space of 256 bytes
        #
        # TODO: Merge with multiple output to merge when an inputs
        # have multiple clients. This can't be done with a local
        # optimiser.

        # TODO: Related: Support composites with multiple outputs

        # TODO: Use Composite to combine Elemwise and Reduce
        # operations.  We have to loop over the data anyway... might
        # as well sum it up while we're at it (this can be trickier
        # than i'm making it seound here. The data-traversal should be
        # done contiguously, and the summing-up might not be easy or
        # worthwhile if the summation axis doesn't line up with a
        # contiguous dimension)

        if type(node.op) is not op_class:
            return False

        if len(node.outputs) > 1:
            # We don't support fusion for nodes with multiple outputs.
            return

        inputs = []  # inputs of the new Elemwise op.
        s_inputs = []  # inputs of the new scalar op used by the Composite.
        # Inputs of the new scalar op that represents the current node.
        s_g = []

        # There is a hard limit of 256 bytes for the formal argument list to a
        # GPU kernel function.
        max_nb_input = max_input_fct(node)
        # The number of inputs to the new fused op if we do not fuse more
        # inputs.
        new_nb_input = len(node.inputs)
        # Did we fuse something?
        # Needed as we can fuse unary op that don't change the number of
        # inputs.
        # And there is a case where the inputs are the same as the current
        # node. That won't change the number of inputs of the new op.
        fused = False

        for i in node.inputs:
            do_fusion = False
            # Will store inputs of the fused node that are not currently inputs
            # of the node we want to create (to avoid duplicating inputs).
            tmp_input = []
            # Same as tmp_input, but for scalars.
            tmp_scalar = []

            # We should not check the number of inputs here
            # As fusing op don't always change the number of input.
            # If a variable is used as multiple into to the same node,
            # we still want to fusion. So we take the set.
            if (
                i.owner
                and isinstance(i.owner.op, op_class)
                and len({n for n, idx in fgraph.clients[i]}) == 1
                and
                # Do not merge elemwise that don't have the same
                # broadcastable pattern to don't redo duplicate
                # computation due to broadcast.
                i.owner.outputs[0].broadcastable == node.outputs[0].broadcastable
            ):
                try:
                    tmp_s_input = []
                    # we should not put duplicate input into s_inputs and inputs
                    for ii in i.owner.inputs:
                        if ii in inputs:
                            tmp_s_input.append(s_inputs[inputs.index(ii)])
                        elif ii in tmp_input:
                            tmp_s_input.append(tmp_scalar[tmp_input.index(ii)])
                        else:
                            tmp = aes.get_scalar_type(ii.dtype).make_variable()
                            try:
                                tv = get_test_value(ii)
                                if tv.size > 0:
                                    tmp.tag.test_value = tv.flatten()[0]
                                else:
                                    _logger.warning(
                                        "Cannot construct a scalar test value"
                                        " from a test value with no size: {}".format(ii)
                                    )
                            except TestValueError:
                                pass

                            tmp_s_input.append(tmp)
                            tmp_input.append(ii)
                            tmp_scalar.append(tmp_s_input[-1])

                    s_op = i.owner.op.scalar_op(*tmp_s_input, return_list=True)

                    # If the scalar_op doesn't have a C implementation, we skip
                    # its fusion to allow fusion of the other ops
                    i.owner.op.scalar_op.c_code(
                        s_op[0].owner,
                        "test_presence_of_c_code",
                        ["x" for x in i.owner.inputs],
                        ["z" for z in i.owner.outputs],
                        {"fail": "%(fail)s"},
                    )

                    do_fusion = True

                except (NotImplementedError, MethodNotDefined):
                    _logger.warning(
                        (
                            "Optimization Warning: "
                            f"The Op {i.owner.op.scalar_op} does not provide a C implementation."
                            " As well as being potentially slow, this also disables "
                            "loop fusion."
                        )
                    )
                    do_fusion = False

            # Compute the number of inputs in case we fuse this input.
            # We subtract 1 because we replace the existing input with the new
            # inputs from `tmp_input`.
            new_nb_input_ = new_nb_input + len(tmp_input) - 1

            # If the new input is already an input of the current node, it was
            # already counted when `new_nb_input` was initialized to
            # len(node.inputs).
            # This can happen when a variable is used both by the Elemwise to
            # fuse and the current node.
            for x in tmp_input:
                if x in node.inputs:
                    new_nb_input_ -= 1

            if do_fusion and (new_nb_input_ <= max_nb_input):
                fused = True
                new_nb_input = new_nb_input_
                inputs.extend(tmp_input)
                s_inputs.extend(tmp_scalar)
                s_g.extend(s_op)
            else:
                # We must support the case where the same variable appears many
                # times within the inputs
                if inputs.count(i) == node.inputs.count(i):
                    s = s_inputs[inputs.index(i)]
                else:
                    s = aes.get_scalar_type(i.dtype).make_variable()
                    try:
                        if config.compute_test_value != "off":
                            v = get_test_value(i)
                            if v.size > 0:
                                s.tag.test_value = v.flatten()[0]
                    except TestValueError:
                        pass

                    inputs.append(i)
                    s_inputs.append(s)
                s_g.append(s)

        if not fused:
            return False

        if new_nb_input != len(inputs) or len(s_inputs) != len(inputs):
            raise Exception(
                """Something has gone wrong with the elemwise
fusion optimization. We skip this optimization. You can ignore this message,
your code will run correctly, but may be slower."""
            )

        s_new_out = node.op.scalar_op(*s_g, return_list=True)
        try:
            s_new_out[0].owner.op.c_code(
                s_new_out[0].owner,
                "test_presence_of_c_code",
                ["x" for x in s_g],
                ["z" for x in s_new_out],
                {"fail": "%(fail)s"},
            )
        except (NotImplementedError, MethodNotDefined):
            name = str(s_new_out[0].owner.op)
            _logger.warning(
                (
                    "Optimization Warning: "
                    f"The Op {name} does not provide a C implementation."
                    " As well as being potentially slow, this also disables "
                    "loop fusion."
                )
            )
            return False

        # create the composite op.
        composite_op = aes.Composite(s_inputs, s_new_out)

        # create the new node.
        # Do not call make_node to have test_value
        new_node = maker(node, composite_op)(*inputs).owner

        assert len(new_node.outputs) == 1
        assert node.outputs[0].dtype == new_node.outputs[0].dtype

        if len(new_node.inputs) > max_nb_input:
            _logger.warning(
                "loop fusion failed because Op would exceed" " kernel argument limit."
            )
            return False

        # we fuse as many that we can at the same time to make debug mode faster
        # debug mode will be faster as it won't test all intermediate step.
        while True:
            ret = local_fuse(fgraph, new_node)
            if ret is not False and ret is not None:
                assert len(ret) == len(new_node.outputs)
                assert len(ret) == 1
                new_node = ret[0].owner
            else:
                break

        return new_node.outputs

    return local_fuse


def elemwise_max_input_fct(node):
    # `Elemwise.perform` uses NumPy ufuncs and they are limited to 31 inputs.
    if not config.cxx:
        return 31
    return 1024


local_elemwise_fusion = local_elemwise_fusion_op(Elemwise, elemwise_max_input_fct)


class FusionOptimizer(GlobalOptimizer):
    """Graph optimizer that simply runs local fusion operations.

    TODO: This is basically a `EquilibriumOptimizer`; we should just use that.

    """

    def __init__(self, local_optimizer):
        super().__init__()
        self.optimizer = local_optimizer

    def add_requirements(self, fgraph):
        fgraph.attach_feature(toolbox.ReplaceValidate())

    def apply(self, fgraph):
        did_something = True
        nb_iter = 0
        nb_replacement = 0
        nb_inconsistency_replace = 0
        time_toposort = 0
        if fgraph.profile:
            validate_before = fgraph.profile.validate_time
            callbacks_before = fgraph.execute_callbacks_times.copy()
            callback_before = fgraph.execute_callbacks_time
        while did_something:
            t0 = time.time()
            nodelist = list(fgraph.toposort())
            time_toposort += time.time() - t0
            nodelist.reverse()
            did_something = False
            for node in nodelist:
                # Don't try to fuse node that have already been fused.
                if node in fgraph.apply_nodes:
                    new_outputs = self.optimizer(fgraph, node)
                    if new_outputs:
                        assert len(new_outputs) == len(node.outputs)
                        try:
                            fgraph.replace_all_validate(
                                list(zip(node.outputs, new_outputs)),
                                reason=self.__class__.__name__,
                            )
                            did_something = True
                            nb_replacement += 1
                        except InconsistencyError:
                            nb_inconsistency_replace += 1
            nb_iter += 1

        if fgraph.profile:
            validate_time = fgraph.profile.validate_time - validate_before
            callback_time = fgraph.execute_callbacks_time - callback_before
            callbacks_time = {}
            for k, v in fgraph.execute_callbacks_times.items():
                if k in callbacks_before:
                    callbacks_time[k] = v - callbacks_before[k]
                else:
                    callbacks_time[k] = v
        else:
            validate_time = None
            callback_time = None
            callbacks_time = {}
        return (
            self,
            nb_iter,
            nb_replacement,
            nb_inconsistency_replace,
            validate_time,
            callback_time,
            callbacks_time,
            time_toposort,
        )

    @staticmethod
    def print_profile(stream, prof, level=0):
        blanc = "    " * level
        print(blanc, "FusionOptimizer", file=stream)
        print(blanc, " nb_iter", prof[1], file=stream)
        print(blanc, " nb_replacement", prof[2], file=stream)
        print(blanc, " nb_inconsistency_replace", prof[3], file=stream)
        print(blanc, " validate_time", prof[4], file=stream)
        print(blanc, " callback_time", prof[5], file=stream)
        if prof[5] > 1:
            print(blanc, " callbacks_time", file=stream)
            for i in sorted(prof[6].items(), key=lambda a: a[1])[::-1]:
                if i[1] > 0:
                    print(blanc, "     ", i)
        print(blanc, " time_toposort", prof[7], file=stream)


if config.tensor__local_elemwise_fusion:
    _logger.debug("Enabling Elemwise fusion optimizations in fast_run")
    # Must be after gpu(48.5) and before AddDestroyHandler(49.5)
    fuse_seqopt = SequenceDB()
    fuse_seqopt.register(
        "composite_elemwise_fusion",
        FusionOptimizer(local_elemwise_fusion),
        1,
        "fast_run",
        "fusion",
    )
    compile.optdb.register(
        "elemwise_fusion",
        fuse_seqopt,
        49,
        "fast_run",
        "fusion",
        "local_elemwise_fusion",
        "FusionOptimizer",
    )
else:
    _logger.debug("not enabling optimization fusion elemwise in fast_run")
    compile.optdb.register(
        "elemwise_fusion",
        FusionOptimizer(local_elemwise_fusion),
        49,
        "fusion",
        "local_elemwise_fusion",
        "FusionOptimizer",
    )


@register_canonicalize
@local_optimizer([Elemwise])
def local_useless_composite(fgraph, node):
    """For elemwise Composite that have multiple outputs, remove the
    outputs that are not used.

    """
    if not isinstance(node.op, Elemwise) or not isinstance(
        node.op.scalar_op, aes.Composite
    ):
        return
    comp = node.op.scalar_op
    idx = [i for i, o_extern in enumerate(node.outputs) if fgraph.clients[o_extern]]
    if len(idx) < len(node.outputs):
        new_outputs = [comp.outputs[i] for i in idx]
        c = aes.Composite(inputs=comp.inputs, outputs=new_outputs)
        e = Elemwise(scalar_op=c)(*node.inputs, return_list=True)
        return dict(zip([node.outputs[i] for i in idx], e))


# ############################
# # Remove consider_constant #
# ############################


# Although the ops ConsiderConstant, ZeroGrad and DisconnectedGrad
# just returns the input, it should be removed from the graph to
@register_canonicalize("fast_compile")
@register_useless("fast_compile")
@local_optimizer(None)
def local_view_op(fgraph, node):
    if isinstance(node.op, ViewOp):
        return node.inputs


@register_useless
@register_canonicalize
@register_stabilize
@register_specialize
@local_optimizer([Alloc])
def local_merge_alloc(fgraph, node):
    # This opt takes care of several cases:
    # Alloc(Alloc(m, x, 1, 1, 1), x, y, z, w) -> Alloc(m, x, y, z, w)
    # Alloc(Alloc(m, y, 1, 1), x, y, z, w) -> Alloc(m, x, y, z, w)
    # Alloc(Alloc(m, y1, 1, 1), x, y2, z, w) -> Alloc(m, x, assert(y1, y1==y2), z, w)
    if not isinstance(node.op, Alloc):
        return False
    if not node.inputs[0].owner or not isinstance(node.inputs[0].owner.op, Alloc):
        return False
    inputs_outer = node.inputs
    inputs_inner = node.inputs[0].owner.inputs
    dims_outer = inputs_outer[1:]
    dims_inner = inputs_inner[1:]
    dims_outer_rev = dims_outer[::-1]
    dims_inner_rev = dims_inner[::-1]
    # check if the pattern of broadcasting is matched, in the reversed ordering.
    # The reverse ordering is needed when an Alloc add an implicit new
    # broadcasted dimensions to its inputs[0]. Eg:
    # Alloc(Alloc(m, y, 1, 1), x, y, z, w) -> Alloc(m, x, y, z, w)
    i = 0
    for dim_inner, dim_outer in zip(dims_inner_rev, dims_outer_rev):
        if dim_inner != dim_outer:
            if isinstance(dim_inner, Constant) and dim_inner.data == 1:
                pass
            else:
                dims_outer[-1 - i] = Assert(
                    "You have a shape error in your graph. To see a better"
                    " error message and a stack trace of where in your code"
                    " the error is created, use the Aesara flags"
                    " optimizer=None or optimizer=fast_compile."
                )(dim_outer, eq(dim_outer, dim_inner))
        i += 1
    return [alloc(inputs_inner[0], *dims_outer)]


@register_useless("fast_compile")
@local_optimizer([TopKOp])
def local_useless_topk(fgraph, node):
    """
    TopKOp generates two outputs by default
    This opt removes the useless ones

    """
    op = node.op
    if not isinstance(op, TopKOp):
        return
    if not (op.return_values and op.return_indices):
        return False

    x, k = node.inputs
    ret_val = bool(fgraph.clients[node.outputs[0]])
    ret_idx = bool(fgraph.clients[node.outputs[1]])

    if not (ret_val ^ ret_idx):
        # both true -> nothing to remove
        # both false -> let pruner handle
        return False

    old_output = node.outputs[ret_idx]
    new_output = TopKOp(
        axis=op.axis,
        sorted=op.sorted,
        idx_dtype=op.idx_dtype,
        return_values=ret_val,
        return_indices=ret_idx,
    )(x, k)
    copy_stack_trace(node.outputs[0], new_output)
    return {old_output: new_output}
