import copy
import logging
import pdb
import sys
import time
from collections import Counter

import numpy as np

import aesara
import aesara.tensor.nlinalg as nlinalg
import aesara.tensor.signal.pool as pool
import aesara.tensor.slinalg as slinalg
from aesara import scalar as aes
from aesara import tensor as aet
from aesara.assert_op import Assert
from aesara.breakpoint import PdbBreakpoint
from aesara.compile import optdb
from aesara.configdefaults import config
from aesara.gpuarray.basic_ops import (
    GpuAlloc,
    GpuAllocEmpty,
    GpuContiguous,
    GpuEye,
    GpuFromHost,
    GpuJoin,
    GpuReshape,
    GpuSplit,
    GpuToGpu,
    GpuTri,
    HostFromGpu,
    as_gpuarray_variable,
    gpu_contiguous,
    gpu_join,
    host_from_gpu,
    infer_context_name,
)
from aesara.gpuarray.blas import (
    GpuCorr3dMM,
    GpuCorr3dMM_gradInputs,
    GpuCorr3dMM_gradWeights,
    GpuCorrMM,
    GpuCorrMM_gradInputs,
    GpuCorrMM_gradWeights,
    GpuGemm,
    GpuGemmBatch,
    GpuGer,
    gpu_dot22,
    gpugemm_inplace,
    gpugemm_no_inplace,
    gpugemmbatch_no_inplace,
    gpugemv_inplace,
    gpugemv_no_inplace,
)
from aesara.gpuarray.blocksparse import (
    GpuSparseBlockGemv,
    GpuSparseBlockOuter,
    gpu_sparse_block_gemv,
    gpu_sparse_block_gemv_inplace,
    gpu_sparse_block_outer,
    gpu_sparse_block_outer_inplace,
)
from aesara.gpuarray.ctc import GpuConnectionistTemporalClassification
from aesara.gpuarray.dnn_opt import (
    local_abstractconv3d_cudnn_alt,
    local_abstractconv_cudnn,
    local_abstractconv_cudnn_alt,
    local_abstractconv_gi_cudnn,
    local_abstractconv_gw_cudnn,
)
from aesara.gpuarray.elemwise import (
    GpuCAReduceCPY,
    GpuCAReduceCuda,
    GpuDimShuffle,
    GpuElemwise,
    gpu_erfcinv,
    gpu_erfinv,
    max_inputs_to_GpuElemwise,
)
from aesara.gpuarray.linalg import (
    MATRIX_STRUCTURES_SOLVE,
    GpuCholesky,
    GpuCublasTriangularSolve,
    GpuCusolverSolve,
    GpuMagmaCholesky,
    GpuMagmaEigh,
    GpuMagmaMatrixInverse,
    cublas_available,
    cusolver_available,
    gpu_qr,
    gpu_svd,
)
from aesara.gpuarray.neighbours import GpuImages2Neibs
from aesara.gpuarray.nnet import (
    gpu_crossentropy_softmax_1hot_with_bias_dx,
    gpu_crossentropy_softmax_argmax_1hot_with_bias,
    gpu_softmax,
    gpu_softmax_with_bias,
)
from aesara.gpuarray.opt_util import (
    alpha_merge,
    op_lifter,
    output_merge,
    pad_dims,
    safe_to_cpu,
    safe_to_gpu,
    unpad_dims,
)
from aesara.gpuarray.optdb import (
    GraphToGPUDB,
    abstract_batch_norm_db,
    abstract_batch_norm_db2,
    abstract_batch_norm_groupopt,
    abstractconv_groupopt,
    gpu_cut_copies,
    gpu_optimizer,
    gpu_seqopt,
    matrix_ops_db,
    matrix_ops_db2,
    pool_db,
    pool_db2,
    register_inplace,
    register_opt,
    register_opt2,
)
from aesara.gpuarray.pool import (
    GpuAveragePoolGrad,
    GpuDownsampleFactorMaxGradGrad,
    GpuMaxPoolGrad,
    GpuMaxPoolRop,
    GpuPool,
)
from aesara.gpuarray.reduction import GpuMaxAndArgmax
from aesara.gpuarray.subtensor import (
    GpuAdvancedIncSubtensor,
    GpuAdvancedIncSubtensor1,
    GpuAdvancedIncSubtensor1_dev20,
    GpuAdvancedSubtensor,
    GpuAdvancedSubtensor1,
    GpuAllocDiag,
    GpuExtractDiag,
    GpuIncSubtensor,
    GpuSubtensor,
)
from aesara.gpuarray.type import (
    ContextNotDefined,
    GpuArrayConstant,
    GpuArrayType,
    get_context,
    move_to_gpu,
)
from aesara.graph import toolbox
from aesara.graph.basic import Constant, Variable, applys_between, clone_replace
from aesara.graph.fg import FunctionGraph
from aesara.graph.opt import (
    GlobalOptimizer,
    LocalMetaOptimizer,
    copy_stack_trace,
    inherit_stack_trace,
    local_optimizer,
)
from aesara.ifelse import IfElse
from aesara.link.c.basic import CLinker
from aesara.misc.ordered_set import OrderedSet
from aesara.scalar.basic import Cast, Pow, Scalar, log, neg, true_div
from aesara.scalar.basic_scipy import Erfcinv, Erfinv
from aesara.scan.op import Scan
from aesara.scan.opt import ScanInplaceOptimizer
from aesara.tensor.basic import (
    Alloc,
    AllocDiag,
    AllocEmpty,
    ExtractDiag,
    Eye,
    Flatten,
    Join,
    Rebroadcast,
    Split,
    Tri,
)
from aesara.tensor.math import MaxAndArgmax
from aesara.tensor.nnet import batchnorm, conv3d2d
from aesara.tensor.nnet.abstract_conv import (
    AbstractConv2d,
    AbstractConv2d_gradInputs,
    AbstractConv2d_gradWeights,
    AbstractConv3d,
    AbstractConv3d_gradInputs,
    AbstractConv3d_gradWeights,
    BaseAbstractConv,
    get_conv_output_shape,
)
from aesara.tensor.nnet.blocksparse import SparseBlockGemv, SparseBlockOuter
from aesara.tensor.nnet.conv import ConvOp
from aesara.tensor.nnet.ctc import ConnectionistTemporalClassification
from aesara.tensor.nnet.neighbours import Images2Neibs
from aesara.tensor.shape import Reshape, Shape, SpecifyShape, shape_i, specify_shape
from aesara.tensor.type import TensorType


_logger = logging.getLogger("aesara.gpuarray.opt")


gpu_seqopt.register(
    "gpuarray_graph_optimization",
    GraphToGPUDB(),
    -0.5,
    "fast_compile",
    "fast_run",
    "gpuarray",
)

gpu_seqopt.register(
    "gpuarray_local_optimizations",
    gpu_optimizer,
    1,
    "fast_compile",
    "fast_run",
    "gpuarray",
    "gpuarray_local_optimiziations",
)
gpu_seqopt.register(
    "gpuarray_cut_transfers", gpu_cut_copies, 2, "fast_compile", "fast_run", "gpuarray"
)

register_opt("fast_compile")(aesara.tensor.basic_opt.local_track_shape_i)
register_opt(final_opt=True, name="gpua_constant_folding")(
    aesara.tensor.basic_opt.constant_folding
)
gpu_optimizer.register(
    "local_remove_all_assert", aesara.tensor.basic_opt.local_remove_all_assert, "unsafe"
)


gpu_log = GpuElemwise(log)
gpu_neg = GpuElemwise(neg)
gpu_true_div = GpuElemwise(true_div)


class InputToGpuOptimizer(GlobalOptimizer):
    """
    Transfer the input to the gpu to start the rolling wave.

    """

    def add_requirements(self, fgraph):
        fgraph.attach_feature(toolbox.ReplaceValidate())

    def apply(self, fgraph):
        for input in fgraph.inputs:
            if isinstance(input.type, GpuArrayType):
                continue

            # If all clients are outputs or transfers don't do anything.
            if all(
                cl[0] == "output" or isinstance(cl[0].op, GpuFromHost)
                for cl in fgraph.clients[input]
            ):
                continue

            target = getattr(input.tag, "target", None)
            if target == "cpu":
                continue
            if isinstance(input.type, TensorType) and not move_to_gpu(input):
                continue

            try:
                new_input = GpuFromHost(target)(input).transfer("cpu")
                fgraph.replace_validate(input, new_input, "InputToGpuOptimizer")
            except TypeError:
                # This could fail if the inputs are not TensorTypes
                pass
            except ContextNotDefined:
                if hasattr(input.tag, "target"):
                    raise
                # If there is no context tag and no default context
                # then it stays on the CPU


gpu_seqopt.register(
    "InputToGpuArrayOptimizer",
    InputToGpuOptimizer(),
    0,
    "fast_run",
    "fast_compile",
    "merge",
)


class GraphToGPU(GlobalOptimizer):
    """
    Transfer the graph as a whole to GPU instead of transferring node by node.

    Parameters
    ----------
    local_optimizers_all : List or SortedSet
        The local optimizations to apply to a node.
    local_optimizers_map : Dict
        Dictionary object containing the mapping of Op to list of
        LocalOptimizers.
    """

    def __init__(self, local_optimizers_all, local_optimizers_map):
        self.local_optimizers_all = local_optimizers_all
        self.local_optimizers_map = local_optimizers_map

    def add_requirements(self, fgraph):
        fgraph.attach_feature(toolbox.ReplaceValidate())

    def apply(self, fgraph):
        mapping = {}
        time_opts = {}
        node_created = {}
        process_count = {}
        t_topo = time.time()
        topo = fgraph.toposort()
        time_topo = time.time()
        toposort_timing = time_topo - t_topo

        # Building a new graph
        # Iterating through inputs of graph
        target = infer_context_name(*fgraph.inputs)
        for i in fgraph.inputs:
            if isinstance(i.type, TensorType) and move_to_gpu(i):
                mapping[i] = i.transfer(getattr(i.tag, "target", target))
            else:
                mapping[i] = i
        for i in fgraph.variables:
            if isinstance(i, Constant):
                mapping[i] = i
        for node in topo:
            for lopt in (
                self.local_optimizers_map.get(node.op, [])
                + self.local_optimizers_map.get(type(node.op), [])
                + self.local_optimizers_all
            ):
                process_count.setdefault(lopt, 0)
                time_opts.setdefault(lopt, 0)
                node_created.setdefault(lopt, 0)

        for node in topo:

            if isinstance(node.op, HostFromGpu):
                mapping[node.outputs[0]] = mapping[node.inputs[0]]
                continue

            # Move only if any of the inputs are on the GPU.
            move_to_GPU = False

            context_name = None
            for i in [mapping[i] for i in node.inputs]:
                if isinstance(i.type, GpuArrayType):
                    context_name = i.type.context_name
                    move_to_GPU = True
                    break
            if not move_to_GPU and isinstance(
                node.op,
                (
                    Alloc,
                    AllocEmpty,
                    Eye,
                    Tri,
                ),
            ):
                # If the Alloc[Empty] have a client that will be moved
                # to the GPU, we should move the Alloc* on the GPU.

                # We approximate this by supposing that if we have an
                # optimization for one of the clients op, then we will
                # move the client to the GPU.
                for c, _ in fgraph.clients[node.outputs[0]]:
                    if c != "output" and (
                        self.local_optimizers_map.get(c.op, [])
                        + self.local_optimizers_map.get(type(c.op), [])
                    ):
                        move_to_GPU = True
            new_ops = None
            if move_to_GPU and any(
                ["complex" in getattr(i, "dtype", "") for i in node.inputs]
            ):
                move_to_GPU = False

            # Apply the lifter
            if move_to_GPU:
                for lopt in (
                    self.local_optimizers_map.get(node.op, [])
                    + self.local_optimizers_map.get(type(node.op), [])
                    + self.local_optimizers_all
                ):
                    t_opt = time.time()
                    new_ops = lopt.transform(
                        node.op,
                        context_name,
                        [mapping[i] for i in node.inputs],
                        node.outputs,
                    )
                    t_opt2 = time.time()
                    time_opts[lopt] += t_opt2 - t_opt

                    if new_ops:
                        process_count[lopt] += 1
                        break
            outputs = []

            if isinstance(new_ops, aesara.graph.op.Op):
                with inherit_stack_trace(node.outputs):
                    outputs = new_ops(
                        *[mapping[i] for i in node.inputs], return_list=True
                    )
            elif not new_ops:
                newnode = node.clone_with_new_inputs(
                    [mapping.get(i) for i in node.inputs]
                )
                outputs = newnode.outputs
            elif isinstance(new_ops, (tuple, list)):
                outputs = new_ops
            elif isinstance(new_ops, Variable):
                outputs = [new_ops]

            for old_output, new_output in zip(node.outputs, outputs):
                copy_stack_trace(old_output, new_output)

            if new_ops:
                node_created[lopt] += len(
                    applys_between([mapping[i] for i in node.inputs], outputs)
                )
                if any(
                    [
                        getattr(old_o, "dtype", None) != getattr(new_o, "dtype", None)
                        for old_o, new_o in zip(outputs, node.outputs)
                    ]
                ):
                    _logger.warning(
                        f"The optimization {lopt} returned bad dtype. Skipping it."
                        " Write to aesara-dev mailing list about this."
                    )
                    newnode = node.clone_with_new_inputs(
                        [mapping.get(i) for i in node.inputs]
                    )
                    outputs = newnode.outputs

            for new_o, old_o in zip(outputs, node.outputs):
                assert len(outputs) == len(node.outputs)
                mapping[old_o] = new_o

        new_nodes = []
        for o in fgraph.outputs:
            new_o = mapping[o]
            if new_o.type != o.type:
                assert isinstance(o.type, TensorType)
                assert isinstance(new_o.type, GpuArrayType)

                # This condition is needed in the case one input is an
                # output of the graph. Without this, it would
                # introduce cycle as we don't replace correctly that
                # case. It would also add extra transfer to/from the
                # gpu.
                if (
                    new_o.owner
                    and isinstance(new_o.owner.op, GpuFromHost)
                    and new_o.owner.inputs[0].type == o.type
                ):
                    new_o = new_o.owner.inputs[0]
                else:
                    new_o = copy_stack_trace(o, safe_to_cpu(new_o))
            new_nodes.append(new_o)
        fgraph.replace_all_validate(
            zip(fgraph.outputs, new_nodes), reason=self.__class__.__name__
        )

        return (self, toposort_timing, time_opts, node_created, process_count)

    @staticmethod
    def print_profile(stream, prof, level=0):
        (opt, toposort_timing, time_opts, node_created, process_count) = prof
        blanc = "    " * level
        print(blanc, "GraphToGPUOptimizer", end=" ", file=stream)

        print(blanc, getattr(opt, "name", getattr(opt, "__name__", "")), file=stream)

        print(blanc, f"  time io_toposort {toposort_timing:.3f}s", file=stream)

        s = sum(time_opts.values())
        print(blanc, f"Total time taken by local optimizers {s:.3f}s ", file=stream)

        count_opt = []
        not_used = []
        not_used_time = 0

        for o, count in process_count.items():
            if count > 0:
                count_opt.append((time_opts[o], count, node_created[o], o))
            else:
                not_used.append((time_opts[o], o))
                not_used_time += time_opts[o]

        if count_opt:
            print(blanc, "  times - times applied - Node created - name:", file=stream)
            count_opt.sort()
            for (t, count, n_created, o) in count_opt[::-1]:
                print(
                    blanc,
                    f"  {t:.3f}s - {int(count)} - {int(n_created)} - {o}",
                    file=stream,
                )
            print(
                blanc,
                f"  {not_used_time:.3f}s - in {len(not_used)} optimization that were not used (display only those with a runtime > 0)",
                file=stream,
            )
            not_used.sort(key=lambda nu: (nu[0], str(nu[1])))
            for (t, o) in not_used[::-1]:
                if t > 0:
                    # Skip opt that have 0 times, they probably wasn't even tried.
                    print(blanc + "  ", f"  {t:.3f}s - {o}", file=stream)
            print(file=stream)

    @staticmethod
    def merge_profile(prof1, prof2):
        # (opt, toposort_timing, time_opts, node_created, process_count) = prof1
        local_optimizers = OrderedSet(prof1[0].local_optimizers_all).union(
            prof2[0].local_optimizers_all
        )

        def merge_dict(d1, d2):
            """
            merge 2 dicts by adding the values.
            """
            d = d1.copy()
            for k, v in d2.items():
                if k in d:
                    d[k] += v
                else:
                    d[k] = v
            return d

        local_optimizers_map = merge_dict(
            prof1[0].local_optimizers_map, prof2[0].local_optimizers_map
        )
        new_opt = GraphToGPU(local_optimizers, local_optimizers_map)

        toposort_timing = prof1[1] + prof2[1]
        time_opts = merge_dict(prof1[2], prof2[2])
        node_created = merge_dict(prof1[3], prof2[3])
        process_count = merge_dict(prof1[4], prof2[4])
        return (new_opt, toposort_timing, time_opts, node_created, process_count)

    def print_summary(self, stream=sys.stdout, level=0, depth=-1):
        print(
            f"{' ' * level}{self.__class__.__name__} ({id(self)})",
            file=stream,
        )
        if depth != 0:
            map_values = []
            for opts in self.local_optimizers_map.values():
                map_values += opts
            for opt in self.local_optimizers_all + map_values:
                opt.print_summary(stream, level=(level + 2), depth=(depth - 1))


@local_optimizer([GpuFromHost, GpuToGpu, HostFromGpu])
def local_cut_gpu_transfers(fgraph, node):
    # gpu[ab] -> host -> gpub
    if (
        isinstance(node.op, GpuFromHost)
        and node.inputs[0].owner
        and isinstance(node.inputs[0].owner.op, HostFromGpu)
    ):
        other = node.inputs[0].owner.inputs[0]
        if node.op.context_name == other.type.context_name:
            return [other]
        else:
            return [GpuToGpu(node.op.context_name)(other)]

    # ? -> gpua -> host
    elif isinstance(node.op, HostFromGpu) and node.inputs[0].owner:
        n2 = node.inputs[0].owner

        # host ->
        if isinstance(n2.op, GpuFromHost):
            return [n2.inputs[0]]

        # gpub ->
        if isinstance(n2.op, GpuToGpu):
            return [n2.inputs[0].transfer("cpu")]

    # ? -> gpua -> gpub
    elif isinstance(node.op, GpuToGpu):
        # Transfer within same context
        if node.inputs[0].type.context_name == node.op.context_name:
            return [node.inputs[0]]

        if node.inputs[0].owner:
            n2 = node.inputs[0].owner

            # host ->
            if isinstance(n2.op, GpuFromHost):
                return [as_gpuarray_variable(n2.inputs[0], node.op.context_name)]

            # gpuc ->
            if isinstance(n2.op, GpuToGpu):
                if node.op.context_name == n2.inputs[0].type.context_name:
                    return [n2.inputs[0]]
                else:
                    return [node.op(n2.inputs[0])]


gpu_cut_copies.register(
    "cut_gpua_host_transfers",
    local_cut_gpu_transfers,
    "fast_compile",
    "fast_run",
    "gpuarray",
)
gpu_cut_copies.register(
    "cut_gpua_constant_transfers",
    aesara.tensor.basic_opt.constant_folding,
    "fast_compile",
    "fast_run",
    "gpuarray",
)
optdb["canonicalize"].register(
    "local_cut_gpua_host_gpua",
    local_cut_gpu_transfers,
    "fast_compile",
    "fast_run",
    "gpuarray",
)


@register_opt("fast_compile")
@local_optimizer([Alloc])
def local_gpua_alloc2(fgraph, node):
    """
    Join(axis, {Alloc or HostFromGPU}, ...) -> Join(axis, GpuAlloc, Alloc, ...)

    Moves an alloc that is an input to join to the gpu.

    """
    try:
        get_context(None)
    except ContextNotDefined:
        # If there is no default context then we do not perform the move here.
        return
    if isinstance(node.op, Alloc) and all(
        c != "output"
        and isinstance(c.op, Join)
        and all(
            i.owner and i.owner.op in [host_from_gpu, aet.alloc] for i in c.inputs[1:]
        )
        for c, idx in fgraph.clients[node.outputs[0]]
    ):
        return [GpuAlloc(None)(*node.inputs).transfer("cpu")]


@register_opt("fast_compile")
@op_lifter([Alloc])
@register_opt2([Alloc], "fast_compile")
def local_gpuaalloc(fgraph, op, context_name, inputs, outputs):
    return GpuAlloc(context_name)(*inputs)


@register_opt("fast_compile")
@op_lifter([AllocEmpty])
@register_opt2([AllocEmpty], "fast_compile")
def local_gpua_alloc_empty(fgraph, op, context_name, inputs, outputs):
    # We use _props_dict() to make sure that the GPU op know all the
    # CPU op props.
    return GpuAllocEmpty(context_name=context_name, **op._props_dict())(*inputs)


@register_opt()
@local_optimizer([GpuAlloc])
def local_gpualloc_memset_0(fgraph, node):
    if isinstance(node.op, GpuAlloc) and not node.op.memset_0:
        inp = node.inputs[0]
        if (
            isinstance(inp, GpuArrayConstant)
            and inp.data.size == 1
            and (np.asarray(inp.data) == 0).all()
        ):
            new_op = GpuAlloc(node.op.context_name, memset_0=True)
            with inherit_stack_trace(node.outputs):
                return new_op(*node.inputs, return_list=True)


# Don't register by default.
@local_optimizer([GpuAllocEmpty])
def local_gpua_alloc_empty_to_zeros(fgraph, node):
    if isinstance(node.op, GpuAllocEmpty):
        context_name = infer_context_name(*node.inputs)
        z = np.asarray(0, dtype=node.outputs[0].dtype)
        with inherit_stack_trace(node.outputs):
            return [
                GpuAlloc(context_name)(
                    as_gpuarray_variable(z, context_name), *node.inputs
                )
            ]


optdb.register(
    "local_gpua_alloc_empty_to_zeros",
    aesara.graph.opt.in2out(local_gpua_alloc_empty_to_zeros),
    # After move to gpu and merge2, before inplace.
    49.3,
    "alloc_empty_to_zeros",
)


@register_opt()
@local_optimizer([GpuContiguous])
def local_gpu_contiguous_gpu_contiguous(fgraph, node):
    """
    gpu_contiguous(gpu_contiguous(x)) -> gpu_contiguous(x)

    """
    if isinstance(node.op, GpuContiguous):
        inp = node.inputs[0]
        if inp.owner and isinstance(inp.owner.op, GpuContiguous):
            return [inp]


@register_opt("fast_compile")
@op_lifter([aesara.tensor.extra_ops.CpuContiguous])
@register_opt2([aesara.tensor.extra_ops.CpuContiguous], "fast_compile")
def local_gpua_contiguous(fgraph, op, context_name, inputs, outputs):
    return gpu_contiguous


@register_opt("fast_compile")
@op_lifter([Reshape])
@register_opt2([Reshape], "fast_compile")
def local_gpua_reshape(fgraph, op, context_name, inputs, outputs):
    res = GpuReshape(op.ndim)
    return res


@register_opt("fast_compile")
@op_lifter([Rebroadcast])
@register_opt2([Rebroadcast], "fast_compile")
def local_gpua_rebroadcast(fgraph, op, context_name, inputs, outputs):
    return op(as_gpuarray_variable(inputs[0], context_name))


@register_opt("fast_compile")
@op_lifter([Flatten])
@register_opt2([Flatten], "fast_compile")
def local_gpua_flatten(fgraph, op, context_name, inputs, outputs):
    shp = []
    if op.outdim != 1:
        shp = [inputs[0].shape[i] for i in range(op.outdim - 1)]
    shp += [-1]
    res = GpuReshape(op.outdim)
    o = res(inputs[0], aet.as_tensor_variable(shp))
    return o


@register_opt("fast_compile")
@op_lifter([aesara.tensor.elemwise.Elemwise])
@register_opt2([aesara.tensor.elemwise.Elemwise], "fast_compile")
def local_gpua_elemwise(fgraph, op, context_name, inputs, outputs):
    scal_op = op.scalar_op
    name = op.name
    if name:
        name = "Gpu" + name
    if len(outputs) > 1:
        return

    have_cuda = False
    have_opencl = False
    if inputs and isinstance(inputs[0].type, GpuArrayType):
        kind = inputs[0].type.context.kind
        if kind.startswith(b"opencl"):
            have_opencl = True
        elif kind.startswith(b"cuda"):
            have_cuda = True
    convert = {Erfinv: gpu_erfinv, Erfcinv: gpu_erfcinv}

    if scal_op.__class__ in convert:
        scal_op = convert[scal_op.__class__]
        if have_opencl:
            _logger.warning(
                f'Function "{scal_op}" is not supported with OpenCL. Use "device=cuda" instead.'
            )
        if not have_cuda:
            return None
    if not scal_op.supports_c_code(inputs, outputs):
        return
    res = GpuElemwise(
        scal_op,
        name=name,
        inplace_pattern=copy.copy(op.inplace_pattern),
        nfunc_spec=op.nfunc_spec,
    )

    # If the elemwise operation is a pow, casts might be required on the
    # inputs and or outputs because only the (float, float)->float and
    # (double, double)->double cases are implemented at the moment.
    if isinstance(op.scalar_op, Pow):

        # Only transfer the computation on the gpu if the output dtype is
        # floating point. Else, give up on the transfer to the gpu.
        out_dtype = outputs[0].dtype
        if out_dtype not in ["float16", "float32", "float64"]:
            return

        # Transfer the inputs on the GPU and cast them to the right dtype.
        new_inputs = []
        for inp in inputs:
            if inp.dtype != out_dtype:
                gpu_cast_op = GpuElemwise(Cast(Scalar(out_dtype)))
                new_inputs.append(gpu_cast_op(as_gpuarray_variable(inp, context_name)))
            else:
                new_inputs.append(as_gpuarray_variable(inp, context_name))

        # Perform the exponent on the gpu and transfer the output back to the
        # cpu.
        gpu_output = res(*new_inputs)
        return [gpu_output]
    elif op.scalar_op in (aes.add, aes.mul):
        try:
            return [split_inputs(inputs, max_inputs_to_GpuElemwise(outputs), res)]
        except ValueError:
            return False
    else:
        return res


def split_inputs(inputs, max_nb_inputs, op):
    """
    For some ops like add and mul, a large number of inputs can make nvcc fail
    compilation of our current code. We don't want node in the graph that can't
    execute as this break DebugMode.

    This should not happen for other GpuElemwise as their is only the fusion
    that can generate op with too much input and it check for that.

    Parameters
    ----------
    inputs: List of aesara variables.
            List of inputs to node.
    max_nb_inputs: int
                   Maximum number of inputs the node can handle without
                   compilation fail.
    op : Aesara operator instance.
         Operator that should be used to rebuild the computation graph with smaller
         number of inputs per node.
    """
    if max_nb_inputs <= 1 and len(inputs) > 1:
        raise ValueError(
            "Can not split nodes because inputs' dimensionality and/or"
            " number of outputs is too large"
        )

    while len(inputs) > max_nb_inputs:
        inner_ops = []
        for i in range(0, len(inputs), max_nb_inputs):
            inner_ops.append(op(*inputs[i : i + max_nb_inputs]))
        inputs = inner_ops

    return op(*inputs)


gpu_local_elemwise_fusion = aesara.tensor.basic_opt.local_elemwise_fusion_op(
    GpuElemwise, max_inputs_to_GpuElemwise
)
optdb.register(
    "gpua_elemwise_fusion",
    # 48.5 move to gpu
    # 48.6 specialize
    # 49 cpu fusion
    # 49.5 add destroy handler
    aesara.tensor.basic_opt.FusionOptimizer(gpu_local_elemwise_fusion),
    49,
    "fast_run",
    "fusion",
    "local_elemwise_fusion",
    "gpuarray",
)

inplace_gpu_elemwise_opt = aesara.tensor.basic_opt.InplaceElemwiseOptimizer(GpuElemwise)
optdb.register(
    "gpua_inplace_opt",
    inplace_gpu_elemwise_opt,
    75,
    "inplace_elemwise_optimizer",
    "fast_run",
    "inplace",
    "gpuarray",
)

register_opt(aesara.tensor.basic_opt.local_useless_elemwise)


@register_opt("fast_compile")
@op_lifter([aesara.tensor.elemwise.DimShuffle])
@register_opt2([aesara.tensor.elemwise.DimShuffle], "fast_compile")
def local_gpua_dimshuffle(fgraph, op, context_name, inputs, outputs):
    return GpuDimShuffle(op.input_broadcastable, op.new_order)


@register_opt("fast_compile")
@op_lifter([SpecifyShape])
@register_opt2([SpecifyShape], "fast_compile")
def local_gpua_specifyShape(fgraph, op, context_name, inputs, outputs):
    if isinstance(inputs[0].type, GpuArrayType):
        return
    return local_gpua_specifyShape_graph(op, context_name, inputs, outputs)


@register_opt2([SpecifyShape], "fast_compile")
def local_gpua_specifyShape_graph(fgraph, op, context_name, inputs, outputs):
    inp = [as_gpuarray_variable(inputs[0], context_name)]
    inp += inputs[1:]
    return specify_shape(*inp)


@register_opt("fast_compile")
@op_lifter([Shape])
def local_gpua_shape(fgraph, op, context_name, inputs, outputs):
    # op_lifter will call this opt too frequently as the output is
    # always on the CPU.
    if isinstance(inputs[0].type, GpuArrayType):
        return
    return local_gpua_shape_graph(op, context_name, inputs, outputs)


@register_opt2([Shape], "fast_compile")
def local_gpua_shape_graph(fgraph, op, context_name, inputs, outputs):
    return [as_gpuarray_variable(inputs[0], context_name).shape]


def gpu_print_wrapper(op, cnda):
    op.old_op.global_fn(op.old_op, np.asarray(cnda))


@register_opt("fast_compile")
@op_lifter([aesara.printing.Print])
@register_opt2([aesara.printing.Print], "fast_compile")
def local_gpua_print_op(fgraph, op, context_name, inputs, outputs):
    (x,) = inputs
    with inherit_stack_trace(outputs):
        gpu_x = as_gpuarray_variable(x, context_name=context_name)
        new_op = op.__class__(global_fn=gpu_print_wrapper)
        new_op.old_op = op
        return new_op(gpu_x)


@register_opt("fast_compile")
@local_optimizer([PdbBreakpoint])
def local_gpu_pdbbreakpoint_op(fgraph, node):
    if isinstance(node.op, PdbBreakpoint):

        old_inputs = node.inputs
        old_outputs = node.outputs

        new_inputs = node.inputs[:1]
        input_transfered = []

        # Go through the monitored variables, only transferring on GPU those
        # for which the input comes from the GPU or the output will be
        # transferred on the GPU.
        nb_monitored_vars = len(node.outputs)
        for i in range(nb_monitored_vars):

            inp = old_inputs[i + 1]
            out = old_outputs[i]

            input_is_from_gpu = inp.owner and isinstance(inp.owner.op, HostFromGpu)
            output_goes_to_gpu = False
            for c in fgraph.clients[out]:
                if c == "output":
                    continue
                if isinstance(c[0].op, GpuFromHost):
                    output_goes_to_gpu = True
                    context_name = c[0].op.context_name
                    break

            if input_is_from_gpu:
                # The op should be applied on the GPU version of the input
                new_inputs.append(inp.owner.inputs[0])
                input_transfered.append(True)

            elif output_goes_to_gpu:
                # The input should be transferred to the gpu
                new_inputs.append(as_gpuarray_variable(inp, context_name))
                input_transfered.append(True)

            else:
                # No transfer is required.
                new_inputs.append(inp)
                input_transfered.append(False)

        # Only continue the optimization if at least one input has been
        # transferred to the gpu
        if not any(input_transfered):
            return False

        # Apply the op on the new inputs
        with inherit_stack_trace(node.outputs):
            new_op_outputs = node.op(*new_inputs, return_list=True)

            # Propagate the transfer to the gpu through the outputs that require
            # it
            new_outputs = []
            for i in range(len(new_op_outputs)):
                if input_transfered[i]:
                    new_outputs.append(new_op_outputs[i].transfer("cpu"))
                else:
                    new_outputs.append(new_op_outputs[i])

            return new_outputs

    return False


@register_opt("fast_compile")
@op_lifter([IfElse])
@register_opt2([IfElse], "fast_compile")
def local_gpua_lazy_ifelse(fgraph, op, context_name, inputs, outputs):
    if op.gpu:
        return
    c = inputs[0]
    inps = []
    falses = []
    # ifelse need corresponding true/false inputs variables to be of the same type.
    # But we can't rely on inputs to respect that, as GraphToGPU don't enforce that.
    # So we need to take care of this here.
    for v1, v2 in zip(inputs[1 : 1 + op.n_outs], inputs[1 + op.n_outs :]):
        if (
            (isinstance(v1.type, TensorType) and move_to_gpu(v1))
            or isinstance(v1.type, GpuArrayType)
            or isinstance(v2.type, GpuArrayType)
        ):
            inps.append(as_gpuarray_variable(v1, context_name))
            falses.append(as_gpuarray_variable(v2, context_name))
        else:
            inps.append(v1)
            falses.append(v2)
    inps.extend(falses)
    return IfElse(op.n_outs, gpu=True)(c, *inps, return_list=True)


@register_opt("fast_compile")
@op_lifter([Join])
@register_opt2([Join], "fast_compile")
def local_gpua_join(fgraph, op, context_name, inputs, outputs):
    return gpu_join


@register_opt("fast_compile")
@local_optimizer([GpuJoin])
def local_gpua_join_1(fgraph, node):
    # join of a single element
    if isinstance(node.op, GpuJoin) and len(node.inputs) == 2:
        return [node.inputs[1]]


@register_opt("fast_compile")
@op_lifter([Split])
@register_opt2([Split], "fast_compile")
def local_gpua_split(fgraph, op, context_name, inputs, outputs):
    # TODO use props
    return GpuSplit(op.len_splits)


@register_opt("fast_compile")
@op_lifter([aesara.tensor.subtensor.Subtensor])
def local_gpua_subtensor(fgraph, op, context_name, inputs, outputs):
    x = inputs[0]
    if x.owner and isinstance(x.owner.op, HostFromGpu):
        gpu_x = x.owner.inputs[0]
        if (
            gpu_x.owner
            and isinstance(gpu_x.owner.op, GpuFromHost)
            and
            # And it is a shared var or an input of the graph.
            not gpu_x.owner.inputs[0].owner
        ):
            if len(fgraph.clients[x]) == 1:
                if any(
                    [
                        n == "output"
                        or any(
                            [
                                isinstance(v.type, GpuArrayType)
                                for v in n.inputs + n.outputs
                            ]
                        )
                        for n, _ in fgraph.clients[outputs[0]]
                    ]
                ):
                    return
                else:
                    return [gpu_x.owner.op(outputs[0]).transfer("cpu")]

    return GpuSubtensor(op.idx_list)


@register_opt2([aesara.tensor.subtensor.Subtensor], "fast_compile")
def local_gpua_subtensor_graph(fgraph, op, context_name, inputs, outputs):
    # We need different code as the condition is different as inputs
    # aren't the same.
    x = inputs[0]
    # We don't want to move the subtensor to the GPU if the inputs is
    # on the CPU and the only client of the CPU node is this
    # subtensor. This allow to have a smaller transfer.

    if x.owner and isinstance(x.owner.op, GpuFromHost):
        cpu_x = x.owner.inputs[0]
        # And it is a shared var or an input of the graph.
        # and is used by only 1 node.
        # x is in the new graph, so we can't tests its number of clients.
        if not cpu_x.owner and len(fgraph.clients[cpu_x]) == 1:
            c = fgraph.clients[outputs[0]]
            # If the subtensor have only 1 client, do it on the CPU.
            # We let the other optimization to take care to move the
            # next node or not.
            if len(c) == 1:
                return
    return GpuSubtensor(op.idx_list)


@register_opt("fast_compile")
@op_lifter([aesara.tensor.subtensor.IncSubtensor])
@register_opt2([aesara.tensor.subtensor.IncSubtensor], "fast_compile")
def local_gpua_inc_subtensor(fgraph, op, context_name, inputs, outputs):
    op = GpuIncSubtensor(
        op.idx_list,
        op.inplace,
        op.set_instead_of_inc,
        op.destroyhandler_tolerate_aliased,
    )
    ret = op(*inputs)
    val = getattr(outputs[0].tag, "nan_guard_mode_check", True)
    ret.tag.nan_guard_mode_check = val
    return ret


@register_opt("fast_compile")
@op_lifter([aesara.tensor.subtensor.AdvancedSubtensor1])
@register_opt2([aesara.tensor.subtensor.AdvancedSubtensor1], "fast_compile")
def local_gpua_advanced_subtensor1(fgraph, op, context_name, inputs, outputs):
    return GpuAdvancedSubtensor1()


@register_opt("fast_compile")
@op_lifter([aesara.tensor.subtensor.AdvancedSubtensor])
@register_opt2([aesara.tensor.subtensor.AdvancedSubtensor], "fast_compile")
def local_gpua_advanced_subtensor(fgraph, op, context_name, inputs, outputs):
    return GpuAdvancedSubtensor()


@register_opt("fast_compile")
@op_lifter([aesara.tensor.subtensor.AdvancedIncSubtensor1])
@register_opt2([aesara.tensor.subtensor.AdvancedIncSubtensor1], "fast_compile")
def local_gpua_advanced_incsubtensor1(fgraph, op, context_name, inputs, outputs):
    x, y, ilist = inputs

    set_instead_of_inc = op.set_instead_of_inc

    if (
        x.ndim == 1
        and y.ndim == 0
        and config.deterministic == "default"
        and x.dtype not in ("int8", "int16")
    ):
        x = x.dimshuffle(0, "x")
        y = y.dimshuffle("x", "x")
        ret = GpuAdvancedIncSubtensor1_dev20(set_instead_of_inc=set_instead_of_inc)(
            x, y, ilist
        )
        ret = GpuDimShuffle(ret.type.broadcastable, [0])(ret)
        return ret
    elif (
        x.ndim != 2
        or y.ndim != 2
        or config.deterministic == "more"
        or x.dtype in ("int8", "int16")
    ):
        return GpuAdvancedIncSubtensor1(set_instead_of_inc=set_instead_of_inc)
    else:
        return GpuAdvancedIncSubtensor1_dev20(set_instead_of_inc=set_instead_of_inc)


# Do not register this optimization for now, as it slows down the
# execution by a lot in important cases.
# @register_opt('fast_compile')
# @op_lifter([aesara.tensor.subtensor.AdvancedIncSubtensor])
# @register_opt2([aesara.tensor.subtensor.AdvancedIncSubtensor], 'fast_compile')
def local_gpua_advanced_incsubtensor(fgraph, op, context_name, inputs, outputs):
    if not op.set_instead_of_inc:
        return GpuAdvancedIncSubtensor()
    else:
        return False


@register_inplace()
@local_optimizer([GpuAdvancedIncSubtensor1, GpuAdvancedIncSubtensor1_dev20])
def local_advincsub1_gpua_inplace(fgraph, node):
    if isinstance(node.op, (GpuAdvancedIncSubtensor1, GpuAdvancedIncSubtensor1_dev20)):
        if not node.op.inplace:
            return [node.op.clone_inplace()(*node.inputs)]


# AllocDiag
@register_opt("fast_compile")
@op_lifter([AllocDiag])
@register_opt2([AllocDiag], "fast_compile")
def local_gpu_alloc_diag(fgraph, op, context_name, inputs, outputs):
    if outputs[0].ndim != 2:
        # AllocDiag only supports 2d output
        return False
    return GpuAllocDiag(offset=op.offset)


# ExtractDiag
@register_opt("fast_compile")
@op_lifter([ExtractDiag])
@register_opt2([ExtractDiag], "fast_compile")
def local_gpu_extract_diag(fgraph, op, context_name, inputs, outputs):
    return GpuExtractDiag(
        offset=op.offset, axis1=op.axis1, axis2=op.axis2, view=op.view
    )


@register_opt("fast_compile")
@op_lifter(
    [
        aesara.tensor.elemwise.CAReduce,
        aesara.tensor.math.Sum,
        aesara.tensor.math.Prod,
    ]
)
@register_opt2(
    [
        aesara.tensor.elemwise.CAReduce,
        aesara.tensor.math.Sum,
        aesara.tensor.math.Prod,
    ],
    "fast_compile",
)
def local_gpua_careduce(fgraph, op, context_name, inputs, outputs):
    if isinstance(
        op.scalar_op,
        (aes.Add, aes.Mul, aes.ScalarMaximum, aes.ScalarMinimum),
    ):

        ctx = get_context(context_name)
        if ctx.kind == b"opencl":
            op2 = GpuCAReduceCPY
            if op.scalar_op not in [aes.add, aes.mul]:
                # We don't support yet all reduction with cpy code.
                return
        elif ctx.kind == b"cuda":
            op2 = GpuCAReduceCuda
        else:
            return False
        (x,) = inputs
        idtype = x.dtype
        adtype = getattr(op, "acc_dtype", None)
        odtype = getattr(op, "dtype", outputs[0].dtype)

        # Force accumulator to float32 for float32 inputs since tree
        # reduction will not loose as much precision as linear
        # accumulation and float64 is much slower on GPU.
        if idtype == "float32" and odtype == "float32":
            adtype = "float32"

        greduce = op2(op.scalar_op, axis=op.axis, dtype=odtype, acc_dtype=adtype)
        with inherit_stack_trace(outputs):
            gvar = greduce(x)
        # We need to have the make node called, otherwise the mask can
        # be None
        if op2 is GpuCAReduceCPY or gvar.owner.op.supports_c_code(
            [as_gpuarray_variable(x, context_name)]
        ):
            return greduce
        else:
            # Try to make a simpler pattern based on reshaping
            # The principle is that if two adjacent dimensions have
            # the same value in the reduce_mask, then we can reshape
            # to make them a single dimension, do the reduction, and
            # then reshape to get them back.

            if op.axis is None:
                reduce_mask = [1] * x.type.ndim
            else:
                reduce_mask = [0] * x.type.ndim
                for a in op.axis:
                    assert reduce_mask[a] == 0
                    reduce_mask[a] = 1

            new_in_shp = [shape_i(x, 0)]
            new_mask = [reduce_mask[0]]
            for i in range(1, x.type.ndim):
                if reduce_mask[i] == reduce_mask[i - 1]:
                    new_in_shp[-1] *= shape_i(x, i)
                else:
                    new_mask.append(reduce_mask[i])
                    new_in_shp.append(shape_i(x, i))
            new_axis = []
            for idx, m in enumerate(new_mask):
                if m == 1:
                    new_axis.append(idx)
            greduce = op2(
                op.scalar_op,
                axis=new_axis,
                reduce_mask=new_mask,
                dtype=odtype,
                acc_dtype=adtype,
            )
            with inherit_stack_trace(outputs):
                reshaped_x = x.reshape(aet.stack(new_in_shp))
                gpu_reshaped_x = as_gpuarray_variable(reshaped_x, context_name)
                # We need to have the make node called, otherwise the mask can
                # be None
                gvar = greduce(gpu_reshaped_x)
                reshaped_gpu_inputs = [gpu_reshaped_x]
                if greduce.supports_c_code(reshaped_gpu_inputs):
                    reduce_reshaped_x = greduce(gpu_reshaped_x)

                    if reduce_reshaped_x.ndim != outputs[0].ndim:
                        out_shp = []
                        for i in range(x.ndim):
                            if i not in op.axis:
                                out_shp.append(shape_i(x, i))
                        unreshaped_reduce = GpuReshape(len(out_shp))(
                            reduce_reshaped_x, aet.stack(out_shp)
                        )
                    else:
                        unreshaped_reduce = reduce_reshaped_x
                    return [unreshaped_reduce]


@register_opt("fast_compile")
@op_lifter([aesara.tensor.blas.Gemv, aesara.tensor.blas_c.CGemv])
@register_opt2([aesara.tensor.blas.Gemv], "fast_compile")
def local_gpua_gemv(fgraph, op, context_name, inputs, outputs):
    if inputs[0].dtype == "float16":
        # Use gemm implementation as cublas gemv don't support float16
        return gpugemm_no_inplace(
            inputs[0][:, None], inputs[1], inputs[2], inputs[3][:, None], inputs[4]
        ).dimshuffle(0)

    if inputs[0].dtype not in ["float32", "float64"]:
        return
    if op.inplace:
        return gpugemv_inplace
    else:
        return gpugemv_no_inplace


@register_opt("fast_compile")
@op_lifter([aesara.tensor.blas.Gemm])
@register_opt2([aesara.tensor.blas.Gemm], "fast_compile")
def local_gpua_gemm(fgraph, op, context_name, inputs, outputs):
    if inputs[0].dtype not in ["float16", "float32", "float64"]:
        return
    if op.inplace:
        return gpugemm_inplace
    else:
        return gpugemm_no_inplace


@register_opt("fast_compile")
@op_lifter([aesara.tensor.blas.BatchedDot])
@register_opt2([aesara.tensor.blas.BatchedDot], "fast_compile")
def local_gpua_gemmbatch(fgraph, op, context_name, inputs, outputs):
    if inputs[0].dtype not in ["float16", "float32", "float64"]:
        return
    with inherit_stack_trace(outputs):
        a, b = inputs
        # Since GpuGemmBatch only supports 3D inputs and output,
        # we need to add broadcastable dims to the inputs, and drop
        # them from outputs
        output_dims = [0, 1, 2]
        if a.ndim == 2:
            a = GpuDimShuffle(a.broadcastable, (0, "x", 1))(a)
            del output_dims[1]
        if b.ndim == 2:
            b = GpuDimShuffle(b.broadcastable, (0, 1, "x"))(b)
            del output_dims[-1]
        # In case of mismatched dtypes, we also have to upcast
        out_dtype = outputs[0].dtype
        if a.dtype != out_dtype or b.dtype != out_dtype:
            gpu_cast_op = GpuElemwise(Cast(Scalar(out_dtype)))
            if a.dtype != out_dtype:
                a = gpu_cast_op(a)
            if b.dtype != out_dtype:
                b = gpu_cast_op(b)

        c = GpuAllocEmpty(out_dtype, context_name)(a.shape[0], a.shape[1], b.shape[2])
        out = gpugemmbatch_no_inplace(
            c, np.asarray(1.0, dtype=out_dtype), a, b, np.asarray(0.0, dtype=out_dtype)
        )
        if len(output_dims) != 3:
            out = GpuDimShuffle(out.broadcastable, output_dims)(out)
        return out


@register_opt()
@alpha_merge(GpuGemm, alpha_in=1, beta_in=4)
def local_gpua_gemm_alpha_merge(fgraph, node, *inputs):
    return [gpugemm_no_inplace(*inputs)]


@register_opt()
@output_merge(GpuGemm, alpha_in=1, beta_in=4, out_in=0)
def local_gpua_gemm_output_merge(fgraph, node, *inputs):
    return [gpugemm_no_inplace(*inputs)]


@register_opt()
@alpha_merge(GpuGemmBatch, alpha_in=1, beta_in=4)
def local_gpua_gemmbatch_alpha_merge(fgraph, node, *inputs):
    return [gpugemmbatch_no_inplace(*inputs)]


@register_opt()
@output_merge(GpuGemmBatch, alpha_in=1, beta_in=4, out_in=0)
def local_gpua_gemmbatch_output_merge(fgraph, node, *inputs):
    return [gpugemmbatch_no_inplace(*inputs)]


@register_opt("fast_compile")
@op_lifter(
    [
        aesara.tensor.blas.Ger,
        aesara.tensor.blas_c.CGer,
        aesara.tensor.blas_scipy.ScipyGer,
    ]
)
@register_opt2(
    [
        aesara.tensor.blas.Ger,
        aesara.tensor.blas_c.CGer,
        aesara.tensor.blas_scipy.ScipyGer,
    ],
    "fast_compile",
)
def local_gpua_ger(fgraph, op, context_name, inputs, outputs):
    if inputs[0].dtype not in ["float32", "float64"]:
        return
    return GpuGer(inplace=op.destructive)


@register_opt("fast_compile")
@op_lifter([aesara.tensor.blas.Dot22])
@register_opt2([aesara.tensor.blas.Dot22], "fast_compile")
def local_gpua_dot22(fgraph, op, context_name, inputs, outputs):
    return gpu_dot22


@register_opt("fast_compile")
@op_lifter([aesara.tensor.blas.Dot22Scalar])
@register_opt2([aesara.tensor.blas.Dot22Scalar], "fast_compile")
def local_gpua_dot22scalar(fgraph, op, context_name, inputs, outputs):
    with inherit_stack_trace(outputs):
        x, y, a = inputs
        x = as_gpuarray_variable(x, context_name)
        y = as_gpuarray_variable(y, context_name)
        z = GpuAllocEmpty(x.dtype, context_name)(x.shape[0], y.shape[1])
        return [gpugemm_no_inplace(z, a, x, y, 0)]


@register_opt("fast_compile")
@op_lifter([Eye])
@register_opt2([Eye], "fast_compile")
def local_gpua_eye(fgraph, op, context_name, inputs, outputs):
    return GpuEye(dtype=op.dtype, context_name=context_name)


@register_opt("fast_compile")
@op_lifter([Tri])
@register_opt2([Tri], "fast_compile")
def local_gpua_tri(fgraph, op, context_name, inputs, outputs):
    return GpuTri(dtype=op.dtype, context_name=context_name)


@register_opt("fast_compile")
@op_lifter([aesara.tensor.nnet.basic.CrossentropySoftmaxArgmax1HotWithBias])
@register_opt2(
    [aesara.tensor.nnet.basic.CrossentropySoftmaxArgmax1HotWithBias], "fast_compile"
)
def local_gpua_crossentropysoftmaxargmax1hotwithbias(
    fgraph, op, context_name, inputs, outputs
):
    return gpu_crossentropy_softmax_argmax_1hot_with_bias


@register_opt("fast_compile")
@op_lifter([aesara.tensor.nnet.basic.CrossentropySoftmax1HotWithBiasDx])
@register_opt2(
    [aesara.tensor.nnet.basic.CrossentropySoftmax1HotWithBiasDx], "fast_compile"
)
def local_gpua_crossentropysoftmax1hotwithbiasdx(
    fgraph, op, context_name, inputs, outputs
):
    return gpu_crossentropy_softmax_1hot_with_bias_dx


@register_opt("fast_compile")
@op_lifter([aesara.tensor.nnet.basic.Softmax])
@register_opt2([aesara.tensor.nnet.basic.Softmax], "fast_compile")
def local_gpua_softmax(fgraph, op, context_name, inputs, outputs):
    return gpu_softmax


@register_opt("fast_compile")
@op_lifter([aesara.tensor.nnet.basic.SoftmaxWithBias])
@register_opt2([aesara.tensor.nnet.basic.SoftmaxWithBias], "fast_compile")
def local_gpua_softmaxwithbias(fgraph, op, context_name, inputs, outputs):
    return gpu_softmax_with_bias


@register_opt("fast_compile")
@op_lifter([aesara.tensor.nnet.basic.CrossentropyCategorical1Hot])
@register_opt2([aesara.tensor.nnet.basic.CrossentropyCategorical1Hot], "fast_compile")
def local_gpu_crossentropycategorical1hot(fgraph, op, context_name, inputs, outputs):
    # There is no corresponding GPU Op, but we can express it as:
    #   coding, one_of_n = inputs
    #   -log(coding[arange(coding.shape[0]), one_of_n])
    coding, one_of_n = inputs
    idx0 = aet.arange(shape_i(coding, 0))
    return [gpu_neg(gpu_log(coding[idx0, one_of_n]))]


@register_opt("fast_compile")
@op_lifter([aesara.tensor.nnet.basic.CrossentropyCategorical1HotGrad])
@register_opt2(
    [aesara.tensor.nnet.basic.CrossentropyCategorical1HotGrad], "fast_compile"
)
def local_gpu_crossentropycategorical1hotgrad(
    fgraph, op, context_name, inputs, outputs
):
    # There is no corresponding GPU Op, but we can express it as:
    #   gy, coding, one_of_n = inputs
    #   gcoding = zeros_like(coding)
    #   gcoding[arange(coding.shape[0]), one_of_n] = -g / (
    #       coding[arange(coding.shape[0]), one_of_n])
    gy, coding, one_of_n = inputs
    idx0 = aet.arange(shape_i(coding, 0))
    z = GpuAlloc(context_name, memset_0=True)(
        as_gpuarray_variable(np.zeros((), dtype=coding.dtype), context_name),
        *[shape_i(coding, i) for i in range(coding.ndim)],
    )
    gcoding = aesara.tensor.subtensor.set_subtensor(
        z[idx0, one_of_n], gpu_neg(gpu_true_div(gy, coding[idx0, one_of_n]))
    )
    return [gcoding.transfer(context_name)]


@register_opt("fast_compile")
@op_lifter([Assert])
def local_gpua_assert(fgraph, op, context_name, inputs, outputs):
    if isinstance(inputs[0].type, GpuArrayType):
        return
    return local_gpua_assert_graph(op, context_name, inputs, outputs)


@register_opt2([Assert], "fast_compile")
def local_gpua_assert_graph(fgraph, op, context_name, inputs, outputs):
    return [op(as_gpuarray_variable(inputs[0], context_name), *inputs[1:])]


@register_opt("fast_compile")
@op_lifter([ConvOp])
@register_opt2([ConvOp], "fast_compile")
def local_gpua_error_convop(fgraph, op, context_name, inputs, outputs):
    raise AssertionError(
        """
ConvOp does not work with the gpuarray backend.

Use the new convolution interface to have GPU convolution working:
aesara.tensor.nnet.conv2d()
"""
    )


@register_opt("fast_compile")
@op_lifter([SparseBlockGemv])
@register_opt2([SparseBlockGemv], "fast_compile")
def local_gpua_sparseblockgemv(fgraph, op, context_name, inputs, outputs):
    if inputs[0].dtype == "float16":
        return
    if op.inplace:
        return gpu_sparse_block_gemv_inplace
    else:
        return gpu_sparse_block_gemv


@register_opt("fast_compile")
@op_lifter([SparseBlockOuter])
@register_opt2([SparseBlockOuter], "fast_compile")
def local_gpua_sparseblockouter(fgraph, op, context_name, inputs, outputs):
    if inputs[0].dtype == "float16":
        return
    if op.inplace:
        return gpu_sparse_block_outer_inplace
    else:
        return gpu_sparse_block_outer


@register_inplace()
@local_optimizer([GpuSparseBlockGemv], inplace=True)
def local_inplace_sparseblockgemv(fgraph, node):
    if isinstance(node.op, GpuSparseBlockGemv) and not node.op.inplace:
        return [gpu_sparse_block_gemv_inplace(*node.inputs)]


@register_inplace()
@local_optimizer([GpuSparseBlockOuter], inplace=True)
def local_inplace_sparseblockouter(fgraph, node):
    if isinstance(node.op, GpuSparseBlockOuter) and not node.op.inplace:
        return [GpuSparseBlockOuter(inplace=True)(*node.inputs)]


# Move to Gpu optimization
@local_optimizer(
    [
        GpuFromHost,
        AbstractConv2d,
        AbstractConv2d_gradWeights,
        AbstractConv2d_gradInputs,
        AbstractConv3d,
        AbstractConv3d_gradWeights,
        AbstractConv3d_gradInputs,
    ]
)
def local_conv_gpu_conv(fgraph, node):
    """
    gpu_from_host(AbstractConv) -> AbstractConv(gpu_from_host)

    AbstractConv(host_from_gpu) -> host_from_gpu(AbstractConv)
    """
    if isinstance(node.op, GpuFromHost):
        host_input = node.inputs[0]
        if host_input.owner and isinstance(host_input.owner.op, BaseAbstractConv):

            conv = host_input.owner.op
            inps = list(host_input.owner.inputs)
            ctx = infer_context_name(*inps)
            inps[0] = as_gpuarray_variable(inps[0], context_name=ctx)
            inps[1] = as_gpuarray_variable(inps[1], context_name=ctx)
            out = conv(*inps)
            # out is on the GPU because both inputs are.
            out = aet.patternbroadcast(out, node.outputs[0].broadcastable)
            return [out]

    if isinstance(node.op, BaseAbstractConv):
        # conv(host_from_gpu) -> host_from_gpu(gpu_conv)
        inp1 = node.inputs[0]
        inp2 = node.inputs[1]
        if isinstance(inp1.type, GpuArrayType) and isinstance(inp2.type, GpuArrayType):
            # Both inputs are already directly on the GPU, nothing to do
            return

        inp1_on_gpu = isinstance(inp1.type, GpuArrayType) or (
            inp1.owner and isinstance(inp1.owner.op, HostFromGpu)
        )
        inp2_on_gpu = isinstance(inp2.type, GpuArrayType) or (
            inp2.owner and isinstance(inp2.owner.op, HostFromGpu)
        )

        if inp1_on_gpu or inp2_on_gpu:
            conv = node.op
            inps = list(node.inputs)
            ctx = infer_context_name(*inps)
            inps[0] = as_gpuarray_variable(inps[0], context_name=ctx)
            inps[1] = as_gpuarray_variable(inps[1], context_name=ctx)
            out = conv(*inps)
            # out is on the GPU because both inputs are.
            out = aet.patternbroadcast(out, node.outputs[0].broadcastable)
            # If the original output was on CPU, we have to transfer it
            if isinstance(node.outputs[0].type, TensorType):
                return [aet.as_tensor_variable(out)]
            else:
                return [out]


register_opt()(local_conv_gpu_conv)


# CorrMM opt
@local_optimizer([AbstractConv2d])
def local_abstractconv_gemm(fgraph, node):
    if not isinstance(node.op, AbstractConv2d):
        return None
    img, kern = node.inputs
    if not isinstance(img.type, GpuArrayType) or not isinstance(
        kern.type, GpuArrayType
    ):
        return None
    ctx = infer_context_name(img, kern)

    border_mode = node.op.border_mode
    subsample = node.op.subsample
    filter_dilation = node.op.filter_dilation
    num_groups = node.op.num_groups
    unshared = node.op.unshared

    flip = (slice(None),) * (kern.ndim - 2) + (slice(None, None, -1),) * 2
    kern_axes = (1, 0) + tuple(i for i in range(2, kern.ndim))
    if (
        (border_mode == "full")
        and (subsample == (1, 1))
        and num_groups == 1
        and not unshared
    ):
        if not node.op.filter_flip:
            kern = kern[flip]
        # need to dimshuffle the kernel for full convolution
        kern = kern.dimshuffle(kern_axes)
        # call GpuCorrMM_gradInputs
        rval = GpuCorrMM_gradInputs("valid", subsample, filter_dilation)(
            gpu_contiguous(kern), gpu_contiguous(img)
        )
    else:
        # need to flip the kernel if necessary
        if node.op.filter_flip:
            kern = kern[flip]
        # By default use GpuCorrMM
        rval = GpuCorrMM(border_mode, subsample, filter_dilation, num_groups, unshared)(
            gpu_contiguous(img), gpu_contiguous(kern)
        )

        # call GpuCorrMM_gradWeights if good
        # (the latter is faster if batchsize * kernelHeight * kernelWidth
        # is larger than inputChannels * outputHeight * outputWidth.
        # GpuConv does not always store information on the batchsize and
        # channels, though, so we only use what information we have.)
        if (
            (subsample == (1, 1))
            and (filter_dilation == (1, 1))
            and (node.op.imshp is not None)
            and (None not in node.op.imshp[-2:])
            and (node.op.kshp is not None)
            and (None not in node.op.kshp)
            and border_mode != "half"
            and num_groups == 1
            and not unshared
        ):
            # we know the kernel and output size
            prod1 = node.op.kshp[0] * node.op.kshp[-3]
            prod2 = (node.op.imshp[-2] - node.op.kshp[0] + 1) * (
                node.op.imshp[-1] - node.op.kshp[-3] + 1
            )
            if None not in node.op.imshp[:1]:
                # we also know batchsize and input channels
                prod1 *= node.op.imshp[0]
                prod2 *= node.op.imshp[1]
            # compare to decide
            if prod1 > prod2:
                rval = GpuCorrMM_gradWeights(border_mode, subsample, filter_dilation)(
                    gpu_contiguous(img.dimshuffle(1, 0, 2, 3)),
                    gpu_contiguous(kern.dimshuffle(1, 0, 2, 3)),
                )
                # (we need to wrap the result in as_gpuarray_variable,
                # because we are not allowed to replace a GpuArray with
                # a DimShuffle instance in a graph optimization)
                rval = as_gpuarray_variable(
                    rval.dimshuffle(1, 0, 2, 3), context_name=ctx
                )
    return [rval]


# CorrMM opt used for Meta-optimizer
@local_optimizer([AbstractConv2d])
def local_abstractconv_gemm_def(fgraph, node):
    if not isinstance(node.op, AbstractConv2d):
        return None
    img, kern = node.inputs
    if not isinstance(img.type, GpuArrayType) or not isinstance(
        kern.type, GpuArrayType
    ):
        return None

    border_mode = node.op.border_mode
    subsample = node.op.subsample
    filter_dilation = node.op.filter_dilation
    num_groups = node.op.num_groups
    unshared = node.op.unshared

    if node.op.filter_flip:
        flip = (slice(None),) * (kern.ndim - 2) + (slice(None, None, -1),) * 2
        kern = kern[flip]
    rval = GpuCorrMM(border_mode, subsample, filter_dilation, num_groups, unshared)(
        gpu_contiguous(img), gpu_contiguous(kern)
    )
    return [rval]


@local_optimizer([AbstractConv2d])
def local_abstractconv_gemm_alt(fgraph, node):
    if not isinstance(node.op, AbstractConv2d):
        return None
    img, kern = node.inputs
    if not isinstance(img.type, GpuArrayType) or not isinstance(
        kern.type, GpuArrayType
    ):
        return None
    ctx = infer_context_name(img, kern)

    border_mode = node.op.border_mode
    subsample = node.op.subsample
    filter_dilation = node.op.filter_dilation
    num_groups = node.op.num_groups
    unshared = node.op.unshared

    if (
        border_mode == "full"
        and subsample == (1, 1)
        and num_groups == 1
        and not unshared
    ):
        if not node.op.filter_flip:
            kern = kern[:, :, ::-1, ::-1]

        kern = kern.dimshuffle(1, 0, 2, 3)
        rval = GpuCorrMM_gradInputs("valid", subsample, filter_dilation)(
            gpu_contiguous(kern), gpu_contiguous(img)
        )

    elif (
        border_mode == "valid"
        and subsample == (1, 1)
        and filter_dilation == (1, 1)
        and num_groups == 1
        and not unshared
    ):
        if node.op.filter_flip:
            kern = kern[:, :, ::-1, ::-1]

        rval = GpuCorrMM_gradWeights(border_mode, subsample, filter_dilation)(
            gpu_contiguous(img.dimshuffle(1, 0, 2, 3)),
            gpu_contiguous(kern.dimshuffle(1, 0, 2, 3)),
        )
        rval = as_gpuarray_variable(rval.dimshuffle(1, 0, 2, 3), context_name=ctx)
    else:
        return None

    return [rval]


@local_optimizer([AbstractConv3d])
def local_abstractconv3d_gemm(fgraph, node):
    if not isinstance(node.op, AbstractConv3d):
        return None
    img, kern = node.inputs
    if not isinstance(img.type, GpuArrayType) or not isinstance(
        kern.type, GpuArrayType
    ):
        return None
    ctx = infer_context_name(img, kern)

    border_mode = node.op.border_mode
    subsample = node.op.subsample
    filter_dilation = node.op.filter_dilation
    num_groups = node.op.num_groups
    if (border_mode == "full") and (subsample == (1, 1, 1)) and num_groups == 1:
        if not node.op.filter_flip:
            kern = kern[:, :, ::-1, ::-1, ::-1]
        # need to dimshuffle the kernel for full convolution
        kern = kern.dimshuffle(1, 0, 2, 3, 4)
        # call GpuCorr3dMM_gradInputs
        rval = GpuCorr3dMM_gradInputs("valid", subsample, filter_dilation)(
            gpu_contiguous(kern), gpu_contiguous(img)
        )
    else:
        # need to flip the kernel if necessary
        if node.op.filter_flip:
            kern = kern[:, :, ::-1, ::-1, ::-1]
        # By default use GpuCorr3dMM
        rval = GpuCorr3dMM(border_mode, subsample, filter_dilation, num_groups)(
            gpu_contiguous(img), gpu_contiguous(kern)
        )

        # call GpuCorr3dMM_gradWeights if good
        # (the latter is faster if batchsize * kernelHeight * kernelWidth * kernelDepth
        # is larger than inputChannels * outputHeight * outputWidth * outputDepth.
        # GpuConv does not always store information on the batchsize and
        # channels, though, so we only use what information we have.)
        if (
            (subsample == (1, 1, 1))
            and (filter_dilation == (1, 1, 1))
            and (node.op.imshp is not None)
            and (None not in node.op.imshp[-3:])
            and (node.op.kshp is not None)
            and (None not in node.op.kshp)
            and border_mode != "half"
            and num_groups == 1
        ):
            # we know the kernel and output size
            prod1 = node.op.kshp[0] * node.op.kshp[1] * node.op.kshp[2]
            prod2 = (
                (node.op.imshp[-3] - node.op.kshp[0] + 1)
                * (node.op.imshp[-2] - node.op.kshp[1] + 1)
                * (node.op.imshp[-1] - node.op.kshp[2] + 1)
            )
            if None not in node.op.imshp[:1]:
                # we also know batchsize and input channels
                prod1 *= node.op.imshp[0]
                prod2 *= node.op.imshp[1]
            # compare to decide
            if prod1 > prod2:
                rval = GpuCorr3dMM_gradWeights(border_mode, subsample, filter_dilation)(
                    gpu_contiguous(img.dimshuffle(1, 0, 2, 3, 4)),
                    gpu_contiguous(kern.dimshuffle(1, 0, 2, 3, 4)),
                )
                # (we need to wrap the result in as_gpuarray_variable,
                # because we are not allowed to replace a GpuArray with
                # a DimShuffle instance in a graph optimization)
                rval = as_gpuarray_variable(
                    rval.dimshuffle(1, 0, 2, 3, 4), context_name=ctx
                )
    return [rval]


# Corr3dMM opt used for Meta-optimizer
@local_optimizer([AbstractConv3d])
def local_abstractconv3d_gemm_def(fgraph, node):
    if not isinstance(node.op, AbstractConv3d):
        return None
    img, kern = node.inputs
    if not isinstance(img.type, GpuArrayType) or not isinstance(
        kern.type, GpuArrayType
    ):
        return None

    border_mode = node.op.border_mode
    subsample = node.op.subsample
    filter_dilation = node.op.filter_dilation
    if node.op.filter_flip:
        kern = kern[:, :, ::-1, ::-1, ::-1]
    # By default use GpuCorr3dMM
    rval = GpuCorr3dMM(border_mode, subsample, filter_dilation, node.op.num_groups)(
        gpu_contiguous(img), gpu_contiguous(kern)
    )
    return [rval]


@local_optimizer([AbstractConv3d])
def local_abstractconv3d_alt(fgraph, node):
    if not isinstance(node.op, AbstractConv3d):
        return None
    img, kern = node.inputs
    if not isinstance(img.type, GpuArrayType) or not isinstance(
        kern.type, GpuArrayType
    ):
        return None
    ctx = infer_context_name(img, kern)

    border_mode = node.op.border_mode
    subsample = node.op.subsample
    filter_dilation = node.op.filter_dilation
    num_groups = node.op.num_groups

    if (border_mode == "full") and (subsample == (1, 1, 1)) and (num_groups == 1):
        if not node.op.filter_flip:
            kern = kern[:, :, ::-1, ::-1, ::-1]
        kern = kern.dimshuffle(1, 0, 2, 3, 4)
        rval = GpuCorr3dMM_gradInputs("valid", subsample, filter_dilation)(
            gpu_contiguous(kern), gpu_contiguous(img)
        )

    elif (
        subsample == (1, 1, 1)
        and filter_dilation == (1, 1, 1)
        and border_mode == "valid"
        and num_groups == 1
    ):
        if node.op.filter_flip:
            kern = kern[:, :, ::-1, ::-1, ::-1]
        rval = GpuCorr3dMM_gradWeights(border_mode, subsample, filter_dilation)(
            gpu_contiguous(img.dimshuffle(1, 0, 2, 3, 4)),
            gpu_contiguous(kern.dimshuffle(1, 0, 2, 3, 4)),
        )
        rval = as_gpuarray_variable(rval.dimshuffle(1, 0, 2, 3, 4), context_name=ctx)
    else:
        return None
    return [rval]


@local_optimizer([AbstractConv3d])
def local_abstractconv3d2d(fgraph, node):
    if not isinstance(node.op, AbstractConv3d):
        return None
    img, kern = node.inputs
    if not isinstance(img.type, GpuArrayType) or not isinstance(
        kern.type, GpuArrayType
    ):
        return None

    ctx = infer_context_name(img, kern)
    border_mode = node.op.border_mode
    subsample = node.op.subsample
    filter_dilation = node.op.filter_dilation
    num_groups = node.op.num_groups

    if subsample == (1, 1, 1) and filter_dilation == (1, 1, 1) and num_groups == 1:
        reorder_array = [0, 2, 1, 3, 4]
        rval = conv3d2d.conv3d(
            gpu_contiguous(img.dimshuffle(*reorder_array)),
            gpu_contiguous(kern.dimshuffle(*reorder_array)),
            [node.op.imshp[i] for i in reorder_array],
            [node.op.kshp[i] for i in reorder_array],
            border_mode=border_mode,
        )
        rval = as_gpuarray_variable(rval.dimshuffle(*reorder_array), context_name=ctx)

        return [rval]
    else:
        return None


@local_optimizer([AbstractConv2d_gradWeights])
def local_abstractconv_gradweights_gemm(fgraph, node):
    if not isinstance(node.op, AbstractConv2d_gradWeights):
        return None
    img, topgrad, shape = node.inputs
    if not isinstance(img.type, GpuArrayType) or not isinstance(
        topgrad.type, GpuArrayType
    ):
        return None
    ctx = infer_context_name(img, topgrad)

    rval = GpuCorrMM_gradWeights(
        border_mode=node.op.border_mode,
        subsample=node.op.subsample,
        filter_dilation=node.op.filter_dilation,
        num_groups=node.op.num_groups,
        unshared=node.op.unshared,
    )(gpu_contiguous(img), gpu_contiguous(topgrad), shape)
    flip = (slice(None),) * (rval.ndim - 2) + (slice(None, None, -1),) * 2
    if node.op.filter_flip:
        rval = rval[flip]
    rval = aet.patternbroadcast(rval, node.outputs[0].broadcastable)
    rval = as_gpuarray_variable(rval, context_name=ctx)
    return [rval]


@local_optimizer([AbstractConv2d_gradWeights])
def local_abstractconv_gemm_gradweights_alt(fgraph, node):
    if not isinstance(node.op, AbstractConv2d_gradWeights):
        return None
    img, topgrad, shape = node.inputs
    if not isinstance(img.type, GpuArrayType) or not isinstance(
        topgrad.type, GpuArrayType
    ):
        return None
    ctx = infer_context_name(img, topgrad)
    border_mode = node.op.border_mode
    subsample = node.op.subsample
    filter_dilation = node.op.filter_dilation
    num_groups = node.op.num_groups
    unshared = node.op.unshared

    if (
        border_mode == "valid"
        and subsample == (1, 1)
        and filter_dilation == (1, 1)
        and num_groups == 1
        and not unshared
    ):
        rval = GpuCorrMM(border_mode, subsample, filter_dilation)(
            gpu_contiguous(img.dimshuffle(1, 0, 2, 3)),
            gpu_contiguous(topgrad.dimshuffle(1, 0, 2, 3)),
        )

        if node.op.filter_flip:
            rval = rval[:, :, ::-1, ::-1]

        rval = rval.dimshuffle(1, 0, 2, 3)
        rval = aet.patternbroadcast(rval, node.outputs[0].broadcastable)
        rval = as_gpuarray_variable(rval, context_name=ctx)
        return [rval]
    else:
        return None


@local_optimizer([AbstractConv3d_gradWeights])
def local_abstractconv3d_gemm_gradweights_alt(fgraph, node):
    if not isinstance(node.op, AbstractConv3d_gradWeights):
        return None
    img, topgrad, shape = node.inputs
    if not isinstance(img.type, GpuArrayType) or not isinstance(
        topgrad.type, GpuArrayType
    ):
        return None
    ctx = infer_context_name(img, topgrad)
    border_mode = node.op.border_mode
    subsample = node.op.subsample
    filter_dilation = node.op.filter_dilation
    num_groups = node.op.num_groups

    if (
        border_mode == "valid"
        and subsample == (1, 1, 1)
        and filter_dilation == (1, 1, 1)
        and num_groups == 1
    ):
        rval = GpuCorr3dMM(border_mode, subsample, filter_dilation)(
            gpu_contiguous(img.dimshuffle(1, 0, 2, 3, 4)),
            gpu_contiguous(topgrad.dimshuffle(1, 0, 2, 3, 4)),
        )

        if node.op.filter_flip:
            rval = rval[:, :, ::-1, ::-1, ::-1]

        rval = rval.dimshuffle(1, 0, 2, 3, 4)
        rval = aet.patternbroadcast(rval, node.outputs[0].broadcastable)
        rval = as_gpuarray_variable(rval, context_name=ctx)
        return [rval]
    else:
        return None


@local_optimizer([AbstractConv3d_gradWeights])
def local_abstractconv3d_gradweights_gemm(fgraph, node):
    if not isinstance(node.op, AbstractConv3d_gradWeights):
        return None
    img, topgrad, shape = node.inputs
    if not isinstance(img.type, GpuArrayType) or not isinstance(
        topgrad.type, GpuArrayType
    ):
        return None
    ctx = infer_context_name(img, topgrad)

    rval = GpuCorr3dMM_gradWeights(
        border_mode=node.op.border_mode,
        subsample=node.op.subsample,
        filter_dilation=node.op.filter_dilation,
        num_groups=node.op.num_groups,
    )(gpu_contiguous(img), gpu_contiguous(topgrad), shape)
    if node.op.filter_flip:
        rval = rval[:, :, ::-1, ::-1, ::-1]
    rval = aet.patternbroadcast(rval, node.outputs[0].broadcastable)
    rval = as_gpuarray_variable(rval, context_name=ctx)
    return [rval]


@local_optimizer([AbstractConv2d_gradInputs])
def local_abstractconv_gradinputs_gemm(fgraph, node):
    if not isinstance(node.op, AbstractConv2d_gradInputs):
        return None
    kern, topgrad, shape = node.inputs
    if not isinstance(kern.type, GpuArrayType) or not isinstance(
        topgrad.type, GpuArrayType
    ):
        return None

    if node.op.filter_flip:
        flip = (slice(None),) * (kern.ndim - 2) + (slice(None, None, -1),) * 2
        kern = kern[flip]

    rval = GpuCorrMM_gradInputs(
        border_mode=node.op.border_mode,
        subsample=node.op.subsample,
        filter_dilation=node.op.filter_dilation,
        num_groups=node.op.num_groups,
        unshared=node.op.unshared,
    )(gpu_contiguous(kern), gpu_contiguous(topgrad), shape)
    return [rval]


@local_optimizer([AbstractConv2d_gradInputs])
def local_abstractconv_gradinputs_gemm_alt(fgraph, node):
    if not isinstance(node.op, AbstractConv2d_gradInputs):
        return None
    kern, topgrad, shape = node.inputs
    if not isinstance(kern.type, GpuArrayType) or not isinstance(
        topgrad.type, GpuArrayType
    ):
        return None
    border_mode = node.op.border_mode
    subsample = node.op.subsample
    filter_dilation = node.op.filter_dilation
    num_groups = node.op.num_groups
    unshared = node.op.unshared

    if (
        border_mode == "valid"
        and subsample == (1, 1)
        and num_groups == 1
        and not unshared
    ):
        if not node.op.filter_flip:
            kern = kern[:, :, ::-1, ::-1]

        rval = GpuCorrMM(
            border_mode="full", subsample=subsample, filter_dilation=filter_dilation
        )(gpu_contiguous(topgrad), gpu_contiguous(kern.dimshuffle(1, 0, 2, 3)))
        return [rval]
    else:
        return None


@local_optimizer([AbstractConv3d_gradInputs])
def local_abstractconv3d_gradinputs_gemm(fgraph, node):
    if not isinstance(node.op, AbstractConv3d_gradInputs):
        return None
    kern, topgrad, shape = node.inputs
    if not isinstance(kern.type, GpuArrayType) or not isinstance(
        topgrad.type, GpuArrayType
    ):
        return None

    if node.op.filter_flip:
        kern = kern[:, :, ::-1, ::-1, ::-1]

    rval = GpuCorr3dMM_gradInputs(
        border_mode=node.op.border_mode,
        subsample=node.op.subsample,
        filter_dilation=node.op.filter_dilation,
        num_groups=node.op.num_groups,
    )(gpu_contiguous(kern), gpu_contiguous(topgrad), shape)
    return [rval]


@local_optimizer([AbstractConv3d_gradInputs])
def local_abstractconv3d_gradinputs_gemm_alt(fgraph, node):
    if not isinstance(node.op, AbstractConv3d_gradInputs):
        return None
    kern, topgrad, shape = node.inputs
    if not isinstance(kern.type, GpuArrayType) or not isinstance(
        topgrad.type, GpuArrayType
    ):
        return None
    border_mode = node.op.border_mode
    subsample = node.op.subsample
    filter_dilation = node.op.filter_dilation
    num_groups = node.op.num_groups

    if border_mode == "valid" and subsample == (1, 1, 1) and num_groups == 1:
        if not node.op.filter_flip:
            kern = kern[:, :, ::-1, ::-1, ::-1]
        rval = GpuCorr3dMM(
            border_mode="full", subsample=subsample, filter_dilation=filter_dilation
        )(gpu_contiguous(topgrad), gpu_contiguous(kern.dimshuffle(1, 0, 2, 3, 4)))
        return [rval]
    else:
        return None


class ConvMetaOptimizer(LocalMetaOptimizer):
    def __init__(self):
        super().__init__()

    def time_call(self, fn):
        start = time.time()
        fn()[0].sync()
        return time.time() - start

    def provide_inputs(self, node, inputs):
        result = {}

        shapes = (node.op.imshp, node.op.kshp)
        if (
            node.op.imshp is None
            or node.op.kshp is None
            or any([s is None for shape in shapes for s in shape])
        ):
            return result

        if type(node.op) in [AbstractConv2d, AbstractConv3d]:
            img, kern = node.inputs
            for (var, shape) in zip((img, kern), shapes):
                result[var] = aesara.shared(
                    np.random.random(shape).astype(var.dtype),
                    var.name,
                    broadcastable=var.broadcastable,
                    borrow=True,
                )

        if type(node.op) in [AbstractConv2d_gradWeights, AbstractConv3d_gradWeights]:
            img, top, kshape = node.inputs

            tshp = get_conv_output_shape(
                node.op.imshp,
                node.op.kshp,
                node.op.border_mode,
                node.op.subsample,
                node.op.filter_dilation,
            )
            convdim = img.ndim - 2

            result[kshape] = aet.as_tensor_variable(node.op.kshp[-convdim:])

            for (var, shape) in zip((img, top), (node.op.imshp, tshp)):
                result[var] = aesara.shared(
                    np.random.random(shape).astype(var.dtype),
                    var.name,
                    broadcastable=var.broadcastable,
                    borrow=True,
                )

        if type(node.op) in [AbstractConv2d_gradInputs, AbstractConv3d_gradInputs]:
            kern, top, ishape = node.inputs

            tshp = get_conv_output_shape(
                node.op.imshp,
                node.op.kshp,
                node.op.border_mode,
                node.op.subsample,
                node.op.filter_dilation,
            )

            result[ishape] = aet.as_tensor_variable(node.op.imshp[2:])

            for (var, shape) in zip((kern, top), (node.op.kshp, tshp)):
                result[var] = aesara.shared(
                    np.random.random(shape).astype(var.dtype),
                    var.name,
                    broadcastable=var.broadcastable,
                    borrow=True,
                )

        return result

    def get_opts(self, node):
        opts = Counter(
            [
                opt
                for opt in self.track_dict[type(node.op)]
                if opt in self.tag_dict["default"]
            ]
        )
        include_tags = config.metaopt__optimizer_including.split(":")
        exclude_tags = config.metaopt__optimizer_excluding.split(":")

        for in_opt in include_tags:
            opts.update(
                [
                    opt
                    for opt in self.track_dict[type(node.op)]
                    if opt in self.tag_dict[in_opt]
                ]
            )

        for ex_opt in exclude_tags:
            opts.subtract(
                [
                    opt
                    for opt in self.track_dict[type(node.op)]
                    if opt in self.tag_dict[ex_opt]
                ]
            )

        opts = list(opts + Counter())
        return opts


# This deals with any abstract convs that have a transfer somewhere
@register_opt("fast_compile", "conv_dnn", "cudnn")
@op_lifter(
    [
        AbstractConv2d,
        AbstractConv2d_gradWeights,
        AbstractConv2d_gradInputs,
        AbstractConv3d,
        AbstractConv3d_gradWeights,
        AbstractConv3d_gradInputs,
    ]
)
def local_gpua_abstractconv(fgraph, op, context_name, inputs, outputs):
    if isinstance(outputs[0].type, GpuArrayType):
        # Don't handle this node here, it's already on the GPU.
        return
    return local_gpua_lift_abstractconv_graph(fgraph, op, context_name, inputs, outputs)


@register_opt2(
    [
        AbstractConv2d,
        AbstractConv2d_gradWeights,
        AbstractConv2d_gradInputs,
        AbstractConv3d,
        AbstractConv3d_gradWeights,
        AbstractConv3d_gradInputs,
    ],
    "fast_compile",
)
def local_gpua_lift_abstractconv_graph(fgraph, op, context_name, inputs, outputs):
    inps = list(inputs)
    inps[0] = as_gpuarray_variable(inputs[0], context_name=context_name)
    inps[1] = as_gpuarray_variable(inputs[1], context_name=context_name)
    return [op(*inps)]


def local_gpu_pool(fgraph, op, ctx_name, inputs, outputs):
    assert op.__props__ == ("ignore_border", "mode", "ndim")
    inp, ws, stride, pad = inputs
    nd = op.ndim
    if nd not in (2, 3):
        return
    inp = gpu_contiguous(as_gpuarray_variable(inp, ctx_name))

    op = GpuPool(op.ignore_border, op.mode, op.ndim)
    if inp.ndim == nd + 2:
        return op(inp, ws, stride, pad)
    else:
        # reshape to 4D or 5D with 2 non-pooling dimensions
        inp_padded = pad_dims(inp, 2, nd)
        ret_padded = op(inp_padded, ws, stride, pad)
        return unpad_dims(ret_padded, inp, 2, nd)


lifter = op_lifter([pool.Pool])(local_gpu_pool)
pool_db.register(
    "local_gpu_pool", lifter, "gpuarray", "fast_compile", "fast_run", position=1
)
pool_db2.register(
    "local_gpu_pool",
    local_optimizer([pool.Pool])(local_gpu_pool),
    "gpuarray",
    "fast_compile",
    "fast_run",
    position=1,
)
register_opt("fast_compile", name="pool_db")(pool_db)
register_opt2([pool.Pool], "fast_compile", name="pool_db2")(pool_db2)


def local_gpu_max_pool_grad(fgraph, op, ctx_name, inputs, outputs):
    assert op.__props__ == ("ignore_border", "mode", "ndim")

    inp, out, out_grad, ws, stride, pad = inputs
    nd = op.ndim
    if nd not in (2, 3):
        return
    inp = gpu_contiguous(as_gpuarray_variable(inp, ctx_name))
    out = gpu_contiguous(as_gpuarray_variable(out, ctx_name))
    out_grad = gpu_contiguous(as_gpuarray_variable(out_grad, ctx_name))

    op = GpuMaxPoolGrad(op.ignore_border, op.mode, op.ndim)
    if inp.ndim == nd + 2:
        return op(inp, out, out_grad, ws, stride, pad)
    else:
        # reshape to 4D or 5D with 2 non-pooling dimensions
        inp_padded = pad_dims(inp, 2, nd)
        out_padded = pad_dims(out, 2, nd)
        out_grad_padded = pad_dims(out_grad, 2, nd)
        ret_padded = op(inp_padded, out_padded, out_grad_padded, ws, stride, pad)
        return unpad_dims(ret_padded, inp, 2, nd)


lifter = op_lifter([pool.MaxPoolGrad])(local_gpu_max_pool_grad)
pool_db.register(
    "local_gpu_max_pool_grad",
    lifter,
    "gpuarray",
    "fast_compile",
    "fast_run",
    position=1,
)
pool_db2.register(
    "local_gpu_max_pool_grad",
    local_optimizer([pool.MaxPoolGrad])(local_gpu_max_pool_grad),
    "gpuarray",
    "fast_compile",
    "fast_run",
    position=1,
)


def local_gpu_average_pool_grad(fgraph, op, ctx_name, inputs, outputs):
    assert op.__props__ == ("ignore_border", "mode", "ndim")

    inp, out_grad, ws, stride, pad = inputs
    nd = op.ndim
    if nd not in (2, 3):
        return
    inp = gpu_contiguous(as_gpuarray_variable(inp, ctx_name))
    out_grad = gpu_contiguous(as_gpuarray_variable(out_grad, ctx_name))

    op = GpuAveragePoolGrad(op.ignore_border, op.mode, op.ndim)
    if inp.ndim == nd + 2:
        return op(inp, out_grad, ws, stride, pad)
    else:
        # reshape to 4D or 5D with 2 non-pooling dimensions
        inp_padded = pad_dims(inp, 2, nd)
        out_grad_padded = pad_dims(out_grad, 2, nd)
        ret_padded = op(inp_padded, out_grad_padded, ws, stride, pad)
        return unpad_dims(ret_padded, inp, 2, nd)


lifter = op_lifter([pool.AveragePoolGrad])(local_gpu_average_pool_grad)
pool_db.register(
    "local_gpu_average_pool_grad",
    lifter,
    "gpuarray",
    "fast_compile",
    "fast_run",
    position=1,
)
pool_db2.register(
    "local_gpu_average_pool_grad",
    local_optimizer([pool.AveragePoolGrad])(local_gpu_average_pool_grad),
    "gpuarray",
    "fast_compile",
    "fast_run",
    position=1,
)


@register_opt()
@op_lifter([pool.DownsampleFactorMaxGradGrad])
@register_opt2([pool.DownsampleFactorMaxGradGrad])
def local_gpu_downsample_factor_max_grad_grad(fgraph, op, ctx_name, inputs, outputs):
    assert op.__props__ == ("ignore_border", "mode", "ndim")
    inp, out, out_grad, ws, stride, pad = inputs
    nd = op.ndim
    if nd not in (2, 3):
        return
    inp = gpu_contiguous(as_gpuarray_variable(inp, ctx_name))
    out = gpu_contiguous(as_gpuarray_variable(out, ctx_name))
    out_grad = gpu_contiguous(as_gpuarray_variable(out_grad, ctx_name))

    op = GpuDownsampleFactorMaxGradGrad(op.ignore_border, op.mode, op.ndim)
    if inp.ndim == nd + 2:
        return op(inp, out, out_grad, ws, stride, pad)
    else:
        # reshape to 4D or 5D with 2 non-pooling dimensions
        inp_padded = pad_dims(inp, 2, nd)
        out_padded = pad_dims(out, 2, nd)
        out_grad_padded = pad_dims(out_grad, 2, nd)
        ret_padded = op(inp_padded, out_padded, out_grad_padded, ws, stride, pad)
        return unpad_dims(ret_padded, inp, 2, nd)


@register_opt()
@op_lifter([pool.MaxPoolRop])
@register_opt2([pool.MaxPoolRop])
def local_gpu_max_pool_rop(fgraph, op, ctx_name, inputs, outputs):
    assert op.__props__ == ("ignore_border", "mode", "ndim")
    inp, eval_inp, ws, stride, pad = inputs
    nd = op.ndim
    if nd not in (2, 3):
        return
    inp = gpu_contiguous(as_gpuarray_variable(inp, ctx_name))
    eval_inp = gpu_contiguous(as_gpuarray_variable(eval_inp, ctx_name))

    op = GpuMaxPoolRop(op.ignore_border, op.mode, op.ndim)
    if inp.ndim == nd + 2:
        return op(inp, eval_inp, ws, stride, pad)
    else:
        # reshape to 4D or 5D with 2 non-pooling dimensions
        inp_padded = pad_dims(inp, 2, nd)
        eval_inp_padded = pad_dims(eval_inp, 2, nd)
        ret_padded = op(inp_padded, eval_inp_padded, ws, stride, pad)
        return unpad_dims(ret_padded, inp, 2, nd)


@register_opt("low_memory")
@local_optimizer([GpuCAReduceCuda])
def local_gpu_elemwise_careduce(fgraph, node):
    """
    Merge some GpuCAReduceCuda and GPUElemwise.
    Currently merged:
     - SUM(X^2)
     - SUM(ABS(X))

    """
    if (
        isinstance(node.op, GpuCAReduceCuda)
        and node.op.pre_scalar_op is None
        and node.inputs[0].owner
        and isinstance(node.inputs[0].owner.op, GpuElemwise)
        and
        # The Op support all scalar with 1 inputs.  We don't
        # automatically add more case, as some like trigonometic
        # operation with some reduction pattern will probably results
        # in slow down.
        isinstance(node.inputs[0].owner.op.scalar_op, (aes.Sqr, aes.Abs))
    ):
        inp = node.inputs[0].owner.inputs[0]
        props = node.op._props_dict()
        props["pre_scalar_op"] = node.inputs[0].owner.op.scalar_op
        with inherit_stack_trace(node.outputs):
            out = GpuCAReduceCuda(**props)(inp)
            return [out]


@local_optimizer(None)
def local_assert_no_cpu_op(fgraph, node):
    if all(
        [var.owner and isinstance(var.owner.op, HostFromGpu) for var in node.inputs]
    ) and any(
        [
            [c for c in fgraph.clients[var] if isinstance(c[0].op, GpuFromHost)]
            for var in node.outputs
        ]
    ):

        if config.assert_no_cpu_op == "warn":
            _logger.warning(f"CPU Op {node} is detected in the computation graph")
        elif config.assert_no_cpu_op == "raise":
            raise AssertionError(f"The Op {node} is on CPU.")
        elif config.assert_no_cpu_op == "pdb":
            pdb.set_trace()


# Register the local_assert_no_cpu_op:
assert_no_cpu_op = aesara.graph.opt.in2out(
    local_assert_no_cpu_op, name="assert_no_cpu_op"
)
# 49.2 is after device specialization & fusion optimizations for last transfers
optdb.register("gpua_assert_no_cpu_op", assert_no_cpu_op, 49.2, "assert_no_cpu_op")


def tensor_to_gpu(x, context_name):
    if isinstance(x.type, TensorType):
        y = GpuArrayType(
            broadcastable=x.type.broadcastable,
            context_name=context_name,
            dtype=x.type.dtype,
        )()
        if x.name:
            y.name = x.name + "[Gpua]"
        return y
    else:
        return x


def gpu_safe_new(x, tag=""):
    """
    Internal function that constructs a new variable from x with the same
    type, but with a different name (old name + tag). This function is used
    by gradient, or the R-op to construct new variables for the inputs of
    the inner graph such that there is no interference between the original
    graph and the newly constructed graph.

    """
    if hasattr(x, "name") and x.name is not None:
        nw_name = x.name + tag
    else:
        nw_name = None

    if isinstance(x, Constant):
        return x.clone()

    nw_x = x.type()
    nw_x.name = nw_name
    return nw_x


def gpu_reconstruct_graph(inputs, outputs, tag=None):
    """
    Different interface to clone, that allows you to pass inputs.
    Compared to clone, this method always replaces the inputs with
    new variables of the same type, and returns those (in the same
    order as the original inputs).

    """
    if tag is None:
        tag = ""
    nw_inputs = [gpu_safe_new(x, tag) for x in inputs]
    givens = {}
    for nw_x, x in zip(nw_inputs, inputs):
        givens[x] = nw_x
    nw_outputs = clone_replace(outputs, replace=givens)
    return (nw_inputs, nw_outputs)


@register_opt("scan", "fast_compile")
@op_lifter([Scan])
@register_opt2([Scan], "fast_compile")
def local_gpua_scan_to_gpua(fgraph, op, context_name, inputs, outputs):
    info = copy.deepcopy(op.info)
    if info.get("gpua", False):
        return
    info["gpua"] = True
    nw_ins = [inputs[0]]
    e = 1 + op.n_seqs + op.n_mit_mot + op.n_mit_sot + op.n_sit_sot + op.n_shared_outs
    nw_ins += [safe_to_gpu(x, context_name) for x in inputs[1:e]]
    b = e
    e = e + op.n_nit_sot
    nw_ins += inputs[b:e]
    nw_ins += [safe_to_gpu(x, context_name) for x in inputs[e:]]
    scan_ins = [tensor_to_gpu(x, context_name) for x in op.inputs]

    # The inner output corresponding to the looping condition should not be
    # moved to the gpu
    if op.info["as_while"]:
        scan_outs = [safe_to_gpu(x, context_name) for x in op.outputs[:-1]]
        scan_outs += [op.outputs[-1]]
    else:
        scan_outs = [safe_to_gpu(x, context_name) for x in op.outputs]
    scan_outs = clone_replace(
        scan_outs, replace=list(zip(op.inputs, (safe_to_cpu(x) for x in scan_ins)))
    )

    # We need to construct the hash here, because scan
    # __init__ does not know about the gpu and can not
    # handle graphs with inputs being on the gpu
    tmp_in, tmp_out = gpu_reconstruct_graph(scan_ins, scan_outs)
    local_fgraph = FunctionGraph(tmp_in, tmp_out, clone=True)
    _cmodule_key = CLinker().cmodule_key_(local_fgraph, [])
    info["gpu_hash"] = hash(_cmodule_key)

    def typebuild(dtype, broadcastable, context_name=context_name):
        return GpuArrayType(
            dtype=dtype, broadcastable=broadcastable, context_name=context_name
        )

    nw_op = Scan(scan_ins, scan_outs, info, typeConstructor=typebuild).make_node(
        *nw_ins
    )
    return nw_op.outputs


def _scan_type_infer(node):
    context_name = infer_context_name(*node.inputs)

    def typebuild(dtype, broadcastable, context_name=context_name):
        return GpuArrayType(
            dtype=dtype, broadcastable=broadcastable, context_name=context_name
        )

    return typebuild


# Add optimization : maxandargmax (CPU -> GPU)
@register_opt("fast_compile")
@op_lifter([MaxAndArgmax])
@register_opt2([MaxAndArgmax], "fast_compile")
def local_gpu_maxandargmax(fgraph, op, context_name, inputs, outputs):
    op = GpuMaxAndArgmax(op.get_params(None))
    if inputs[0].dtype == "float16":
        # For now it is better to copy/cast on the GPU then transfer to the CPU
        casted_inputs = inputs[0].astype("float32")
        ret = op(casted_inputs)
        return [ret[0].astype("float16"), ret[1]]
    return op


@register_opt("fast_compile")
@op_lifter([Images2Neibs])
@register_opt2([Images2Neibs], "fast_compile")
def local_gpua_images2neibs(fgraph, op, context_name, inputs, outputs):
    if op.mode in ["valid", "half", "full", "ignore_borders", "wrap_centered"]:
        return GpuImages2Neibs(op.mode)


# solve
@register_opt("fast_compile")
@op_lifter([slinalg.Solve])
@register_opt2([aesara.tensor.slinalg.Solve], "fast_compile")
def local_gpu_solve(fgraph, op, context_name, inputs, outputs):
    if inputs[0].dtype not in ["float16", "float32", "float64"]:
        return
    if op.A_structure not in MATRIX_STRUCTURES_SOLVE:
        return

    if op.A_structure in ["lower_triangular", "upper_triangular"]:
        if not cublas_available:
            return
        lower = op.A_structure == "lower_triangular"
        op = GpuCublasTriangularSolve(lower)
    else:
        if not cusolver_available:
            return
        op = GpuCusolverSolve(A_structure=op.A_structure)

    if inputs[0].dtype == "float16":
        return op(inputs[0].astype("float32"), inputs[1].astype("float32")).astype(
            "float16"
        )
    return op


@register_inplace()
@local_optimizer([GpuCusolverSolve], inplace=True)
def local_inplace_gpu_solve(fgraph, node):
    if isinstance(node.op, GpuCusolverSolve) and not node.op.inplace:
        with inherit_stack_trace(node.outputs):
            return [
                GpuCusolverSolve(
                    A_structure=node.op.A_structure, trans=node.op.trans, inplace=True
                )(*node.inputs)
            ]


# Cholesky decomposition
def local_gpu_cholesky(fgraph, op, context_name, inputs, outputs):
    if not cusolver_available:
        return
    if inputs[0].dtype not in ["float16", "float32", "float64"]:
        return
    op = GpuCholesky(lower=op.lower, inplace=op.destructive)
    if inputs[0].dtype == "float16":
        return op(inputs[0].astype("float32")).astype("float16")

    return op


# For Cholesky decomposition, magma 2.2 is slower than cusolver 8 (tested for
# matrices of size 1000). Thus, cusolver is prioritized during graph
# optimizations. To explicitly use magma, you should disable cusolver using
# `optimizer_excluding=cusolver` in Aesara config.
lifter = op_lifter([slinalg.Cholesky])(local_gpu_cholesky)
matrix_ops_db.register(
    "local_gpu_cholesky",
    lifter,
    "gpuarray",
    "fast_compile",
    "fast_run",
    "cusolver",
    position=0,
)
matrix_ops_db2.register(
    "local_gpu_cholesky",
    local_optimizer([slinalg.Cholesky])(local_gpu_cholesky),
    "gpuarray",
    "fast_compile",
    "fast_run",
    "cusolver",
    position=0,
)
register_opt("fast_compile", name="matrix_ops_db")(matrix_ops_db)
register_opt2([slinalg.Solve], "fast_compile", name="matrix_ops_db2")(matrix_ops_db2)


@register_inplace()
@local_optimizer([GpuCholesky], inplace=True)
def local_inplace_gpu_cholesky(fgraph, node):
    if isinstance(node.op, GpuCholesky) and not node.op.inplace:
        with inherit_stack_trace(node.outputs):
            return [node.op.clone_inplace()(*node.inputs)]


def local_gpu_magma_cholesky(fgraph, op, context_name, inputs, outputs):
    if not config.magma__enabled:
        return
    if inputs[0].dtype not in ["float16", "float32"]:
        return
    op = GpuMagmaCholesky(lower=op.lower, inplace=op.destructive)
    if inputs[0].dtype == "float16":
        return op(inputs[0].astype("float32")).astype("float16")
    return op


lifter = op_lifter([slinalg.Cholesky])(local_gpu_magma_cholesky)
matrix_ops_db.register(
    "local_gpu_magma_cholesky",
    lifter,
    "gpuarray",
    "fast_compile",
    "fast_run",
    "magma",
    position=1,
)
matrix_ops_db2.register(
    "local_gpu_magma_cholesky",
    local_optimizer([slinalg.Cholesky])(local_gpu_magma_cholesky),
    "gpuarray",
    "fast_compile",
    "fast_run",
    "magma",
    position=1,
)


@register_inplace()
@local_optimizer([GpuMagmaCholesky], inplace=True)
def local_inplace_gpu_magma_cholesky(fgraph, node):
    if isinstance(node.op, GpuMagmaCholesky) and not node.op.inplace:
        return [node.op.clone_inplace()(*node.inputs)]


# QR decomposition
@register_opt("magma", "fast_compile")
@op_lifter([nlinalg.QRFull])
@register_opt2([nlinalg.QRFull], "magma", "fast_compile")
def local_gpu_magma_qr(fgraph, op, context_name, inputs, outputs):
    if not config.magma__enabled or op.mode != "reduced":
        return
    if inputs[0].dtype not in ["float16", "float32"]:
        return
    x = inputs[0]
    if inputs[0].dtype == "float16":
        x = inputs[0].astype("float32")
    out = gpu_qr(x, complete=True)
    if inputs[0].dtype == "float16":
        return [o.astype("float16") for o in out]
    return out


# Matrix inverse
@register_opt("magma", "fast_compile")
@op_lifter([nlinalg.MatrixInverse])
@register_opt2([nlinalg.MatrixInverse], "magma", "fast_compile")
def local_gpu_magma_matrix_inverse(fgraph, op, context_name, inputs, outputs):
    if not config.magma__enabled:
        return
    if inputs[0].dtype not in ["float16", "float32"]:
        return
    op = GpuMagmaMatrixInverse()
    if inputs[0].dtype == "float16":
        return op(inputs[0].astype("float32")).astype("float16")
    return op


@register_inplace()
@local_optimizer([GpuMagmaMatrixInverse])
def local_inplace_gpu_magma_matrix_inverse(fgraph, node):
    if isinstance(node.op, GpuMagmaMatrixInverse) and not node.op.inplace:
        with inherit_stack_trace(node.outputs):
            return [node.op.clone_inplace()(*node.inputs)]


# Eigen decomposition of a symmetric matrix
@register_opt("magma", "fast_compile")
@op_lifter([nlinalg.Eigh])
@register_opt2([nlinalg.Eigh], "magma", "fast_compile")
def local_gpu_magma_eigh(fgraph, op, context_name, inputs, outputs):
    if not config.magma__enabled:
        return
    if inputs[0].dtype not in ["float16", "float32"]:
        return
    op = GpuMagmaEigh(UPLO=op.UPLO, compute_v=True)
    if inputs[0].dtype == "float16":
        return op(inputs[0].astype("float32")).astype("float16")
    return op


# Singular Value Decomposition
@register_opt("magma", "fast_compile")
@op_lifter([nlinalg.SVD])
@register_opt2([nlinalg.SVD], "magma", "fast_compile")
def local_gpu_magma_svd(fgraph, op, context_name, inputs, outputs):
    if not config.magma__enabled:
        return
    if inputs[0].dtype not in ["float16", "float32"]:
        return
    x = inputs[0]
    if inputs[0].dtype == "float16":
        x = inputs[0].astype("float32")
    out = gpu_svd(x, compute_uv=op.compute_uv, full_matrices=op.full_matrices)
    if inputs[0].dtype == "float16":
        if op.compute_uv:
            out = [o.astype("float16") for o in out]
        else:
            out = [out.astype("float16")]
    return out


@register_opt("ctc", "fast_compile")
@op_lifter([aesara.tensor.nnet.ctc.ConnectionistTemporalClassification])
@register_opt2([ConnectionistTemporalClassification], "ctc", "fast_compile")
def local_gpu_ctc(fgraph, op, context_name, inputs, outputs):
    op = GpuConnectionistTemporalClassification(compute_grad=op.compute_grad)
    return op.make_node(*inputs).outputs


# Do not register in fast_run or fast_compile.
# It will be added to fast_run if the GPU is enabled.
optdb.register(
    "gpua_scanOp_make_inplace",
    ScanInplaceOptimizer(typeInfer=_scan_type_infer, gpua_flag=True),
    75,
    "gpuarray",
    "inplace",
    "scan",
)

abstractconv_groupopt.register(
    "local_abstractconv_dnn",
    local_abstractconv_cudnn,
    20,
    "conv_dnn",
    "gpuarray",
    "fast_compile",
    "fast_run",
    "cudnn",
)
abstractconv_groupopt.register(
    "local_abstractconv_gw_dnn",
    local_abstractconv_gw_cudnn,
    20,
    "conv_dnn",
    "gpuarray",
    "fast_compile",
    "fast_run",
    "cudnn",
)
abstractconv_groupopt.register(
    "local_abstractconv_gi_dnn",
    local_abstractconv_gi_cudnn,
    20,
    "conv_dnn",
    "gpuarray",
    "fast_compile",
    "fast_run",
    "cudnn",
)
# The GEMM-based convolution comes last to catch all remaining cases.
# It can be disabled by excluding 'conv_gemm'.
abstractconv_groupopt.register(
    "local_abstractconv_gemm",
    local_abstractconv_gemm,
    30,
    "conv_gemm",
    "gpuarray",
    "fast_compile",
    "fast_run",
)
abstractconv_groupopt.register(
    "local_abstractconv3d_gemm",
    local_abstractconv3d_gemm,
    30,
    "conv_gemm",
    "gpuarray",
    "fast_compile",
    "fast_run",
)
abstractconv_groupopt.register(
    "local_abstractconv_gradweights_gemm",
    local_abstractconv_gradweights_gemm,
    30,
    "conv_gemm",
    "gpuarray",
    "fast_compile",
    "fast_run",
)
abstractconv_groupopt.register(
    "local_abstractconv3d_gradweights_gemm",
    local_abstractconv3d_gradweights_gemm,
    30,
    "conv_gemm",
    "gpuarray",
    "fast_compile",
    "fast_run",
)
abstractconv_groupopt.register(
    "local_abstractconv_gradinputs",
    local_abstractconv_gradinputs_gemm,
    30,
    "conv_gemm",
    "gpuarray",
    "fast_compile",
    "fast_run",
)
abstractconv_groupopt.register(
    "local_abstractconv3d_gradinputs",
    local_abstractconv3d_gradinputs_gemm,
    30,
    "conv_gemm",
    "gpuarray",
    "fast_compile",
    "fast_run",
)

conv_metaopt = ConvMetaOptimizer()

conv_metaopt.register(local_abstractconv_cudnn, ["default", "cudnn", "conv_dnn"])
conv_metaopt.register(local_abstractconv_gw_cudnn, ["default", "cudnn", "conv_dnn"])
conv_metaopt.register(local_abstractconv_gi_cudnn, ["default", "cudnn", "conv_dnn"])
conv_metaopt.register(local_abstractconv_gemm_def, ["default", "conv_gemm"])
conv_metaopt.register(local_abstractconv3d_gemm_def, ["default", "conv_gemm"])
conv_metaopt.register(local_abstractconv_gradweights_gemm, ["default", "conv_gemm"])
conv_metaopt.register(local_abstractconv3d_gradweights_gemm, ["default", "conv_gemm"])
conv_metaopt.register(local_abstractconv_gradinputs_gemm, ["default", "conv_gemm"])
conv_metaopt.register(local_abstractconv3d_gradinputs_gemm, ["default", "conv_gemm"])
conv_metaopt.register(
    local_abstractconv_gemm_alt, ["default", "alternative", "conv_gemm"]
)
conv_metaopt.register(
    local_abstractconv_gemm_gradweights_alt, ["default", "alternative", "conv_gemm"]
)
conv_metaopt.register(
    local_abstractconv_gradinputs_gemm_alt, ["default", "alternative", "conv_gemm"]
)
conv_metaopt.register(
    local_abstractconv_cudnn_alt, ["default", "alternative", "cudnn", "conv_dnn"]
)
conv_metaopt.register(
    local_abstractconv3d_cudnn_alt, ["default", "alternative", "cudnn", "conv_dnn"]
)
conv_metaopt.register(local_abstractconv3d_alt, ["default", "alternative", "conv_gemm"])
conv_metaopt.register(
    local_abstractconv3d_gemm_gradweights_alt, ["default", "alternative", "conv_gemm"]
)
conv_metaopt.register(
    local_abstractconv3d_gradinputs_gemm_alt, ["default", "alternative", "conv_gemm"]
)
conv_metaopt.register(local_abstractconv3d2d, ["alternative", "conv3d2d"])

abstractconv_groupopt.register("conv_metaopt", conv_metaopt, "conv_meta", position=0)

# Register cuDNN batch normalization implementation

# We import these opts here instead of at the top of this file
# to avoid a circular dependency problem with dnn
from aesara.gpuarray.dnn import (  # noqa: E402
    local_abstract_batch_norm_inference_cudnn,
    local_abstract_batch_norm_train_cudnn,
    local_abstract_batch_norm_train_grad_cudnn,
)


register_opt("fast_compile")(abstract_batch_norm_groupopt)

register_opt("fast_compile", name="abstract_batch_norm_db")(abstract_batch_norm_db)
register_opt2(
    [
        batchnorm.AbstractBatchNormTrain,
        batchnorm.AbstractBatchNormTrainGrad,
        batchnorm.AbstractBatchNormInference,
    ],
    "fast_compile",
    name="abstract_batch_norm_db2",
)(abstract_batch_norm_db2)

for op, fct, cpu in [
    (
        batchnorm.AbstractBatchNormTrain,
        local_abstract_batch_norm_train_cudnn,
        batchnorm.local_abstract_batch_norm_train,
    ),
    (
        batchnorm.AbstractBatchNormTrainGrad,
        local_abstract_batch_norm_train_grad_cudnn,
        batchnorm.local_abstract_batch_norm_train_grad,
    ),
    (
        batchnorm.AbstractBatchNormInference,
        local_abstract_batch_norm_inference_cudnn,
        batchnorm.local_abstract_batch_norm_inference,
    ),
]:
    lifter = op_lifter([op])(fct)
    abstract_batch_norm_db.register(
        fct.__name__,
        lifter,
        "gpuarray",
        "fast_compile",
        "fast_run",
        "cudnn",
        "batchnorm_dnn",
        position=1,
    )
    abstract_batch_norm_db2.register(
        fct.__name__,
        local_optimizer([op])(fct),
        "gpuarray",
        "fast_compile",
        "fast_run",
        "cudnn",
        "batchnorm_dnn",
        position=1,
    )
    # cpu is a normal optimization. We can't register it in
    # GraphToGPU.  So for now, only add it to the slower EQ phase.  If
    # there is no cuDNN, we still want to move it to the GPU now with
    # an Aesara graph so to have this graph on the GPU.
    abstract_batch_norm_db.register(
        cpu.__name__, cpu, "gpuarray", "fast_compile", "fast_run", position="last"
    )
