import os
import sys
from typing import Any, Optional

from typing_extensions import Protocol

import aesara.tensor as aet
from aesara.configdefaults import config
from aesara.gradient import grad_undefined
from aesara.graph.basic import Apply
from aesara.graph.op import ExternalCOp, OpenMPOp
from aesara.graph.opt import local_optimizer
from aesara.link.c.cmodule import GCC_compiler
from aesara.tensor.basic_opt import register_canonicalize
from aesara.tensor.blas import batched_dot
from aesara.tensor.extra_ops import cpu_contiguous
from aesara.tensor.type import ftensor3, fvector


def _ctc_find_lib():
    """
    Find the directory that contains libwarpctc.so
    """
    if config.ctc__root != "":
        for lib_dir in ["build", "lib", "lib64"]:
            lib_path = os.path.join(config.ctc__root, lib_dir)
            if os.path.isdir(lib_path) and os.path.exists(lib_path):
                lib_found = os.path.exists(os.path.join(lib_path, "libwarpctc.so"))
                if lib_found:
                    return lib_path
    return None


def _ctc_check_compile(ctc_lib_path):
    preambule = """
#include <string.h>
#include "ctc.h"
"""

    body = """
ctcOptions options;
memset(&options, 0, sizeof(ctcOptions));
options.loc = CTC_CPU;
options.num_threads = 1;
"""

    params = [f"-I{os.path.dirname(__file__)}"]
    if ctc_lib_path is not None:
        params.extend([f"-I{os.path.join(config.ctc__root, 'include')}"])
        params.extend([f"-L{ctc_lib_path}"])
    params.extend(["-l", "warpctc"])
    compiler_res = GCC_compiler.try_flags(
        params, preambule=preambule, body=body, try_run=False, output=True
    )

    avail, out, err = (
        compiler_res if isinstance(compiler_res, tuple) else (compiler_res, None, None)
    )
    if not avail:
        return (
            False,
            ("cannot compile with warp-ctc. " "We got this error:\n" + str(err)),
        )
    return True, None


class Ctc_present(Protocol):
    avail: Optional[bool]
    msg: Optional[str]
    path: Optional[str]


def ctc_present_decorator(func: Any) -> Ctc_present:
    return func


@ctc_present_decorator
def ctc_present():
    if ctc_present.avail is not None:
        return ctc_present.avail
    ctc_lib_path = _ctc_find_lib()
    ctc_present.path = ctc_lib_path
    ctc_present.avail, ctc_present.msg = _ctc_check_compile(ctc_present.path)
    return ctc_present.avail


ctc_present.avail = None
ctc_present.msg = None
ctc_present.path = None


class Ctc_available(Protocol):
    avail: Optional[bool]
    msg: Optional[str]
    path: Optional[str]


def ctc_available_decorator(func: Any) -> Ctc_available:
    return func


@ctc_available_decorator
def ctc_available():
    if os.name == "nt":
        ctc_available.msg = ("Windows platforms are currently not supported ",)
        "by underlying CTC library (warp-ctc)."
        return False
    elif not ctc_present():
        ctc_available.msg = ctc_present.msg
        return False

    ctc_available.path = ctc_present.path
    return True


ctc_available.msg = None
ctc_available.path = None


class ConnectionistTemporalClassification(ExternalCOp, OpenMPOp):
    """
    CTC loss function wrapper.

    Notes
    -----
    Using the wrapper requires that Baidu's warp-ctc library is installed.
    If the warp-ctc library is not on your compiler's default library path,
    you must set the configuration variable ``config.ctc__root`` appropriately.

    Parameters
    ----------
    compute_grad
        If set to True, enables the computation of gradients of the CTC loss function.
    """

    __props__ = ("compute_grad",)

    _cop_num_inputs = 3
    _cop_num_outputs = 2

    func_file = os.path.join("c_code", "ctc_wrapper.c")
    func_name = "APPLY_SPECIFIC(ctc_cost_cpu)"

    def __init__(self, compute_grad=True, openmp=None):
        if not ctc_available():
            raise RuntimeError(
                "Baidu CTC is not available and "
                "ConnectionistTemporalClassification Op "
                "can not be constructed."
            )

        super().__init__(self.func_file, self.func_name)
        OpenMPOp.__init__(self, openmp=openmp)

        self.compute_grad = compute_grad
        # Return only the cost. Gradient will be returned by grad()
        self.default_output = 0

    def c_lib_dirs(self, **kwargs):
        lib_dirs = []
        if ctc_available.path is not None:
            lib_dirs += [ctc_available.path]
        return lib_dirs

    def c_compile_args(self, **kwargs):
        if ctc_available.path is not None:
            if sys.platform != "darwin" and " " in ctc_available.path:
                return ['-Wl,-rpath,"' + ctc_available.path + '"']
            else:
                return ["-Wl,-rpath," + ctc_available.path]
        return []

    def c_libraries(self, **kwargs):
        return ["warpctc"]

    def c_header_dirs(self, **kwargs):
        header_dirs = []
        if config.ctc__root != "":
            # We assume here that the header is available at the include directory
            # of the CTC root directory.
            header_dirs += [os.path.join(config.ctc__root, "include")]
        return header_dirs

    def c_headers(self, **kwargs):
        return ["ctc.h"] + super().c_headers(**kwargs)

    def make_node(self, activations, labels, input_lengths):
        t_activations = aet.as_tensor_variable(activations)
        # Ensure activations array is C-contiguous
        t_activations = cpu_contiguous(t_activations)

        t_labels = aet.as_tensor_variable(labels)
        t_input_lengths = aet.as_tensor_variable(input_lengths)

        if t_activations.type.dtype != "float32":
            raise TypeError("activations must use the float32 type!")

        if t_activations.ndim != 3:
            raise ValueError("activations must have 3 dimensions.")

        if t_labels.type.dtype != "int32":
            raise TypeError("labels must use the int32 type!")

        if t_labels.ndim != 2:
            raise ValueError("labels must have 2 dimensions.")

        if t_input_lengths.type.dtype != "int32":
            raise TypeError("input_lengths must use the int32 type!")

        if t_input_lengths.ndim != 1:
            raise ValueError("input_lengths must have 1 dimension.")

        costs = fvector(name="ctc_cost")
        outputs = [costs]
        if self.compute_grad:
            gradients = ftensor3(name="ctc_grad")
            outputs += [gradients]

        return Apply(
            self, inputs=[t_activations, t_labels, t_input_lengths], outputs=outputs
        )

    def L_op(self, inputs, outputs, output_grads):
        assert self.compute_grad and len(outputs) == 2
        gradients = outputs[1]
        assert gradients is not None

        grad_op = output_grads[0]
        total_grad = batched_dot(grad_op, gradients.dimshuffle(1, 0, 2)).dimshuffle(
            1, 0, 2
        )
        return [
            total_grad,
            grad_undefined(self, 1, inputs[1]),
            grad_undefined(self, 2, inputs[2]),
        ]


def ctc(activations, labels, input_lengths):
    """
    Compute CTC loss function.

    Notes
    -----
    Using the loss function requires that the Baidu's warp-ctc library be installed.
    If the warp-ctc library is not on the compiler's default library path, the
    configuration variable ``config.ctc__root`` must be properly set.

    Parameters
    ----------
    activations
        Three-dimensional tensor, which has a shape of (t, m, p), where
        t is the time index, m is the minibatch index, and p is the index
        over the probabilities of each symbol in the alphabet. The memory
        layout is assumed to be in C-order, which consists in the slowest
        to the fastest changing dimension, from left to right. In this case,
        p is the fastest changing dimension.
    labels
        A 2-D tensor of all the labels for the minibatch. In each row, there
        is a sequence of target labels. Negative values are assumed to be padding,
        and thus are ignored. Blank symbol is assumed to have index 0 in the
        alphabet.
    input_lengths
        A 1-D tensor with the number of time steps for each sequence in
        the minibatch.

    Returns
    -------
    1-D array
        Cost of each example in the minibatch.
    """
    return ConnectionistTemporalClassification()(activations, labels, input_lengths)


# Disable gradient computation if not needed
@register_canonicalize("fast_compile")
@local_optimizer([ConnectionistTemporalClassification])
def local_ctc_no_grad(fgraph, node):
    if isinstance(node.op, ConnectionistTemporalClassification):
        if len(node.outputs) > 1:
            if len(fgraph.clients[node.outputs[1]]) == 0:  # gradient is not used
                return [
                    ConnectionistTemporalClassification(compute_grad=False)(
                        *node.inputs
                    ),
                    None,
                ]
    return False
