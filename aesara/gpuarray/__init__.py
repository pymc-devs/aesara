import logging
import os
import sys
import warnings
from typing import Any, Callable, Dict, Optional, TypeVar

from typing_extensions import Protocol

import aesara
from aesara.compile import optdb
from aesara.configdefaults import config
from aesara.tensor.basic import register_transfer


_logger_name = "aesara.gpuarray"
_logger = logging.getLogger(_logger_name)

error = _logger.error
info = _logger.info

pygpu_activated = False
# Used to skip initialization checking when we are in the same processus.
aesara_gpu_is_already_active = False
try:
    import pygpu
    import pygpu.gpuarray
except ImportError:
    pygpu = None

from aesara.gpuarray import (
    ctc,
    dnn,
    extra_ops,
    fft,
    multinomial,
    opt,
    reduction,
    rng_mrg,
    sort,
)
from aesara.gpuarray.basic_ops import as_gpuarray_variable

# This is for documentation not to depend on the availability of pygpu
from aesara.gpuarray.type import (
    ContextNotDefined,
    GpuArrayConstant,
    GpuArraySharedVariable,
    GpuArrayType,
    GpuArrayVariable,
    get_context,
    gpuarray_shared_constructor,
    reg_context,
)


def transfer(x, target):
    try:
        get_context(target)
        return as_gpuarray_variable(x, target)
    except ContextNotDefined:
        pass


register_transfer(transfer)


def pygpu_parse_version(version_string):
    from collections import namedtuple

    version_type = namedtuple(
        "version_type", ("major", "minor", "patch", "fullversion")
    )
    pieces = version_string.split(".", 2)
    assert len(pieces) == 3, version_string
    major = int(pieces[0])
    minor = int(pieces[1])
    if "+" in pieces[2]:  # It contain a git commit.
        patch = int(pieces[2].split("+", 1)[0])
    else:  # Maybe it end with .devN
        patch = int(pieces[2].split(".", 1)[0])
    fullversion = f"{int(major)}.{int(minor)}.{pieces[2]}"
    return version_type(major=major, minor=minor, patch=patch, fullversion=fullversion)


F = TypeVar("F", bound=Callable[..., object])


class Init_dev(Protocol[F]):
    devmap: Dict
    __call__: F


def init_dev_decorator(func: Any) -> Init_dev:
    return func


@init_dev_decorator
def init_dev(dev, name: Optional[str] = None, preallocate: Optional[int] = None):
    global pygpu_activated
    global aesara_gpu_is_already_active
    if (
        not aesara_gpu_is_already_active
        and os.environ.get("AESARA_GPU_IS_ALREADY_ACTIVE", "") == "Yes"
    ):
        raise RuntimeError(
            "You can't initialize the GPU in a subprocess if the parent process already did it"
        )
    if not config.cxx:
        raise RuntimeError("The new gpu-backend need a c++ compiler.")
    pygpu_version = pygpu_parse_version(pygpu.__version__)
    if pygpu_version.major != 0 or pygpu_version.minor != 7 or pygpu_version.patch < 0:
        raise ValueError(
            "Your installed version of pygpu(%s) is too old, please upgrade to 0.7.0 or later (but below 0.8.0)"
            % pygpu_version.fullversion
        )
    # This is for the C headers API, we need to match the exact version.
    gpuarray_version_major_supported = 2
    gpuarray_version_major_detected = pygpu.gpuarray.api_version()[0]
    if gpuarray_version_major_detected != gpuarray_version_major_supported:
        raise ValueError(
            "Your installed version of libgpuarray is not in sync with the current Aesara"
            f" version. The installed libgpuarray version supports API version {int(gpuarray_version_major_detected)},"
            f" while current Aesara supports API version {int(gpuarray_version_major_supported)}. Change the version of"
            " libgpuarray or Aesara to fix this problem.",
        )
    if dev not in init_dev.devmap:
        args = dict()
        if config.gpuarray__cache_path != "":
            args["kernel_cache_path"] = config.gpuarray__cache_path
        if preallocate is None:
            preallocate = config.gpuarray__preallocate
        if preallocate < 0:
            args["max_cache_size"] = 0
        else:
            args["initial_cache_size"] = preallocate
        context = pygpu.init(
            dev,
            sched=config.gpuarray__sched,
            single_stream=config.gpuarray__single_stream,
            **args,
        )
        os.environ["AESARA_GPU_IS_ALREADY_ACTIVE"] = "Yes"
        aesara_gpu_is_already_active = True
        context.dev = dev
        init_dev.devmap[dev] = context
        reg_context(name, context)

        MB = 1024 * 1024
        if dev.startswith("cuda"):
            avail = dnn.dnn_available(name)
            # If we try to enable cudnn and there isn't enough GPU
            # memory, there will be an unclear error message. So do
            # not even try a clear error.
            if avail and context.free_gmem < 75 * MB:
                raise RuntimeError(
                    f"Can not enable cuDNN as there is only {int(context.free_gmem / MB)} MB of free GPU memory."
                )
            elif avail:
                context.cudnn_handle = dnn._make_handle(context)
            elif config.dnn__enabled == "True":
                raise RuntimeError(
                    "You enabled cuDNN, but we aren't able to use it: %s"
                    % dnn.dnn_available.msg
                )
            if config.print_active_device:
                if avail:
                    print(
                        f"Using cuDNN version {int(dnn.version())} on context {name}",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"Can not use cuDNN on context {name}: {dnn.dnn_available.msg}",
                        file=sys.stderr,
                    )
        if preallocate < 0:
            print(f"Disabling allocation cache on {dev}")
        elif preallocate > 0:
            if preallocate <= 1:
                gmem = min(preallocate, 0.95) * context.total_gmem
            else:
                gmem = preallocate * MB
            if gmem > context.free_gmem:
                raise RuntimeError(
                    f"Trying to preallocate {int(gmem / MB)} MB of GPU memory while only"
                    f" {int(context.free_gmem / MB)} MB are available."
                )
            elif gmem > context.free_gmem - 50 * MB:
                warnings.warn(
                    "Preallocating too much memory can prevent cudnn and cublas from working properly"
                )

            # This will allocate and immediately free an object of size gmem
            # which will reserve that amount of memory on the GPU.
            pygpu.empty((gmem,), dtype="int8", context=context)
            if config.print_active_device:
                print(
                    f"Preallocating {int(gmem // MB)}/{int(context.total_gmem // MB)} Mb ({gmem / context.total_gmem}) on {dev}",
                    file=sys.stderr,
                )

        # Initialise the blas kernels.  We do this after the
        # preallocation to not fragment the heap accidentally.
        tmp = pygpu.empty((2, 2), dtype="float32", context=context)
        if dev.startswith("cuda"):
            # In OpenCL, BLAS isn't always available
            pygpu.blas.gemm(0, tmp, tmp, 0, tmp, overwrite_c=True)
        del tmp
    else:
        context = init_dev.devmap[dev]
    # This will map the context name to the real context object.
    if config.print_active_device:
        try:
            unique_id = "(" + context.unique_id + ")"
        except pygpu.gpuarray.UnsupportedException:
            unique_id = ""

        print(
            f"Mapped name {name} to device {dev}: {context.devname} {unique_id}",
            file=sys.stderr,
        )
    pygpu_activated = True


# This maps things like 'cuda0' to the context object on that device.
init_dev.devmap = {}


def use(
    device,
    force=False,
    default_to_move_computation_to_gpu=True,
    move_shared_to_gpu=True,
    preallocate=None,
):
    """
    Error and warning about CUDA should be displayed only when this
    function is called. We need to be able to load this module only
    to check if it is available!

    Parameters
    ----------
    device : string
        "cuda", "cuda0", "cudaN", "" (N is the device number to use).
        "" mean do all the rest and don't init a device.
    force
        Will always raise an exception if we can't use the gpu.
    default_to_move_computation_to_gpu
        If gpu init succeeded, enable by default optimizations to move
        computations to the gpu.
    move_shared_to_gpu
        If gpu init succeeded, put new shared variables on the gpu.
    preallocate
        If specified, will use this value for preallocation instead of
        gpuarray__preallocate.

    """
    if force:
        if not (device.startswith("cuda") or device.startswith("opencl")):
            raise Exception("forced the init and bad device provided: " + device)
        else:
            # If we force, the device should not already be initialized.
            assert device not in init_dev.devmap
    if device:
        init_dev(device, preallocate=preallocate)
    if default_to_move_computation_to_gpu:
        optdb.add_tags("gpuarray_opt", "fast_run", "fast_compile")
        optdb.add_tags("gpua_scanOp_make_inplace", "fast_run")
    if move_shared_to_gpu:
        import aesara.compile

        aesara.compile.shared_constructor(gpuarray_shared_constructor)


if pygpu:
    try:
        if config.device.startswith("cuda") or config.device.startswith("opencl"):
            use(config.device)
        elif config.init_gpu_device.startswith(
            "cuda"
        ) or config.init_gpu_device.startswith("opencl"):
            if config.device != "cpu":
                raise ValueError("you must set device=cpu to use init_gpu_device.")
            if config.contexts != "":
                print(
                    "Using contexts will make init_gpu_device act like device and move all computations by default, which might not be what you want."
                )
            init_dev(config.init_gpu_device)
        if config.contexts != "":
            for n, d in (c.split("->") for c in config.contexts.split(";")):
                init_dev(d.strip(), n.strip())
            # To have shared var default on the GPU and opt to move to the GPU.
            use("")

    except Exception:
        error("Could not initialize pygpu, support disabled", exc_info=True)

    from .basic_ops import (
        GpuAlloc,
        GpuAllocEmpty,
        GpuContiguous,
        GpuEye,
        GpuFromHost,
        GpuJoin,
        GpuReshape,
        GpuSplit,
        HostFromGpu,
        host_from_gpu,
    )
    from .elemwise import GpuElemwise
    from .subtensor import GpuAdvancedIncSubtensor1, GpuIncSubtensor, GpuSubtensor

else:
    if (
        config.init_gpu_device.startswith("cuda")
        or config.init_gpu_device.startswith("opencl")
        or config.device.startswith("opencl")
        or config.device.startswith("cuda")
        or config.contexts != ""
    ):
        error(
            "pygpu was configured but could not be imported or is too old (version 0.7 or higher required)",
            exc_info=True,
        )
