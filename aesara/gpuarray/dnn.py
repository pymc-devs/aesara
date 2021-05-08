import ctypes
import os
import sys
import warnings
from functools import reduce
from typing import Any, Callable, Optional, TypeVar

import numpy as np
from typing_extensions import Protocol

import aesara
import aesara.gpuarray.pathparse
import aesara.tensor.basic as aet
import aesara.tensor.math as tm
from aesara.assert_op import Assert
from aesara.compile.io import Out
from aesara.compile.mode import Mode
from aesara.configdefaults import SUPPORTED_DNN_CONV_ALGO_RUNTIME, config
from aesara.gpuarray import cudnn_defs, pygpu
from aesara.gpuarray.basic_ops import (
    GpuAllocEmpty,
    GpuArrayType,
    HostFromGpu,
    as_gpuarray_variable,
    empty_like,
    gpu_contiguous,
    gpuarray_helper_inc_dir,
    infer_context_name,
)
from aesara.gpuarray.type import GpuArraySharedVariable, get_context, gpu_context_type
from aesara.gradient import DisconnectedType, grad_not_implemented
from aesara.graph.basic import Apply, Variable
from aesara.graph.op import ExternalCOp, _NoPythonCOp, _NoPythonExternalCOp
from aesara.graph.params_type import ParamsType
from aesara.graph.type import CDataType, EnumList, Generic
from aesara.link.c.cmodule import GCC_compiler
from aesara.scalar import as_scalar
from aesara.scalar import bool as bool_t
from aesara.scalar import constant, get_scalar_type
from aesara.scalar import int32 as int_t
from aesara.scalar import uint32 as uint32_t
from aesara.tensor.basic import as_tensor_variable
from aesara.tensor.exceptions import NotScalarConstantError
from aesara.tensor.extra_ops import cpu_contiguous
from aesara.tensor.nnet.abstract_conv import (
    AbstractConv2d,
    AbstractConv2d_gradInputs,
    AbstractConv2d_gradWeights,
    AbstractConv3d,
    AbstractConv3d_gradInputs,
    AbstractConv3d_gradWeights,
    assert_conv_shape,
    get_conv_output_shape,
)
from aesara.tensor.shape import reshape, shape_i, shape_i_op, shape_padright
from aesara.tensor.type import int_dtypes, integer_dtypes, values_eq_approx_always_true


DNN_CONV_ALGO_CHOOSE_ONCE = ["guess_once", "time_once"]
DNN_CONV_ALGO_CHOOSE_TIME = ["time_once", "time_on_shape_change"]

try:
    from pygpu import gpuarray
except ImportError:
    pass

# Update these names when new versions of cudnn are supported.
WIN32_CUDNN_NAMES = ["cudnn64_7.dll", "cudnn64_6.dll", "cudnn64_5.dll"]

if sys.platform == "win32":
    try:
        from aesara.gpuarray.pathparse import PathParser

        PathParser(config.dnn__bin_path)
    except ImportError:
        pass


def _load_lib(name):
    try:
        return ctypes.cdll.LoadLibrary(name)
    except OSError:
        return None


class Dnn_lib(Protocol):
    handle: Optional[str]


def _dnn_lib_decorator(func: Any) -> Dnn_lib:
    return func


@_dnn_lib_decorator
def _dnn_lib():
    if _dnn_lib.handle is None:
        import ctypes.util

        if config.dnn__bin_path != "":
            if sys.platform == "darwin":
                dnn_handle = _load_lib(
                    os.path.join(config.dnn__bin_path, "libcudnn.dylib")
                )
            elif sys.platform == "win32":
                for name in WIN32_CUDNN_NAMES:
                    dnn_handle = _load_lib(os.path.join(config.dnn__bin_path, name))
                    if dnn_handle is not None:
                        break
            else:
                dnn_handle = _load_lib(
                    os.path.join(config.dnn__bin_path, "libcudnn.so")
                )
        else:
            lib_name = ctypes.util.find_library("cudnn")
            if lib_name is None and sys.platform == "win32":
                for name in WIN32_CUDNN_NAMES:
                    lib_name = ctypes.util.find_library(name)
                    if lib_name:
                        break
            if lib_name is None:
                raise RuntimeError(
                    "Could not find cudnn library (looked for v5* to v7*)."
                    " Check your cudnn installation. Maybe using the Aesara"
                    f' flag dnn__base_path can help you. Current value "{config.dnn__base_path}"'
                )
            else:
                dnn_handle = ctypes.cdll.LoadLibrary(lib_name)
        if dnn_handle is None:
            raise RuntimeError(
                "Could not load cudnn library. Check your cudnn"
                " installation. Maybe using the Aesara"
                f' flag dnn__base_path can help you. Current value "{config.dnn__base_path}"'
            )
        _dnn_lib.handle = dnn_handle
        cudnn = _dnn_lib.handle
        cudnn.cudnnCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        cudnn.cudnnCreate.restype = ctypes.c_int
        cudnn.cudnnDestroy.argtypes = [ctypes.c_void_p]
        cudnn.cudnnDestroy.restype = ctypes.c_int
    return _dnn_lib.handle


_dnn_lib.handle = None


def _make_handle(ctx):
    cudnn = _dnn_lib()
    handle = ctypes.c_void_p()
    with ctx:
        err = cudnn.cudnnCreate(ctypes.byref(handle))
    if err != 0:
        raise RuntimeError(
            "Error creating cudnn handle. " "This can be a sign of a too old driver.",
            err,
        )
    return handle


def _dnn_check_compile():
    preambule = """
#include <stdio.h>
#include <cudnn.h>
#include <cudnn_helper.h>
"""

    # No need for the context in here since we won't execute that code
    body = """
cudnnHandle_t _handle = NULL;
cudnnStatus_t err;
if ((err = cudnnCreate(&_handle)) != CUDNN_STATUS_SUCCESS) {
  fprintf(stderr, "could not create cuDNN handle: %s",
          cudnnGetErrorString(err));
  return 1;
}
"""

    path_wrapper = '"' if os.name == "nt" else ""
    params = ["-l", "cudnn"]
    params.extend([f"-I{path_wrapper}{gpuarray_helper_inc_dir()}{path_wrapper}"])
    if config.dnn__include_path:
        params.extend([f"-I{path_wrapper}{config.dnn__include_path}{path_wrapper}"])
    if config.cuda__include_path:
        params.extend([f"-I{path_wrapper}{config.cuda__include_path}{path_wrapper}"])
    if config.dnn__library_path:
        params.extend([f"-L{path_wrapper}{config.dnn__library_path}{path_wrapper}"])
    # Do not run here the test program. It would run on the
    # default gpu, not the one selected by the user. If mixed
    # GPU are installed or if the GPUs are configured in
    # exclusive mode, this cause bad detection.

    # NB: GCC_compiler.try_flags() may return just a boolean instead of a tuple (avail, out, here).
    compiler_res = GCC_compiler.try_flags(
        params, preambule=preambule, body=body, try_run=False, output=True
    )

    avail, out, err = (
        compiler_res if isinstance(compiler_res, tuple) else (compiler_res, None, None)
    )

    if not avail:
        return False, ("cannot compile with cuDNN. " "We got this error:\n" + str(err))
    return True, None


def _dnn_check_version():
    v = version()
    if v < 5000:
        return (
            False,
            f"cuDNN version is too old. Update to v5* or higher, was {int(v)}.",
        )
    if v >= 7200:
        warnings.warn(
            "Your cuDNN version is more recent than "
            "Aesara. If you encounter problems, try "
            "updating Aesara or downgrading cuDNN to "
            "a version >= v5 and <= v7."
        )
    return True, None


F = TypeVar("F", bound=Callable[..., object])


class Dnn_present(Protocol[F]):
    avail: Optional[bool]
    msg: Optional[str]
    __call__: F


def dnn_present_decorator(func: Any) -> Dnn_present:
    return func


@dnn_present_decorator
def dnn_present():
    if dnn_present.avail is not None:
        return dnn_present.avail
    if config.dnn__enabled == "False":
        dnn_present.msg = "Disabled by dnn__enabled flag"
        dnn_present.avail = False
        return False

    if pygpu is None:
        dnn_present.msg = "PyGPU not available"
        dnn_present.avail = False
        return False

    if config.dnn__enabled == "no_check":
        dnn_present.avail, dnn_present.msg = (
            True,
            "presence check disabled by dnn__enabled flag",
        )
    else:
        dnn_present.avail, dnn_present.msg = _dnn_check_compile()
    if dnn_present.avail:
        dnn_present.avail, dnn_present.msg = _dnn_check_version()
        if not dnn_present.avail:
            return False

    return dnn_present.avail


dnn_present.avail = None
dnn_present.msg = None


class Dnn_available(Protocol[F]):
    msg: Optional[bool]
    __call__: F


def dnn_available_decorator(func: Any) -> Dnn_available:
    return func


@dnn_available_decorator
def dnn_available(context_name):
    if not dnn_present():
        dnn_available.msg = dnn_present.msg
        return False

    ctx = get_context(context_name)

    if not ctx.kind == b"cuda":
        dnn_available.msg = "Not on a CUDA device."
        return False

    # This is a hack because bin_id is in the from of
    # "<something>_<major><minor>" for cuda devices.
    if int(ctx.bin_id[-2:]) < 30:
        dnn_available.msg = "Device not supported"
        return False

    # On V100, cuDNN lower then 7002 don't raise error but
    # takes hours to load or execute! So raise a good user error.
    if version() < 7002:
        if int(ctx.bin_id[-2:]) >= 70:
            dnn_available.msg = "Use cuDNN 7.0.2 or higher for Volta."
            return False
    return True


dnn_available.msg = None


class MakerCDataType(CDataType):
    """This `CDataType` provides a `make_value` method.

    It also has `CDataType._fn` field that caches a compiled function used
    by `CDataType.make_value`.

    This was a very lame hack that was removed from `CDataType` itself.

    """

    def _get_func(self):
        """
        Return a function that makes a value from an integer.

        The integer value is assumed to be a valid pointer for the
        type and no check is done to ensure that.
        """
        from aesara.scalar import get_scalar_type

        if self._fn is None:
            with config.change_flags(compute_test_value="off"):
                v = get_scalar_type("int64")()
                self._fn = aesara.function(
                    [v],
                    CDataMaker(self)(v),
                    mode=Mode(optimizer=None),
                    profile=False,
                )
        return self._fn

    def make_value(self, ptr):
        """
        Make a value of this type.

        Parameters
        ----------
        ptr : int
            Integer representation of a valid pointer value

        """
        return self._get_func()(ptr)


class CDataMaker(_NoPythonCOp):
    """This is the equally lame `Op` that accompanies `MakerCDataType`."""

    __props__ = ("rtype",)

    def __init__(self, rtype):
        assert isinstance(rtype, MakerCDataType)
        self.rtype = rtype

    def do_constant_folding(self, fgraph, node):
        return False

    def make_node(self, val):
        from aesara.scalar import as_scalar

        val = as_scalar(val).astype("uint64")
        return Apply(self, [val], [self.rtype()])

    def c_code(self, node, name, inputs, outputs, sub):
        return """
        %(out)s = (%(ctype)s)%(inp)s;
        """ % dict(
            ctype=self.rtype.ctype, out=outputs[0], inp=inputs[0]
        )

    def c_code_cache_version(self):
        if self.rtype.version is None:
            return None
        return (0, self.rtype.version)


def CUDNNDataType(name, freefunc=None):
    cargs = []
    if config.dnn__bin_path and sys.platform != "win32":
        cargs.append("-Wl,-rpath," + config.dnn__bin_path)

    return MakerCDataType(
        name,
        freefunc,
        headers=["cudnn.h"],
        header_dirs=[config.dnn__include_path, config.cuda__include_path],
        libraries=["cudnn"],
        lib_dirs=[config.dnn__library_path],
        compile_args=cargs,
        version=version(raises=False),
    )


class DnnVersion(_NoPythonCOp):
    __props__ = ()

    def c_headers(self, **kwargs):
        return ["cudnn.h"]

    def c_header_dirs(self, **kwargs):
        return [config.dnn__include_path, config.cuda__include_path]

    def c_libraries(self, **kwargs):
        return ["cudnn"]

    def c_lib_dirs(self, **kwargs):
        return [config.dnn__library_path]

    def c_compile_args(self, **kwargs):
        if config.dnn__bin_path and sys.platform != "win32":
            return ["-Wl,-rpath," + config.dnn__bin_path]
        return []

    def c_support_code(self, **kwargs):
        return """
#if PY_MAJOR_VERSION >= 3
#define PyInt_FromLong PyLong_FromLong
#endif
"""

    def make_node(self):
        return Apply(self, [], [Generic()()])

    def c_code(self, node, name, inputs, outputs, sub):
        o = outputs[0]
        return (
            """
        %(o)s = PyTuple_Pack(2, PyInt_FromLong(CUDNN_VERSION), PyInt_FromLong(cudnnGetVersion()));
        """
            % locals()
        )

    def do_constant_folding(self, fgraph, node):
        # Needed as we do not want to cache this information.
        return False

    def c_code_cache_version(self):
        # Not needed, but make it clear that we do not want to cache this.
        return None


class Version(Protocol[F]):
    v: Optional[int]
    __call__: F


def version_decorator(func: Any) -> Version:
    return func


@version_decorator
def version(raises: bool = True) -> int:
    """Return the current cuDNN version we link with.

    This also does a check that the header version matches the runtime version.

    :raises: If True, raise an exception if cuDNN is not present.
        Otherwise, return -1.

    It always raise an RuntimeError if the header and library version
    are not the same.

    """
    if not dnn_present():
        if raises:
            raise RuntimeError(
                "We can't determine the cudnn version as it is not available",
                dnn_available.msg,
            )
        else:
            return -1

    if version.v is None:
        f = aesara.function([], DnnVersion()(), Mode(optimizer=None), profile=False)
        v = f()
        if v[0] != v[1]:
            raise RuntimeError(
                f"Mixed dnn version. The header is version {v[0]} "
                f"while the library is version {v[1]}."
            )
        version.v = v[1]
    return version.v


version.v = None

handle_type = CUDNNDataType("cudnnHandle_t", "cudnnDestroy")

# Get cuDNN definitions to be used.
cudnn = cudnn_defs.get_definitions(version(raises=False))


def get_precision(precision, inputs, for_grad=False):
    common_dtype = aesara.scalar.upcast(*[i.dtype for i in inputs])
    if not common_dtype.startswith("float"):
        raise TypeError("cuDNN convolution only works on real numbers")

    if precision is None:
        precision = config.dnn__conv__precision
    if precision == "as_input" or precision == "as_input_f32":
        if common_dtype == "float16" and precision == "as_input_f32":
            precision = "float32"
        else:
            precision = common_dtype
    if for_grad and precision == "float16":
        raise TypeError(
            "Float16 precision is disabled for cuDNN backward convolutions due to computation errors."
        )
    return precision, common_dtype


class DnnBase(_NoPythonExternalCOp):

    """
    Creates a handle for cudnn and pulls in the cudnn libraries and headers.

    """

    # dnn does not know about broadcasting, so we do not need to assert
    # the input broadcasting pattern.
    check_broadcast = False
    params_type = handle_type

    def dnn_context(self, node):
        return node.outputs[0].type.context_name

    def get_params(self, node):
        ctx_name = self.dnn_context(node)
        ctx = get_context(ctx_name)
        if not hasattr(ctx, "cudnn_handle_param"):
            ptr = ctx.cudnn_handle.value
            res = handle_type.make_value(ptr)
            ctx.cudnn_handle_param = res
        if isinstance(self.params_type, ParamsType):
            if not self.params_type.has_type(handle_type):
                raise TypeError(
                    "DnnBase: params_type must take into account the cuDNN handle type."
                )
            handle_field = self.params_type.get_field(handle_type)
            return self.params_type.get_params(
                self, **{handle_field: ctx.cudnn_handle_param}
            )
        return ctx.cudnn_handle_param

    def __init__(self, files=None, c_func=None):
        if files is None:
            files = []
        super().__init__(["c_code/dnn_base.c"] + files, c_func)

    def c_headers(self, **kwargs):
        return [
            "gpuarray/types.h",
            "gpuarray/array.h",
            "gpuarray/kernel.h",
            "gpuarray/util.h",
            "gpuarray/ext_cuda.h",
            "gpuarray_api.h",
            "numpy_compat.h",
            "cudnn.h",
            "cudnn_helper.h",
            "gpuarray_helper.h",
        ]

    def c_header_dirs(self, **kwargs):
        return [
            gpuarray_helper_inc_dir(),
            pygpu.get_include(),
            config.dnn__include_path,
            config.cuda__include_path,
        ]

    def c_libraries(self, **kwargs):
        return ["cudnn", "gpuarray"]

    def c_lib_dirs(self, **kwargs):
        return [config.dnn__library_path]

    def c_compile_args(self, **kwargs):
        if config.dnn__bin_path and sys.platform != "win32":
            return ["-Wl,-rpath," + config.dnn__bin_path]
        return []

    def c_code_cache_version(self):
        return (super().c_code_cache_version(), version(), 4)


class GpuDnnConvDesc(_NoPythonExternalCOp):

    """
    This Op builds a convolution descriptor for use in the other convolution
    operations.

    See the doc of :func:`dnn_conv` for a description of the parameters

    """

    __props__ = (
        "border_mode",
        "subsample",
        "dilation",
        "conv_mode",
        "precision",
        "num_groups",
    )
    params_type = ParamsType(
        pad0=int_t,
        pad1=int_t,
        pad2=int_t,
        sub0=int_t,
        sub1=int_t,
        sub2=int_t,
        dil0=int_t,
        dil1=int_t,
        dil2=int_t,
        nb_dims=int_t,
        bmode=EnumList(
            ("BORDER_MODE_FULL", "full"),
            ("BORDER_MODE_VALID", "valid"),
            ("BORDER_MODE_HALF", "half"),
        ),
        conv_mode=cudnn.cudnnConvolutionMode_t,
        precision=cudnn.cudnnDataType_t,
        num_groups=int_t,
    )

    def c_headers(self, **kwargs):
        return ["cudnn.h", "cudnn_helper.h"]

    def c_header_dirs(self, **kwargs):
        return [
            gpuarray_helper_inc_dir(),
            config.dnn__include_path,
            config.cuda__include_path,
        ]

    def c_libraries(self, **kwargs):
        return ["cudnn"]

    def c_lib_dirs(self, **kwargs):
        return [config.dnn__library_path]

    def c_compile_args(self, **kwargs):
        if config.dnn__bin_path and sys.platform != "win32":
            return ["-Wl,-rpath," + config.dnn__bin_path]
        return []

    def do_constant_folding(self, fgraph, node):
        return False

    def __init__(
        self,
        border_mode,
        subsample=(1, 1),
        dilation=(1, 1),
        conv_mode="conv",
        precision="float32",
        num_groups=1,
    ):
        super().__init__(["c_code/conv_desc.c"], "APPLY_SPECIFIC(conv_desc)")

        if version() < 6000 and any([d != 1 for d in dilation]):
            raise RuntimeError("Dilation > 1 not supported for cuDNN version < 6.")

        if isinstance(border_mode, int):
            border_mode = (border_mode,) * len(subsample)
        if isinstance(border_mode, tuple):
            assert len(border_mode) == len(subsample)
            border_mode = tuple(map(int, border_mode))
        if not (
            (isinstance(border_mode, tuple) and min(border_mode) >= 0)
            or border_mode in ("valid", "full", "half")
        ):
            raise ValueError(
                "invalid border_mode {}, which must be either "
                '"valid", "full", "half", an integer or a pair of'
                " integers".format(border_mode)
            )
        self.border_mode = border_mode
        assert len(subsample) in (2, 3)
        self.subsample = subsample
        assert cudnn.cudnnConvolutionMode_t.has_alias(conv_mode)
        self.conv_mode = conv_mode
        self.num_groups = num_groups

        assert len(dilation) == len(subsample)
        self.dilation = dilation

        assert cudnn.cudnnDataType_t.has_alias(precision)
        self.precision = precision

    def make_node(self, kern_shape):
        kern_shape = as_tensor_variable(kern_shape)
        if kern_shape.type.ndim != 1 or kern_shape.dtype not in int_dtypes:
            raise TypeError("kern must be an int64 1D shape tensor")
        kern_shape = aet.cast(kern_shape, "int64")

        node = Apply(
            self,
            [kern_shape],
            [
                CUDNNDataType(
                    "cudnnConvolutionDescriptor_t",
                    freefunc="cudnnDestroyConvolutionDescriptor",
                )()
            ],
        )
        # DebugMode cannot compare the values of CDataType variables, so by
        # default it returns False all the time. To prevent DebugMode from
        # complaining because of the MergeOptimizer, we make this variable
        # always compare to True.
        out = node.outputs[0]
        out.tag.values_eq_approx = values_eq_approx_always_true
        return node

    bmode = property(
        lambda self: "valid"
        if isinstance(self.border_mode, tuple)
        else self.border_mode
    )
    pad0 = property(
        lambda self: self.border_mode[0] if isinstance(self.border_mode, tuple) else 0
    )
    pad1 = property(
        lambda self: self.border_mode[1] if isinstance(self.border_mode, tuple) else 0
    )
    pad2 = property(
        lambda self: self.border_mode[2]
        if (isinstance(self.border_mode, tuple) and len(self.border_mode) > 2)
        else 0
    )
    sub0 = property(lambda self: self.subsample[0])
    sub1 = property(lambda self: self.subsample[1])
    sub2 = property(lambda self: self.subsample[2] if len(self.subsample) > 2 else 0)
    dil0 = property(lambda self: self.dilation[0])
    dil1 = property(lambda self: self.dilation[1])
    dil2 = property(lambda self: self.dilation[2] if len(self.dilation) > 2 else 0)
    nb_dims = property(lambda self: len(self.subsample))

    def c_code_cache_version(self):
        return (super().c_code_cache_version(), version())

    def __setstate__(self, d):
        self.__dict__.update(d)
        if not hasattr(self, "dilation"):
            self.dilation = (1,) * len(self.subsample)
        if not hasattr(self, "num_groups"):
            self.num_groups = 1


# scalar constants
_zero = constant(np.asarray(0.0, dtype="float64"))
_one = constant(np.asarray(1.0, dtype="float64"))


def ensure_dt(val, default, name, dtype):
    if dtype == "float16":
        dtype = "float32"
    if val is None:
        val = default.clone()
    if not isinstance(val, Variable):
        val = constant(val)
    if hasattr(val, "ndim") and val.ndim == 0:
        val = as_scalar(val)
    if not isinstance(val.type, aesara.scalar.Scalar):
        raise TypeError(f"{name}: expected a scalar value")
    if not val.type.dtype == dtype:
        val = val.astype(dtype)
    return val


class GpuDnnConv(DnnBase):
    """
    The forward convolution.

    Parameters
    ----------
    image
    kernel
    descr :
        The convolution descriptor.
    algo : {'small', 'none', 'large', 'fft', 'fft_tiling', 'winograd', 'guess_once',
            'guess_on_shape_change', 'time_once', 'time_on_shape_change'}
        Default is the value of :attr:`config.dnn__conv__algo_fwd`.
    num_groups :
        Divides the image, kernel and output tensors into num_groups
        separate groups. Each which carry out convolutions separately

    """

    _f16_ok = True
    __props__ = ("algo", "inplace", "num_groups")

    check_input = False
    params_type = ParamsType(
        conv_algo=cudnn.cudnnConvolutionFwdAlgo_t,
        choose_algo=bool_t,
        choose_once=bool_t,
        choose_time=bool_t,
        inplace=bool_t,
        handle=handle_type,
        num_groups=int_t,
    )

    def __init__(self, algo=None, inplace=False, num_groups=1):
        super().__init__(
            ["c_code/dnn_conv_base.c", "c_code/dnn_fwd.c"],
            "APPLY_SPECIFIC(conv_fwd)",
        )

        if algo is None:
            algo = config.dnn__conv__algo_fwd
        self.algo = algo

        self.inplace = bool(inplace)
        if self.inplace:
            self.destroy_map = {0: [2]}

        assert (
            cudnn.cudnnConvolutionFwdAlgo_t.has_alias(self.algo)
            or self.algo in SUPPORTED_DNN_CONV_ALGO_RUNTIME
        )

        self.conv_algo = (
            cudnn.cudnnConvolutionFwdAlgo_t.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
        )
        if self.algo not in SUPPORTED_DNN_CONV_ALGO_RUNTIME:
            self.conv_algo = self.algo
        self.choose_algo = self.algo in SUPPORTED_DNN_CONV_ALGO_RUNTIME
        self.choose_once = self.algo in DNN_CONV_ALGO_CHOOSE_ONCE
        self.choose_time = self.algo in DNN_CONV_ALGO_CHOOSE_TIME
        self.num_groups = num_groups

    def __setstate__(self, d):
        self.__dict__.update(d)
        if not hasattr(self, "algo"):
            if hasattr(self, "workmem"):
                self.algo = self.workmem
            else:
                self.algo = config.dnn__conv__algo_fwd
        if not hasattr(self, "inplace"):
            self.inplace = False
        if not hasattr(self, "num_groups"):
            self.num_groups = 1

    def make_node(self, img, kern, output, desc, alpha=None, beta=None):
        ctx_name = infer_context_name(img, kern, output)
        img = as_gpuarray_variable(img, ctx_name)
        kern = as_gpuarray_variable(kern, ctx_name)
        output = as_gpuarray_variable(output, ctx_name)

        if img.type.ndim not in (4, 5):
            raise TypeError("img must be 4D or 5D tensor")
        if kern.type.ndim not in (4, 5):
            raise TypeError("kern must be 4D or 5D tensor")
        if output.type.ndim not in (4, 5):
            raise TypeError("output must be a 4D or 5D tensor")

        if img.type.ndim != kern.type.ndim or img.type.ndim != output.type.ndim:
            raise TypeError(
                "The number of dimensions of " "img, kern and output must match"
            )

        if img.type.ndim == 5 and self.algo not in (
            cudnn.conv3d_fwd_algorithms + SUPPORTED_DNN_CONV_ALGO_RUNTIME
        ):
            raise ValueError(
                f"convolution algo {self.algo} can't be used for 3d convolutions"
            )

        if (
            not isinstance(desc.type, CDataType)
            or desc.type.ctype != "cudnnConvolutionDescriptor_t"
        ):
            raise TypeError("desc must be cudnnConvolutionDescriptor_t")

        alpha = ensure_dt(alpha, _one, "alpha", img.dtype)
        beta = ensure_dt(beta, _zero, "beta", img.dtype)

        return Apply(self, [img, kern, output, desc, alpha, beta], [output.type()])

    def grad(self, inp, grads):
        img, kerns, output, desc, alpha, beta = inp
        (top,) = grads

        top = gpu_contiguous(top)

        d_img = GpuDnnConvGradI(num_groups=self.num_groups)(
            kerns, top, empty_like(img), desc
        )
        d_kerns = GpuDnnConvGradW(num_groups=self.num_groups)(
            img, top, empty_like(kerns), desc
        )
        d_alpha = grad_not_implemented(self, 4, alpha)
        d_beta = grad_not_implemented(self, 5, beta)

        return [
            d_img * alpha,
            d_kerns * alpha,
            top * beta,
            DisconnectedType()(),
            d_alpha,
            d_beta,
        ]

    def connection_pattern(self, node):
        # not connected to desc
        return [[1], [1], [1], [0], [1], [1]]

    @staticmethod
    def get_out_shape(ishape, kshape, border_mode, subsample, dilation):
        """
        This function computes the output shape for a convolution with
        the specified parameters. `ishape` and `kshape` can be symbolic
        or scalar.

        """

        # if ishape and/or kshape are not tuples or list, but rather symbolic
        # vectors, turn them into lists of symbolic scalars.
        if not isinstance(ishape, (list, tuple)):
            ishape = [ishape[i] for i in range(len(subsample) + 2)]
        if not isinstance(kshape, (list, tuple)):
            kshape = [kshape[i] for i in range(len(subsample) + 2)]

        return get_conv_output_shape(ishape, kshape, border_mode, subsample, dilation)

    def infer_shape(self, fgraph, node, shape):
        return [shape[2]]


class GpuDnnConvGradW(DnnBase):

    """
    The convolution gradient with respect to the weights.

    Parameters
    ----------
    image
    kernel
    descr :
        The convolution descriptor.
    algo : {'none', 'deterministic', 'fft', 'small', 'guess_once',
            'guess_on_shape_change', 'time_once', 'time_on_shape_change'}
        Default is the value of :attr:`config.dnn__conv__algo_bwd_filter`.
    num_groups :
        Divides the image, kernel and output tensors into num_groups
        separate groups. Each which carry out convolutions separately

    """

    _f16_ok = True
    __props__ = ("algo", "inplace", "num_groups")

    check_input = False
    params_type = ParamsType(
        conv_algo=cudnn.cudnnConvolutionBwdFilterAlgo_t,
        choose_algo=bool_t,
        choose_once=bool_t,
        choose_time=bool_t,
        inplace=bool_t,
        handle=handle_type,
        num_groups=int_t,
    )

    def __init__(self, inplace=False, algo=None, num_groups=1):
        super().__init__(
            ["c_code/dnn_conv_base.c", "c_code/dnn_gw.c"],
            "APPLY_SPECIFIC(conv_gw)",
        )
        self.inplace = bool(inplace)
        if self.inplace:
            self.destroy_map = {0: [2]}
        if algo is None:
            algo = config.dnn__conv__algo_bwd_filter
        self.algo = algo

        assert (
            cudnn.cudnnConvolutionBwdFilterAlgo_t.has_alias(self.algo)
            or self.algo in SUPPORTED_DNN_CONV_ALGO_RUNTIME
        )

        self.conv_algo = (
            cudnn.cudnnConvolutionBwdFilterAlgo_t.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0
        )
        if self.algo not in SUPPORTED_DNN_CONV_ALGO_RUNTIME:
            self.conv_algo = self.algo
        self.choose_algo = self.algo in SUPPORTED_DNN_CONV_ALGO_RUNTIME
        self.choose_once = self.algo in DNN_CONV_ALGO_CHOOSE_ONCE
        self.choose_time = self.algo in DNN_CONV_ALGO_CHOOSE_TIME
        self.num_groups = num_groups

    def __setstate__(self, d):
        self.__dict__.update(d)
        if not hasattr(self, "inplace"):
            self.inplace = False
        if not hasattr(self, "algo"):
            self.algo = config.dnn__conv__algo_bwd_filter
        if not hasattr(self, "num_groups"):
            self.num_groups = 1

    def grad(self, inp, grads):
        img, top, output, desc, alpha, beta = inp
        (kerns,) = grads

        kerns = gpu_contiguous(kerns)

        d_img = GpuDnnConvGradI(num_groups=self.num_groups)(
            kerns, top, empty_like(img), desc
        )
        d_top = GpuDnnConv(num_groups=self.num_groups)(
            img, kerns, empty_like(top), desc
        )
        d_alpha = grad_not_implemented(self, 4, alpha)
        d_beta = grad_not_implemented(self, 5, beta)

        return (
            d_img * alpha,
            d_top * alpha,
            kerns * beta,
            DisconnectedType()(),
            d_alpha,
            d_beta,
        )

    def connection_pattern(self, node):
        # not connected to desc
        return [[1], [1], [1], [0], [1], [1]]

    def op_may_fail_with_subsample(self, img, desc):
        return (
            version() < 6000
            and img.type.dtype == "float32"
            and img.type.ndim == 5
            and self.algo != "none"
            and desc.owner.op.subsample != (1, 1, 1)
        )

    def op_may_fail_with_beta(self, img, beta):
        return (
            version() < 6000
            and img.type.dtype == "float32"
            and self.algo not in ("none", "deterministic", "fft", "small")
            and beta is not None
            and aet.extract_constant(beta) != 1
        )

    def make_node(self, img, topgrad, output, desc, alpha=None, beta=None):
        if self.op_may_fail_with_subsample(img, desc):
            warnings.warn(
                "cuDNN backward filter operation for 3D convolutions may produce bad results "
                "with certain cuDNN algorithms depending on the compute capability of your GPU "
                "if subsample is not (1, 1, 1). If you encounter problems, consider "
                'setting the aesara flag "dnn__conv__algo_bwd_filter" to "none".'
            )
        if self.op_may_fail_with_beta(img, beta):
            warnings.warn(
                "cuDNN backward filter operation for convolutions may produce bad results "
                "with certain cuDNN algorithms depending on the compute capability of your GPU "
                "if beta != 1. If you encounter problems, consider "
                'setting the aesara flag "dnn__conv__algo_bwd_filter" to '
                '"none", "deterministic", "fft", or "small".'
            )
        ctx_name = infer_context_name(img, topgrad, output)
        img = as_gpuarray_variable(img, ctx_name)
        topgrad = as_gpuarray_variable(topgrad, ctx_name)
        output = as_gpuarray_variable(output, ctx_name)
        if img.type.ndim not in (4, 5):
            raise TypeError("img must be 4D or 5D tensor")
        if topgrad.type.ndim not in (4, 5):
            raise TypeError("topgrad must be 4D or 5D tensor")
        if output.type.ndim not in (4, 5):
            raise TypeError("output must be 4D or 5D tensor")

        if img.type.ndim != topgrad.type.ndim or img.type.ndim != output.type.ndim:
            raise TypeError(
                "The number of dimensions of " "img, topgrad and output must match"
            )

        if img.type.ndim == 5 and self.algo not in (
            cudnn.conv3d_bwd_filter_algorithms + SUPPORTED_DNN_CONV_ALGO_RUNTIME
        ):
            raise ValueError(
                f"convolution algo {self.algo} can't be used for 3d convolutions"
            )

        if (
            not isinstance(desc.type, CDataType)
            or desc.type.ctype != "cudnnConvolutionDescriptor_t"
        ):
            raise TypeError("desc must be cudnnConvolutionDescriptor_t")

        alpha = ensure_dt(alpha, _one, "alpha", img.dtype)
        beta = ensure_dt(beta, _zero, "beta", img.dtype)

        return Apply(self, [img, topgrad, output, desc, alpha, beta], [output.type()])

    def infer_shape(self, fgraph, node, shape):
        return [shape[2]]


class GpuDnnConvGradI(DnnBase):
    """
    The convolution gradient with respect to the inputs.

    Parameters
    ----------
    image
    kernel
    descr
        The convolution descriptor.
    algo : {'none', 'deterministic', 'fft', 'fft_tiling', 'winograd', 'guess_once',
            'guess_on_shape_change', 'time_once', 'time_on_shape_change'}
        Default is the value of :attr:`config.dnn__conv__algo_bwd_data`.
    num_groups :
        Divides the image, kernel and output tensors into num_groups
        separate groups. Each which carry out convolutions separately

    """

    _f16_ok = True
    __props__ = ("algo", "inplace", "num_groups")

    check_input = False
    params_type = ParamsType(
        conv_algo=cudnn.cudnnConvolutionBwdDataAlgo_t,
        choose_algo=bool_t,
        choose_once=bool_t,
        choose_time=bool_t,
        inplace=bool_t,
        handle=handle_type,
        num_groups=int_t,
    )

    def __init__(self, inplace=False, algo=None, num_groups=1):
        super().__init__(
            ["c_code/dnn_conv_base.c", "c_code/dnn_gi.c"],
            "APPLY_SPECIFIC(conv_gi)",
        )
        self.inplace = bool(inplace)
        if self.inplace:
            self.destroy_map = {0: [2]}
        if algo is None:
            algo = config.dnn__conv__algo_bwd_data
        self.algo = algo
        assert (
            cudnn.cudnnConvolutionBwdDataAlgo_t.has_alias(self.algo)
            or self.algo in SUPPORTED_DNN_CONV_ALGO_RUNTIME
        )

        self.conv_algo = (
            cudnn.cudnnConvolutionBwdDataAlgo_t.CUDNN_CONVOLUTION_BWD_DATA_ALGO_0
        )
        if self.algo not in SUPPORTED_DNN_CONV_ALGO_RUNTIME:
            self.conv_algo = self.algo
        self.choose_algo = self.algo in SUPPORTED_DNN_CONV_ALGO_RUNTIME
        self.choose_once = self.algo in DNN_CONV_ALGO_CHOOSE_ONCE
        self.choose_time = self.algo in DNN_CONV_ALGO_CHOOSE_TIME
        self.num_groups = num_groups

    def __setstate__(self, d):
        self.__dict__.update(d)
        if not hasattr(self, "algo"):
            self.algo = config.dnn__conv__algo_bwd_data
        if not hasattr(self, "inplace"):
            self.inplace = False
        if not hasattr(self, "num_groups"):
            self.num_groups = 1

    def grad(self, inp, grads):
        kerns, top, output, desc, alpha, beta = inp
        (img,) = grads

        img = gpu_contiguous(img)

        d_kerns = GpuDnnConvGradW(num_groups=self.num_groups)(
            img, top, empty_like(kerns), desc
        )
        d_top = GpuDnnConv(num_groups=self.num_groups)(
            img, kerns, empty_like(top), desc
        )
        d_alpha = grad_not_implemented(self, 4, alpha)
        d_beta = grad_not_implemented(self, 5, beta)

        return (
            d_kerns * alpha,
            d_top * alpha,
            img * beta,
            DisconnectedType()(),
            d_alpha,
            d_beta,
        )

    def connection_pattern(self, node):
        # not connected to desc
        return [[1], [1], [1], [0], [1], [1]]

    def make_node(self, kern, topgrad, output, desc, alpha=None, beta=None):
        ctx_name = infer_context_name(kern, topgrad, output)
        kern = as_gpuarray_variable(kern, ctx_name)
        topgrad = as_gpuarray_variable(topgrad, ctx_name)
        output = as_gpuarray_variable(output, ctx_name)
        if kern.type.ndim not in (4, 5):
            raise TypeError("kern must be 4D or 5D tensor")
        if topgrad.type.ndim not in (4, 5):
            raise TypeError("topgrad must be 4D or 5D tensor")
        if output.type.ndim not in (4, 5):
            raise TypeError("output must be 4D or 5D tensor")

        if kern.type.ndim != topgrad.type.ndim or kern.type.ndim != output.type.ndim:
            raise TypeError(
                "The number of dimensions of " "kern, topgrad and output must match"
            )

        if kern.type.ndim == 5 and self.algo not in (
            cudnn.conv3d_bwd_data_algorithms + SUPPORTED_DNN_CONV_ALGO_RUNTIME
        ):
            raise ValueError(
                f"convolution algo {self.algo} can't be used for 3d convolutions"
            )

        if (
            not isinstance(desc.type, CDataType)
            or desc.type.ctype != "cudnnConvolutionDescriptor_t"
        ):
            raise TypeError("desc must be cudnnConvolutionDescriptor_t")

        alpha = ensure_dt(alpha, _one, "alpha", kern.dtype)
        beta = ensure_dt(beta, _zero, "beta", kern.dtype)

        return Apply(self, [kern, topgrad, output, desc, alpha, beta], [output.type()])

    def infer_shape(self, fgraph, node, shape):
        return [shape[2]]


# These internal implementations for dnn_conv, dnn_gradweight and dnn_gradinput
# support alpha, beta and out as parameters. Public interfaces follow without
# underscore prefix.


def _dnn_conv(
    img,
    kerns,
    alpha=1,
    beta=0,
    out=None,
    border_mode="valid",
    subsample=(1, 1),
    dilation=(1, 1),
    conv_mode="conv",
    algo=None,
    precision=None,
    num_groups=1,
):
    ctx_name = infer_context_name(img, kerns)

    img = as_gpuarray_variable(img, ctx_name)
    kerns = as_gpuarray_variable(kerns, ctx_name)

    precision, dt = get_precision(precision, [img, kerns])

    img = gpu_contiguous(img.astype(dt))
    kerns = gpu_contiguous(kerns.astype(dt))

    desc = GpuDnnConvDesc(
        border_mode=border_mode,
        subsample=subsample,
        dilation=dilation,
        conv_mode=conv_mode,
        precision=precision,
        num_groups=num_groups,
    )(kerns.shape)
    desc_op = desc.owner.op
    # We can use Shape_i and bypass the infer_shape here as this is on
    # the input of node and it will always be present.
    ishape = [shape_i_op(i)(img) for i in range(img.ndim)]
    kshape = [shape_i_op(i)(kerns) for i in range(kerns.ndim)]
    out_shp = get_conv_output_shape(
        ishape, kshape, desc_op.border_mode, desc_op.subsample, filter_dilation=dilation
    )
    out_shp = assert_conv_shape(out_shp)
    if beta == 0:
        real_out = GpuAllocEmpty(dtype=img.dtype, context_name=ctx_name)(*out_shp)
    else:
        assert out is not None
        out = gpu_contiguous(as_gpuarray_variable(out, ctx_name))
        check = Assert(
            "GpuDnnConv: given output (for beta not null) does not have expected shape"
        )
        real_out = check(out, tm.all(tm.eq(out.shape, out_shp)))
    return GpuDnnConv(algo=algo, num_groups=num_groups)(
        img, kerns, real_out, desc, alpha, beta
    )


def _dnn_gradweight(
    img,
    topgrad,
    kerns_shp,
    alpha=1,
    beta=0,
    out=None,
    border_mode="valid",
    subsample=(1, 1),
    dilation=(1, 1),
    conv_mode="conv",
    algo=None,
    precision=None,
    num_groups=1,
):
    ctx_name = infer_context_name(img, topgrad)

    img = as_gpuarray_variable(img, ctx_name)
    topgrad = as_gpuarray_variable(topgrad, ctx_name)
    kerns_shp = aet.as_tensor_variable(kerns_shp)

    precision, dt = get_precision(precision, [img, topgrad], for_grad=True)

    img = gpu_contiguous(img.astype(dt))
    topgrad = gpu_contiguous(topgrad.astype(dt))

    desc = GpuDnnConvDesc(
        border_mode=border_mode,
        subsample=subsample,
        dilation=dilation,
        conv_mode=conv_mode,
        precision=precision,
        num_groups=num_groups,
    )(kerns_shp)
    if beta == 0:
        real_out = GpuAllocEmpty(dtype=img.dtype, context_name=ctx_name)(*kerns_shp)
    else:
        assert out is not None
        out = gpu_contiguous(as_gpuarray_variable(out, ctx_name))
        check = Assert(
            "GpuDnnConvGradW: given output (for beta not null) does not have expected shape"
        )
        real_out = check(out, tm.all(tm.eq(out.shape, kerns_shp)))
    return GpuDnnConvGradW(algo=algo, num_groups=num_groups)(
        img, topgrad, real_out, desc, alpha, beta
    )


def _dnn_gradinput(
    kerns,
    topgrad,
    img_shp,
    alpha=1,
    beta=0,
    out=None,
    border_mode="valid",
    subsample=(1, 1),
    dilation=(1, 1),
    conv_mode="conv",
    algo=None,
    precision=None,
    num_groups=1,
):
    ctx_name = infer_context_name(kerns, topgrad)

    kerns = as_gpuarray_variable(kerns, ctx_name)
    topgrad = as_gpuarray_variable(topgrad, ctx_name)
    img_shp = aet.as_tensor_variable(img_shp)

    precision, dt = get_precision(precision, [kerns, topgrad], for_grad=True)

    kerns = gpu_contiguous(kerns.astype(dt))
    topgrad = gpu_contiguous(topgrad.astype(dt))

    desc = GpuDnnConvDesc(
        border_mode=border_mode,
        subsample=subsample,
        dilation=dilation,
        conv_mode=conv_mode,
        precision=precision,
        num_groups=num_groups,
    )(kerns.shape)
    if beta == 0:
        real_out = GpuAllocEmpty(dtype=kerns.dtype, context_name=ctx_name)(*img_shp)
    else:
        assert out is not None
        out = gpu_contiguous(as_gpuarray_variable(out, ctx_name))
        check = Assert(
            "GpuDnnConvGradI: given output (for beta not null) does not have expected shape"
        )
        real_out = check(out, tm.all(tm.eq(out.shape, img_shp)))
    return GpuDnnConvGradI(algo=algo, num_groups=num_groups)(
        kerns, topgrad, real_out, desc, alpha, beta
    )


def dnn_conv(
    fgraph,
    img,
    kerns,
    border_mode="valid",
    subsample=(1, 1),
    dilation=(1, 1),
    conv_mode="conv",
    direction_hint=None,
    workmem=None,
    algo=None,
    precision=None,
    num_groups=1,
):
    """
    GPU convolution using cuDNN from NVIDIA.

    The memory layout to use is 'bc01', that is 'batch', 'channel',
    'first dim', 'second dim' in that order.

    Parameters
    ----------
    fgraph : FunctionGraph
        The function graph containing `img`.
    img
        Images to do the convolution over.
    kerns
        Convolution filters.
    border_mode
        One of 'valid', 'full', 'half'; additionally, the padding size
        could be directly specified by an integer or a pair of integers.
    subsample
        Perform subsampling of the output (default: (1, 1)).
    dilation
        Filter dilation factor. A dilation factor of d is equivalent to a
        convolution with d - 1 zeros inserted between neighboring filter
        values.
    conv_mode
        Perform convolution (kernels flipped) or cross-correlation.
        One of 'conv', 'cross' (default: 'conv').
    direction_hint
        Used by graph optimizers to change algorithm choice.
        By default, GpuDnnConv will be used to carry out the convolution.
        If border_mode is 'valid', subsample is (1, 1), dilation is (1, 1), and
        direction_hint is 'bprop weights', it will use GpuDnnConvGradW.
        If border_mode is 'full', subsample is (1, 1), dilation is (1, 1), and
        direction_hint is *not* 'forward!', it will use GpuDnnConvGradI.
        This parameter is used internally by graph optimizers and may be
        removed at any time without a deprecation period. You have been warned.
    algo : {'none', 'small', 'large', 'fft', 'guess_once', 'guess_on_shape_change', 'time_once', 'time_on_shape_change'}
        Convolution implementation to use. Some of its values may
        require certain versions of cuDNN to be installed. Default is
        the value of :attr:`config.dnn__conv__algo_fwd`.
    precision : {'as_input_f32', 'as_input', 'float16', 'float32', 'float64'}
        Description of the dtype in which the computation of the convolution
        should be done. Possible values are 'as_input', 'float16', 'float32'
        and 'float64'. Default is the value of
        :attr:`config.dnn__conv__precision`.
    num_groups :
        Divides the image, kernel and output tensors into num_groups
        separate groups. Each which carry out convolutions separately


    .. warning:: The cuDNN library only works with GPUs that have a compute
        capability of 3.0 or higher. This means that older GPUs will not
        work with this Op.

    """

    if workmem is not None:
        if algo is not None:
            raise ValueError("You can't use both algo and workmem")
        warnings.warn("workmem is deprecated, use algo instead", stacklevel=2)
        algo = workmem
    ctx_name = infer_context_name(img, kerns)
    if (
        border_mode == "valid"
        and subsample == (1, 1)
        and dilation == (1, 1)
        and direction_hint == "bprop weights"
        and num_groups == 1
    ):
        # Special case: We are asked to use GpuDnnConvGradW. We need to set
        # up a suitable 'fake' convolution to compute the gradient for.
        img = gpu_contiguous(img.dimshuffle(1, 0, 2, 3))
        if conv_mode == "conv":
            # We need to flip manually. These 'kerns' are not the kernels
            # that would be flipped by conv_mode='conv' in GpuDnnConvGradW.
            kerns = kerns[:, :, ::-1, ::-1]
        kerns = gpu_contiguous(kerns.dimshuffle(1, 0, 2, 3))
        out_shp = (
            shape_i(kerns, 1, fgraph),
            shape_i(img, 1, fgraph),
            shape_i(img, 2, fgraph) - shape_i(kerns, 2, fgraph) + 1,
            shape_i(img, 3, fgraph) - shape_i(kerns, 3, fgraph) + 1,
        )
        out_shp = assert_conv_shape(out_shp)
        out = GpuAllocEmpty(dtype=img.dtype, context_name=ctx_name)(*out_shp)
        precision, _ = get_precision(precision, [img, kerns], for_grad=True)
        desc = GpuDnnConvDesc(
            border_mode="valid",
            subsample=(1, 1),
            dilation=(1, 1),
            num_groups=num_groups,
            conv_mode="cross",
            precision=precision,
        )(out.shape)
        conv = GpuDnnConvGradW(num_groups=num_groups)(img, kerns, out, desc)
        return as_gpuarray_variable(conv.dimshuffle(1, 0, 2, 3), ctx_name)

    elif (
        border_mode == "full"
        and subsample == (1, 1)
        and direction_hint != "forward!"
        and num_groups == 1
    ):
        # Special case: We can be faster by using GpuDnnConvGradI to compute
        # the full convolution as the backward pass of a valid convolution.
        # We just need to set up a suitable 'fake' valid convolution.
        img = gpu_contiguous(img)  # cudnn v2 rc3 need contiguous data
        kerns = gpu_contiguous(kerns.dimshuffle(1, 0, 2, 3))
        conv_mode = "cross" if conv_mode == "conv" else "conv"
        out_shp = (
            shape_i(img, 0, fgraph),
            shape_i(kerns, 1, fgraph),
            shape_i(img, 2, fgraph) + (shape_i(kerns, 2, fgraph) - 1) * dilation[0],
            shape_i(img, 3, fgraph) + (shape_i(kerns, 3, fgraph) - 1) * dilation[1],
        )
        out_shp = assert_conv_shape(out_shp)
        out = GpuAllocEmpty(dtype=img.dtype, context_name=ctx_name)(*out_shp)
        precision, _ = get_precision(precision, [img, kerns], for_grad=True)
        desc = GpuDnnConvDesc(
            border_mode="valid",
            subsample=(1, 1),
            dilation=dilation,
            num_groups=num_groups,
            conv_mode=conv_mode,
            precision=precision,
        )(kerns.shape)
        return GpuDnnConvGradI(num_groups=num_groups)(kerns, img, out, desc)

    # Standard case: We use GpuDnnConv with suitable padding.
    return _dnn_conv(
        img,
        kerns,
        algo=algo,
        border_mode=border_mode,
        subsample=subsample,
        dilation=dilation,
        conv_mode=conv_mode,
        precision=precision,
        num_groups=num_groups,
    )


def dnn_conv3d(
    fgraph,
    img,
    kerns,
    border_mode="valid",
    subsample=(1, 1, 1),
    dilation=(1, 1, 1),
    conv_mode="conv",
    direction_hint=None,
    algo=None,
    precision=None,
    num_groups=1,
):
    """
    GPU convolution using cuDNN from NVIDIA.

    The memory layout to use is 'bc012', that is 'batch', 'channel',
    'first dim', 'second dim', 'third dim' in that order.

    Parameters
    ----------
    fgraph : FunctionGraph
        The `FunctionGraph` containing `img`.
    img
        Images to do the convolution over.
    kerns
        Convolution filters.
    border_mode
        One of 'valid', 'full', 'half'; additionally, the padding size
        could be directly specified by an integer or a pair of integers.
    subsample
        Perform subsampling of the output (default: (1, 1, 1)).
    dilation
        Filter dilation factor. A dilation factor of d is equivalent to a
        convolution with d - 1 zeros inserted between neighboring filter
        values.
    conv_mode
        Perform convolution (kernels flipped) or cross-correlation.
        One of 'conv', 'cross' (default: 'conv').
    direction_hint
        Used by graph optimizers to change algorithm choice.
        By default, GpuDnnConv will be used to carry out the convolution.
        If border_mode is 'valid', subsample is (1, 1, 1), dilation is
        (1, 1, 1), and direction_hint is 'bprop weights', it will use
        GpuDnnConvGradW.
        If border_mode is 'full', subsample is (1, 1, 1), dilation is
        (1, 1, 1), and direction_hint is *not* 'forward!', it will use
        GpuDnnConvGradI.
        This parameter is used internally by graph optimizers and may be
        removed at any time without a deprecation period. You have been warned.
    algo : convolution implementation to use. Only 'none' is implemented
        for the conv3d. Default is the value of :attr:`config.dnn__conv__algo_fwd`.
    precision : {'as_input_f32', 'as_input', 'float16', 'float32', 'float64'}
        Description of the dtype in which the computation of the convolution
        should be done. Possible values are 'as_input', 'float16', 'float32'
        and 'float64'. Default is the value of
        :attr:`config.dnn__conv__precision`.
    num_groups :
        Divides the image, kernel and output tensors into num_groups
        separate groups. Each which carry out convolutions separately


    .. warning:: The cuDNN library only works with GPUs that have a compute
        capability of 3.0 or higher. This means that older GPUs will not
        work with this Op.

    """
    ctx_name = infer_context_name(img, kerns)
    if (
        border_mode == "valid"
        and subsample == (1, 1, 1)
        and dilation == (1, 1, 1)
        and direction_hint == "bprop weights"
        and num_groups == 1
    ):
        # Special case: We are asked to use GpuDnnConvGradW. We need to set
        # up a suitable 'fake' convolution to compute the gradient for.
        img = gpu_contiguous(img.dimshuffle(1, 0, 2, 3, 4))
        if conv_mode == "conv":
            # We need to flip manually. These 'kerns' are not the kernels
            # that would be flipped by conv_mode='conv' in GpuDnnConvGradW.
            kerns = kerns[:, :, ::-1, ::-1, ::-1]
        kerns = gpu_contiguous(kerns.dimshuffle(1, 0, 2, 3, 4))
        out_shp = (
            shape_i(kerns, 1, fgraph),
            shape_i(img, 1, fgraph),
            shape_i(img, 2, fgraph) - shape_i(kerns, 2, fgraph) + 1,
            shape_i(img, 3, fgraph) - shape_i(kerns, 3, fgraph) + 1,
            shape_i(img, 4, fgraph) - shape_i(kerns, 4, fgraph) + 1,
        )
        out_shp = assert_conv_shape(out_shp)
        out = GpuAllocEmpty(dtype=img.dtype, context_name=ctx_name)(*out_shp)
        precision, _ = get_precision(precision, [img, kerns], for_grad=True)
        desc = GpuDnnConvDesc(
            border_mode="valid",
            subsample=(1, 1, 1),
            dilation=(1, 1, 1),
            num_groups=num_groups,
            conv_mode="cross",
            precision=precision,
        )(out.shape)
        conv = GpuDnnConvGradW(num_groups=num_groups)(img, kerns, out, desc)
        return as_gpuarray_variable(conv.dimshuffle(1, 0, 2, 3, 4), ctx_name)

    elif (
        border_mode == "full"
        and subsample == (1, 1, 1)
        and direction_hint != "forward!"
        and num_groups == 1
    ):
        # Special case: We can be faster by using GpuDnnConvGradI to compute
        # the full convolution as the backward pass of a valid convolution.
        # We just need to set up a suitable 'fake' valid convolution.
        img = gpu_contiguous(img)  # cudnn v2 rc3 need contiguous data
        kerns = gpu_contiguous(kerns.dimshuffle(1, 0, 2, 3, 4))
        conv_mode = "cross" if conv_mode == "conv" else "conv"
        out_shp = (
            shape_i(img, 0, fgraph),
            shape_i(kerns, 1, fgraph),
            shape_i(img, 2, fgraph) + (shape_i(kerns, 2, fgraph) - 1) * dilation[0],
            shape_i(img, 3, fgraph) + (shape_i(kerns, 3, fgraph) - 1) * dilation[1],
            shape_i(img, 4, fgraph) + (shape_i(kerns, 4, fgraph) - 1) * dilation[2],
        )
        out_shp = assert_conv_shape(out_shp)
        out = GpuAllocEmpty(dtype=img.dtype, context_name=ctx_name)(*out_shp)
        precision, _ = get_precision(precision, [img, kerns], for_grad=True)
        desc = GpuDnnConvDesc(
            border_mode="valid",
            subsample=(1, 1, 1),
            dilation=dilation,
            num_groups=num_groups,
            conv_mode=conv_mode,
            precision=precision,
        )(kerns.shape)
        return GpuDnnConvGradI(num_groups=num_groups)(kerns, img, out, desc)

    # Standard case: We use GpuDnnConv with suitable padding.
    return _dnn_conv(
        img,
        kerns,
        algo=algo,
        border_mode=border_mode,
        subsample=subsample,
        dilation=dilation,
        conv_mode=conv_mode,
        precision=precision,
        num_groups=num_groups,
    )


def dnn_gradweight(
    img,
    topgrad,
    kerns_shp,
    border_mode="valid",
    subsample=(1, 1),
    dilation=(1, 1),
    conv_mode="conv",
    precision=None,
    algo=None,
    num_groups=1,
):
    """
    TODO: document this
    """
    return _dnn_gradweight(
        img,
        topgrad,
        kerns_shp,
        border_mode=border_mode,
        subsample=subsample,
        dilation=dilation,
        conv_mode=conv_mode,
        algo=algo,
        precision=precision,
        num_groups=num_groups,
    )


def dnn_gradweight3d(
    img,
    topgrad,
    kerns_shp,
    border_mode="valid",
    subsample=(1, 1, 1),
    dilation=(1, 1, 1),
    conv_mode="conv",
    precision=None,
    algo=None,
    num_groups=1,
):
    """
    3d version of dnn_gradweight
    """
    return dnn_gradweight(
        img,
        topgrad,
        kerns_shp,
        border_mode,
        subsample,
        dilation,
        conv_mode,
        precision,
        algo,
        num_groups,
    )


def dnn_gradinput(
    kerns,
    topgrad,
    img_shp,
    border_mode="valid",
    subsample=(1, 1),
    dilation=(1, 1),
    conv_mode="conv",
    precision=None,
    algo=None,
    num_groups=1,
):
    """
    TODO: document this
    """
    return _dnn_gradinput(
        kerns,
        topgrad,
        img_shp,
        border_mode=border_mode,
        subsample=subsample,
        dilation=dilation,
        conv_mode=conv_mode,
        algo=algo,
        precision=precision,
        num_groups=num_groups,
    )


def dnn_gradinput3d(
    kerns,
    topgrad,
    img_shp,
    border_mode="valid",
    subsample=(1, 1, 1),
    dilation=(1, 1, 1),
    conv_mode="conv",
    precision=None,
    algo=None,
    num_groups=1,
):
    """
    3d version of `dnn_gradinput`.
    """
    return dnn_gradinput(
        kerns,
        topgrad,
        img_shp,
        border_mode,
        subsample,
        dilation,
        conv_mode,
        precision,
        algo,
        num_groups,
    )


class GpuDnnPoolDesc(_NoPythonCOp):
    """
    This Op builds a pooling descriptor for use in the other
    pooling operations.

    `ws`, `stride` and `pad` must have the same length.

    Parameters
    ----------
    ws : tuple
        Window size.
    stride : tuple
        (dx, dy) or (dx, dy, dz).
    mode : {'max', 'average_inc_pad', 'average_exc_pad'}
        The old deprecated name 'average' corresponds to 'average_inc_pad'.
    pad : tuple
        (padX, padY) or (padX, padY, padZ)

    Notes
    -----
    Not used anymore. Only needed to reload old pickled files.
    """

    __props__ = ("ws", "stride", "mode", "pad")

    def c_headers(self, **kwargs):
        return ["cudnn.h", "cudnn_helper.h"]

    def c_header_dirs(self, **kwargs):
        return [gpuarray_helper_inc_dir(), config.dnn__include_path]

    def c_libraries(self, **kwargs):
        return ["cudnn"]

    def c_lib_dirs(self, **kwargs):
        return [config.dnn__library_path]

    def do_constant_folding(self, fgraph, node):
        return False

    def __init__(self, ws=(1, 1), stride=(1, 1), mode="max", pad=(0, 0)):
        if mode == "average":
            mode = "average_inc_pad"
        assert mode in ("max", "average_inc_pad", "average_exc_pad")
        self.mode = mode

        assert len(ws) == len(stride) and len(stride) == len(pad)
        assert len(ws) in (2, 3)
        self.ws = ws
        self.stride = stride
        self.pad = pad

    def get_ndim(self):
        return len(self.ws)

    def __setstate__(self, d):
        self.__dict__.update(d)
        if not hasattr(self, "pad"):
            self.pad = (0, 0)

    def make_node(self):
        node = Apply(
            self,
            [],
            [
                CUDNNDataType(
                    "cudnnPoolingDescriptor_t", freefunc="cudnnDestroyPoolingDescriptor"
                )()
            ],
        )
        # DebugMode cannot compare the values of CDataType variables, so by
        # default it returns False all the time. To prevent DebugMode from
        # complaining because of the MergeOptimizer, we make this variable
        # always compare to True.
        out = node.outputs[0]
        out.tag.values_eq_approx = values_eq_approx_always_true
        return node

    def c_code(self, node, name, inputs, outputs, sub):
        (desc,) = outputs

        if self.mode == "max":
            mode_flag = "CUDNN_POOLING_MAX"
        elif self.mode == "average_inc_pad":
            mode_flag = "CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING"
        elif self.mode == "average_exc_pad":
            mode_flag = "CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING"
        else:
            raise NotImplementedError("Unsupported pooling model.")

        return """
{
  cudnnStatus_t err;

  if ((err = cudnnCreatePoolingDescriptor(&%(desc)s)) != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_MemoryError, "could not allocate pooling "
                 "descriptor: %%s", cudnnGetErrorString(err));
    %(fail)s
  }

  static const int win[%(nd)d] = {%(win)s};
  static const int pad[%(nd)d] = {%(pad)s};
  static const int str[%(nd)d] = {%(str)s};

    err = cudnnSetPoolingNdDescriptor(%(desc)s, %(mode_flag)s, CUDNN_PROPAGATE_NAN, %(nd)d, win, pad, str);

  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "could not set op descriptor: %%s",
                 cudnnGetErrorString(err));
    %(fail)s
  }
}
""" % dict(
            name=name,
            desc=desc,
            mode_flag=mode_flag,
            fail=sub["fail"],
            nd=self.get_ndim(),
            win=", ".join(map(str, self.ws)),
            pad=", ".join(map(str, self.pad)),
            str=", ".join(map(str, self.stride)),
        )

    def c_code_cache_version(self):
        return (4, version())


class GpuDnnPoolBase(DnnBase):

    """
    Abstract base class for GpuDnnPool and GpuDnnPoolGrad.

    """

    # c_file and c_function must be defined in sub-classes.
    c_file: Optional[str] = None
    c_function: Optional[str] = None

    _f16_ok = True
    __props__ = ("mode",)
    check_input = False
    params_type = ParamsType(mode=cudnn.cudnnPoolingMode_t, handle=handle_type)

    def __init__(self, mode="max"):
        super().__init__([self.c_file], self.c_function)
        if mode == "average":
            mode = "average_inc_pad"
        # Supported modes depend on runtime cuDNN version.
        assert cudnn.cudnnPoolingMode_t.has_alias(mode)
        self.mode = mode


class GpuDnnPool(GpuDnnPoolBase):

    """
    Parameters
    ----------
    img : tensor
        The image 4d or 5d tensor.
    ws : tensor
        Window size.
    stride : tensor
        (dx, dy) or (dx, dy, dz).
    mode : {'max', 'average_inc_pad', 'average_exc_pad'}
        The old deprecated name 'average' corresponds to 'average_inc_pad'.
    pad : tensor
        (padX, padY) or (padX, padY, padZ)

    """

    c_file = "c_code/dnn_pool.c"
    c_function = "APPLY_SPECIFIC(dnn_pool)"

    def make_node(self, img, ws, stride, pad):
        ctx_name = infer_context_name(img)
        img = as_gpuarray_variable(img, ctx_name)

        ws = aet.as_tensor_variable(ws)
        stride = aet.as_tensor_variable(stride)
        pad = aet.as_tensor_variable(pad)
        assert ws.type.ndim == stride.type.ndim and ws.type.ndim == pad.type.ndim
        assert ws.type.ndim == 1

        return Apply(self, [img, ws, stride, pad], [img.type()])

    def infer_shape(self, fgraph, node, shape):
        w = node.inputs[1]
        s = node.inputs[2]
        p = node.inputs[3]

        res = [
            shape[0][0],
            shape[0][1],
            (shape[0][2] + 2 * p[0] - w[0]) // s[0] + 1,
            (shape[0][3] + 2 * p[1] - w[1]) // s[1] + 1,
        ]
        if node.inputs[0].ndim == 5:
            res.append((shape[0][4] + 2 * p[2] - w[2]) // s[2] + 1)
        return [res]

    def L_op(self, inp, outputs, grads):
        img, ws, stride, pad = inp
        (grad,) = grads

        grad = gpu_contiguous(grad)

        (out,) = outputs

        g_out = GpuDnnPoolGrad(mode=self.mode)(img, out, grad, ws, stride, pad)

        return (
            g_out,
            aesara.gradient.DisconnectedType()(),
            aesara.gradient.DisconnectedType()(),
            aesara.gradient.DisconnectedType()(),
        )

    def connection_pattern(self, node):
        # not connected to parameters
        return [[1], [0], [0], [0]]


class GpuDnnPoolGrad(GpuDnnPoolBase):

    """
    The pooling gradient.

    Parameters
    ----------
    inp
        The input of the pooling.
    out
        The output of the pooling in the forward.
    out_grad
        Same size as out, but is the corresponding gradient information.
    ws : tensor variable
        Window size.
    stride : tensor variable
        (dx, dy) or (dx, dy, dz).
    mode : {'max', 'average_inc_pad', 'average_exc_pad'}
        The old deprecated name 'average' corresponds to 'average_inc_pad'.
    pad : tensor
        (padX, padY) or (padX, padY, padZ)

    """

    c_file = "c_code/dnn_pool_grad.c"
    c_function = "APPLY_SPECIFIC(dnn_pool_grad)"

    def make_node(self, inp, out, out_grad, ws, stride, pad):
        ctx_name = infer_context_name(inp, out, out_grad)
        inp = as_gpuarray_variable(inp, ctx_name)
        assert inp.ndim in [4, 5]
        out_grad = as_gpuarray_variable(out_grad, ctx_name)
        assert out_grad.ndim in [4, 5]
        out = as_gpuarray_variable(out, ctx_name)
        assert out.ndim in [4, 5]

        assert out_grad.ndim == inp.ndim
        assert inp.ndim == out.ndim

        ws = aet.as_tensor_variable(ws)
        stride = aet.as_tensor_variable(stride)
        pad = aet.as_tensor_variable(pad)
        assert ws.type.ndim == stride.type.ndim and ws.type.ndim == pad.type.ndim
        assert ws.type.ndim == 1

        return Apply(self, [inp, out, out_grad, ws, stride, pad], [inp.type()])

    def infer_shape(self, fgraph, node, shape):
        return [shape[0]]


def dnn_pool(img, ws, stride=None, mode="max", pad=None):
    """
    GPU pooling using cuDNN from NVIDIA.

    The memory layout to use is 'bc01', that is 'batch', 'channel',
    'first dim', 'second dim' in that order.

    `ws`, `stride` and `pad` must have the same length.

    Parameters
    ----------
    img
        Images to do the pooling over.
    ws : tuple
        Subsampling window size.  Should have 2 or 3 elements.
    stride : tuple
        Subsampling stride (default: (1, 1) or (1, 1, 1)).
    mode : {'max', 'average_inc_pad', 'average_exc_pad', 'sum', 'max_deterministic'}
        **NB**: 'max_deterministic' is supported since cuDNN v6.
    pad : tuple
        (padX, padY) or (padX, padY, padZ)
        default: (0, 0) or (0, 0, 0)


    .. warning:: The cuDNN library only works with GPU that have a compute
        capability of 3.0 or higher.  This means that older GPU will not
        work with this Op.

    Notes
    -----
    This Op implements the ignore_border=True of max_pool_2d.

    """
    img = gpu_contiguous(img)
    if stride is None:
        stride = (1,) * len(ws)
    if pad is None:
        pad = (0,) * len(ws)
    if mode == "sum":
        ret = GpuDnnPool(mode="average_inc_pad")(img, ws, stride, pad)
        context_name = ret.type.context_name
        window_elem = tm.prod(ws).astype(ret.dtype)
        return as_gpuarray_variable(ret * window_elem, context_name)
    return GpuDnnPool(mode=mode)(img, ws, stride, pad)


class GpuDnnSoftmaxBase(DnnBase):

    """
    Op for the cuDNN Softmax.

    Parameters
    ----------
    algo : {'fast', 'accurate', 'log'}
        Indicating whether, respectively, computations should be optimized for
        speed, for accuracy, or if cuDNN should rather compute the log-softmax instead.
    mode : {'instance', 'channel'}
        Indicating whether the softmax should be computed per image across 'c01'
        or per spatial location '01' per image across 'c'.

    """

    __props__ = ("mode", "algo")
    # Neither inputs nor output types properties are used
    # neither in dnn_base.c nor in dnn_softmax*.c,
    # so we can disable input checking.
    check_input = False
    params_type = ParamsType(
        algo=cudnn.cudnnSoftmaxAlgorithm_t,
        mode=cudnn.cudnnSoftmaxMode_t,
        handle=handle_type,
    )

    def __init__(self, algo, mode):
        super().__init__([self.file], self.c_func)

        assert cudnn.cudnnSoftmaxAlgorithm_t.has_alias(algo)
        self.algo = algo

        assert cudnn.cudnnSoftmaxMode_t.has_alias(mode)
        self.mode = mode

    def infer_shape(self, fgraph, node, shape):
        if self.direction == "forward":
            return [shape[0]]
        else:
            return [shape[1]]


class GpuDnnSoftmax(GpuDnnSoftmaxBase):

    """
    Op for the cuDNN Softmax.

    algo : {'fast', 'accurate', 'log'}
        Indicating whether, respectively, computations should be optimized for
        speed, for accuracy, or if cuDNN should rather compute the log-softmax instead.
    mode : {'instance', 'channel'}
        Indicating whether the softmax should be computed per image across 'c01'
        or per spatial location '01' per image across 'c'.

    """

    _f16_ok = True
    direction = "forward"
    file = "c_code/dnn_softmax.c"
    c_func = "APPLY_SPECIFIC(softmax)"

    def make_node(self, x):
        x = as_gpuarray_variable(x, infer_context_name(x))
        assert x.ndim == 4
        return Apply(self, [x], [x.type()])

    def L_op(self, inp, outputs, grads):
        (x,) = inp
        (g_sm,) = grads
        (sm,) = outputs
        return [GpuDnnSoftmaxGrad(self.algo, self.mode)(g_sm, sm)]


class GpuDnnSoftmaxGrad(GpuDnnSoftmaxBase):

    """
    Op for the cuDNN SoftmaxGrad.

    Parameters
    ----------
    algo
        'fast', 'accurate' or 'log' indicating whether, respectively,
        computations should be optimized for speed, for accuracy, or if cuDNN
        should rather compute the gradient of the log-softmax instead.
    mode
        'instance' or 'channel' indicating whether the softmax should
        be computed per image across 'c01' or per spatial location '01' per
        image across 'c'.

    """

    _f16_ok = True
    direction = "backward"
    file = "c_code/dnn_softmax_grad.c"
    c_func = "APPLY_SPECIFIC(softmax_grad)"

    def make_node(self, dy, sm):
        ctx_name = infer_context_name(dy, sm)
        dy = as_gpuarray_variable(dy, ctx_name)
        sm = as_gpuarray_variable(sm, ctx_name)
        assert dy.ndim == 4
        assert sm.ndim == 4
        return Apply(self, [dy, sm], [sm.type()])


class GpuDnnReduction(DnnBase):
    check_input = False
    _f16_ok = True
    _cop_num_outputs = 2

    __props__ = ("red_op", "axis", "acc_dtype", "dtype", "return_indices")

    params_type = ParamsType(
        red_op=cudnn.cudnnReduceTensorOp_t,
        acc_dtype=cudnn.cudnnDataType_t,
        c_axis=uint32_t,
        handle=handle_type,
    )

    def __init__(self, red_op, axis, acc_dtype, dtype, return_indices):
        super().__init__(["c_code/dnn_redux.c"], "APPLY_SPECIFIC(dnn_redux)")
        assert cudnn.cudnnReduceTensorOp_t.has_alias(red_op)
        self.red_op = red_op
        assert acc_dtype in ["float16", "float32", "float64"]
        self.acc_dtype = acc_dtype
        assert dtype in ["float16", "float32", "float64"]
        self.dtype = dtype
        # 8 is the current limit for cudnn
        if axis is not None:
            if len(axis) > 8:
                raise ValueError("Too many axes to reduce on")
            if any(a >= 8 for a in axis):
                raise ValueError("Axes larger than 8 not supported")
            axis = tuple(axis)
        # c_axis is a bitfield (1 to reduce)
        self.c_axis = self._convert_axis(axis)
        # axis is a list of axes to reduce on
        self.axis = axis
        if return_indices and (red_op != "maximum" and red_op != "minimum"):
            raise ValueError(
                "Can't request indices for something other than" " minimum or maximum"
            )
        self.return_indices = return_indices

    def _convert_axis(self, axis):
        if axis is None:
            return np.uint32(-1)
        else:
            return reduce(lambda a, b: a | b, map(lambda a: 1 << a, axis), 0)

    def make_node(self, inp):
        ctx_name = infer_context_name(inp)
        inp = as_gpuarray_variable(inp, ctx_name)
        inp = gpu_contiguous(inp)
        if inp.ndim > 8:
            raise ValueError("cuDNN reduction doesn't support nd > 8")
        assert inp.dtype in ["float16", "float32", "float64"]

        # These restrictions where guessed from vague clues since
        # there is no actual documentation on this
        if inp.dtype == "float64":
            assert self.acc_dtype == "float64"
        if inp.dtype == "float32":
            assert self.acc_dtype == "float32"
        if inp.dtype == "float16":
            assert self.acc_dtype != "float64"

        bcast = []
        for i in range(inp.ndim):
            if not (self.c_axis & (1 << i)):
                bcast.append(inp.broadcastable[i])
        outs = [inp.type.clone(dtype=self.dtype, broadcastable=bcast)()]
        if self.return_indices:
            outs.append(
                GpuArrayType(
                    dtype="uint32", broadcastable=bcast, context_name=ctx_name
                )()
            )

        return Apply(self, [inp], outs)


class GpuDnnBatchNorm(DnnBase):
    """
    Base Op for cuDNN Batch Normalization.

    Parameters
    ----------
    mode : {'per-activation', 'spatial'}
        Whether to normalize per activation (in this mode, bias and scale
        tensor dimensions are 1xCxHxW) or share normalization factors across
        spatial dimensions (in this mode, bias and scale tensor dimensions
        are 1xCx1x1).
    epsilon
        Epsilon value used in the batch normalization formula. Minimum allowed
        value is 1e-5 (imposed by cuDNN).
    running_average_factor : float
        Factor for updating the values or `running_mean` and `running_var`.
        If the factor is close to one, the running averages will update quickly,
        if the factor is close to zero it will update slowly.
    running_mean : tensor or None
        Previous value of the running mean. If this is given, the new value
        ``running_mean * (1 - r_a_factor) + batch mean * r_a_factor``
        will be returned as one of the outputs of this function.
        `running_mean` and `running_var` should either both be given or
        both be None.
    running_var : tensor or None
        Previous value of the running variance. If this is given, the new value
        ``running_var * (1 - r_a_factor) + (m / (m - 1)) * batch var * r_a_factor``
        will be returned as one of the outputs of this function,
        where `m` is the product of lengths of the averaged-over dimensions.
        `running_mean` and `running_var` should either both be given or
        both be None.
    """

    __props__ = (
        "mode",
        "running_averages",
        "inplace_running_mean",
        "inplace_running_var",
        "inplace_output",
    )
    _cop_num_inputs = 7
    _cop_num_outputs = 5
    check_input = False
    params_type = ParamsType(
        mode=cudnn.cudnnBatchNormMode_t,
        inplace_output=bool_t,
        inplace_running_mean=bool_t,
        inplace_running_var=bool_t,
        handle=handle_type,
    )

    def __init__(
        self,
        mode="per-activation",
        running_averages=False,
        inplace_running_mean=False,
        inplace_running_var=False,
        inplace_output=False,
    ):
        super().__init__(
            ["c_code/dnn_batchnorm_base.c", "c_code/dnn_batchnorm.c"],
            "dnn_batchnorm_op",
        )

        assert cudnn.cudnnBatchNormMode_t.has_alias(mode)
        self.mode = mode
        self.running_averages = running_averages
        self.inplace_output = inplace_output
        self.inplace_running_mean = inplace_running_mean
        self.inplace_running_var = inplace_running_var
        self.destroy_map = {}
        if self.inplace_output:
            self.destroy_map[0] = [0]
        if self.running_averages and self.inplace_running_mean:
            self.destroy_map[3] = [5]
        if self.running_averages and self.inplace_running_var:
            self.destroy_map[4] = [6]

    def __setstate__(self, d):
        self.__dict__.update(d)
        if not hasattr(self, "running_average_factor"):
            self.running_average_factor = 0
        if not hasattr(self, "running_averages"):
            self.running_averages = False
        if not (
            hasattr(self, "inplace_running_mean")
            and hasattr(self, "inplace_running_var")
            and hasattr(self, "inplace_output")
        ):
            self.inplace_running_mean = False
            self.inplace_running_var = False
            self.inplace_output = False
            self.destroy_map = {}

    def infer_shape(self, fgraph, node, shape):
        return [shape[0]] + [shape[1]] * (len(node.outputs) - 1)

    def make_node(
        self,
        x,
        scale,
        bias,
        epsilon=1e-4,
        running_average_factor=0.1,
        running_mean=None,
        running_var=None,
    ):
        assert x.ndim == scale.ndim == bias.ndim
        assert x.ndim in (4, 5)
        assert (
            self.running_averages
            == (running_mean is not None)
            == (running_var is not None)
        )
        assert running_mean is None or running_mean.ndim == x.ndim
        assert running_var is None or running_var.ndim == x.ndim
        ctx_name = infer_context_name(x, scale, bias)
        x = as_gpuarray_variable(x, ctx_name)
        scale = as_gpuarray_variable(scale, ctx_name)
        bias = as_gpuarray_variable(bias, ctx_name)
        epsilon = as_scalar(epsilon).astype("float64")
        running_average_factor = as_scalar(running_average_factor).astype("float64")
        inputs = [x, scale, bias, epsilon, running_average_factor]
        output_types = [x.type(), scale.type(), scale.type()]
        if running_mean is not None and running_var is not None:
            inputs.append(as_gpuarray_variable(running_mean, ctx_name))
            inputs.append(as_gpuarray_variable(running_var, ctx_name))
            output_types.append(scale.type())
            output_types.append(scale.type())
        return Apply(self, inputs, output_types)

    def L_op(self, inputs, outputs, grads):
        x, scale, bias, epsilon, running_average_factor = inputs[:5]
        dy = grads[0]
        _, x_mean, x_invstd = outputs[:3]
        disconnected_outputs = [
            DisconnectedType()(),  # epsilon
            DisconnectedType()(),
        ]  # running_average_factor
        # Optional running_mean and running_var.
        for i in range(5, len(inputs)):
            disconnected_outputs.append(DisconnectedType()())
        return (
            GpuDnnBatchNormGrad(self.mode)(x, dy, scale, x_mean, x_invstd, epsilon)
            + disconnected_outputs
        )

    def connection_pattern(self, node):
        # Specificy that epsilon and running_average_factor are not connected to outputs.
        patterns = [
            [True, True, True],  # x
            [True, True, True],  # scale
            [True, True, True],  # bias
            [False, False, False],  # epsilon
            [False, False, False],
        ]  # running_average_factor
        # Optional running_mean and running_var are only
        # connected to their new values.
        for i in range(5, len(node.inputs)):
            patterns[0].append(True)
            for pattern in patterns[1:]:
                pattern.append(False)
            patterns.append([False] * (3 + i - 5) + [True])
        return patterns


class GpuDnnBatchNormInference(DnnBase):
    """
    Base Op for cuDNN Batch Normalization.

    Parameters
    ----------
    mode : {'per-activation', 'spatial'}
        Whether to normalize per activation (in this mode, bias and scale
        tensor dimensions are 1xCxHxW) or share normalization factors across
        spatial dimensions (in this mode, bias and scale tensor dimensions
        are 1xCx1x1).
    epsilon
        Epsilon value used in the batch normalization formula. Minimum allowed
        value is 1e-5 (imposed by cuDNN).
    """

    __props__ = ("mode", "inplace")

    check_input = False
    params_type = ParamsType(
        mode=cudnn.cudnnBatchNormMode_t, inplace=bool_t, handle=handle_type
    )

    def __init__(self, mode="per-activation", inplace=False):
        super().__init__(
            ["c_code/dnn_batchnorm_base.c", "c_code/dnn_batchnorm_inf.c"],
            "dnn_batchnorm_op",
        )

        assert cudnn.cudnnBatchNormMode_t.has_alias(mode)
        self.mode = mode
        self.inplace = bool(inplace)
        if self.inplace:
            self.destroy_map = {0: [0]}

    def __setstate__(self, d):
        self.__dict__.update(d)
        if not hasattr(self, "inplace"):
            self.inplace = False

    def infer_shape(self, fgraph, node, shape):
        return [shape[0]]

    def make_node(
        self, x, scale, bias, estimated_mean, estimated_variance, epsilon=1e-4
    ):
        ctx_name = infer_context_name(
            x, scale, bias, estimated_mean, estimated_variance
        )
        x = as_gpuarray_variable(x, ctx_name)
        scale = as_gpuarray_variable(scale, ctx_name)
        bias = as_gpuarray_variable(bias, ctx_name)
        estimated_mean = as_gpuarray_variable(estimated_mean, ctx_name)
        estimated_variance = as_gpuarray_variable(estimated_variance, ctx_name)
        epsilon = as_scalar(epsilon).astype("float64")
        assert (
            x.ndim
            == scale.ndim
            == bias.ndim
            == estimated_mean.ndim
            == estimated_variance.ndim
        )
        assert x.ndim in (4, 5)
        return Apply(
            self,
            [x, scale, bias, estimated_mean, estimated_variance, epsilon],
            [x.type()],
        )

    def grad(self, inputs, grads):
        x, scale, bias, est_mean, est_var, epsilon = inputs
        dy = grads[0]

        if self.mode == "per-activation":
            axes = (0,)
        elif self.mode == "spatial":
            axes = (0,) + tuple(range(2, x.ndim))
        scale, bias, est_mean, est_var = (
            aet.addbroadcast(t, *axes) for t in (scale, bias, est_mean, est_var)
        )

        # define helper expressions
        est_var_eps = est_var + epsilon
        est_std = tm.sqrt(est_var_eps)
        two = aet.constant(2.0)

        # define and return gradients
        dx = dy * (scale / est_std)
        dscale = (dy * (x - est_mean)).sum(axes, keepdims=True) / est_std
        dbias = dy.sum(axes, keepdims=True)
        dmean = -dy.sum(axes, keepdims=True) * (scale / est_std)
        dvar = -(dy * (x - est_mean)).sum(axes, keepdims=True) * (
            scale / (two * est_var_eps * est_std)
        )
        return [dx, dscale, dbias, dmean, dvar, DisconnectedType()()]

    def connection_pattern(self, node):
        # Specificy that epsilon is not connected to outputs.
        return [[True], [True], [True], [True], [True], [False]]


class GpuDnnBatchNormGrad(DnnBase):
    __props__ = ("mode",)

    check_input = False
    params_type = ParamsType(mode=cudnn.cudnnBatchNormMode_t, handle=handle_type)

    def __init__(self, mode="per-activation"):
        super().__init__(
            ["c_code/dnn_batchnorm_base.c", "c_code/dnn_batchnorm_grad.c"],
            "dnn_batchnorm_grad",
        )

        assert cudnn.cudnnBatchNormMode_t.has_alias(mode)
        self.mode = mode

    def make_node(self, x, dy, scale, x_mean, x_invstd, epsilon=1e-4):
        ctx_name = infer_context_name(x, dy, scale, x_mean, x_invstd)
        x = as_gpuarray_variable(x, ctx_name)
        dy = as_gpuarray_variable(dy, ctx_name)
        scale = as_gpuarray_variable(scale, ctx_name)
        x_mean = as_gpuarray_variable(x_mean, ctx_name)
        x_invstd = as_gpuarray_variable(x_invstd, ctx_name)
        epsilon = as_scalar(epsilon).astype("float64")
        assert x.ndim == dy.ndim == scale.ndim == x_mean.ndim == x_invstd.ndim
        assert x.ndim in (4, 5)
        return Apply(
            self,
            [x, dy, scale, x_mean, x_invstd, epsilon],
            [x.type(), scale.type(), scale.type()],
        )

    def infer_shape(self, fgraph, node, shape):
        return [shape[0], shape[2], shape[2]]


gpudata_type = CDataType("gpudata *", "gpudata_release")
dropoutdesc_type = CUDNNDataType(
    "cudnnDropoutDescriptor_t", "cudnnDestroyDropoutDescriptor"
)


class GpuDnnDropoutOp(DnnBase):
    __props__ = ("inplace",)

    def __init__(self, inplace=False):
        super().__init__(["c_code/dnn_dropout_fwd.c"], "dnn_dropout_fwd")
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {1: [2]}

    def make_node(self, inp, descriptor, state):
        ctx_name = infer_context_name(inp)
        inp = as_gpuarray_variable(inp, ctx_name)
        return Apply(
            self, [inp, descriptor, state], [inp.type(), state.type(), gpudata_type()]
        )

    def prepare_node(self, node, storage_map, compute_map, impl):
        assert self.inplace, "GpuDnnDropoutOp not inplace"


class _DropoutDescriptor(DnnBase):
    __props__ = ("context_name",)

    def __init__(self, context_name):
        super().__init__(["c_code/dnn_dropout_desc.c"], "dnn_dropout_desc")
        self.context_name = context_name

    def dnn_context(self, node):
        return self.context_name

    def do_constant_folding(self, fgraph, node):
        return False

    def make_node(self, dropout, seed, context_name):
        dropout = as_scalar(dropout).astype("float32")
        seed = as_scalar(seed).astype("uint64")

        assert context_name == self.context_name
        # This is a dirty hack to pass the context because params is
        # occupied by the cudnn handle
        context = gpu_context_type.make_constant(get_context(context_name))

        return Apply(
            self,
            [dropout, seed, context],
            [
                dropoutdesc_type(),
                GpuArrayType("uint8", (False,), context_name=context_name)(),
            ],
        )

    def c_code_cache_version_apply(self, node):
        # disable the cache since we can't pickle contexts
        return None


def _make_dropout_desc(dropout, seed, context_name):
    desc, states = aesara.function(
        [],
        _DropoutDescriptor(context_name)(dropout, seed, context_name),
        Mode(optimizer=None),
        profile=False,
    )()
    return desc, states


def dropout(x, dropout=0.0, seed=4242):
    desc, states = _make_dropout_desc(dropout, seed, x.type.context_name)
    y, odesc = GpuDnnDropoutOp()(x, desc)
    return y, desc, odesc, states


rnndesc_type = CUDNNDataType("cudnnRNNDescriptor_t", "cudnnDestroyRNNDescriptor")


def as_i32(v):
    return as_scalar(v).astype("int32")


class _RNNDescriptor(DnnBase):
    __props__ = ("context_name",)

    def __init__(self, context_name):
        if version() < 5005:
            raise RuntimeError("cudnn RNN require cudnn v5 final or higher.")
        super().__init__(["c_code/dnn_rnn_desc.c"], "dnn_rnn_desc")
        self.context_name = context_name

    def dnn_context(self, node):
        return self.context_name

    def do_constant_folding(self, fgraph, node):
        return False

    def make_node(
        self,
        hidden_size,
        num_layers,
        ddesc,
        input_mode,
        direction_mode,
        rnn_mode,
        dtype,
    ):

        hidden_size = as_i32(hidden_size)
        num_layers = as_i32(num_layers)

        if version() < 5005:
            raise RuntimeError("cudnn RNN require cudnn v5 final or higher.")

        if input_mode == "linear":
            input_mode = as_i32(0)
        elif input_mode == "skip":
            input_mode = as_i32(1)
        else:
            raise ValueError("input_mode")

        if direction_mode == "unidirectional":
            direction_mode = as_i32(0)
        elif direction_mode == "bidirectional":
            direction_mode = as_i32(1)
        else:
            raise ValueError("direction_mode")

        if rnn_mode == "rnn_relu":
            rnn_mode = as_i32(0)
        elif rnn_mode == "rnn_tanh":
            rnn_mode = as_i32(1)
        elif rnn_mode == "lstm":
            rnn_mode = as_i32(2)
        elif rnn_mode == "gru":
            rnn_mode = as_i32(3)
        else:
            raise ValueError("rnn_mode")

        dtype = as_i32(gpuarray.dtype_to_typecode(dtype))

        return Apply(
            self,
            [
                hidden_size,
                num_layers,
                dropoutdesc_type.make_constant(ddesc),
                input_mode,
                direction_mode,
                rnn_mode,
                dtype,
            ],
            [rnndesc_type()],
        )


def _make_rnn_desc(
    hidden_size,
    num_layers,
    ddesc,
    rnn_mode,
    input_mode,
    direction_mode,
    dtype,
    context_name,
):
    desc = aesara.function(
        [],
        _RNNDescriptor(context_name)(
            hidden_size, num_layers, ddesc, input_mode, direction_mode, rnn_mode, dtype
        ),
        Mode(optimizer=None),
        profile=False,
    )()
    return desc


class _RNNParamSize(DnnBase):
    __props__ = ("context_name",)

    def __init__(self, context_name):
        super().__init__(["c_code/dnn_rnn_paramsize.c"], "dnn_rnn_paramsize")
        self.context_name = context_name

    def dnn_context(self, node):
        return self.context_name

    def do_constant_folding(self, fgraph, node):
        return False

    def make_node(self, desc, input_size, typecode):
        input_size = as_tensor_variable(input_size).astype("uint64")
        typecode = as_i32(typecode)
        return Apply(
            self,
            [rnndesc_type.make_constant(desc), input_size, typecode],
            [get_scalar_type("uint64")()],
        )


def _get_param_size(desc, input_size, dtype, context_name):
    typecode = gpuarray.dtype_to_typecode(dtype)
    return aesara.function(
        [],
        _RNNParamSize(context_name)(desc, input_size, typecode),
        Mode(optimizer=None),
        profile=False,
    )()


class _RNNSplitParams(DnnBase):
    __props__ = ("rnn_mode",)

    def __init__(self, rnn_mode):
        super().__init__()
        self.rnn_mode = rnn_mode

    def make_node(self, w, desc, layer, isize, typecode):
        w = as_gpuarray_variable(w, infer_context_name(w))
        assert w.ndim == 1
        layer = as_scalar(layer).astype("int32")
        isize = as_tensor_variable(isize).astype("uint64")
        assert isize.ndim == 1
        typecode = as_scalar(typecode).astype("int32")
        _1d = GpuArrayType(w.type.dtype, [False], context_name=w.type.context_name)
        _2d = GpuArrayType(
            w.type.dtype, [False, False], context_name=w.type.context_name
        )
        outputs = []
        if self.rnn_mode == "rnn_relu" or self.rnn_mode == "rnn_tanh":
            outputs.extend([_2d(), _1d()])  # input
            outputs.extend([_2d(), _1d()])  # recurrent
        elif self.rnn_mode == "lstm":
            outputs.extend([_2d(), _1d()])  # input input
            outputs.extend([_2d(), _1d()])  # input forget
            outputs.extend([_2d(), _1d()])  # input newmem
            outputs.extend([_2d(), _1d()])  # input output
            outputs.extend([_2d(), _1d()])  # recur input
            outputs.extend([_2d(), _1d()])  # recur forget
            outputs.extend([_2d(), _1d()])  # recur newmem
            outputs.extend([_2d(), _1d()])  # recur output
        elif self.rnn_mode == "gru":
            outputs.extend([_2d(), _1d()])  # input reset
            outputs.extend([_2d(), _1d()])  # input update
            outputs.extend([_2d(), _1d()])  # input newmem
            outputs.extend([_2d(), _1d()])  # recur reset
            outputs.extend([_2d(), _1d()])  # recur update
            outputs.extend([_2d(), _1d()])  # recur newmem

        return Apply(
            self, [w, layer, rnndesc_type.make_constant(desc), isize, typecode], outputs
        )

    def c_code(self, node, name, inputs, outputs, sub):
        kw = dict(
            fail=sub["fail"],
            w=inputs[0],
            layer=inputs[1],
            desc=inputs[2],
            isize=inputs[3],
            typecode=inputs[4],
            handle=sub["params"],
        )
        code = (
            """
  cudnnTensorDescriptor_t xdesc;
  cudnnFilterDescriptor_t wdesc;
  cudnnFilterDescriptor_t odesc;
  size_t nshp[2];
  void *w;
  void *o;
  ptrdiff_t off;
#if CUDNN_VERSION < 7100
  size_t bshp;
#endif
  cudnnStatus_t err;
  cudnnDataType_t dt;
  cudnnTensorFormat_t tf;
  int nd;
  int dims[3];
  int strs[3];

  if (PyArray_DIM(%(isize)s, 0) != 2) {
    PyErr_SetString(PyExc_ValueError, "input_size should be of length two");
    %(fail)s;
  }

  switch (%(typecode)s) {
  case GA_FLOAT:
    dt = CUDNN_DATA_FLOAT;
    break;
  case GA_DOUBLE:
    dt = CUDNN_DATA_DOUBLE;
    break;
  case GA_HALF:
    dt = CUDNN_DATA_HALF;
    break;
  default:
    PyErr_SetString(PyExc_ValueError, "Unsupported data type");
    %(fail)s;
  }

  err = cudnnCreateTensorDescriptor(&xdesc);
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_SetString(PyExc_RuntimeError, "Could not create xdesc");
    %(fail)s;
  }

  dims[0] = *(npy_uint64 *)PyArray_GETPTR1(%(isize)s, 0);
  dims[1] = *(npy_uint64 *)PyArray_GETPTR1(%(isize)s, 1);
  dims[2] = 1;
  strs[0] = dims[2] * dims[1];
  strs[1] = dims[2];
  strs[2] = 1;

  err = cudnnSetTensorNdDescriptor(xdesc, dt, 3, dims, strs);
  if (err != CUDNN_STATUS_SUCCESS) {
    cudnnDestroyTensorDescriptor(xdesc);
    PyErr_Format(PyExc_RuntimeError, "Could not set xdesc: %%s",
                 cudnnGetErrorString(err));
    %(fail)s;
  }

  if (c_make_filter(%(w)s, &wdesc)) {
    cudnnDestroyTensorDescriptor(xdesc);
    %(fail)s
  }

  err = cudnnCreateFilterDescriptor(&odesc);
  if (err != CUDNN_STATUS_SUCCESS) {
    PyErr_SetString(PyExc_RuntimeError, "could not create odesc");
    cudnnDestroyTensorDescriptor(xdesc);
    cudnnDestroyFilterDescriptor(wdesc);
    %(fail)s
  }

  w = PyGpuArray_DEV_DATA(%(w)s);
  nshp[0] = PyGpuArray_DIM(%(w)s, 0);
  nshp[1] = 1;
        """
            % kw
        )

        def get_params(id, m, b):
            kw2 = kw.copy()
            kw2["id"] = id
            kw2["m"] = m
            kw2["b"] = b
            return (
                """
  err = cudnnGetRNNLinLayerBiasParams(%(handle)s, %(desc)s, %(layer)s, xdesc, wdesc, w, %(id)s, odesc, &o);
  if (err != CUDNN_STATUS_SUCCESS) {
    cudnnDestroyTensorDescriptor(xdesc);
    cudnnDestroyFilterDescriptor(wdesc);
    cudnnDestroyFilterDescriptor(odesc);
    PyErr_SetString(PyExc_RuntimeError, "can't fetch bias for id %(id)s");
    %(fail)s
  }
  off = (intptr_t)o - (intptr_t)w;
  assert(off >= 0 && "bias");

  err = cudnnGetFilterNdDescriptor(odesc, 3, &dt, &tf, &nd, dims);
  if (err != CUDNN_STATUS_SUCCESS) {
    cudnnDestroyTensorDescriptor(xdesc);
    cudnnDestroyFilterDescriptor(wdesc);
    cudnnDestroyFilterDescriptor(odesc);
    PyErr_SetString(PyExc_RuntimeError, "could not get bias shape for id %(id)s");
    %(fail)s;
  }
  // We assume that the typecode matches
#if CUDNN_VERSION < 7100
  assert(dims[2] == 1 && "bias");
  assert(dims[1] == 1 && "bias");
  %(b)s = pygpu_view(%(w)s, Py_None);
  %(b)s->ga.offset += off;
  %(b)s->ga.dimensions[0] = dims[0];
  bshp = dims[0];
#else
  assert(dims[0] == 1 && "bias");
  assert(dims[2] == 1 && "bias");
  %(b)s = pygpu_view(%(w)s, Py_None);
  %(b)s->ga.offset += off;
  %(b)s->ga.dimensions[0] = dims[1];
#endif
  GpuArray_fix_flags(&%(b)s->ga);

  err = cudnnGetRNNLinLayerMatrixParams(%(handle)s, %(desc)s, %(layer)s, xdesc, wdesc, w, %(id)s, odesc, &o);
  if (err != CUDNN_STATUS_SUCCESS) {
    cudnnDestroyTensorDescriptor(xdesc);
    cudnnDestroyFilterDescriptor(wdesc);
    cudnnDestroyFilterDescriptor(odesc);
    PyErr_SetString(PyExc_RuntimeError, "can't fetch matrix for id %(id)s");
    %(fail)s
  }
  off = (intptr_t)o - (intptr_t)w;
  assert(off >= 0 && "matrix");

  // This is 3d because of cudnn limitations.
  err = cudnnGetFilterNdDescriptor(odesc, 3, &dt, &tf, &nd, dims);
  if (err != CUDNN_STATUS_SUCCESS) {
    cudnnDestroyTensorDescriptor(xdesc);
    cudnnDestroyFilterDescriptor(wdesc);
    cudnnDestroyFilterDescriptor(odesc);
    PyErr_SetString(PyExc_RuntimeError, "could not get matrix shape for id %(id)s");
    %(fail)s;
  }

#if CUDNN_VERSION < 7100
  assert(dims[1] == 1 && "matrix");
  assert(dims[2] == 1 && "matrix");
  // We assume that the typecode matches
  %(m)s = pygpu_reshape(%(w)s, 2, nshp, GA_F_ORDER, 1, -1);
  %(m)s->ga.offset += off;
  assert(dims[0] %% bshp == 0);
  %(m)s->ga.dimensions[0] = dims[0] / bshp;
  %(m)s->ga.dimensions[1] = bshp;
#else
  assert(dims[0] == 1 && "matrix");
  // We assume that the typecode matches
  %(m)s = pygpu_reshape(%(w)s, 2, nshp, GA_F_ORDER, 1, -1);
  %(m)s->ga.offset += off;
  %(m)s->ga.dimensions[1] = dims[1];
  %(m)s->ga.dimensions[0] = dims[2];
#endif
  %(m)s->ga.strides[1] = %(m)s->ga.dimensions[0] * gpuarray_get_elsize(%(m)s->ga.typecode);
  GpuArray_fix_flags(&%(m)s->ga);
            """
                % kw2
            )

        for i in range(len(outputs) // 2):
            code += get_params(i, outputs[2 * i], outputs[(2 * i) + 1])

        code += """
  cudnnDestroyTensorDescriptor(xdesc);
  cudnnDestroyFilterDescriptor(wdesc);
  cudnnDestroyFilterDescriptor(odesc);
        """
        return code

    def c_code_cache_version(self):
        return (5, version())


def _split_rnn_params(w, desc, layer, input_size, dtype, rnn_mode):
    typecode = gpuarray.dtype_to_typecode(dtype)
    outs = _RNNSplitParams(rnn_mode)(w, desc, layer, input_size, typecode)
    outs = [Out(o, borrow=True) for o in outs]
    return aesara.function([], outs, Mode(optimizer=None), profile=False)()


class GpuDnnRNNOp(DnnBase):
    __props__ = ()
    _cop_num_inputs = 6
    _cop_num_outputs = 4

    def __init__(self, rnn_mode, direction_mode):
        super().__init__(["c_code/dnn_rnn_fwd.c"], "dnn_rnn_fwd")
        self.rnn_mode = rnn_mode
        if direction_mode == "bidirectional":
            self.num_dirs = 2
        elif direction_mode == "unidirectional":
            self.num_dirs = 1
        else:
            raise ValueError(f"direction_mode is invalid (got {direction_mode})")

    def dnn_context(self, node):
        return node.outputs[1].type.context_name

    def make_node(self, desc, w, x, hx, cx=None):
        if cx is None:
            context_name = infer_context_name(w, x, hx)
        else:
            context_name = infer_context_name(w, x, hx, cx)

        w = as_gpuarray_variable(w, context_name)
        x = as_gpuarray_variable(x, context_name)
        hx = as_gpuarray_variable(hx, context_name)
        inputs = [desc, as_i32(self.num_dirs), w, x, hx]
        assert w.ndim == 1
        assert x.ndim == 3  # seqLength, minibatch, inputSize
        assert hx.ndim == 3  # numLayers, minibatch, hiddenSize * bidi

        if self.rnn_mode == "lstm":
            cx = as_gpuarray_variable(cx, context_name)
            assert cx.ndim == 3  # numLayers, minibatch, hiddenSize * bidi
            inputs.append(cx)

        _3d = GpuArrayType(
            dtype=x.dtype,
            broadcastable=(False, False, False),
            context_name=context_name,
        )
        reserve = gpudata_type()
        y = _3d()  # seqLength, minibatch, hiddenSize * bidi
        hy = _3d()  # numLayers, miniBatch, hiddenSize * bidi
        outputs = [reserve, y, hy]

        if self.rnn_mode == "lstm":
            cy = _3d()  # numLayers, miniBatch, hiddenSize * bidi
            outputs.append(cy)

        return Apply(self, inputs, outputs)

    def L_op(self, inputs, outputs, output_grads):
        desc, numDirs, w, x, hx = inputs[:5]
        cx = inputs[5] if len(inputs) == 6 else None
        reserve, y, hy = outputs[:3]
        _, dy, dhy = output_grads[:3]
        dcy = output_grads[3] if len(output_grads) == 4 else None
        # Since the op return two outputs which contain essentially
        # the same information, the user will most likely only use one
        # of them. This leads to the situation that the other is
        # considered "disconnected" by aesara in the gradient.
        # However we know that this isn't really the case so we fix it
        # here.

        # If all the ys are disconnected, then you get a boring
        # gradient instead of an error.  But in that case you
        # shouldn't call this method anyway.
        if isinstance(dy.type, DisconnectedType):
            dy = as_gpuarray_variable(y.zeros_like(), context_name=y.type.context_name)
        if isinstance(dhy.type, DisconnectedType):
            dhy = None
        if dcy and isinstance(dcy.type, DisconnectedType):
            dcy = None
        dinputs = GpuDnnRNNGradInputs(
            rnn_mode=self.rnn_mode, grad_h=(dhy is not None), grad_c=(dcy is not None)
        )(desc, x, y, dy, dhy, dcy, w, hx, cx, reserve, return_list=True)
        reserve2, dx, dhx = dinputs[:3]
        dw = GpuDnnRNNGradWeights()(desc, x, hx, y, reserve2, w)
        res = [DisconnectedType()(), DisconnectedType()(), dw, dx, dhx]
        if cx is not None:
            res.append(dinputs[3])  # dcx
        return res

    def connection_pattern(self, node):
        deconn = [[False] * len(node.outputs)] * 2
        conn = [[True] * len(node.outputs)] * (len(node.inputs) - 2)
        return deconn + conn


class GpuDnnRNNGradInputs(DnnBase):
    __props__ = ("rnn_mode", "grad_c", "grad_h")
    _cop_num_inputs = 10
    _cop_num_outputs = 4

    def __init__(self, rnn_mode, grad_h, grad_c):
        super().__init__(["c_code/dnn_rnn_gi.c"], "dnn_rnn_gi")
        self.rnn_mode = rnn_mode
        self.grad_h = grad_h
        self.grad_c = grad_c
        if self.grad_c:
            assert self.rnn_mode == "lstm"

    def dnn_context(self, node):
        return node.outputs[1].type.context_name

    def make_node(self, desc, x, y, dy, dhy, dcy, w, hx, cx, reserve):
        # We trust the callers here
        xshp = as_scalar(x.shape[2]).astype("uint64")
        inputs = [desc, xshp, y, dy, w, hx, reserve]
        outputs = [reserve.type(), x.type(), hx.type()]
        if self.rnn_mode == "lstm":
            inputs.append(cx)
            outputs.append(cx.type())
        if self.grad_h:
            inputs.append(dhy)
        if self.grad_c:
            inputs.append(dcy)

        return Apply(self, inputs, outputs)

    # We have special requirements so this is hooking into ExternalCOp
    def format_c_function_args(self, inp, out):
        rinp = inp[:7]
        others = inp[7:]
        if self.rnn_mode == "lstm":
            rinp.append(others.pop(0))
        else:
            rinp.append("NULL")
        if self.grad_h:
            rinp.append(others.pop(0))
        else:
            rinp.append("NULL")
        if self.grad_c:
            rinp.append(others.pop(0))
        else:
            rinp.append("NULL")
        assert len(others) == 0
        return ExternalCOp.format_c_function_args(self, rinp, out)


class GpuDnnRNNGradWeights(DnnBase):
    __props__ = ()

    def __init__(self):
        super().__init__(["c_code/dnn_rnn_gw.c"], "dnn_rnn_gw")

    def make_node(self, desc, x, hx, y, reserve, w):
        # We trust the callers here
        wsize = as_scalar(w.shape[0]).astype("uint64")
        inputs = [desc, wsize, x, hx, y, reserve]
        outputs = [w.type()]
        return Apply(self, inputs, outputs)


class RNNBlock:
    """
    An object that allow us to use CuDNN RNN implementation.
    TODO: make an example how to use. You can check Aesara tests
    test_dnn_rnn_gru() and test_dnn_rnn_lstm() in the file
    aesara/gpuarray/tests/test_dnn.py for now.


    Parameters
    ----------
    dtype : data type of computation
    hidden_size : int
        hidden layer dimension.
    num_layers : int
        number of the recurrent layer you want to set.
    rnn_mode : {'rnn_relu', 'rnn_tanh', 'lstm', 'gru'}
        rnn_relu: A single-gate recurrent neural network with a ReLU activation function.

        .. math::

        h_t=ReLU(W_ix_t+U_ih_{t-1}+b_{wi}+b_{Ri})
        rnn_tanh: A single-gate recurrent neural network with a tanh activation function.

        .. math::

        h_t=tanh(W_ix_t+U_ih_{t-1}+b_{wi}+b_{Ri})

        lstm: A four-gate Long Short-Term Memory network with no peephole connections.
        gru: A three-gate network consisting of Gated Recurrent Units.
    input_mode : {'linear', 'skip'}
        linear: input will be multiplied by a biased matrix
        skip: No operation is performed on the input.  The size must match the hidden size.
    direction_mode : {'unidirectional', 'bidirectional'}
        unidirectional: The network operates recurrently from the first input to the last.
        bidirectional: The network operates from first to last then from last to first and concatenates the results at each layer.

    """

    def __init__(
        self,
        dtype,
        hidden_size,
        num_layers,
        rnn_mode,
        input_mode="linear",
        direction_mode="unidirectional",
        context_name=None,
    ):
        # This is not supported for any value other than 0, so don't change it
        ddesc, states = _make_dropout_desc(0, 4242, context_name)
        self.ddesc = ddesc
        self.dstates = states
        self.desc = _make_rnn_desc(
            hidden_size,
            num_layers,
            ddesc,
            rnn_mode,
            input_mode,
            direction_mode,
            dtype,
            context_name,
        )
        self.rnn_mode = rnn_mode
        self.direction_mode = direction_mode
        self.context_name = context_name
        self.dtype = dtype

    def get_param_size(self, input_size):
        """
        Get the size of the shared variable for the parameters of the RNN.

        This will return a size (in items) necessary to store all the
        parameters for the RNN.  You should allocate a variable of
        that size to store those parameters.  The order and layout of
        the parameters is opaque.

        Parameters
        ----------
        input_size: (int, int)
            Size of the input blocks

        """
        bytesize = _get_param_size(self.desc, input_size, self.dtype, self.context_name)
        bytesize = int(bytesize)
        assert bytesize % np.dtype(self.dtype).itemsize == 0
        return bytesize // np.dtype(self.dtype).itemsize

    def split_params(self, w, layer, input_size):
        """
        Split the opaque parameter block into components.

        Parameters
        ----------
        w: GpuArraySharedVariable
            opaque parameter block
        layer: int
            ID of the layer
        input_size: (int, int)
            Size of the input blocks

        """
        if not isinstance(w, GpuArraySharedVariable):
            raise TypeError("split_params only works on gpuarray shared variables")
        return _split_rnn_params(
            w, self.desc, layer, input_size, self.dtype, self.rnn_mode
        )

    def apply(self, w, x, hx, cx=None):
        """
        Apply the RNN to some data

        Parameters
        ----------
        w:
            opaque parameter block
        x:
            input
        hx:
            initial hidden state
        cx:
            initial cell state (for LSTM)
        """
        # Don't return the reserve as an output
        return GpuDnnRNNOp(self.rnn_mode, self.direction_mode)(
            rnndesc_type.make_constant(self.desc), w, x, hx, cx, return_list=True
        )[1:]


def dnn_batch_normalization_train(
    inputs,
    gamma,
    beta,
    mode="per-activation",
    epsilon=1e-4,
    running_average_factor=0.1,
    running_mean=None,
    running_var=None,
):
    """
    Performs batch normalization of the given inputs, using the mean and
    variance of the inputs.

    Parameters
    ----------
    mode : {'per-activation', 'spatial'}
        Whether to normalize per activation or share normalization factors
        across spatial dimensions (i.e., all dimensions past the second).
    gamma : tensor
        Learnable scale factors. Must match the dimensionality of `inputs`,
        but have sizes of `1` for all axes normalized over (i.e., in the first
        dimension for ``mode='per-activation'`, and additionally in all
        dimensions past the second for ``mode='spatial'``).
    beta : tensor
        Learnable biases. Must match the tensor layout of `gamma`.
    epsilon : float
        Epsilon value used in the batch normalization formula. Minimum allowed
        value is 1e-5 (imposed by cuDNN).
    running_average_factor : float
        Factor for updating the values or `running_mean` and `running_var`.
        If the factor is close to one, the running averages will update quickly,
        if the factor is close to zero it will update slowly.
    running_mean : tensor or None
        Previous value of the running mean. If this is given, the new value
        ``running_mean * (1 - r_a_factor) + batch mean * r_a_factor``
        will be returned as one of the outputs of this function.
        `running_mean` and `running_var` should either both be given or
        both be None.
    running_var : tensor or None
        Previous value of the running variance. If this is given, the new value
        ``running_var * (1 - r_a_factor) + (m / (m - 1)) * batch var * r_a_factor``
        will be returned as one of the outputs of this function,
        where `m` is the product of lengths of the averaged-over dimensions.
        `running_mean` and `running_var` should either both be given or
        both be None.

    Returns
    -------
    out : tensor
        Batch-normalized inputs.
    mean : tensor
        Means of `inputs` across the normalization axes.
    invstd : tensor
        Inverse standard deviations of `inputs` across the normalization axes.
    new_running_mean : tensor
        New value of the running mean (only if both `running_mean` and
        `running_var` were given).
    new_running_var : tensor
        New value of the running variance (only if both `running_var` and
        `running_mean` were given).

    Notes
    -----
    Requires cuDNN 5 and Aesara 0.9dev2 or more recent.

    For 4d tensors, returned values are equivalent to:

    .. code-block:: python

        axes = 0 if mode == 'per-activation' else (0, 2, 3)
        mean = inputs.mean(axes, keepdims=True)
        var = inputs.var(axes, keepdims=True)
        invstd = aet.inv(aet.sqrt(var + epsilon))
        out = (inputs - mean) * gamma * invstd + beta

        m = aet.cast(aet.prod(inputs.shape) / aet.prod(mean.shape), 'float32')
        running_mean = running_mean * (1 - running_average_factor) + \\
                       mean * running_average_factor
        running_var = running_var * (1 - running_average_factor) + \\
                      (m / (m - 1)) * var * running_average_factor

    For 5d tensors, the axes are (0, 2, 3, 4).
    """
    ndim = inputs.ndim
    if gamma.ndim != ndim or beta.ndim != ndim:
        raise ValueError(
            "gamma and beta must be of the same dimensionality "
            f"as inputs; got {int(gamma.ndim)} and {int(beta.ndim)} instead of {int(ndim)}"
        )
    if (running_mean is None) != (running_var is None):
        raise ValueError(
            "running_mean and running_var must either both be " "given or both be None"
        )
    if running_mean is not None and running_mean.ndim != ndim:
        raise ValueError(
            "running_mean must be of the same dimensionality "
            f"as inputs; got {int(running_mean.ndim)} instead of {int(ndim)}"
        )
    if running_var is not None and running_var.ndim != ndim:
        raise ValueError(
            "running_var must be of the same dimensionality "
            f"as inputs; got {int(running_var.ndim)} instead of {int(ndim)}"
        )
    if epsilon < 1e-5:
        raise ValueError(f"epsilon must be at least 1e-5, got {epsilon:f}")

    running_averages = running_mean is not None and running_var is not None

    if ndim < 4:
        inputs = shape_padright(inputs, 4 - ndim)
        gamma = shape_padright(gamma, 4 - ndim)
        beta = shape_padright(beta, 4 - ndim)
        if running_averages:
            running_mean = shape_padright(running_mean, 4 - ndim)
            running_var = shape_padright(running_var, 4 - ndim)
    elif ndim > 5:
        inputs_shape = inputs.shape
        params_shape = gamma.shape
        inputs = aet.flatten(inputs, 5)
        gamma = aet.flatten(gamma, 5)
        beta = aet.flatten(beta, 5)
        if running_averages:
            running_mean = aet.flatten(running_mean, 5)
            running_var = aet.flatten(running_var, 5)

    batchnorm_op = GpuDnnBatchNorm(mode=mode, running_averages=running_averages)
    if running_averages:
        out, mean, invstd, new_running_mean, new_running_var = batchnorm_op(
            gpu_contiguous(inputs),
            gpu_contiguous(gamma),
            gpu_contiguous(beta),
            epsilon=epsilon,
            running_average_factor=running_average_factor,
            running_mean=gpu_contiguous(running_mean),
            running_var=gpu_contiguous(running_var),
        )
        if new_running_mean.broadcastable != running_mean.broadcastable:
            new_running_mean = aet.patternbroadcast(
                new_running_mean, running_mean.broadcastable
            )
        if new_running_var.broadcastable != running_var.broadcastable:
            new_running_var = aet.patternbroadcast(
                new_running_var, running_var.broadcastable
            )
        result = (out, mean, invstd, new_running_mean, new_running_var)
    else:
        result = batchnorm_op(
            gpu_contiguous(inputs),
            gpu_contiguous(gamma),
            gpu_contiguous(beta),
            epsilon=epsilon,
        )
    if ndim < 4:
        result = tuple(aet.flatten(r, ndim) for r in result)
    elif ndim > 5:
        result = (reshape(result[0], inputs_shape),) + tuple(
            reshape(r, params_shape) for r in result[1:]
        )
    return result


def dnn_batch_normalization_test(
    inputs, gamma, beta, mean, var, mode="per-activation", epsilon=1e-4
):
    """
    Performs batch normalization of the given inputs, using the given mean and
    variance.

    Parameters
    ----------
    mode : {'per-activation', 'spatial'}
        Whether to normalize per activation or share normalization factors
        across spatial dimensions (i.e., all dimensions past the second).
    gamma : tensor
        Scale factors. Must match the dimensionality of `inputs`, but have
        sizes of `1` for all axes normalized over (i.e., in the first dimension
        for ``mode='per-activation'`, and additionally in all dimensions past
        the second for ``mode='spatial'``).
    beta : tensor
        Biases. Must match the tensor layout of `gamma`.
    mean : tensor
        Means. Usually these are running averages computed during training.
        Must match the tensor layout of `gamma`.
    var : tensor
        Variances. Usually these are running averages computed during training.
        Must match the tensor layout of `gamma`.
    epsilon : float
        Epsilon value used in the batch normalization formula. Minimum allowed
        value is 1e-5 (imposed by cuDNN).

    Returns
    -------
    out : tensor
        Batch-normalized inputs.

    Notes
    -----
    Requires cuDNN 5 and Aesara 0.9dev2 or more recent.

    For 4d tensors, the returned value is equivalent to:

    .. code-block:: python

        axes = (0,) if mode == 'per-activation' else (0, 2, 3)
        gamma, beta, mean, var = (aet.addbroadcast(t, *axes)
                                  for t in (gamma, beta, mean, var))
        out = (inputs - mean) * gamma / aet.sqrt(var + epsilon) + beta

    For 5d tensors, the axes would be (0, 2, 3, 4).
    """
    ndim = inputs.ndim
    if gamma.ndim != ndim or beta.ndim != ndim:
        raise ValueError(
            "gamma and beta must be of the same dimensionality "
            f"as inputs; got {int(gamma.ndim)} and {int(beta.ndim)} instead of {int(ndim)}"
        )
    if mean.ndim != ndim or var.ndim != ndim:
        raise ValueError(
            "mean and var must be of the same dimensionality "
            f"as inputs; got {int(mean.ndim)} and {int(var.ndim)} instead of {int(ndim)}"
        )
    if epsilon < 1e-5:
        raise ValueError(f"epsilon must be at least 1e-5, got {epsilon:f}")

    if ndim < 4:
        inputs = shape_padright(inputs, 4 - ndim)
        gamma = shape_padright(gamma, 4 - ndim)
        beta = shape_padright(beta, 4 - ndim)
        mean = shape_padright(mean, 4 - ndim)
        var = shape_padright(var, 4 - ndim)
    elif ndim > 5:
        inputs_shape = inputs.shape
        inputs = aet.flatten(inputs, 5)
        gamma = aet.flatten(gamma, 5)
        beta = aet.flatten(beta, 5)
        mean = aet.flatten(mean, 5)
        var = aet.flatten(var, 5)
    batchnorm_op = GpuDnnBatchNormInference(mode=mode)
    result = batchnorm_op(
        gpu_contiguous(inputs),
        gpu_contiguous(gamma),
        gpu_contiguous(beta),
        gpu_contiguous(mean),
        gpu_contiguous(var),
        epsilon=epsilon,
    )
    if ndim < 4:
        result = aet.flatten(result, ndim)
    elif ndim > 5:
        result = reshape(result, inputs_shape)
    return result


class GpuDnnTransformerGrid(DnnBase):
    """
    Grid generator Op for cuDNN Spatial Transformer.
    """

    __props__ = ()
    _cop_num_inputs = 2
    _cop_num_outputs = 1
    _f16_ok = True
    check_input = False

    def __init__(self):
        super().__init__(["c_code/dnn_sptf_grid.c"], "APPLY_SPECIFIC(dnn_sptf_grid)")

    def make_node(self, theta, out_dims):
        """
        Create a grid generator node for a cuDNN Spatial Transformer

        Parameters
        ----------
        theta : tensor
            Affine transformation tensor containing one affine transformation
            matrix per image. ``theta`` is usually generated by the localization
            network.

        out_dims : tuple
            Dimensions of the transformed inputs, containing four elements, and is given
            by (N, C, H, W), where N is the number of inputs, C the number of channels,
            H and W are the height and width of each input.
        """
        context_name = infer_context_name(theta)

        theta = gpu_contiguous(as_gpuarray_variable(theta, context_name))
        assert theta.dtype in ("float16", "float32", "float64")
        assert theta.ndim == 3

        out_dims = cpu_contiguous(as_tensor_variable(out_dims))
        assert out_dims.dtype in integer_dtypes
        assert out_dims.ndim == 1
        # Ensure 64-bit ints are passed to the C code
        out_dims = aet.cast(out_dims, "int64")
        grid = GpuArrayType(
            dtype=theta.dtype,
            broadcastable=(theta.type.ndim + 1) * (False,),
            context_name=context_name,
        )()

        inputs = [theta, out_dims]
        outputs = [grid]
        return Apply(self, inputs, outputs)

    def grad(self, inputs, grads):
        theta, out_dims = inputs
        dgrid = grads[0]

        dtheta = GpuDnnTransformerGradT()(dgrid)
        return [dtheta, grad_not_implemented(self, 1, out_dims)]


class GpuDnnTransformerSampler(DnnBase):
    """
    Grid sampler Op for cuDNN Spatial Transformer.
    """

    __props__ = ()
    _cop_num_inputs = 2
    _cop_num_outputs = 1
    _f16_ok = True
    check_input = False

    def __init__(self):
        super().__init__(
            ["c_code/dnn_sptf_sampler.c"], "APPLY_SPECIFIC(dnn_sptf_sampler)"
        )

    def make_node(self, img, grid):
        """
        Create a grid sampler node for a cuDNN Spatial Transformer

        Parameters
        ----------
        img : tensor
            Images from which the pixels will be sampled. The implementation
            assumes the tensor is in NCHW format, where N is the number of images,
            C is the number of color channels, H is the height of the inputs, and
            W is width of the inputs.

        grid : GpuDnnTransformerGrid
            Grid that contains the coordinates of the pixels to be sampled from
            the inputs images.
        """
        context_name = infer_context_name(img, grid)

        img = gpu_contiguous(as_gpuarray_variable(img, context_name))
        if img.type.ndim != 4:
            raise TypeError("img must be a 4D tensor")
        elif img.dtype not in ("float16", "float32", "float64"):
            raise TypeError("img type must be floating-point")

        grid = gpu_contiguous(as_gpuarray_variable(grid, context_name))
        if grid.type.ndim != 4:
            raise TypeError("grid must be a 4D tensor")
        elif grid.dtype not in ("float16", "float32", "float64"):
            raise TypeError("grid type must be floating-point")

        out = GpuArrayType(
            dtype=img.dtype,
            broadcastable=img.type.ndim * (False,),
            context_name=context_name,
        )()

        inputs = [img, grid]
        outputs = [out]
        return Apply(self, inputs, outputs)

    def grad(self, inputs, grads):
        img, grid = inputs
        dy = grads[0]

        dimg, dgrid = GpuDnnTransformerGradI()(img, grid, dy)
        return [dimg, dgrid]


class GpuDnnTransformerGradI(DnnBase):
    """
    Gradient of inputs Op for cuDNN Spatial Transformer.
    """

    __props__ = ()
    _cop_num_inputs = 3
    _cop_num_outputs = 2
    _f16_ok = True
    check_input = False

    def __init__(self):
        super().__init__(["c_code/dnn_sptf_gi.c"], "APPLY_SPECIFIC(dnn_sptf_gi)")

    def make_node(self, img, grid, dy):
        context_name = infer_context_name(img, grid, dy)

        img = as_gpuarray_variable(gpu_contiguous(img), context_name)
        if img.ndim != 4:
            raise TypeError("img must have 4 dimensions.")

        grid = as_gpuarray_variable(gpu_contiguous(grid), context_name)
        if img.ndim != grid.ndim:
            raise TypeError("grid should have the same number of dimensions as img")

        dy = as_gpuarray_variable(dy, context_name)
        if dy.ndim != 4:
            raise TypeError("dy must have 4 dimensions.")

        dimg = img.type()
        dgrid = grid.type()

        inputs = [img, grid, dy]
        outputs = [dimg, dgrid]

        return Apply(self, inputs, outputs)


class GpuDnnTransformerGradT(DnnBase):
    """
    Gradient of affine transformations Op for cuDNN Spatial Transformer.
    """

    __props__ = ()
    _cop_num_inputs = 1
    _cop_num_outputs = 1
    _f16_ok = True
    check_input = False

    def __init__(self):
        super().__init__(["c_code/dnn_sptf_gt.c"], "APPLY_SPECIFIC(dnn_sptf_gt)")

    def make_node(self, dgrid):
        context_name = infer_context_name(dgrid)

        dgrid = as_gpuarray_variable(dgrid, context_name)
        assert dgrid.dtype in ("float16", "float32", "float64")
        assert dgrid.ndim == 4

        dtheta = GpuArrayType(
            dtype=dgrid.dtype,
            broadcastable=(dgrid.type.ndim - 1) * (False,),
            context_name=context_name,
        )()
        inputs = [dgrid]
        outputs = [dtheta]

        return Apply(self, inputs, outputs)


def dnn_spatialtf(img, theta, scale_width=1, scale_height=1):
    """
    GPU spatial transformer using cuDNN from NVIDIA.

    Parameters
    ----------
    img : tensor
        Images to which the transformations will be applied. The implementation
        assumes the tensor is in NCHW format, where N is the number of images,
        C is the number of color channels, H is the height of the inputs, and
        W is width of the inputs.
    theta : tensor
        Affine transformation tensor containing one affine transformation
        matrix per image. ``theta`` is usually generated by the localization
        network.
    scale_height: float
        A float specifying the scaling factor for the height of the output
        image. A value of 1 will keep the original height of the input. Values
        larger than 1 will upsample the input. Values below 1 will downsample
        the input.
    scale_width: float
        A float specifying the scaling factor for the width of the output
        image. A value of 1 will keep the original width of the input. Values
        larger than 1 will upsample the input. Values below 1 will downsample
        the input.

    Returns
    -------
    out : tensor
        Transformed images with width and height properly scaled.

    Notes
    -----
    Currently, cuDNN only supports 2D transformations with 2x3 affine
    transformation matrices.

    Bilinear interpolation is the only grid sampler method available.
    """
    out_dims = (
        img.shape[0],
        img.shape[1],
        tm.ceil(img.shape[2] * scale_height),
        tm.ceil(img.shape[3] * scale_width),
    )
    out_dims = tuple([as_scalar(v).astype("int64") for v in out_dims])
    # Setup spatial transformer
    grid = GpuDnnTransformerGrid()(theta, out_dims)
    sampler = GpuDnnTransformerSampler()(img, grid)
    return sampler


def local_abstractconv_cudnn_graph(fgraph, op, context_name, inputs, outputs):
    if not isinstance(
        op, (AbstractConv2d, AbstractConv2d_gradWeights, AbstractConv2d_gradInputs)
    ):
        return

    if version(raises=False) < 6000 and op.filter_dilation != (1, 1):
        return None

    if op.unshared:
        return None

    if isinstance(op.border_mode, tuple) and any(
        isinstance(p, tuple) for p in op.border_mode
    ):
        # Asymmetric padding not yet supported
        return None

    inp1 = inputs[0]
    inp2 = inputs[1]

    if not dnn_available(inp1.type.context_name):
        return

    if op.filter_flip:
        conv_mode = "conv"
    else:
        conv_mode = "cross"

    if isinstance(op, AbstractConv2d):
        rval = dnn_conv(
            fgraph,
            inp1,
            inp2,
            border_mode=op.border_mode,
            subsample=op.subsample,
            dilation=op.filter_dilation,
            direction_hint="forward!",
            conv_mode=conv_mode,
            num_groups=op.num_groups,
        )
    elif isinstance(op, AbstractConv2d_gradWeights):
        shape = (
            inp2.shape[1],
            inp1.shape[1] // op.num_groups,
            inputs[2][0],
            inputs[2][1],
        )
        rval = dnn_gradweight(
            inp1,
            inp2,
            shape,
            border_mode=op.border_mode,
            subsample=op.subsample,
            dilation=op.filter_dilation,
            conv_mode=conv_mode,
            num_groups=op.num_groups,
        )
    elif isinstance(op, AbstractConv2d_gradInputs):
        shape = (
            inp2.shape[0],
            inp1.shape[1] * op.num_groups,
            inputs[2][0],
            inputs[2][1],
        )
        rval = dnn_gradinput(
            inp1,
            inp2,
            shape,
            border_mode=op.border_mode,
            subsample=op.subsample,
            dilation=op.filter_dilation,
            conv_mode=conv_mode,
            num_groups=op.num_groups,
        )
    return [rval]


def local_abstractconv3d_cudnn_graph(fgraph, op, context_name, inputs, outputs):
    if not isinstance(
        op, (AbstractConv3d, AbstractConv3d_gradWeights, AbstractConv3d_gradInputs)
    ):
        return

    if version(raises=False) < 6000 and op.filter_dilation != (1, 1, 1):
        return None

    inp1 = inputs[0]
    inp2 = inputs[1]

    if not dnn_available(inp1.type.context_name):
        return

    if op.filter_flip:
        conv_mode = "conv"
    else:
        conv_mode = "cross"

    if isinstance(op, AbstractConv3d):
        rval = dnn_conv3d(
            fgraph,
            inp1,
            inp2,
            border_mode=op.border_mode,
            subsample=op.subsample,
            dilation=op.filter_dilation,
            direction_hint="forward!",
            conv_mode=conv_mode,
            num_groups=op.num_groups,
        )
    elif isinstance(op, AbstractConv3d_gradWeights):
        shape = (
            inp2.shape[1],
            inp1.shape[1] // op.num_groups,
            inputs[2][0],
            inputs[2][1],
            inputs[2][2],
        )
        rval = dnn_gradweight3d(
            inp1,
            inp2,
            shape,
            border_mode=op.border_mode,
            subsample=op.subsample,
            dilation=op.filter_dilation,
            conv_mode=conv_mode,
            num_groups=op.num_groups,
        )
    elif isinstance(op, AbstractConv3d_gradInputs):
        shape = (
            inp2.shape[0],
            inp1.shape[1] * op.num_groups,
            inputs[2][0],
            inputs[2][1],
            inputs[2][2],
        )
        rval = dnn_gradinput3d(
            inp1,
            inp2,
            shape,
            border_mode=op.border_mode,
            subsample=op.subsample,
            dilation=op.filter_dilation,
            conv_mode=conv_mode,
            num_groups=op.num_groups,
        )
    return [rval]


def local_abstract_batch_norm_train_cudnn(fgraph, op, ctx_name, inputs, outputs):
    x, scale, bias, epsilon, running_average_factor = inputs[:5]
    running_mean = inputs[5] if len(inputs) > 5 else None
    running_var = inputs[6] if len(inputs) > 6 else None

    # convert axes to cuDNN mode
    axes = tuple(op.axes)
    if axes == (0,):
        mode = "per-activation"
    elif axes == (0,) + tuple(range(2, x.ndim)):
        mode = "spatial"
    else:
        return None

    try:
        eps = aet.get_scalar_constant_value(epsilon)
    except NotScalarConstantError:
        return None
    if eps < 1e-5:
        return None
    try:
        running_average_factor = aet.get_scalar_constant_value(running_average_factor)
    except NotScalarConstantError:
        return None

    ctx = infer_context_name(*inputs)
    if not dnn_available(ctx):
        return
    x = as_gpuarray_variable(x, context_name=ctx)
    scale = as_gpuarray_variable(scale, context_name=ctx)
    bias = as_gpuarray_variable(bias, context_name=ctx)

    inputs = [x, scale, bias, mode, eps, running_average_factor]
    if running_mean is not None and running_var is not None:
        inputs.append(running_mean)
        inputs.append(running_var)

    results = list(dnn_batch_normalization_train(*inputs))

    return results


def local_abstract_batch_norm_train_grad_cudnn(fgraph, op, ctx_name, inputs, outputs):
    x, dy, scale, x_mean, x_invstd, epsilon = inputs

    # input on gpu?  TODO what about the output?
    x_on_gpu = isinstance(x.type, GpuArrayType) or (
        x.owner and isinstance(x.owner.op, HostFromGpu)
    )
    dy_on_gpu = isinstance(dy.type, GpuArrayType) or (
        dy.owner and isinstance(dy.owner.op, HostFromGpu)
    )
    if not (x_on_gpu or dy_on_gpu):
        return None

    # convert axes to cuDNN mode
    axes = tuple(op.axes)
    if axes == (0,):
        mode = "per-activation"
    elif axes == (0,) + tuple(range(2, x.ndim)):
        mode = "spatial"
    else:
        return None

    ndim = x.ndim
    if ndim < 4:
        x = shape_padright(x, 4 - ndim)
        dy = shape_padright(dy, 4 - ndim)
        scale = shape_padright(scale, 4 - ndim)
        x_mean = shape_padright(x_mean, 4 - ndim)
        x_invstd = shape_padright(x_invstd, 4 - ndim)
    elif ndim > 5:
        x_shape = x.shape
        params_shape = scale.shape
        x = aet.flatten(x, 5)
        dy = aet.flatten(dy, 5)
        scale = aet.flatten(scale, 5)
        x_mean = aet.flatten(x_mean, 5)
        x_invstd = aet.flatten(x_invstd, 5)

    try:
        eps = aet.get_scalar_constant_value(epsilon)
    except NotScalarConstantError:
        return None
    if eps < 1e-5:
        return None

    ctx = infer_context_name(*inputs)
    if not dnn_available(ctx):
        return
    x = as_gpuarray_variable(x, context_name=ctx)
    dy = as_gpuarray_variable(dy, context_name=ctx)
    scale = as_gpuarray_variable(scale, context_name=ctx)
    x_mean = as_gpuarray_variable(x_mean, context_name=ctx)
    x_invstd = as_gpuarray_variable(x_invstd, context_name=ctx)

    g_wrt_inputs, g_wrt_scale, g_wrt_bias = GpuDnnBatchNormGrad(mode)(
        x, dy, scale, x_mean, x_invstd, eps
    )

    if ndim < 4:
        g_wrt_inputs = aet.flatten(g_wrt_inputs, ndim)
        g_wrt_scale = aet.flatten(g_wrt_scale, ndim)
        g_wrt_bias = aet.flatten(g_wrt_bias, ndim)
    elif ndim > 5:
        g_wrt_inputs = reshape(g_wrt_inputs, x_shape)
        g_wrt_scale = reshape(g_wrt_scale, params_shape)
        g_wrt_bias = reshape(g_wrt_bias, params_shape)

    return [g_wrt_inputs, g_wrt_scale, g_wrt_bias]


def local_abstract_batch_norm_inference_cudnn(fgraph, op, ctx_name, inputs, outputs):
    x, scale, bias, estimated_mean, estimated_variance, epsilon = inputs

    axes = tuple(op.axes)
    if axes == (0,):
        mode = "per-activation"
    elif axes == (0,) + tuple(range(2, x.ndim)):
        mode = "spatial"
    else:
        return None

    try:
        eps = aet.get_scalar_constant_value(epsilon)
    except NotScalarConstantError:
        return None
    if eps < 1e-5:
        return None

    ctx = infer_context_name(*inputs)
    if not dnn_available(ctx):
        return
    x = as_gpuarray_variable(x, context_name=ctx)
    scale = as_gpuarray_variable(scale, context_name=ctx)
    bias = as_gpuarray_variable(bias, context_name=ctx)
    estimated_mean = as_gpuarray_variable(estimated_mean, context_name=ctx)
    estimated_variance = as_gpuarray_variable(estimated_variance, context_name=ctx)

    out = dnn_batch_normalization_test(
        x, scale, bias, estimated_mean, estimated_variance, mode, eps
    )

    return [out]
