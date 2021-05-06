import logging

import numpy as np

import aesara
from aesara import scalar as aes
from aesara.configdefaults import config
from aesara.graph.basic import Variable
from aesara.graph.type import CType
from aesara.graph.utils import MetaType
from aesara.misc.safe_asarray import _asarray
from aesara.utils import apply_across_args


_logger = logging.getLogger("aesara.tensor.type")


# Define common subsets of dtypes (as strings).
complex_dtypes = list(map(str, aes.complex_types))
continuous_dtypes = list(map(str, aes.continuous_types))
float_dtypes = list(map(str, aes.float_types))
integer_dtypes = list(map(str, aes.integer_types))
discrete_dtypes = list(map(str, aes.discrete_types))
all_dtypes = list(map(str, aes.all_types))
int_dtypes = list(map(str, aes.int_types))
uint_dtypes = list(map(str, aes.uint_types))

# TODO: add more type correspondances for e.g. int32, int64, float32,
# complex64, etc.
dtype_specs_map = {
    "float16": (float, "npy_float16", "NPY_FLOAT16"),
    "float32": (float, "npy_float32", "NPY_FLOAT32"),
    "float64": (float, "npy_float64", "NPY_FLOAT64"),
    "bool": (bool, "npy_bool", "NPY_BOOL"),
    "uint8": (int, "npy_uint8", "NPY_UINT8"),
    "int8": (int, "npy_int8", "NPY_INT8"),
    "uint16": (int, "npy_uint16", "NPY_UINT16"),
    "int16": (int, "npy_int16", "NPY_INT16"),
    "uint32": (int, "npy_uint32", "NPY_UINT32"),
    "int32": (int, "npy_int32", "NPY_INT32"),
    "uint64": (int, "npy_uint64", "NPY_UINT64"),
    "int64": (int, "npy_int64", "NPY_INT64"),
    "complex128": (complex, "aesara_complex128", "NPY_COMPLEX128"),
    "complex64": (complex, "aesara_complex64", "NPY_COMPLEX64"),
}


class TensorType(CType):
    """
    Symbolic `Type` representing a numpy.ndarray value.

    Initialize self.dtype and self.broadcastable.

    Parameters
    ----------
    dtype: str
        Corresponding to numpy dtype (e.g., 'int64')
        The value (ndarray) associated to a `Variable` of this `Type` will
        have this dtype.
    broadcastable: tuple, list, or array of boolean values
        This argument serves two purposes. First, the True elements of this
        list indicate the dimensions where the shape of an associated value
        must be 1. Secondly, the length of this list is the number of
        dimensions that an associated value must have. See
        doc:`broadcasting` for an explanation of how this list is used.
    name : str
        Optional name for this type.

    """

    dtype_specs_map = dtype_specs_map
    context_name = "cpu"
    filter_checks_isfinite = False
    """
    When this is True, strict filtering rejects data containing NaN or
    Inf entries. (Used in `DebugMode`)
    """

    def __init__(self, dtype, broadcastable, name=None):
        self.dtype = str(dtype)
        if self.dtype == "floatX":
            self.dtype = config.floatX
        # broadcastable is immutable, and all elements are either
        # True or False
        self.broadcastable = tuple(bool(b) for b in broadcastable)
        self.dtype_specs()  # error checking is done there
        self.name = name
        self.numpy_dtype = np.dtype(self.dtype)

    def clone(self, dtype=None, broadcastable=None):
        """
        Return a copy of the type optionally with a new dtype or
        broadcastable pattern.

        """
        if dtype is None:
            dtype = self.dtype
        if broadcastable is None:
            broadcastable = self.broadcastable
        return self.__class__(dtype, broadcastable, name=self.name)

    def filter(self, data, strict=False, allow_downcast=None):
        """
        Convert `data` to something which can be associated to a
        `TensorVariable`.

        This function is not meant to be called in user code. It is for
        `Linker` instances to use when running a compiled graph.

        """
        # Explicit error message when one accidentally uses a Variable as
        # input (typical mistake, especially with shared variables).
        if isinstance(data, Variable):
            raise TypeError(
                "Expected an array-like object, but found a Variable: "
                "maybe you are trying to call a function on a (possibly "
                "shared) variable instead of a numeric array?"
            )

        if isinstance(data, np.memmap) and (data.dtype == self.numpy_dtype):
            # numpy.memmap is a "safe" subclass of ndarray,
            # so we can use it wherever we expect a base ndarray.
            # however, casting it would defeat the purpose of not
            # loading the whole data into memory
            pass
        elif isinstance(data, np.ndarray) and (data.dtype == self.numpy_dtype):
            if data.dtype.num != self.numpy_dtype.num:
                data = _asarray(data, dtype=self.dtype)
            # -- now fall through to ndim check
        elif strict:
            # If any of the two conditions above was not met,
            # we raise a meaningful TypeError.
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{self} expected a ndarray object (got {type(data)}).")
            if data.dtype != self.numpy_dtype:
                raise TypeError(
                    f"{self} expected an ndarray with dtype={self.numpy_dtype} (got {data.dtype})."
                )
        else:
            if allow_downcast:
                # Convert to self.dtype, regardless of the type of data
                data = _asarray(data, dtype=self.dtype)
                # TODO: consider to pad shape with ones to make it consistent
                # with self.broadcastable... like vector->row type thing
            else:
                if isinstance(data, np.ndarray):
                    # Check if self.dtype can accurately represent data
                    # (do not try to convert the data)
                    up_dtype = aes.upcast(self.dtype, data.dtype)
                    if up_dtype == self.dtype:
                        # Bug in the following line when data is a
                        # scalar array, see
                        # http://projects.scipy.org/numpy/ticket/1611
                        # data = data.astype(self.dtype)
                        data = _asarray(data, dtype=self.dtype)
                    if up_dtype != self.dtype:
                        err_msg = (
                            f"{self} cannot store a value of dtype {data.dtype} without "
                            "risking loss of precision. If you do not mind "
                            "this loss, you can: "
                            f"1) explicitly cast your data to {self.dtype}, or "
                            '2) set "allow_input_downcast=True" when calling '
                            f'"function". Value: "{repr(data)}"'
                        )
                        raise TypeError(err_msg)
                elif (
                    allow_downcast is None
                    and isinstance(data, (float, np.floating))
                    and self.dtype == config.floatX
                ):
                    # Special case where we allow downcasting of Python float
                    # literals to floatX, even when floatX=='float32'
                    data = _asarray(data, self.dtype)
                else:
                    # data has to be converted.
                    # Check that this conversion is lossless
                    converted_data = _asarray(data, self.dtype)
                    # We use the `values_eq` static function from TensorType
                    # to handle NaN values.
                    if TensorType.values_eq(
                        np.asarray(data), converted_data, force_same_dtype=False
                    ):
                        data = converted_data
                    else:
                        # Do not print a too long description of data
                        # (ndarray truncates it, but it's not sure for data)
                        str_data = str(data)
                        if len(str_data) > 80:
                            str_data = str_data[:75] + "(...)"

                        err_msg = (
                            f"{self} cannot store accurately value {data}, "
                            f"it would be represented as {converted_data}. "
                            "If you do not mind this precision loss, you can: "
                            "1) explicitly convert your data to a numpy array "
                            f"of dtype {self.dtype}, or "
                            '2) set "allow_input_downcast=True" when calling '
                            '"function".'
                        )
                        raise TypeError(err_msg)

        if self.ndim != data.ndim:
            raise TypeError(
                f"Wrong number of dimensions: expected {self.ndim},"
                f" got {data.ndim} with shape {data.shape}."
            )
        if not data.flags.aligned:
            try:
                msg = "object buffer" + str(data.data)
            except AttributeError:
                msg = ""
            raise TypeError(
                "The numpy.ndarray object is not aligned."
                " Aesara C code does not support that.",
                msg,
                "object shape",
                data.shape,
                "object strides",
                data.strides,
                "object dtype",
                data.dtype,
            )

        i = 0
        for b in self.broadcastable:
            if b and data.shape[i] != 1:
                raise TypeError(
                    "Non-unit value on shape on a broadcastable" " dimension.",
                    data.shape,
                    self.broadcastable,
                )
            i += 1
        if self.filter_checks_isfinite and not np.all(np.isfinite(data)):
            raise ValueError("non-finite elements not allowed")
        return data

    def filter_variable(self, other, allow_convert=True):
        """
        Convert a symbolic Variable into a TensorType, if compatible.

        For the moment, only a TensorType and GpuArrayType will be
        converted, provided they have the same number of dimensions
        and dtype and have "compatible" broadcastable pattern.

        """
        if not isinstance(other, Variable):
            # The value is not a Variable: we cast it into
            # a Constant of the appropriate Type.
            other = self.Constant(type=self, data=other)

        if other.type == self:
            return other

        if allow_convert:
            # Attempt safe broadcast conversion.
            other2 = self.convert_variable(other)
            if other2 is not None and other2.type == self:
                return other2

        raise TypeError(
            f"Cannot convert Type {other.type} "
            f"(of Variable {other}) into Type {self}. "
            f"You can try to manually convert {other} into a {self}."
        )

    def value_validity_msg(self, a):
        try:
            self.filter(a, strict=True)
        except Exception as e:
            return str(e)
        return "value is valid"

    def dtype_specs(self):
        """
        Return a tuple (python type, c type, numpy typenum) that corresponds
        to self.dtype.

        This function is used internally as part of C code generation.

        """
        try:
            return self.dtype_specs_map[self.dtype]
        except KeyError:
            raise TypeError(
                f"Unsupported dtype for {self.__class__.__name__}: {self.dtype}"
            )

    def to_scalar_type(self):
        return aes.get_scalar_type(dtype=self.dtype)

    def __eq__(self, other):
        """
        Compare True iff other is the same kind of TensorType.

        """
        return (
            type(self) == type(other)
            and other.dtype == self.dtype
            and other.broadcastable == self.broadcastable
        )

    def convert_variable(self, var):
        if (
            isinstance(self, type(var.type))
            and self.dtype == var.type.dtype  # noqa
            and self.ndim == var.type.ndim
            and all(
                sb == ob or ob
                for sb, ob in zip(self.broadcastable, var.type.broadcastable)
            )
        ):
            return aesara.tensor.patternbroadcast(var, self.broadcastable)

    @staticmethod
    def may_share_memory(a, b):
        # This is a method of TensorType, so both a and b should be ndarrays
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return np.may_share_memory(a, b)
        else:
            return False

    @staticmethod
    def values_eq(a, b, force_same_dtype=True):
        # TODO: check to see if the shapes must match
        #      for now, we err on safe side...
        if a.shape != b.shape:
            return False
        if force_same_dtype and a.dtype != b.dtype:
            return False
        a_eq_b = a == b
        r = np.all(a_eq_b)
        if r:
            return True
        # maybe the trouble is that there are NaNs
        a_missing = np.isnan(a)
        if a_missing.any():
            b_missing = np.isnan(b)
            return np.all(a_eq_b + (a_missing == b_missing))
        else:
            return False

    @staticmethod
    def values_eq_approx(
        a, b, allow_remove_inf=False, allow_remove_nan=False, rtol=None, atol=None
    ):
        return values_eq_approx(a, b, allow_remove_inf, allow_remove_nan, rtol, atol)

    def __hash__(self):
        return hash((type(self), self.dtype, self.broadcastable))

    ndim = property(lambda self: len(self.broadcastable), doc="number of dimensions")
    """
    Number of dimensions.

    This read-only property is the preferred way to get the number of
    dimensions of a `TensorType`.

    """

    def make_variable(self, name=None):
        """
        Return a `TensorVariable` of this type.

        Parameters
        ----------
        name : str
            A pretty name to identify this `Variable` when printing and
            debugging

        """
        return self.Variable(self, name=name)

    def __str__(self):
        if self.name:
            return self.name
        else:
            b = self.broadcastable
            named_broadcastable = {
                (): "scalar",
                (False,): "vector",
                (False, True): "col",
                (True, False): "row",
                (False, False): "matrix",
            }
            if b in named_broadcastable:
                bcast = named_broadcastable[b]
            else:
                if any(b):
                    bcast = str(b)
                else:
                    bcast = f"{len(b)}D"
            return f"TensorType({self.dtype}, {bcast})"

    def __repr__(self):
        return str(self)

    def c_element_type(self):
        return self.dtype_specs()[1]

    def c_declare(self, name, sub, check_input=True):
        if check_input:
            check = """
            typedef %(dtype)s dtype_%(name)s;
            """ % dict(
                sub, name=name, dtype=self.dtype_specs()[1]
            )
        else:
            check = ""
        declaration = """
        PyArrayObject* %(name)s;
        """ % dict(
            sub, name=name, dtype=self.dtype_specs()[1]
        )

        return declaration + check

    def c_init(self, name, sub):
        return """
        %(name)s = NULL;
        """ % dict(
            sub, name=name, type_num=self.dtype_specs()[2]
        )

    def c_extract(self, name, sub, check_input=True, **kwargs):
        if check_input:
            check = """
            %(name)s = NULL;
            if (py_%(name)s == Py_None) {
                // We can either fail here or set %(name)s to NULL and rely on Ops
                // using tensors to handle the NULL case, but if they fail to do so
                // they'll end up with nasty segfaults, so this is public service.
                PyErr_SetString(PyExc_ValueError, "expected an ndarray, not None");
                %(fail)s
            }
            if (!PyArray_Check(py_%(name)s)) {
                PyErr_SetString(PyExc_ValueError, "expected an ndarray");
                %(fail)s
            }
            // We expect %(type_num)s
            if (!PyArray_ISALIGNED((PyArrayObject*) py_%(name)s)) {
                PyArrayObject * tmp = (PyArrayObject*) py_%(name)s;
                PyErr_Format(PyExc_NotImplementedError,
                             "expected an aligned array of type %%ld "
                             "(%(type_num)s), got non-aligned array of type %%ld"
                             " with %%ld dimensions, with 3 last dims "
                             "%%ld, %%ld, %%ld"
                             " and 3 last strides %%ld %%ld, %%ld.",
                             (long int) %(type_num)s,
                             (long int) PyArray_TYPE((PyArrayObject*) py_%(name)s),
                             (long int) PyArray_NDIM(tmp),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_DIMS(tmp)[PyArray_NDIM(tmp)-1] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 3 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-3] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 2 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-2] : -1),
                             (long int) (PyArray_NDIM(tmp) >= 1 ?
            PyArray_STRIDES(tmp)[PyArray_NDIM(tmp)-1] : -1)
            );
                %(fail)s
            }
            // This is a TypeError to be consistent with DEBUG_MODE
            // Note: DEBUG_MODE also tells the name of the container
            if (PyArray_TYPE((PyArrayObject*) py_%(name)s) != %(type_num)s) {
                PyErr_Format(PyExc_TypeError,
                             "expected type_num %%d (%(type_num)s) got %%d",
                             %(type_num)s, PyArray_TYPE((PyArrayObject*) py_%(name)s));
                %(fail)s
            }
            """ % dict(
                sub, name=name, type_num=self.dtype_specs()[2]
            )
        else:
            check = ""
        return (
            check
            + """
        %(name)s = (PyArrayObject*)(py_%(name)s);
        Py_XINCREF(%(name)s);
        """
            % dict(sub, name=name, type_num=self.dtype_specs()[2])
        )

    def c_cleanup(self, name, sub):
        return (
            """
        if (%(name)s) {
            Py_XDECREF(%(name)s);
        }
        """
            % locals()
        )

    def c_sync(self, name, sub):
        fail = sub["fail"]
        type_num = self.dtype_specs()[2]
        return (
            """
        {Py_XDECREF(py_%(name)s);}
        if (!%(name)s) {
            Py_INCREF(Py_None);
            py_%(name)s = Py_None;
        }
        else if ((void*)py_%(name)s != (void*)%(name)s) {
            py_%(name)s = (PyObject*)%(name)s;
        }

        {Py_XINCREF(py_%(name)s);}

        if (%(name)s && !PyArray_ISALIGNED((PyArrayObject*) py_%(name)s)) {
            PyErr_Format(PyExc_NotImplementedError,
                         "c_sync: expected an aligned array, got non-aligned array of type %%ld"
                         " with %%ld dimensions, with 3 last dims "
                         "%%ld, %%ld, %%ld"
                         " and 3 last strides %%ld %%ld, %%ld.",
                         (long int) PyArray_TYPE((PyArrayObject*) py_%(name)s),
                         (long int) PyArray_NDIM(%(name)s),
                         (long int) (PyArray_NDIM(%(name)s) >= 3 ?
        PyArray_DIMS(%(name)s)[PyArray_NDIM(%(name)s)-3] : -1),
                         (long int) (PyArray_NDIM(%(name)s) >= 2 ?
        PyArray_DIMS(%(name)s)[PyArray_NDIM(%(name)s)-2] : -1),
                         (long int) (PyArray_NDIM(%(name)s) >= 1 ?
        PyArray_DIMS(%(name)s)[PyArray_NDIM(%(name)s)-1] : -1),
                         (long int) (PyArray_NDIM(%(name)s) >= 3 ?
        PyArray_STRIDES(%(name)s)[PyArray_NDIM(%(name)s)-3] : -1),
                         (long int) (PyArray_NDIM(%(name)s) >= 2 ?
        PyArray_STRIDES(%(name)s)[PyArray_NDIM(%(name)s)-2] : -1),
                         (long int) (PyArray_NDIM(%(name)s) >= 1 ?
        PyArray_STRIDES(%(name)s)[PyArray_NDIM(%(name)s)-1] : -1)
        );
            %(fail)s
        }
        """
            % locals()
        )

    def c_headers(self, **kwargs):
        return aes.get_scalar_type(self.dtype).c_headers(**kwargs)

    def c_libraries(self, **kwargs):
        return aes.get_scalar_type(self.dtype).c_libraries(**kwargs)

    def c_compile_args(self, **kwargs):
        return aes.get_scalar_type(self.dtype).c_compile_args(**kwargs)

    def c_support_code(self, **kwargs):
        return aes.get_scalar_type(self.dtype).c_support_code(**kwargs)

    def c_init_code(self, **kwargs):
        return aes.get_scalar_type(self.dtype).c_init_code(**kwargs)

    def c_code_cache_version(self):
        scalar_version = aes.get_scalar_type(self.dtype).c_code_cache_version()
        if scalar_version:
            return (11,) + scalar_version
        else:
            return ()

    def value_zeros(self, shape):
        """Create an numpy ndarray full of 0 values."""
        return np.zeros(shape, dtype=self.dtype)

    def get_shape_info(self, obj):
        """Return the information needed to compute the memory size of ``obj``.

        The memory size is only the data, so this excludes the container.
        For an ndarray, this is the data, but not the ndarray object and
        other data structures such as shape and strides.

        ``get_shape_info()`` and ``get_size()`` work in tandem for the memory
        profiler.

        ``get_shape_info()`` is called during the execution of the function.
        So it is better that it is not too slow.

        ``get_size()`` will be called on the output of this function
        when printing the memory profile.

        Parameters
        ----------
        obj
            The object that this Type represents during execution.

        Returns
        -------
        object
            Python object that ``self.get_size()`` understands.

        """
        return obj.shape

    def get_size(self, shape_info):
        """Number of bytes taken by the object represented by `shape_info`.

        Parameters
        ----------
        shape_info
            The output of the call to get_shape_info().

        Returns
        -------
        int
            The number of bytes taken by the object described by ``shape_info``.

        """
        if shape_info:
            return np.prod(shape_info) * np.dtype(self.dtype).itemsize
        else:  # a scalar
            return np.dtype(self.dtype).itemsize


aesara.compile.ops.expandable_types += (TensorType,)


class DenseTypeMeta(MetaType):
    def __instancecheck__(self, o):
        if type(o) == TensorType or isinstance(o, DenseTypeMeta):
            return True
        return False


class DenseTensorType(TensorType, metaclass=DenseTypeMeta):
    """A `Type` for dense tensors."""


def values_eq_approx(
    a, b, allow_remove_inf=False, allow_remove_nan=False, rtol=None, atol=None
):
    """
    Parameters
    ----------
    allow_remove_inf
        If True, when there is an inf in a, we allow any value in b in
        that position. Event -inf
    allow_remove_nan
        If True, when there is a nan in a, we allow any value in b in
        that position. Event +-inf
    rtol
        Relative tolerance, passed to _allclose.
    atol
        Absolute tolerance, passed to _allclose.

    """
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.shape != b.shape:
            return False
        if a.dtype != b.dtype:
            return False
        if str(a.dtype) not in continuous_dtypes:
            return np.all(a == b)
        else:
            cmp = aesara.tensor.math._allclose(a, b, rtol=rtol, atol=atol)
            if cmp:
                # Numpy claims they are close, this is good enough for us.
                return True
            # Numpy is unhappy, but it does not necessarily mean that a and
            # b are different. Indeed, Numpy does not like missing values
            # and will return False whenever some are found in a or b.
            # The proper way would be to use the MaskArray stuff available
            # in Numpy. However, it looks like it has been added to Numpy's
            # core recently, so it may not be available to everyone. Thus,
            # for now we use a home-made recipe, that should probably be
            # revisited in the future.
            a_missing = np.isnan(a)
            a_inf = np.isinf(a)

            if not (a_missing.any() or (allow_remove_inf and a_inf.any())):
                # There are no missing values in a, thus this is not the
                # reason why numpy.allclose(a, b) returned False.
                _logger.info(
                    f"numpy allclose failed for abs_err {np.max(abs(a - b))} and rel_err {np.max(abs(a - b) / (abs(a) + abs(b)))}"
                )
                return False
            # The following line is what numpy.allclose bases its decision
            # upon, according to its documentation.
            rtol = 1.0000000000000001e-05
            atol = 1e-8
            cmp_elemwise = np.absolute(a - b) <= (atol + rtol * np.absolute(b))
            # Find places where both a and b have missing values.
            both_missing = a_missing * np.isnan(b)

            # Find places where both a and b have inf of the same sign.
            both_inf = a_inf * np.isinf(b)

            # cmp_elemwise is weird when we have inf and -inf.
            # set it to False
            cmp_elemwise = np.where(both_inf & cmp_elemwise, a == b, cmp_elemwise)

            # check the sign of the inf
            both_inf = np.where(both_inf, (a == b), both_inf)

            if allow_remove_inf:
                both_inf += a_inf
            if allow_remove_nan:
                both_missing += a_missing

            # Combine all information.
            return (cmp_elemwise + both_missing + both_inf).all()

    return False


def values_eq_approx_remove_inf(a, b):
    return values_eq_approx(a, b, True)


def values_eq_approx_remove_nan(a, b):
    return values_eq_approx(a, b, False, True)


def values_eq_approx_remove_inf_nan(a, b):
    return values_eq_approx(a, b, True, True)


def values_eq_approx_always_true(a, b):
    return True


aesara.compile.register_view_op_c_code(
    TensorType,
    """
    Py_XDECREF(%(oname)s);
    %(oname)s = %(iname)s;
    Py_XINCREF(%(oname)s);
    """,
    version=1,
)


aesara.compile.register_deep_copy_op_c_code(
    TensorType,
    """
    int alloc = %(oname)s == NULL;
    for(int i=0; !alloc && i<PyArray_NDIM(%(oname)s); i++) {
       if(PyArray_DIMS(%(iname)s)[i] != PyArray_DIMS(%(oname)s)[i]) {
           alloc = true;
           break;
       }
    }
    if(alloc) {
        Py_XDECREF(%(oname)s);
        %(oname)s = (PyArrayObject*)PyArray_NewCopy(%(iname)s,
                                                    NPY_ANYORDER);
        if (!%(oname)s)
        {
            PyErr_SetString(PyExc_ValueError,
                            "DeepCopyOp: the copy failed!");
            %(fail)s;
        }
    } else {
        if(PyArray_CopyInto(%(oname)s, %(iname)s)){
            PyErr_SetString(PyExc_ValueError,
        "DeepCopyOp: the copy failed into already allocated space!");
            %(fail)s;
        }
    }
    """,
    version=2,
)


def tensor(*args, **kwargs):
    name = kwargs.pop("name", None)
    return TensorType(*args, **kwargs)(name=name)


cscalar = TensorType("complex64", ())
zscalar = TensorType("complex128", ())
fscalar = TensorType("float32", ())
dscalar = TensorType("float64", ())
bscalar = TensorType("int8", ())
wscalar = TensorType("int16", ())
iscalar = TensorType("int32", ())
lscalar = TensorType("int64", ())


def scalar(name=None, dtype=None):
    """Return a symbolic scalar variable.

    Parameters
    ----------
    dtype: numeric
        None means to use aesara.config.floatX.
    name
        A name to attach to this variable.

    """
    if dtype is None:
        dtype = config.floatX
    type = TensorType(dtype, ())
    return type(name)


scalars, fscalars, dscalars, iscalars, lscalars = apply_across_args(
    scalar, fscalar, dscalar, iscalar, lscalar
)

int_types = bscalar, wscalar, iscalar, lscalar
float_types = fscalar, dscalar
complex_types = cscalar, zscalar
int_scalar_types = int_types
float_scalar_types = float_types
complex_scalar_types = complex_types

cvector = TensorType("complex64", (False,))
zvector = TensorType("complex128", (False,))
fvector = TensorType("float32", (False,))
dvector = TensorType("float64", (False,))
bvector = TensorType("int8", (False,))
wvector = TensorType("int16", (False,))
ivector = TensorType("int32", (False,))
lvector = TensorType("int64", (False,))


def vector(name=None, dtype=None):
    """Return a symbolic vector variable.

    Parameters
    ----------
    dtype: numeric
        None means to use aesara.config.floatX.
    name
        A name to attach to this variable

    """
    if dtype is None:
        dtype = config.floatX
    type = TensorType(dtype, (False,))
    return type(name)


vectors, fvectors, dvectors, ivectors, lvectors = apply_across_args(
    vector, fvector, dvector, ivector, lvector
)

int_vector_types = bvector, wvector, ivector, lvector
float_vector_types = fvector, dvector
complex_vector_types = cvector, zvector

cmatrix = TensorType("complex64", (False, False))
zmatrix = TensorType("complex128", (False, False))
fmatrix = TensorType("float32", (False, False))
dmatrix = TensorType("float64", (False, False))
bmatrix = TensorType("int8", (False, False))
wmatrix = TensorType("int16", (False, False))
imatrix = TensorType("int32", (False, False))
lmatrix = TensorType("int64", (False, False))


def matrix(name=None, dtype=None):
    """Return a symbolic matrix variable.

    Parameters
    ----------
    dtype: numeric
        None means to use aesara.config.floatX.
    name
        A name to attach to this variable.

    """
    if dtype is None:
        dtype = config.floatX
    type = TensorType(dtype, (False, False))
    return type(name)


matrices, fmatrices, dmatrices, imatrices, lmatrices = apply_across_args(
    matrix, fmatrix, dmatrix, imatrix, lmatrix
)

int_matrix_types = bmatrix, wmatrix, imatrix, lmatrix
float_matrix_types = fmatrix, dmatrix
complex_matrix_types = cmatrix, zmatrix

crow = TensorType("complex64", (True, False))
zrow = TensorType("complex128", (True, False))
frow = TensorType("float32", (True, False))
drow = TensorType("float64", (True, False))
brow = TensorType("int8", (True, False))
wrow = TensorType("int16", (True, False))
irow = TensorType("int32", (True, False))
lrow = TensorType("int64", (True, False))


def row(name=None, dtype=None):
    """Return a symbolic row variable (ndim=2, broadcastable=[True,False]).

    Parameters
    ----------
    dtype: numeric type
        None means to use aesara.config.floatX.
    name
        A name to attach to this variable.

    """
    if dtype is None:
        dtype = config.floatX
    type = TensorType(dtype, (True, False))
    return type(name)


rows, frows, drows, irows, lrows = apply_across_args(row, frow, drow, irow, lrow)

ccol = TensorType("complex64", (False, True))
zcol = TensorType("complex128", (False, True))
fcol = TensorType("float32", (False, True))
dcol = TensorType("float64", (False, True))
bcol = TensorType("int8", (False, True))
wcol = TensorType("int16", (False, True))
icol = TensorType("int32", (False, True))
lcol = TensorType("int64", (False, True))


def col(name=None, dtype=None):
    """Return a symbolic column variable (ndim=2, broadcastable=[False,True]).

    Parameters
    ----------
    dtype : numeric
        None means to use aesara.config.floatX.
    name
        A name to attach to this variable.

    """
    if dtype is None:
        dtype = config.floatX
    type = TensorType(dtype, (False, True))
    return type(name)


cols, fcols, dcols, icols, lcols = apply_across_args(col, fcol, dcol, icol, lcol)

ctensor3 = TensorType("complex64", ((False,) * 3))
ztensor3 = TensorType("complex128", ((False,) * 3))
ftensor3 = TensorType("float32", ((False,) * 3))
dtensor3 = TensorType("float64", ((False,) * 3))
btensor3 = TensorType("int8", ((False,) * 3))
wtensor3 = TensorType("int16", ((False,) * 3))
itensor3 = TensorType("int32", ((False,) * 3))
ltensor3 = TensorType("int64", ((False,) * 3))


def tensor3(name=None, dtype=None):
    """Return a symbolic 3-D variable.

    Parameters
    ----------
    dtype: numeric type
        None means to use aesara.config.floatX.
    name
        A name to attach to this variable.

    """
    if dtype is None:
        dtype = config.floatX
    type = TensorType(dtype, (False, False, False))
    return type(name)


tensor3s, ftensor3s, dtensor3s, itensor3s, ltensor3s = apply_across_args(
    tensor3, ftensor3, dtensor3, itensor3, ltensor3
)

ctensor4 = TensorType("complex64", ((False,) * 4))
ztensor4 = TensorType("complex128", ((False,) * 4))
ftensor4 = TensorType("float32", ((False,) * 4))
dtensor4 = TensorType("float64", ((False,) * 4))
btensor4 = TensorType("int8", ((False,) * 4))
wtensor4 = TensorType("int16", ((False,) * 4))
itensor4 = TensorType("int32", ((False,) * 4))
ltensor4 = TensorType("int64", ((False,) * 4))


def tensor4(name=None, dtype=None):
    """Return a symbolic 4-D variable.

    Parameters
    ----------
    dtype: numeric type
        None means to use aesara.config.floatX.
    name
        A name to attach to this variable.

    """
    if dtype is None:
        dtype = config.floatX
    type = TensorType(dtype, (False, False, False, False))
    return type(name)


tensor4s, ftensor4s, dtensor4s, itensor4s, ltensor4s = apply_across_args(
    tensor4, ftensor4, dtensor4, itensor4, ltensor4
)

ctensor5 = TensorType("complex64", ((False,) * 5))
ztensor5 = TensorType("complex128", ((False,) * 5))
ftensor5 = TensorType("float32", ((False,) * 5))
dtensor5 = TensorType("float64", ((False,) * 5))
btensor5 = TensorType("int8", ((False,) * 5))
wtensor5 = TensorType("int16", ((False,) * 5))
itensor5 = TensorType("int32", ((False,) * 5))
ltensor5 = TensorType("int64", ((False,) * 5))


def tensor5(name=None, dtype=None):
    """Return a symbolic 5-D variable.

    Parameters
    ----------
    dtype: numeric type
        None means to use aesara.config.floatX.
    name
        A name to attach to this variable.

    """
    if dtype is None:
        dtype = config.floatX
    type = TensorType(dtype, (False, False, False, False, False))
    return type(name)


tensor5s, ftensor5s, dtensor5s, itensor5s, ltensor5s = apply_across_args(
    tensor5, ftensor5, dtensor5, itensor5, ltensor5
)

ctensor6 = TensorType("complex64", ((False,) * 6))
ztensor6 = TensorType("complex128", ((False,) * 6))
ftensor6 = TensorType("float32", ((False,) * 6))
dtensor6 = TensorType("float64", ((False,) * 6))
btensor6 = TensorType("int8", ((False,) * 6))
wtensor6 = TensorType("int16", ((False,) * 6))
itensor6 = TensorType("int32", ((False,) * 6))
ltensor6 = TensorType("int64", ((False,) * 6))


def tensor6(name=None, dtype=None):
    """Return a symbolic 6-D variable.

    Parameters
    ----------
    dtype: numeric type
        None means to use aesara.config.floatX.
    name
        A name to attach to this variable.

    """
    if dtype is None:
        dtype = config.floatX
    type = TensorType(dtype, (False,) * 6)
    return type(name)


tensor6s, ftensor6s, dtensor6s, itensor6s, ltensor6s = apply_across_args(
    tensor6, ftensor6, dtensor6, itensor6, ltensor6
)

ctensor7 = TensorType("complex64", ((False,) * 7))
ztensor7 = TensorType("complex128", ((False,) * 7))
ftensor7 = TensorType("float32", ((False,) * 7))
dtensor7 = TensorType("float64", ((False,) * 7))
btensor7 = TensorType("int8", ((False,) * 7))
wtensor7 = TensorType("int16", ((False,) * 7))
itensor7 = TensorType("int32", ((False,) * 7))
ltensor7 = TensorType("int64", ((False,) * 7))


def tensor7(name=None, dtype=None):
    """Return a symbolic 7-D variable.

    Parameters
    ----------
    dtype: numeric type
        None means to use aesara.config.floatX.
    name
        A name to attach to this variable.

    """
    if dtype is None:
        dtype = config.floatX
    type = TensorType(dtype, (False,) * 7)
    return type(name)


tensor7s, ftensor7s, dtensor7s, itensor7s, ltensor7s = apply_across_args(
    tensor7, ftensor7, dtensor7, itensor7, ltensor7
)
