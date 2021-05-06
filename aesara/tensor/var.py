import copy
import traceback as tb
import warnings
from collections.abc import Iterable

import numpy as np

from aesara import tensor as aet
from aesara.configdefaults import config
from aesara.graph.basic import Constant, Variable
from aesara.graph.utils import MetaType
from aesara.scalar import ComplexError, IntegerDivisionError
from aesara.tensor.exceptions import AdvancedIndexingError
from aesara.tensor.type import TensorType
from aesara.tensor.utils import hash_from_ndarray


class _tensor_py_operators:
    def __abs__(self):
        return aet.math.abs_(self)

    def __neg__(self):
        return aet.math.neg(self)

    # These won't work because Python requires an int return value
    # def __int__(self): return convert_to_int32(self)
    # def __float__(self): return convert_to_float64(self)
    # def __complex__(self): return convert_to_complex128(self)

    _is_nonzero = True

    def __lt__(self, other):
        rval = aet.math.lt(self, other)
        rval._is_nonzero = False
        return rval

    def __le__(self, other):
        rval = aet.math.le(self, other)
        rval._is_nonzero = False
        return rval

    def __gt__(self, other):
        rval = aet.math.gt(self, other)
        rval._is_nonzero = False
        return rval

    def __ge__(self, other):
        rval = aet.math.ge(self, other)
        rval._is_nonzero = False
        return rval

    def __nonzero__(self):
        # Python 2.x
        return self.__bool__()

    def __bool__(self):
        # This is meant to prohibit stuff like a < b < c, which is internally
        # implemented as (a < b) and (b < c). The trouble with this is the
        # side-effect that checking for a non-NULL a by typing "if a: ..."
        # uses the same __nonzero__ method.  We want these both to work, but
        # it seems impossible.  Currently, all vars evaluate to nonzero except
        # the return values of comparison operators, which raise this
        # exception.  If you can think of a better solution, go for it!
        #
        # __bool__ is Python 3.x data model. __nonzero__ is Python 2.x.
        if self._is_nonzero:
            return True
        else:
            raise TypeError("Variables do not support boolean operations.")

    def __invert__(self):
        return aet.math.invert(self)

    def __and__(self, other):
        return aet.math.and_(self, other)

    def __or__(self, other):
        return aet.math.or_(self, other)

    def __xor__(self, other):
        return aet.math.xor(self, other)

    def __rand__(self, other):
        return aet.math.and_(other, self)

    def __ror__(self, other):
        return aet.math.or_(other, self)

    def __rxor__(self, other):
        return aet.math.xor(other, self)

    # def __iand__(self, other):
    #    return _and_inplace(self, other)
    #
    # def __ior__(self, other):
    #    return _or_inplace(self, other)
    #
    # def __ixor__(self, other):
    #    return _xor_inplace(self, other)

    def __add__(self, other):
        try:
            return aet.math.add(self, other)
        # We should catch the minimum number of exception here.
        # Otherwise this will convert error when Aesara flags
        # compute_test_value is used
        # Evidently, we need to catch NotImplementedError
        # TypeError from as_tensor_variable are caught in Elemwise.make_node
        # Oterwise TensorVariable * SparseVariable won't work!
        except (NotImplementedError, TypeError):
            # We must return NotImplemented and not an
            # NotImplementedError or raise an NotImplementedError.
            # That way python will give a good error message like this
            # `TypeError: unsupported operand type(s) for +:
            # 'TensorVariable' and 'TensorVariable'`
            return NotImplemented

    def __sub__(self, other):
        # See explanation in __add__ for the error catched
        # and the return value in that case
        try:
            return aet.math.sub(self, other)
        except (NotImplementedError, TypeError):
            return NotImplemented

    def __mul__(self, other):
        # See explanation in __add__ for the error catched
        # and the return value in that case
        try:
            return aet.math.mul(self, other)
        except (NotImplementedError, TypeError):
            return NotImplemented

    def __div__(self, other):
        # See explanation in __add__ for the error catched
        # and the return value in that case
        try:
            return aet.math.div_proxy(self, other)
        except IntegerDivisionError:
            # This is to raise the exception that occurs when trying to divide
            # two integer arrays (currently forbidden).
            raise
        except (NotImplementedError, TypeError):
            return NotImplemented

    __truediv__ = __div__

    def __pow__(self, other):
        # See explanation in __add__ for the error catched
        # and the return value in that case
        try:
            return aet.math.pow(self, other)
        except (NotImplementedError, TypeError):
            return NotImplemented

    def __mod__(self, other):
        # See explanation in __add__ for the error catched
        # and the return value in that case
        try:
            return aet.math.mod_check(self, other)
        except ComplexError:
            # This is to raise the exception that occurs when trying to compute
            # x % y with either x or y a complex number.
            raise
        except (NotImplementedError, TypeError):
            return NotImplemented

    def __divmod__(self, other):
        return aet.math.divmod(self, other)

    def __truediv__(self, other):
        return aet.math.true_div(self, other)

    def __floordiv__(self, other):
        return aet.math.floor_div(self, other)

    def __rtruediv__(self, other):
        return aet.math.true_div(other, self)

    def __rfloordiv__(self, other):
        return aet.math.floor_div(other, self)

    # Do not use these; in-place `Op`s should be inserted by optimizations
    # only!
    # def __iadd__(self, other):
    #    return _add_inplace(self, other)
    # def __isub__(self, other):
    #    return _sub_inplace(self, other)
    #
    # def __imul__(self, other):
    #    return _mul_inplace(self, other)
    #
    # def __idiv__(self, other):
    #    return _div_inplace(self, other)
    #
    # def __ipow__(self, other):
    #    return _pow_inplace(self, other)

    def __radd__(self, other):
        return aet.math.add(other, self)

    def __rsub__(self, other):
        return aet.math.sub(other, self)

    def __rmul__(self, other):
        return aet.math.mul(other, self)

    def __rdiv__(self, other):
        return aet.math.div_proxy(other, self)

    def __rmod__(self, other):
        return aet.math.mod(other, self)

    def __rdivmod__(self, other):
        return aet.math.divmod(other, self)

    def __rpow__(self, other):
        return aet.math.pow(other, self)

    def __ceil__(self):
        return aet.math.ceil(self)

    def __floor__(self):
        return aet.math.floor(self)

    def __trunc__(self):
        return aet.math.trunc(self)

    # NumPy-like transpose property
    @property
    def T(self):
        return aet.basic.transpose(self)

    def transpose(self, *axes):
        """Transpose this array.

        Returns
        -------
        object
            `tensor.transpose(self, axes)` or `tensor.transpose(self, axes[0])`.

        If only one `axes` argument is provided and it is iterable, then it is
        assumed to be the entire axes tuple, and passed intact to
        tensor.transpose.

        """
        if len(axes) == 0:
            return aet.basic.transpose(self)
        try:
            iter(axes[0])
            iterable = True
        except TypeError:
            iterable = False
        if len(axes) == 1 and iterable:
            return aet.basic.transpose(self, axes[0])
        else:
            return aet.basic.transpose(self, axes)

    @property
    def shape(self):
        return aet.shape(self)

    @property
    def size(self):
        if self.ndim == 1:
            return self.shape[0]
        else:
            return aet.math.prod(self.shape)

    def any(self, axis=None, keepdims=False):
        return aet.math.any(self, axis=axis, keepdims=keepdims)

    def all(self, axis=None, keepdims=False):
        return aet.math.all(self, axis=axis, keepdims=keepdims)

    # Old note: "We can't implement this because Python requests that this
    # function returns an integer."
    # TODO: We could use `get_vector_length` and let it raise an exception just like
    # `__iter__` does
    # def __len__(self):
    #     raise Exception("Aesara Variables can't work with len(Aesara "
    #                     "Variable) due to Python restriction. You can use "
    #                     "AesaraVariable.shape[0] instead.")

    def reshape(self, shape, ndim=None):
        """Return a reshaped view/copy of this variable.

        Parameters
        ----------
        shape
            Something that can be converted to a symbolic vector of integers.
        ndim
            The length of the shape. Passing None here means for
            Aesara to try and guess the length of `shape`.


        .. warning:: This has a different signature than numpy's
                     ndarray.reshape!
                     In numpy you do not need to wrap the shape arguments
                     in a tuple, in aesara you do need to.

        """
        if ndim is not None:
            if not isinstance(ndim, int):
                raise ValueError(
                    "Expected ndim to be an integer, is " + str(type(ndim))
                )

        return aet.reshape(self, shape, ndim=ndim)

    def dimshuffle(self, *pattern):
        """
        Reorder the dimensions of this variable, optionally inserting
        broadcasted dimensions.

        Parameters
        ----------
        pattern
            List/tuple of int mixed with 'x' for broadcastable dimensions.

        Examples
        --------
        For example, to create a 3D view of a [2D] matrix, call
        ``dimshuffle([0,'x',1])``.  This will create a 3D view such that the
        middle dimension is an implicit broadcasted dimension.  To do the same
        thing on the transpose of that matrix, call ``dimshuffle([1, 'x', 0])``.

        Notes
        -----
        This function supports the pattern passed as a tuple, or as a
        variable-length argument (e.g. ``a.dimshuffle(pattern)`` is equivalent
        to ``a.dimshuffle(*pattern)`` where ``pattern`` is a list/tuple of ints
        mixed with 'x' characters).

        See Also
        --------
        DimShuffle

        """
        if (len(pattern) == 1) and (isinstance(pattern[0], (list, tuple))):
            pattern = pattern[0]
        op = aet.elemwise.DimShuffle(list(self.type.broadcastable), pattern)
        return op(self)

    def flatten(self, ndim=1):
        return aet.basic.flatten(self, ndim)

    def ravel(self):
        return aet.basic.flatten(self)

    def diagonal(self, offset=0, axis1=0, axis2=1):
        return aet.basic.diagonal(self, offset, axis1, axis2)

    def transfer(self, target):
        """Transfer this this array's data to another device.

        If `target` is `'cpu'` this will transfer to a TensorType (if
        not already one).  Other types may define additional targets.

        Parameters
        ----------
        target : str
            The desired location of the output variable
        """
        return aet.basic.transfer(self, target)

    def arccos(self):
        return aet.math.arccos(self)

    def arccosh(self):
        return aet.math.arccosh(self)

    def arcsin(self):
        return aet.math.arcsin(self)

    def arcsinh(self):
        return aet.math.arcsinh(self)

    def arctan(self):
        return aet.math.arctan(self)

    def arctanh(self):
        return aet.math.arctanh(self)

    def ceil(self):
        return aet.math.ceil(self)

    def cos(self):
        return aet.math.cos(self)

    def cosh(self):
        return aet.math.cosh(self)

    def deg2rad(self):
        return aet.math.deg2rad(self)

    def exp(self):
        return aet.math.exp(self)

    def exp2(self):
        return aet.math.exp2(self)

    def expm1(self):
        return aet.math.expm1(self)

    def floor(self):
        return aet.math.floor(self)

    def log(self):
        return aet.math.log(self)

    def log10(self):
        return aet.math.log10(self)

    def log1p(self):
        return aet.math.log1p(self)

    def log2(self):
        return aet.math.log2(self)

    def rad2deg(self):
        return aet.math.rad2deg(self)

    def sin(self):
        return aet.math.sin(self)

    def sinh(self):
        return aet.math.sinh(self)

    def sqrt(self):
        return aet.math.sqrt(self)

    def tan(self):
        return aet.math.tan(self)

    def tanh(self):
        return aet.math.tanh(self)

    def trunc(self):
        return aet.math.trunc(self)

    def astype(self, dtype):
        return aet.basic.cast(self, dtype)

    def __getitem__(self, args):
        def includes_bool(args_el):
            if isinstance(args_el, (np.bool_, bool)) or (
                hasattr(args_el, "dtype") and args_el.dtype == "bool"
            ):
                return True
            if not isinstance(args_el, Variable) and isinstance(args_el, Iterable):
                for el in args_el:
                    if includes_bool(el):
                        return True
            return False

        if isinstance(args, list) and any([isinstance(a, slice) for a in args]):
            pass
        elif not isinstance(args, tuple):
            args = (args,)

        # Count the dimensions, check for bools and find ellipses.
        ellipses = []
        index_dim_count = 0
        for i, arg in enumerate(args):
            if arg is np.newaxis:
                # no increase in index_dim_count
                pass
            elif arg is Ellipsis:
                # no increase in index_dim_count
                ellipses.append(i)
            elif (
                isinstance(arg, (np.ndarray, Variable))
                and hasattr(arg, "dtype")
                and arg.dtype == "bool"
            ):
                index_dim_count += arg.ndim
            else:
                # Python arrays can contain a mixture of bools and integers,
                # which requires complex rules to handle all special cases.
                # These rules differ slightly between NumPy versions.
                # Since earlier versions of Aesara did not support any boolean
                # indexing, it is safe to throw an error if we encounter
                # any of these difficult cases.
                if includes_bool(arg):
                    raise TypeError(
                        "TensorType does not support Python bools "
                        "for indexing, such as tensor[[True, False]]. "
                        "To use a boolean mask, convert the mask to "
                        "a NumPy array first, e.g., "
                        "tensor[numpy.array([True, False])]."
                    )
                index_dim_count += 1

        # Check if the number of dimensions isn't too large.
        if self.ndim < index_dim_count:
            raise IndexError("too many indices for array")

        # Convert an Ellipsis if provided into an appropriate number of
        # slice(None).
        if len(ellipses) > 1:
            raise IndexError("an index can only have a single Ellipsis (`...`)")
        elif len(ellipses) == 1:
            ellipsis_at = ellipses[0]
            args = list(args)
            args[ellipsis_at : ellipsis_at + 1] = [slice(None)] * (
                self.ndim - index_dim_count
            )

        def is_empty_array(val):
            return (isinstance(val, (tuple, list)) and len(val) == 0) or (
                isinstance(val, np.ndarray) and val.size == 0
            )

        # Force input to be int64 datatype if input is an empty list or tuple
        # Else leave it as is if it is a real number
        # Convert python literals to aesara constants
        args = tuple(
            [
                aet.subtensor.as_index_constant(
                    np.array(inp, dtype=np.int64) if is_empty_array(inp) else inp
                )
                for inp in args
            ]
        )

        # Determine if advanced indexing is needed or not.  The logic is
        # already in `Subtensor.convert`: if it succeeds, standard indexing is
        # used; if it fails with AdvancedIndexingError, advanced indexing is
        # used
        advanced = False
        axis = None
        for i, arg in enumerate(args):
            if includes_bool(arg):
                advanced = True
                axis = None
                break

            if arg is not np.newaxis:
                try:
                    aet.subtensor.Subtensor.convert(arg)
                except AdvancedIndexingError:
                    if advanced:
                        axis = None
                        break
                    else:
                        advanced = True
                        axis = i

        if advanced:
            if (
                axis is not None
                and all(isinstance(a, slice) and a == slice(None) for a in args[:axis])
                and all(
                    isinstance(a, slice) and a == slice(None) for a in args[axis + 1 :]
                )
                # I.e. if the first advanced index is a tensor or NumPy array,
                # then it can't be boolean (in order to meet this condition).
                # How could this possibly occur; we filter for booleans above,
                # right?
                # and (not hasattr(args[axis], "dtype") or args[axis].dtype != "bool")
                and isinstance(
                    args[axis],
                    (
                        np.ndarray,
                        list,
                        TensorVariable,
                        TensorConstant,
                        aet.sharedvar.TensorSharedVariable,
                    ),
                )
            ):
                # If we're here, it means that an advanced index was found
                # (e.g. an array of indices) and it was surrounded by full
                # slices--or no slices (e.g. `x[:, :, idx, ...]`).  The
                # `take` function/`Op` serves exactly this type of indexing,
                # so we simply return its result.
                return self.take(args[axis], axis)
            else:
                return aet.subtensor.advanced_subtensor(self, *args)
        else:
            if np.newaxis in args:
                # `np.newaxis` (i.e. `None`) in NumPy indexing mean "add a new
                # broadcastable dimension at this location".  Since Aesara adds
                # new broadcastable dimensions via the `DimShuffle` `Op`, the
                # following code uses said `Op` to add one of the new axes and
                # then uses recursion to apply any other indices and add any
                # remaining new axes.

                counter = 0
                pattern = []
                new_args = []
                for arg in args:
                    if arg == np.newaxis:
                        pattern.append("x")
                        new_args.append(slice(None, None, None))
                    else:
                        pattern.append(counter)
                        counter += 1
                        new_args.append(arg)
                view = self.dimshuffle(pattern)
                full_slices = True
                for arg in new_args:
                    # We can't do arg == slice(None, None, None) as in
                    # Python 2.7, this call __lt__ if we have a slice
                    # with some symbolic variable.
                    if not (
                        isinstance(arg, slice)
                        and arg.start is None
                        and arg.stop is None
                        and arg.step is None
                    ):
                        full_slices = False
                if full_slices:
                    return view
                else:
                    return view.__getitem__(tuple(new_args))
            else:
                return aet.subtensor.Subtensor(args)(
                    self,
                    *aet.subtensor.Subtensor.collapse(
                        args, lambda entry: isinstance(entry, Variable)
                    ),
                )

    def take(self, indices, axis=None, mode="raise"):
        return aet.subtensor.take(self, indices, axis, mode)

    def copy(self, name=None):
        """Return a symbolic copy and optionally assign a name.

        Does not copy the tags.
        """
        copied_variable = aet.basic.tensor_copy(self)
        copied_variable.name = name
        return copied_variable

    def __iter__(self):
        try:
            for i in range(aet.basic.get_vector_length(self)):
                yield self[i]
        except TypeError:
            # This prevents accidental iteration via sum(self)
            raise TypeError(
                "TensorType does not support iteration. "
                "Maybe you are using builtins.sum instead of "
                "aesara.tensor.math.sum? (Maybe .max?)"
            )

    @property
    def ndim(self):
        """The rank of this tensor."""
        return self.type.ndim

    @property
    def broadcastable(self):
        """
        The broadcastable signature of this tensor.

        See Also
        --------
        broadcasting

        """
        return self.type.broadcastable

    @property
    def dtype(self):
        """The dtype of this tensor."""
        return self.type.dtype

    def __dot__(left, right):
        return aet.math.dense_dot(left, right)

    def __rdot__(right, left):
        return aet.math.dense_dot(left, right)

    dot = __dot__

    def sum(self, axis=None, dtype=None, keepdims=False, acc_dtype=None):
        """See `aesara.tensor.math.sum`."""
        return aet.math.sum(
            self, axis=axis, dtype=dtype, keepdims=keepdims, acc_dtype=acc_dtype
        )

    def prod(self, axis=None, dtype=None, keepdims=False, acc_dtype=None):
        """See `aesara.tensor.math.prod`."""
        return aet.math.prod(
            self, axis=axis, dtype=dtype, keepdims=keepdims, acc_dtype=acc_dtype
        )

    def norm(self, L, axis=None, keepdims=False):
        if L == 0:
            raise NotImplementedError()
        if np.isinf(L):
            raise NotImplementedError()
        # optimizations will/should catch cases like L=1, L=2
        y = aet.math.pow(
            aet.math.pow(aet.math.abs_(self), L).sum(axis=axis),
            1.0 / L,
        )
        if keepdims:
            return aet.math.makeKeepDims(self, y, axis)
        else:
            return y

    def mean(self, axis=None, dtype=None, keepdims=False, acc_dtype=None):
        """See `aesara.tensor.math.mean`."""
        return aet.math.mean(
            self, axis=axis, dtype=dtype, keepdims=keepdims, acc_dtype=acc_dtype
        )

    def var(self, axis=None, ddof=0, keepdims=False, corrected=False):
        """See `aesara.tensor.math.var`."""
        return aet.math.var(
            self, axis=axis, ddof=ddof, keepdims=keepdims, corrected=corrected
        )

    def std(self, axis=None, ddof=0, keepdims=False, corrected=False):
        """See `aesara.tensor.math.std`."""
        return aet.math.std(
            self, axis=axis, ddof=ddof, keepdims=keepdims, corrected=corrected
        )

    def min(self, axis=None, keepdims=False):
        """See `aesara.tensor.math.min`."""
        return aet.math.min(self, axis, keepdims=keepdims)

    def max(self, axis=None, keepdims=False):
        """See `aesara.tensor.math.max`."""
        return aet.math.max(self, axis, keepdims=keepdims)

    def argmin(self, axis=None, keepdims=False):
        """See `aesara.tensor.math.argmin`."""
        return aet.math.argmin(self, axis, keepdims=keepdims)

    def argmax(self, axis=None, keepdims=False):
        """See `aesara.tensor.math.argmax`."""
        return aet.math.argmax(self, axis, keepdims=keepdims)

    def nonzero(self, return_matrix=False):
        """See `aesara.tensor.basic.nonzero`."""
        return aet.nonzero(self, return_matrix=return_matrix)

    def nonzero_values(self):
        """See `aesara.tensor.basic.nonzero_values`."""
        return aet.nonzero_values(self)

    def sort(self, axis=-1, kind="quicksort", order=None):
        """See `aesara.tensor.sort.sort`."""
        return aet.sort(self, axis, kind, order)

    def argsort(self, axis=-1, kind="quicksort", order=None):
        """See `aesara.tensor.sort.argsort`."""
        from aesara.tensor.sort import argsort

        return argsort(self, axis, kind, order)

    def clip(self, a_min, a_max):
        "See `aesara.tensor.math.clip`."
        return aet.math.clip(self, a_min, a_max)

    def conj(self):
        """See `aesara.tensor.math.conj`."""
        return aet.math.conj(self)

    conjugate = conj

    def repeat(self, repeats, axis=None):
        """See `aesara.tensor.basic.repeat`."""
        return aet.extra_ops.repeat(self, repeats, axis)

    def round(self, mode=None):
        """See `aesara.tensor.math.round`."""
        return aet.math.round(self, mode)

    def trace(self):
        return aet.nlinalg.trace(self)

    # This value is set so that Aesara arrays will trump NumPy operators.
    __array_priority__ = 1000

    def get_scalar_constant_value(self):
        return aet.basic.get_scalar_constant_value(self)

    def zeros_like(model, dtype=None):
        return aet.basic.zeros_like(model, dtype=dtype)

    def ones_like(model, dtype=None):
        return aet.basic.ones_like(model, dtype=dtype)

    def cumsum(self, axis=None):
        return aet.extra_ops.cumsum(self, axis)

    def cumprod(self, axis=None):
        return aet.extra_ops.cumprod(self, axis)

    def searchsorted(self, v, side="left", sorter=None):
        return aet.extra_ops.searchsorted(self, v, side, sorter)

    def ptp(self, axis=None):
        """See `aesara.tensor.math.ptp`."""

        return aet.math.ptp(self, axis)

    def swapaxes(self, axis1, axis2):
        """See `aesara.tensor.basic.swapaxes`.

        If a matrix is provided with the right axes, its transpose
        will be returned.

        """
        return aet.basic.swapaxes(self, axis1, axis2)

    def fill(self, value):
        """Fill inputted tensor with the assigned value."""
        return aet.basic.fill(self, value)

    def choose(self, choices, out=None, mode="raise"):
        """
        Construct an array from an index array and a set of arrays to choose
        from.

        """
        return aet.basic.choose(self, choices, out=None, mode="raise")

    def squeeze(self):
        """
        Remove broadcastable dimensions from the shape of an array.

        It returns the input array, but with the broadcastable dimensions
        removed. This is always `x` itself or a view into `x`.

        """
        return aet.extra_ops.squeeze(self)

    def compress(self, a, axis=None):
        """Return selected slices only."""
        return aet.extra_ops.compress(self, a, axis=axis)


class TensorVariable(_tensor_py_operators, Variable):
    """
    Subclass to add the tensor operators to the basic `Variable` class.

    """

    def __init__(self, type, owner=None, index=None, name=None):
        super().__init__(type, owner=owner, index=index, name=name)
        if config.warn_float64 != "ignore" and type.dtype == "float64":
            msg = (
                "You are creating a TensorVariable "
                "with float64 dtype. You requested an action via "
                "the Aesara flag warn_float64={ignore,warn,raise,pdb}."
            )
            if config.warn_float64 == "warn":
                # Get the user stack. We don't want function inside the
                # tensor and graph directory to be shown to the user.
                x = tb.extract_stack()
                nb_rm = 0
                while x:
                    file_path = x[-1][0]
                    rm = False
                    for p in [
                        "aesara/tensor/",
                        "aesara\\tensor\\",
                        "aesara/graph/",
                        "aesara\\tensor\\",
                    ]:
                        if p in file_path:
                            x = x[:-1]
                            nb_rm += 1
                            rm = True
                            break
                    if not rm:
                        break
                warnings.warn(msg, stacklevel=1 + nb_rm)
            elif config.warn_float64 == "raise":
                raise Exception(msg)
            elif config.warn_float64 == "pdb":
                import pdb

                pdb.set_trace()


TensorType.Variable = TensorVariable


class TensorConstantSignature(tuple):
    """
    A Signature object for comparing TensorConstant instances.

    An instance is a pair: (Type instance, ndarray).

    """

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        try:
            (t0, d0), (t1, d1) = self, other
        except Exception:
            return False

        # N.B. compare shape to ensure no broadcasting in ==
        if t0 != t1 or d0.shape != d1.shape:
            return False

        self.no_nan  # Ensure has_nan is computed.
        # Note that in the comparisons below, the elementwise comparisons
        # come last because they are the most expensive checks.
        if self.has_nan:
            other.no_nan  # Ensure has_nan is computed.
            return (
                other.has_nan
                and self.sum == other.sum
                and (self.no_nan.mask == other.no_nan.mask).all()
                and
                # Note that the second test below (==) may crash e.g. for
                # a single scalar NaN value, so we do not run it when all
                # values are missing.
                (self.no_nan.mask.all() or (self.no_nan == other.no_nan).all())
            )
        else:
            # Simple case where we do not need to worry about NaN values.
            # (note that if there are NaN values in d1, this will return
            # False, which is why we do not bother with testing `other.has_nan`
            # here).
            return (self.sum == other.sum) and np.all(d0 == d1)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        t, d = self
        return hash((type(self), t, d.shape, self.sum))

    def aesara_hash(self):
        _, d = self
        return hash_from_ndarray(d)

    def _get_sum(self):
        """Compute sum of non NaN / Inf values in the array."""
        try:
            return self._sum
        except AttributeError:
            self._sum = self.no_nan.sum()
            # The following 2 lines are needede as in Python 3.3 with NumPy
            # 1.7.1, numpy.ndarray and numpy.memmap aren't hashable.
            if isinstance(self._sum, np.memmap):
                self._sum = np.asarray(self._sum).item()
            if self.has_nan and self.no_nan.mask.all():
                # In this case the sum is not properly computed by numpy.
                self._sum = 0
            if np.isinf(self._sum) or np.isnan(self._sum):
                # NaN may happen when there are both -inf and +inf values.
                if self.has_nan:
                    # Filter both NaN and Inf values.
                    mask = self.no_nan.mask + np.isinf(self[1])
                else:
                    # Filter only Inf values.
                    mask = np.isinf(self[1])
                if mask.all():
                    self._sum = 0
                else:
                    self._sum = np.ma.masked_array(self[1], mask).sum()
                # At this point there should be no more NaN.
                assert not np.isnan(self._sum)
        return self._sum

    sum = property(_get_sum)

    def _get_no_nan(self):
        try:
            return self._no_nan
        except AttributeError:
            nan_mask = np.isnan(self[1])
            if nan_mask.any():
                self._no_nan = np.ma.masked_array(self[1], nan_mask)
                self.has_nan = True
            else:
                self._no_nan = self[1]
                self.has_nan = False
        return self._no_nan

    no_nan = property(_get_no_nan)


class TensorConstant(_tensor_py_operators, Constant):
    """Subclass to add the tensor operators to the basic `Constant` class.

    To create a TensorConstant, use the `constant` function in this module.

    """

    def __init__(self, type, data, name=None):
        Constant.__init__(self, type, data, name)
        self.tag.unique_value = None
        if isinstance(data, np.ndarray) and data.ndim > 0:
            flat_data = data.ravel()
            if flat_data.shape[0]:
                if (flat_data == flat_data[0]).all():
                    self.tag.unique_value = flat_data[0]

    def __str__(self):
        if self.tag.unique_value is not None:
            name = f"{self.data.shape} of {self.tag.unique_value}"
        else:
            name = f"{self.data}"
        if len(name) > 20:
            name = name[:10] + ".." + name[-10:]

        return "TensorConstant{%s}" % name

    def signature(self):
        return TensorConstantSignature((self.type, self.data))

    def equals(self, other):
        # Override Contant.equals to allow to compare with
        # numpy.ndarray, and python type.
        if isinstance(other, (np.ndarray, int, float)):
            # Make a TensorConstant to be able to compare
            other = aet.basic.constant(other)
        return (
            isinstance(other, TensorConstant) and self.signature() == other.signature()
        )

    def __copy__(self):
        # We need to do this to remove the cached attribute
        return type(self)(self.type, self.data, self.name)

    def __deepcopy__(self, memo):
        # We need to do this to remove the cached attribute
        return type(self)(
            copy.deepcopy(self.type, memo),
            copy.deepcopy(self.data, memo),
            copy.deepcopy(self.name, memo),
        )


TensorType.Constant = TensorConstant


class DenseVariableMeta(MetaType):
    def __instancecheck__(self, o):
        if type(o) == TensorVariable or isinstance(o, DenseVariableMeta):
            return True
        return False


class DenseTensorVariable(TensorType, metaclass=DenseVariableMeta):
    """A `Variable` for dense tensors."""


class DenseConstantMeta(MetaType):
    def __instancecheck__(self, o):
        if type(o) == TensorConstant or isinstance(o, DenseConstantMeta):
            return True
        return False


class DenseTensorConstant(TensorType, metaclass=DenseConstantMeta):
    """A `Constant` for dense tensors."""
