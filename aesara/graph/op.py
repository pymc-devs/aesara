"""
Defines base classes `Op` and `CLinkerOp`.

The `Op` class is the base interface for all operations
compatible with `graph`'s :doc:`graph` routines.

"""
import copy
import inspect
import os
import re
import sys
import warnings
from abc import abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    NoReturn,
    Optional,
    Pattern,
    Set,
    Text,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
from typing_extensions import Protocol

import aesara
from aesara.configdefaults import config
from aesara.graph.basic import Apply, NoParams, Variable
from aesara.graph.fg import FunctionGraph
from aesara.graph.params_type import Params, ParamsType
from aesara.graph.utils import (
    MetaObject,
    MethodNotDefined,
    TestValueError,
    add_tag_trace,
    get_variable_trace_string,
)
from aesara.link.c.interface import CLinkerOp


if TYPE_CHECKING:
    from aesara.gpuarray.type import GpuContextType
    from aesara.graph.type import EnumList

__docformat__ = "restructuredtext en"

StorageMapType = Union[List, Dict]
ComputeMapType = Union[List, Dict]
OutputStorageType = List[Optional[List[Any]]]
ParamsInputType = Optional[Tuple[Any]]
PerformMethodType = Callable[
    [Apply, List[Any], OutputStorageType, ParamsInputType], NoReturn
]
ThunkType = Callable[[PerformMethodType, StorageMapType, ComputeMapType, Apply], Any]


def compute_test_value(node: Apply):
    """Computes the test value of a node.

    Parameters
    ----------
    node : Apply
        The `Apply` node for which the test value is computed.

    Returns
    -------
    None
        The `tag.test_value`s are updated in each `Variable` in `node.outputs`.

    """
    # Gather the test values for each input of the node
    storage_map = {}
    compute_map = {}
    for i, ins in enumerate(node.inputs):
        try:
            storage_map[ins] = [ins.get_test_value()]
            compute_map[ins] = [True]
        except TestValueError:
            # no test-value was specified, act accordingly
            if config.compute_test_value == "warn":
                warnings.warn(
                    f"Warning, Cannot compute test value: input {i} ({ins}) of Op {node} missing default value",
                    stacklevel=2,
                )
                return
            elif config.compute_test_value == "raise":
                detailed_err_msg = get_variable_trace_string(ins)

                raise ValueError(
                    f"Cannot compute test value: input {i} ({ins}) of Op {node} missing default value. {detailed_err_msg}"
                )
            elif config.compute_test_value == "ignore":
                return
            elif config.compute_test_value == "pdb":
                import pdb

                pdb.post_mortem(sys.exc_info()[2])
            else:
                raise ValueError(
                    f"{config.compute_test_value} is invalid for option config.compute_test_value"
                )

    # All inputs have test-values; perform the `Op`'s computation

    # The original values should not be destroyed, so we copy the values of the
    # inputs in `destroy_map`
    destroyed_inputs_idx = set()
    if node.op.destroy_map:
        for i_pos_list in node.op.destroy_map.values():
            destroyed_inputs_idx.update(i_pos_list)
    for inp_idx in destroyed_inputs_idx:
        inp = node.inputs[inp_idx]
        storage_map[inp] = [copy.copy(storage_map[inp][0])]

    # Prepare `storage_map` and `compute_map` for the outputs
    for o in node.outputs:
        storage_map[o] = [None]
        compute_map[o] = [False]

    # Create a thunk that performs the computation
    thunk = node.op.make_thunk(node, storage_map, compute_map, no_recycling=[])
    thunk.inputs = [storage_map[v] for v in node.inputs]
    thunk.outputs = [storage_map[v] for v in node.outputs]

    required = thunk()
    assert not required  # We provided all inputs

    for output in node.outputs:
        # Check that the output has been computed
        assert compute_map[output][0], (output, storage_map[output][0])

        # Add 'test_value' to output tag, so that downstream `Op`s can use
        # these numerical values as test values
        output.tag.test_value = storage_map[output][0]


F = TypeVar("F", bound=Callable[..., object])


class Rval(Protocol[F]):
    inputs: List
    outputs: List
    perform: List
    lazy: bool
    __call__: F


def rval_decorator(func: Any) -> Rval:
    return func


class Op(MetaObject):
    """A class that models and constructs operations in a graph.

    A `Op` instance has several responsibilities:

    - construct `Apply` nodes via `Op.make_node` method,

    - perform the numeric calculation of the modeled operation via
    the `Op.perform` method,

    - and (optionally) build the gradient-calculating sub-graphs via the
    `Op.grad` method.

    To see how `Op`, `Type`, `Variable`, and `Apply` fit together see the
    page on :doc:`graph`.

    For more details regarding how these methods should behave: see the `Op
    Contract` in the sphinx docs (advanced tutorial on `Op`-making).

    """

    default_output: Optional[int] = None
    itypes: Optional[List] = None
    otypes: Optional[List] = None
    params_type: "Optional[Union[ParamsType, GpuContextType, EnumList]]" = None
    """
    An `int` that specifies which output `Op.__call__` should return.  If
    `None`, then all outputs are returned.

    A subclass should not change this class variable, but instead override it
    with a subclass variable or an instance variable.

    """

    view_map: Dict[int, List[int]] = {}
    """
    A ``dict`` that maps output indices to the input indices of which they are
    a view.

    Examples
    ========
        view_map = {0: [1]} # first output is a view of second input
        view_map = {1: [0]} # second output is a view of first input
    """

    destroy_map: Dict[int, List[int]] = {}
    """
    A ``dict`` that maps output indices to the input indices upon which they
    operate in-place.

    Examples
    ========
        destroy_map = {0: [1]} # first output operates in-place on second input
        destroy_map = {1: [0]} # second output operates in-place on first input

    """

    def make_node(self, *inputs: Variable) -> Apply:
        """Construct an `Apply` node that represent the application of this operation to the given inputs.

        This must be implemented by sub-classes.

        Returns
        -------
        node: Apply
            The constructed `Apply` node.

        """
        if not hasattr(self, "itypes"):
            raise NotImplementedError(
                "You can either define itypes and otypes,\
             or implement make_node"
            )

        if not hasattr(self, "otypes"):
            raise NotImplementedError(
                "You can either define itypes and otypes,\
             or implement make_node"
            )

        if len(inputs) != len(self.itypes):
            raise ValueError(
                f"We expected {len(self.itypes)} inputs but got {len(inputs)}."
            )
        if not all(inp.type == it for inp, it in zip(inputs, self.itypes)):
            raise TypeError(
                f"We expected inputs of types '{str(self.itypes)}' but got types '{str([inp.type for inp in inputs])}'"
            )
        return Apply(self, inputs, [o() for o in self.otypes])

    def __call__(
        self, *inputs: Any, **kwargs
    ) -> Union[Variable, List[Variable], Tuple[Variable]]:
        """Construct an `Apply` node using `self.make_node` and return its outputs.

        This method is just a wrapper around `Op.make_node`.

        It is called by code such as:

        .. python::

           x = tensor.matrix()

           y = tensor.exp(x)

        `tensor.exp` is an Op instance, so `tensor.exp(x)` calls
        `tensor.exp.__call__` (i.e. this method) and returns its single output
        `Variable`, `y`.  The `Apply` node constructed by `self.make_node`
        behind the scenes is available via `y.owner`.

        `Op` authors are able to determine which output is returned by this method
        via the `Op.default_output` property., but subclasses are free to override this
        function and ignore `default_output`.

        Parameters
        ----------
        inputs : tuple of Variable
            The `Op`'s inputs.
        kwargs
            Additional keyword arguments to be forwarded to
            `make_node()` *except* for optional argument `return_list` (which
            defaults to `False`). If `return_list` is `True`, then the returned
            value is always a `list`. Otherwise it is either a single `Variable`
            when the output of `make_node()` contains a single element, or this
            output (unchanged) when it contains multiple elements.

        Returns
        -------
        outputs : list of Variable or Variable
            Either a list of output `Variable`s, or a single `Variable`.
            This is determined by the number of outputs produced by the
            `Op`, the value of the keyword `return_list`, and the value of
            the `Op.default_output` property.

        """
        return_list = kwargs.pop("return_list", False)
        node = self.make_node(*inputs, **kwargs)

        if config.compute_test_value != "off":
            compute_test_value(node)

        if self.default_output is not None:
            rval = node.outputs[self.default_output]
            if return_list:
                rval = [rval]
            return rval
        else:
            if return_list:
                return list(node.outputs)
            elif len(node.outputs) == 1:
                return node.outputs[0]
            else:
                return node.outputs

    def __ne__(self, other: Any) -> bool:
        return not (self == other)

    # Convenience so that subclass implementers don't have to import utils
    # just to self.add_tag_trace
    add_tag_trace = staticmethod(add_tag_trace)

    def grad(
        self, inputs: List[Variable], output_grads: List[Variable]
    ) -> List[Variable]:
        """Construct a graph for the gradient with respect to each input variable.

        Each returned `Variable` represents the gradient with respect to that
        input computed based on the symbolic gradients with respect to each
        output. If the output is not differentiable with respect to an input,
        then this method should return an instance of type `NullType` for that
        input.

        Parameters
        ----------
        inputs : list of Variable
            The input variables.
        output_grads : list of Variable
            The gradients of the output variables.

        Returns
        -------
        grads : list of Variable
            The gradients with respect to each `Variable` in `inputs`.

        """
        raise NotImplementedError()

    def L_op(
        self,
        inputs: List[Variable],
        outputs: List[Variable],
        output_grads: List[Variable],
    ) -> List[Variable]:
        r"""Construct a graph for the L-operator.

        This method is primarily used by `Lop` and dispatches to
        `Op.grad` by default.

        The *L-operator* computes a *row* vector times the Jacobian. The
        mathematical relationship is
        :math:`v \frac{\partial f(x)}{\partial x}`.
        The *L-operator* is also supported for generic tensors (not only for
        vectors).

        Parameters
        ----------
        inputs : list of Variable
        outputs : list of Variable
        output_grads : list of Variable

        """
        return self.grad(inputs, output_grads)

    def R_op(
        self, inputs: List[Variable], eval_points: Union[Variable, List[Variable]]
    ) -> List[Variable]:
        """Construct a graph for the R-operator.

        This method is primarily used by `Rop`

        Suppose the op outputs

        [ f_1(inputs), ..., f_n(inputs) ]

        Parameters
        ----------
        inputs : a Variable or list of Variables
        eval_points
            A Variable or list of Variables with the same length as inputs.
            Each element of eval_points specifies the value of the
            corresponding input at the point where the R op is to be
            evaluated.

        Returns
        -------
        list of n elements
            rval[i] should be Rop(f=f_i(inputs),
                                  wrt=inputs,
                                  eval_points=eval_points)

        """
        raise NotImplementedError()

    @abstractmethod
    def perform(
        self,
        node: Apply,
        inputs: List[Variable],
        output_storage: OutputStorageType,
        params: ParamsInputType = None,
    ) -> NoReturn:
        """Calculate the function on the inputs and put the variables in the output storage.

        Parameters
        ----------
        node : Apply
            The symbolic `Apply` node that represents this computation.
        inputs : Sequence
            Immutable sequence of non-symbolic/numeric inputs.  These
            are the values of each `Variable` in `node.inputs`.
        output_storage : list of list
            List of mutable single-element lists (do not change the length of
            these lists).  Each sub-list corresponds to value of each
            `Variable` in `node.outputs`.  The primary purpose of this method
            is to set the values of these sub-lists.
        params : tuple
            A tuple containing the values of each entry in `__props__`.

        Notes
        -----
        The `output_storage` list might contain data. If an element of
        output_storage is not `None`, it has to be of the right type, for
        instance, for a `TensorVariable`, it has to be a NumPy `ndarray`
        with the right number of dimensions and the correct dtype.
        Its shape and stride pattern can be arbitrary. It is not
        guaranteed that such pre-set values were produced by a previous call to
        this `Op.perform`; they could've been allocated by another
        `Op`'s `perform` method.
        A `Op` is free to reuse `output_storage` as it sees fit, or to
        discard it and allocate new memory.

        """

    def do_constant_folding(self, fgraph: FunctionGraph, node: Apply) -> bool:
        """Determine whether or not constant folding should be performed for the given node.

        This allows each `Op` to determine if it wants to be constant
        folded when all its inputs are constant. This allows it to choose where
        it puts its memory/speed trade-off. Also, it could make things faster
        as constants can't be used for in-place operations (see
        `*IncSubtensor`).

        Parameters
        ----------
        node : Apply
            The node for which the constant folding determination is made.

        Returns
        -------
        res : bool

        """
        return True

    def get_params(self, node: Apply) -> Params:
        """Try to detect params from the op if `Op.params_type` is set to a `ParamsType`."""
        if hasattr(self, "params_type") and isinstance(self.params_type, ParamsType):
            wrapper = self.params_type
            if not all(hasattr(self, field) for field in wrapper.fields):
                # Let's print missing attributes for debugging.
                not_found = tuple(
                    field for field in wrapper.fields if not hasattr(self, field)
                )
                raise AttributeError(
                    f"{type(self).__name__}: missing attributes {not_found} for ParamsType."
                )
            # ParamsType.get_params() will apply filtering to attributes.
            return self.params_type.get_params(self)
        raise MethodNotDefined("get_params")

    def prepare_node(
        self,
        node: Apply,
        storage_map: StorageMapType,
        compute_map: ComputeMapType,
        impl: Optional[Text],
    ) -> NoReturn:
        """Make any special modifications that the Op needs before doing `Op.make_thunk`.

        This can modify the node inplace and should return nothing.

        It can be called multiple time with different impl. It is the
        op responsibility to don't re-prepare the node when it isn't
        good to do so.

        """

    def make_py_thunk(
        self,
        node: Apply,
        storage_map: StorageMapType,
        compute_map: ComputeMapType,
        no_recycling: bool,
        debug: bool = False,
    ) -> ThunkType:
        """Make a Python thunk.

        Like `Op.make_thunk` but only makes python thunks.

        """
        node_input_storage = [storage_map[r] for r in node.inputs]
        node_output_storage = [storage_map[r] for r in node.outputs]

        if debug:
            p = node.op.debug_perform
        else:
            p = node.op.perform

        params = node.run_params()

        if params is NoParams:
            # default arguments are stored in the closure of `rval`
            @rval_decorator
            def rval(p=p, i=node_input_storage, o=node_output_storage, n=node):
                r = p(n, [x[0] for x in i], o)
                for o in node.outputs:
                    compute_map[o][0] = True
                return r

        else:
            params_val = node.params_type.filter(params)

            @rval_decorator
            def rval(
                p=p,
                i=node_input_storage,
                o=node_output_storage,
                n=node,
                params=params_val,
            ):
                r = p(n, [x[0] for x in i], o, params)
                for o in node.outputs:
                    compute_map[o][0] = True
                return r

        rval.inputs = node_input_storage
        rval.outputs = node_output_storage
        rval.perform = p
        rval.lazy = False
        return rval

    def make_thunk(
        self,
        node: Apply,
        storage_map: StorageMapType,
        compute_map: ComputeMapType,
        no_recycling: bool,
        impl: Optional[Text] = None,
    ) -> ThunkType:
        """Create a thunk.

        This function must return a thunk, that is a zero-arguments
        function that encapsulates the computation to be performed
        by this op on the arguments of the node.

        Parameters
        ----------
        node
            Something previously returned by self.make_node.
        storage_map
            dict variable -> one-element-list where a computed
            value for this variable may be found.
        compute_map
            dict variable -> one-element-list where a boolean
            value will be found. The boolean indicates whether the
            variable's storage_map container contains a valid value (True)
            or if it has not been computed yet (False).
        no_recycling
            List of variables for which it is forbidden to reuse memory
            allocated by a previous call.
        impl: str
            Description for the type of node created (e.g. ``"c"``, ``"py"``,
            etc.)

        Notes
        -----
        If the thunk consults the storage_map on every call, it is safe
        for it to ignore the no_recycling argument, because elements of the
        no_recycling list will have a value of None in the storage map.  If
        the thunk can potentially cache return values (like CLinker does),
        then it must not do so for variables in the no_recycling list.

        self.prepare_node(node, ...) is always called. If we try 'c' and it
        fail and we try again 'py', prepare_node will be called twice.
        """
        self.prepare_node(
            node, storage_map=storage_map, compute_map=compute_map, impl="py"
        )
        return self.make_py_thunk(node, storage_map, compute_map, no_recycling)

    def __str__(self):
        return getattr(type(self), "__name__", super().__str__())


class Rval_c(Protocol):
    thunk: ThunkType
    cthunk: ThunkType
    inputs: List
    outputs: List
    lazy: bool


def rval_c_decorator(func: Any) -> Rval_c:
    return func


# Caching GpuCAReduceCuda
class COp(Op, CLinkerOp):
    """An `Op` with a C implementation."""

    def make_c_thunk(
        self,
        node: Apply,
        storage_map: StorageMapType,
        compute_map: ComputeMapType,
        no_recycling: bool,
    ) -> ThunkType:
        """Create a thunk for a C implementation.

        Like `Op.make_thunk`, but will only try to make a C thunk.

        """
        # FIXME: Putting the following import on the module level causes an import cycle.
        #        The conclusion should be that the antire "make_c_thunk" method should be defined
        #        in aesara.link.c and dispatched onto the Op!
        import aesara.link.c.basic

        node_input_storage = [storage_map[r] for r in node.inputs]
        node_output_storage = [storage_map[r] for r in node.outputs]

        e = FunctionGraph(node.inputs, node.outputs)
        e_no_recycling = [
            new_o
            for (new_o, old_o) in zip(e.outputs, node.outputs)
            if old_o in no_recycling
        ]
        cl = aesara.link.c.basic.CLinker().accept(e, no_recycling=e_no_recycling)
        # float16 gets special treatment since running
        # unprepared C code will get bad results.
        if not getattr(self, "_f16_ok", False):

            def is_f16(t):
                return getattr(t, "dtype", "") == "float16"

            if any(is_f16(i.type) for i in node.inputs) or any(
                is_f16(o.type) for o in node.outputs
            ):
                # get_dynamic_module is a subset of make_thunk that is reused.
                # This just try to build the c code
                # It will raise an error for ops
                # that don't implement c code. In those cases, we
                # don't want to print a warning.
                cl.get_dynamic_module()
                print(f"Disabling C code for {self} due to unsupported float16")
                raise NotImplementedError("float16")
        outputs = cl.make_thunk(
            input_storage=node_input_storage, output_storage=node_output_storage
        )
        thunk, node_input_filters, node_output_filters = outputs

        @rval_c_decorator
        def rval_c():
            thunk()
            for o in node.outputs:
                compute_map[o][0] = True

        rval_c.thunk = thunk
        rval_c.cthunk = thunk.cthunk
        rval_c.inputs = node_input_storage
        rval_c.outputs = node_output_storage
        rval_c.lazy = False
        return rval_c

    def make_thunk(self, node, storage_map, compute_map, no_recycling, impl=None):
        """Create a thunk.

        See `Op.make_thunk`.

        Parameters
        ----------
        impl
            Currently, None, 'c' or 'py'. If 'c' or 'py' we will only try
            that version of the code.

        """
        if (impl is None and config.cxx) or impl == "c":
            self.prepare_node(
                node, storage_map=storage_map, compute_map=compute_map, impl="c"
            )
            try:
                return self.make_c_thunk(node, storage_map, compute_map, no_recycling)
            except (NotImplementedError, MethodNotDefined):
                # We requested the c code, so don't catch the error.
                if impl == "c":
                    raise

        return super().make_thunk(
            node, storage_map, compute_map, no_recycling, impl=impl
        )


def get_test_value(v: Variable) -> Any:
    """Get the test value for `v`.

    If input `v` is not already a variable, it is turned into one by calling
    `as_tensor_variable(v)`.

    Raises
    ------
    AttributeError if no test value is set.

    """
    if not isinstance(v, Variable):
        v = aesara.tensor.as_tensor_variable(v)

    return v.get_test_value()


def missing_test_message(msg: Text) -> NoReturn:
    """
    Displays msg, a message saying that some test_value is missing,
    in the appropriate form based on config.compute_test_value:

        off: The interactive debugger is off, so we do nothing.
        ignore: The interactive debugger is set to ignore missing inputs,
                so do nothing.
        warn: Display msg as a warning.

    Raises
    ------
    AttributeError
        With msg as the exception text.

    """
    action = config.compute_test_value
    if action == "raise":
        raise TestValueError(msg)
    elif action == "warn":
        warnings.warn(msg, stacklevel=2)
    else:
        assert action in ["ignore", "off"]


def get_test_values(*args: Variable) -> Union[Any, List[Any]]:
    """Get test values for multiple `Variable`s.

    Intended use:

        for val_1, ..., val_n in get_debug_values(var_1, ..., var_n):
            if some condition on val_1, ..., val_n is not met:
                missing_test_message("condition was not met")

    Given a list of variables, get_debug_values does one of three things:

        1. If the interactive debugger is off, returns an empty list
        2. If the interactive debugger is on, and all variables have
            debug values, returns a list containing a single element.
            This single element is either:
                a) if there is only one variable, the element is its
                   value
                b) otherwise, a tuple containing debug values of all
                   the variables.
        3. If the interactive debugger is on, and some variable does
            not have a debug value, issue a missing_test_message about
            the variable, and, if still in control of execution, return
            an empty list.

    """

    if config.compute_test_value == "off":
        return []

    rval = []

    for i, arg in enumerate(args):
        try:
            rval.append(get_test_value(arg))
        except TestValueError:
            if hasattr(arg, "name") and arg.name is not None:
                missing_test_message(f"Argument {i} ('{arg.name}') has no test value")
            else:
                missing_test_message(f"Argument {i} has no test value")
            return []

    if len(rval) == 1:
        return rval

    return [tuple(rval)]


ops_with_inner_function: Dict[Op, Text] = {}
"""
Registry of Ops that have an inner compiled Aesara function.

The keys are Op classes (not instances), and values are the name of the
attribute that contains the function. For instance, if the function is
self.fn, the value will be 'fn'.

We need that to be able not to run debug checks a number of times that is
exponential in the nesting level of those ops.
For instance, Scan will be registered here.

"""


class OpenMPOp(COp):
    """
    All op using OpenMP code should inherit from this Op.

    This op will check that the compiler support correctly OpenMP code.
    If not, it will print a warning and disable openmp for this Op.
    Then it will generate the not OpenMP code.

    This is needed as EPD on Windows g++ version spec information tell
    it support OpenMP, but does not include the OpenMP files.

    We also add the correct compiler flags in c_compile_args.

    """

    gxx_support_openmp: Optional[bool] = None
    """
    True/False after we tested this.

    """

    def __init__(self, openmp: Optional[bool] = None):
        if openmp is None:
            openmp = config.openmp
        self.openmp = openmp

    def __setstate__(self, d: Dict):
        self.__dict__.update(d)
        # If we unpickle old op
        if not hasattr(self, "openmp"):
            self.openmp = False

    def c_compile_args(self, **kwargs):
        """
        Return the compilation arg "fopenmp" if openMP is supported
        """
        self.update_self_openmp()
        if self.openmp:
            return ["-fopenmp"]
        return []

    def c_headers(self, **kwargs):
        """
        Return the header file name "omp.h" if openMP is supported
        """
        self.update_self_openmp()
        if self.openmp:
            return ["omp.h"]
        return []

    @staticmethod
    def test_gxx_support():
        """Check if openMP is supported."""
        from aesara.link.c.cmodule import GCC_compiler

        code = """
        #include <omp.h>
int main( int argc, const char* argv[] )
{
        int res[10];

        for(int i=0; i < 10; i++){
            res[i] = i;
        }
}
        """
        default_openmp = GCC_compiler.try_compile_tmp(
            src_code=code, tmp_prefix="test_omp_", flags=["-fopenmp"], try_run=False
        )
        return default_openmp

    def update_self_openmp(self) -> NoReturn:
        """
        Make sure self.openmp is not True if there is no support in gxx.

        """
        if self.openmp:
            if OpenMPOp.gxx_support_openmp is None:
                OpenMPOp.gxx_support_openmp = OpenMPOp.test_gxx_support()
                if not OpenMPOp.gxx_support_openmp:
                    # We want to warn only once.
                    warnings.warn(
                        "Your g++ compiler fails to compile OpenMP code. We"
                        " know this happen with some version of the EPD mingw"
                        " compiler and LLVM compiler on Mac OS X."
                        " We disable openmp everywhere in Aesara."
                        " To remove this warning set the aesara flags `openmp`"
                        " to False.",
                        stacklevel=3,
                    )
            if OpenMPOp.gxx_support_openmp is False:
                self.openmp = False
                config.openmp = False

    def prepare_node(self, node, storage_map, compute_map, impl):
        if impl == "c":
            self.update_self_openmp()


def lquote_macro(txt: Text) -> Text:
    """Turn the last line of text into a ``\\``-commented line."""
    res = []
    spl = txt.split("\n")
    for l in spl[:-1]:
        res.append(l + " \\")
    res.append(spl[-1])
    return "\n".join(res)


def get_sub_macros(sub: Dict[Text, Text]) -> Union[Tuple[Text], Tuple[Text, Text]]:
    define_macros = []
    undef_macros = []
    define_macros.append(f"#define FAIL {lquote_macro(sub['fail'])}")
    undef_macros.append("#undef FAIL")
    if "params" in sub:
        define_macros.append(f"#define PARAMS {sub['params']}")
        undef_macros.append("#undef PARAMS")

    return "\n".join(define_macros), "\n".join(undef_macros)


def get_io_macros(
    inputs: List[Text], outputs: List[Text]
) -> Union[Tuple[List[Text]], Tuple[str, str]]:
    define_macros = []
    undef_macros = []

    for i, inp in enumerate(inputs):
        define_macros.append(f"#define INPUT_{int(i)} {inp}")
        undef_macros.append(f"#undef INPUT_{int(i)}")

    for i, out in enumerate(outputs):
        define_macros.append(f"#define OUTPUT_{int(i)} {out}")
        undef_macros.append(f"#undef OUTPUT_{int(i)}")

    return "\n".join(define_macros), "\n".join(undef_macros)


class ExternalCOp(COp):
    """Class for an `Op` with an external C implementation.

    One can inherit from this class, provide its constructor with a path to
    an external C source file and the name of a function within it, and define
    an `Op` for said function.

    """

    section_re: ClassVar[Pattern] = re.compile(
        r"^#section ([a-zA-Z0-9_]+)$", re.MULTILINE
    )
    backward_re: ClassVar[Pattern] = re.compile(
        r"^AESARA_(APPLY|SUPPORT)_CODE_SECTION$", re.MULTILINE
    )
    # This is the set of allowed markers
    SECTIONS: ClassVar[Set[Text]] = {
        "init_code",
        "init_code_apply",
        "init_code_struct",
        "support_code",
        "support_code_apply",
        "support_code_struct",
        "cleanup_code_struct",
        "code",
        "code_cleanup",
    }

    @classmethod
    def get_path(cls, f: Text) -> Text:
        """Convert a path relative to the location of the class file into an absolute path.

        Paths that are already absolute are passed through unchanged.

        """
        if not os.path.isabs(f):
            class_file = inspect.getfile(cls)
            class_dir = os.path.dirname(class_file)
            f = os.path.realpath(os.path.join(class_dir, f))
        return f

    def __init__(
        self, func_files: Union[Text, List[Text]], func_name: Optional[Text] = None
    ):
        """
        Sections are loaded from files in order with sections in later
        files overriding sections in previous files.

        """
        if not isinstance(func_files, list):
            func_files = [func_files]

        self.func_name = func_name
        # Keep the original name. If we reload old pickle, we want to
        # find the new path and new version of the file in Aesara.
        self.func_files = func_files
        self.load_c_code(func_files)

        if len(self.code_sections) == 0:
            raise ValueError("No sections where defined in C files")

        if self.func_name is not None:
            if "op_code" in self.code_sections:
                # maybe a warning instead (and clearing the key)
                raise ValueError(
                    'Cannot have an "op_code" section and ' "specify the func_name"
                )
            if "op_code_cleanup" in self.code_sections:
                # maybe a warning instead (and clearing the key)
                raise ValueError(
                    'Cannot have an "op_code_cleanup" section '
                    "and specify the func_name"
                )

    def load_c_code(self, func_files: List[Text]) -> NoReturn:
        """Loads the C code to perform the `Op`."""
        func_files = [self.get_path(f) for f in func_files]
        self.func_codes = []
        for func_file in func_files:
            # U (universal) will convert all new lines format to \n.
            with open(func_file) as f:
                self.func_codes.append(f.read())

        # If both the old section markers and the new section markers are
        # present, raise an error because we don't know which ones to follow.
        old_markers_present = False
        new_markers_present = False
        for code in self.func_codes:
            if self.backward_re.search(code):
                old_markers_present = True
            if self.section_re.search(code):
                new_markers_present = True

        if old_markers_present and new_markers_present:
            raise ValueError(
                "Both the new and the old syntax for "
                "identifying code sections are present in the "
                "provided C code. These two syntaxes should not "
                "be used at the same time."
            )

        self.code_sections = dict()
        for i, code in enumerate(self.func_codes):
            if self.backward_re.search(code):
                # This is backward compat code that will go away in a while

                # Separate the code into the proper sections
                split = self.backward_re.split(code)
                n = 1
                while n < len(split):
                    if split[n] == "APPLY":
                        self.code_sections["support_code_apply"] = split[n + 1]
                    elif split[n] == "SUPPORT":
                        self.code_sections["support_code"] = split[n + 1]
                    n += 2
                continue

            elif self.section_re.search(code):

                # Check for code outside of the supported sections
                split = self.section_re.split(code)
                if split[0].strip() != "":
                    raise ValueError(
                        "Stray code before first #section "
                        f"statement (in file {func_files[i]}): {split[0]}"
                    )

                # Separate the code into the proper sections
                n = 1
                while n < len(split):
                    if split[n] not in self.SECTIONS:
                        raise ValueError(
                            f"Unknown section type (in file {func_files[i]}): {split[n]}"
                        )
                    if split[n] not in self.code_sections:
                        self.code_sections[split[n]] = ""
                    self.code_sections[split[n]] += split[n + 1]
                    n += 2

            else:
                raise ValueError(
                    f"No valid section marker was found in file {func_files[i]}"
                )

    def __get_op_params(self) -> Union[List, List[Text], List[Tuple[str, Any]]]:
        """Construct name, value pairs that will be turned into macros for use within the `Op`'s code.

        The names must be strings that are not a C keyword and the
        values must be strings of literal C representations.

        If op uses a :class:`aesara.graph.params_type.ParamsType` as ``params_type``,
        it returns:
         - a default macro ``PARAMS_TYPE`` which defines the class name of the
           corresponding C struct.
         - a macro ``DTYPE_PARAM_key`` for every ``key`` in the ParamsType for which associated
           type implements the method :func:`aesara.graph.type.CLinkerType.c_element_type`.
           ``DTYPE_PARAM_key`` defines the primitive C type name of an item in a variable
           associated to ``key``.

        """
        if hasattr(self, "params_type") and isinstance(self.params_type, ParamsType):
            wrapper = self.params_type
            params = [("PARAMS_TYPE", wrapper.name)]
            for i in range(wrapper.length):
                c_type = wrapper.types[i].c_element_type()
                if c_type:
                    # NB (reminder): These macros are currently used only in ParamsType example test
                    # (`aesara/graph/tests/test_quadratic_function.c`), to demonstrate how we can
                    # access params dtypes when dtypes may change (e.g. if based on config.floatX).
                    # But in practice, params types generally have fixed types per op.
                    params.append(
                        (
                            "DTYPE_PARAM_" + wrapper.fields[i],
                            c_type,
                        )
                    )
            return params
        return []

    def c_code_cache_version(self):
        version = (hash(tuple(self.func_codes)),)
        if hasattr(self, "params_type"):
            version += (self.params_type.c_code_cache_version(),)
        return version

    def c_init_code(self, **kwargs):
        if "init_code" in self.code_sections:
            return [self.code_sections["init_code"]]
        else:
            return super().c_init_code(**kwargs)

    def c_support_code(self, **kwargs):
        if "support_code" in self.code_sections:
            return self.code_sections["support_code"]
        else:
            return super().c_support_code(**kwargs)

    def c_init_code_apply(self, node, name):
        if "init_code_apply" in self.code_sections:
            code = self.code_sections["init_code_apply"]

            define_macros, undef_macros = self.get_c_macros(node, name)
            return "\n".join(["", define_macros, code, undef_macros])
        else:
            return super().c_init_code_apply(node, name)

    def c_support_code_apply(self, node, name):
        if "support_code_apply" in self.code_sections:
            code = self.code_sections["support_code_apply"]

            define_macros, undef_macros = self.get_c_macros(node, name)
            return "\n".join(["", define_macros, code, undef_macros])
        else:
            return super().c_support_code_apply(node, name)

    def c_support_code_struct(self, node, name):
        if "support_code_struct" in self.code_sections:
            code = self.code_sections["support_code_struct"]

            define_macros, undef_macros = self.get_c_macros(node, name)
            return "\n".join(["", define_macros, code, undef_macros])
        else:
            return super().c_support_code_struct(node, name)

    def c_cleanup_code_struct(self, node, name):
        if "cleanup_code_struct" in self.code_sections:
            code = self.code_sections["cleanup_code_struct"]

            define_macros, undef_macros = self.get_c_macros(node, name)
            return "\n".join(["", define_macros, code, undef_macros])
        else:
            return super().c_cleanup_code_struct(node, name)

    def format_c_function_args(self, inp: List[Text], out: List[Text]) -> Text:
        """Generate a string containing the arguments sent to the external C function.

        The result will have the format: ``"input0, input1, input2, &output0, &output1"``.

        """
        inp = list(inp)
        numi = getattr(self, "_cop_num_inputs", len(inp))
        while len(inp) < numi:
            inp.append("NULL")
        out = [f"&{o}" for o in out]
        numo = getattr(self, "_cop_num_outputs", len(out))
        while len(out) < numo:
            out.append("NULL")
        return ", ".join(inp + out)

    def get_c_macros(
        self, node: Apply, name: Text, check_input: Optional[bool] = None
    ) -> Union[Tuple[str], Tuple[str, str]]:
        "Construct a pair of C ``#define`` and ``#undef`` code strings."
        define_template = "#define %s %s"
        undef_template = "#undef %s"
        define_macros = []
        undef_macros = []

        if check_input is None:
            check_input = getattr(self, "check_input", True)

        if check_input:
            # Extract the various properties of the input and output variables
            variables = node.inputs + node.outputs
            variable_names = [f"INPUT_{i}" for i in range(len(node.inputs))] + [
                f"OUTPUT_{i}" for i in range(len(node.outputs))
            ]

            # Generate dtype macros
            for i, v in enumerate(variables):
                if not hasattr(v, "dtype"):
                    continue
                vname = variable_names[i]

                macro_name = "DTYPE_" + vname
                macro_value = "npy_" + v.dtype

                define_macros.append(define_template % (macro_name, macro_value))
                undef_macros.append(undef_template % macro_name)

                d = np.dtype(v.dtype)

                macro_name = "TYPENUM_" + vname
                macro_value = d.num

                define_macros.append(define_template % (macro_name, macro_value))
                undef_macros.append(undef_template % macro_name)

                macro_name = "ITEMSIZE_" + vname
                macro_value = d.itemsize

                define_macros.append(define_template % (macro_name, macro_value))
                undef_macros.append(undef_template % macro_name)

        # Generate a macro to mark code as being apply-specific
        define_macros.append(define_template % ("APPLY_SPECIFIC(str)", f"str##_{name}"))
        undef_macros.append(undef_template % "APPLY_SPECIFIC")

        for n, v in self.__get_op_params():
            define_macros.append(define_template % (n, v))
            undef_macros.append(undef_template % (n,))

        return "\n".join(define_macros), "\n".join(undef_macros)

    def c_init_code_struct(self, node, name, sub):
        """
        Stitches all the macros and "init_code" together

        """
        if "init_code_struct" in self.code_sections:
            op_code = self.code_sections["init_code_struct"]

            def_macros, undef_macros = self.get_c_macros(node, name)
            def_sub, undef_sub = get_sub_macros(sub)

            return "\n".join(
                ["", def_macros, def_sub, op_code, undef_sub, undef_macros]
            )
        else:
            return super().c_init_code_struct(node, name, sub)

    def c_code(self, node, name, inp, out, sub):
        if self.func_name is not None:
            assert "code" not in self.code_sections

            define_macros, undef_macros = self.get_c_macros(
                node, name, check_input=False
            )

            params = ""
            if "params" in sub:
                params = f", {sub['params']}"

            # Generate the C code
            return """
                %(define_macros)s
                {
                  if (%(func_name)s(%(func_args)s%(params)s) != 0) {
                    %(fail)s
                  }
                }
                %(undef_macros)s
                """ % dict(
                func_name=self.func_name,
                fail=sub["fail"],
                params=params,
                func_args=self.format_c_function_args(inp, out),
                define_macros=define_macros,
                undef_macros=undef_macros,
            )
        else:
            if "code" in self.code_sections:
                op_code = self.code_sections["code"]

                def_macros, undef_macros = self.get_c_macros(node, name)
                def_sub, undef_sub = get_sub_macros(sub)
                def_io, undef_io = get_io_macros(inp, out)

                return "\n".join(
                    [
                        def_macros,
                        def_sub,
                        def_io,
                        op_code,
                        undef_io,
                        undef_sub,
                        undef_macros,
                    ]
                )
            else:
                raise NotImplementedError()

    def c_code_cleanup(self, node, name, inputs, outputs, sub):
        """
        Stitches all the macros and "code_cleanup" together
        """
        if "code_cleanup" in self.code_sections:
            op_code = self.code_sections["code_cleanup"]

            def_macros, undef_macros = self.get_c_macros(node, name)
            def_sub, undef_sub = get_sub_macros(sub)
            def_io, undef_io = get_io_macros(inputs, outputs)

            return "\n".join(
                [
                    def_macros,
                    def_sub,
                    def_io,
                    op_code,
                    undef_io,
                    undef_sub,
                    undef_macros,
                ]
            )
        else:
            return super().c_code_cleanup(node, name, inputs, outputs, sub)


class _NoPythonOp(Op):
    """A class used to indicate that an `Op` does not provide a Python implementation.

    XXX: Do not use this class; it's only for tracking bad implementations internally.

    """

    def perform(self, node, inputs, output_storage, params=None):
        raise NotImplementedError("No Python implementation is provided by this Op.")


class _NoPythonCOp(COp):
    """A class used to indicate that a `COp` does not provide a Python implementation.

    XXX: Do not use this class; it's only for tracking bad implementations internally.

    """

    def perform(self, node, inputs, output_storage, params=None):
        raise NotImplementedError("No Python implementation is provided by this COp.")


class _NoPythonExternalCOp(ExternalCOp):
    """A class used to indicate that a `ExternalCOp` does not provide a Python implementation.

    XXX: Do not use this class; it's only for tracking bad implementations internally.

    """

    def perform(self, node, inputs, output_storage, params=None):
        raise NotImplementedError(
            "No Python implementation is provided by this ExternalCOp."
        )
