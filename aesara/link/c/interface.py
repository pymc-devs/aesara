from abc import abstractmethod
from typing import Callable, Dict, List, Text, Tuple, Union

from aesara.graph.basic import Apply, Constant
from aesara.graph.utils import MethodNotDefined


class CLinkerObject:
    """Standard methods for an `Op` or `Type` used with the `CLinker`."""

    def c_headers(self, **kwargs) -> List[Text]:
        """Return a list of header files required by code returned by this class.

        These strings will be prefixed with ``#include``  and inserted at the
        beginning of the C source code.

        Strings in this list that start neither with ``<`` nor ``"`` will be
        enclosed in double-quotes.

        Examples
        --------

            def c_headers(self, **kwargs):
                return ['<iostream>', '<math.h>', '/full/path/to/header.h']


        """
        return []

    def c_header_dirs(self, **kwargs) -> List[Text]:
        """Return a list of header search paths required by code returned by this class.

        Provides search paths for headers, in addition to those in any relevant
        environment variables.

        Note: for Unix compilers, these are the things that get `-I` prefixed
        in the compiler command line arguments.

        Examples
        --------

            def c_header_dirs(self, **kwargs):
                return ['/usr/local/include', '/opt/weirdpath/src/include']

        """
        return []

    def c_libraries(self, **kwargs) -> List[Text]:
        """Return a list of libraries required by code returned by this class.

        The compiler will search the directories specified by the environment
        variable LD_LIBRARY_PATH in addition to any returned by `c_lib_dirs`.

        Note: for Unix compilers, these are the things that get ``-l`` prefixed
        in the compiler command line arguments.


        Examples
        --------

            def c_libraries(self, **kwargs):
                return ['gsl', 'gslcblas', 'm', 'fftw3', 'g2c'].

        """
        return []

    def c_lib_dirs(self, **kwargs) -> List[Text]:
        """Return a list of library search paths required by code returned by this class.

        Provides search paths for libraries, in addition to those in any
        relevant environment variables (e.g. ``LD_LIBRARY_PATH``).

        Note: for Unix compilers, these are the things that get ``-L`` prefixed
        in the compiler command line arguments.

        Examples
        --------

            def c_lib_dirs(self, **kwargs):
                return ['/usr/local/lib', '/opt/weirdpath/build/libs'].

        """
        return []

    def c_support_code(self, **kwargs) -> Text:
        """Return utility code for use by a `Variable` or `Op`.

        This is included at global scope prior to the rest of the code for this class.

        Question: How many times will this support code be emitted for a graph
        with many instances of the same type?

        Returns
        -------
        str

        """
        return ""

    def c_compile_args(self, **kwargs) -> List[Text]:
        """Return a list of recommended compile arguments for code returned by other methods in this class.

        Compiler arguments related to headers, libraries and search paths
        should be provided via the functions `c_headers`, `c_libraries`,
        `c_header_dirs`, and `c_lib_dirs`.

        Examples
        --------

            def c_compile_args(self, **kwargs):
                return ['-ffast-math']

        """
        return []

    def c_no_compile_args(self, **kwargs) -> List[Text]:
        """Return a list of incompatible ``gcc`` compiler arguments.

        We will remove those arguments from the command line of ``gcc``. So if
        another Op adds a compile arg in the graph that is incompatible
        with this Op, the incompatible arg will not be used.

        This is used, for instance, to remove ``-ffast-math``.

        """
        return []

    def c_init_code(self, **kwargs) -> List[Text]:
        """Return a list of code snippets to be inserted in module initialization."""
        return []

    def c_code_cache_version(self) -> Union[Tuple[int], Tuple]:
        """Return a tuple of integers indicating the version of this `Op`.

        An empty tuple indicates an 'unversioned' `Op` that will not be cached
        between processes.

        The cache mechanism may erase cached modules that have been superceded
        by newer versions. See `ModuleCache` for details.

        See Also
        --------
        c_code_cache_version_apply()

        """
        return ()


class CLinkerOp(CLinkerObject):
    """Interface definition for `Op` subclasses compiled by `CLinker`.

    A subclass should implement WRITEME.

    WRITEME: structure of automatically generated C code.
    Put this in doc/code_structure.txt

    """

    @abstractmethod
    def c_code(
        self,
        node: Apply,
        name: Text,
        inputs: List[Text],
        outputs: List[Text],
        sub: Dict[Text, Text],
    ) -> Text:
        """Return the C implementation of an `Op`.

        Returns C code that does the computation associated to this `Op`,
        given names for the inputs and outputs.

        Parameters
        ----------
        node : Apply instance
            The node for which we are compiling the current c_code.
           The same Op may be used in more than one node.
        name : str
            A name that is automatically assigned and guaranteed to be
            unique.
        inputs : list of strings
            There is a string for each input of the function, and the
            string is the name of a C variable pointing to that input.
            The type of the variable depends on the declared type of
            the input.  There is a corresponding python variable that
            can be accessed by prepending "py_" to the name in the
            list.
        outputs : list of strings
            Each string is the name of a C variable where the Op should
            store its output.  The type depends on the declared type of
            the output.  There is a corresponding python variable that
            can be accessed by prepending "py_" to the name in the
            list.  In some cases the outputs will be preallocated and
            the value of the variable may be pre-filled.  The value for
            an unallocated output is type-dependent.
        sub : dict of strings
            Extra symbols defined in `CLinker` sub symbols (such as 'fail').
            WRITEME

        """
        raise NotImplementedError()

    def c_code_cache_version_apply(self, node: Apply) -> Union[Tuple[int], Tuple]:
        """Return a tuple of integers indicating the version of this `Op`.

        An empty tuple indicates an 'unversioned' `Op` that will not be
        cached between processes.

        The cache mechanism may erase cached modules that have been
        superceded by newer versions.  See `ModuleCache` for details.

        See Also
        --------
        c_code_cache_version()

        Notes
        -----
            This function overrides `c_code_cache_version` unless it explicitly
            calls `c_code_cache_version`. The default implementation simply
            calls `c_code_cache_version` and ignores the `node` argument.

        """
        return self.c_code_cache_version()

    def c_code_cleanup(
        self,
        node: Apply,
        name: Text,
        inputs: List[Text],
        outputs: List[Text],
        sub: Dict[Text, Text],
    ) -> Text:
        """Return C code to run after `CLinkerOp.c_code`, whether it failed or not.

        This is a convenient place to clean up things allocated by `CLinkerOp.c_code`.

        Parameters
        ----------
        node : Apply
            WRITEME
        name : str
            A name that is automatically assigned and guaranteed to be
            unique.
        inputs : list of str
            There is a string for each input of the function, and the
            string is the name of a C variable pointing to that input.
            The type of the variable depends on the declared type of
            the input. There is a corresponding python variable that
            can be accessed by prepending "py_" to the name in the
            list.
        outputs : list of str
            Each string is the name of a C variable correspoinding to
            one of the outputs of the Op. The type depends on the
            declared type of the output. There is a corresponding
            python variable that can be accessed by prepending "py_" to
            the name in the list.
        sub : dict of str
            extra symbols defined in `CLinker` sub symbols (such as 'fail').
            WRITEME

        """
        return ""

    def c_support_code_apply(self, node: Apply, name: Text) -> Text:
        """Return `Apply`-specialized utility code for use by an `Op` that will be inserted at global scope.

        Parameters
        ----------
        node: Apply
            The node in the graph being compiled.
        name: str
            A string or number that serves to uniquely identify this node.
            Symbol names defined by this support code should include the name,
            so that they can be called from the `CLinkerOp.c_code`, and so that
            they do not cause name collisions.

        Notes
        -----
        This function is called in addition to `CLinkerObject.c_support_code`
        and will supplement whatever is returned from there.

        """
        return ""

    def c_init_code_apply(self, node: Apply, name: Text) -> Text:
        """Return a code string specific to the apply to be inserted in the module initialization code.

        Parameters
        ----------
        node : an Apply instance in the graph being compiled
        name : str
            A string or number that serves to uniquely identify this node.
            Symbol names defined by this support code should include the name,
            so that they can be called from the c_code, and so that they do not
            cause name collisions.

        Notes
        -----
        This function is called in addition to c_init_code and will supplement
        whatever is returned from there.

        """
        return ""

    def c_init_code_struct(self, node: Apply, name, sub) -> Text:
        """Return an `Apply`-specific code string to be inserted in the struct initialization code.

        Parameters
        ----------
        node: Apply
            The node in the graph being compiled.
        name: str
            A unique name to distinguish variables from those of other nodes.
        sub: dict of str
            A dictionary of values to substitute in the code.
            Most notably it contains a ``'fail'`` entry that you should place
            in your code after setting a Python exception to indicate an error.

        """
        return ""

    def c_support_code_struct(self, node: Apply, name: Text) -> Text:
        """Return `Apply`-specific utility code for use by an `Op` that will be inserted at struct scope.

        Parameters
        ----------
        node : Apply
            The node in the graph being compiled
        name : str
            A unique name to distinguish you variables from those of other
            nodes.

        """
        return ""

    def c_cleanup_code_struct(self, node: Apply, name: Text) -> Text:
        """Return an `Apply`-specific code string to be inserted in the struct cleanup code.

        Parameters
        ----------
        node : Apply
            The node in the graph being compiled
        name : str
            A unique name to distinguish variables from those of other nodes.

        """
        return ""


class CLinkerType(CLinkerObject):
    """
    Interface specification for Types that can be arguments to a `CLinkerOp`.

    A CLinkerType instance is mainly responsible  for providing the C code that
    interfaces python objects with a C `CLinkerOp` implementation.

    See WRITEME for a general overview of code generation by `CLinker`.

    """

    @abstractmethod
    def c_declare(
        self, name: Text, sub: Dict[Text, Text], check_input: bool = True
    ) -> Text:
        """Return C code to declare variables that will be instantiated by `CLinkerType.c_extract`.

        Parameters
        ----------
        name: str
            The name of the ``PyObject *`` pointer that will
            the value for this Type
        sub: dict string -> string
            a dictionary of special codes.  Most importantly
            sub['fail']. See CLinker for more info on `sub` and ``fail``.

        Notes
        -----
        It is important to include the `name` inside of variables which
        are declared here, so that name collisions do not occur in the
        source file that is generated.

        The variable called ``name`` is not necessarily defined yet
        where this code is inserted. This code might be inserted to
        create class variables for example, whereas the variable ``name``
        might only exist inside certain functions in that class.

        TODO: Why should variable declaration fail?  Is it even allowed to?

        Examples
        --------
        .. code-block: python

            def c_declare(self, name, sub, check_input=True):
                return "PyObject ** addr_of_%(name)s;"

        """

    @abstractmethod
    def c_init(self, name: Text, sub: Dict[Text, Text]) -> Text:
        """Return C code to initialize the variables that were declared by `CLinkerType.c_declare`.

        Notes
        -----
        The variable called ``name`` is not necessarily defined yet
        where this code is inserted. This code might be inserted in a
        class constructor for example, whereas the variable ``name``
        might only exist inside certain functions in that class.

        TODO: Why should variable initialization fail?  Is it even allowed to?

        Examples
        --------
        .. code-block: python

            def c_init(self, name, sub):
                return "addr_of_%(name)s = NULL;"

        """

    @abstractmethod
    def c_extract(
        self, name: Text, sub: Dict[Text, Text], check_input: bool = True, **kwargs
    ) -> Text:
        """Return C code to extract a ``PyObject *`` instance.

        The code returned from this function must be templated using
        ``%(name)s``, representing the name that the caller wants to
        call this `Variable`. The Python object ``self.data`` is in a
        variable called ``"py_%(name)s"`` and this code must set the
        variables declared by c_declare to something representative
        of ``py_%(name)``s. If the data is improper, set an appropriate
        exception and insert ``"%(fail)s"``.

        TODO: Point out that template filling (via sub) is now performed
        by this function. --jpt

        Parameters
        ----------
        name: str
            The name of the ``PyObject *`` pointer that will store the value
            for this type.
        sub: dict string -> string
            A dictionary of special codes. Most importantly
            ``sub['fail']``. See `CLinker` for more info on ``sub`` and
            ``fail``.

        Examples
        --------
        .. code-block: python

            def c_extract(self, name, sub, check_input=True, **kwargs):
                return "if (py_%(name)s == Py_None)" + \\\
                            addr_of_%(name)s = &py_%(name)s;" + \\\
                    "else" + \\\
                    { PyErr_SetString(PyExc_ValueError, \\\
                            'was expecting None'); %(fail)s;}"

        """

    @abstractmethod
    def c_sync(self, name: Text, sub: Dict[Text, Text]) -> Text:
        """Return C code to pack C types back into a ``PyObject``.

        The code returned from this function must be templated using
        ``"%(name)s"``, representing the name that the caller wants to
        call this `Variable`. The returned code may set ``"py_%(name)s"``
        to a ``PyObject*`` and that ``PyObject*`` will be accessible from
        Python via ``variable.data``. Do not forget to adjust reference
        counts if ``"py_%(name)s"`` is changed from its original value.

        Parameters
        ----------
        name : str
            WRITEME
        sub : dict of str
            WRITEME

        """

    def c_element_type(self) -> Text:
        """Return the name of the primitive C type of items into variables handled by this type.

        e.g:

         - For ``TensorType(dtype='int64', ...)``: should return ``"npy_int64"``.
         - For ``GpuArrayType(dtype='int32', ...)``: should return ``"ga_int"``.

        """
        return ""

    def c_is_simple(self) -> bool:
        """Return ``True`` for small or builtin C types.

        A hint to tell the compiler that this type is a builtin C type or a
        small struct and that its memory footprint is negligible. Simple
        objects may be passed on the stack.

        """
        return False

    def c_literal(self, data: Constant) -> Text:
        """Provide a C literal string value for the specified `data`.

        Parameters
        ----------
        data : Constant
            The data to be converted into a C literal string.

        """
        return ""

    def c_extract_out(
        self, name: Text, sub: Dict[Text, Text], check_input: bool = True, **kwargs
    ) -> Text:
        """Return C code to extract a ``PyObject *`` instance.

        Unlike `CLinkerType.c_extract`, `CLinkerType.c_extract_out` has to
        accept ``Py_None``, meaning that the variable should be left
        uninitialized.

        """
        return """
        if (py_%(name)s == Py_None)
        {
            %(c_init_code)s
        }
        else
        {
            %(c_extract_code)s
        }
        """ % dict(
            name=name,
            c_init_code=self.c_init(name, sub),
            c_extract_code=self.c_extract(name, sub, check_input),
        )

    def c_cleanup(self, name: Text, sub: Dict[Text, Text]) -> Text:
        """Return C code to clean up after `CLinkerType.c_extract`.

        This returns C code that should deallocate whatever
        `CLinkerType.c_extract` allocated or decrease the reference counts. Do
        not decrease ``py_%(name)s``'s reference count.

        Parameters
        ----------
        name : str
            WRITEME
        sub : dict of str
            WRITEME

        """
        return ""

    def c_code_cache_version(self) -> Union[Tuple, Tuple[int]]:
        """Return a tuple of integers indicating the version of this type.

        An empty tuple indicates an 'unversioned' type that will not
        be cached between processes.

        The cache mechanism may erase cached modules that have been
        superceded by newer versions. See `ModuleCache` for details.

        """
        return ()


class HideC(CLinkerOp):
    def __hide(*args):
        raise MethodNotDefined()

    c_code: Callable = __hide
    c_code_cleanup: Callable = __hide

    c_headers: Callable = __hide
    c_header_dirs: Callable = __hide
    c_libraries: Callable = __hide
    c_lib_dirs: Callable = __hide

    c_support_code: Callable = __hide
    c_support_code_apply: Callable = __hide

    c_compile_args: Callable = __hide
    c_no_compile_args: Callable = __hide
    c_init_code: Callable = __hide
    c_init_code_apply: Callable = __hide

    c_init_code_struct: Callable = __hide
    c_support_code_struct: Callable = __hide
    c_cleanup_code_struct: Callable = __hide

    def c_code_cache_version(self):
        return ()

    def c_code_cache_version_apply(self, node):
        return self.c_code_cache_version()
