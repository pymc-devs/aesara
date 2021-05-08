import warnings
from typing import Any, Dict

import numpy as np
from typing_extensions import Protocol

import aesara
from aesara.gradient import DisconnectedType
from aesara.graph.basic import Apply, Variable
from aesara.graph.op import COp
from aesara.graph.params_type import ParamsType
from aesara.misc.safe_asarray import _asarray
from aesara.scalar import int32
from aesara.tensor import basic as aet
from aesara.tensor.exceptions import NotScalarConstantError
from aesara.tensor.type import TensorType, int_dtypes, tensor
from aesara.tensor.var import TensorConstant, TensorVariable


def register_shape_c_code(type, code, version=()):
    """
    Tell Shape Op how to generate C code for an Aesara Type.

    Parameters
    ----------
    typ : Aesara type
        It must be the Aesara class itself and not an instance of the class.
    code : C code
        Returns a vector representing the shape for the Aesara type 'typ'.
        Use %(iname)s and %(oname)s for the input and output C variable names
        respectively.
    version
        A number indicating the version of the code, for cache.

    """
    Shape.c_code_and_version[type] = (code, version)


class Shape(COp):
    """
    L{Op} to return the shape of a matrix.

    Notes
    -----
    Non-differentiable.

    """

    _f16_ok = True

    # Mapping from Type to C code (and version) to use.
    # In the C code, the name of the input variable is %(iname)s,
    # the output variable is %(oname)s.
    c_code_and_version: Dict = {}

    check_input = False
    __props__ = ()

    def make_node(self, x):
        # Must work for all type that have a shape attribute.
        # This will fail at execution time.
        if not isinstance(x, Variable):
            x = aet.as_tensor_variable(x)
        return Apply(self, [x], [aesara.tensor.type.lvector()])

    def perform(self, node, inp, out_):
        (x,) = inp
        (out,) = out_
        out[0] = _asarray(np.shape(x), dtype="int64")

    def infer_shape(self, fgraph, node, in_shapes):
        return [[len(in_shapes[0])]]

    def connection_pattern(self, node):
        # the grad returns the gradient with respect to the
        # elements of a tensor variable
        # the elements of the tensor variable do not participate
        # in the computation of the shape, so they are not really
        # part of the graph
        return [[False]]

    def grad(self, inp, grads):
        # the grad returns the gradient with respect to the
        # elements of a tensor variable
        # the elements of the tensor variable do not participate
        # in the computation of the shape, so they are not really
        # part of the graph
        return [aesara.gradient.DisconnectedType()()]

    def R_op(self, inputs, eval_points):
        return [None]

    def c_code(self, node, name, inames, onames, sub):
        (iname,) = inames
        (oname,) = onames
        fail = sub["fail"]

        itype = node.inputs[0].type.__class__
        if itype in self.c_code_and_version:
            code, version = self.c_code_and_version[itype]
            return code % locals()

        # Else, no C code
        raise NotImplementedError()

    def c_code_cache_version(self):
        version = []
        # If any of the c code is unversionned, we have to return ()
        # Else, we will return a list of (type name, version) pairs.
        for t, (c, v) in sorted(
            self.c_code_and_version.items(), key=lambda pair: str(pair[0])
        ):
            if not v:
                warnings.warn(
                    f"Type {t} has C code for Shape, but it has no "
                    "version. You should add a 'version' keyword "
                    "arg when calling register_shape_c_code.",
                    stacklevel=2,
                )
                return ()
            version.append((str(t), v))

        if version:
            version.append(1)

        return tuple(version)


shape = Shape()
_shape = shape  # was used in the past, now use shape directly.


class Shape_i(COp):
    """
    L{Op} to return the shape of a matrix.

    Notes
    -----
    Non-differentiable.

    """

    _f16_ok = True

    # Mapping from Type to C code (and version) to use.
    # In the C code, the name of the input variable is %(iname)s,
    # the output variable is %(oname)s.
    c_code_and_version: Dict = {}

    check_input = False

    __props__ = ("i",)

    def __init__(self, i):
        # As i will be used in the hash and that ndarray are not hashable,
        # we need to convert it to an int as it is hashable.
        if isinstance(i, np.ndarray):
            assert i.dtype in aesara.tensor.type.integer_dtypes
        assert i == int(i)
        i = int(i)
        self.i = i

    # NB:
    # 1) params_type is defined as a property to avoid
    #    loop in Python import caused by importing aesara.scalar below
    #    when params_type is defined directly in class code.
    # 2) We wrap scalar into ParamsType (instead of directly using scalar as op param)
    #    to avoid Aesara converting scalar param to constant that would be later
    #    hardcoded as litteral in C code, making us loose all the advantages of
    #    using params.
    @property
    def params_type(self):
        return ParamsType(i=aesara.scalar.basic.int64)

    def __str__(self):
        return "%s{%i}" % (self.__class__.__name__, self.i)

    def make_node(self, x):
        if not isinstance(x, Variable):
            raise TypeError("x must be Variable with ndim attribute", x)
        if x.ndim <= self.i:
            raise TypeError("x has too few dimensions for Shape_i", (x, self.i))
        return Apply(self, [x], [aesara.tensor.type.lscalar()])

    def perform(self, node, inp, out_, params):
        (x,) = inp
        (out,) = out_
        if out[0] is None:
            out[0] = _asarray(np.shape(x)[self.i], dtype="int64")
        else:
            out[0][...] = np.shape(x)[self.i]

    def c_code_cache_version(self):
        version = []
        # If any of the c code is unversionned, we have to return ()
        # Else, we will return a list of (type name, version) pairs.
        for t, (c, ci, v) in sorted(
            self.c_code_and_version.items(), key=lambda pair: str(pair[0])
        ):
            if not v:
                warnings.warn(
                    f"Type {t} has C code for Shape_i, but it has "
                    "no version. You should add a 'version' keyword "
                    "arg when calling register_shape_i_c_code.",
                    stacklevel=2,
                )
                return ()
            version.append((str(t), v))

        if version:
            version.append(2)

        return tuple(version)

    def c_code(self, node, name, inames, onames, sub):
        (iname,) = inames
        (oname,) = onames
        fail = sub["fail"]
        # i is then 'params->i', not just 'params'.
        i = sub["params"] + "->i"

        itype = node.inputs[0].type.__class__
        if itype in self.c_code_and_version:
            code, check_input, version = self.c_code_and_version[itype]
            return (check_input + code) % locals()

        # Else, no C code
        raise NotImplementedError()

    def infer_shape(self, fgraph, node, input_shapes):
        return [()]

    def connection_pattern(self, node):
        # the grad returns the gradient with respect to the
        # elements of a tensor variable
        # the elements of the tensor variable do not participate
        # in the computation of the shape, so they are not really
        # part of the graph
        return [[False]]

    def grad(self, inp, grads):
        return [
            aesara.gradient.grad_not_implemented(
                op=self,
                x_pos=0,
                x=inp[0],
                comment=("No gradient for the shape of a matrix " "is implemented."),
            )
        ]


class Shape_i_decorator(Protocol):
    cache: Dict


def shape_i_decorator(func: Any) -> Shape_i_decorator:
    return func


@shape_i_decorator
def shape_i(var, i, fgraph=None):
    """
    Equivalent of var.shape[i], but apply if possible the shape feature
    optimization.

    This is useful in optimization that need to get the shape. This
    remove the need of the following shape_feature optimization that
    convert it. So this speed up optimization and remove Equilibrium
    max iteration problems.

    Parameters
    ----------
    var : Variable
        The variable we want to take the shape of.
    i : int
        The shape dimensions we want
    fgraph : FunctionGraph (optional)

    """
    if fgraph and hasattr(fgraph, "shape_feature"):
        shape_feature = fgraph.shape_feature
        shape_of = shape_feature.shape_of

        def recur(node):
            if not node.outputs[0] in shape_of:
                for inp in node.inputs:
                    if inp.owner:
                        recur(inp.owner)
                # If the output var isn't marked as being in the graph,
                # we need to add it in the ShapeFeature.
                shape_feature.on_import(fgraph, node, "graph.ops.shape_i")

        if var not in shape_of:
            recur(var.owner)
        return shape_of[var][i]

    # If we are not able to use the shape feature, we should not put
    # Shape_i in the graph. Otherwise, the shape feature optimization
    # won't get applied.
    return shape(var)[i]


def shape_i_op(i):
    key = i
    if key not in shape_i_op.cache:
        shape_i_op.cache[key] = Shape_i(i)
    return shape_i_op.cache[key]


shape_i_op.cache = {}


def register_shape_i_c_code(typ, code, check_input, version=()):
    """
    Tell Shape_i how to generate C code for an Aesara Type.

    Parameters
    ----------
    typ : Aesara type
        It must be the Aesara class itself and not an instance of the class.
    code : C code
        Gets the shape of dimensions %(i)s for the Aesara type 'typ'.
        Use %(iname)s and %(oname)s for the input and output C variable names
        respectively.
    version
        A number indicating the version of the code, for cache.

    """
    Shape_i.c_code_and_version[typ] = (code, check_input, version)


def register_specify_shape_c_code(typ, code, version=(), c_support_code_apply=None):
    """
    Tell SpecifyShape how to generate C code for an Aesara Type.

    Parameters
    ----------
    typ : Aesara type
        It must be the Aesara class itself and not an instance of the class.
    code : C code
        Checks the shape and returns a view for the Aesara type 'typ'.
        Use %(iname)s and %(oname)s for the input and output C variable names
        respectively. %(shape)s is the vector of shape of %(iname)s.
        Check that its length is good.
    version
        A number indicating the version of the code, for cache.
    c_support_code_apply
        Extra code.

    """
    SpecifyShape.c_code_and_version[typ] = (code, version, c_support_code_apply)


class SpecifyShape(COp):
    """
    L{Op} that puts into the graph the user-provided shape.

    In the case where this `Op` stays in the final graph, we assert the shape.
    For this the output of this op must be used in the graph. This is not
    the case most of the time if we only take the shape of the output.
    Maybe there are other optimizations that will mess with this.

    Notes
    -----
    Maybe in the future we will never do the assert!

    We currently don't support specifying partial shape information.

    TODO : test this op with sparse. Do C code for them too.

    """

    view_map = {0: [0]}
    # Mapping from Type to C code (and version) to use.
    # In the C code, the name of the input variable is %(iname)s,
    # the output variable is %(oname)s.
    c_code_and_version: Dict = {}
    __props__ = ()
    _f16_ok = True

    def make_node(self, x, shape):
        if not isinstance(x, Variable):
            x = aet.as_tensor_variable(x)
        if shape == () or shape == []:
            tshape = aet.constant([], dtype="int64")
        else:
            tshape = aet.as_tensor_variable(shape, ndim=1)
            if tshape.dtype not in aesara.tensor.type.integer_dtypes:
                raise AssertionError(
                    f"The `shape` must be an integer type. Got {tshape.dtype} instead."
                )
        if isinstance(tshape, TensorConstant) and tshape.data.size != x.ndim:
            ndim = len(tshape.data)
            raise AssertionError(
                f"Input `x` is {x.ndim}-dimensional and will never match a {ndim}-dimensional shape."
            )
        return Apply(self, [x, tshape], [x.type()])

    def perform(self, node, inp, out_):
        x, shape = inp
        (out,) = out_
        ndim = len(shape)
        if x.ndim != ndim:
            raise AssertionError(
                f"SpecifyShape: Got {x.ndim} dimensions (shape {x.shape}), expected {ndim} dimensions with shape {tuple(shape)}."
            )
        if not np.all(x.shape == shape):
            raise AssertionError(
                f"SpecifyShape: Got shape {x.shape}, expected {tuple(shape)}."
            )
        out[0] = x

    def infer_shape(self, fgraph, node, shapes):
        xshape, sshape = shapes
        new_shape = []
        for dim in range(node.inputs[0].ndim):
            try:
                s = aet.get_scalar_constant_value(node.inputs[1][dim])
                s = aet.as_tensor_variable(s)
                new_shape.append(s)
            except NotScalarConstantError:
                new_shape.append(node.inputs[1][dim])

        assert len(new_shape) == len(xshape)
        return [new_shape]

    def connection_pattern(self, node):
        return [[True], [False]]

    def grad(self, inp, grads):
        x, s = inp
        (gz,) = grads
        # Should I set an SpecifyShape on gz? I think so
        # But I don't do it now as we need to make an optimization
        # to remove that op from the graph to don't block other optimization
        # Should I do an optimizer that will remove the SpecifyShape?
        # I think Yes
        return [gz, aesara.gradient.DisconnectedType()()]
        return [specify_shape(gz, s), aesara.gradient.DisconnectedType()()]

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            # It means that the this op sits on top of a non-differentiable
            # path
            return [None]
        return self.make_node(eval_points[0], *inputs[1:]).outputs

    def c_support_code_apply(self, node, name):
        itype = node.inputs[0].type.__class__
        if itype in self.c_code_and_version:
            _, _, support_code = self.c_code_and_version[itype]
            if support_code:
                return support_code
        return super().c_support_code_apply(node, name)

    def c_code(self, node, name, inames, onames, sub):
        iname, shape = inames
        (oname,) = onames
        fail = sub["fail"]

        itype = node.inputs[0].type.__class__
        if itype in self.c_code_and_version:
            code, version, _ = self.c_code_and_version[itype]
            return code % locals()

        raise NotImplementedError()

    def c_code_cache_version(self):
        version = []
        # If any of the c code is unversionned, we have to return ()
        # Else, we will return a list of (type name, version) pairs.
        for t, (c, v, _) in sorted(
            self.c_code_and_version.items(), key=lambda pair: str(pair[0])
        ):
            if not v:
                warnings.warn(
                    "Type %s has C code for SpecifyShape, but it "
                    "has no version. You should add a 'version' "
                    "keyword arg when calling "
                    "register_specify_shape_c_code." % t,
                    stacklevel=2,
                )
                return ()
            version.append((str(t), v))

        return tuple(version)


specify_shape = SpecifyShape()


class Reshape(COp):
    """Perform a reshape operation of the input x to the new shape shp.
    The number of dimensions to which to reshape to (ndim) must be
    known at graph build time.
    """

    view_map = {0: [0]}  # output 0 is potentially aliased to inputs [0]
    _f16_ok = True

    check_input = False
    __props__ = ("ndim",)
    params_type = ParamsType(ndim=int32)
    # name does not participate because it doesn't affect computations

    def __init__(self, ndim, name=None):
        self.ndim = int(ndim)
        if ndim < 0:
            raise ValueError("The output dimensions after reshape must be 0 or greater")
        assert name is None, "name attribute for Reshape has been deprecated"

    def __str__(self):
        return f"{self.__class__.__name__}{{{self.ndim}}}"

    def make_node(self, x, shp):
        x = aet.as_tensor_variable(x)
        shp_orig = shp
        shp = aet.as_tensor_variable(shp, ndim=1)
        if not (
            shp.dtype in int_dtypes
            or (isinstance(shp, TensorConstant) and shp.data.size == 0)
        ):
            # It raises an error if shp is not of integer type,
            # except when shp is constant and empty
            # (in this case, shp.dtype does not matter anymore).
            raise TypeError("Shape must be integers", shp, shp.dtype)
        assert shp.ndim == 1
        if isinstance(shp, TensorConstant):
            bcast = [s == 1 for s in shp.data]
            return Apply(self, [x, shp], [tensor(x.type.dtype, bcast)])
        else:
            bcasts = [False] * self.ndim
            shp_list = shp_orig
            if hasattr(shp_orig, "ndim") and shp_orig.ndim == 0:
                shp_list = [shp_orig]
            for index in range(self.ndim):
                y = shp_list[index]
                y = aet.as_tensor_variable(y)
                # Try to see if we can infer that y has a constant value of 1.
                # If so, that dimension should be broadcastable.
                try:
                    bcasts[index] = (
                        hasattr(y, "get_scalar_constant_value")
                        and y.get_scalar_constant_value() == 1
                    )
                except NotScalarConstantError:
                    pass
            return Apply(self, [x, shp], [tensor(x.type.dtype, bcasts)])

    def perform(self, node, inp, out_, params):
        x, shp = inp
        (out,) = out_
        if len(shp) != self.ndim:
            raise ValueError(
                (
                    "shape argument to Reshape.perform has incorrect"
                    f" length {len(shp)}"
                    f", should be {self.ndim}"
                ),
                shp,
            )
        try:
            out[0] = np.reshape(x, shp)
        except Exception:
            raise ValueError(f"Cannot reshape input of shape {x.shape} to shape {shp}")

    def connection_pattern(self, node):
        return [[True], [False]]

    def grad(self, inp, grads):
        x, shp = inp
        (g_out,) = grads
        return [reshape(g_out, shape(x), ndim=x.ndim), DisconnectedType()()]

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return [None]
        return self(eval_points[0], *inputs[1:], **dict(return_list=True))

    def infer_shape(self, fgraph, node, ishapes):
        from aesara.tensor.math import eq, maximum, mul

        # inputs[1] can contain at most one value of '-1', meaning the actual
        # shape of the output will be automatically computed by reshape, so
        # that the total number of elements stays the same.
        # TODO: Maybe put that formula here?
        # It's not trivial, because we would have to check if the product of
        # all the non-minus-one shapes is a divisor of the product of the
        # original shapes.
        # The following expression leads to cycles in feature_shape,
        # because it tries to replace the Shape_i node by the switch
        # statement, which depends on Shape_i.
        # return [tuple([switch(eq(node.inputs[1][i], -1),
        #                      Shape_i(i)(node.outputs[0]),
        #                      node.inputs[1][i])
        #                    for i in range(self.ndim)]
        #    )]
        # Here, we only simplify if the shape (node.inputs[1]) is a constant,
        # ideally it would suffice to check that it is always non-negative.
        # If current variable is a scalar and its dimensionality should
        # change to self.ndim, then use size 1 for all new dimensions.
        if len(ishapes[0]) == 0:
            return [(1,) * self.ndim]

        requ = node.inputs[1]
        input_size = mul(*ishapes[0])
        if isinstance(requ, TensorConstant):
            requ = list(requ.data)
            requ_part = [ele for ele in requ if ele != -1]
            crit = len(requ) - len(requ_part)
            if crit == 1 and len(requ_part) > 0:
                # If there are both 0 and -1 in requ_size, it is impossible
                # to determine a right output, but we can at least prevent
                # a division by 0. We do not want to keep a negative
                # size here as it could lead to further weird errors
                # after other optimizations.
                requ_size = mul(*requ_part)
                missing = input_size // (1 if requ_size == 0 else requ_size)
                for i, ele in enumerate(requ):
                    if ele == -1:
                        requ[i] = missing
            elif crit == 1:  # we reshape to -1
                requ = [input_size] if ishapes[0] else [1]
            elif crit > 1:
                raise ValueError(
                    "shape argument to Reshape.perform"
                    " must have at most one entry equal to -1"
                )
            return [requ]
        else:

            requ = [requ[i] for i in range(self.ndim)]
            # since new_dims can have negative value (-1), the
            # multiplication of all values should be negated
            # to give a positive value.
            # To avoid optimization complexity, we avoid checking
            # for the case when there are two or more '-1' values.
            if self.ndim:
                requ_size = -mul(*requ)
                # If there are both 0 and -1 in requ_size, it is impossible
                # to determine a right output, but we can at least prevent
                # a division by 0. We do not want to keep a negative
                # size here as it could lead to further weird errors
                # after other optimizations.
                rest_size = input_size // maximum(requ_size, 1)
            return [
                tuple(
                    [
                        aet.switch(eq(requ[i], -1), rest_size, requ[i])
                        for i in range(self.ndim)
                    ]
                )
            ]

    def c_code_cache_version(self):
        return (8,)

    def c_code(self, node, name, inputs, outputs, sub):
        if isinstance(node.inputs[0], TensorVariable):
            x, shp = inputs
            (z,) = outputs
            sdtype = node.inputs[1].type.dtype_specs()[1]
            fail = sub["fail"]
            params = sub["params"]
            return (
                """
            assert (PyArray_NDIM(%(shp)s) == 1);
            npy_intp new_dims[%(params)s->ndim];
            PyArray_Dims newshape;
            newshape.ptr = new_dims;
            newshape.len = %(params)s->ndim;
            for (int ii = 0; ii < %(params)s->ndim; ++ii)
            {
                // -- We do not want an explicit cast here. the shp can be any
                // -- int* dtype. The compiler will explicitly upcast it, but
                // -- will err if this will downcast. This could happen if the
                // -- user pass an int64 dtype, but npy_intp endup being int32.
                new_dims[ii] = ((%(sdtype)s*)(
                        PyArray_BYTES(%(shp)s) +
                        ii * PyArray_STRIDES(%(shp)s)[0]))[0];
            }
            Py_XDECREF(%(z)s);
            %(z)s = (PyArrayObject *) PyArray_Newshape(%(x)s, &newshape, NPY_CORDER);
            if (!%(z)s)
            {
                //The error message should have been set by PyArray_Newshape
                %(fail)s;
            }
            """
                % locals()
            )
        else:
            raise NotImplementedError()


def reshape(x, newshape, ndim=None):
    if ndim is None:
        newshape = aet.as_tensor_variable(newshape)
        if newshape.ndim != 1:
            raise TypeError(
                "New shape in reshape must be a vector or a list/tuple of"
                f" scalar. Got {newshape} after conversion to a vector."
            )
        try:
            ndim = aet.get_vector_length(newshape)
        except ValueError:
            raise ValueError(
                f"The length of the provided shape ({newshape}) cannot "
                "be automatically determined, so Aesara is not able "
                "to know what the number of dimensions of the reshaped "
                "variable will be. You can provide the 'ndim' keyword "
                "argument to 'reshape' to avoid this problem."
            )
    op = Reshape(ndim)
    rval = op(x, newshape)
    return rval


def shape_padleft(t, n_ones=1):
    """Reshape `t` by left-padding the shape with `n_ones` 1s.

    See Also
    --------
    shape_padaxis
    shape_padright
    Dimshuffle

    """
    _t = aet.as_tensor_variable(t)

    pattern = ["x"] * n_ones + [i for i in range(_t.type.ndim)]
    return _t.dimshuffle(pattern)


def shape_padright(t, n_ones=1):
    """Reshape `t` by right-padding the shape with `n_ones` 1s.

    See Also
    --------
    shape_padaxis
    shape_padleft
    Dimshuffle

    """
    _t = aet.as_tensor_variable(t)

    pattern = [i for i in range(_t.type.ndim)] + ["x"] * n_ones
    return _t.dimshuffle(pattern)


def shape_padaxis(t, axis):
    """Reshape `t` by inserting 1 at the dimension `axis`.

    Examples
    --------
    >>> tensor = aesara.tensor.type.tensor3()
    >>> aesara.tensor.shape_padaxis(tensor, axis=0)
    DimShuffle{x,0,1,2}.0
    >>> aesara.tensor.shape_padaxis(tensor, axis=1)
    DimShuffle{0,x,1,2}.0
    >>> aesara.tensor.shape_padaxis(tensor, axis=3)
    DimShuffle{0,1,2,x}.0
    >>> aesara.tensor.shape_padaxis(tensor, axis=-1)
    DimShuffle{0,1,2,x}.0

    See Also
    --------
    shape_padleft
    shape_padright
    Dimshuffle

    """
    _t = aet.as_tensor_variable(t)

    ndim = _t.ndim + 1
    if not -ndim <= axis < ndim:
        msg = "axis {0} is out of bounds [-{1}, {1})".format(axis, ndim)
        raise IndexError(msg)
    if axis < 0:
        axis += ndim

    pattern = [i for i in range(_t.type.ndim)]
    pattern.insert(axis, "x")
    return _t.dimshuffle(pattern)


register_shape_c_code(
    TensorType,
    """
    npy_intp shape[] = {PyArray_NDIM(%(iname)s)};
    if(%(oname)s == NULL || (PyArray_DIMS(%(oname)s)[0] != shape[0]))
    {
        Py_XDECREF(%(oname)s);
        %(oname)s = (PyArrayObject*) PyArray_SimpleNew(1, shape, NPY_INT64);
    }
    for(int i=0;i<shape[0];i++)
    {
        ((npy_int64*)PyArray_GETPTR1(%(oname)s, i))[0] = PyArray_DIMS(%(iname)s)[i];
    }
    """,
    version=1,
)


register_shape_i_c_code(
    TensorType,
    """
    if(!%(oname)s)
        %(oname)s=(PyArrayObject*)PyArray_EMPTY(0, NULL, NPY_INT64, 0);
    ((npy_int64*)PyArray_DATA(%(oname)s))[0]=PyArray_DIMS(%(iname)s)[%(i)s];
    """,
    """
    if (%(i)s>=PyArray_NDIM(%(iname)s)){
        PyErr_SetString(PyExc_TypeError,
            "Number of dimensions lower than expected");
        %(fail)s
    }
    """,
    version=3,
)


register_specify_shape_c_code(
    TensorType,
    """
        if (PyArray_NDIM(%(iname)s) != PyArray_DIMS(%(shape)s)[0]) {
            PyErr_Format(PyExc_AssertionError,
                "SpecifyShape: Got %%d dimensions, expected %%d dimensions.",
                PyArray_NDIM(%(iname)s),
                PyArray_DIMS(%(shape)s)[0]
            );
            %(fail)s;
        }
        for(int i = 0; i < PyArray_NDIM(%(iname)s); i++){
            dtype_%(shape)s shp = ((dtype_%(shape)s*)PyArray_GETPTR1(%(shape)s,
                                                                     i))[0];
            if (PyArray_DIMS(%(iname)s)[i] != shp) {
                PyErr_Format(PyExc_AssertionError,
                             "SpecifyShape: dim %%d of input has shape %%d,"
                             " expected %%d.",
                             i, PyArray_DIMS(%(iname)s)[i],
                             shp);
                %(fail)s;
            }
        }
        Py_XDECREF(%(oname)s);
        %(oname)s = %(iname)s;
        Py_XINCREF(%(oname)s);
    """,
    version=1,
)
