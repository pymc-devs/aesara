import logging
from functools import partial
from typing import Tuple, Union

import numpy as np

from aesara import scalar as aes
from aesara.gradient import DisconnectedType
from aesara.graph.basic import Apply
from aesara.graph.op import Op
from aesara.tensor import basic as aet
from aesara.tensor import math as tm
from aesara.tensor.basic import as_tensor_variable, extract_diag
from aesara.tensor.type import dvector, lscalar, matrix, scalar, vector


logger = logging.getLogger(__name__)


class MatrixPinv(Op):
    """Computes the pseudo-inverse of a matrix :math:`A`.

    The pseudo-inverse of a matrix :math:`A`, denoted :math:`A^+`, is
    defined as: "the matrix that 'solves' [the least-squares problem]
    :math:`Ax = b`," i.e., if :math:`\\bar{x}` is said solution, then
    :math:`A^+` is that matrix such that :math:`\\bar{x} = A^+b`.

    Note that :math:`Ax=AA^+b`, so :math:`AA^+` is close to the identity matrix.
    This method is not faster than `matrix_inverse`. Its strength comes from
    that it works for non-square matrices.
    If you have a square matrix though, `matrix_inverse` can be both more
    exact and faster to compute. Also this op does not get optimized into a
    solve op.

    """

    __props__ = ()

    def __init__(self):
        pass

    def make_node(self, x):
        x = as_tensor_variable(x)
        assert x.ndim == 2
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (z,) = outputs
        z[0] = np.linalg.pinv(x).astype(x.dtype)

    def L_op(self, inputs, outputs, g_outputs):
        r"""The gradient function should return

            .. math:: V\frac{\partial X^+}{\partial X},

        where :math:`V` corresponds to ``g_outputs`` and :math:`X` to
        ``inputs``. According to `Wikipedia
        <https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_pseudoinverse#Derivative>`_,
        this corresponds to

            .. math:: (-X^+ V^T X^+ + X^+ X^{+T} V (I - X X^+) + (I - X^+ X) V X^{+T} X^+)^T.
        """
        (x,) = inputs
        (z,) = outputs
        (gz,) = g_outputs

        x_dot_z = tm.dot(x, z)
        z_dot_x = tm.dot(z, x)

        grad = (
            -matrix_dot(z, gz.T, z)
            + matrix_dot(z, z.T, gz, (aet.identity_like(x_dot_z) - x_dot_z))
            + matrix_dot((aet.identity_like(z_dot_x) - z_dot_x), gz, z.T, z)
        ).T
        return [grad]


pinv = MatrixPinv()


class MatrixInverse(Op):
    r"""Computes the inverse of a matrix :math:`A`.

    Given a square matrix :math:`A`, ``matrix_inverse`` returns a square
    matrix :math:`A_{inv}` such that the dot product :math:`A \cdot A_{inv}`
    and :math:`A_{inv} \cdot A` equals the identity matrix :math:`I`.

    Notes
    -----
    When possible, the call to this op will be optimized to the call
    of ``solve``.

    """

    __props__ = ()

    def __init__(self):
        pass

    def make_node(self, x):
        x = as_tensor_variable(x)
        assert x.ndim == 2
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (z,) = outputs
        z[0] = np.linalg.inv(x).astype(x.dtype)

    def grad(self, inputs, g_outputs):
        r"""The gradient function should return

            .. math:: V\frac{\partial X^{-1}}{\partial X},

        where :math:`V` corresponds to ``g_outputs`` and :math:`X` to
        ``inputs``. Using the `matrix cookbook
        <http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=3274>`_,
        one can deduce that the relation corresponds to

            .. math:: (X^{-1} \cdot V^{T} \cdot X^{-1})^T.

        """
        (x,) = inputs
        xi = self(x)
        (gz,) = g_outputs
        # tm.dot(gz.T,xi)
        return [-matrix_dot(xi, gz.T, xi).T]

    def R_op(self, inputs, eval_points):
        r"""The gradient function should return

            .. math:: \frac{\partial X^{-1}}{\partial X}V,

        where :math:`V` corresponds to ``g_outputs`` and :math:`X` to
        ``inputs``. Using the `matrix cookbook
        <http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=3274>`_,
        one can deduce that the relation corresponds to

            .. math:: X^{-1} \cdot V \cdot X^{-1}.

        """
        (x,) = inputs
        xi = self(x)
        (ev,) = eval_points
        if ev is None:
            return [None]
        return [-matrix_dot(xi, ev, xi)]

    def infer_shape(self, fgraph, node, shapes):
        return shapes


matrix_inverse = MatrixInverse()


def matrix_dot(*args):
    r"""Shorthand for product between several dots.

    Given :math:`N` matrices :math:`A_0, A_1, .., A_N`, ``matrix_dot`` will
    generate the matrix product between all in the given order, namely
    :math:`A_0 \cdot A_1 \cdot A_2 \cdot .. \cdot A_N`.

    """
    rval = args[0]
    for a in args[1:]:
        rval = tm.dot(rval, a)
    return rval


def trace(X):
    """
    Returns the sum of diagonal elements of matrix X.

    Notes
    -----
    Works on GPU since 0.6rc4.

    """
    return extract_diag(X).sum()


class Det(Op):
    """
    Matrix determinant. Input should be a square matrix.

    """

    __props__ = ()

    def make_node(self, x):
        x = as_tensor_variable(x)
        assert x.ndim == 2
        o = scalar(dtype=x.dtype)
        return Apply(self, [x], [o])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (z,) = outputs
        try:
            z[0] = np.asarray(np.linalg.det(x), dtype=x.dtype)
        except Exception:
            print("Failed to compute determinant", x)
            raise

    def grad(self, inputs, g_outputs):
        (gz,) = g_outputs
        (x,) = inputs
        return [gz * self(x) * matrix_inverse(x).T]

    def infer_shape(self, fgraph, node, shapes):
        return [()]

    def __str__(self):
        return "Det"


det = Det()


class Eig(Op):
    """
    Compute the eigenvalues and right eigenvectors of a square array.

    """

    _numop = staticmethod(np.linalg.eig)
    __props__: Union[Tuple, Tuple[str]] = ()

    def make_node(self, x):
        x = as_tensor_variable(x)
        assert x.ndim == 2
        w = vector(dtype=x.dtype)
        v = matrix(dtype=x.dtype)
        return Apply(self, [x], [w, v])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (w, v) = outputs
        w[0], v[0] = [z.astype(x.dtype) for z in self._numop(x)]

    def infer_shape(self, fgraph, node, shapes):
        n = shapes[0][0]
        return [(n,), (n, n)]


eig = Eig()


class Eigh(Eig):
    """
    Return the eigenvalues and eigenvectors of a Hermitian or symmetric matrix.

    """

    _numop = staticmethod(np.linalg.eigh)
    __props__ = ("UPLO",)

    def __init__(self, UPLO="L"):
        assert UPLO in ["L", "U"]
        self.UPLO = UPLO

    def make_node(self, x):
        x = as_tensor_variable(x)
        assert x.ndim == 2
        # Numpy's linalg.eigh may return either double or single
        # presision eigenvalues depending on installed version of
        # LAPACK.  Rather than trying to reproduce the (rather
        # involved) logic, we just probe linalg.eigh with a trivial
        # input.
        w_dtype = self._numop([[np.dtype(x.dtype).type()]])[0].dtype.name
        w = vector(dtype=w_dtype)
        v = matrix(dtype=w_dtype)
        return Apply(self, [x], [w, v])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (w, v) = outputs
        w[0], v[0] = self._numop(x, self.UPLO)

    def grad(self, inputs, g_outputs):
        r"""The gradient function should return

           .. math:: \sum_n\left(W_n\frac{\partial\,w_n}
                           {\partial a_{ij}} +
                     \sum_k V_{nk}\frac{\partial\,v_{nk}}
                           {\partial a_{ij}}\right),

        where [:math:`W`, :math:`V`] corresponds to ``g_outputs``,
        :math:`a` to ``inputs``, and  :math:`(w, v)=\mbox{eig}(a)`.

        Analytic formulae for eigensystem gradients are well-known in
        perturbation theory:

           .. math:: \frac{\partial\,w_n}
                          {\partial a_{ij}} = v_{in}\,v_{jn}


           .. math:: \frac{\partial\,v_{kn}}
                          {\partial a_{ij}} =
                \sum_{m\ne n}\frac{v_{km}v_{jn}}{w_n-w_m}

        """
        (x,) = inputs
        w, v = self(x)
        # Replace gradients wrt disconnected variables with
        # zeros. This is a work-around for issue #1063.
        gw, gv = _zero_disconnected([w, v], g_outputs)
        return [EighGrad(self.UPLO)(x, w, v, gw, gv)]


def _zero_disconnected(outputs, grads):
    l = []
    for o, g in zip(outputs, grads):
        if isinstance(g.type, DisconnectedType):
            l.append(o.zeros_like())
        else:
            l.append(g)
    return l


class EighGrad(Op):
    """
    Gradient of an eigensystem of a Hermitian matrix.

    """

    __props__ = ("UPLO",)

    def __init__(self, UPLO="L"):
        assert UPLO in ["L", "U"]
        self.UPLO = UPLO
        if UPLO == "L":
            self.tri0 = np.tril
            self.tri1 = partial(np.triu, k=1)
        else:
            self.tri0 = np.triu
            self.tri1 = partial(np.tril, k=-1)

    def make_node(self, x, w, v, gw, gv):
        x, w, v, gw, gv = map(as_tensor_variable, (x, w, v, gw, gv))
        assert x.ndim == 2
        assert w.ndim == 1
        assert v.ndim == 2
        assert gw.ndim == 1
        assert gv.ndim == 2
        out_dtype = aes.upcast(x.dtype, w.dtype, v.dtype, gw.dtype, gv.dtype)
        out = matrix(dtype=out_dtype)
        return Apply(self, [x, w, v, gw, gv], [out])

    def perform(self, node, inputs, outputs):
        """
        Implements the "reverse-mode" gradient for the eigensystem of
        a square matrix.

        """
        x, w, v, W, V = inputs
        N = x.shape[0]
        outer = np.outer

        def G(n):
            return sum(
                v[:, m] * V.T[n].dot(v[:, m]) / (w[n] - w[m])
                for m in range(N)
                if m != n
            )

        g = sum(outer(v[:, n], v[:, n] * W[n] + G(n)) for n in range(N))

        # Numpy's eigh(a, 'L') (eigh(a, 'U')) is a function of tril(a)
        # (triu(a)) only.  This means that partial derivative of
        # eigh(a, 'L') (eigh(a, 'U')) with respect to a[i,j] is zero
        # for i < j (i > j).  At the same time, non-zero components of
        # the gradient must account for the fact that variation of the
        # opposite triangle contributes to variation of two elements
        # of Hermitian (symmetric) matrix. The following line
        # implements the necessary logic.
        out = self.tri0(g) + self.tri1(g).T

        # Make sure we return the right dtype even if NumPy performed
        # upcasting in self.tri0.
        outputs[0][0] = np.asarray(out, dtype=node.outputs[0].dtype)

    def infer_shape(self, fgraph, node, shapes):
        return [shapes[0]]


def eigh(a, UPLO="L"):
    return Eigh(UPLO)(a)


class QRFull(Op):
    """
    Full QR Decomposition.

    Computes the QR decomposition of a matrix.
    Factor the matrix a as qr, where q is orthonormal
    and r is upper-triangular.

    """

    _numop = staticmethod(np.linalg.qr)
    __props__ = ("mode",)

    def __init__(self, mode):
        self.mode = mode

    def make_node(self, x):
        x = as_tensor_variable(x)

        assert x.ndim == 2, "The input of qr function should be a matrix."

        in_dtype = x.type.numpy_dtype
        out_dtype = np.dtype(f"f{in_dtype.itemsize}")

        q = matrix(dtype=out_dtype)

        if self.mode != "raw":
            r = matrix(dtype=out_dtype)
        else:
            r = vector(dtype=out_dtype)

        if self.mode != "r":
            q = matrix(dtype=out_dtype)
            outputs = [q, r]
        else:
            outputs = [r]

        return Apply(self, [x], outputs)

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        assert x.ndim == 2, "The input of qr function should be a matrix."
        res = self._numop(x, self.mode)
        if self.mode != "r":
            outputs[0][0], outputs[1][0] = res
        else:
            outputs[0][0] = res


def qr(a, mode="reduced"):
    """
    Computes the QR decomposition of a matrix.
    Factor the matrix a as qr, where q
    is orthonormal and r is upper-triangular.

    Parameters
    ----------
    a : array_like, shape (M, N)
        Matrix to be factored.

    mode : {'reduced', 'complete', 'r', 'raw'}, optional
        If K = min(M, N), then

        'reduced'
          returns q, r with dimensions (M, K), (K, N)

        'complete'
           returns q, r with dimensions (M, M), (M, N)

        'r'
          returns r only with dimensions (K, N)

        'raw'
          returns h, tau with dimensions (N, M), (K,)

        Note that array h returned in 'raw' mode is
        transposed for calling Fortran.

        Default mode is 'reduced'

    Returns
    -------
    q : matrix of float or complex, optional
        A matrix with orthonormal columns. When mode = 'complete' the
        result is an orthogonal/unitary matrix depending on whether or
        not a is real/complex. The determinant may be either +/- 1 in
        that case.
    r : matrix of float or complex, optional
        The upper-triangular matrix.

    """
    return QRFull(mode)(a)


class SVD(Op):
    """

    Parameters
    ----------
    full_matrices : bool, optional
        If True (default), u and v have the shapes (M, M) and (N, N),
        respectively.
        Otherwise, the shapes are (M, K) and (K, N), respectively,
        where K = min(M, N).
    compute_uv : bool, optional
        Whether or not to compute u and v in addition to s.
        True by default.

    """

    # See doc in the docstring of the function just after this class.
    _numop = staticmethod(np.linalg.svd)
    __props__ = ("full_matrices", "compute_uv")

    def __init__(self, full_matrices=True, compute_uv=True):
        self.full_matrices = full_matrices
        self.compute_uv = compute_uv

    def make_node(self, x):
        x = as_tensor_variable(x)
        assert x.ndim == 2, "The input of svd function should be a matrix."

        in_dtype = x.type.numpy_dtype
        out_dtype = np.dtype(f"f{in_dtype.itemsize}")

        s = vector(dtype=out_dtype)

        if self.compute_uv:
            u = matrix(dtype=out_dtype)
            vt = matrix(dtype=out_dtype)
            return Apply(self, [x], [u, s, vt])
        else:
            return Apply(self, [x], [s])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        assert x.ndim == 2, "The input of svd function should be a matrix."
        if self.compute_uv:
            u, s, vt = outputs
            u[0], s[0], vt[0] = self._numop(x, self.full_matrices, self.compute_uv)
        else:
            (s,) = outputs
            s[0] = self._numop(x, self.full_matrices, self.compute_uv)

    def infer_shape(self, fgraph, node, shapes):
        (x_shape,) = shapes
        M, N = x_shape
        K = tm.minimum(M, N)
        s_shape = (K,)
        if self.compute_uv:
            u_shape = (M, M) if self.full_matrices else (M, K)
            vt_shape = (N, N) if self.full_matrices else (K, N)
            return [u_shape, s_shape, vt_shape]
        else:
            return [s_shape]


def svd(a, full_matrices=1, compute_uv=1):
    """
    This function performs the SVD on CPU.

    Parameters
    ----------
    full_matrices : bool, optional
        If True (default), u and v have the shapes (M, M) and (N, N),
        respectively.
        Otherwise, the shapes are (M, K) and (K, N), respectively,
        where K = min(M, N).
    compute_uv : bool, optional
        Whether or not to compute u and v in addition to s.
        True by default.

    Returns
    -------
    U, V,  D : matrices

    """
    return SVD(full_matrices, compute_uv)(a)


class lstsq(Op):

    __props__ = ()

    def make_node(self, x, y, rcond):
        x = as_tensor_variable(x)
        y = as_tensor_variable(y)
        rcond = as_tensor_variable(rcond)
        return Apply(
            self,
            [x, y, rcond],
            [
                matrix(),
                dvector(),
                lscalar(),
                dvector(),
            ],
        )

    def perform(self, node, inputs, outputs):
        zz = np.linalg.lstsq(inputs[0], inputs[1], inputs[2])
        outputs[0][0] = zz[0]
        outputs[1][0] = zz[1]
        outputs[2][0] = np.array(zz[2])
        outputs[3][0] = zz[3]


def matrix_power(M, n):
    r"""Raise a square matrix, ``M``, to the (integer) power ``n``.

    This implementation uses exponentiation by squaring which is
    significantly faster than the naive implementation.
    The time complexity for exponentiation by squaring is
    :math: `\mathcal{O}((n \log M)^k)`

    Parameters
    ----------
    M: TensorVariable
    n: int

    """
    if n < 0:
        M = pinv(M)
        n = abs(n)

    # Shortcuts when 0 < n <= 3
    if n == 0:
        return aet.eye(M.shape[-2])

    elif n == 1:
        return M

    elif n == 2:
        return tm.dot(M, M)

    elif n == 3:
        return tm.dot(tm.dot(M, M), M)

    result = z = None

    while n > 0:
        z = M if z is None else tm.dot(z, z)
        n, bit = divmod(n, 2)
        if bit:
            result = z if result is None else tm.dot(result, z)

    return result


def norm(x, ord):
    x = as_tensor_variable(x)
    ndim = x.ndim
    if ndim == 0:
        raise ValueError("'axis' entry is out of bounds.")
    elif ndim == 1:
        if ord is None:
            return tm.sum(x ** 2) ** 0.5
        elif ord == "inf":
            return tm.max(abs(x))
        elif ord == "-inf":
            return tm.min(abs(x))
        elif ord == 0:
            return x[x.nonzero()].shape[0]
        else:
            try:
                z = tm.sum(abs(x ** ord)) ** (1.0 / ord)
            except TypeError:
                raise ValueError("Invalid norm order for vectors.")
            return z
    elif ndim == 2:
        if ord is None or ord == "fro":
            return tm.sum(abs(x ** 2)) ** (0.5)
        elif ord == "inf":
            return tm.max(tm.sum(abs(x), 1))
        elif ord == "-inf":
            return tm.min(tm.sum(abs(x), 1))
        elif ord == 1:
            return tm.max(tm.sum(abs(x), 0))
        elif ord == -1:
            return tm.min(tm.sum(abs(x), 0))
        else:
            raise ValueError(0)
    elif ndim > 2:
        raise NotImplementedError("We don't support norm with ndim > 2")


class TensorInv(Op):
    """
    Class wrapper for tensorinv() function;
    Aesara utilization of numpy.linalg.tensorinv;
    """

    _numop = staticmethod(np.linalg.tensorinv)
    __props__ = ("ind",)

    def __init__(self, ind=2):
        self.ind = ind

    def make_node(self, a):
        a = as_tensor_variable(a)
        out = a.type()
        return Apply(self, [a], [out])

    def perform(self, node, inputs, outputs):
        (a,) = inputs
        (x,) = outputs
        x[0] = self._numop(a, self.ind)

    def infer_shape(self, fgraph, node, shapes):
        sp = shapes[0][self.ind :] + shapes[0][: self.ind]
        return [sp]


def tensorinv(a, ind=2):
    """
    Does not run on GPU;
    Aesara utilization of numpy.linalg.tensorinv;

    Compute the 'inverse' of an N-dimensional array.
    The result is an inverse for `a` relative to the tensordot operation
    ``tensordot(a, b, ind)``, i. e., up to floating-point accuracy,
    ``tensordot(tensorinv(a), a, ind)`` is the "identity" tensor for the
    tensordot operation.

    Parameters
    ----------
    a : array_like
        Tensor to 'invert'. Its shape must be 'square', i. e.,
        ``prod(a.shape[:ind]) == prod(a.shape[ind:])``.
    ind : int, optional
        Number of first indices that are involved in the inverse sum.
        Must be a positive integer, default is 2.

    Returns
    -------
    b : ndarray
        `a`'s tensordot inverse, shape ``a.shape[ind:] + a.shape[:ind]``.

    Raises
    ------
    LinAlgError
        If `a` is singular or not 'square' (in the above sense).
    """
    return TensorInv(ind)(a)


class TensorSolve(Op):
    """
    Aesara utilization of numpy.linalg.tensorsolve
    Class wrapper for tensorsolve function.

    """

    _numop = staticmethod(np.linalg.tensorsolve)
    __props__ = ("axes",)

    def __init__(self, axes=None):
        self.axes = axes

    def make_node(self, a, b):
        a = as_tensor_variable(a)
        b = as_tensor_variable(b)
        out_dtype = aes.upcast(a.dtype, b.dtype)
        x = matrix(dtype=out_dtype)
        return Apply(self, [a, b], [x])

    def perform(self, node, inputs, outputs):
        (
            a,
            b,
        ) = inputs
        (x,) = outputs
        x[0] = self._numop(a, b, self.axes)


def tensorsolve(a, b, axes=None):
    """
    Aesara utilization of numpy.linalg.tensorsolve. Does not run on GPU!

    Solve the tensor equation ``a x = b`` for x.
    It is assumed that all indices of `x` are summed over in the product,
    together with the rightmost indices of `a`, as is done in, for example,
    ``tensordot(a, x, axes=len(b.shape))``.

    Parameters
    ----------
    a : array_like
        Coefficient tensor, of shape ``b.shape + Q``. `Q`, a tuple, equals
        the shape of that sub-tensor of `a` consisting of the appropriate
        number of its rightmost indices, and must be such that
        ``prod(Q) == prod(b.shape)`` (in which sense `a` is said to be
        'square').
    b : array_like
        Right-hand tensor, which can be of any shape.
    axes : tuple of ints, optional
        Axes in `a` to reorder to the right, before inversion.
        If None (default), no reordering is done.
    Returns
    -------
    x : ndarray, shape Q
    Raises
    ------
    LinAlgError
        If `a` is singular or not 'square' (in the above sense).
    """

    return TensorSolve(axes)(a, b)
