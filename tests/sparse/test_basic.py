import time
from itertools import product

import numpy as np
import pytest
from packaging import version

import aesara
import aesara.tensor as aet
from aesara import sparse
from aesara.compile.function import function
from aesara.compile.io import In, Out
from aesara.configdefaults import config
from aesara.gradient import GradientError
from aesara.graph.basic import Apply, Constant, applys_between
from aesara.graph.op import Op
from aesara.misc.safe_asarray import _asarray
from aesara.sparse import (
    CSC,
    CSM,
    CSR,
    AddSD,
    AddSS,
    AddSSData,
    Cast,
    ConstructSparseFromList,
    CSMGrad,
    CSMProperties,
    DenseFromSparse,
    Diag,
    Dot,
    EnsureSortedIndices,
    GetItemScalar,
    HStack,
    MulSD,
    MulSS,
    Neg,
    Remove0,
    SamplingDot,
    SparseFromDense,
    SparseType,
    SquareDiagonal,
    StructuredDot,
    StructuredDotGradCSC,
    StructuredDotGradCSR,
    Transpose,
    TrueDot,
    Usmm,
    VStack,
    add,
    add_s_s_data,
    as_sparse_or_tensor_variable,
    as_sparse_variable,
    cast,
    clean,
    construct_sparse_from_list,
    csc_from_dense,
    csm_properties,
    csr_from_dense,
    dense_from_sparse,
    diag,
    ensure_sorted_indices,
    ge,
    gt,
    le,
    lt,
    mul,
    mul_s_v,
    sampling_dot,
    sp_ones_like,
    square_diagonal,
    structured_add,
    structured_add_s_v,
    structured_dot,
    structured_maximum,
    structured_minimum,
    transpose,
    true_dot,
)
from aesara.sparse.basic import (
    SparseConstant,
    _is_dense_variable,
    _is_sparse,
    _is_sparse_variable,
    _mtypes,
)
from aesara.sparse.opt import CSMGradC, StructuredDotCSC, UsmmCscDense
from aesara.tensor.basic import MakeVector
from aesara.tensor.elemwise import DimShuffle, Elemwise
from aesara.tensor.math import sum as aet_sum
from aesara.tensor.shape import Shape_i
from aesara.tensor.subtensor import AdvancedIncSubtensor1, AdvancedSubtensor1, Subtensor
from aesara.tensor.type import (
    TensorType,
    float_dtypes,
    fscalar,
    iscalar,
    ivector,
    lvector,
    matrix,
    scalar,
    tensor,
    vector,
)
from tests import unittest_tools as utt
from tests.tensor.test_sharedvar import makeSharedTester


sp = pytest.importorskip("scipy", minversion="0.7.0")


# Probability distributions are currently tested in test_sp2.py
# from aesara.sparse import (
#    Poisson, poisson, Binomial, Multinomial, multinomial)


def as_sparse_format(data, format):
    if format == "csc":
        return sp.sparse.csc_matrix(data)
    elif format == "csr":
        return sp.sparse.csr_matrix(data)
    else:
        raise NotImplementedError()


def eval_outputs(outputs):
    return function([], outputs)()[0]


# scipy 0.17 will return sparse values in all cases while previous
# version sometimes wouldn't.  This will make everything dense so that
# we can use assert_allclose.
def as_ndarray(val):
    if hasattr(val, "toarray"):
        return val.toarray()
    return val


def random_lil(shape, dtype, nnz):
    rval = sp.sparse.lil_matrix(shape, dtype=dtype)
    huge = 2 ** 30
    for k in range(nnz):
        # set non-zeros in random locations (row x, col y)
        idx = np.random.randint(1, huge + 1, size=2) % shape
        value = np.random.rand()
        # if dtype *int*, value will always be zeros!
        if dtype in sparse.integer_dtypes:
            value = int(value * 100)
        # The call to tuple is needed as scipy 0.13.1 do not support
        # ndarray with length 2 as idx tuple.
        rval.__setitem__(tuple(idx), value)
    return rval


def sparse_random_inputs(
    format,
    shape,
    n=1,
    out_dtype=None,
    p=0.5,
    gap=None,
    explicit_zero=False,
    unsorted_indices=False,
):
    """
    Return a tuple containing everything needed to perform a test.

    If `out_dtype` is `None`, aesara.config.floatX is used.

    :param format: Sparse format.
    :param shape: Shape of data.
    :param n: Number of variable.
    :param out_dtype: dtype of output.
    :param p: Sparsity proportion.
    :param gap: Tuple for the range of the random sample. When
                length is 1, it is assumed to be the exclusive
                max, when `gap` = (`a`, `b`) it provide a sample
                from [a, b[. If `None` is used, it provide [0, 1]
                for float dtypes and [0, 50[ for integer dtypes.
    :param explicit_zero: When True, we add explicit zero in the
                          returned sparse matrix
    :param unsorted_indices: when True, we make sure there is
                             unsorted indices in the returned
                             sparse matrix.
    :return: (variable, data) where both `variable` and `data` are list.

    :note: explicit_zero and unsorted_indices was added in Aesara 0.6rc4
    """

    if out_dtype is None:
        out_dtype = aesara.config.floatX

    assert 0 <= p <= 1
    assert len(shape) == 2
    assert out_dtype in sparse.all_dtypes
    assert gap is None or isinstance(gap, (tuple, list))
    if gap is not None and out_dtype.startswith("u"):
        assert gap[0] >= 0

    def _rand():
        where = np.random.binomial(1, p, size=shape).astype("int8")

        if out_dtype in sparse.discrete_dtypes:
            if not gap:
                value = np.random.randint(50, size=shape)
            elif len(gap) == 2:
                value = np.random.randint(gap[0], gap[1], size=shape)
            else:
                value = np.random.randint(gap[0], size=shape)
        else:
            if not gap:
                value = np.random.random(shape)
            elif len(gap) == 2:
                a, b = gap
                value = a + np.random.random(shape) * (b - a)
            else:
                value = np.random.random(shape) * gap[0]
        return (where * value).astype(out_dtype)

    variable = [
        getattr(aesara.sparse, format + "_matrix")(dtype=out_dtype) for k in range(n)
    ]
    data = [
        getattr(sp.sparse, format + "_matrix")(_rand(), dtype=out_dtype)
        for k in range(n)
    ]
    if unsorted_indices:
        for idx in range(n):
            d = data[idx]
            # these flip the matrix, but it's random anyway
            if format == "csr":
                d = sp.sparse.csr_matrix(
                    (d.data, d.shape[1] - 1 - d.indices, d.indptr), shape=d.shape
                )
            if format == "csc":
                d = sp.sparse.csc_matrix(
                    (d.data, d.shape[0] - 1 - d.indices, d.indptr), shape=d.shape
                )
            assert not d.has_sorted_indices
            data[idx] = d
    if explicit_zero:
        for idx in range(n):
            assert data[idx].nnz > 1, "can't make a sparse matrix with explicit 0"
            d_idx = np.random.randint(data[idx].nnz)
            data[idx].data[d_idx] = 0

    # numpy 1.5.0 with scipy 0.9.0 have sp.sparse.XXX_matrix return
    # typenum 10(ulonglong) instead of 8(uint64) event if they are the same!
    # Aesara don't like ulonglong type_num
    dtype = np.dtype(out_dtype)  # Convert into dtype object.
    if data[0].dtype.num != dtype.num and dtype.str == data[0].dtype.str:
        data[0].data = _asarray(data[0].data, out_dtype)
    assert data[0].dtype.num == dtype.num
    return (variable, data)


def verify_grad_sparse(op, pt, structured=False, *args, **kwargs):
    """
    Wrapper for aesara.test.unittest_tools.py:verify_grad which
    converts sparse variables back and forth.

    Parameters
    ----------
    op
        Op to check.
    pt
        List of inputs to realize the tests.
    structured
        True to tests with a structured grad, False otherwise.
    args
        Other `verify_grad` parameters if any.
    kwargs
        Other `verify_grad` keywords if any.

    Returns
    -------
    None
    """

    def conv_none(x):
        return x

    def conv_csr(ind, indptr, shp):
        def f(spdata):
            return CSR(spdata, ind, indptr, shp)

        return f

    def conv_csc(ind, indptr, shp):
        def f(spdata):
            return CSC(spdata, ind, indptr, shp)

        return f

    iconv = []
    dpt = []

    for p in pt:
        if _is_sparse(p):
            if structured:
                dpt.append(p.data)
            else:
                dpt.append(p.toarray())
            if p.format == "csr":
                if structured:
                    iconv.append(conv_csr(p.indices[: p.size], p.indptr, p.shape))
                else:
                    iconv.append(csr_from_dense)
            elif p.format == "csc":
                if structured:
                    iconv.append(conv_csc(p.indices[: p.size], p.indptr, p.shape))
                else:
                    iconv.append(csc_from_dense)
            else:
                raise NotImplementedError(f"No conv for {p.format}")
        else:
            dpt.append(p)
            iconv.append(conv_none)
    output = op(*[as_sparse_or_tensor_variable(p) for p in pt])
    if isinstance(output, (list, tuple)):
        raise NotImplementedError("verify_grad can't deal with " "multiple outputs")
    if _is_sparse_variable(output):
        oconv = DenseFromSparse(structured=structured)
    else:
        oconv = conv_none

    def conv_op(*inputs):
        ipt = [conv(i) for i, conv in zip(inputs, iconv)]
        out = op(*ipt)
        return oconv(out)

    return utt.verify_grad(conv_op, dpt, *args, **kwargs)


class TestVerifyGradSparse:
    class FailOp(Op):
        def __init__(self, structured):
            self.structured = structured

        def __eq__(self, other):
            return (type(self) == type(other)) and self.structured == other.structured

        def __hash__(self):
            return hash(type(self)) ^ hash(self.structured)

        def make_node(self, x):
            x = as_sparse_variable(x)
            return Apply(self, [x], [x.type()])

        def perform(self, node, inputs, outputs):
            (x,) = inputs
            (out,) = outputs
            assert _is_sparse(x)
            out[0] = -x

        def grad(self, inputs, gout):
            (x,) = inputs
            (gz,) = gout
            assert _is_sparse_variable(x) and _is_sparse_variable(gz)
            if self.structured:
                return (sp_ones_like(x) * dense_from_sparse(gz),)
            else:
                return (gz,)

        def infer_shape(self, fgraph, node, shapes):
            return [shapes[0]]

    def test_grad_fail(self):
        with pytest.raises(GradientError):
            verify_grad_sparse(
                self.FailOp(structured=False),
                [sp.sparse.csr_matrix(random_lil((10, 40), config.floatX, 3))],
            )

        with pytest.raises(GradientError):
            verify_grad_sparse(
                self.FailOp(structured=True),
                [sp.sparse.csr_matrix(random_lil((10, 40), config.floatX, 3))],
            )


class TestTranspose:
    def setup_method(self):
        utt.seed_rng()

    def test_transpose_csc(self):
        spe = sp.sparse.csc_matrix(sp.sparse.eye(5, 3))
        a = as_sparse_variable(spe)
        assert a.data is not spe
        assert a.data.shape == (5, 3)
        assert a.type.dtype == "float64", a.type.dtype
        assert a.type.format == "csc", a.type.format
        ta = transpose(a)
        assert ta.type.dtype == "float64", ta.type.dtype
        assert ta.type.format == "csr", ta.type.format

        vta = eval_outputs([ta])
        assert vta.shape == (3, 5)

    def test_transpose_csr(self):
        a = as_sparse_variable(sp.sparse.csr_matrix(sp.sparse.eye(5, 3)))
        assert a.data.shape == (5, 3)
        assert a.type.dtype == "float64"
        assert a.type.format == "csr"
        ta = transpose(a)
        assert ta.type.dtype == "float64", ta.type.dtype
        assert ta.type.format == "csc", ta.type.format

        vta = eval_outputs([ta])
        assert vta.shape == (3, 5)


class TestSparseInferShape(utt.InferShapeTester):
    @pytest.mark.skip(reason="infer_shape not implemented for GetItem2d yet")
    def test_getitem_2d(self):
        pass

    def test_getitem_scalar(self):
        x = SparseType("csr", dtype=config.floatX)()
        self._compile_and_check(
            [x],
            [x[2, 2]],
            [sp.sparse.csr_matrix(random_lil((10, 40), config.floatX, 3))],
            GetItemScalar,
        )

    def test_csm(self):
        for sparsetype in ("csr", "csc"):
            x = vector()
            y = ivector()
            z = ivector()
            s = ivector()
            call = getattr(sp.sparse, sparsetype + "_matrix")
            spm = call(random_lil((300, 400), config.floatX, 5))
            out = CSM(sparsetype)(x, y, z, s)
            self._compile_and_check(
                [x, y, z, s], [out], [spm.data, spm.indices, spm.indptr, spm.shape], CSM
            )

    def test_csm_grad(self):
        for sparsetype in ("csr", "csc"):
            x = vector()
            y = ivector()
            z = ivector()
            s = ivector()
            call = getattr(sp.sparse, sparsetype + "_matrix")
            spm = call(random_lil((300, 400), config.floatX, 5))
            out = aesara.grad(dense_from_sparse(CSM(sparsetype)(x, y, z, s)).sum(), x)
            self._compile_and_check(
                [x, y, z, s],
                [out],
                [spm.data, spm.indices, spm.indptr, spm.shape],
                (CSMGrad, CSMGradC),
            )

    def test_transpose(self):
        x = SparseType("csr", dtype=config.floatX)()
        self._compile_and_check(
            [x],
            [x.T],
            [sp.sparse.csr_matrix(random_lil((10, 40), config.floatX, 3))],
            Transpose,
        )

    def test_neg(self):
        x = SparseType("csr", dtype=config.floatX)()
        self._compile_and_check(
            [x],
            [-x],
            [sp.sparse.csr_matrix(random_lil((10, 40), config.floatX, 3))],
            Neg,
        )

    def test_add_ss(self):
        x = SparseType("csr", dtype=config.floatX)()
        y = SparseType("csr", dtype=config.floatX)()
        self._compile_and_check(
            [x, y],
            [x + y],
            [
                sp.sparse.csr_matrix(random_lil((10, 40), config.floatX, 3)),
                sp.sparse.csr_matrix(random_lil((10, 40), config.floatX, 3)),
            ],
            AddSS,
        )

    def test_add_sd(self):
        x = SparseType("csr", dtype=config.floatX)()
        y = matrix()
        self._compile_and_check(
            [x, y],
            [x + y],
            [
                sp.sparse.csr_matrix(random_lil((10, 40), config.floatX, 3)),
                np.random.randn(10, 40).astype(config.floatX),
            ],
            (AddSD, sparse.opt.AddSD_ccode),
        )

    def test_mul_ss(self):
        x = SparseType("csr", dtype=config.floatX)()
        y = SparseType("csr", dtype=config.floatX)()
        self._compile_and_check(
            [x, y],
            [x * y],
            [
                sp.sparse.csr_matrix(random_lil((10, 40), config.floatX, 3)),
            ]
            * 2,
            MulSS,
        )

    def test_mul_sd(self):
        x = SparseType("csr", dtype=config.floatX)()
        y = matrix()
        self._compile_and_check(
            [x, y],
            [x * y],
            [
                sp.sparse.csr_matrix(random_lil((10, 40), config.floatX, 3)),
                np.random.randn(10, 40).astype(config.floatX),
            ],
            MulSD,
            excluding=["local_mul_s_d"],
        )

    def test_remove0(self):
        x = SparseType("csr", dtype=config.floatX)()
        self._compile_and_check(
            [x],
            [Remove0()(x)],
            [sp.sparse.csr_matrix(random_lil((10, 40), config.floatX, 3))],
            Remove0,
        )

    def test_dot(self):
        x = SparseType("csc", dtype=config.floatX)()
        y = SparseType("csc", dtype=config.floatX)()
        self._compile_and_check(
            [x, y],
            [Dot()(x, y)],
            [
                sp.sparse.csc_matrix(random_lil((4, 5), config.floatX, 3)),
                sp.sparse.csc_matrix(random_lil((5, 3), config.floatX, 3)),
            ],
            Dot,
        )

    def test_dot_broadcast(self):
        for x, y in [
            (SparseType("csr", "float32")(), vector()[:, None]),
            (SparseType("csr", "float32")(), vector()[None, :]),
            (SparseType("csr", "float32")(), matrix()),
            (vector()[:, None], SparseType("csr", "float32")()),
            (vector()[None, :], SparseType("csr", "float32")()),
            (matrix(), SparseType("csr", "float32")()),
        ]:

            sparse_out = aet.dot(x, y)
            if isinstance(x, sparse.SparseVariable):
                x = matrix()
            if isinstance(y, sparse.SparseVariable):
                y = matrix()
            dense_out = aet.dot(x, y)
            assert dense_out.broadcastable == sparse_out.broadcastable

    def test_structured_dot(self):
        x = SparseType("csc", dtype=config.floatX)()
        y = SparseType("csc", dtype=config.floatX)()
        self._compile_and_check(
            [x, y],
            [structured_dot(x, y)],
            [
                sp.sparse.csc_matrix(random_lil((4, 5), config.floatX, 3)),
                sp.sparse.csc_matrix(random_lil((5, 3), config.floatX, 3)),
            ],
            StructuredDot,
        )

    @pytest.mark.skip(
        reason="infer_shape not implemented for the grad of structured_dot"
    )
    def test_structured_dot_grad(self):
        # We also need the grad of CSM to be implemetned.
        for format, op in [
            ("csc", StructuredDotGradCSC),
            ("csr", StructuredDotGradCSR),
        ]:
            x = SparseType(format, dtype=config.floatX)()
            y = SparseType(format, dtype=config.floatX)()
            grads = aesara.grad(dense_from_sparse(structured_dot(x, y)).sum(), [x, y])
            self._compile_and_check(
                [x, y],
                [grads[0]],
                [
                    as_sparse_format(random_lil((4, 5), config.floatX, 3), format),
                    as_sparse_format(random_lil((5, 3), config.floatX, 3), format),
                ],
                op,
            )
            self._compile_and_check(
                [x, y],
                [grads[1]],
                [
                    as_sparse_format(random_lil((4, 5), config.floatX, 3), format),
                    as_sparse_format(random_lil((5, 3), config.floatX, 3), format),
                ],
                op,
            )

    def test_dense_from_sparse(self):
        x = SparseType("csr", dtype=config.floatX)()
        self._compile_and_check(
            [x],
            [dense_from_sparse(x)],
            [sp.sparse.csr_matrix(random_lil((10, 40), config.floatX, 3))],
            dense_from_sparse.__class__,
        )

    def test_sparse_from_dense(self):
        x = matrix()
        self._compile_and_check(
            [x],
            [csc_from_dense(x)],
            [np.random.randn(10, 40).astype(config.floatX)],
            csc_from_dense.__class__,
        )

    def test_sparse_from_list(self):
        x = matrix("x")
        vals = matrix("vals")
        ilist = lvector("ilist")

        out = construct_sparse_from_list(x, vals, ilist)
        self._compile_and_check(
            [x, vals, ilist],
            [out],
            [
                np.zeros((40, 10), dtype=config.floatX),
                np.random.randn(12, 10).astype(config.floatX),
                np.random.randint(low=0, high=40, size=(12,)),
            ],
            ConstructSparseFromList,
        )


class TestConstructSparseFromList:
    def test_adv_sub1_sparse_grad(self):
        v = ivector()

        # Assert we don't create a sparse grad by default
        m = matrix()
        sub = m[v]
        g = aesara.grad(sub.sum(), m)
        assert isinstance(g.owner.op, AdvancedIncSubtensor1)

        # Test that we create a sparse grad when asked
        # USER INTERFACE
        m = matrix()
        v = ivector()
        sub = aesara.sparse.sparse_grad(m[v])
        g = aesara.grad(sub.sum(), m)
        assert isinstance(g.owner.op, ConstructSparseFromList)

        # Test that we create a sparse grad when asked
        # Op INTERFACE
        m = matrix()
        v = ivector()
        sub = AdvancedSubtensor1(sparse_grad=True)(m, v)
        g = aesara.grad(sub.sum(), m)
        assert isinstance(g.owner.op, ConstructSparseFromList)

        # Test the sparse grad
        valm = np.random.rand(5, 4).astype(config.floatX)
        valv = np.random.randint(0, 5, 10)
        m = matrix()
        shared_v = aesara.shared(valv)

        def fn(m):
            return aesara.sparse.sparse_grad(m[shared_v])

        verify_grad_sparse(fn, [valm])

    def test_err(self):
        for ndim in [1, 3]:
            t = TensorType(dtype=config.floatX, broadcastable=(False,) * ndim)()
            v = ivector()
            sub = t[v]

            # Assert we don't create a sparse grad by default
            g = aesara.grad(sub.sum(), t)
            assert isinstance(g.owner.op, AdvancedIncSubtensor1)

            # Test that we raise an error, as we can't create a sparse
            # grad from tensors that don't have 2 dimensions.
            sub = aesara.sparse.sparse_grad(sub)
            with pytest.raises(TypeError):
                aesara.grad(sub.sum(), t)


class TestAddMul:
    def test_AddSS(self):
        self._testSS(add)

    def test_AddSD(self):
        self._testSD(add)

    def test_AddDS(self):
        self._testDS(add)

    def test_MulSS(self):
        self._testSS(
            mul,
            np.array([[1.0, 0], [3, 0], [0, 6]]),
            np.array([[1.0, 2], [3, 0], [0, 6]]),
        )

    def test_MulSD(self):
        self._testSD(
            mul,
            np.array([[1.0, 0], [3, 0], [0, 6]]),
            np.array([[1.0, 2], [3, 0], [0, 6]]),
        )

    def test_MulDS(self):
        self._testDS(
            mul,
            np.array([[1.0, 0], [3, 0], [0, 6]]),
            np.array([[1.0, 2], [3, 0], [0, 6]]),
        )

    def _testSS(
        self,
        op,
        array1=None,
        array2=None,
    ):
        if array1 is None:
            array1 = np.array([[1.0, 0], [3, 0], [0, 6]])
        if array2 is None:
            array2 = np.asarray([[0, 2.0], [0, 4], [5, 0]])

        for mtype1, mtype2 in product(_mtypes, _mtypes):
            for dtype1, dtype2 in [
                ("float64", "int8"),
                ("int8", "float64"),
                ("float64", "float64"),
            ]:
                a = mtype1(array1).astype(dtype1)
                aR = as_sparse_variable(a)
                assert aR.data is not a
                assert _is_sparse(a)
                assert _is_sparse_variable(aR)

                b = mtype2(array2).astype(dtype2)
                bR = as_sparse_variable(b)
                assert bR.data is not b
                assert _is_sparse(b)
                assert _is_sparse_variable(bR)

                apb = op(aR, bR)
                assert _is_sparse_variable(apb)

                assert apb.type.format == aR.type.format, apb.type.format

                val = eval_outputs([apb])
                assert val.shape == (3, 2)
                if op is add:
                    assert np.all(val.todense() == (array1 + array2))
                    if dtype1.startswith("float") and dtype2.startswith("float"):
                        verify_grad_sparse(op, [a, b], structured=False)
                elif op is mul:
                    assert np.all(val.todense() == (array1 * array2))
                    if dtype1.startswith("float") and dtype2.startswith("float"):
                        verify_grad_sparse(op, [a, b], structured=False)

    def _testSD(
        self,
        op,
        array1=None,
        array2=None,
    ):
        if array1 is None:
            array1 = np.array([[1.0, 0], [3, 0], [0, 6]])
        if array2 is None:
            array2 = np.asarray([[0, 2.0], [0, 4], [5, 0]])

        for mtype in _mtypes:
            for a in [
                np.array(array1),
                aet.as_tensor_variable(array1),
                aesara.shared(array1),
            ]:
                for dtype1, dtype2 in [
                    ("float64", "int8"),
                    ("int8", "float64"),
                    # Needed to test the grad
                    ("float32", "float64"),
                ]:
                    a = a.astype(dtype1)
                    b = mtype(array2).astype(dtype2)
                    bR = as_sparse_variable(b)
                    assert bR.data is not b  # constants are copied
                    assert _is_sparse(b)
                    assert _is_sparse_variable(bR)

                    apb = op(a, bR)

                    val = eval_outputs([apb])
                    assert val.shape == (3, 2)
                    if op is add:
                        assert _is_dense_variable(apb)
                        assert np.all(val == (array1 + b))
                        ans = np.array([[1.0, 2], [3, 4], [5, 6]])
                        assert np.all(val == ans)
                        if isinstance(a, Constant):
                            a = a.data
                        if getattr(a, "owner", None):
                            continue
                        if dtype1.startswith("float") and dtype2.startswith("float"):
                            verify_grad_sparse(op, [a, b], structured=True)
                    elif op is mul:
                        assert _is_sparse_variable(apb)
                        assert np.all(val.todense() == (b.multiply(array1)))
                        assert np.all(
                            val.todense() == np.array([[1, 0], [9, 0], [0, 36]])
                        )
                        if isinstance(a, Constant):
                            a = a.data
                        if getattr(a, "owner", None):
                            continue
                        if dtype1.startswith("float") and dtype2.startswith("float"):
                            verify_grad_sparse(op, [a, b], structured=False)

    def _testDS(
        self,
        op,
        array1=None,
        array2=None,
    ):
        if array1 is None:
            array1 = np.array([[1.0, 0], [3, 0], [0, 6]])
        if array2 is None:
            array2 = np.asarray([[0, 2.0], [0, 4], [5, 0]])

        for mtype in _mtypes:
            for b in [
                np.asarray(array2),
                aet.as_tensor_variable(array2),
                aesara.shared(array2),
            ]:
                for dtype1, dtype2 in [
                    ("float64", "int8"),
                    ("int8", "float64"),
                ]:
                    a = mtype(array1).astype(dtype1)
                    aR = as_sparse_variable(a)
                    assert aR.data is not a
                    assert _is_sparse(a)
                    assert _is_sparse_variable(aR)
                    b = b.astype(dtype2)

                    apb = op(aR, b)

                    val = eval_outputs([apb])
                    assert val.shape == (3, 2)
                    if op is add:
                        assert _is_dense_variable(apb)
                        assert np.all(val == (a + array2))
                        ans = np.array([[1.0, 2], [3, 4], [5, 6]])
                        assert np.all(val == ans)
                        if isinstance(b, Constant):
                            b = b.data
                        if dtype1.startswith("float") and dtype2.startswith("float"):
                            verify_grad_sparse(op, [a, b], structured=True)
                    elif op is mul:
                        assert _is_sparse_variable(apb)
                        ans = np.array([[1, 0], [9, 0], [0, 36]])
                        assert np.all(val.todense() == (a.multiply(array2)))
                        assert np.all(val.todense() == ans)
                        if isinstance(b, Constant):
                            b = b.data
                        if dtype1.startswith("float") and dtype2.startswith("float"):
                            verify_grad_sparse(op, [a, b], structured=False)


class TestComparison:
    def setup_method(self):
        utt.seed_rng()

    # took from tensor basic_test.py
    def _rand_ranged(self, min, max, shape):
        return np.asarray(
            np.random.rand(*shape) * (max - min) + min, dtype=config.floatX
        )

    tests = [
        lambda x, y: x > y,
        lambda x, y: x < y,
        lambda x, y: x >= y,
        lambda x, y: x <= y,
    ]

    testsDic = {
        gt: lambda x, y: x > y,
        lt: lambda x, y: x < y,
        ge: lambda x, y: x >= y,
        le: lambda x, y: x <= y,
    }

    @pytest.mark.skipif(
        version.parse(sp.__version__) < version.parse("0.13"),
        reason="Comparison operators need newer release of scipy",
    )
    def __generalized_ss_test(self, aesarap, symbolicType, testOp, scipyType):
        x = symbolicType()
        y = symbolicType()

        op = aesarap(x, y)

        f = aesara.function([x, y], op)

        m1 = scipyType(random_lil((10, 40), config.floatX, 3))
        m2 = scipyType(random_lil((10, 40), config.floatX, 3))

        assert np.array_equal(f(m1, m2).data, testOp(m1, m2).data)

    @pytest.mark.skipif(
        version.parse(sp.__version__) < version.parse("0.13"),
        reason="Comparison operators need newer release of scipy",
    )
    def __generalized_sd_test(self, aesarap, symbolicType, testOp, scipyType):
        x = symbolicType()
        y = matrix()

        op = aesarap(x, y)

        f = aesara.function([x, y], op)

        m1 = scipyType(random_lil((10, 40), config.floatX, 3))
        m2 = self._rand_ranged(1000, -1000, [10, 40])

        assert np.array_equal(f(m1, m2).data, testOp(m1, m2).data)

    @pytest.mark.skipif(
        version.parse(sp.__version__) < version.parse("0.13"),
        reason="Comparison operators need newer release of scipy",
    )
    def __generalized_ds_test(self, aesarap, symbolicType, testOp, scipyType):
        x = symbolicType()
        y = matrix()

        op = aesarap(y, x)

        f = aesara.function([y, x], op)

        m1 = scipyType(random_lil((10, 40), config.floatX, 3))
        m2 = self._rand_ranged(1000, -1000, [10, 40])

        assert np.array_equal(f(m2, m1).data, testOp(m2, m1).data)

    def test_ss_csr_comparison(self):

        for op in self.tests:
            self.__generalized_ss_test(op, sparse.csr_matrix, op, sp.sparse.csr_matrix)

    def test_ss_csc_comparison(self):

        for op in self.tests:
            self.__generalized_ss_test(op, sparse.csc_matrix, op, sp.sparse.csc_matrix)

    def test_sd_csr_comparison(self):

        for op in self.tests:
            self.__generalized_sd_test(op, sparse.csr_matrix, op, sp.sparse.csr_matrix)

    def test_sd_csc_comparison(self):

        for op in self.tests:
            self.__generalized_sd_test(op, sparse.csc_matrix, op, sp.sparse.csc_matrix)

    def test_ds_csc_comparison(self):

        for op in self.testsDic:
            self.__generalized_ds_test(
                op, sparse.csc_matrix, self.testsDic[op], sp.sparse.csc_matrix
            )

    def test_ds_csr_comparison(self):

        for op in self.testsDic:
            self.__generalized_ds_test(
                op, sparse.csr_matrix, self.testsDic[op], sp.sparse.csr_matrix
            )

    @pytest.mark.skipif(
        version.parse(sp.__version__) < version.parse("0.13"),
        reason="Comparison operators need newer release of scipy",
    )
    def test_equality_case(self):
        # Test assuring normal behaviour when values
        # in the matrices are equal
        x = sparse.csc_matrix()
        y = matrix()

        m1 = sp.sparse.csc_matrix((2, 2), dtype=aesara.config.floatX)
        m2 = np.asarray([[0, 0], [0, 0]], dtype=aesara.config.floatX)

        for func in self.testsDic:

            op = func(y, x)
            f = aesara.function([y, x], op)

            assert np.array_equal(f(m2, m1), self.testsDic[func](m2, m1))


class TestConversion:
    def setup_method(self):
        utt.seed_rng()

    def test_basic(self):
        test_val = np.random.rand(5).astype(config.floatX)
        a = aet.as_tensor_variable(test_val)
        s = csc_from_dense(a)
        val = eval_outputs([s])
        assert str(val.dtype) == config.floatX
        assert val.format == "csc"

        a = aet.as_tensor_variable(test_val)
        s = csr_from_dense(a)
        val = eval_outputs([s])
        assert str(val.dtype) == config.floatX
        assert val.format == "csr"

        test_val = np.eye(3).astype(config.floatX)
        a = sp.sparse.csr_matrix(test_val)
        s = as_sparse_or_tensor_variable(a)
        res = aet.as_tensor_variable(s)
        assert isinstance(res, SparseConstant)

        a = sp.sparse.csr_matrix(test_val)
        s = as_sparse_or_tensor_variable(a)
        from aesara.tensor.exceptions import NotScalarConstantError

        with pytest.raises(NotScalarConstantError):
            aet.get_scalar_constant_value(s, only_process_constants=True)

    # TODO:
    # def test_sparse_as_tensor_variable(self):
    #     csr = sp.sparse.csr_matrix(np.eye(3))
    #     val = aet.as_tensor_variable(csr)
    #     assert str(val.dtype) == config.floatX
    #     assert val.format == "csr"
    #
    #     csr = sp.sparse.csc_matrix(np.eye(3))
    #     val = aet.as_tensor_variable(csr)
    #     assert str(val.dtype) == config.floatX
    #     assert val.format == "csc"

    def test_dense_from_sparse(self):
        # call dense_from_sparse
        for t in _mtypes:
            s = t(sp.sparse.identity(5))
            s = as_sparse_variable(s)
            d = dense_from_sparse(s)
            val = eval_outputs([d])
            assert str(val.dtype) == s.dtype
            assert np.all(val[0] == [1, 0, 0, 0, 0])

    def test_todense(self):
        # call sparse_var.todense()
        for t in _mtypes:
            s = t(sp.sparse.identity(5))
            s = as_sparse_variable(s)
            d = s.toarray()
            val = eval_outputs([d])
            assert str(val.dtype) == s.dtype
            assert np.all(val[0] == [1, 0, 0, 0, 0])

    @staticmethod
    def check_format_ndim(format, ndim):
        x = tensor(dtype=config.floatX, broadcastable=([False] * ndim), name="x")

        s = SparseFromDense(format)(x)
        s_m = -s
        d = dense_from_sparse(s_m)
        c = d.sum()
        g = aesara.grad(c, x)
        f = aesara.function([x], [s, g])
        f(np.array(0, dtype=config.floatX, ndmin=ndim))
        f(np.array(7, dtype=config.floatX, ndmin=ndim))

    def test_format_ndim(self):
        for format in "csc", "csr":
            for ndim in 0, 1, 2:
                self.check_format_ndim(format, ndim)

            with pytest.raises(TypeError):
                self.check_format_ndim(format, 3)
            with pytest.raises(TypeError):
                self.check_format_ndim(format, 4)


class TestCsmProperties:
    def setup_method(self):
        utt.seed_rng()

    def test_csm_properties_grad(self):
        sp_types = {"csc": sp.sparse.csc_matrix, "csr": sp.sparse.csr_matrix}

        for format in ["csc", "csr"]:
            for dtype in ["float32", "float64"]:
                spmat = sp_types[format](random_lil((4, 3), dtype, 3))

                verify_grad_sparse(
                    lambda *x: CSMProperties()(*x)[0], [spmat], structured=True
                )

                verify_grad_sparse(
                    lambda *x: CSMProperties()(*x)[1], [spmat], structured=True
                )

                verify_grad_sparse(
                    lambda *x: CSMProperties()(*x)[2], [spmat], structured=True
                )

                verify_grad_sparse(
                    lambda *x: CSMProperties()(*x)[2], [spmat], structured=True
                )

    def test_csm_properties(self):
        sp_types = {"csc": sp.sparse.csc_matrix, "csr": sp.sparse.csr_matrix}

        for format in ["csc", "csr"]:
            for dtype in ["float32", "float64"]:
                x = SparseType(format, dtype=dtype)()
                f = aesara.function([x], csm_properties(x))

                spmat = sp_types[format](random_lil((4, 3), dtype, 3))

                data, indices, indptr, shape = f(spmat)

                assert np.all(data == spmat.data)
                assert np.all(indices == spmat.indices)
                assert np.all(indptr == spmat.indptr)
                assert np.all(shape == spmat.shape)


class TestCsm:
    def setup_method(self):
        utt.seed_rng()

    def test_csm_grad(self):
        sp_types = {"csc": sp.sparse.csc_matrix, "csr": sp.sparse.csr_matrix}

        for format in ["csc", "csr"]:
            for dtype in ["float32", "float64"]:
                spmat = sp_types[format](random_lil((4, 3), dtype, 3))

                verify_grad_sparse(
                    lambda x: CSM(format)(
                        x, spmat.indices, spmat.indptr, np.asarray(spmat.shape, "int32")
                    ),
                    [spmat.data],
                    structured=True,
                )

    def test_csm_sparser(self):
        # Test support for gradients sparser than the input.

        sp_types = {"csc": sp.sparse.csc_matrix, "csr": sp.sparse.csr_matrix}

        for format in ["csc", "csr"]:
            for dtype in ["float32", "float64"]:
                x = tensor(dtype=dtype, broadcastable=(False,))
                y = ivector()
                z = ivector()
                s = ivector()

                a = as_sparse_variable(sp_types[format](random_lil((4, 3), dtype, 1)))

                f = aesara.function(
                    [x, y, z, s],
                    aesara.grad(
                        dense_from_sparse(a * CSM(format)(x, y, z, s)).sum(), x
                    ),
                )

                spmat = sp_types[format](random_lil((4, 3), dtype, 3))

                res = f(
                    spmat.data,
                    spmat.indices,
                    spmat.indptr,
                    np.asarray(spmat.shape, "int32"),
                )

                assert len(spmat.data) == len(res)

    def test_csm_unsorted(self):
        # Test support for gradients of unsorted inputs.

        for format in [
            "csr",
            "csc",
        ]:
            for dtype in ["float32", "float64"]:
                # Sparse advanced indexing produces unsorted sparse matrices
                a = sparse_random_inputs(
                    format, (8, 6), out_dtype=dtype, unsorted_indices=True
                )[1][0]
                # Make sure it's unsorted
                assert not a.has_sorted_indices

                def my_op(x):
                    y = aet.constant(a.indices)
                    z = aet.constant(a.indptr)
                    s = aet.constant(a.shape)
                    return aet_sum(dense_from_sparse(CSM(format)(x, y, z, s) * a))

                verify_grad_sparse(my_op, [a.data])

    def test_csm(self):
        sp_types = {"csc": sp.sparse.csc_matrix, "csr": sp.sparse.csr_matrix}

        for format in ["csc", "csr"]:
            for dtype in ["float32", "float64"]:
                x = tensor(dtype=dtype, broadcastable=(False,))
                y = ivector()
                z = ivector()
                s = ivector()
                f = aesara.function([x, y, z, s], CSM(format)(x, y, z, s))

                spmat = sp_types[format](random_lil((4, 3), dtype, 3))

                res = f(
                    spmat.data,
                    spmat.indices,
                    spmat.indptr,
                    np.asarray(spmat.shape, "int32"),
                )

                assert np.all(res.data == spmat.data)
                assert np.all(res.indices == spmat.indices)
                assert np.all(res.indptr == spmat.indptr)
                assert np.all(res.shape == spmat.shape)


class TestStructuredDot:
    def setup_method(self):
        utt.seed_rng()

    def test_structureddot_csc_grad(self):

        # shortcut: testing csc in float32, testing csr in float64

        # allocate a random sparse matrix
        spmat = sp.sparse.csc_matrix(random_lil((4, 3), "float32", 3))

        mat = np.asarray(np.random.randn(3, 2), "float32")

        verify_grad_sparse(structured_dot, [spmat, mat], structured=True)

        def buildgraph_T(spmat, mat):
            return structured_dot(mat.T, spmat.T)

        verify_grad_sparse(buildgraph_T, [spmat, mat], structured=True)

    def test_structureddot_csr_grad(self):

        # shortcut: testing csc in float32, testing csr in float64

        # allocate a random sparse matrix
        spmat = sp.sparse.csr_matrix(random_lil((4, 3), "float64", 3))

        mat = np.asarray(np.random.randn(3, 2), "float64")

        verify_grad_sparse(structured_dot, [spmat, mat], structured=True)

        def buildgraph_T(spmat, mat):
            return structured_dot(mat.T, spmat.T)

        verify_grad_sparse(buildgraph_T, [spmat, mat], structured=True)

    def test_upcast(self):

        typenames = (
            "float32",
            "int64",
            "int8",
            "int32",
            "int16",
            "float64",
            "complex64",
            "complex128",
        )
        for dense_dtype in typenames:
            for sparse_dtype in typenames:
                correct_dtype = aesara.scalar.upcast(sparse_dtype, dense_dtype)
                a = SparseType("csc", dtype=sparse_dtype)()
                b = matrix(dtype=dense_dtype)
                d = structured_dot(a, b)
                assert d.type.dtype == correct_dtype

                # compile and run a function

                f = aesara.function([a, b], d)

                M, N, K, nnz = (4, 3, 5, 3)
                spmat = sp.sparse.csc_matrix(random_lil((M, N), sparse_dtype, nnz))
                # the following madness is necessary to workaround
                # an intc vs. int32 bug.
                # The lil makes an intc on my computer when sparse_dtype
                # is int32.
                spmat.dtype = np.dtype(sparse_dtype)
                mat = np.asarray(np.random.randn(N, K) * 9, dtype=dense_dtype)
                # print 'DTYPES', sparse_dtype, dense_dtype
                # print 'sym types', a.type, b.type
                # print 'dtype strings', spmat.dtype, mat.dtype
                # print 'numpy dtype num', mat.dtype.num
                # print 'scipy dtype num', spmat.data.dtype.num
                aesara_result = f(spmat, mat)
                scipy_result = spmat * mat
                assert aesara_result.shape == scipy_result.shape
                assert aesara_result.dtype == scipy_result.dtype
                utt.assert_allclose(scipy_result, aesara_result)

    def test_opt_unpack(self):
        #
        # Test that a graph involving
        # structured_dot(assembled_csc_matrix) is optimized to be just
        # a structured_dot_csc Op and no assembly of a csc_matrix.
        #
        # The optimization from structured_dot -> structured_dot_csc
        # is currently disabled, So this test is not expected to pass

        return
        #
        kerns = TensorType(dtype="int64", broadcastable=[False])("kerns")
        spmat = sp.sparse.lil_matrix((4, 6), dtype="int64")
        for i in range(5):
            # set non-zeros in random locations (row x, col y)
            x = np.floor(np.random.rand() * spmat.shape[0])
            y = np.floor(np.random.rand() * spmat.shape[1])
            spmat[x, y] = np.random.rand() * 10
        spmat = sp.sparse.csc_matrix(spmat)

        images = TensorType(dtype="float32", broadcastable=[False, False])("images")

        cscmat = CSC(kerns, spmat.indices[: spmat.size], spmat.indptr, spmat.shape)
        f = aesara.function([kerns, images], structured_dot(cscmat, images.T))

        sdcscpresent = False
        for node in f.maker.fgraph.toposort():
            # print node.op
            assert not isinstance(node.op, CSM)
            assert not isinstance(node.op, CSMProperties)
            if isinstance(f.maker.fgraph.toposort()[1].op, StructuredDotCSC):
                sdcscpresent = True
        assert sdcscpresent

        kernvals = np.array(spmat.data[: spmat.size])
        # print 'kdtype', kernvals.dtype, kernvals.shape,
        # print kernvals.ndim, kernvals.dtype.num
        # print 'type of kernvals = ', kernvals.dtype
        bsize = 3
        imvals = 1.0 * np.array(
            np.arange(bsize * spmat.shape[1]).reshape(bsize, spmat.shape[1]),
            dtype="float32",
        )
        f(kernvals, imvals)
        # print outvals

    def test_dot_sparse_sparse(self):
        # test dot for 2 input sparse matrix
        sparse_dtype = "float64"
        sp_mat = {
            "csc": sp.sparse.csc_matrix,
            "csr": sp.sparse.csr_matrix,
            "bsr": sp.sparse.csr_matrix,
        }

        for sparse_format_a in ["csc", "csr", "bsr"]:
            for sparse_format_b in ["csc", "csr", "bsr"]:
                a = SparseType(sparse_format_a, dtype=sparse_dtype)()
                b = SparseType(sparse_format_b, dtype=sparse_dtype)()
                d = aet.dot(a, b)
                f = aesara.function([a, b], Out(d, borrow=True))
                for M, N, K, nnz in [
                    (4, 3, 2, 3),
                    (40, 30, 20, 3),
                    (40, 30, 20, 30),
                    (400, 3000, 200, 6000),
                ]:
                    a_val = sp_mat[sparse_format_a](
                        random_lil((M, N), sparse_dtype, nnz)
                    )
                    b_val = sp_mat[sparse_format_b](
                        random_lil((N, K), sparse_dtype, nnz)
                    )
                    f(a_val, b_val)

    def test_csc_correct_output_faster_than_scipy(self):
        sparse_dtype = "float64"
        dense_dtype = "float64"

        a = SparseType("csc", dtype=sparse_dtype)()
        b = matrix(dtype=dense_dtype)
        d = aet.dot(a, b)
        f = aesara.function([a, b], Out(d, borrow=True))

        for M, N, K, nnz in [
            (4, 3, 2, 3),
            (40, 30, 20, 3),
            (40, 30, 20, 30),
            (400, 3000, 200, 6000),
        ]:
            spmat = sp.sparse.csc_matrix(random_lil((M, N), sparse_dtype, nnz))
            mat = np.asarray(np.random.randn(N, K), dense_dtype)
            aesara_times = []
            scipy_times = []
            for i in range(5):
                t0 = time.time()
                aesara_result = f(spmat, mat)
                t1 = time.time()
                scipy_result = spmat * mat
                t2 = time.time()

                aesara_times.append(t1 - t0)
                scipy_times.append(t2 - t1)

            aesara_time = np.min(aesara_times)
            scipy_time = np.min(scipy_times)

            # speedup = scipy_time / aesara_time
            # print scipy_times
            # print aesara_times
            # print ('M=%(M)s N=%(N)s K=%(K)s nnz=%(nnz)s aesara_time'
            #       '=%(aesara_time)s speedup=%(speedup)s') % locals()

            # fail if Aesara is slower than scipy by more than a certain amount
            overhead_tol = 0.003  # seconds overall
            overhead_rtol = 1.2  # times as long
            utt.assert_allclose(scipy_result, aesara_result)
            if aesara.config.mode == "FAST_RUN" and aesara.config.cxx:
                assert aesara_time <= overhead_rtol * scipy_time + overhead_tol

    def test_csr_correct_output_faster_than_scipy(self):

        # contrast with test_grad, we put csr in float32, csc in float64

        sparse_dtype = "float32"
        dense_dtype = "float32"

        a = SparseType("csr", dtype=sparse_dtype)()
        b = matrix(dtype=dense_dtype)
        d = aet.dot(a, b)
        f = aesara.function([a, b], d)

        for M, N, K, nnz in [
            (4, 3, 2, 3),
            (40, 30, 20, 3),
            (40, 30, 20, 30),
            (400, 3000, 200, 6000),
        ]:
            spmat = sp.sparse.csr_matrix(random_lil((M, N), sparse_dtype, nnz))
            mat = np.asarray(np.random.randn(N, K), dense_dtype)
            t0 = time.time()
            aesara_result = f(spmat, mat)
            t1 = time.time()
            scipy_result = spmat * mat
            t2 = time.time()

            aesara_time = t1 - t0
            scipy_time = t2 - t1
            # print 'aesara took', aesara_time,
            # print 'scipy took', scipy_time
            overhead_tol = 0.002  # seconds
            overhead_rtol = 1.1  # times as long
            utt.assert_allclose(scipy_result, aesara_result)
            if aesara.config.mode == "FAST_RUN" and aesara.config.cxx:
                assert aesara_time <= overhead_rtol * scipy_time + overhead_tol, (
                    aesara_time,
                    overhead_rtol * scipy_time + overhead_tol,
                    scipy_time,
                    overhead_rtol,
                    overhead_tol,
                )


class TestDots(utt.InferShapeTester):
    def setup_method(self):
        super().setup_method()
        x_size = (10, 100)
        y_size = (100, 1000)
        utt.seed_rng()

        self.x_csr = sp.sparse.csr_matrix(
            np.random.binomial(1, 0.5, x_size), dtype=aesara.config.floatX
        )
        self.x_csc = sp.sparse.csc_matrix(
            np.random.binomial(1, 0.5, x_size), dtype=aesara.config.floatX
        )
        self.y = np.asarray(
            np.random.uniform(-1, 1, y_size), dtype=aesara.config.floatX
        )
        self.y_csr = sp.sparse.csr_matrix(
            np.random.binomial(1, 0.5, y_size), dtype=aesara.config.floatX
        )
        self.y_csc = sp.sparse.csc_matrix(
            np.random.binomial(1, 0.5, y_size), dtype=aesara.config.floatX
        )
        self.v_10 = np.asarray(np.random.uniform(-1, 1, 10), dtype=aesara.config.floatX)
        self.v_100 = np.asarray(
            np.random.uniform(-1, 1, 100), dtype=aesara.config.floatX
        )

    def test_csr_dense(self):
        x = sparse.csr_matrix("x")
        y = matrix("y")
        v = vector("v")

        for (x, y, x_v, y_v) in [
            (x, y, self.x_csr, self.y),
            (x, v, self.x_csr, self.v_100),
            (v, x, self.v_10, self.x_csr),
        ]:
            f_a = aesara.function([x, y], sparse.dot(x, y))

            def f_b(x, y):
                return x * y

            utt.assert_allclose(f_a(x_v, y_v), f_b(x_v, y_v))

            # Test infer_shape
            self._compile_and_check(
                [x, y], [sparse.dot(x, y)], [x_v, y_v], (Dot, Usmm, UsmmCscDense)
            )

    def test_csc_dense(self):
        x = sparse.csc_matrix("x")
        y = matrix("y")
        v = vector("v")

        for (x, y, x_v, y_v) in [
            (x, y, self.x_csc, self.y),
            (x, v, self.x_csc, self.v_100),
            (v, x, self.v_10, self.x_csc),
        ]:

            f_a = aesara.function([x, y], sparse.dot(x, y))

            def f_b(x, y):
                return x * y

            utt.assert_allclose(f_a(x_v, y_v), f_b(x_v, y_v))

            # Test infer_shape
            self._compile_and_check(
                [x, y], [sparse.dot(x, y)], [x_v, y_v], (Dot, Usmm, UsmmCscDense)
            )

    def test_sparse_sparse(self):
        for d1, d2 in [
            ("float32", "float32"),
            ("float32", "float64"),
            ("float64", "float32"),
            ("float64", "float64"),
            ("float32", "int16"),
            ("float32", "complex64"),
        ]:
            for x_f, y_f in [
                ("csc", "csc"),
                ("csc", "csr"),
                ("csr", "csc"),
                ("csr", "csr"),
            ]:
                x = sparse.SparseType(format=x_f, dtype=d1)("x")
                y = sparse.SparseType(format=x_f, dtype=d2)("x")

                def f_a(x, y):
                    return x * y

                f_b = aesara.function([x, y], sparse.dot(x, y))

                vx = getattr(self, "x_" + x_f).astype(d1)
                vy = getattr(self, "y_" + y_f).astype(d2)
                utt.assert_allclose(f_a(vx, vy).toarray(), f_b(vx, vy))

                # Test infer_shape
                f_a = aesara.function([x, y], sparse.dot(x, y).shape)

                def f_b(x, y):
                    return (x * y).shape

                assert np.all(f_a(vx, vy) == f_b(vx, vy))
                topo = f_a.maker.fgraph.toposort()
                if aesara.config.mode != "FAST_COMPILE":
                    nb = 0
                else:
                    nb = 1
                assert (
                    sum(
                        [
                            isinstance(node.op, (Dot, Usmm, UsmmCscDense))
                            for node in topo
                        ]
                    )
                    == nb
                )

    def test_int32_dtype(self):
        # Reported on the theano-user mailing-list:
        # https://groups.google.com/d/msg/theano-users/MT9ui8LtTsY/rwatwEF9zWAJ
        size = 9
        intX = "int32"

        C = matrix("C", dtype=intX)
        I = matrix("I", dtype=intX)

        fI = I.flatten()
        data = aet.ones_like(fI)
        indptr = aet.arange(data.shape[0] + 1, dtype="int32")

        m1 = sparse.CSR(data, fI, indptr, (8, size))
        m2 = sparse.dot(m1, C)
        y = m2.reshape(shape=(2, 4, 9), ndim=3)

        f = aesara.function(inputs=[I, C], outputs=y)
        i = np.asarray([[4, 3, 7, 7], [2, 8, 4, 5]], dtype=intX)
        a = np.asarray(np.random.randint(0, 100, (size, size)), dtype=intX)
        f(i, a)

    def test_tensor_dot_types(self):

        x = sparse.csc_matrix("x")
        x_d = aet.matrix("x_d")
        y = sparse.csc_matrix("y")

        res = aet.dot(x, y)
        op_types = set(type(n.op) for n in applys_between([x, y], [res]))
        assert sparse.basic.StructuredDot in op_types
        assert aet.math.Dot not in op_types

        res = aet.dot(x_d, y)
        op_types = set(type(n.op) for n in applys_between([x, y], [res]))
        assert sparse.basic.StructuredDot in op_types
        assert aet.math.Dot not in op_types

        res = aet.dot(x, x_d)
        op_types = set(type(n.op) for n in applys_between([x, y], [res]))
        assert sparse.basic.StructuredDot in op_types
        assert aet.math.Dot not in op_types

        res = aet.dot(aet.second(1, x), y)
        op_types = set(type(n.op) for n in applys_between([x, y], [res]))
        assert sparse.basic.StructuredDot in op_types
        assert aet.math.Dot not in op_types

    # @aesara.config.change_flags(compute_test_value="raise", optimizer_verbose=True)
    def test_csr_dense_grad(self):

        # shortcut: testing csc in float32, testing csr in float64

        # allocate a random sparse matrix
        spmat = sp.sparse.csr_matrix(random_lil((4, 3), "float64", 3))

        mat = np.asarray(np.random.randn(2, 4), "float64")

        def buildgraph_T(mat):
            return Dot()(mat, spmat)

        utt.verify_grad(buildgraph_T, [mat])


class TestUsmm:
    """
    Test the Usmm and UsmmCscDense class and related optimization
    """

    def setup_method(self):
        x_size = (10, 100)
        y_size = (100, 200)
        z_size = (x_size[0], y_size[1])

        self.rng = np.random.RandomState(seed=utt.fetch_seed())
        self.x = np.asarray(
            self.rng.binomial(1, 0.5, x_size), dtype=aesara.config.floatX
        )
        self.y = np.asarray(self.rng.uniform(-1, 1, y_size), dtype=aesara.config.floatX)
        self.z = np.asarray(self.rng.uniform(-1, 1, z_size), dtype=aesara.config.floatX)

    @pytest.mark.slow
    def test_basic(self):
        # this is slow, but it's the only test for the op.
        def mat(format, name, dtype):
            if format == "dense":
                return matrix(name, dtype=dtype)
            else:
                return sparse.matrix(format, name, dtype=dtype)

        params = product(
            *(
                [["float32", "float64", "int16", "complex64"]] * 4
                + [["dense", "csc", "csr"]] * 2
            )
        )

        # All test are too slow, so we randomly take 100 of them.
        # The buildbot change the seed, so we will finish by running them all.
        # As of this writing they where all passing.
        # params = self.rng.permutation(list(params))[:500]

        for dtype1, dtype2, dtype3, dtype4, format1, format2 in params:
            if format1 == "dense" and format2 == "dense":
                # Usmm won't be used!
                continue
            x = mat(format1, "x", dtype1)
            y = mat(format2, "y", dtype2)
            a = scalar("a", dtype=dtype3)
            z = aesara.shared(np.asarray(self.z, dtype=dtype4).copy())

            def f_b(z, a, x, y):
                return z - a * (x * y)

            x_data = np.asarray(self.x, dtype=dtype1)
            if format1 != "dense":
                x_data = as_sparse_format(x_data, format1)
            y_data = np.asarray(self.y, dtype=dtype2)
            if format2 != "dense":
                y_data = as_sparse_format(y_data, format2)
            a_data = np.asarray(1.5, dtype=dtype3)
            z_data = np.asarray(self.z, dtype=dtype4)

            f_b_out = f_b(z_data, a_data, x_data, y_data)

            # Can it work inplace?
            inplace = dtype4 == aesara.scalar.upcast(dtype1, dtype2, dtype3)

            # To make it easier to check the toposort
            mode = aesara.compile.mode.get_default_mode().excluding("fusion")

            if inplace:
                updates = [(z, z - a * sparse.dot(x, y))]
                f_a = aesara.function([a, x, y], [], updates=updates, mode=mode)
                f_a(a_data, x_data, y_data)
                f_a_out = z.get_value(borrow=True)
            else:
                f_a = aesara.function([a, x, y], z - a * sparse.dot(x, y), mode=mode)
                # In DebugMode there is a strange difference with complex
                # So we raise a little the threshold a little.
                try:
                    orig_atol = aesara.tensor.math.float64_atol
                    orig_rtol = aesara.tensor.math.float64_rtol
                    aesara.tensor.math.float64_atol = 1e-7
                    aesara.tensor.math.float64_rtol = 1e-6
                    f_a_out = f_a(a_data, x_data, y_data)
                finally:
                    aesara.tensor.math.float64_atol = orig_atol
                    aesara.tensor.math.float64_rtol = orig_rtol

            # As we do a dot product of 2 vector of 100 element,
            # This mean we can have 2*100*eps abs error.
            if f_a_out.dtype in ["float64", "complex128"]:
                atol = 3e-8
                rtol = 1e-5
            else:
                atol = None
                rtol = None
            utt.assert_allclose(f_a_out, f_b_out, rtol=rtol, atol=atol)
            topo = f_a.maker.fgraph.toposort()
            up = aesara.scalar.upcast(dtype1, dtype2, dtype3, dtype4)

            fast_compile = aesara.config.mode == "FAST_COMPILE"

            if not aesara.config.blas__ldflags:
                # Usmm should not be inserted, because it relies on BLAS
                assert len(topo) == 4, topo
                assert isinstance(topo[0].op, sparse.Dot)
                assert isinstance(topo[1].op, DimShuffle)
                assert isinstance(topo[2].op, Elemwise) and isinstance(
                    topo[2].op.scalar_op, aesara.scalar.Mul
                )
                assert isinstance(topo[3].op, Elemwise) and isinstance(
                    topo[3].op.scalar_op, aesara.scalar.Sub
                )
            elif (
                y.type.dtype == up
                and format1 == "csc"
                and format2 == "dense"
                and not fast_compile
                and aesara.config.cxx
                and up in ("float32", "float64")
            ):
                # The op UsmmCscDense should be inserted
                assert (
                    sum(
                        [
                            isinstance(node.op, Elemwise)
                            and isinstance(node.op.scalar_op, aesara.scalar.basic.Cast)
                            for node in topo
                        ]
                    )
                    == len(topo) - 5
                )
                new_topo = []
                for node in topo:
                    if not (
                        isinstance(node.op, Elemwise)
                        and isinstance(node.op.scalar_op, aesara.scalar.basic.Cast)
                    ):
                        new_topo.append(node)
                topo = new_topo
                assert len(topo) == 5, topo
                # Usmm is tested at the same time in debugmode
                # Check if the optimization local_usmm and local_usmm_csx is
                # applied

                def check_once(x):
                    assert sum([isinstance(n.op, x) for n in topo]) == 1

                check_once(sparse.basic.CSMProperties)
                check_once(DimShuffle)
                check_once(Subtensor)
                check_once(UsmmCscDense)
                check_once(Elemwise)
                if inplace:
                    assert topo[4].op.inplace
            elif not fast_compile:
                # The op Usmm should be inserted
                assert len(topo) == 3, topo
                assert isinstance(topo[0].op, DimShuffle)
                assert topo[1].op == aesara.tensor.neg
                assert isinstance(topo[2].op, sparse.Usmm)

    def test_infer_shape(self):
        def mat(format, name, dtype):
            if format == "dense":
                return matrix(name, dtype=dtype)
            else:
                return sparse.matrix(format, name, dtype=dtype)

        params = [
            ("float32", "float64", "int16", "complex64", "csc", "dense"),
            ("float32", "float64", "int16", "complex64", "csr", "dense"),
        ]
        for dtype1, dtype2, dtype3, dtype4, format1, format2 in params:
            if format1 == "dense" and format2 == "dense":
                # Usmm won't be used!
                continue
            x = mat(format1, "x", dtype1)
            y = mat(format2, "y", dtype2)
            a = scalar("a", dtype=dtype3)
            z = aesara.shared(np.asarray(self.z, dtype=dtype4).copy())

            def f_b(z, a, x, y):
                return z - a * (x * y)

            x_data = np.asarray(self.x, dtype=dtype1)
            if format1 != "dense":
                x_data = as_sparse_format(x_data, format1)
            y_data = np.asarray(self.y, dtype=dtype2)
            if format2 != "dense":
                y_data = as_sparse_format(y_data, format2)
            a_data = np.asarray(1.5, dtype=dtype3)
            z_data = np.asarray(self.z, dtype=dtype4)

            f_b_out = f_b(z_data, a_data, x_data, y_data)

            # Can it work inplace?
            # inplace = dtype4 == aesara.scalar.upcast(dtype1, dtype2, dtype3)

            # To make it easier to check the toposort
            mode = aesara.compile.mode.get_default_mode().excluding("fusion")

            # test infer_shape of Dot got applied
            f_shape = aesara.function(
                [a, x, y], (z - a * sparse.dot(x, y)).shape, mode=mode
            )
            assert all(f_shape(a_data, x_data, y_data) == f_b_out.shape)
            topo = f_shape.maker.fgraph.toposort()
            if aesara.config.mode != "FAST_COMPILE":
                nb = 0
            else:
                nb = 1
            assert (
                sum([isinstance(node.op, (Dot, Usmm, UsmmCscDense)) for node in topo])
                == nb
            )


class TestZerosLike:
    def test(self):
        x = sparse.csr_matrix()
        f = aesara.function([x], sparse.sp_zeros_like(x))
        vx = sp.sparse.csr_matrix(
            np.asarray(
                np.random.binomial(1, 0.5, (100, 100)), dtype=aesara.config.floatX
            )
        )

        fx = f(vx)

        assert fx.nnz == 0
        assert fx.shape == vx.shape


def test_shape_i():
    sparse_dtype = "float32"

    a = SparseType("csr", dtype=sparse_dtype)()
    f = aesara.function([a], a.shape[1])
    assert f(sp.sparse.csr_matrix(random_lil((100, 10), sparse_dtype, 3))) == 10


def test_shape():
    # Test that getting the shape of a sparse variable
    # does not actually create a dense tensor in the process.
    sparse_dtype = "float32"

    a = SparseType("csr", dtype=sparse_dtype)()
    f = aesara.function([a], a.shape)
    assert np.all(
        f(sp.sparse.csr_matrix(random_lil((100, 10), sparse_dtype, 3))) == (100, 10)
    )
    if aesara.config.mode != "FAST_COMPILE":
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 3
        assert isinstance(topo[0].op, Shape_i)
        assert isinstance(topo[1].op, Shape_i)
        assert isinstance(topo[2].op, MakeVector)


def test_may_share_memory():
    a = sp.sparse.csc_matrix(sp.sparse.eye(5, 3))
    b = sp.sparse.csc_matrix(sp.sparse.eye(4, 3))

    def as_ar(a):
        return _asarray(a, dtype="int32")

    for a_, b_, rep in [
        (a, a, True),
        (b, b, True),
        (a, b, False),
        (a, a.data, True),
        (a, a.indptr, True),
        (a, a.indices, True),
        (a, as_ar(a.shape), False),
        (a.data, a, True),
        (a.indptr, a, True),
        (a.indices, a, True),
        (as_ar(a.shape), a, False),
        (b, b.data, True),
        (b, b.indptr, True),
        (b, b.indices, True),
        (b, as_ar(b.shape), False),
        (b.data, b, True),
        (b.indptr, b, True),
        (b.indices, b, True),
        (as_ar(b.shape), b, False),
        (b.data, a, False),
        (b.indptr, a, False),
        (b.indices, a, False),
        (as_ar(b.shape), a, False),
        (a.transpose(), a, True),
        (b.transpose(), b, True),
        (a.transpose(), b, False),
        (b.transpose(), a, False),
    ]:

        assert SparseType.may_share_memory(a_, b_) == rep


def test_sparse_shared_memory():
    # Note : There are no inplace ops on sparse matrix yet. If one is
    # someday implemented, we could test it here.
    a = random_lil((3, 4), "float32", 3).tocsr()
    m1 = random_lil((4, 4), "float32", 3).tocsr()
    m2 = random_lil((4, 4), "float32", 3).tocsr()
    x = SparseType("csr", dtype="float32")()
    y = SparseType("csr", dtype="float32")()

    sdot = sparse.structured_dot
    z = sdot(x * 3, m1) + sdot(y * 2, m2)

    f = aesara.function([In(x, mutable=True), In(y, mutable=True)], z, mode="FAST_RUN")

    def f_(x, y, m1=m1, m2=m2):
        return ((x * 3) * m1) + ((y * 2) * m2)

    assert SparseType.may_share_memory(a, a)  # This is trivial
    result = f(a, a)
    result_ = f_(a, a)
    assert (result_.todense() == result.todense()).all()


def test_size():
    # Ensure the `size` attribute of sparse matrices behaves as in numpy.

    for sparse_type in ("csc_matrix", "csr_matrix"):
        x = getattr(aesara.sparse, sparse_type)()
        y = getattr(sp.sparse, sparse_type)((5, 7)).astype(config.floatX)
        get_size = aesara.function([x], x.size)

        def check():
            assert y.size == get_size(y)

        # We verify that the size is correctly updated as we store more data
        # into the sparse matrix (including zeros).
        check()
        y[0, 0] = 1
        check()
        y[0, 1] = 0
        check()


class TestColScaleCSC(utt.InferShapeTester):
    def setup_method(self):
        super().setup_method()
        self.op = sparse.col_scale

    def test_op(self):
        for format in sparse.sparse_formats:
            variable, data = sparse_random_inputs(format, shape=(8, 10))
            variable.append(vector())
            data.append(np.random.random(10).astype(config.floatX))

            f = aesara.function(variable, self.op(*variable))

            tested = f(*data)
            x, s = data[0].toarray(), data[1][np.newaxis, :]
            expected = x * s

            assert tested.format == format
            utt.assert_allclose(expected, tested.toarray())

    def test_infer_shape(self):
        for format, cls in [("csc", sparse.ColScaleCSC), ("csr", sparse.RowScaleCSC)]:
            variable, data = sparse_random_inputs(format, shape=(8, 10))
            variable.append(vector())
            data.append(np.random.random(10).astype(config.floatX))

            self._compile_and_check(variable, [self.op(*variable)], data, cls)

    def test_grad(self):
        for format in sparse.sparse_formats:
            variable, data = sparse_random_inputs(format, shape=(8, 10))
            variable.append(vector())
            data.append(np.random.random(10).astype(config.floatX))

            verify_grad_sparse(self.op, data, structured=True)


class TestRowScaleCSC(utt.InferShapeTester):
    def setup_method(self):
        super().setup_method()
        self.op = sparse.row_scale

    def test_op(self):
        for format in sparse.sparse_formats:
            variable, data = sparse_random_inputs(format, shape=(8, 10))
            variable.append(vector())
            data.append(np.random.random(8).astype(config.floatX))

            f = aesara.function(variable, self.op(*variable))

            tested = f(*data)
            x, s = data[0].toarray(), data[1][:, np.newaxis]
            expected = x * s

            assert tested.format == format
            utt.assert_allclose(expected, tested.toarray())

    def test_infer_shape(self):
        for format, cls in [("csc", sparse.RowScaleCSC), ("csr", sparse.ColScaleCSC)]:
            variable, data = sparse_random_inputs(format, shape=(8, 10))
            variable.append(vector())
            data.append(np.random.random(8).astype(config.floatX))

            self._compile_and_check(variable, [self.op(*variable)], data, cls)

    def test_grad(self):
        for format in sparse.sparse_formats:
            variable, data = sparse_random_inputs(format, shape=(8, 10))
            variable.append(vector())
            data.append(np.random.random(8).astype(config.floatX))

            verify_grad_sparse(self.op, data, structured=True)


class TestSpSum(utt.InferShapeTester):
    possible_axis = [None, 0, 1]

    def setup_method(self):
        super().setup_method()
        self.op_class = sparse.SpSum
        self.op = sparse.sp_sum

    def test_op(self):
        for format in sparse.sparse_formats:
            for axis in self.possible_axis:
                variable, data = sparse_random_inputs(format, shape=(10, 10))

                z = sparse.sp_sum(variable[0], axis=axis)
                if axis is None:
                    assert z.type.broadcastable == ()
                else:
                    assert z.type.broadcastable == (False,)

                f = aesara.function(variable, self.op(variable[0], axis=axis))
                tested = f(*data)
                expected = data[0].todense().sum(axis).ravel()
                utt.assert_allclose(expected, tested)

    def test_infer_shape(self):
        for format in sparse.sparse_formats:
            for axis in self.possible_axis:
                variable, data = sparse_random_inputs(format, shape=(9, 10))
                self._compile_and_check(
                    variable, [self.op(variable[0], axis=axis)], data, self.op_class
                )

    def test_grad(self):
        for format in sparse.sparse_formats:
            for axis in self.possible_axis:
                for struct in [True, False]:
                    variable, data = sparse_random_inputs(format, shape=(9, 10))
                    verify_grad_sparse(
                        self.op_class(axis=axis, sparse_grad=struct),
                        data,
                        structured=struct,
                    )


class TestDiag(utt.InferShapeTester):
    def setup_method(self):
        super().setup_method()
        self.op_class = Diag
        self.op = diag

    def test_op(self):
        for format in sparse.sparse_formats:
            variable, data = sparse_random_inputs(format, shape=(10, 10))

            z = self.op(*variable)
            assert z.type.broadcastable == (False,)

            f = aesara.function(variable, z)
            tested = f(*data)
            expected = data[0].toarray().diagonal()

            utt.assert_allclose(expected, tested)

    def test_infer_shape(self):
        for format in sparse.sparse_formats:
            variable, data = sparse_random_inputs(format, shape=(10, 10))
            self._compile_and_check(
                variable, [self.op(*variable)], data, self.op_class, warn=False
            )

    def test_grad(self):
        for format in sparse.sparse_formats:
            variable, data = sparse_random_inputs(format, shape=(10, 10))
            verify_grad_sparse(self.op, data, structured=False)


class TestSquareDiagonal(utt.InferShapeTester):
    def setup_method(self):
        super().setup_method()
        self.op_class = SquareDiagonal
        self.op = square_diagonal

    def test_op(self):
        for format in sparse.sparse_formats:
            for size in range(5, 9):
                variable = [vector()]
                data = [np.random.random(size).astype(config.floatX)]

                f = aesara.function(variable, self.op(*variable))
                tested = f(*data).toarray()

                expected = np.diag(*data)
                utt.assert_allclose(expected, tested)
                assert tested.dtype == expected.dtype
                assert tested.shape == expected.shape

    def test_infer_shape(self):
        for format in sparse.sparse_formats:
            for size in range(5, 9):
                variable = [vector()]
                data = [np.random.random(size).astype(config.floatX)]

                self._compile_and_check(
                    variable, [self.op(*variable)], data, self.op_class
                )

    def test_grad(self):
        for format in sparse.sparse_formats:
            for size in range(5, 9):
                data = [np.random.random(size).astype(config.floatX)]

                verify_grad_sparse(self.op, data, structured=False)


class TestEnsureSortedIndices(utt.InferShapeTester):
    def setup_method(self):
        super().setup_method()
        self.op_class = EnsureSortedIndices
        self.op = ensure_sorted_indices

    def test_op(self):
        for format in sparse.sparse_formats:
            for shape in zip(range(5, 9), range(3, 7)[::-1]):
                variable, data = sparse_random_inputs(format, shape=shape)

                f = aesara.function(variable, self.op(*variable))
                tested = f(*data).toarray()
                expected = data[0].sorted_indices().toarray()

                utt.assert_allclose(expected, tested)

    def test_infer_shape(self):
        for format in sparse.sparse_formats:
            for shape in zip(range(5, 9), range(3, 7)[::-1]):
                variable, data = sparse_random_inputs(format, shape=shape)
                self._compile_and_check(
                    variable, [self.op(*variable)], data, self.op_class
                )

    def test_grad(self):
        for format in sparse.sparse_formats:
            for shape in zip(range(5, 9), range(3, 7)[::-1]):
                variable, data = sparse_random_inputs(format, shape=shape)
                verify_grad_sparse(self.op, data, structured=False)


class TestClean(utt.InferShapeTester):
    def setup_method(self):
        super().setup_method()
        self.op = clean

    def test_op(self):
        for format in sparse.sparse_formats:
            for shape in zip(range(5, 9), range(3, 7)[::-1]):
                variable, data = sparse_random_inputs(format, shape=shape)

                data[0][0, 0] = data[0][1, 1] = 0

                f = aesara.function(variable, self.op(*variable))
                tested = f(*data)
                expected = data[0]
                expected.eliminate_zeros()

                assert all(tested.data == expected.data)
                assert not all(tested.data == 0)

                tested = tested.toarray()
                expected = expected.toarray()
                utt.assert_allclose(expected, tested)

    def test_grad(self):
        for format in sparse.sparse_formats:
            for shape in zip(range(5, 9), range(3, 7)[::-1]):
                variable, data = sparse_random_inputs(format, shape=shape)
                verify_grad_sparse(self.op, data, structured=False)


class TestRemove0(utt.InferShapeTester):
    def setup_method(self):
        super().setup_method()
        self.op_class = Remove0

    def test_remove0(self):
        configs = [
            # structure type, numpy matching class
            ("csc", sp.sparse.csc_matrix),
            ("csr", sp.sparse.csr_matrix),
        ]

        for format, matrix_class in configs:
            for zero, unsor in [
                (True, True),
                (True, False),
                (False, True),
                (False, False),
            ]:
                (x,), (mat,) = sparse_random_inputs(
                    format,
                    (6, 8),
                    out_dtype=config.floatX,
                    explicit_zero=zero,
                    unsorted_indices=unsor,
                )
                assert 0 in mat.data or not zero
                assert not mat.has_sorted_indices or not unsor

                # the In thingy has to be there because aesara has as rule not
                # to optimize inputs
                f = aesara.function([In(x, borrow=True, mutable=True)], Remove0()(x))

                # assert optimization local_inplace_remove0 is applied in
                # modes with optimization
                if aesara.config.mode not in ["FAST_COMPILE"]:
                    # list of apply nodes in the optimized graph.
                    nodes = f.maker.fgraph.toposort()
                    # Check there isn't any Remove0 instance not inplace.
                    assert not any(
                        [
                            isinstance(node.op, Remove0) and not node.op.inplace
                            for node in nodes
                        ]
                    ), "Inplace optimization should have been applied"
                    # Check there is at least one Remove0 inplace.
                    assert any(
                        [
                            isinstance(node.op, Remove0) and node.op.inplace
                            for node in nodes
                        ]
                    )
                # checking
                # makes sense to change its name
                target = mat
                result = f(mat)
                mat.eliminate_zeros()
                msg = "Matrices sizes differ. Have zeros been removed ?"
                assert result.size == target.size, msg
                if unsor:
                    assert not result.has_sorted_indices
                    assert not target.has_sorted_indices
                else:
                    assert result.has_sorted_indices
                    assert target.has_sorted_indices

    def test_infer_shape(self):
        mat = (np.arange(12) + 1).reshape((4, 3))
        mat[0, 1] = mat[1, 0] = mat[2, 2] = 0

        x_csc = sparse.csc_matrix(dtype=aesara.config.floatX)
        mat_csc = sp.sparse.csc_matrix(mat, dtype=aesara.config.floatX)
        self._compile_and_check([x_csc], [Remove0()(x_csc)], [mat_csc], self.op_class)

        x_csr = sparse.csr_matrix(dtype=aesara.config.floatX)
        mat_csr = sp.sparse.csr_matrix(mat, dtype=aesara.config.floatX)
        self._compile_and_check([x_csr], [Remove0()(x_csr)], [mat_csr], self.op_class)

    def test_grad(self):
        mat = (np.arange(9) + 1).reshape((3, 3))
        mat[0, 1] = mat[1, 0] = mat[2, 2] = 0

        mat_csc = sp.sparse.csc_matrix(mat, dtype=aesara.config.floatX)
        verify_grad_sparse(Remove0(), [mat_csc])

        mat_csr = sp.sparse.csr_matrix(mat, dtype=aesara.config.floatX)
        verify_grad_sparse(Remove0(), [mat_csr])


class TestGetItem:
    def setup_method(self):
        self.rng = np.random.RandomState(utt.fetch_seed())

    def test_GetItemList(self):

        a, A = sparse_random_inputs("csr", (4, 5))
        b, B = sparse_random_inputs("csc", (4, 5))
        y = a[0][[0, 1, 2, 3, 1]]
        z = b[0][[0, 1, 2, 3, 1]]

        fa = aesara.function([a[0]], y)
        fb = aesara.function([b[0]], z)

        t_geta = fa(A[0]).todense()
        t_getb = fb(B[0]).todense()

        s_geta = sp.sparse.csr_matrix(A[0])[[0, 1, 2, 3, 1]].todense()
        s_getb = sp.sparse.csc_matrix(B[0])[[0, 1, 2, 3, 1]].todense()

        utt.assert_allclose(t_geta, s_geta)
        utt.assert_allclose(t_getb, s_getb)

    def test_GetItemList_wrong_index(self):
        a, A = sparse_random_inputs("csr", (4, 5))
        y = a[0][[0, 4]]
        f = aesara.function([a[0]], y)

        with pytest.raises(IndexError):
            f(A[0])

    def test_get_item_list_grad(self):
        op = sparse.basic.GetItemList()

        def op_with_fixed_index(x):
            return op(x, index=np.asarray([0, 1]))

        x, x_val = sparse_random_inputs("csr", (4, 5))

        try:
            verify_grad_sparse(op_with_fixed_index, x_val)
        except NotImplementedError as e:
            assert "Scipy version is to old" in str(e)

    def test_GetItem2Lists(self):

        a, A = sparse_random_inputs("csr", (4, 5))
        b, B = sparse_random_inputs("csc", (4, 5))
        y = a[0][[0, 0, 1, 3], [0, 1, 2, 4]]
        z = b[0][[0, 0, 1, 3], [0, 1, 2, 4]]

        fa = aesara.function([a[0]], y)
        fb = aesara.function([b[0]], z)

        t_geta = fa(A[0])
        t_getb = fb(B[0])

        s_geta = np.asarray(sp.sparse.csr_matrix(A[0])[[0, 0, 1, 3], [0, 1, 2, 4]])
        s_getb = np.asarray(sp.sparse.csc_matrix(B[0])[[0, 0, 1, 3], [0, 1, 2, 4]])

        utt.assert_allclose(t_geta, s_geta)
        utt.assert_allclose(t_getb, s_getb)

    def test_GetItem2Lists_wrong_index(self):
        a, A = sparse_random_inputs("csr", (4, 5))
        y1 = a[0][[0, 5], [0, 3]]
        y2 = a[0][[0, 3], [0, 5]]

        f1 = aesara.function([a[0]], y1)
        f2 = aesara.function([a[0]], y2)

        with pytest.raises(IndexError):
            f1(A[0])
        with pytest.raises(IndexError):
            f2(A[0])

    def test_get_item_2lists_grad(self):
        op = sparse.basic.GetItem2Lists()

        def op_with_fixed_index(x):
            return op(x, ind1=np.asarray([0, 1]), ind2=np.asarray([2, 3]))

        x, x_val = sparse_random_inputs("csr", (4, 5))

        verify_grad_sparse(op_with_fixed_index, x_val)

    def test_GetItem2D(self):
        is_supported_version = version.parse(sp.__version__) >= version.parse("0.14")

        sparse_formats = ("csc", "csr")
        for format in sparse_formats:
            x = sparse.matrix(format, name="x")
            a = iscalar("a")
            b = iscalar("b")
            c = iscalar("c")
            d = iscalar("d")
            e = iscalar("e")
            f = iscalar("f")

            # index
            m = 1
            n = 5
            p = 10
            q = 15
            if is_supported_version:
                j = 2
                k = 3
            else:
                j = None
                k = None

            vx = as_sparse_format(self.rng.binomial(1, 0.5, (100, 97)), format).astype(
                aesara.config.floatX
            )

            # mode_no_debug = aesara.compile.mode.get_default_mode()
            # if isinstance(mode_no_debug, aesara.compile.debugmode.DebugMode):
            #    mode_no_debug = 'FAST_RUN'
            if is_supported_version:
                f1 = aesara.function([x, a, b, c, d, e, f], x[a:b:e, c:d:f])
                r1 = f1(vx, m, n, p, q, j, k)
                t1 = vx[m:n:j, p:q:k]
            else:
                f1 = aesara.function([x, a, b, c, d], x[a:b, c:d])
                r1 = f1(vx, m, n, p, q)
                t1 = vx[m:n, p:q]
            assert r1.shape == t1.shape
            assert np.all(t1.toarray() == r1.toarray())

            """
            Important: based on a discussion with both Fred and James
            The following indexing methods is not supported because the rval
            would be a sparse matrix rather than a sparse vector, which is a
            deviation from numpy indexing rule. This decision is made largely
            for keeping the consistency between numpy and aesara.

            f2 = aesara.function([x, a, b, c], x[a:b, c])
            r2 = f2(vx, m, n, p)
            t2 = vx[m:n, p]
            assert r2.shape == t2.shape
            assert np.all(t2.toarray() == r2.toarray())

            f3 = aesara.function([x, a, b, c], x[a, b:c])
            r3 = f3(vx, m, n, p)
            t3 = vx[m, n:p]
            assert r3.shape == t3.shape
            assert np.all(t3.toarray() == r3.toarray())

            f5 = aesara.function([x], x[1:2,3])
            r5 = f5(vx)
            t5 = vx[1:2, 3]
            assert r5.shape == t5.shape
            assert np.all(r5.toarray() == t5.toarray())

            f7 = aesara.function([x], x[50])
            r7 = f7(vx)
            t7 = vx[50]
            assert r7.shape == t7.shape
            assert np.all(r7.toarray() == t7.toarray())
            """
            if is_supported_version:
                f4 = aesara.function([x, a, b, e], x[a:b:e])
                r4 = f4(vx, m, n, j)
                t4 = vx[m:n:j]
            else:
                f4 = aesara.function([x, a, b], x[a:b])
                r4 = f4(vx, m, n)
                t4 = vx[m:n]
            assert r4.shape == t4.shape
            assert np.all(t4.toarray() == r4.toarray())

            # -----------------------------------------------------------
            # test cases using int indexing instead of aesara variable
            f6 = aesara.function([x], x[1:10:j, 10:20:k])
            r6 = f6(vx)
            t6 = vx[1:10:j, 10:20:k]
            assert r6.shape == t6.shape
            assert np.all(r6.toarray() == t6.toarray())

            # ----------------------------------------------------------
            # test cases with indexing both with aesara variable and int
            if is_supported_version:
                f8 = aesara.function([x, a, b, e], x[a:b:e, 10:20:1])
                r8 = f8(vx, m, n, j)
                t8 = vx[m:n:j, 10:20:1]
            else:
                f8 = aesara.function([x, a, b], x[a:b, 10:20])
                r8 = f8(vx, m, n)
                t8 = vx[m:n, 10:20]
            assert r8.shape == t8.shape
            assert np.all(r8.toarray() == t8.toarray())

            f9 = aesara.function([x, a, b], x[1:a:j, 1:b:k])
            r9 = f9(vx, p, q)
            t9 = vx[1:p:j, 1:q:k]
            assert r9.shape == t9.shape
            assert np.all(r9.toarray() == t9.toarray())

            # -----------------------------------------------------------
            # Test mixing None and variables
            f10 = aesara.function([x, a, b], x[:a, :b])
            r10 = f10(vx, p, q)
            t10 = vx[:p, :q]
            assert r10.shape == t10.shape
            assert np.all(r10.toarray() == t10.toarray())

            f11 = aesara.function([x, a], x[:, a:])
            r11 = f11(vx, p)
            t11 = vx[:, p:]
            assert r11.shape == t11.shape
            assert np.all(r11.toarray() == t11.toarray())

            # Test that is work with shared variable
            sx = aesara.shared(vx)
            f12 = aesara.function([a], sx[:, a:])
            r12 = f12(p)
            t12 = vx[:, p:]
            assert r12.shape == t12.shape
            assert np.all(r12.toarray() == t12.toarray())

            # ------------------------------------------------------------
            # Invalid things
            # The syntax is a bit awkward because assertRaises forbids
            # the [] shortcut for getitem.
            # x[a:b] is not accepted because we don't have sparse vectors
            with pytest.raises(NotImplementedError):
                x.__getitem__((slice(a, b), c))

            # x[a:b:step, c:d] is not accepted because scipy silently drops
            # the step (!)
            if not is_supported_version:
                with pytest.raises(ValueError):
                    x.__getitem__((slice(a, b, -1), slice(c, d)))
                with pytest.raises(ValueError):
                    x.__getitem__((slice(a, b), slice(c, d, 2)))

                # Advanced indexing is not supported
                with pytest.raises(ValueError):
                    x.__getitem__((ivector("l"), slice(a, b)))

                # Indexing with random things is not supported either
                with pytest.raises(ValueError):
                    x.__getitem__(slice(fscalar("f"), None))
                with pytest.raises(ValueError):
                    x.__getitem__((slice(None), slice([1, 3, 4], None)))

    def test_GetItemScalar(self):
        sparse_formats = ("csc", "csr")
        for format in sparse_formats:
            x = sparse.csc_matrix("x")
            a = iscalar()
            b = iscalar()

            m = 50
            n = 42

            vx = as_sparse_format(self.rng.binomial(1, 0.5, (97, 100)), format).astype(
                aesara.config.floatX
            )

            f1 = aesara.function([x, a, b], x[a, b])
            r1 = f1(vx, 10, 10)
            t1 = vx[10, 10]
            assert r1.shape == t1.shape
            assert np.all(t1 == r1)

            f2 = aesara.function([x, a], x[50, a])
            r2 = f2(vx, m)
            t2 = vx[50, m]
            assert r2.shape == t2.shape
            assert np.all(t2 == r2)

            f3 = aesara.function([x, a], x[a, 50])
            r3 = f3(vx, m)
            t3 = vx[m, 50]
            assert r3.shape == t3.shape
            assert np.all(t3 == r3)

            f4 = aesara.function([x], x[50, 42])
            r4 = f4(vx)
            t4 = vx[m, n]
            assert r3.shape == t3.shape
            assert np.all(t4 == r4)

            # Test that is work with shared variable
            sx = aesara.shared(vx)
            f1 = aesara.function([a, b], sx[a, b])
            r1 = f1(10, 10)
            t1 = vx[10, 10]
            assert r1.shape == t1.shape
            assert np.all(t1 == r1)


class TestCasting(utt.InferShapeTester):
    def setup_method(self):
        super().setup_method()

    # slow but only test
    def test_cast(self):
        for format in sparse.sparse_formats:
            for i_dtype in sparse.all_dtypes:
                for o_dtype in sparse.all_dtypes:
                    (variable,), (data,) = sparse_random_inputs(
                        format, shape=(4, 7), out_dtype=i_dtype
                    )

                    func = aesara.function([variable], cast(variable, o_dtype))
                    cls = aesara.function([variable], Cast(o_dtype)(variable))
                    prop = aesara.function([variable], variable.astype(o_dtype))

                    t_func, t_cls, t_prop = func(data), cls(data), prop(data)

                    expected = data.toarray().astype(o_dtype)

                    assert t_func.format == format
                    assert t_cls.format == format
                    assert t_prop.format == format

                    t_func = t_func.toarray()
                    t_cls = t_cls.toarray()
                    t_prop = t_prop.toarray()

                    utt.assert_allclose(expected, t_func)
                    utt.assert_allclose(expected, t_cls)
                    utt.assert_allclose(expected, t_prop)

    @pytest.mark.slow
    def test_infer_shape(self):
        for format in sparse.sparse_formats:
            for i_dtype in sparse.all_dtypes:
                for o_dtype in sparse.all_dtypes:
                    variable, data = sparse_random_inputs(
                        format, shape=(4, 7), out_dtype=i_dtype
                    )
                    self._compile_and_check(
                        variable, [Cast(o_dtype)(*variable)], data, Cast
                    )

    def test_grad(self):
        for format in sparse.sparse_formats:
            for i_dtype in sparse.float_dtypes:
                for o_dtype in float_dtypes:
                    if o_dtype == "float16":
                        # Don't test float16 output.
                        continue
                    _, data = sparse_random_inputs(
                        format, shape=(4, 7), out_dtype=i_dtype
                    )

                    eps = None
                    if o_dtype == "float32":
                        eps = 1e-2

                    verify_grad_sparse(Cast(o_dtype), data, eps=eps)


def _format_info(nb):
    x = {}
    mat = {}

    for format in sparse.sparse_formats:
        variable = getattr(aesara.sparse, format + "_matrix")
        spa = getattr(sp.sparse, format + "_matrix")

        x[format] = [variable() for t in range(nb)]
        mat[format] = [
            spa(random_lil((3, 4), aesara.config.floatX, 8)) for t in range(nb)
        ]
    return x, mat


class _TestHVStack(utt.InferShapeTester):
    """
    Test for both HStack and VStack.
    """

    nb = 3  # Number of sparse matrix to stack
    x, mat = _format_info(nb)

    def test_op(self):
        for format in sparse.sparse_formats:
            for out_f in sparse.sparse_formats:
                for dtype in sparse.all_dtypes:
                    blocks = self.mat[format]

                    f = aesara.function(
                        self.x[format],
                        self.op_class(format=out_f, dtype=dtype)(*self.x[format]),
                        allow_input_downcast=True,
                    )

                    tested = f(*blocks)
                    expected = self.expected_f(blocks, format=out_f, dtype=dtype)

                    utt.assert_allclose(expected.toarray(), tested.toarray())
                    assert tested.format == expected.format
                    assert tested.dtype == expected.dtype

    def test_infer_shape(self):
        for format in sparse.sparse_formats:
            self._compile_and_check(
                self.x[format],
                [self.op_class(dtype="float64")(*self.x[format])],
                self.mat[format],
                self.op_class,
            )

    def test_grad(self):
        for format in sparse.sparse_formats:
            for out_f in sparse.sparse_formats:
                for dtype in sparse.float_dtypes:
                    verify_grad_sparse(
                        self.op_class(format=out_f, dtype=dtype),
                        self.mat[format],
                        structured=False,
                        eps=1e-2,
                    )


def _hv_switch(op, expected_function):
    """
    Return the right test class for HStack or VStack.

    :Parameters:
    - `op`: HStack or VStack class.
    - `expected_function`: function from scipy for comparaison.
    """

    class TestXStack(_TestHVStack):
        op_class = op

        def expected_f(self, a, format=None, dtype=None):
            return expected_function(a, format, dtype)

    TestXStack.__name__ = op.__name__ + "Tester"
    if hasattr(TestXStack, "__qualname__"):
        TestXStack.__qualname__ = TestXStack.__name__
    return TestXStack


TestHStack = _hv_switch(HStack, sp.sparse.hstack)
TestVStack = _hv_switch(VStack, sp.sparse.vstack)


class TestAddSSData(utt.InferShapeTester):
    x = {}
    a = {}

    def setup_method(self):
        super().setup_method()
        self.op_class = AddSSData

        for format in sparse.sparse_formats:
            variable = getattr(aesara.sparse, format + "_matrix")

            rand = np.array(
                np.random.randint(1, 4, size=(3, 4)) - 1, dtype=aesara.config.floatX
            )
            constant = as_sparse_format(rand, format)

            self.x[format] = [variable() for t in range(2)]
            self.a[format] = [constant for t in range(2)]

    def test_op(self):
        for format in sparse.sparse_formats:
            f = aesara.function(self.x[format], add_s_s_data(*self.x[format]))

            tested = f(*self.a[format])
            expected = 2 * self.a[format][0]

            utt.assert_allclose(expected.toarray(), tested.toarray())
            assert tested.format == expected.format
            assert tested.dtype == expected.dtype

    def test_infer_shape(self):
        for format in sparse.sparse_formats:
            self._compile_and_check(
                self.x[format],
                [add_s_s_data(*self.x[format])],
                self.a[format],
                self.op_class,
            )

    def test_grad(self):
        for format in sparse.sparse_formats:
            verify_grad_sparse(self.op_class(), self.a[format], structured=True)


def elemwise_checker(
    op, expected_f, gap=None, test_dtypes=None, grad_test=True, name=None, gap_grad=None
):
    """
    Return the appropriate test class for the elemwise on sparse.

    :param op: Op to test.
    :expected_f: Function use to compare. This function must act
                 on dense matrix. If the op is structured
                 see the `structure_function` decorator to make
                 this function structured.
    :param gap: Tuple for the range of the random sample. When
                length is 1, it is assumed to be the exclusive
                max, when `gap` = (`a`, `b`) it provide a sample
                from [a, b[. If `None` is used, it provide [0, 1]
                for float dtypes and [0, 50[ for integer dtypes.
    :param test_dtypes: Particular dtypes for testing the op.
                        If `None`, this is set to the most common
                        dtypes.
    :param grad_test: True for testing the grad. False will
                      skip this test.
    :param gap_grad: If None, we reuse gap. Otherwise it is the same as gap
                     but for testing the gradiant of the op.

    :return: The class that perform the tests, not an instance
             of the class.
    """

    if test_dtypes is None:
        test_dtypes = sparse.all_dtypes

    class TestElemwise:
        def setup_method(self):
            super().setup_method()
            self.op = op
            self.expected_f = expected_f
            self.gap = gap
            if gap_grad is not None:
                self.gap_grad = gap_grad
            else:
                self.gap_grad = gap
            # Ensure the test's name is correct.
            utt.seed_rng()
            assert eval(self.__class__.__name__) is self.__class__

        def test_op(self):
            for format in sparse.sparse_formats:
                for dtype in test_dtypes:
                    if dtype == "int8" or dtype == "uint8":
                        continue

                    # When testing with unsigned integers,
                    # we must check if the gap contains
                    # negative numbers.
                    if dtype.startswith("uint"):
                        if self.gap and len(self.gap) == 2 and self.gap[0] < 0:
                            if self.gap[1] >= 1:
                                self.gap = (0, self.gap[1])
                            else:
                                raise TypeError(
                                    "Gap not suitable for", dtype, self.__name__
                                )

                    variable, data = sparse_random_inputs(
                        format, shape=(4, 7), out_dtype=dtype, gap=self.gap
                    )

                    f = aesara.function(variable, self.op(*variable))

                    tested = f(*data)
                    data = [m.toarray() for m in data]
                    expected = self.expected_f(*data)

                    assert tested.format == format
                    tested = tested.toarray()

                    try:
                        utt.assert_allclose(expected, tested)
                    except AssertionError:
                        raise AssertionError(self.__name__)

                # Test with int8 as dtype
                # These tests are not in the loop for two reasons.
                # First, in recent version of numpy, when a numpy
                # function have int8 as input dtype, it returns a
                # float16 as output dtype. Since this does not provide
                # enough precision, we upcast the data before we apply the
                # function.
                # Second, the tolerance for the checkup in DebugMode
                # is too high.
                for dtype in ["int8", "uint8"]:
                    if dtype in test_dtypes:
                        if self.gap:
                            domain = self.gap
                            # When testing with unsigned integers,
                            # we must check if the gap contains
                            # negative numbers.
                            if dtype == "uint8":
                                if len(domain) == 2 and domain[0] < 0:
                                    if domain[1] >= 1:
                                        domain = (0, domain[1])
                                    else:
                                        raise TypeError(
                                            "Gap not suitable for", dtype, self.__name__
                                        )

                        else:
                            domain = (0, 5)

                        variable, data = sparse_random_inputs(
                            format, shape=(4, 7), out_dtype=dtype, gap=domain
                        )

                        f = aesara.function(variable, self.op(*variable))

                        old_value = (
                            tensor.math.float32_atol,
                            tensor.math.float32_rtol,
                            tensor.math.float64_atol,
                            tensor.math.float64_rtol,
                        )
                        tensor.math.float32_atol = 1e-4
                        tensor.math.float32_rtol = 1e-3
                        tensor.math.float64_atol = 1e-3
                        tensor.math.float64_rtol = 1e-4
                        try:
                            tested = f(*data)
                        finally:
                            (
                                tensor.math.float32_atol,
                                tensor.math.float32_rtol,
                                tensor.math.float64_atol,
                                tensor.math.float64_rtol,
                            ) = old_value

                        data = [m.toarray().astype("float32") for m in data]
                        expected = self.expected_f(*data)

                        assert tested.format == format
                        tested = tested.toarray()

                        try:
                            utt.assert_allclose(tested, expected, rtol=1e-2)
                        except AssertionError:
                            raise AssertionError(self.__name__)

        if grad_test:

            def test_grad(self):
                for format in sparse.sparse_formats:
                    for dtype in sparse.float_dtypes:
                        variable, data = sparse_random_inputs(
                            format, shape=(4, 7), out_dtype=dtype, gap=self.gap_grad
                        )

                        verify_grad_sparse(self.op, data, structured=True)

    # Set proper class name to uniquely identify tests.
    # Note that it is important to run this code *outside* of the `Tester`
    # class itself, otherwise it will not work properly for some reason.
    if name is None:
        name = op.__name__.capitalize() + "Tester"
    TestElemwise.__name__ = name
    if hasattr(TestElemwise, "__qualname__"):
        TestElemwise.__qualname__ = name
    assert "Roundhalftoeven" not in TestElemwise.__name__

    return TestElemwise


def test_hstack_vstack():
    # Tests sparse.hstack and sparse.vstack (as opposed to the HStack and VStack
    # classes that they wrap).

    def make_block(dtype):
        return sparse.csr_matrix(name=f"{dtype} block", dtype=dtype)

    def get_expected_dtype(blocks, to_dtype):
        if to_dtype is None:
            block_dtypes = tuple(b.dtype for b in blocks)
            return aesara.scalar.upcast(*block_dtypes)
        else:
            return to_dtype

    # a deliberately weird mix of dtypes to stack
    dtypes = ("complex128", aesara.config.floatX)

    blocks = [make_block(dtype) for dtype in dtypes]

    for stack_dimension, stack_function in enumerate((sparse.vstack, sparse.hstack)):

        for to_dtype in (None,) + dtypes:
            stacked_blocks = stack_function(blocks, dtype=to_dtype)
            expected_dtype = get_expected_dtype(blocks, to_dtype)
            assert stacked_blocks.dtype == expected_dtype


def structure_function(f, index=0):
    """
    Decorator to structure a function which
    apply on dense matrix.

    Here, the inputs of the function must be
    dense matrix. The sparse pattern is
    determined by finding the zeros.

    :param index: The index of the parameter
                  from which the function must
                  be structured.

    :return: The structured function for its
             `index` parameter.
    """

    def structured_function(*args):
        pattern = args[index]
        evaluated = f(*args)
        evaluated[pattern == 0] = 0
        return evaluated

    return structured_function


StructuredSigmoidTester = elemwise_checker(
    sparse.structured_sigmoid,
    structure_function(lambda x: 1.0 / (1.0 + np.exp(-x))),
    test_dtypes=[
        m
        for m in sparse.all_dtypes
        if (m not in sparse.complex_dtypes and not m.startswith("uint"))
    ],
    gap=(-5, 5),
    name="StructuredSigmoidTester",
)

StructuredExpTester = elemwise_checker(
    sparse.structured_exp, structure_function(np.exp), name="StructuredExpTester"
)

StructuredLogTester = elemwise_checker(
    sparse.structured_log,
    structure_function(np.log),
    gap=(0.5, 10),
    name="StructuredLogTester",
)

StructuredPowTester = elemwise_checker(
    lambda x: sparse.structured_pow(x, 2),
    structure_function(lambda x: np.power(x, 2)),
    name="StructuredPowTester",
)

StructuredMinimumTester = elemwise_checker(
    lambda x: structured_minimum(x, 2),
    structure_function(lambda x: np.minimum(x, 2)),
    name="StructuredMinimumTester",
)

StructuredMaximumTester = elemwise_checker(
    lambda x: structured_maximum(x, 2),
    structure_function(lambda x: np.maximum(x, 2)),
    name="StructuredMaximumTester",
)

StructuredAddTester = elemwise_checker(
    lambda x: structured_add(x, 2),
    structure_function(lambda x: np.add(x, 2)),
    name="StructuredAddTester",
)

SinTester = elemwise_checker(sparse.sin, np.sin)

TanTester = elemwise_checker(sparse.tan, np.tan, gap=(-1, 1))

ArcsinTester = elemwise_checker(
    sparse.arcsin, np.arcsin, gap=(-1, 1), gap_grad=(-0.99, 0.99)
)

ArctanTester = elemwise_checker(sparse.arctan, np.arctan)

SinhTester = elemwise_checker(sparse.sinh, np.sinh)

ArcsinhTester = elemwise_checker(sparse.arcsinh, np.arcsinh, gap=(-1, 1))

TanhTester = elemwise_checker(sparse.tanh, np.tanh, gap=(-1, 1))

ArctanhTester = elemwise_checker(
    sparse.arctanh, np.arctanh, gap=(-0.9, 1), gap_grad=(-0.9, 0.95)
)

RintTester = elemwise_checker(
    sparse.rint, np.rint, grad_test=False, test_dtypes=sparse.float_dtypes
)

SgnTester = elemwise_checker(
    sparse.sgn,
    np.sign,
    grad_test=False,
    test_dtypes=[
        m
        for m in sparse.all_dtypes
        if (m not in sparse.complex_dtypes and not m.startswith("uint"))
    ],
)

CeilTester = elemwise_checker(
    sparse.ceil,
    np.ceil,
    grad_test=False,
    test_dtypes=[m for m in sparse.all_dtypes if m not in sparse.complex_dtypes],
)

FloorTester = elemwise_checker(
    sparse.floor,
    np.floor,
    grad_test=False,
    test_dtypes=[m for m in sparse.all_dtypes if m not in sparse.complex_dtypes],
)

Log1pTester = elemwise_checker(sparse.log1p, np.log1p, gap=(0.5, 10))

Expm1Tester = elemwise_checker(sparse.expm1, np.expm1)

Deg2radTester = elemwise_checker(
    sparse.deg2rad,
    np.deg2rad,
    test_dtypes=[m for m in sparse.all_dtypes if m not in sparse.complex_dtypes],
)

Rad2degTester = elemwise_checker(
    sparse.rad2deg,
    np.rad2deg,
    test_dtypes=[m for m in sparse.all_dtypes if m not in sparse.complex_dtypes],
)


TruncTester = elemwise_checker(
    sparse.trunc,
    np.trunc,
    test_dtypes=[m for m in sparse.all_dtypes if m not in sparse.complex_dtypes],
    grad_test=False,
)


SqrTester = elemwise_checker(sparse.sqr, lambda x: x * x)

SqrtTester = elemwise_checker(sparse.sqrt, np.sqrt, gap=(0, 10))

ConjTester = elemwise_checker(sparse.conj, np.conj, grad_test=False)


class TestMulSV:
    def setup_method(self):
        utt.seed_rng()

    def test_mul_s_v_grad(self):
        sp_types = {"csc": sp.sparse.csc_matrix, "csr": sp.sparse.csr_matrix}

        for format in ["csr", "csc"]:
            for dtype in ["float32", "float64"]:
                spmat = sp_types[format](random_lil((4, 3), dtype, 3))
                mat = np.asarray(np.random.rand(3), dtype=dtype)

                verify_grad_sparse(mul_s_v, [spmat, mat], structured=True)

    def test_mul_s_v(self):
        sp_types = {"csc": sp.sparse.csc_matrix, "csr": sp.sparse.csr_matrix}

        for format in ["csr", "csc"]:
            for dtype in ["float32", "float64"]:
                x = sparse.SparseType(format, dtype=dtype)()
                y = vector(dtype=dtype)
                f = aesara.function([x, y], mul_s_v(x, y))

                spmat = sp_types[format](random_lil((4, 3), dtype, 3))
                mat = np.asarray(np.random.rand(3), dtype=dtype)

                out = f(spmat, mat)

                utt.assert_allclose(spmat.toarray() * mat, out.toarray())


class TestStructuredAddSV:
    def setup_method(self):
        utt.seed_rng()

    def test_structured_add_s_v_grad(self):
        sp_types = {"csc": sp.sparse.csc_matrix, "csr": sp.sparse.csr_matrix}

        for format in ["csr", "csc"]:
            for dtype in ["float32", "float64"]:
                spmat = sp_types[format](random_lil((4, 3), dtype, 3))
                mat = np.asarray(np.random.rand(3), dtype=dtype)

                verify_grad_sparse(structured_add_s_v, [spmat, mat], structured=True)

    def test_structured_add_s_v(self):
        sp_types = {"csc": sp.sparse.csc_matrix, "csr": sp.sparse.csr_matrix}

        for format in ["csr", "csc"]:
            for dtype in ["float32", "float64"]:
                x = sparse.SparseType(format, dtype=dtype)()
                y = vector(dtype=dtype)
                f = aesara.function([x, y], structured_add_s_v(x, y))

                spmat = sp_types[format](random_lil((4, 3), dtype, 3))
                spones = spmat.copy()
                spones.data = np.ones_like(spones.data)
                mat = np.asarray(np.random.rand(3), dtype=dtype)

                out = f(spmat, mat)

                utt.assert_allclose(
                    as_ndarray(spones.multiply(spmat + mat)), out.toarray()
                )


class TestTrueDot(utt.InferShapeTester):
    def setup_method(self):
        super().setup_method()
        self.op = true_dot
        self.op_class = TrueDot

    def test_op_ss(self):
        for format in sparse.sparse_formats:
            for dtype in sparse.all_dtypes:
                variable, data = sparse_random_inputs(
                    format, shape=(10, 10), out_dtype=dtype, n=2, p=0.1
                )

                f = aesara.function(variable, self.op(*variable))

                tested = f(*data)

                x, y = [m.toarray() for m in data]
                expected = np.dot(x, y)

                assert tested.format == format
                assert tested.dtype == expected.dtype
                tested = tested.toarray()
                utt.assert_allclose(tested, expected)

    def test_op_sd(self):
        for format in sparse.sparse_formats:
            for dtype in sparse.all_dtypes:
                variable, data = sparse_random_inputs(
                    format, shape=(10, 10), out_dtype=dtype, n=2, p=0.1
                )
                variable[1] = TensorType(dtype=dtype, broadcastable=(False, False))()
                data[1] = data[1].toarray()

                f = aesara.function(variable, self.op(*variable))

                tested = f(*data)
                expected = np.dot(data[0].toarray(), data[1])

                assert tested.format == format
                assert tested.dtype == expected.dtype
                tested = tested.toarray()
                utt.assert_allclose(tested, expected)

    def test_infer_shape(self):
        for format in sparse.sparse_formats:
            for dtype in sparse.all_dtypes:
                (x,), (x_value,) = sparse_random_inputs(
                    format, shape=(9, 10), out_dtype=dtype, p=0.1
                )
                (y,), (y_value,) = sparse_random_inputs(
                    format, shape=(10, 24), out_dtype=dtype, p=0.1
                )
                variable = [x, y]
                data = [x_value, y_value]
                self._compile_and_check(
                    variable, [self.op(*variable)], data, self.op_class
                )

    def test_grad(self):
        for format in sparse.sparse_formats:
            for dtype in sparse.float_dtypes:
                (x,), (x_value,) = sparse_random_inputs(
                    format, shape=(9, 10), out_dtype=dtype, p=0.1
                )
                (y,), (y_value,) = sparse_random_inputs(
                    format, shape=(10, 24), out_dtype=dtype, p=0.1
                )
                data = [x_value, y_value]
                verify_grad_sparse(self.op, data, structured=False)


class TestSamplingDot(utt.InferShapeTester):
    x = [matrix() for t in range(2)]
    x.append(sparse.csr_matrix())
    # unsquare shape
    a = [
        np.array(np.random.randint(1, 6, size=(4, 3)) - 1, dtype=aesara.config.floatX),
        np.array(np.random.randint(1, 6, size=(5, 3)) - 1, dtype=aesara.config.floatX),
        np.array(np.random.randint(1, 3, size=(4, 5)) - 1, dtype=aesara.config.floatX),
    ]
    a[2] = sp.sparse.csr_matrix(a[2])

    def setup_method(self):
        super().setup_method()
        self.op_class = SamplingDot

    def test_op(self):
        f = aesara.function(self.x, sampling_dot(*self.x))

        tested = f(*self.a)
        x, y, p = self.a
        expected = p.multiply(np.dot(x, y.T))

        utt.assert_allclose(as_ndarray(expected), tested.toarray())
        assert tested.format == "csr"
        assert tested.dtype == expected.dtype

    def test_negative_stride(self):
        f = aesara.function(self.x, sampling_dot(*self.x))

        a2 = [self.a[0][::-1, :], self.a[1][:, ::-1], self.a[2]]
        tested = f(*a2)
        x, y, p = a2
        expected = p.multiply(np.dot(x, y.T))

        utt.assert_allclose(as_ndarray(expected), tested.toarray())
        assert tested.format == "csr"
        assert tested.dtype == expected.dtype

    def test_infer_shape(self):
        self._compile_and_check(
            self.x,
            [sampling_dot(*self.x)],
            self.a,
            self.op_class,
            excluding=["local_sampling_dot_csr"],
        )

    def test_grad(self):
        def _helper(x, y):
            return sampling_dot(x, y, self.a[2])

        verify_grad_sparse(_helper, self.a[:2])


@makeSharedTester(
    shared_constructor_=sparse.shared,
    dtype_="float64",
    get_value_borrow_true_alias_=True,
    shared_borrow_true_alias_=True,
    set_value_borrow_true_alias_=True,
    set_value_inplace_=False,
    set_cast_value_inplace_=False,
    shared_constructor_accept_ndarray_=False,
    internal_type_=sp.sparse.csc_matrix,
    check_internal_type_=sp.sparse.issparse,
    aesara_fct_=lambda a: dense_from_sparse(a * 2.0),
    ref_fct_=lambda a: np.asarray((a * 2).todense()),
    cast_value_=sp.sparse.csr_matrix,
    expect_fail_fast_shape_inplace=False,
)
class TestSharedOptions:
    pass
