import os
import pickle
import re
import shutil
import tempfile
from collections import OrderedDict

import numpy as np
import pytest

from aesara.compile import shared
from aesara.compile.function import function, function_dump
from aesara.compile.io import In
from aesara.configdefaults import config
from aesara.tensor.math import sum as aet_sum
from aesara.tensor.type import (
    bscalar,
    bvector,
    dscalar,
    dvector,
    fmatrix,
    fscalar,
    fvector,
    vector,
    wvector,
)


floatX = "float32"


def test_function_dump():
    v = vector()
    fct1 = function([v], v + 1)

    try:
        tmpdir = tempfile.mkdtemp()
        fname = os.path.join(tmpdir, "test_function_dump.pkl")
        function_dump(fname, [v], v + 1)
        with open(fname, "rb") as f:
            l = pickle.load(f)
    finally:
        if tmpdir is not None:
            shutil.rmtree(tmpdir)

    fct2 = function(**l)
    x = [1, 2, 3]
    assert np.allclose(fct1(x), fct2(x))


def test_function_name():
    x = vector("x")
    func = function([x], x + 1.0)

    regex = re.compile(os.path.basename(".*test_function.pyc?"))
    assert regex.match(func.name) is not None


class TestFunctionIn:
    def test_in_strict(self):

        a = dvector()
        b = shared(7)
        out = a + b

        f = function([In(a, strict=False)], out)
        # works, rand generates float64 by default
        f(np.random.rand(8))
        # works, casting is allowed
        f(np.array([1, 2, 3, 4], dtype="int32"))

        f = function([In(a, strict=True)], out)
        try:
            # fails, f expects float64
            f(np.array([1, 2, 3, 4], dtype="int32"))
        except TypeError:
            pass

    def test_explicit_shared_input(self):
        # This is not a test of the In class per se, but the In class relies
        # on the fact that shared variables cannot be explicit inputs
        a = shared(1.0)
        with pytest.raises(TypeError):
            function([a], a + 1)

    def test_in_shared_variable(self):
        # Ensure that an error is raised if the In wrapped is used to wrap
        # a shared variable
        a = shared(1.0)
        a_wrapped = In(a, update=a + 1)
        with pytest.raises(TypeError):
            function([a_wrapped])

    def test_in_mutable(self):
        a = dvector()
        a_out = a * 2  # assuming the op which makes this "in place" triggers

        # using mutable=True will let f change the value in aval
        f = function([In(a, mutable=True)], a_out, mode="FAST_RUN")
        aval = np.random.rand(10)
        aval2 = aval.copy()
        assert np.all(f(aval) == (aval2 * 2))
        assert not np.all(aval == aval2)

        # using mutable=False should leave the input untouched
        f = function([In(a, mutable=False)], a_out, mode="FAST_RUN")
        aval = np.random.rand(10)
        aval2 = aval.copy()
        assert np.all(f(aval) == (aval2 * 2))
        assert np.all(aval == aval2)

    def test_in_update(self):
        a = dscalar("a")
        f = function([In(a, value=0.0, update=a + 1)], a, mode="FAST_RUN")

        # Ensure that, through the executions of the function, the state of the
        # input is persistent and is updated as it should
        assert f() == 0.0
        assert f() == 1.0
        assert f() == 2.0

    def test_in_update_wrong_dtype(self):
        # Ensure that an error is raised if an In-wrapped variables has
        # an update of a different type
        a = dscalar("a")
        b = dvector("b")
        with pytest.raises(TypeError):
            In(a, update=b)

    def test_in_update_shared(self):
        # Test that using both In() with updates and shared variables with
        # updates in the same function behaves as expected
        shared_var = shared(1.0)
        a = dscalar("a")
        a_wrapped = In(a, value=0.0, update=shared_var)
        f = function([a_wrapped], [], updates={shared_var: a}, mode="FAST_RUN")

        # Ensure that, through the executions of the function, the state of
        # the input and the shared variable are appropriate (after N execution,
        # the values have swapped N times). This allows testing that the
        # changes occur at the same time and one doesn't overwrite the other.
        for i in range(5):
            f()
            assert np.allclose(shared_var.get_value(), i % 2)

    def test_in_allow_downcast_int(self):
        a = wvector("a")  # int16
        b = bvector("b")  # int8
        c = bscalar("c")  # int8
        f = function(
            [
                In(a, allow_downcast=True),
                In(b, allow_downcast=False),
                In(c, allow_downcast=None),
            ],
            (a + b + c),
        )

        # Both values are in range. Since they're not ndarrays (but lists),
        # they will be converted, and their value checked.
        assert np.all(f([3], [6], 1) == 10)

        # Values are in range, but a dtype too large has explicitly been given
        # For performance reasons, no check of the data is explicitly performed
        # (It might be OK to change this in the future.)
        with pytest.raises(TypeError):
            f([3], np.array([6], dtype="int16"), 1)

        # Value too big for a, silently ignored
        assert np.all(f([2 ** 20], np.ones(1, dtype="int8"), 1) == 2)

        # Value too big for b, raises TypeError
        with pytest.raises(TypeError):
            f([3], [312], 1)

        # Value too big for c, raises TypeError
        with pytest.raises(TypeError):
            f([3], [6], 806)

    def test_in_allow_downcast_floatX(self):
        a = fscalar("a")
        b = fscalar("b")
        c = fscalar("c")

        f = function(
            [
                In(a, allow_downcast=True),
                In(b, allow_downcast=False),
                In(c, allow_downcast=None),
            ],
            (a + b + c),
        )

        # If the values can be accurately represented, everything is OK
        assert np.all(f(0, 0, 0) == 0)

        # If allow_downcast is True, idem
        assert np.allclose(f(0.1, 0, 0), 0.1)

        # If allow_downcast is False, nope
        with pytest.raises(TypeError):
            f(0, 0.1, 0)

        # If allow_downcast is None, it should work iff floatX=float32
        if config.floatX == "float32":
            assert np.allclose(f(0, 0, 0.1), 0.1)
        else:
            with pytest.raises(TypeError):
                f(0, 0, 0.1)

    def test_in_allow_downcast_vector_floatX(self):
        a = fvector("a")
        b = fvector("b")
        c = fvector("c")

        f = function(
            [
                In(a, allow_downcast=True),
                In(b, allow_downcast=False),
                In(c, allow_downcast=None),
            ],
            (a + b + c),
        )

        # If the values can be accurately represented, everything is OK
        z = [0]
        assert np.all(f(z, z, z) == 0)

        # If allow_downcast is True, idem
        assert np.allclose(f([0.1], z, z), 0.1)

        # If allow_downcast is False, nope
        with pytest.raises(TypeError):
            f(z, [0.1], z)

        # If allow_downcast is None, like False
        with pytest.raises(TypeError):
            f(z, z, [0.1])


def test_pickle_unpickle_with_reoptimization():
    mode = config.mode
    if mode in ["DEBUG_MODE", "DebugMode"]:
        mode = "FAST_RUN"
    x1 = fmatrix("x1")
    x2 = fmatrix("x2")
    x3 = shared(np.ones((10, 10), dtype=floatX))
    x4 = shared(np.ones((10, 10), dtype=floatX))
    y = aet_sum(aet_sum(aet_sum(x1 ** 2 + x2) + x3) + x4)

    updates = OrderedDict()
    updates[x3] = x3 + 1
    updates[x4] = x4 + 1
    f = function([x1, x2], y, updates=updates, mode=mode)

    # now pickle the compiled aesara fn
    string_pkl = pickle.dumps(f, -1)

    in1 = np.ones((10, 10), dtype=floatX)
    in2 = np.ones((10, 10), dtype=floatX)

    # test unpickle with optimization
    default = config.reoptimize_unpickled_function
    try:
        # the default is True
        config.reoptimize_unpickled_function = True
        f_ = pickle.loads(string_pkl)
        assert f(in1, in2) == f_(in1, in2)
    finally:
        config.reoptimize_unpickled_function = default


def test_pickle_unpickle_without_reoptimization():
    mode = config.mode
    if mode in ["DEBUG_MODE", "DebugMode"]:
        mode = "FAST_RUN"
    x1 = fmatrix("x1")
    x2 = fmatrix("x2")
    x3 = shared(np.ones((10, 10), dtype=floatX))
    x4 = shared(np.ones((10, 10), dtype=floatX))
    y = aet_sum(aet_sum(aet_sum(x1 ** 2 + x2) + x3) + x4)

    updates = OrderedDict()
    updates[x3] = x3 + 1
    updates[x4] = x4 + 1
    f = function([x1, x2], y, updates=updates, mode=mode)

    # now pickle the compiled aesara fn
    string_pkl = pickle.dumps(f, -1)

    # compute f value
    in1 = np.ones((10, 10), dtype=floatX)
    in2 = np.ones((10, 10), dtype=floatX)

    # test unpickle without optimization
    default = config.reoptimize_unpickled_function
    try:
        # the default is True
        config.reoptimize_unpickled_function = False
        f_ = pickle.loads(string_pkl)
        assert f(in1, in2) == f_(in1, in2)
    finally:
        config.reoptimize_unpickled_function = default
