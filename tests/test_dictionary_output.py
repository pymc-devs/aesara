import pytest

from aesara import function
from aesara.tensor.type import scalar


class TestDictionaryOutput:
    def test_output_dictionary(self):
        # Tests that function works when outputs is a dictionary

        x = scalar()
        f = function([x], outputs={"a": x, "c": x * 2, "b": x * 3, "1": x * 4})

        outputs = f(10.0)

        assert outputs["a"] == 10.0
        assert outputs["b"] == 30.0
        assert outputs["1"] == 40.0
        assert outputs["c"] == 20.0

    def test_input_named_variables(self):
        # Tests that named variables work when outputs is a dictionary

        x = scalar("x")
        y = scalar("y")

        f = function([x, y], outputs={"a": x + y, "b": x * y})

        assert f(2, 4) == {"a": 6, "b": 8}
        assert f(2, y=4) == f(2, 4)
        assert f(x=2, y=4) == f(2, 4)

    def test_output_order_sorted(self):
        # Tests that the output keys are sorted correctly.

        x = scalar("x")
        y = scalar("y")
        z = scalar("z")
        e1 = scalar("1")
        e2 = scalar("2")

        f = function(
            [x, y, z, e1, e2], outputs={"x": x, "y": y, "z": z, "1": e1, "2": e2}
        )

        assert "1" in str(f.outputs[0])
        assert "2" in str(f.outputs[1])
        assert "x" in str(f.outputs[2])
        assert "y" in str(f.outputs[3])
        assert "z" in str(f.outputs[4])

    def test_composing_function(self):
        # Tests that one can compose two aesara functions when the outputs are
        # provided in a dictionary.

        x = scalar("x")
        y = scalar("y")

        a = x + y
        b = x * y

        f = function([x, y], outputs={"a": a, "b": b})

        a = scalar("a")
        b = scalar("b")

        l = a + b
        r = a * b

        g = function([a, b], outputs=[l, r])

        result = g(**f(5, 7))

        assert result[0] == 47.0
        assert result[1] == 420.0

    def test_output_list_still_works(self):
        # Test that function works if outputs is a list.

        x = scalar("x")

        f = function([x], outputs=[x * 3, x * 2, x * 4, x])

        result = f(5.0)

        assert result[0] == 15.0
        assert result[1] == 10.0
        assert result[2] == 20.0
        assert result[3] == 5.0

    def test_debug_mode_dict(self):
        # Tests that debug mode works where outputs is a dictionary.

        x = scalar("x")

        f = function([x], outputs={"1": x, "2": 2 * x, "3": 3 * x}, mode="DEBUG_MODE")

        result = f(3.0)

        assert result["1"] == 3.0
        assert result["2"] == 6.0
        assert result["3"] == 9.0

    def test_debug_mode_list(self):
        # Tests that debug mode works where the outputs argument is a list.

        x = scalar("x")

        f = function([x], outputs=[x, 2 * x, 3 * x], mode="DEBUG_MODE")

        result = f(5.0)

        assert result[0] == 5.0
        assert result[1] == 10.0
        assert result[2] == 15.0

    def test_key_string_requirement(self):
        # Tests that an exception is thrown if a non-string key is used in
        # the outputs dictionary.
        x = scalar("x")

        with pytest.raises(AssertionError):
            function([x], outputs={1.0: x})

        with pytest.raises(AssertionError):
            function([x], outputs={1.0: x, "a": x ** 2})

        with pytest.raises(AssertionError):
            function([x], outputs={(1, "b"): x, 1.0: x ** 2})
