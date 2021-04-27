# Test of memory profiling


from io import StringIO

import numpy as np

import aesara.tensor as aet
from aesara.compile import ProfileStats
from aesara.compile.function import function
from aesara.configdefaults import config
from aesara.ifelse import ifelse
from aesara.tensor.type import fvector, scalars


class TestProfiling:
    # Test of Aesara profiling with min_peak_memory=True

    def test_profiling(self):

        config1 = config.profile
        config2 = config.profile_memory
        config3 = config.profiling__min_peak_memory
        try:
            config.profile = True
            config.profile_memory = True
            config.profiling__min_peak_memory = True

            x = [fvector("val%i" % i) for i in range(3)]

            z = []
            z += [aet.outer(x[i], x[i + 1]).sum(axis=1) for i in range(len(x) - 1)]
            z += [x[i] + x[i + 1] for i in range(len(x) - 1)]

            p = ProfileStats(False, gpu_checks=False)

            if config.mode in ["DebugMode", "DEBUG_MODE", "FAST_COMPILE"]:
                m = "FAST_RUN"
            else:
                m = None

            f = function(x, z, profile=p, name="test_profiling", mode=m)

            inp = [np.arange(1024, dtype="float32") + 1 for i in range(len(x))]
            f(*inp)

            buf = StringIO()
            f.profile.summary(buf)

            # regression testing for future algo speed up
            the_string = buf.getvalue()
            lines1 = [l for l in the_string.split("\n") if "Max if linker" in l]
            lines2 = [l for l in the_string.split("\n") if "Minimum peak" in l]
            if config.device == "cpu":
                assert "CPU: 4112KB (4104KB)" in the_string, (lines1, lines2)
                assert "CPU: 8204KB (8196KB)" in the_string, (lines1, lines2)
                assert "CPU: 8208KB" in the_string, (lines1, lines2)
                assert (
                    "Minimum peak from all valid apply node order is 4104KB"
                    in the_string
                ), (lines1, lines2)
            else:
                assert "CPU: 16KB (16KB)" in the_string, (lines1, lines2)
                assert "GPU: 8204KB (8204KB)" in the_string, (lines1, lines2)
                assert "GPU: 12300KB (12300KB)" in the_string, (lines1, lines2)
                assert "GPU: 8212KB" in the_string, (lines1, lines2)
                assert (
                    "Minimum peak from all valid apply node order is 4116KB"
                    in the_string
                ), (lines1, lines2)

        finally:
            config.profile = config1
            config.profile_memory = config2
            config.profiling__min_peak_memory = config3

    def test_ifelse(self):
        config1 = config.profile
        config2 = config.profile_memory

        try:
            config.profile = True
            config.profile_memory = True

            a, b = scalars("a", "b")
            x, y = scalars("x", "y")

            z = ifelse(aet.lt(a, b), x * 2, y * 2)

            p = ProfileStats(False, gpu_checks=False)

            if config.mode in ["DebugMode", "DEBUG_MODE", "FAST_COMPILE"]:
                m = "FAST_RUN"
            else:
                m = None

            f_ifelse = function([a, b, x, y], z, profile=p, name="test_ifelse", mode=m)

            val1 = 0.0
            val2 = 1.0
            big_mat1 = 10
            big_mat2 = 11

            f_ifelse(val1, val2, big_mat1, big_mat2)

        finally:
            config.profile = config1
            config.profile_memory = config2
