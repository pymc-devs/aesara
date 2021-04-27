"""
Test compilation modes
"""

import copy

from aesara.compile.function import function
from aesara.compile.mode import Mode, get_default_mode
from aesara.configdefaults import config
from aesara.tensor.type import matrix, vector


class TestBunchOfModes:
    def test_modes(self):
        # this is a quick test after the LazyLinker branch merge
        # to check that all the current modes can still be used.
        linker_classes_involved = []

        predef_modes = ["FAST_COMPILE", "FAST_RUN", "DEBUG_MODE"]

        # Linkers to use with regular Mode
        if config.cxx:
            linkers = ["py", "c|py", "c|py_nogc", "vm", "vm_nogc", "cvm", "cvm_nogc"]
        else:
            linkers = ["py", "c|py", "c|py_nogc", "vm", "vm_nogc"]
        modes = predef_modes + [Mode(linker, "fast_run") for linker in linkers]

        for mode in modes:
            x = matrix()
            y = vector()
            f = function([x, y], x + y, mode=mode)
            # test that it runs something
            f([[1, 2], [3, 4]], [5, 6])
            linker_classes_involved.append(f.maker.mode.linker.__class__)
            # print 'MODE:', mode, f.maker.mode.linker, 'stop'

        # regression check:
        # there should be
        # - `VMLinker`
        # - OpWiseCLinker (FAST_RUN)
        # - PerformLinker (FAST_COMPILE)
        # - DebugMode's Linker  (DEBUG_MODE)
        assert 4 == len(set(linker_classes_involved))


class TestOldModesProblem:
    def test_modes(self):
        # Then, build a mode with the same linker, and a modified optimizer
        default_mode = get_default_mode()
        modified_mode = default_mode.including("specialize")

        # The following line used to fail, with Python 2.4, in July 2012,
        # because an fgraph was associated to the default linker
        copy.deepcopy(modified_mode)

        # More straightforward test
        linker = get_default_mode().linker
        assert not hasattr(linker, "fgraph") or linker.fgraph is None
