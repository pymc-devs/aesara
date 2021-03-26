"""

To update the `Scan` Cython code you must
- update the version value in this file and `scan_perform.py`, and
- run `cython scan_perform.pyx; mv scan_perform.c c_code`

"""
import logging
import os
import sys
from importlib import reload

import aesara
from aesara.compile.compilelock import lock_ctx
from aesara.configdefaults import config
from aesara.link.c import cmodule


if not config.cxx:
    raise ImportError("No C compiler; cannot compile Cython-generated code")

_logger = logging.getLogger("aesara.scan.scan_perform")

version = 0.298  # must match constant returned in function get_version()

need_reload = False


def try_import():
    global scan_perform
    sys.path[0:0] = [config.compiledir]
    import scan_perform

    del sys.path[0]


def try_reload():
    sys.path[0:0] = [config.compiledir]
    reload(scan_perform)
    del sys.path[0]


try:
    try_import()
    need_reload = True
    if version != getattr(scan_perform, "_version", None):
        raise ImportError("Scan code version mismatch")
except ImportError:

    dirname = "scan_perform"
    loc = os.path.join(config.compiledir, dirname)

    os.makedirs(loc, exist_ok=True)

    with lock_ctx(loc):
        # Maybe someone else already finished compiling it while we were
        # waiting for the lock?
        try:
            if need_reload:
                # The module was successfully imported earlier: we need to
                # reload it to check if the version was updated.
                try_reload()
            else:
                try_import()
                need_reload = True

            if version != getattr(scan_perform, "_version", None):
                raise ImportError()

        except ImportError:
            _logger.info("Compiling C code for scan")

            cfile = os.path.join(aesara.__path__[0], "scan", "c_code", "scan_perform.c")  # type: ignore

            if not os.path.exists(cfile):
                raise ImportError(
                    "The file scan_perform.c is not available, so scan "
                    "will not use its Cython implementation."
                )

            preargs = ["-fwrapv", "-O2", "-fno-strict-aliasing"]
            preargs += cmodule.GCC_compiler.compile_args()

            with open(cfile) as f:
                code = f.read()

            cmodule.GCC_compiler.compile_str(
                dirname, code, location=loc, preargs=preargs, hide_symbols=False
            )
            # Save version into the __init__.py file.
            init_py = os.path.join(loc, "__init__.py")

            with open(init_py, "w") as f:
                f.write(f"_version = {version}\n")

            # If we just compiled the module for the first time, then it was
            # imported at the same time.  We need to make sure we do not reload
            # the now outdated __init__.pyc below.
            init_pyc = os.path.join(loc, "__init__.pyc")

            if os.path.isfile(init_pyc):
                os.remove(init_pyc)

            try_import()

            try_reload()

            from scan_perform import scan_perform as scan_c

            assert scan_perform._version == scan_c.get_version()

            _logger.info(f"New version {scan_perform._version}")

from scan_perform.scan_perform import get_version, perform  # noqa: F401, E402


assert version == get_version()
