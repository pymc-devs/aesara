"""
ProfileStats object for runtime and memory profiling.

"""
#
# TODO: add tip to use specify_shape (is specify_shape even in library doc?)
# TODO: ensure field width for string fields makes columns line up
# TODO: what to do about 'diff summary'? (ask Fred?)
#

import atexit
import copy
import logging
import operator
import os
import sys
import time
import warnings
from collections import defaultdict
from typing import Dict, List

import numpy as np

import aesara
from aesara.configdefaults import config
from aesara.graph.basic import Constant, Variable


__authors__ = "James Bergstra" "PyMC Developers"
__copyright__ = "(c) 2011, Universite de Montreal"

__docformat__ = "restructuredtext en"

logger = logging.getLogger("aesara.compile.profiling")

aesara_imported_time = time.time()
total_fct_exec_time = 0.0
total_graph_opt_time = 0.0
total_time_linker = 0.0

_atexit_print_list: List = []
_atexit_registered = False


def _atexit_print_fn():
    """
    Print ProfileStat objects in _atexit_print_list to _atexit_print_file.

    """
    if config.profile:
        to_sum = []

        if config.profiling__destination == "stderr":
            destination_file = sys.stderr
        elif config.profiling__destination == "stdout":
            destination_file = sys.stdout
        else:
            destination_file = open(config.profiling__destination, "w")

        # Reverse sort in the order of compile+exec time
        for ps in sorted(
            _atexit_print_list, key=lambda a: a.compile_time + a.fct_call_time
        )[::-1]:
            if (
                ps.fct_callcount >= 1
                or ps.compile_time > 1
                or getattr(ps, "callcount", 0) > 1
            ):
                ps.summary(
                    file=destination_file,
                    n_ops_to_print=config.profiling__n_ops,
                    n_apply_to_print=config.profiling__n_apply,
                )
                if not isinstance(ps, ScanProfileStats):
                    to_sum.append(ps)
            else:
                # TODO print the name if there is one!
                print("Skipping empty Profile")
        if len(to_sum) > 1:
            # Make a global profile
            cum = copy.copy(to_sum[0])
            msg = (
                f"Sum of all({len(to_sum)}) printed profiles at exit excluding Scan op"
                " profile."
            )
            cum.message = msg
            for ps in to_sum[1:]:
                for attr in [
                    "compile_time",
                    "fct_call_time",
                    "fct_callcount",
                    "vm_call_time",
                    "optimizer_time",
                    "linker_time",
                    "validate_time",
                    "import_time",
                    "linker_node_make_thunks",
                ]:
                    setattr(cum, attr, getattr(cum, attr) + getattr(ps, attr))

                # merge dictonary
                for attr in [
                    "apply_time",
                    "apply_callcount",
                    "apply_cimpl",
                    "variable_shape",
                    "variable_strides",
                    "variable_offset",
                    "linker_make_thunk_time",
                ]:
                    cum_attr = getattr(cum, attr)
                    for key, val in getattr(ps, attr.items()):
                        assert key not in cum_attr, (key, cum_attr)
                        cum_attr[key] = val

                if cum.optimizer_profile and ps.optimizer_profile:
                    try:
                        merge = cum.optimizer_profile[0].merge_profile(
                            cum.optimizer_profile[1], ps.optimizer_profile[1]
                        )
                        assert len(merge) == len(cum.optimizer_profile[1])
                        cum.optimizer_profile = (cum.optimizer_profile[0], merge)
                    except Exception as e:
                        print(e)
                        cum.optimizer_profile = None
                else:
                    cum.optimizer_profile = None

            cum.summary(
                file=destination_file,
                n_ops_to_print=config.profiling__n_ops,
                n_apply_to_print=config.profiling__n_apply,
            )

    if config.print_global_stats:
        print_global_stats()


def print_global_stats():
    """
    Print the following stats:
      -- Time elapsed since Aesara was imported
      -- Time spent inside Aesara functions
      -- Time spent in compiling Aesara functions
           -- on graph optimization
           -- on linker
    """

    if config.profiling__destination == "stderr":
        destination_file = sys.stderr
    elif config.profiling__destination == "stdout":
        destination_file = sys.stdout
    else:
        destination_file = open(config.profiling__destination, "w")

    print("=" * 50, file=destination_file)
    print(
        (
            "Global stats: ",
            f"Time elasped since Aesara import = {time.time() - aesara_imported_time:6.3f}s, "
            f"Time spent in Aesara functions = {total_fct_exec_time:6.3f}s, "
            "Time spent compiling Aesara functions: "
            f" optimization = {total_graph_opt_time:6.3f}s, linker = {total_time_linker:6.3f}s ",
        ),
        file=destination_file,
    )
    print("=" * 50, file=destination_file)


_profiler_printers = []


def register_profiler_printer(fct):
    _profiler_printers.append(fct)
    return fct


class ProfileStats:

    """
    Object to store runtime and memory profiling information for all of
    Aesara's operations: compilation, optimization, execution.

    Parameters
    ----------
    atexit_print : bool
        True means that this object will be printed to stderr (using .summary())
        at the end of the program.
    **kwargs : misc initializers
        These should (but need not) match the names of the class vars declared
        in this class.

    """

    def reset(self):
        """ Ignore previous function call"""
        # self.compile_time = 0.
        self.fct_call_time = 0.0
        self.fct_callcount = 0
        self.vm_call_time = 0.0
        self.apply_time = {}
        self.apply_callcount = {}
        # self.apply_cimpl = None
        # self.message = None

    #
    # Note on implementation:
    # Class variables are used here so that each one can be
    # documented and initialized together.
    # dictionary variables are initialized with None.
    #

    compile_time = 0.0
    # Total time spent in body of orig_function,
    # dominated by graph optimization and compilation of C
    #

    fct_call_time = 0.0
    # The total time spent in Function.__call__
    #

    fct_callcount = 0
    # Number of calls to Function.__call__
    #

    vm_call_time = 0.0
    # Total time spent in Function.fn.__call__
    #

    apply_time = None
    # dict from `(FunctionGraph, Variable)` to float runtime
    #

    apply_callcount = None
    # dict from `(FunctionGraph, Variable)` to number of executions
    #

    apply_cimpl = None
    # dict from node -> bool (1 if c, 0 if py)
    #

    message = None
    # pretty string to print in summary, to identify this output
    #

    variable_shape: Dict = {}
    # Variable -> shapes
    #

    variable_strides: Dict = {}
    # Variable -> strides
    #

    variable_offset: Dict = {}
    # Variable -> offset
    #

    optimizer_time = 0.0
    # time spent optimizing graph (FunctionMaker.__init__)

    validate_time = 0.0
    # time spent in fgraph.validate
    # This is a subset of optimizer_time that is dominated by toposort()
    # when the destorymap feature is included.

    linker_time = 0.0
    # time spent linking graph (FunctionMaker.create)

    import_time = 0.0
    # time spent in importing compiled python module.

    linker_node_make_thunks = 0.0

    linker_make_thunk_time: Dict = {}

    line_width = config.profiling__output_line_width

    nb_nodes = -1
    # The number of nodes in the graph. We need the information separately in
    # case we print the profile when the function wasn't executed, or if there
    # is a lazy operation in the graph.

    optimizer_profile = None
    # None or tuple (the optimizer, the profile it returned)

    # param is called flag_time_thunks because most other attributes with time
    # in the name are times *of* something, rather than configuration flags.
    def __init__(
        self, atexit_print=True, flag_time_thunks=None, gpu_checks=True, **kwargs
    ):
        if (
            gpu_checks
            and (hasattr(aesara, "gpuarray") and aesara.gpuarray.pygpu_activated)
            and os.environ.get("CUDA_LAUNCH_BLOCKING", "0") != "1"
        ):
            msg = (
                "You are running the Aesara profiler with CUDA enabled."
                " Aesara GPU ops execution is asynchronous by default."
                " So by default, the profile is useless."
                " You must set the environment variable"
                " CUDA_LAUNCH_BLOCKING to 1 to tell the CUDA driver to"
                " synchronize the execution to get a meaningful profile."
            )
            if config.profile:
                raise Exception(msg)
            else:
                warnings.warn(msg)

        if (
            config.profile
            and gpu_checks
            and hasattr(aesara, "gpuarray")
            and aesara.gpuarray.pygpu_activated
            and not config.profiling__ignore_first_call
        ):
            warnings.warn(
                "Aesara flag profiling__ignore_first_call is False. "
                "This cause bad profiling result in the gpu "
                "back-end, as sometimes we compile at the first call."
            )

        self.apply_callcount = {}
        self.output_size = {}
        # Keys are `(FunctionGraph, Variable)`
        self.apply_time = {}
        self.apply_cimpl = {}
        self.variable_shape = {}
        self.variable_strides = {}
        self.variable_offset = {}
        if flag_time_thunks is None:
            self.flag_time_thunks = config.profiling__time_thunks
        else:
            self.flag_time_thunks = flag_time_thunks
        self.__dict__.update(kwargs)
        if atexit_print:
            global _atexit_print_list
            _atexit_print_list.append(self)
            global _atexit_registered
            if not _atexit_registered:
                atexit.register(_atexit_print_fn)
                _atexit_registered = True
        self.ignore_first_call = config.profiling__ignore_first_call

    def class_time(self):
        """
        dict op -> total time on thunks

        """
        # timing is stored by node, we compute timing by class on demand
        rval = {}
        for (fgraph, node), t in self.apply_time.items():
            typ = type(node.op)
            rval.setdefault(typ, 0)
            rval[typ] += t
        return rval

    def class_callcount(self):
        """
        dict op -> total number of thunk calls

        """
        # timing is stored by node, we compute timing by class on demand
        rval = {}
        for (fgraph, node), count in self.apply_callcount.items():
            typ = type(node.op)
            rval.setdefault(typ, 0)
            rval[typ] += count
        return rval

    def class_nodes(self):
        """
        dict op -> total number of nodes

        """
        # timing is stored by node, we compute timing by class on demand
        rval = {}
        for (fgraph, node), count in self.apply_callcount.items():
            typ = type(node.op)
            rval.setdefault(typ, 0)
            rval[typ] += 1
        return rval

    def class_impl(self):
        """
        dict op -> total number of nodes

        """
        # timing is stored by node, we compute timing by class on demand
        rval = {}
        for (fgraph, node) in self.apply_callcount:
            typ = type(node.op)
            if self.apply_cimpl[node]:
                impl = "C "
            else:
                impl = "Py"
            rval.setdefault(typ, impl)
            if rval[typ] != impl and len(rval[typ]) == 2:
                rval[typ] += impl
        return rval

    def op_time(self):
        """
        dict op -> total time on thunks

        """
        # timing is stored by node, we compute timing by Op on demand
        rval = {}
        for (fgraph, node), t in self.apply_time.items():
            rval.setdefault(node.op, 0)
            rval[node.op] += t
        return rval

    def fill_node_total_time(self, fgraph, node, total_times):
        """
        node -> fill total time including its parents (returns nothing)

        """
        # timing is stored by node, we compute total time on demand
        total = self.apply_time.get((fgraph, node), 0.0)
        for parent in node.get_parents():
            if (fgraph, parent.owner) in self.apply_time:
                if parent.owner not in total_times:
                    self.fill_node_total_time(fgraph, parent.owner, total_times)
                total += total_times[parent.owner]
        total_times[node] = total

    def compute_total_times(self):
        """
        dict op -> total time icluding the time for parents

        """
        rval = {}
        for (fgraph, node) in self.apply_time:
            if node not in rval:
                self.fill_node_total_time(fgraph, node, rval)
        return rval

    def op_callcount(self):
        """
        dict op -> total number of thunk calls

        """
        # timing is stored by node, we compute timing by Op on demand
        rval = {}
        for (fgraph, node), count in self.apply_callcount.items():
            rval.setdefault(node.op, 0)
            rval[node.op] += count
        return rval

    def op_nodes(self):
        """
        dict op -> total number of nodes

        """
        # timing is stored by node, we compute timing by Op on demand
        rval = {}
        for (fgraph, node), count in self.apply_callcount.items():
            rval.setdefault(node.op, 0)
            rval[node.op] += 1
        return rval

    def op_impl(self):
        """
        dict op -> 'C' or 'Py' depending how the op is implemented

        """
        # timing is stored by node, we compute timing by Op on demand
        rval = {}
        for (fgraph, node) in self.apply_callcount:
            if self.apply_cimpl[node]:
                rval[node.op] = "C "
            else:
                rval[node.op] = "Py"
        return rval

    def summary_class(self, file=sys.stderr, N=None):
        if self.apply_time:
            local_time = sum(self.apply_time.values())
        else:
            local_time = 0
        if local_time == 0:
            print(
                (
                    "ProfileStats.summary_class: total time 0"
                    " (did you forget to enable counters?)"
                ),
                file=file,
            )
            return
        class_time = self.class_time()
        class_call = self.class_callcount()
        class_apply = self.class_nodes()
        class_impl = self.class_impl()
        if N is None:
            N = len(self.class_time)
        otimes = [
            (
                t * 100 / local_time,
                t,
                clas,
                class_impl.get(clas, "  "),
                class_call.get(clas, 0),
                class_apply.get(clas, 0),
            )
            for clas, t in class_time.items()
        ]
        otimes.sort(key=lambda t: (t[1], t[4], t[5]), reverse=True)
        tot = 0
        print("Class", file=file)
        print("---", file=file)
        hs = []
        # formatting string
        es = []

        hs += ["<% time>"]
        es += ["  %4.1f%% "]

        hs += ["<sum %>"]
        es += [" %5.1f%% "]

        hs += ["<apply time>"]
        es += ["   %7.3fs "]

        hs += ["<time per call>"]
        es += ["     %8.2es "]

        hs += ["<type>"]
        es += ["   %2s "]

        hs += ["<#call>"]
        es += ["%6d  "]

        hs += ["<#apply>"]
        es += [" %4d  "]

        upto_length = np.sum([len(x) for x in hs]) + len(hs)
        maxlen = max(self.line_width - upto_length, 0)
        hs += ["<Class name>"]
        es += ["%s"]
        header_str = " ".join(hs)
        format_str = " ".join(es)

        print(header_str, file=file)

        for f, t, a, impl, nb_call, nb_apply in otimes[:N]:
            if nb_call == 0:
                assert t == 0
                continue
            tot += t
            ftot = tot * 100 / local_time
            # Remove the useless start and end of the class name:
            # "<class 'aesara.gpuarray.blas.GpuDot22'>" ->
            #  "aesara.gpuarray.blas.GpuDot22"
            class_name = str(a)[8:-2][:maxlen]
            print(
                format_str
                % (f, ftot, t, t / nb_call, impl, nb_call, nb_apply, class_name),
                file=file,
            )
            # While this carries over less information, it is arranged such
            # that it way more readeable that the previous output of the
            # profiler
        print(
            "   ... (remaining %i Classes account for %6.2f%%(%.2fs) of "
            "the runtime)"
            % (
                max(0, len(otimes) - N),
                sum(f for f, t, a, ci, nb_call, nb_op in otimes[N:]),
                sum(t for f, t, a, ci, nb_call, nb_op in otimes[N:]),
            ),
            file=file,
        )
        print("", file=file)

    def summary_ops(self, file=sys.stderr, N=None):
        if self.apply_time:
            local_time = sum(self.apply_time.values())
        else:
            local_time = 0
        if local_time == 0:
            print(
                (
                    "ProfileStats.summary_ops: total time 0"
                    " (did you forget to enable counters?)"
                ),
                file=file,
            )
            return
        op_time = self.op_time()
        op_call = self.op_callcount()
        op_apply = self.op_nodes()
        op_impl = self.op_impl()
        otimes = [
            (
                t * 100 / local_time,
                t,
                op,
                op_impl.get(op, "  "),
                op_call.get(op, 0),
                op_apply.get(op, 0),
            )
            for op, t in op_time.items()
        ]
        otimes.sort(key=lambda t: (t[1], t[4], t[5]), reverse=True)
        tot = 0
        print("Ops", file=file)
        print("---", file=file)
        hs = []
        # formatting string
        es = []

        hs += ["<% time>"]
        es += ["  %4.1f%% "]

        hs += ["<sum %>"]
        es += [" %5.1f%% "]

        hs += ["<apply time>"]
        es += ["   %7.3fs "]

        hs += ["<time per call>"]
        es += ["     %8.2es "]

        hs += ["<type>"]
        es += ["   %2s "]

        hs += ["<#call>"]
        es += ["  %4d  "]

        hs += ["<#apply>"]
        es += ["  %4d  "]

        upto_length = np.sum([len(x) for x in hs]) + len(hs)
        maxlen = max(self.line_width - upto_length, 0)
        hs += ["<Op name>"]
        es += ["%s"]
        header_str = " ".join(hs)
        format_str = " ".join(es)

        print(header_str, file=file)

        for f, t, a, impl, nb_call, nb_apply in otimes[:N]:
            if nb_call == 0:
                assert t == 0
                continue
            tot += t
            ftot = tot * 100 / local_time
            print(
                format_str
                % (f, ftot, t, t / nb_call, impl, nb_call, nb_apply, str(a)[:maxlen]),
                file=file,
            )
            # While this carries over less information, it is arranged such
            # that it way more readeable that the previous output of the
            # profiler
        print(
            "   ... (remaining %i Ops account for %6.2f%%(%.2fs) of "
            "the runtime)"
            % (
                max(0, len(otimes) - N),
                sum(f for f, t, a, ci, nb_call, nb_op in otimes[N:]),
                sum(t for f, t, a, ci, nb_call, nb_op in otimes[N:]),
            ),
            file=file,
        )
        print("", file=file)

    def summary_nodes(self, file=sys.stderr, N=None):
        if self.apply_time:
            local_time = sum(self.apply_time.values())
        else:
            local_time = 0
        if local_time == 0:
            print(
                (
                    "ProfileStats.summary_nodes: total time 0"
                    " (did you forget to enable counters?)"
                ),
                file=file,
            )
            return

        print("Apply", file=file)
        print("------", file=file)
        # headers
        hs = []
        # formatting string
        es = []

        hs += ["<% time>"]
        es += ["  %4.1f%% "]

        hs += ["<sum %>"]
        es += [" %5.1f%% "]

        hs += ["<apply time>"]
        es += ["   %7.3fs "]

        hs += ["<time per call>"]
        es += ["     %8.2es "]

        hs += ["<#call>"]
        es += [" %4d  "]

        hs += ["<id>"]
        es += ["%3d"]

        es += ["%s", "%s"]
        if self.variable_shape:
            hs += ["<Mflops>", "<Gflops/s>"]

        upto_length = np.sum([len(x) for x in hs]) + len(hs)
        maxlen = max(self.line_width - upto_length, 0)
        hs += ["<Apply name>"]
        es += ["%s"]

        header_str = " ".join(hs)
        format_str = " ".join(es)

        print(header_str, file=file)

        topos = {}  # Only do the topo once per fct.
        atimes = []
        for (fgraph, a), t in self.apply_time.items():
            if fgraph not in topos:
                topo = fgraph.toposort()
                topos[fgraph] = topo
            else:
                topo = topos[fgraph]
            atimes.append(
                (
                    t * 100 / local_time,
                    t,
                    a,
                    topo.index(a),
                    self.apply_callcount[(fgraph, a)],
                )
            )
        del topos

        atimes.sort(reverse=True, key=lambda t: (t[1], t[3]))
        tot = 0
        for (f, t, a, nd_id, nb_call) in atimes[:N]:
            tot += t
            ftot = tot * 100 / local_time
            if nb_call == 0:
                continue
            if not self.variable_shape:
                flops = ""
                flops_s = ""
            elif hasattr(a.op, "flops"):
                fl = a.op.flops(
                    [self.variable_shape[var] for var in a.inputs],
                    [self.variable_shape[var] for var in a.outputs],
                )
                flops = f"{fl / 1024.0 / 1024:8.1f}"
                flops_s = f"{fl / 1024.0 / 1024 / 1024 / t:10.1f}"
            else:
                flops = "        "
                flops_s = "          "
            print(
                format_str
                % (
                    f,
                    ftot,
                    t,
                    t / nb_call,
                    nb_call,
                    nd_id,
                    flops,
                    flops_s,
                    str(a)[:maxlen],
                ),
                file=file,
            )
            if not config.profile_memory:
                continue
            for idx, var in enumerate(a.inputs):
                sh = self.variable_shape.get(var, "no shape")
                st = self.variable_strides.get(var, "no strides")
                off = self.variable_offset.get(var, "")
                if off != "":
                    off = f", offset={off}"
                dtype = getattr(var, "dtype", "no dtype")
                print(
                    f"    input {int(idx)}: dtype={dtype}, shape={sh}, strides={st}{off}",
                    file=file,
                )
            for idx, var in enumerate(a.outputs):
                sh = self.variable_shape.get(var, "no shape")
                st = self.variable_strides.get(var, "no strides")
                off = self.variable_offset.get(var, "")
                if off != "":
                    off = f", offset={off}"
                dtype = getattr(var, "dtype", "no dtype")
                print(
                    f"    output {int(idx)}: dtype={dtype}, shape={sh}, strides={st}{off}",
                    file=file,
                )
            # Same as before, this I've sacrificied some information making
            # the output more readable
        print(
            "   ... (remaining %i Apply instances account for "
            "%.2f%%(%.2fs) of the runtime)"
            % (
                max(0, len(atimes) - N),
                sum(f for f, t, a, nd_id, nb_call in atimes[N:]),
                sum(t for f, t, a, nd_id, nb_call in atimes[N:]),
            ),
            file=file,
        )
        print("", file=file)

    def summary_function(self, file):
        print("Function profiling", file=file)
        print("==================", file=file)
        print(f"  Message: {self.message}", file=file)
        print(
            f"  Time in {self.fct_callcount} calls to Function.__call__: {self.fct_call_time:e}s",
            file=file,
        )
        if self.fct_call_time > 0:
            print(
                f"  Time in Function.fn.__call__: {self.vm_call_time}s ({100 * self.vm_call_time / self.fct_call_time:.3f}%)",
                file=file,
            )
            local_time = sum(self.apply_time.values())
            if local_time > 0:
                print(
                    f"  Time in thunks: {local_time}s ({100 * local_time / self.fct_call_time:.3f}%)",
                    file=file,
                )
        print(f"  Total compile time: {self.compile_time:e}s", file=file)
        print(f"    Number of Apply nodes: {int(self.nb_nodes)}", file=file)
        print(f"    Aesara Optimizer time: {self.optimizer_time:e}s", file=file)
        print(f"       Aesara validate time: {self.validate_time:e}s", file=file)
        print(
            (
                "    Aesara Linker time (includes C, CUDA code "
                f"generation/compiling): {self.linker_time}s"
            ),
            file=file,
        )
        print(f"       Import time {self.import_time:e}s", file=file)
        print(
            f"       Node make_thunk time {self.linker_node_make_thunks:e}s", file=file
        )

        for node, t in sorted(
            self.linker_make_thunk_time.items(), key=operator.itemgetter(1)
        )[::-1][:5]:
            print(f"           Node {node} time {t:e}s", file=file)
        print("", file=file)

        # The validation time is a subset of optimizer_time
        if self.optimizer_time > 0:
            assert self.validate_time < self.optimizer_time

    def summary_globals(self, file):
        print(
            f"Time in all call to aesara.grad() {aesara.gradient.grad_time:e}s",
            file=file,
        )
        total_time = time.time() - aesara_imported_time
        print(f"Time since aesara import {total_time:.3f}s", file=file)

    def summary_memory(self, file, N=None):
        fct_memory = {}  # fgraph->dict(node->[outputs size])
        fct_shapes = {}  # fgraph->dict(node->[outputs shapes]))
        var_mem = {}  # variable->size in bytes; don't include input variables
        node_mem = {}  # (fgraph, node)->total outputs size (only dense outputs)

        for (fgraph, node) in self.apply_callcount:
            fct_memory.setdefault(fgraph, {})
            fct_memory[fgraph].setdefault(node, [])
            fct_shapes.setdefault(fgraph, {})
            fct_shapes[fgraph].setdefault(node, [])
            sum_dense = 0
            for out in node.outputs:
                if out in self.variable_shape:
                    sh = self.variable_shape[out]
                    if hasattr(out.type, "get_size"):
                        v = out.type.get_size(sh)
                        sum_dense += v
                    else:
                        v = 0  # 'Unknown'
                else:
                    v = 0  # 'Variable isn't created'

                var_mem[out] = v
                fct_memory[fgraph][node].append(v)
                fct_shapes[fgraph][node].append(sh)
            node_mem[(fgraph, node)] = sum_dense
        del v

        # Find the function that used the most of that statistic
        max_sum_size = 0

        # statistics with the old and new order
        stats = [
            [[0, 0, 0], [0, 0, 0], 0, 0],  # old, with dmap
            [[0, 0, 0], [0, 0, 0], 0, 0],  # old, without dmap
            [[0, 0, 0], [0, 0, 0], 0, 0],  # new, with dmap
            [[0, 0, 0], [0, 0, 0], 0, 0],
        ]  # new, without dmap

        # track min peak memory usage
        min_max_peak = 0
        min_peak_time = 0

        def count_running_memory(order, fgraph, nodes_mem, ignore_dmap=False):
            """
            Calculate memory with specific node order.

            Return a list including the following values
            1.  node_memory_size
                Sum of the size of all variables that actually allocate
                memory (excluding views, and inplace).
            2.  running_memory_size
                The memory allocated after the current apply node.
            3.  running_max_memory_size
                The maximum of running_memory_size during the function.
            4.  node_memory_saved_by_view
                The sum of memory saved by returning view instead of new
                allocation.
            5.  node_memory_saved_by_inplace
                The sum of memory saved by reusing the input instead of
                new allocation.

            """
            from aesara.gpuarray import GpuArrayType

            # Initial Mem info values [CPU, GPU]
            node_memory_size = [0, 0]
            running_memory_size = [0, 0]
            running_max_memory_size = [0, 0]
            node_memory_saved_by_view = 0
            node_memory_saved_by_inplace = 0
            # This take only the inputs/outputs dependencies.
            dependencies = fgraph.profile.dependencies

            # Initial compute_map which is used to check if a node is valid
            compute_map = defaultdict(lambda: [0])
            for var in fgraph.inputs:
                compute_map[var][0] = 1

            # two data structure used to mimic Python gc
            viewed_by = {}  # {var1: [vars that view var1]}
            # The len of the list is the value of python ref
            # count. But we use a list, not just the ref count value.
            # This is more safe to help detect potential bug  in the algo
            for var in fgraph.variables:
                viewed_by[var] = []
            view_of = {}  # {var1: original var viewed by var1}
            # The orignal mean that we don't keep trac of all the intermediate
            # relationship in the view.

            for node in order:
                for var in node.outputs:
                    compute_map[var][0] = 1
                idx = 0
                if ignore_dmap:
                    dmap = None
                else:
                    dmap = node.op.destroy_map
                vmap = node.op.view_map
                val = nodes_mem[node]

                for v in val:
                    # TODO check the op returned a view
                    if dmap and idx in dmap:
                        node_memory_saved_by_inplace += v
                    # TODO check the op returned a view
                    elif vmap and idx in vmap:
                        node_memory_saved_by_view += v
                    idx += 1

                # Update the Python emulating dicts and add the memory
                # allocated by the node
                idx2 = 0
                for out in node.outputs:
                    if isinstance(out.type, GpuArrayType):
                        cg = 1
                    else:
                        cg = 0
                    ins = None
                    if dmap and idx2 in dmap:
                        vidx = dmap[idx2]
                        assert len(vidx) == 1, (
                            "Here we only support the "
                            "possibility to destroy one "
                            "input"
                        )
                        ins = node.inputs[vidx[0]]
                    if vmap and idx2 in vmap:
                        assert ins is None
                        vidx = vmap[idx2]
                        assert len(vidx) == 1, (
                            "Here we only support the "
                            "possibility to view one "
                            "input"
                        )
                        ins = node.inputs[vidx[0]]
                    if ins is not None:
                        # This is needed for destroy_map in case it
                        # return a partial view that is destroyed.  So
                        # the output could be different then the
                        # input.
                        assert isinstance(ins, Variable)
                        # we keep trac of view only again the origin
                        origin = view_of.get(ins, ins)
                        view_of[out] = origin
                        viewed_by[origin].append(out)
                    else:
                        running_memory_size[cg] += var_mem[out]
                        node_memory_size[cg] += var_mem[out]
                    idx2 += 1

                running_max_memory_size[0] = max(
                    running_max_memory_size[0], running_memory_size[0]
                )
                running_max_memory_size[1] = max(
                    running_max_memory_size[1], running_memory_size[1]
                )

                # Mimic the combination of Aesara and Python gc
                for ins in set(node.inputs):
                    assert not (ins in view_of and viewed_by[ins])
                    # we trac the original var, so this shouldn't happen
                    if isinstance(ins.type, GpuArrayType):
                        cg = 1
                    else:
                        cg = 0
                    if (
                        dependencies[ins]
                        and ins not in fgraph.outputs
                        and ins.owner
                        and all([compute_map[v][0] for v in dependencies[ins]])
                    ):
                        if ins not in view_of and not viewed_by.get(ins, []):
                            running_memory_size[cg] -= var_mem[ins]
                        elif ins in view_of:
                            origin = view_of[ins]
                            viewed_by[origin].remove(ins)
                            if (
                                not viewed_by[origin]
                                and origin not in fgraph.inputs
                                and not isinstance(origin, Constant)
                            ):
                                running_memory_size[cg] -= var_mem[origin]
                    else:
                        # ins is viewed_by something else, so its
                        # memory isn't freed
                        pass

            return [
                node_memory_size,
                running_memory_size,
                running_max_memory_size,
                node_memory_saved_by_inplace,
                node_memory_saved_by_view,
            ]

        def count_minimum_peak(node_list, fgraph, nodes_mem):
            global mem_count, mem_bound, max_mem_count
            node_list = list(node_list)
            mem_count = 0
            max_mem_count = 0
            mem_bound = np.inf
            # This take only the inputs/outputs dependencies.
            dependencies = fgraph.profile.dependencies
            done_set = set()
            done_dict = {}

            # Initial compute_map which is used to check if a node is valid
            compute_map = defaultdict(lambda: [0])
            for var in fgraph.inputs:
                compute_map[var][0] = 1
            for var in node_list:
                for val in var.inputs:
                    if isinstance(val, Constant):
                        compute_map[val][0] = 1

            # Initial executable_nodes
            executable_nodes = set()
            for var in fgraph.inputs:
                for c, _ in fgraph.clients[var]:
                    if c != "output":
                        deps = c.inputs + c.destroy_dependencies
                        if all(compute_map[v][0] for v in deps):
                            executable_nodes.add(c)

            def min_memory_generator(executable_nodes, viewed_by, view_of):
                """
                Generate all valid node order from node_list and compute its
                memory peak.

                Parameters
                ----------
                executable_nodes
                    Set of executable nodes.

                """
                global mem_count, mem_bound, max_mem_count

                for node in executable_nodes:
                    new_exec_nodes = executable_nodes.copy()
                    new_exec_nodes.remove(node)

                    # Check if cut path now
                    if max_mem_count > mem_bound:
                        continue

                    viewof_change = []
                    # Use to track view_of changes

                    viewedby_add = defaultdict(lambda: [])
                    viewedby_remove = defaultdict(lambda: [])
                    # Use to track viewed_by changes

                    for var in node.outputs:
                        compute_map[var][0] = 1

                    mem_created = 0
                    mem_freed = 0
                    max_storage = max_mem_count

                    dmap = node.op.destroy_map
                    vmap = node.op.view_map

                    idx = 0
                    # Update the Python emulating dicts and add the
                    # memory allocated by the node
                    for out in node.outputs:
                        ins = None
                        if dmap and idx in dmap:
                            vidx = dmap[idx]
                            assert len(vidx) == 1, (
                                "Here we only support "
                                "the possibility to "
                                "destroy one input"
                            )
                            ins = node.inputs[vidx[0]]
                        if vmap and idx in vmap:
                            assert ins is None
                            vidx = vmap[idx]
                            assert len(vidx) == 1, (
                                "Here we only support "
                                "the possibility to "
                                "view one input"
                            )
                            ins = node.inputs[vidx[0]]
                        if ins is not None:
                            # This is needed for destroy_map in case it
                            # return a partial view that is destroyed. So
                            # the output could be different then the
                            # input.
                            assert isinstance(ins, Variable)
                            # We keep track of view only again the original
                            origin = view_of.get(ins, ins)
                            view_of[out] = origin
                            viewof_change.append(out)
                            viewed_by[origin].append(out)
                            viewedby_add[origin].append(out)
                        else:
                            mem_created += var_mem[out]
                        idx += 1

                    mem_count += mem_created
                    max_mem_count = max(max_mem_count, mem_count)

                    # Mimic the combination of Aesara and Python gc.
                    for ins in node.inputs:
                        assert not (ins in view_of and viewed_by[ins])
                        # We track of the original var, so this shouldn't
                        # happen
                        if (
                            dependencies[ins]
                            and ins not in fgraph.outputs
                            and ins.owner
                            and all([compute_map[v][0] for v in dependencies[ins]])
                        ):
                            if ins not in view_of and not viewed_by.get(ins, []):
                                mem_freed += var_mem[ins]
                            elif ins in view_of:
                                origin = view_of[ins]
                                viewed_by[origin].remove(ins)
                                viewedby_remove[origin].append(ins)
                                if (
                                    not viewed_by[origin]
                                    and origin not in fgraph.inputs
                                    and not isinstance(origin, Constant)
                                ):
                                    mem_freed += var_mem[origin]
                        else:
                            # ins is viewed_by something else, so its
                            # memory isn't freed
                            pass

                    mem_count -= mem_freed

                    done_set.add(node)
                    frozen_set = frozenset(done_set)
                    if done_dict.get(frozen_set, max_mem_count + 1) > max_mem_count:
                        # check if frozen_set is in done_set
                        # no, add it to done_set
                        # yes, then compare the past mem and current mem
                        # bigger, update the value and continue
                        # smaller, stop this iteration, move to next node
                        done_dict[frozen_set] = max_mem_count

                        for var in node.outputs:
                            for c, _ in fgraph.clients[var]:
                                if c != "output":
                                    deps = c.inputs + c.destroy_dependencies
                                    if all(compute_map[v][0] for v in deps):
                                        new_exec_nodes.add(c)

                        if not new_exec_nodes:
                            # Check and Update mem_bound
                            if max_mem_count < mem_bound:
                                mem_bound = max_mem_count
                        else:
                            min_memory_generator(new_exec_nodes, viewed_by, view_of)

                    # Reset track variables
                    done_set.remove(node)
                    mem_count -= mem_created
                    max_mem_count = max_storage
                    mem_count += mem_freed
                    for var in node.outputs:
                        compute_map[var][0] = 0

                    for k_remove, v_remove in viewedby_remove.items():
                        for i in v_remove:
                            viewed_by[k_remove].append(i)

                    for k_add, v_add in viewedby_add.items():
                        for i in v_add:
                            viewed_by[k_add].remove(i)

                    for k in viewof_change:
                        del view_of[k]

            # two data structure used to mimic Python gc
            viewed_by = {}  # {var1: [vars that view var1]}
            # The len of the list is the value of python ref
            # count. But we use a list, not just the ref count value.
            # This is more safe to help detect potential bug  in the algo
            for var in fgraph.variables:
                viewed_by[var] = []
            view_of = {}  # {var1: original var viewed by var1}
            # The orignal mean that we don't keep trac of all the intermediate
            # relationship in the view.

            min_memory_generator(executable_nodes, viewed_by, view_of)

            return mem_bound

        for fgraph, nodes_mem in fct_memory.items():
            # Sum of the size of all variables in bytes
            sum_size = sum(
                sum(v for v in val if not isinstance(v, str))
                for key, val in nodes_mem.items()
            )

            order = fgraph.toposort()
            # A list of intermediate variable that are not need
            # after the execution of the corresponding node.
            # It mean that after executing the node,
            # the corresponding variable can be gc.

            # Store the max of some stats by any function in this profile.
            max_sum_size = max(max_sum_size, sum_size)

            def compute_max_stats(running_memory, stats):
                (
                    max_node_memory_size,
                    max_running_max_memory_size,
                    max_node_memory_saved_by_view,
                    max_node_memory_saved_by_inplace,
                ) = stats

                max_node_memory_size[0] = max(
                    max_node_memory_size[0], sum(running_memory[0])
                )
                max_running_max_memory_size[0] = max(
                    max_running_max_memory_size[0], sum(running_memory[2])
                )

                # Separate CPU and GPU
                max_node_memory_size[1] = max(
                    max_node_memory_size[1], running_memory[0][0]
                )
                max_node_memory_size[2] = max(
                    max_node_memory_size[2], running_memory[0][1]
                )
                max_running_max_memory_size[1] = max(
                    max_running_max_memory_size[1], running_memory[2][0]
                )
                max_running_max_memory_size[2] = max(
                    max_running_max_memory_size[2], running_memory[2][1]
                )

                max_node_memory_saved_by_inplace = max(
                    max_node_memory_saved_by_inplace, running_memory[3]
                )
                max_node_memory_saved_by_view = max(
                    max_node_memory_saved_by_view, running_memory[4]
                )
                return (
                    max_node_memory_size,
                    max_running_max_memory_size,
                    max_node_memory_saved_by_view,
                    max_node_memory_saved_by_inplace,
                )

            new_order = fgraph.profile.node_executed_order
            # A list of new executed node order
            for i, (ord, ignore_dmap) in enumerate(
                [(order, False), (order, True), (new_order, False), (new_order, True)]
            ):
                running_memory = count_running_memory(
                    ord, fgraph, nodes_mem, ignore_dmap=ignore_dmap
                )

                stats[i] = compute_max_stats(running_memory, stats[i])

            # Config: whether print min memory peak
            if config.profiling__min_peak_memory:
                node_list = fgraph.apply_nodes
                ttt = time.time()
                min_peak = count_minimum_peak(node_list, fgraph, nodes_mem)
                min_peak_time += time.time() - ttt
                min_max_peak = max(min_max_peak, min_peak)

            del fgraph, nodes_mem

        if len(fct_memory) > 1:
            print(
                "Memory Profile (the max between all functions in " "that profile)",
                file=file,
            )
        else:
            print("Memory Profile", file=file)

        print("(Sparse variables are ignored)", file=file)
        print("(For values in brackets, it's for linker = c|py", file=file)

        def print_stats(stats1, stats2):
            (_, max_running_max_memory_size, _, _) = stats1
            (_, new_max_running_max_memory_size, _, _) = stats2

            print(
                (
                    f"        CPU: {int(round(new_max_running_max_memory_size[1] / 1024.0))}KB "
                    f"({int(round(max_running_max_memory_size[1] / 1024.0))}KB)"
                ),
                file=file,
            )
            print(
                (
                    f"        GPU: {int(round(new_max_running_max_memory_size[2] / 1024.0))}KB "
                    f"({int(round(max_running_max_memory_size[2] / 1024.0))}KB)"
                ),
                file=file,
            )
            print(
                (
                    f"        CPU + GPU: {int(round(new_max_running_max_memory_size[0] / 1024.0))}KB "
                    f"({int(round(max_running_max_memory_size[0] / 1024.0))}KB)"
                ),
                file=file,
            )

        print("---", file=file)
        print("    Max peak memory with current setting", file=file)
        print_stats(stats[0], stats[2])
        print(
            "    Max peak memory with current setting and Aesara flag optimizer_excluding=inplace",
            file=file,
        )
        print_stats(stats[1], stats[3])

        (max_node_memory_size, _, _, _) = stats[0]
        (new_max_node_memory_size, _, _, _) = stats[2]
        print(
            "    Max peak memory if allow_gc=False (linker don't make a difference)",
            file=file,
        )
        print(
            f"        CPU: {int(round(new_max_node_memory_size[1] / 1024.0))}KB",
            file=file,
        )
        print(
            f"        GPU: {int(round(new_max_node_memory_size[2] / 1024.0))}KB",
            file=file,
        )
        print(
            f"        CPU + GPU: {int(round(new_max_node_memory_size[0] / 1024.0))}KB",
            file=file,
        )
        print("---", file=file)

        if min_max_peak:
            print(
                "    Minimum peak from all valid apply node order is "
                f"{int(round(min_max_peak / 1024.0))}KB(took {min_peak_time:3f}s to compute)",
                file=file,
            )

            print("---", file=file)

        print("", file=file)
        if len(fct_memory) > 1:
            print("    This list is based on all functions in the profile", file=file)
        print(
            "    <Sum apply outputs (bytes)>"
            " <Apply outputs shape>"
            " <created/inplace/view>"
            " <Apply node>",
            file=file,
        )
        print("", file=file)
        items = list(node_mem.items())
        items.sort(key=lambda a: a[1], reverse=True)
        for idx, ((fgraph, node), node_outputs_size) in enumerate(items[:N]):
            code = ["c"] * len(node.outputs)
            for out, inp in node.op.destroy_map.items():
                code[out] = "i"
            for out, inp in node.op.view_map.items():
                code[out] = "v"
            shapes = str(fct_shapes[fgraph][node])

            if all([hasattr(out.type, "get_size") for out in node.outputs]):
                size = "{node_outputs_size:9d}B"
                if node_outputs_size < config.profiling__min_memory_size:
                    N = idx
                    break
            else:
                size = "   Unknown"

            print(
                f"     {size}  {shapes} {' '.join(code)} {node}",
                file=file,
            )

        sum_remaining = sum(size for _, size in items[N:])
        size_sum_dense = sum(node_mem.values())
        if size_sum_dense == 0:
            p = "0%"
        else:
            p = f"({float(sum_remaining) / size_sum_dense * 100:.2f}%)"
        print(
            (
                f"   ... (remaining {max(0, len(node_mem) - N)} Apply account for "
                f"{sum_remaining:4d}B/{size_sum_dense :d}B ({p}) of the"
                " Apply with dense outputs sizes)"
            ),
            file=file,
        )
        print("", file=file)
        if N == 0:
            print(
                "    All Apply nodes have output sizes that take less "
                f"than {int(config.profiling__min_memory_size)}B.",
                file=file,
            )
        print(
            "    <created/inplace/view> is taken from the Op's declaration.", file=file
        )
        print(
            "    Apply nodes marked 'inplace' or 'view' may"
            " actually allocate memory, this is not reported"
            " here. If you use DebugMode, warnings will be"
            " emitted in those cases.",
            file=file,
        )
        print("", file=file)

    def summary(self, file=sys.stderr, n_ops_to_print=20, n_apply_to_print=20):
        self.summary_function(file)
        self.summary_globals(file)
        local_time = sum(self.apply_time.values())
        if local_time > 0:
            self.summary_class(file, n_ops_to_print)
            self.summary_ops(file, n_ops_to_print)
            self.summary_nodes(file, n_apply_to_print)
        elif self.fct_callcount > 0:
            print(
                "  No execution time accumulated "
                "(hint: try config profiling__time_thunks=1)",
                file=file,
            )
        if config.profiling__debugprint:
            fcts = {fgraph for (fgraph, n) in self.apply_time.keys()}
            aesara.printing.debugprint(fcts, print_type=True)
        if self.variable_shape or self.variable_strides:
            self.summary_memory(file, n_apply_to_print)
        if self.optimizer_profile:
            print("Optimizer Profile", file=file)
            print("-----------------", file=file)
            self.optimizer_profile[0].print_profile(file, self.optimizer_profile[1])
        self.print_extra(file)
        self.print_tips(file)

    def print_tips(self, file):
        print(
            """Here are tips to potentially make your code run faster
                 (if you think of new ones, suggest them on the mailing list).
                 Test them first, as they are not guaranteed to always provide a speedup.""",
            file=file,
        )

        from aesara import scalar as aes
        from aesara.tensor.elemwise import Elemwise
        from aesara.tensor.math import Dot
        from aesara.tensor.random.op import RandomVariable

        scalar_op_amdlibm_no_speed_up = [
            aes.LT,
            aes.GT,
            aes.LE,
            aes.GE,
            aes.EQ,
            aes.NEQ,
            aes.InRange,
            aes.Switch,
            aes.OR,
            aes.XOR,
            aes.AND,
            aes.Invert,
            aes.ScalarMaximum,
            aes.ScalarMinimum,
            aes.Add,
            aes.Mul,
            aes.Sub,
            aes.TrueDiv,
            aes.IntDiv,
            aes.Clip,
            aes.Second,
            aes.Identity,
            aes.Cast,
            aes.Sgn,
            aes.Neg,
            aes.Inv,
            aes.Sqr,
        ]
        scalar_op_amdlibm_speed_up = [
            aes.Mod,
            aes.Pow,
            aes.Ceil,
            aes.Floor,
            aes.RoundHalfToEven,
            aes.RoundHalfAwayFromZero,
            aes.Log,
            aes.Log2,
            aes.Log10,
            aes.Log1p,
            aes.Exp,
            aes.Sqrt,
            aes.Abs,
            aes.Cos,
            aes.Sin,
            aes.Tan,
            aes.Tanh,
            aes.Cosh,
            aes.Sinh,
            aes.Sigmoid,
            aes.Softplus,
        ]

        def get_scalar_ops(s):
            if isinstance(s, aes.Composite):
                l = []
                for node in s.fgraph.toposort():
                    l += get_scalar_ops(node.op)
                return l
            else:
                return [s]

        def list_scalar_op(op):
            if isinstance(op.scalar_op, aes.Composite):
                return get_scalar_ops(op.scalar_op)
            else:
                return [op.scalar_op]

        def amdlibm_speed_up(op):
            if not isinstance(op, Elemwise):
                return False
            else:
                l = list_scalar_op(op)
                for s_op in l:
                    if s_op.__class__ in scalar_op_amdlibm_speed_up:
                        return True
                    elif s_op.__class__ not in scalar_op_amdlibm_no_speed_up:
                        print(
                            "We don't know if amdlibm will accelerate "
                            "this scalar op.",
                            s_op,
                            file=file,
                        )
                return False

        def exp_float32_op(op):
            if not isinstance(op, Elemwise):
                return False
            else:
                l = list_scalar_op(op)
                return any([s_op.__class__ in [aes.Exp] for s_op in l])

        printed_tip = False
        # tip 1
        if config.floatX == "float64":
            print("  - Try the Aesara flag floatX=float32", file=file)
            printed_tip = True

        # tip 2
        if not config.lib__amblibm and any(
            [amdlibm_speed_up(a.op) for (fgraph, a) in self.apply_time]
        ):
            print(
                "  - Try installing amdlibm and set the Aesara flag "
                "lib__amblibm=True. This speeds up only some Elemwise "
                "operation.",
                file=file,
            )
            printed_tip = True

        # tip 3
        if not config.lib__amblibm and any(
            [
                exp_float32_op(a.op) and a.inputs[0].dtype == "float32"
                for (fgraph, a) in self.apply_time
            ]
        ):
            print(
                "  - With the default gcc libm, exp in float32 is slower "
                "than in float64! Try Aesara flag floatX=float64, or "
                "install amdlibm and set the aesara flags lib__amblibm=True",
                file=file,
            )
            printed_tip = True

        # tip 4
        for (fgraph, a) in self.apply_time:
            node = a
            if isinstance(node.op, Dot) and all(
                [len(i.type.broadcastable) == 2 for i in node.inputs]
            ):
                print(
                    (
                        "  - You have a dot operation that was not optimized to"
                        " dot22 (which is faster). Make sure the inputs are "
                        "float32 or float64, and are the same for both inputs. "
                        f"Currently they are: {[i.type for i in node.inputs]}"
                    ),
                    file=file,
                )
                printed_tip = True

        # tip 5
        for (fgraph, a) in self.apply_time:
            node = a
            if isinstance(node.op, RandomVariable):
                printed_tip = True
                print(
                    "  - Replace the default random number generator by "
                    "'from aesara.sandbox.rng_mrg import MRG_RandomStream "
                    "as RandomStream', as this is is faster. It is still "
                    "experimental, but seems to work correctly.",
                    file=file,
                )
                if config.device.startswith("gpu"):
                    print(
                        "     - MRG_RandomStream is the only random number"
                        " generator supported on the GPU.",
                        file=file,
                    )
                break

        # tip 6
        for (fgraph, a) in self.apply_time:
            node = a
            if isinstance(node.op, Dot) and len({i.dtype for i in node.inputs}) != 1:
                print(
                    (
                        "  - You have a dot operation that has different dtype "
                        f" for inputs ({[i.type for i in node.inputs]}). Make sure that the inputs have same "
                        " dtype."
                    ),
                    file=file,
                )
                printed_tip = True

        # tip 7
        import aesara.gpuarray
        import aesara.tensor.signal.pool as pool
        from aesara.tensor.nnet.basic import LogSoftmax

        for (fgraph, a) in self.apply_time:
            node = a
            if isinstance(node.op, pool.Pool):
                if not aesara.gpuarray.dnn.dnn_present():
                    print(
                        "Install CuDNN to do pooling faster"
                        "this allows the operation to run on GPU"
                    )
                    printed_tip = True
            if isinstance(node.op, LogSoftmax):
                if not aesara.gpuarray.dnn.dnn_present():
                    print(
                        "Install CuDNN to do LogSoftmax faster"
                        "this allows the operation to run on GPU"
                    )
                    printed_tip = True

        if not printed_tip:
            print("  Sorry, no tip for today.", file=file)

    def print_extra(self, file):
        params = [
            self.message,
            self.compile_time,
            self.fct_call_time,
            self.apply_time,
            self.apply_cimpl,
            self.output_size,
        ]
        for f in _profiler_printers:
            f(*params, file=file)


class ScanProfileStats(ProfileStats):
    callcount = 0.0
    nbsteps = 0.0
    call_time = 0.0

    def __init__(self, atexit_print=True, name=None, **kwargs):
        super().__init__(atexit_print, **kwargs)
        self.name = name

    def summary_globals(self, file):
        # Do nothing, we don't want to print extra global summary
        # here.
        pass

    def summary_function(self, file):
        # RP: every time we compile a function a ProfileStats is created for
        # that function. This means that every time a optimization replaces
        # some scan op, some orphane ProfileStats remains in the air ..
        # also even without any optimization, scan compiles a dummy function
        # that will produce a ProfileStats that will correspond to a
        # function that will never be called. Printing several empty
        # Function profiling is just extremely confusing
        if self.callcount == 0:
            return
        print("", file=file)

        if self.name is not None:
            print("Scan Op profiling (", self.name, ")", file=file)
        else:
            print("Scan Op profiling", file=file)
        print("==================", file=file)
        print(f"  Message: {self.message}", file=file)

        print(
            (
                f"  Time in {self.callcount} calls of the op (for a total of {self.nbsteps} "
                f"steps) {self.call_time:3}s"
            ),
            file=file,
        )
        print("", file=file)
        val = 0
        if self.call_time > 0:
            val = self.vm_call_time * 100 / self.call_time
        print(
            f"  Total time spent in calling the VM {self.vm_call_time:e}s ({val:.3f}%)",
            file=file,
        )
        val = 100
        if self.call_time > 0:
            val = 100.0 - self.vm_call_time * 100 / self.call_time
        print(
            f"  Total overhead (computing slices..) {self.call_time - self.vm_call_time:e}s ({val:.3f}%)",
            file=file,
        )
        print("", file=file)
