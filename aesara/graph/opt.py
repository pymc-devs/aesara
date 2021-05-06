"""
Defines the base class for optimizations as well as a certain
amount of useful generic optimization tools.

"""
import abc
import contextlib
import copy
import inspect
import logging
import pdb
import sys
import time
import traceback
import warnings
from collections import OrderedDict, UserList, defaultdict, deque
from collections.abc import Iterable
from functools import reduce

import numpy as np

import aesara
from aesara.assert_op import Assert, assert_op
from aesara.configdefaults import config
from aesara.graph import destroyhandler as dh
from aesara.graph.basic import (
    Apply,
    Constant,
    Variable,
    applys_between,
    io_toposort,
    nodes_constructed,
)
from aesara.graph.fg import InconsistencyError
from aesara.graph.op import Op
from aesara.graph.toolbox import Feature, NodeFinder
from aesara.graph.utils import AssocList
from aesara.misc.ordered_set import OrderedSet
from aesara.utils import flatten


_logger = logging.getLogger("aesara.graph.opt")
_optimizer_idx = [0]


class LocalMetaOptimizerSkipAssertionError(AssertionError):
    """This is an AssertionError, but instead of having the
    LocalMetaOptimizer print the error, it just skip that
    compilation.

    """


class GlobalOptimizer(abc.ABC):
    """

    A L{GlobalOptimizer} can be applied to an L{FunctionGraph} to transform it.
    It can represent an optimization or in general any kind
    of transformation you could apply to an L{FunctionGraph}.

    """

    @abc.abstractmethod
    def apply(self, fgraph):
        """

        Applies the optimization to the provided L{FunctionGraph}. It may
        use all the methods defined by the L{FunctionGraph}. If the
        L{GlobalOptimizer} needs to use a certain tool, such as an
        L{InstanceFinder}, it can do so in its L{add_requirements} method.

        """
        raise NotImplementedError()

    def optimize(self, fgraph, *args, **kwargs):
        """

        This is meant as a shortcut to:
          opt.add_requirements(fgraph)
          opt.apply(fgraph)

        """
        self.add_requirements(fgraph)
        ret = self.apply(fgraph, *args, **kwargs)
        return ret

    def __call__(self, fgraph):
        """

        Same as self.optimize(fgraph).

        """
        return self.optimize(fgraph)

    def add_requirements(self, fgraph):
        """

        Add features to the fgraph that are required to apply the optimization.
        For example:
          fgraph.attach_feature(History())
          fgraph.attach_feature(MyFeature())
          etc.

        """

    def print_summary(self, stream=sys.stdout, level=0, depth=-1):
        name = getattr(self, "name", None)
        print(
            f"{' ' * level}{self.__class__.__name__} {name} id={id(self)}",
            file=stream,
        )

    @staticmethod
    def print_profile(stream, prof, level=0):
        if prof is not None:
            raise NotImplementedError(
                "The function print_profile must be overrided if the"
                " optimizer return profiling information."
            )

    def __hash__(self):
        if not hasattr(self, "_optimizer_idx"):
            self._optimizer_idx = _optimizer_idx[0]
            _optimizer_idx[0] += 1
        return self._optimizer_idx

    def __str__(self):
        if hasattr(self, "name"):
            return f"{type(self).__name__}[{self.name}]"
        return repr(self)


class FromFunctionOptimizer(GlobalOptimizer):
    """A `GlobalOptimizer` constructed from a given function."""

    def __init__(self, fn, requirements=()):
        self.fn = fn
        self.requirements = requirements

    def apply(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def add_requirements(self, fgraph):
        for req in self.requirements:
            req(fgraph)

    def print_summary(self, stream=sys.stdout, level=0, depth=-1):
        print(f"{' ' * level}{self.apply} id={id(self)}", file=stream)

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def __str__(self):
        return self.__name__


def optimizer(f):
    """
    Decorator for FromFunctionOptimizer.

    """
    rval = FromFunctionOptimizer(f)
    rval.__name__ = f.__name__
    return rval


def inplace_optimizer(f):
    """
    Decorator for FromFunctionOptimizer.

    """
    dh_handler = dh.DestroyHandler
    requirements = (lambda fgraph: fgraph.attach_feature(dh_handler()),)
    rval = FromFunctionOptimizer(f, requirements)
    rval.__name__ = f.__name__
    return rval


class SeqOptimizer(GlobalOptimizer, UserList):
    """A `GlobalOptimizer` that applies a list of optimizers sequentially."""

    @staticmethod
    def warn(exc, self, optimizer):
        """
        Default failure_callback for SeqOptimizer.

        """
        _logger.error(f"SeqOptimizer apply {optimizer}")
        _logger.error("Traceback:")
        _logger.error(traceback.format_exc())
        if config.on_opt_error == "raise":
            raise exc
        elif config.on_opt_error == "pdb":
            pdb.post_mortem(sys.exc_info()[2])

    def __init__(self, *opts, **kw):
        """
        Parameters
        ----------
        *opts :
            The List of optimizers to be applied to a node
        failure_callback : callable or None
            Keyword only argument. A callback used when a failure
            happen during optimization.

        """
        if len(opts) == 1 and isinstance(opts[0], (list, tuple)):
            opts = opts[0]

        super().__init__(opts)

        self.failure_callback = kw.pop("failure_callback", None)
        assert len(kw) == 0

    def apply(self, fgraph):
        """

        Applies each L{GlobalOptimizer} in self in turn.

        """
        l = []
        if fgraph.profile:
            validate_before = fgraph.profile.validate_time
            sub_validate_time = [validate_before]
            callbacks_before = fgraph.execute_callbacks_times.copy()
        else:
            sub_validate_time = []
            callbacks_before = []
        callback_before = fgraph.execute_callbacks_time
        nb_node_before = len(fgraph.apply_nodes)
        sub_profs = []
        nb_nodes = []

        self.pre_profile = (
            self,
            l,
            -1,
            -1,
            nb_node_before,
            -1,
            sub_profs,
            sub_validate_time,
            nb_nodes,
            {},
        )
        try:
            for optimizer in self.data:
                try:
                    nb_nodes_before = len(fgraph.apply_nodes)
                    t0 = time.time()
                    sub_prof = optimizer.optimize(fgraph)
                    l.append(float(time.time() - t0))
                    sub_profs.append(sub_prof)
                    nb_nodes.append((nb_nodes_before, len(fgraph.apply_nodes)))
                    if fgraph.profile:
                        sub_validate_time.append(fgraph.profile.validate_time)
                except AssertionError:
                    # do not catch Assertion failures
                    raise
                except Exception as e:
                    if self.failure_callback:
                        self.failure_callback(e, self, optimizer)
                        continue
                    else:
                        raise
        finally:

            if fgraph.profile:
                validate_time = fgraph.profile.validate_time - validate_before
                callbacks_time = {}
                for k, v in fgraph.execute_callbacks_times.items():
                    if k in callbacks_before:
                        t = v - callbacks_before[k]
                        if t > 0:
                            callbacks_time[k] = t
                    else:
                        callbacks_time[k] = v
            else:
                validate_time = None
                callbacks_time = {}
            callback_time = fgraph.execute_callbacks_time - callback_before
            self.pre_profile = (
                self,
                l,
                validate_time,
                callback_time,
                nb_node_before,
                len(fgraph.apply_nodes),
                sub_profs,
                sub_validate_time,
                nb_nodes,
                callbacks_time,
            )
        return self.pre_profile

    def __repr__(self):
        return f"SeqOpt({self.data})"

    def print_summary(self, stream=sys.stdout, level=0, depth=-1):
        name = getattr(self, "name", None)
        print(
            f"{' ' * level}{self.__class__.__name__} {name} id={id(self)}", file=stream
        )
        # This way, -1 will do all depth
        if depth != 0:
            depth -= 1
            for opt in self.data:
                opt.print_summary(stream, level=(level + 2), depth=depth)

    @staticmethod
    def print_profile(stream, prof, level=0):
        (
            opts,
            prof,
            validate_time,
            callback_time,
            nb_node_before,
            nb_node_after,
            sub_profs,
            sub_validate_time,
            nb_nodes,
            callbacks_time,
        ) = prof
        blanc = "    " * level

        print(blanc, "SeqOptimizer", end=" ", file=stream)
        if hasattr(opts, "name"):
            print(blanc, opts.name, end=" ", file=stream)
        elif hasattr(opts, "__name__"):
            print(blanc, opts.__name__, end=" ", file=stream)
        print(
            (
                f" time {sum(prof):.3f}s for {int(nb_node_before)}/{int(nb_node_after)} nodes"
                " before/after optimization"
            ),
            file=stream,
        )
        print(blanc, f"  {callback_time:.3f}s for callback", file=stream)
        print(blanc, f"      {validate_time:.3f}s for fgraph.validate()", file=stream)
        if callback_time > 1:
            print(blanc, "  callbacks_time", file=stream)
            for i in sorted(callbacks_time.items(), key=lambda a: -a[1]):
                if i[1] > 0:
                    # We want to have the __str__ called, so we can't
                    # just print i.
                    print(blanc, "      ", i[0], ",", i[1], file=stream)

        if level == 0:
            print(
                blanc,
                "  time      - (name, class, index, nodes before, nodes after) - validate time",
                file=stream,
            )
        ll = []
        for (opt, nb_n) in zip(opts, nb_nodes):
            if hasattr(opt, "__name__"):
                name = opt.__name__
            else:
                name = opt.name
            idx = opts.index(opt)
            ll.append((name, opt.__class__.__name__, idx) + nb_n)
        lll = sorted(zip(prof, ll), key=lambda a: a[0])

        for (t, opt) in lll[::-1]:
            i = opt[2]
            if sub_validate_time:
                val_time = sub_validate_time[i + 1] - sub_validate_time[i]
                print(
                    blanc,
                    f"  {t:.6f}s - {opt} - {val_time:.3f}s",
                    file=stream,
                )
            else:
                print(blanc, f"  {t:.6f}s - {opt}", file=stream)

            if sub_profs[i]:
                opts[i].print_profile(stream, sub_profs[i], level=level + 1)
        print(file=stream)

    @staticmethod
    def merge_profile(prof1, prof2):
        """
        Merge 2 profiles returned by this cass apply() fct.

        """
        new_t = []  # the time for the optimization
        new_l = []  # the optimization
        new_sub_profile = []
        # merge common(same object) opt
        for l in set(prof1[0]).intersection(set(prof2[0])):
            idx1 = prof1[0].index(l)
            idx2 = prof2[0].index(l)
            new_t.append(prof1[1][idx1] + prof2[1][idx2])
            new_l.append(l)
            if hasattr(l, "merge_profile"):
                assert len(prof1[6][idx1]) == len(prof2[6][idx2])
                new_sub_profile.append(l.merge_profile(prof1[6][idx1], prof2[6][idx2]))
            else:
                new_sub_profile.append(None)

        # merge not common opt
        from io import StringIO

        for l in set(prof1[0]).symmetric_difference(set(prof2[0])):
            # The set trick above only work for the same object optimization
            # It don't work for equivalent optimization.
            # So we try to merge equivalent optimization here.
            new_l_names = [o.name for o in new_l]
            if l.name in new_l_names:
                idx = new_l_names.index(l.name)
                io1 = StringIO()
                io2 = StringIO()
                l.print_summary(io1)
                new_l[idx].print_summary(io2)
                if io1.read() == io2.read():
                    if l in prof1[0]:
                        p = prof1
                    else:
                        p = prof2
                    new_t[idx] += p[1][p[0].index(l)]
                    if hasattr(l, "merge_profile"):
                        assert len(p[6][p[0].index(l)]) == len(new_sub_profile[idx])
                        new_sub_profile[idx] = l.merge_profile(
                            new_sub_profile[idx], p[6][p[0].index(l)]
                        )
                    else:
                        new_sub_profile[idx] = None
                continue
            if l in prof1[0]:
                p = prof1
            else:
                p = prof2
            new_t.append(p[1][p[0].index(l)])
            idx = p[0].index(l)
            new_l.append(l)
            new_sub_profile.append(p[6][idx])

        new_opt = SeqOptimizer(*new_l)
        new_nb_nodes = []
        for p1, p2 in zip(prof1[8], prof2[8]):
            new_nb_nodes.append((p1[0] + p2[0], p1[1] + p2[1]))
        new_nb_nodes.extend(prof1[8][len(new_nb_nodes) :])
        new_nb_nodes.extend(prof2[8][len(new_nb_nodes) :])

        new_callbacks_times = merge_dict(prof1[9], prof2[9])
        # We need to assert based on the name as we merge also based on
        # the name.
        assert {l.name for l in prof1[0]}.issubset({l.name for l in new_l})
        assert {l.name for l in prof2[0]}.issubset({l.name for l in new_l})
        assert len(new_t) == len(new_opt) == len(new_sub_profile)
        return (
            new_opt,
            new_t,
            prof1[2] + prof2[2],
            prof1[3] + prof2[3],
            -1,
            -1,
            new_sub_profile,
            [],
            new_nb_nodes,
            new_callbacks_times,
        )


class MergeFeature(Feature):
    """Keeps track of variables in a `FunctionGraph` that cannot be merged together.

    That way, the `MergeOptimizer` can remember the result of the last
    merge-pass on the `FunctionGraph`.

    """

    def on_attach(self, fgraph):
        assert not hasattr(fgraph, "merge_feature")
        fgraph.merge_feature = self

        # For constants
        self.seen_constants = set()
        # variable -> signature (for constants)
        self.const_sig = AssocList()
        # signature -> variable (for constants)
        self.const_sig_inv = AssocList()

        # For all Apply nodes
        # Set of distinct (not mergeable) nodes
        self.nodes_seen = set()
        # Ordered set of distinct (not mergeable) nodes without any input
        self.noinput_nodes = OrderedSet()

        # Each element of scheduled is a list of list of (out, new_out) pairs.
        # Each list of pairs represent the substitution needed to replace all
        # the outputs of a node with the outputs of a replacement candidate.
        # Each node can have several candidates. For instance, if "node" has
        # 2 outputs, and there are 3 replacement candidates, we will have:
        # shelf.scheduled = [
        #    [[(node.out1, cand1.out1), (node.out2, cand1.out2)],
        #     [(node.out1, cand2.out1), (node.out2, cand2.out2)],
        #     [(node.out1, cand3.out1), (node.out2, cand3.out2)]]]
        self.scheduled = []

        # List of (node, candidate) pairs, where we tried to replace node by
        # candidate, but it failed. This is used to avoid infinite loops
        # during the replacement phase.
        self.blacklist = []

        for node in fgraph.toposort():
            self.on_import(fgraph, node, "on_attach")

    def on_change_input(self, fgraph, node, i, r, new_r, reason):
        # If inputs to node change, it is not guaranteed that it is distinct
        # from the other nodes in nodes_seen
        if node in self.nodes_seen:
            self.nodes_seen.discard(node)
            self.process_node(fgraph, node)

        # Since we are in on_change_input, node should have inputs.
        if not isinstance(node, str):
            assert node.inputs

        if isinstance(new_r, Constant):
            self.process_constant(fgraph, new_r)

    def on_import(self, fgraph, node, reason):
        for c in node.inputs:
            if isinstance(c, Constant):
                self.process_constant(fgraph, c)

        self.process_node(fgraph, node)

    def on_prune(self, fgraph, node, reason):
        self.nodes_seen.discard(node)
        if not node.inputs:
            self.noinput_nodes.discard(node)
        for c in node.inputs:
            if isinstance(c, Constant) and (len(fgraph.clients[c]) <= 1):
                # This was the last node using this constant
                sig = self.const_sig[c]
                self.const_sig.discard(c)
                self.const_sig_inv.discard(sig)
                self.seen_constants.discard(id(c))

    def process_constant(self, fgraph, c):
        """
        Check if a constant can be merged, and queue that replacement.

        """
        if id(c) in self.seen_constants:
            return
        sig = c.merge_signature()
        other_c = self.const_sig_inv.get(sig, None)
        if other_c is not None:
            # multiple names will clobber each other..
            # we adopt convention to keep the last name
            if c.name:
                other_c.name = c.name
            self.scheduled.append([[(c, other_c, "merge")]])
        else:
            # this is a new constant
            self.const_sig[c] = sig
            self.const_sig_inv[sig] = c
            self.seen_constants.add(id(c))

    def process_node(self, fgraph, node):
        """
        Check if a node can be merged, and queue that replacement.

        """
        if node in self.nodes_seen:
            return

        node_has_assert = False

        # These asserts ensure that the fgraph has set the clients field
        # properly.
        # The clients should at least contain `node` itself!
        if node.inputs:
            # Take the smallest clients list. Some ops like elemwise
            # have optimization that put constant as the first inputs.
            # As constant have in general more clients than other type of nodes
            # using always inputs[0] make us look at more nodes.
            # Always pick the smallest clints list between inputs 0
            # and -1 speed up optimization.

            a_clients = fgraph.clients[node.inputs[0]]
            b_clients = fgraph.clients[node.inputs[-1]]
            if len(a_clients) < len(b_clients):
                clients = a_clients
            else:
                clients = b_clients
            assert len(clients) > 0

            merge_candidates = [c for c, i in clients if c in self.nodes_seen]

            # Put all clients of Assert inputs (if exist) into merge_candidates
            # TODO: Deactivated for now as this cause cycle in the graph.
            # (There is a second deactivation part below.)
            for i in []:  # node.inputs:
                if i.owner and isinstance(i.owner.op, Assert):
                    node_has_assert = True
                    i_clients = fgraph.clients[i.owner.inputs[0]]
                    assert_clients = [c for (c, _) in i_clients if c in self.nodes_seen]

                    for idx in range(len(assert_clients)):
                        client = assert_clients[idx]
                        if isinstance(i.owner.op, Assert):
                            o_clients = fgraph.clients[client.outputs[0]]
                            for c in o_clients:
                                if c[0] in self.nodes_seen:
                                    assert_clients.append(c[0])

                    merge_candidates.extend(assert_clients)
        else:
            # If two nodes have no input, but perform the same operation,
            # they are not always constant-folded, so we want to merge them.
            # In that case, the candidates are all the nodes without inputs.
            merge_candidates = self.noinput_nodes

        replacement_candidates = []
        for candidate in merge_candidates:
            if candidate is node:
                continue
            if len(node.inputs) != len(candidate.inputs):
                continue

            cand_has_assert = False

            # Get input list of the candidate with assert removed
            cand_inputs_assert_removed = []
            # TODO: Deactivated while Assert merging is disabled. (See above and below.)
            for i in []:  # candidate.inputs:
                if i.owner and isinstance(i.owner.op, Assert):
                    cand_has_assert = True
                    cand_inputs_assert_removed.append(i.owner.inputs[0])
                else:
                    cand_inputs_assert_removed.append(i)

            # TODO: Remove this when Assert merging is re-enabled. (See above.)
            # Without Assert merging we can still look for identical Asserts,
            # so we should not treat Asserts separately for now.
            cand_inputs_assert_removed = candidate.inputs

            # Get input list of the node with assert removed
            if node_has_assert:
                node_inputs_assert_removed = []
                for i in node.inputs:
                    if i.owner and isinstance(i.owner.op, Assert):
                        node_inputs_assert_removed.append(i.owner.inputs[0])
                    else:
                        node_inputs_assert_removed.append(i)
            else:
                node_inputs_assert_removed = node.inputs

            inputs_match = all(
                node_in is cand_in
                for node_in, cand_in in zip(
                    node_inputs_assert_removed, cand_inputs_assert_removed
                )
            )

            if inputs_match and node.op == candidate.op:
                if (node, candidate) in self.blacklist:
                    # They were already tried, and there was an error
                    continue

                # replace node with candidate
                if not (node_has_assert or cand_has_assert):
                    # Schedule transfer of clients from node to candidate
                    pairs = list(
                        zip(
                            node.outputs,
                            candidate.outputs,
                            ["merge"] * len(node.outputs),
                        )
                    )

                # if the current node has assert input, it should not be
                # replaced with a candidate node which has no assert input
                elif node_has_assert and not cand_has_assert:
                    pairs = list(
                        zip(
                            candidate.outputs,
                            node.outputs,
                            ["merge"] * len(node.outputs),
                        )
                    )
                else:
                    new_inputs = self.get_merged_assert_input(node, candidate)
                    new_node = node.op(*new_inputs)
                    pairs = list(
                        zip(
                            node.outputs,
                            new_node.owner.outputs,
                            ["new_node"] * len(node.outputs),
                        )
                    ) + list(
                        zip(
                            candidate.outputs,
                            new_node.owner.outputs,
                            ["new_node"] * len(node.outputs),
                        )
                    )

                # transfer names
                for pair in pairs:
                    node_output, cand_output = pair[:2]
                    # clobber old name with new one
                    # it's arbitrary... one of the names has to go
                    if node_output.name:
                        cand_output.name = node_output.name

                replacement_candidates.append(pairs)

        if replacement_candidates:
            self.scheduled.append(replacement_candidates)
        else:
            self.nodes_seen.add(node)
            if not node.inputs:
                self.noinput_nodes.add(node)

    def get_merged_assert_input(self, node, candidate):
        new_inputs = []
        for node_i, cand_i in zip(node.inputs, candidate.inputs):
            # if node_i is assert
            if node_i.owner and isinstance(node_i.owner.op, Assert):
                # node_i is assert, cand_i is assert
                if cand_i.owner and isinstance(cand_i.owner.op, Assert):
                    # Here two assert nodes are merged.
                    # Step 1. Merge conditions of both assert nodes.
                    # Step 2. Make the new assert node
                    node_cond = node_i.owner.inputs[1:]
                    cand_cond = cand_i.owner.inputs[1:]
                    new_cond = list(set(node_cond + cand_cond))
                    new_inputs.append(assert_op(node_i.owner.inputs[0], *new_cond))

                # node_i is assert, cand_i is not assert
                else:
                    new_inputs.append(node_i)
            else:
                # if node_i is not an assert node, append cand_i
                new_inputs.append(cand_i)

        return new_inputs


class MergeOptimizer(GlobalOptimizer):
    """
    Merges parts of the graph that are identical and redundant.

    The basic principle is that if two Applies have ops that compare equal, and
    identical inputs, then they do not both need to be computed. The clients of
    one are transferred to the other and one of them is removed from the graph.
    This procedure is carried out in input->output order through the graph.

    The first step of merging is constant-merging, so that all clients of an
    int(1) for example, are transferred to a particular instance of int(1).

    """

    def add_requirements(self, fgraph):
        if not hasattr(fgraph, "merge_feature"):
            fgraph.attach_feature(MergeFeature())

    def apply(self, fgraph):
        # Constant and non-constant are now applied in the same phase.
        # I am not sure why, but it seems to be faster this way.
        sched = fgraph.merge_feature.scheduled
        nb_fail = 0
        t0 = time.time()
        if fgraph.profile:
            validate_before = fgraph.profile.validate_time
            callback_before = fgraph.execute_callbacks_time
            callbacks_before = fgraph.execute_callbacks_times.copy()

        nb_merged = 0
        nb_constant = 0
        while sched:
            pairs_list = sched.pop()
            success = True
            for pairs_ in pairs_list:
                # We must check again the equivalence, as the graph
                # could've changed. If so, doing the replacement can
                # introduce a node that depends on itself.  Doing the
                # full check of such cycles every time is very time
                # consuming. I think this double check is faster than
                # doing the full cycle check. The full cycle check is
                # skipped by validate() if the graph doesn't contain
                # destroyers.
                var, candidate, merge_mode = pairs_[0]
                if merge_mode == "new_node" and var in fgraph.variables:
                    pass
                elif var not in fgraph.variables or candidate not in fgraph.variables:
                    continue

                # Keep len(item) == 2 for item in pairs
                pairs = [pair[:2] for pair in pairs_]

                if var.owner and candidate.owner:
                    node = var.owner
                    candidate = candidate.owner

                    # Get input list of the candidate node with assert
                    # nodes removed
                    cand_inputs_assert_removed = []
                    for i in candidate.inputs:
                        if i.owner and isinstance(i.owner.op, Assert):
                            cand_inputs_assert_removed.append(i.owner.inputs[0])
                        else:
                            cand_inputs_assert_removed.append(i)

                    # Get input list of the node with assert nodes removed
                    node_inputs_assert_removed = []
                    for i in node.inputs:
                        if i.owner and isinstance(i.owner.op, Assert):
                            node_inputs_assert_removed.append(i.owner.inputs[0])
                        else:
                            node_inputs_assert_removed.append(i)

                    if merge_mode == "new_node":
                        inputs_match = True
                    else:
                        inputs_match = all(
                            node_in is cand_in
                            for node_in, cand_in in zip(
                                node_inputs_assert_removed, cand_inputs_assert_removed
                            )
                        )

                    # No need to compare the op again, as it don't change.
                    if not inputs_match:
                        continue

                    if hasattr(fgraph, "destroy_handler"):
                        # If both nodes have clients that destroy them, we
                        # can't merge them.
                        clients = (
                            fgraph.clients[pairs[0][0]] + fgraph.clients[pairs[0][1]]
                        )
                        if (
                            sum(
                                [
                                    i in flatten(c.op.destroy_map.values())
                                    for c, i in clients
                                    if c != "output" and c.op.destroy_map
                                ]
                            )
                            > 1
                        ):
                            continue

                if len(pairs) == 1 and pairs[0][0].type != pairs[0][1].type:
                    res = pairs[0][0].type.convert_variable(pairs[0][1])

                    # Since the fgraph.replace only checks the convert_variable
                    # in one way, we change the order in the case that
                    # convert_variable will not be successful.
                    if not res:
                        pairs = [(pairs[0][1], pairs[0][0])]

                try:
                    # If all Constants, no need to call validate.
                    # Only need to check one of the var of each pairs.
                    # If it is a Constant, the other must also be a Constant as we merge them.
                    if all([isinstance(old, Constant) for old, new in pairs]):
                        fgraph.replace_all(pairs, reason="MergeOptimizer")
                    else:
                        fgraph.replace_all_validate(pairs, reason="MergeOptimizer")
                except InconsistencyError:
                    success = False
                    nb_fail += 1
                    fgraph.merge_feature.blacklist.append(
                        (pairs[0][0].owner, pairs[0][1].owner)
                    )

                if success:
                    nb_merged += len(pairs)
                    if isinstance(pairs[0][0], Constant):
                        nb_constant += 1
                        # print pairs, pairs[0][0].type
                    break

        if fgraph.profile:
            validate_time = fgraph.profile.validate_time - validate_before
            callback_time = fgraph.execute_callbacks_time - callback_before
            callbacks_time = {}
            for k, v in fgraph.execute_callbacks_times.items():
                if k in callbacks_before:
                    t = v - callbacks_before[k]
                    if t > 0:
                        callbacks_time[k] = t
                else:
                    callbacks_time[k] = v
        else:
            validate_time = None
            callback_time = None
            callbacks_time = {}
        # clear blacklist
        fgraph.merge_feature.blacklist = []
        return (
            nb_fail,
            time.time() - t0,
            validate_time,
            callback_time,
            callbacks_time,
            nb_merged,
            nb_constant,
        )

    def __str__(self):
        return self.__class__.__name__

    @staticmethod
    def print_profile(stream, prof, level=0):

        (
            nb_fail,
            replace_time,
            validate_time,
            callback_time,
            callbacks_time,
            nb_merged,
            nb_constant,
        ) = prof

        blanc = "    " * level
        print(blanc, "MergeOptimizer", file=stream)
        print(
            blanc,
            f"  nb fail={nb_fail:5d} merged={nb_merged:5d} constant={nb_constant:5d}",
            file=stream,
        )
        print(
            blanc,
            f"  time replace={replace_time:2.2f} validate={validate_time:2.2f} callback={callback_time:2.2f}",
            file=stream,
        )
        if callback_time > 1:
            print(blanc, "  callbacks_time", file=stream)
            for i in sorted(callbacks_time.items(), key=lambda a: a[1]):
                if i[1] > 0:
                    # We want to have the __str__ called, so we can't
                    # just print i.
                    print(blanc, "      ", i[0], ",", i[1], file=stream)

    @staticmethod
    def merge_profile(prof1, prof2):
        def merge_none_number(v1, v2):
            if v1 is None:
                return v2
            if v2 is None:
                return v1
            return v1 + v2

        nb_fail = prof1[0] + prof2[0]
        replace_time = prof1[1] + prof2[1]
        validate_time = merge_none_number(prof1[2], prof2[2])
        callback_time = merge_none_number(prof1[3], prof2[3])
        callbacks_time = merge_dict(prof1[4], prof2[4])
        nb_merged = prof1[5] + prof2[5]
        nb_constant = prof1[6] + prof2[6]
        return (
            nb_fail,
            replace_time,
            validate_time,
            callback_time,
            callbacks_time,
            nb_merged,
            nb_constant,
        )


def pre_constant_merge(fgraph, variables):
    """Merge constants in the graphs for a list of `variables`.

    XXX: This changes the nodes in a graph in-place!

    `variables` is a list of nodes, and we want to merge together nodes that
    are constant inputs used to compute nodes in that list.

    We also want to avoid terms in the graphs for `variables` that are
    contained in the `FunctionGraph` given by `fgraph`.  The reason for that:
    it will break consistency of `fgraph` and its features
    (e.g. `ShapeFeature`).

    Notes
    -----
    It is used to pre-merge nodes generated inside an optimization.  It is
    useful if there are many such replacements to make, so that `DebugMode`
    will not check each of them.

    """
    seen_var = set()
    # signature -> variable (for constants)
    const_sig_inv = {}
    if isinstance(variables, Variable):
        variables = [variables]

    def recursive_merge(var):

        if var in seen_var:
            return var

        if not hasattr(var, "owner"):
            return var

        # We don't want to merge constants that are *within* the
        # `FunctionGraph`
        if var.owner in fgraph.apply_nodes:
            return var

        seen_var.add(var)

        if isinstance(var, Constant):
            sig = var.signature()

            if sig in const_sig_inv:
                return const_sig_inv[sig]

            const_sig_inv[sig] = var

            return var

        if var.owner:
            for idx, inp in enumerate(var.owner.inputs):
                # XXX: This is changing the graph in place!
                var.owner.inputs[idx] = recursive_merge(inp)
        return var

    return [recursive_merge(v) for v in variables]


class LocalOptimizer(abc.ABC):
    """A node-based optimizer."""

    def __hash__(self):
        if not hasattr(self, "_optimizer_idx"):
            self._optimizer_idx = _optimizer_idx[0]
            _optimizer_idx[0] += 1
        return self._optimizer_idx

    def tracks(self):
        """
        Return the list of op classes that this opt applies to.

        Return None to apply to all nodes.

        """
        return None

    @abc.abstractmethod
    def transform(self, fgraph, node, *args, **kwargs):
        """
        Transform a subgraph whose output is `node`.

        Subclasses should implement this function so that it returns one of two
        kinds of things:

        - False to indicate that no optimization can be applied to this `node`;
          or
        - <list of variables> to use in place of `node`'s outputs in the
          greater graph.
        - dict(old variables -> new variables). A dictionary that map
          from old variables to new variables to replace.

        Parameters
        ----------
        node : an Apply instance

        """

        raise NotImplementedError()

    def add_requirements(self, fgraph):
        """
        If this local optimization wants to add some requirements to the
        fgraph, this is the place to do it.

        """

    def print_summary(self, stream=sys.stdout, level=0, depth=-1):
        print(f"{' ' * level}{self.__class__.__name__} id={id(self)}", file=stream)

    def __str__(self):
        if hasattr(self, "name"):
            return f"{type(self).__name__}[{self.name}]"
        return repr(self)


class LocalMetaOptimizer(LocalOptimizer):
    """
    Base class for meta-optimizers that try a set of LocalOptimizers
    to replace a node and choose the one that executes the fastest.

    If the error LocalMetaOptimizerSkipAssertionError is raised during
    compilation, we will skip that function compilation and not print
    the error.

    """

    def __init__(self):
        self.verbose = config.metaopt__verbose
        self.track_dict = defaultdict(lambda: [])
        self.tag_dict = defaultdict(lambda: [])
        self._tracks = []
        self.optimizers = []

    def register(self, optimizer, tag_list):
        self.optimizers.append(optimizer)
        for c in optimizer.tracks():
            self.track_dict[c].append(optimizer)
            self._tracks.append(c)
        for tag in tag_list:
            self.tag_dict[tag].append(optimizer)

    def tracks(self):
        return self._tracks

    def transform(self, fgraph, node, *args, **kwargs):
        # safety check: depending on registration, tracks may have been ignored
        if self._tracks is not None:
            if not isinstance(node.op, tuple(self._tracks)):
                return
        # first, we need to provide dummy values for all inputs
        # to the node that are not shared variables anyway
        givens = {}
        missing = set()
        for input in node.inputs:
            if isinstance(input, aesara.compile.SharedVariable):
                pass
            elif hasattr(input.tag, "test_value"):
                givens[input] = aesara.shared(
                    input.type.filter(input.tag.test_value),
                    input.name,
                    broadcastable=input.broadcastable,
                    borrow=True,
                )
            else:
                missing.add(input)
        if missing:
            givens.update(self.provide_inputs(node, missing))
            missing.difference_update(givens.keys())
        # ensure we have data for all input variables that need it
        if missing:
            if self.verbose > 0:
                print(
                    f"{self.__class__.__name__} cannot meta-optimize {node}, "
                    f"{len(missing)} of {int(node.nin)} input shapes unknown"
                )
            return
        # now we can apply the different optimizations in turn,
        # compile the resulting subgraphs and time their execution
        if self.verbose > 1:
            print(
                f"{self.__class__.__name__} meta-optimizing {node} ({len(self.get_opts(node))} choices):"
            )
        timings = []
        for opt in self.get_opts(node):
            outputs = opt.transform(fgraph, node, *args, **kwargs)
            if outputs:
                try:
                    fn = aesara.function(
                        [], outputs, givens=givens, on_unused_input="ignore"
                    )
                    fn.trust_input = True
                    timing = min(self.time_call(fn) for _ in range(2))
                except LocalMetaOptimizerSkipAssertionError:
                    continue
                except Exception as e:
                    if self.verbose > 0:
                        print(f"* {opt}: exception", e)
                    continue
                else:
                    if self.verbose > 1:
                        print(f"* {opt}: {timing:.5g} sec")
                    timings.append((timing, outputs, opt))
            else:
                if self.verbose > 0:
                    print(f"* {opt}: not applicable")
        # finally, we choose the fastest one
        if timings:
            timings.sort()
            if self.verbose > 1:
                print(f"= {timings[0][2]}")
            return timings[0][1]
        return

    def provide_inputs(self, node, inputs):
        """
        If implemented, returns a dictionary mapping all symbolic variables
        in ``inputs`` to SharedVariable instances of suitable dummy values.
        The ``node`` can be inspected to infer required input shapes.

        """
        raise NotImplementedError()

    def get_opts(self, node):
        """
        Can be overrided to change the way opts are selected
        """
        return self.track_dict[type(node.op)]

    def time_call(self, fn):
        start = time.time()
        fn()
        return time.time() - start


class FromFunctionLocalOptimizer(LocalOptimizer):
    """An optimizer constructed from a given function."""

    def __init__(self, fn, tracks=None, requirements=()):
        self.fn = fn
        self._tracks = tracks
        self.requirements = requirements

    def transform(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def add_requirements(self, fgraph):
        for req in self.requirements:
            req(fgraph)

    def tracks(self):
        return self._tracks

    def __str__(self):
        return getattr(self, "__name__", "<FromFunctionLocalOptimizer instance>")

    def print_summary(self, stream=sys.stdout, level=0, depth=-1):
        print(f"{' ' * level}{self.transform} id={id(self)}", file=stream)


def local_optimizer(tracks, inplace=False, requirements=()):
    def decorator(f):
        """
        WRITEME

        """
        if tracks is not None:
            if len(tracks) == 0:
                raise ValueError(
                    "Use None instead of an empty list to apply to all nodes.",
                    f.__module__,
                    f.__name__,
                )
            for t in tracks:
                if not (isinstance(t, Op) or issubclass(t, Op)):
                    raise ValueError(
                        "Tracks are op classes or instances", f.__module__, f.__name__
                    )
        req = requirements
        if inplace:
            dh_handler = dh.DestroyHandler
            req = tuple(requirements) + (
                lambda fgraph: fgraph.attach_feature(dh_handler()),
            )
        rval = FromFunctionLocalOptimizer(f, tracks, req)
        rval.__name__ = f.__name__
        return rval

    return decorator


class LocalOptGroup(LocalOptimizer):
    """Takes a list of LocalOptimizer and applies them to the node.

    This is where the "tracks" parameters are largely used.  If you set
    one of the `LocalOptimizer` in `LocalOptGroup.optimizers` to track a
    specific `Op` instance, this optimizer will only apply said
    `LocalOptimizer` when it's acting on a node that exactly matches the object
    object tracked `Op` (the matching is performed using a `dict` lookup).

    TODO: Use type-based matching (e.g. like `singledispatch`).

    Parameters
    ----------
    optimizers :
        The List of optimizers to be applied to a node
    reentrant : bool (Default True)
        Keyword only argument. Reentrant information. Some global
        optimizer like NavigatorOptimizer can use this value to
        determine if it ignore new nodes during a pass on the
        nodes. Sometimes, ignore_newtrees is not reentrant.
    apply_all_opts : bool (Default False)
        If False, it will return after the new node after the first optimizer
        applied. Otherwise, it will start again with the new node until no new
        optimization apply.

    """

    def __init__(self, *optimizers, apply_all_opts=False, profile=False, name=None):
        self.name = name

        if len(optimizers) == 1 and isinstance(optimizers[0], list):
            optimizers = tuple(optimizers[0])

        self.opts = optimizers
        assert isinstance(self.opts, tuple)

        self.reentrant = any(getattr(opt, "reentrant", True) for opt in optimizers)
        self.retains_inputs = all(
            getattr(opt, "retains_inputs", False) for opt in optimizers
        )

        self.apply_all_opts = apply_all_opts
        self.profile = profile
        self.track_map = defaultdict(list)

        if self.profile:
            self.time_opts = {}
            self.process_count = {}
            self.applied_true = {}
            self.node_created = {}

        for o in self.opts:
            if self.profile:
                self.time_opts.setdefault(o, 0)
                self.process_count.setdefault(o, 0)
                self.applied_true.setdefault(o, 0)
                self.node_created.setdefault(o, 0)
            tracks = o.tracks()
            if tracks is None:
                self.track_map[None].append(o)
            else:
                for c in tracks:
                    self.track_map[c].append(o)

    def __repr__(self):
        return f"LocalOptGroup([{', '.join([str(o) for o in self.opts])}])"

    def tracks(self):
        t = []
        for l in self.opts:
            aet = l.tracks()
            if aet:
                t.extend(aet)
        return t

    def transform(self, fgraph, node):
        if len(self.opts) == 0:
            return
        repl = None
        while True:
            opts = (
                self.track_map[type(node.op)]
                + self.track_map[node.op]
                + self.track_map[None]
            )
            new_repl = None
            for opt in opts:
                opt_start = time.time()
                new_repl = opt.transform(fgraph, node)
                opt_finish = time.time()
                if self.profile:
                    self.time_opts[opt] += opt_start - opt_finish
                    self.process_count[opt] += 1
                if not new_repl:
                    continue
                if isinstance(new_repl, (tuple, list)):
                    new_vars = new_repl
                else:  # It must be a dict
                    new_vars = list(new_repl.values())
                if self.profile:
                    self.node_created[opt] += len(
                        list(applys_between(fgraph.variables, new_vars))
                    )
                    self.applied_true[opt] += 1
                break  # break from the for loop over optimization.
            if not new_repl:  # No optimization applied in the last iteration
                return repl
            # only 1 iteration
            if not self.apply_all_opts:
                return new_repl
            if not new_vars[0].owner:
                # We are at the start of the graph.
                return new_repl
            if len(new_repl) > 1:
                s = {v.owner for v in new_repl}
                assert len(s) == 1
            repl = new_repl
            node = new_vars[0].owner

    @staticmethod
    def print_profile(stream, prof, level=0):
        (time_opts, process_count, applied_true, node_created, profile) = prof

        if not profile:
            return

        blanc = "    " * int(level)
        print(blanc, "LocalOptGroup", file=stream)
        print(blanc, "---------------------", file=stream)
        count_opt = []
        not_used = []
        not_used_time = 0
        for o, count in process_count.items():
            if count > 0:
                count_opt.append(
                    (time_opts[o], applied_true[o], count, o, node_created[o])
                )
            else:
                not_used.append((time_opts[o], o))
                not_used_time += time_opts[o]
        if count_opt:
            print(
                blanc,
                "  time taken - times applied - times tried - name - node_created:",
                file=stream,
            )
            count_opt.sort()
            for (t, a_t, count, o, n_c) in count_opt[::-1]:
                print(
                    blanc,
                    f"  {t:.3f}s - {int(a_t)} - {int(count)} - {o} - {int(n_c)}",
                    file=stream,
                )
            print(
                blanc,
                f"  {not_used_time:.3f}s - in {len(not_used)} optimization that were not used (display those with runtime greater than 0)",
                file=stream,
            )
            not_used.sort(key=lambda nu: (nu[0], str(nu[1])))
            for (t, o) in not_used[::-1]:
                if t > 0:
                    # Skip opt that have 0 times, they probably wasn't even tried.
                    print(blanc + "  ", f"  {t:.3f}s - {o}", file=stream)
        else:
            print(blanc, " The optimizer wasn't successful ", file=stream)

        print(file=stream)

    def merge_profile(prof1, prof2):
        raise NotImplementedError

    def print_summary(self, stream=sys.stdout, level=0, depth=-1):
        print(f"{' ' * level}{self.__class__.__name__} id={id(self)}", file=stream)
        if depth != 0:
            depth -= 1
            for lopt in self.opts:
                lopt.print_summary(stream, level=(level + 2), depth=depth)

    def add_requirements(self, fgraph):
        for opt in self.opts:
            opt.add_requirements(fgraph)


class GraphToGPULocalOptGroup(LocalOptGroup):
    """This is the equivalent of LocalOptGroup for GraphToGPU.

    The main different is the function signature of the local
    optimizer that use the GraphToGPU signature and not the normal
    LocalOptimizer signature.

    apply_all_opts=True is not supported

    """

    def __init__(self, *optimizers, **kwargs):
        super().__init__(*optimizers, **kwargs)
        assert self.apply_all_opts is False

    def transform(self, fgraph, op, context_name, inputs, outputs):
        if len(self.opts) == 0:
            return
        opts = self.track_map[type(op)] + self.track_map[op] + self.track_map[None]
        for opt in opts:
            opt_start = time.time()
            new_repl = opt.transform(fgraph, op, context_name, inputs, outputs)
            opt_finish = time.time()
            if self.profile:
                self.time_opts[opt] += opt_start - opt_finish
                self.process_count[opt] += 1
            if not new_repl:
                continue
            if self.profile:
                self.node_created[opt] += len(
                    list(applys_between(fgraph.variables, new_repl))
                )
                self.applied_true[opt] += 1

            return new_repl


class OpSub(LocalOptimizer):
    """

    Replaces the application of a certain op by the application of
    another op that takes the same inputs as what they are replacing.

    Parameters
    ----------
    op1, op2
        op1.make_node and op2.make_node must take the same number of
        inputs and have the same number of outputs.

    Examples
    --------
    OpSub(add, sub) ==>
        add(div(x, y), add(y, x)) -> sub(div(x, y), sub(y, x))

    """

    # an OpSub does not apply to the nodes it produces
    reentrant = False
    # all the inputs of the original node are transferred to the outputs
    retains_inputs = True

    def __init__(self, op1, op2, transfer_tags=True):
        self.op1 = op1
        self.op2 = op2
        self.transfer_tags = transfer_tags

    def op_key(self):
        return self.op1

    def tracks(self):
        return [self.op1]

    def transform(self, fgraph, node):
        if node.op != self.op1:
            return False
        repl = self.op2.make_node(*node.inputs)
        if self.transfer_tags:
            repl.tag = copy.copy(node.tag)
            for output, new_output in zip(node.outputs, repl.outputs):
                new_output.tag = copy.copy(output.tag)
        return repl.outputs

    def __str__(self):
        return f"{self.op1} -> {self.op2}"


class OpRemove(LocalOptimizer):
    """

    Removes all applications of an op by transferring each of its
    outputs to the corresponding input.

    """

    reentrant = False  # no nodes are added at all

    def __init__(self, op):
        self.op = op

    def op_key(self):
        return self.op

    def tracks(self):
        return [self.op]

    def transform(self, fgraph, node):
        if node.op != self.op:
            return False
        return node.inputs

    def __str__(self):
        return f"{self.op}(x) -> x"

    def print_summary(self, stream=sys.stdout, level=0, depth=-1):
        print(
            f"{' ' * level}{self.__class__.__name__}(self.op) id={id(self)}",
            file=stream,
        )


class PatternSub(LocalOptimizer):
    """

    @todo update

    Replaces all occurrences of the input pattern by the output pattern:

    input_pattern ::= (op, <sub_pattern1>, <sub_pattern2>, ...)
    input_pattern ::= dict(pattern = <input_pattern>,
                            constraint = <constraint>)
    sub_pattern ::= input_pattern
    sub_pattern ::= string
    sub_pattern ::= a Constant instance
    sub_pattern ::= int
    sub_pattern ::= float
    constraint ::= lambda fgraph, expr: additional matching condition

    output_pattern ::= (op, <output_pattern1>, <output_pattern2>, ...)
    output_pattern ::= string
    output_pattern ::= int
    output_pattern ::= float

    Each string in the input pattern is a variable that will be set to
    whatever expression is found in its place. If the same string is
    used more than once, the same expression must be found in those
    places. If a string used in the input pattern is used in the
    output pattern, the matching expression will be inserted in its
    place. The input pattern cannot just be a string but the output
    pattern can.

    If you put a constant variable in the input pattern, there will be a
    match iff a constant variable with the same value and the same type
    is found in its place.

    You can add a constraint to the match by using the dict(...)  form
    described above with a 'constraint' key. The constraint must be a
    function that takes the fgraph and the current Variable that we are
    trying to match and returns True or False according to an
    arbitrary criterion.

    The constructor creates a PatternSub that replaces occurrences of
    in_pattern by occurrences of out_pattern.

    Parameters
    ----------
    in_pattern
        The input pattern that we want to replace.
    out_pattern
        The replacement pattern.
    allow_multiple_clients : bool
        If False, the pattern matching will fail if one of the subpatterns has
        more than one client.
    skip_identities_fn : TODO
    name
        Allows to override this optimizer name.
    pdb : bool
        If True, we invoke pdb when the first node in the pattern matches.
    tracks : optional
        The values that self.tracks() will return. Useful to speed up
        optimization sometimes.
    get_nodes : optional
        If you provide `tracks`, you must provide this parameter. It must be a
        function that takes the tracked node and returns a list of nodes on
        which we will try this optimizer.

    Notes
    -----
    `tracks` and `get_nodes` can be used to make this optimizer track a less
    frequent Op, so this will make this optimizer tried less frequently.

    Examples
    --------
    PatternSub((add, 'x', 'y'), (add, 'y', 'x'))
    PatternSub((multiply, 'x', 'x'), (square, 'x'))
    PatternSub((subtract, (add, 'x', 'y'), 'y'), 'x')
    PatternSub((power, 'x', Constant(double, 2.0)), (square, 'x'))
    PatternSub((boggle, {'pattern': 'x',
                         'constraint': lambda expr: expr.type == scrabble}),
               (scrabble, 'x'))
    """

    def __init__(
        self,
        in_pattern,
        out_pattern,
        allow_multiple_clients=False,
        skip_identities_fn=None,
        name=None,
        pdb=False,
        tracks=(),
        get_nodes=None,
        values_eq_approx=None,
    ):
        self.in_pattern = in_pattern
        self.out_pattern = out_pattern
        self.values_eq_approx = values_eq_approx
        if isinstance(in_pattern, (list, tuple)):
            self.op = self.in_pattern[0]
        elif isinstance(in_pattern, dict):
            self.op = self.in_pattern["pattern"][0]
        else:
            raise TypeError(
                "The pattern to search for must start with " "a specific Op instance."
            )
        self.__doc__ = (
            self.__class__.__doc__ + "\n\nThis instance does: " + str(self) + "\n"
        )
        self.allow_multiple_clients = allow_multiple_clients
        self.skip_identities_fn = skip_identities_fn
        if name:
            self.__name__ = name
        self.pdb = pdb
        self._tracks = tracks
        self.get_nodes = get_nodes
        if tracks != ():
            assert get_nodes

    def op_key(self):
        return self.op

    def tracks(self):
        if self._tracks != ():
            return self._tracks
        return [self.op]

    def transform(self, fgraph, node, get_nodes=True):
        """
        Checks if the graph from node corresponds to in_pattern. If it does,
        constructs out_pattern and performs the replacement.

        """
        from aesara.graph import unify

        if get_nodes and self.get_nodes is not None:
            for real_node in self.get_nodes(fgraph, node):
                if real_node == "output":
                    continue
                ret = self.transform(fgraph, real_node, get_nodes=False)
                if ret is not False and ret is not None:
                    assert len(real_node.outputs) == len(ret)
                    if self.values_eq_approx:
                        ret.tag.values_eq_approx = self.values_eq_approx
                    return dict(zip(real_node.outputs, ret))

        if node.op != self.op:
            return False
        # TODO: if we remove pdb, do this speed things up?

        def match(pattern, expr, u, allow_multiple_clients=False, pdb=False):
            # TODO move outside match
            def retry_with_equiv():
                if not self.skip_identities_fn:
                    return False
                expr_equiv = self.skip_identities_fn(expr)
                if expr_equiv is None:
                    return False
                # TODO: Not sure how to handle multiple_clients flag
                # print 'retrying match', pattern, expr_equiv
                return match(
                    pattern,
                    expr_equiv,
                    u,
                    allow_multiple_clients=allow_multiple_clients,
                )

            if isinstance(pattern, (list, tuple)):
                if expr.owner is None:
                    return False
                if not (expr.owner.op == pattern[0]) or (
                    not allow_multiple_clients and len(fgraph.clients[expr]) > 1
                ):
                    return retry_with_equiv()
                if len(pattern) - 1 != len(expr.owner.inputs):
                    return retry_with_equiv()
                for p, v in zip(pattern[1:], expr.owner.inputs):
                    u = match(p, v, u, self.allow_multiple_clients)
                    if not u:
                        return False
            elif isinstance(pattern, dict):
                try:
                    real_pattern = pattern["pattern"]
                except KeyError:
                    raise KeyError(
                        f"Malformed pattern: {pattern} (expected key 'pattern')"
                    )
                constraint = pattern.get("constraint", lambda expr: True)
                if constraint(expr):
                    return match(
                        real_pattern,
                        expr,
                        u,
                        pattern.get("allow_multiple_clients", allow_multiple_clients),
                    )
                else:
                    return retry_with_equiv()
            elif isinstance(pattern, str):
                v = unify.Var(pattern)
                if u[v] is not v and u[v] is not expr:
                    return retry_with_equiv()
                else:
                    u = u.merge(expr, v)
            elif isinstance(pattern, (int, float)) and isinstance(expr, Constant):
                if np.all(aesara.tensor.constant(pattern).value == expr.value):
                    return u
                else:
                    return retry_with_equiv()
            elif (
                isinstance(pattern, Constant)
                and isinstance(expr, Constant)
                and pattern.equals(expr)
            ):
                return u
            else:
                return retry_with_equiv()
            if pdb:
                import pdb

                pdb.set_trace()
            return u

        u = match(self.in_pattern, node.out, unify.Unification(), True, self.pdb)
        if u:

            def build(pattern, u):
                if isinstance(pattern, (list, tuple)):
                    args = [build(p, u) for p in pattern[1:]]
                    return pattern[0](*args)
                elif isinstance(pattern, str):
                    return u[unify.Var(pattern)]
                elif isinstance(pattern, (int, float)):
                    return pattern
                else:
                    return pattern.clone()

            p = self.out_pattern
            ret = build(p, u)
            if self.values_eq_approx:
                ret.tag.values_eq_approx = self.values_eq_approx
            return [ret]
        else:
            return False

    def __str__(self):
        if getattr(self, "__name__", None):
            return self.__name__

        def pattern_to_str(pattern):
            if isinstance(pattern, (list, tuple)):
                return "{}({})".format(
                    str(pattern[0]),
                    ", ".join([pattern_to_str(p) for p in pattern[1:]]),
                )
            elif isinstance(pattern, dict):
                return "{} subject to {}".format(
                    pattern_to_str(pattern["pattern"]),
                    str(pattern.get("constraint", "no conditions")),
                )
            else:
                return str(pattern)

        return "{} -> {}".format(
            pattern_to_str(self.in_pattern),
            pattern_to_str(self.out_pattern),
        )

    def __repr__(self):
        return str(self)

    def print_summary(self, stream=sys.stdout, level=0, depth=-1):
        name = getattr(self, "__name__", getattr(self, "name", None))
        print(
            f"{' ' * level}{self.__class__.__name__} {name}({self.in_pattern}, {self.out_pattern}) id={id(self)}",
            file=stream,
        )


class Updater(Feature):
    def __init__(self, importer, pruner, chin, name=None):
        self.importer = importer
        self.pruner = pruner
        self.chin = chin
        self.name = name

    def __str__(self):
        return "Updater{%s}" % str(self.name)

    def on_import(self, fgraph, node, reason):
        if self.importer:
            self.importer(node)

    def on_prune(self, fgraph, node, reason):
        if self.pruner:
            self.pruner(node)

    def on_change_input(self, fgraph, node, i, r, new_r, reason):
        if self.chin:
            self.chin(node, i, r, new_r, reason)

    def on_detach(self, fgraph):
        # To allow pickling this object
        self.importer = None
        self.pruner = None
        self.chin = None


class NavigatorOptimizer(GlobalOptimizer):
    """
    Abstract class.

    Parameters
    ----------
    local_opt
        A LocalOptimizer to apply over a FunctionGraph (or None is Ok too).
    ignore_newtrees
        - True: new subgraphs returned by an optimization is not a
          candidate for optimization.
        - False: new subgraphs returned by an optimization is a candidate
          for optimization.
        - 'auto': let the local_opt set this parameter via its 'reentrant'
          attribute.
    failure_callback
            A function that takes (exception, navigator, [(old, new),
            (old,new),...]) and we call it if there's an exception.

            If the trouble is from local_opt.transform(), the new variables
            will be 'None'.

            If the trouble is from validation (the new types don't match for
            example) then the new variables will be the ones created by
            transform().

            If this parameter is None, then exceptions are not caught here
            (raised normally).

    """

    @staticmethod
    def warn(exc, nav, repl_pairs, local_opt, node):
        """
        Failure_callback for NavigatorOptimizer: print traceback.

        """
        if config.on_opt_error != "ignore":
            _logger.error(f"Optimization failure due to: {local_opt}")
            _logger.error(f"node: {node}")
            _logger.error("TRACEBACK:")
            _logger.error(traceback.format_exc())
        if config.on_opt_error == "pdb":
            pdb.post_mortem(sys.exc_info()[2])
        elif isinstance(exc, AssertionError) or config.on_opt_error == "raise":
            # We always crash on AssertionError because something may be
            # seriously wrong if such an exception is raised.
            raise exc

    @staticmethod
    def warn_inplace(exc, nav, repl_pairs, local_opt, node):
        """
        Failure_callback for NavigatorOptimizer.

        Ignore InconsistencyErrors, print traceback.

        If error during replacement repl_pairs is set. Otherwise None.

        """
        if isinstance(exc, InconsistencyError):
            return
        return NavigatorOptimizer.warn(exc, nav, repl_pairs, local_opt, node)

    @staticmethod
    def warn_ignore(exc, nav, repl_pairs, local_opt, node):
        """
        Failure_callback for NavigatorOptimizer: ignore all errors.

        """

    def __init__(self, local_opt, ignore_newtrees="auto", failure_callback=None):
        self.local_opt = local_opt
        if ignore_newtrees == "auto":
            self.ignore_newtrees = not getattr(local_opt, "reentrant", True)
        else:
            self.ignore_newtrees = ignore_newtrees
        self.failure_callback = failure_callback

    def attach_updater(self, fgraph, importer, pruner, chin=None, name=None):
        """
        Install some FunctionGraph listeners to help the navigator deal with
        the ignore_trees-related functionality.

        Parameters
        ----------
        importer
            Function that will be called whenever optimizations add stuff
            to the graph.
        pruner
            Function to be called when optimizations remove stuff
            from the graph.
        chin
            "on change input" called whenever a node's inputs change.
        name
            name of the Updater to attach.

        Returns
        -------
        object
            The FunctionGraph plugin that handles the three tasks.
            Keep this around so that you can detach later!

        """
        if self.ignore_newtrees:
            importer = None

        if importer is None and pruner is None:
            return None

        u = Updater(importer, pruner, chin, name=name)
        fgraph.attach_feature(u)
        return u

    def detach_updater(self, fgraph, u):
        """
        Undo the work of attach_updater.

        Parameters
        ----------
        u
            A return-value of attach_updater.

        Returns
        -------
        None

        """
        if u is not None:
            fgraph.remove_feature(u)

    def process_node(self, fgraph, node, lopt=None):
        """
        This function will use `lopt` to `transform` the `node`. The
        `transform` method will return either False or a list of Variables
        that are intended to replace `node.outputs`.

        If the fgraph accepts the replacement, then the optimization is
        successful, and this function returns True.

        If there are no replacement candidates or the fgraph rejects the
        replacements, this function returns False.

        Parameters
        ----------
        fgraph
            A FunctionGraph.
        node
            An Apply instance in `fgraph`
        lopt
            A LocalOptimizer instance that may have a better idea for
            how to compute node's outputs.

        Returns
        -------
        bool
            True iff the `node`'s outputs were replaced in the `fgraph`.

        """
        lopt = lopt or self.local_opt
        try:
            replacements = lopt.transform(fgraph, node)
        except Exception as e:
            if self.failure_callback is not None:
                self.failure_callback(
                    e, self, [(x, None) for x in node.outputs], lopt, node
                )
                return False
            else:
                raise
        if replacements is False or replacements is None:
            return False
        old_vars = node.outputs
        remove = []
        if isinstance(replacements, dict):
            if "remove" in replacements:
                remove = replacements.pop("remove")
            old_vars = list(replacements.keys())
            replacements = list(replacements.values())
        elif not isinstance(replacements, (tuple, list)):
            raise TypeError(
                f"Optimizer {lopt} gave wrong type of replacement. "
                f"Expected list or tuple. Got {replacements}"
            )
        if len(old_vars) != len(replacements):
            raise ValueError(f"Optimizer {lopt} gave wrong number of replacements")
        # None in the replacement mean that this variable isn't used
        # and we want to remove it
        for r, rnew in zip(old_vars, replacements):
            if rnew is None and len(fgraph.clients[r]) > 0:
                raise ValueError(
                    "A local optimizer tried to remove a Variable that is used"
                )
        # If an output would be replaced by itself, no need to perform
        # the replacement
        repl_pairs = [
            (r, rnew)
            for r, rnew in zip(old_vars, replacements)
            if rnew is not r and rnew is not None
        ]

        if len(repl_pairs) == 0:
            return False
        try:
            fgraph.replace_all_validate_remove(repl_pairs, reason=lopt, remove=remove)
            return True
        except Exception as e:
            # This means the replacements were rejected by the fgraph.
            #
            # This is not supposed to happen.  The default failure_callback
            # will print a traceback as a warning.
            if self.failure_callback is not None:
                self.failure_callback(e, self, repl_pairs, lopt, node)
                return False
            else:
                raise

    def add_requirements(self, fgraph):
        super().add_requirements(fgraph)
        # Added by default
        # fgraph.attach_feature(ReplaceValidate())
        if self.local_opt:
            self.local_opt.add_requirements(fgraph)

    def print_summary(self, stream=sys.stdout, level=0, depth=-1):
        print(f"{' ' * level}{self.__class__.__name__} id={id(self)}", file=stream)
        if depth != 0:
            self.local_opt.print_summary(stream, level=(level + 2), depth=(depth - 1))


class TopoOptimizer(NavigatorOptimizer):
    """
    TopoOptimizer has one local optimizer. It tries to apply to each node, in topological order (or reverse).
    Each time the local optimizer applies, the node gets replaced, and the topooptimizer moves on to the next one.

    """

    def __init__(
        self, local_opt, order="in_to_out", ignore_newtrees=False, failure_callback=None
    ):
        if order not in ["out_to_in", "in_to_out"]:
            raise ValueError("order must be 'out_to_in' or 'in_to_out'")
        self.order = order
        super().__init__(local_opt, ignore_newtrees, failure_callback)

    def apply(self, fgraph, start_from=None):
        if start_from is None:
            start_from = fgraph.outputs
        callback_before = fgraph.execute_callbacks_time
        nb_nodes_start = len(fgraph.apply_nodes)
        t0 = time.time()
        q = deque(io_toposort(fgraph.inputs, start_from))
        io_t = time.time() - t0

        def importer(node):
            if node is not current_node:
                q.append(node)

        u = self.attach_updater(
            fgraph, importer, None, name=getattr(self, "name", None)
        )
        nb = 0
        try:
            t0 = time.time()
            while q:
                if self.order == "out_to_in":
                    node = q.pop()
                else:
                    node = q.popleft()
                if node not in fgraph.apply_nodes:
                    continue
                current_node = node
                nb += self.process_node(fgraph, node)
            loop_t = time.time() - t0
        finally:
            self.detach_updater(fgraph, u)

        callback_time = fgraph.execute_callbacks_time - callback_before
        nb_nodes_end = len(fgraph.apply_nodes)
        return (
            self,
            nb,
            nb_nodes_start,
            nb_nodes_end,
            io_t,
            loop_t,
            callback_time,
            self.local_opt,
        )

    @staticmethod
    def print_profile(stream, prof, level=0):
        blanc = "    " * level
        if prof is None:  # Happen as merge_profile() isn't implemented
            print(blanc, "TopoOptimizer merge_profile not implemented", file=stream)
            return

        (
            opt,
            nb,
            nb_nodes_start,
            nb_nodes_end,
            io_t,
            loop_t,
            callback_time,
            lopt,
        ) = prof

        print(
            blanc,
            "TopoOptimizer ",
            getattr(opt, "name", getattr(opt, "__name__", "")),
            file=stream,
        )

        print(
            blanc,
            "  nb_node (start, end, changed)",
            (nb_nodes_start, nb_nodes_end, nb),
            file=stream,
        )
        print(blanc, "  init io_toposort", io_t, file=stream)
        print(blanc, "  loop time", loop_t, file=stream)
        print(blanc, "  callback_time", callback_time, file=stream)
        if isinstance(lopt, LocalOptGroup):
            if lopt.profile:
                lopt.print_profile(
                    stream,
                    (
                        lopt.time_opts,
                        lopt.process_count,
                        lopt.applied_true,
                        lopt.node_created,
                        lopt.profile,
                    ),
                    level=level + 1,
                )


def out2in(*local_opts, **kwargs):
    """
    Uses the TopoOptimizer from the output nodes to input nodes of the graph.
    """
    name = kwargs and kwargs.pop("name", None)
    if len(local_opts) > 1:
        # Don't wrap it uselessly if their is only 1 optimization.
        local_opts = LocalOptGroup(*local_opts)
    else:
        (local_opts,) = local_opts
        if not name:
            name = local_opts.__name__
    ret = TopoOptimizer(
        local_opts,
        order="out_to_in",
        failure_callback=TopoOptimizer.warn_inplace,
        **kwargs,
    )
    if name:
        ret.__name__ = name
    return ret


def in2out(*local_opts, **kwargs):
    """
    Uses the TopoOptimizer from the input nodes to output nodes of the graph.
    """
    name = kwargs and kwargs.pop("name", None)
    if len(local_opts) > 1:
        # Don't wrap it uselessly if their is only 1 optimization.
        local_opts = LocalOptGroup(*local_opts)
    else:
        (local_opts,) = local_opts
        if not name:
            name = local_opts.__name__
    ret = TopoOptimizer(
        local_opts,
        order="in_to_out",
        failure_callback=TopoOptimizer.warn_inplace,
        **kwargs,
    )
    if name:
        ret.__name__ = name
    return ret


class OpKeyOptimizer(NavigatorOptimizer):
    """
    WRITEME

    """

    def __init__(self, local_opt, ignore_newtrees=False, failure_callback=None):
        if not hasattr(local_opt, "op_key"):
            raise TypeError(
                "LocalOptimizer for OpKeyOptimizer must have " "an 'op_key' method."
            )
        super().__init__(local_opt, ignore_newtrees, failure_callback)

    def apply(self, fgraph):
        op = self.local_opt.op_key()
        if isinstance(op, (list, tuple)):
            q = reduce(list.__iadd__, map(fgraph.get_nodes, op))
        else:
            q = list(fgraph.get_nodes(op))

        def importer(node):
            if node is not current_node:
                if node.op == op:
                    q.append(node)

        u = self.attach_updater(
            fgraph, importer, None, name=getattr(self, "name", None)
        )
        try:
            while q:
                node = q.pop()
                if node not in fgraph.apply_nodes:
                    continue
                current_node = node
                self.process_node(fgraph, node)
        finally:
            self.detach_updater(fgraph, u)

    def add_requirements(self, fgraph):
        """
        Requires the following features:
          - NodeFinder
          - ReplaceValidate(Added by default)

        """
        super().add_requirements(fgraph)
        fgraph.attach_feature(NodeFinder())


class ChangeTracker(Feature):
    def __init__(self):
        self.changed = False
        self.nb_imported = 0

    def on_import(self, fgraph, node, reason):
        self.nb_imported += 1
        self.changed = True

    def on_change_input(self, fgraph, node, i, r, new_r, reason):
        self.changed = True

    def reset(self):
        self.changed = False

    def on_attach(self, fgraph):
        fgraph.change_tracker = self

    def on_detach(self, fgraph):
        del fgraph.change_tracker


def merge_dict(d1, d2):
    """
    merge 2 dicts by adding the values.
    """
    d = d1.copy()
    for k, v in d2.items():
        if k in d:
            d[k] += v
        else:
            d[k] = v
    return d


class EquilibriumOptimizer(NavigatorOptimizer):
    """
    Apply optimizations until equilibrium point.

    Parameters
    ----------
    optimizers : list or set
        Local or global optimizations to apply until equilibrium.
        The global optimizer will be run at the start of each iteration before
        the local optimizer.
    max_use_ratio : int or float
        Each optimizer can be applied at most (size of graph * this number)
        times.
    ignore_newtrees
        See EquilibriumDB ignore_newtrees parameter definition.
    final_optimizers
        Global optimizers that will be run after each iteration.
    cleanup_optimizers
        Global optimizers that apply a list of pre determined optimization.
        They must not traverse the graph as they are called very frequently.
        The MergeOptimizer is one example of optimization that respect this.
        They are applied after all global optimizers, then when one local
        optimizer is applied, then after all final optimizers.

    """

    def __init__(
        self,
        optimizers,
        failure_callback=None,
        ignore_newtrees=True,
        tracks_on_change_inputs=False,
        max_use_ratio=None,
        final_optimizers=None,
        cleanup_optimizers=None,
    ):
        super().__init__(
            None, ignore_newtrees=ignore_newtrees, failure_callback=failure_callback
        )
        self.local_optimizers_map = OrderedDict()
        self.local_optimizers_all = []
        self.global_optimizers = []
        self.final_optimizers = []
        self.cleanup_optimizers = []
        self.tracks_on_change_inputs = tracks_on_change_inputs
        for opt in optimizers:
            if isinstance(opt, LocalOptimizer):
                if opt.tracks() is None:
                    self.local_optimizers_all.append(opt)
                else:
                    for c in opt.tracks():
                        self.local_optimizers_map.setdefault(c, []).append(opt)
            else:
                self.global_optimizers.append(opt)
        if final_optimizers:
            self.final_optimizers = final_optimizers
        if cleanup_optimizers:
            self.cleanup_optimizers = cleanup_optimizers
        self.max_use_ratio = max_use_ratio
        assert self.max_use_ratio is not None, "max_use_ratio has to be a number"

    def get_local_optimizers(self):
        for opt in self.local_optimizers_all:
            yield opt
        # if repeat is not a problem we can drop the set
        s = set()
        for lopt in self.local_optimizers_map.values():
            for opt in lopt:
                if opt not in s:
                    yield opt
                    s.add(opt)

    def add_requirements(self, fgraph):
        super().add_requirements(fgraph)
        for opt in self.get_local_optimizers():
            opt.add_requirements(fgraph)
        for opt in self.global_optimizers:
            opt.add_requirements(fgraph)
        for opt in self.final_optimizers:
            opt.add_requirements(fgraph)
        for opt in self.cleanup_optimizers:
            opt.add_requirements(fgraph)

    def apply(self, fgraph, start_from=None):
        change_tracker = ChangeTracker()
        fgraph.attach_feature(change_tracker)
        if start_from is None:
            start_from = fgraph.outputs
        else:
            for node in start_from:
                assert node in fgraph.outputs

        changed = True
        max_use_abort = False
        opt_name = None
        global_process_count = {}
        start_nb_nodes = len(fgraph.apply_nodes)
        max_nb_nodes = len(fgraph.apply_nodes)
        max_use = max_nb_nodes * self.max_use_ratio

        loop_timing = []
        loop_process_count = []
        global_opt_timing = []
        time_opts = {}
        io_toposort_timing = []
        nb_nodes = []
        node_created = {}
        global_sub_profs = []
        final_sub_profs = []
        cleanup_sub_profs = []
        for opt in (
            self.global_optimizers
            + list(self.get_local_optimizers())
            + self.final_optimizers
            + self.cleanup_optimizers
        ):
            global_process_count.setdefault(opt, 0)
            time_opts.setdefault(opt, 0)
            node_created.setdefault(opt, 0)

        def apply_cleanup(profs_dict):
            changed = False
            for copt in self.cleanup_optimizers:
                change_tracker.reset()
                nb = change_tracker.nb_imported
                t_opt = time.time()
                sub_prof = copt.apply(fgraph)
                time_opts[copt] += time.time() - t_opt
                profs_dict[copt].append(sub_prof)
                if change_tracker.changed:
                    process_count.setdefault(copt, 0)
                    process_count[copt] += 1
                    global_process_count[copt] += 1
                    changed = True
                    node_created[copt] += change_tracker.nb_imported - nb
            return changed

        while changed and not max_use_abort:
            process_count = {}
            t0 = time.time()
            changed = False
            iter_cleanup_sub_profs = {}
            for copt in self.cleanup_optimizers:
                iter_cleanup_sub_profs[copt] = []

            # apply global optimizers
            sub_profs = []
            for gopt in self.global_optimizers:
                change_tracker.reset()
                nb = change_tracker.nb_imported
                t_opt = time.time()
                sub_prof = gopt.apply(fgraph)
                time_opts[gopt] += time.time() - t_opt
                sub_profs.append(sub_prof)
                if change_tracker.changed:
                    process_count.setdefault(gopt, 0)
                    process_count[gopt] += 1
                    global_process_count[gopt] += 1
                    changed = True
                    node_created[gopt] += change_tracker.nb_imported - nb
                    if global_process_count[gopt] > max_use:
                        max_use_abort = True
                        opt_name = getattr(gopt, "name", None) or getattr(
                            gopt, "__name__", ""
                        )
            global_sub_profs.append(sub_profs)

            global_opt_timing.append(float(time.time() - t0))

            # apply clean up as global opt can have done changes that
            # request that
            changed |= apply_cleanup(iter_cleanup_sub_profs)

            # apply local optimizer
            topo_t0 = time.time()
            q = deque(io_toposort(fgraph.inputs, start_from))
            io_toposort_timing.append(time.time() - topo_t0)

            nb_nodes.append(len(q))
            max_nb_nodes = max(max_nb_nodes, len(q))
            max_use = max_nb_nodes * self.max_use_ratio

            def importer(node):
                if node is not current_node:
                    q.append(node)

            chin = None
            if self.tracks_on_change_inputs:

                def chin(node, i, r, new_r, reason):
                    if node is not current_node and not isinstance(node, str):
                        q.append(node)

            u = self.attach_updater(
                fgraph, importer, None, chin=chin, name=getattr(self, "name", None)
            )
            try:
                while q:
                    node = q.pop()
                    if node not in fgraph.apply_nodes:
                        continue
                    current_node = node
                    for lopt in (
                        self.local_optimizers_all
                        + self.local_optimizers_map.get(type(node.op), [])
                        + self.local_optimizers_map.get(node.op, [])
                    ):
                        nb = change_tracker.nb_imported
                        t_opt = time.time()
                        lopt_change = self.process_node(fgraph, node, lopt)
                        time_opts[lopt] += time.time() - t_opt
                        if not lopt_change:
                            continue
                        process_count.setdefault(lopt, 0)
                        process_count[lopt] += 1
                        global_process_count[lopt] += 1
                        changed = True
                        node_created[lopt] += change_tracker.nb_imported - nb
                        changed |= apply_cleanup(iter_cleanup_sub_profs)
                        if global_process_count[lopt] > max_use:
                            max_use_abort = True
                            opt_name = getattr(lopt, "name", None) or getattr(
                                lopt, "__name__", ""
                            )
                        if node not in fgraph.apply_nodes:
                            # go to next node
                            break
            finally:
                self.detach_updater(fgraph, u)

            # Apply final optimizers
            sub_profs = []
            t_before_final_opt = time.time()
            for gopt in self.final_optimizers:
                change_tracker.reset()
                nb = change_tracker.nb_imported
                t_opt = time.time()
                sub_prof = gopt.apply(fgraph)
                time_opts[gopt] += time.time() - t_opt
                sub_profs.append(sub_prof)
                if change_tracker.changed:
                    process_count.setdefault(gopt, 0)
                    process_count[gopt] += 1
                    global_process_count[gopt] += 1
                    changed = True
                    node_created[gopt] += change_tracker.nb_imported - nb
                    if global_process_count[gopt] > max_use:
                        max_use_abort = True
                        opt_name = getattr(gopt, "name", None) or getattr(
                            gopt, "__name__", ""
                        )
            final_sub_profs.append(sub_profs)

            global_opt_timing[-1] += time.time() - t_before_final_opt
            # apply clean up as final opt can have done changes that
            # request that
            changed |= apply_cleanup(iter_cleanup_sub_profs)
            # merge clean up profiles during that iteration.
            c_sub_profs = []
            for copt, sub_profs in iter_cleanup_sub_profs.items():
                sub_prof = sub_profs[0]
                for s_p in sub_profs[1:]:
                    sub_prof = copt.merge_profile(sub_prof, s_p)
                c_sub_profs.append(sub_prof)
            cleanup_sub_profs.append(c_sub_profs)

            loop_process_count.append(process_count)
            loop_timing.append(float(time.time() - t0))

        end_nb_nodes = len(fgraph.apply_nodes)

        if max_use_abort:
            msg = (
                f"EquilibriumOptimizer max'ed out by '{opt_name}'"
                + ". You can safely raise the current threshold of "
                + "{config.optdb__max_use_ratio:f} with the aesara flag 'optdb__max_use_ratio'."
            )
            if config.on_opt_error == "raise":
                raise AssertionError(msg)
            else:
                _logger.error(msg)
        fgraph.remove_feature(change_tracker)
        assert len(loop_process_count) == len(loop_timing)
        assert len(loop_process_count) == len(global_opt_timing)
        assert len(loop_process_count) == len(nb_nodes)
        assert len(loop_process_count) == len(io_toposort_timing)
        assert len(loop_process_count) == len(global_sub_profs)
        assert len(loop_process_count) == len(final_sub_profs)
        assert len(loop_process_count) == len(cleanup_sub_profs)
        return (
            self,
            loop_timing,
            loop_process_count,
            (start_nb_nodes, end_nb_nodes, max_nb_nodes),
            global_opt_timing,
            nb_nodes,
            time_opts,
            io_toposort_timing,
            node_created,
            global_sub_profs,
            final_sub_profs,
            cleanup_sub_profs,
        )

    def print_summary(self, stream=sys.stdout, level=0, depth=-1):
        name = getattr(self, "name", None)
        print(
            f"{' ' * level}{self.__class__.__name__} {name} id={id(self)}", file=stream
        )
        if depth != 0:
            for lopt in self.get_local_optimizers():
                lopt.print_summary(stream, level=(level + 2), depth=(depth - 1))

    @staticmethod
    def print_profile(stream, prof, level=0):
        (
            opt,
            loop_timing,
            loop_process_count,
            (start_nb_nodes, end_nb_nodes, max_nb_nodes),
            global_opt_timing,
            nb_nodes,
            time_opts,
            io_toposort_timing,
            node_created,
            global_sub_profs,
            final_sub_profs,
            cleanup_sub_profs,
        ) = prof

        blanc = "    " * level
        print(blanc, "EquilibriumOptimizer", end=" ", file=stream)
        print(blanc, getattr(opt, "name", getattr(opt, "__name__", "")), file=stream)
        print(
            blanc,
            f"  time {sum(loop_timing):.3f}s for {len(loop_timing)} passes",
            file=stream,
        )
        print(
            blanc,
            f"  nb nodes (start, end,  max) {int(start_nb_nodes)} {int(end_nb_nodes)} {int(max_nb_nodes)}",
            file=stream,
        )
        print(blanc, f"  time io_toposort {sum(io_toposort_timing):.3f}s", file=stream)
        s = sum([time_opts[o] for o in opt.get_local_optimizers()])
        print(blanc, f"  time in local optimizers {s:.3f}s", file=stream)
        s = sum([time_opts[o] for o in opt.global_optimizers])
        print(blanc, f"  time in global optimizers {s:.3f}s", file=stream)
        s = sum([time_opts[o] for o in opt.final_optimizers])
        print(blanc, f"  time in final optimizers {s:.3f}s", file=stream)
        s = sum([time_opts[o] for o in opt.cleanup_optimizers])
        print(blanc, f"  time in cleanup optimizers {s:.3f}s", file=stream)
        for i in range(len(loop_timing)):
            lopt = ""
            if loop_process_count[i]:
                d = list(
                    reversed(sorted(loop_process_count[i].items(), key=lambda a: a[1]))
                )
                lopt = " ".join([str((str(k), v)) for k, v in d[:5]])
                if len(d) > 5:
                    lopt += " ..."
            print(
                blanc,
                (
                    f"  {int(i):2d} - {loop_timing[i]:.3f}s {int(sum(loop_process_count[i].values()))} ({global_opt_timing[i]:.3f}s in global opts, "
                    f"{io_toposort_timing[i]:.3f}s io_toposort) - {int(nb_nodes[i])} nodes - {lopt}"
                ),
                file=stream,
            )

        count_opt = []
        not_used = []
        not_used_time = 0
        process_count = {}
        for o in (
            opt.global_optimizers
            + list(opt.get_local_optimizers())
            + list(opt.final_optimizers)
            + list(opt.cleanup_optimizers)
        ):
            process_count.setdefault(o, 0)
        for count in loop_process_count:
            for o, v in count.items():
                process_count[o] += v
        for o, count in process_count.items():
            if count > 0:
                count_opt.append((time_opts[o], count, node_created[o], o))
            else:
                not_used.append((time_opts[o], o))
                not_used_time += time_opts[o]

        if count_opt:
            print(
                blanc, "  times - times applied - nb node created - name:", file=stream
            )
            count_opt.sort()
            for (t, count, n_created, o) in count_opt[::-1]:
                print(
                    blanc,
                    f"  {t:.3f}s - {int(count)} - {int(n_created)} - {o}",
                    file=stream,
                )
            print(
                blanc,
                f"  {not_used_time:.3f}s - in {len(not_used)} optimization that were not used (display only those with a runtime > 0)",
                file=stream,
            )
            not_used.sort(key=lambda nu: (nu[0], str(nu[1])))
            for (t, o) in not_used[::-1]:
                if t > 0:
                    # Skip opt that have 0 times, they probably wasn't even tried.
                    print(blanc + "  ", f"  {t:.3f}s - {o}", file=stream)
            print(file=stream)
        gf_opts = [
            o
            for o in (
                opt.global_optimizers
                + list(opt.final_optimizers)
                + list(opt.cleanup_optimizers)
            )
            if o.print_profile.__code__ is not GlobalOptimizer.print_profile.__code__
        ]
        if not gf_opts:
            return
        print(blanc, "Global, final and clean up optimizers", file=stream)
        for i in range(len(loop_timing)):
            print(blanc, f"Iter {int(i)}", file=stream)
            for o, prof in zip(opt.global_optimizers, global_sub_profs[i]):
                try:
                    o.print_profile(stream, prof, level + 2)
                except NotImplementedError:
                    print(blanc, "merge not implemented for ", o)
            for o, prof in zip(opt.final_optimizers, final_sub_profs[i]):
                try:
                    o.print_profile(stream, prof, level + 2)
                except NotImplementedError:
                    print(blanc, "merge not implemented for ", o)
            for o, prof in zip(opt.cleanup_optimizers, cleanup_sub_profs[i]):
                try:
                    o.print_profile(stream, prof, level + 2)
                except NotImplementedError:
                    print(blanc, "merge not implemented for ", o)

    @staticmethod
    def merge_profile(prof1, prof2):
        # (opt, loop_timing, loop_process_count, max_nb_nodes,
        # global_opt_timing, nb_nodes, time_opts, io_toposort_timing) = prof1
        local_optimizers = OrderedSet(prof1[0].get_local_optimizers()).union(
            prof2[0].get_local_optimizers()
        )
        global_optimizers = OrderedSet(prof1[0].global_optimizers).union(
            prof2[0].global_optimizers
        )
        final_optimizers = list(
            OrderedSet(prof1[0].final_optimizers).union(prof2[0].final_optimizers)
        )
        cleanup_optimizers = list(
            OrderedSet(prof1[0].cleanup_optimizers).union(prof2[0].cleanup_optimizers)
        )
        new_opt = EquilibriumOptimizer(
            local_optimizers.union(global_optimizers),
            max_use_ratio=1,
            final_optimizers=final_optimizers,
            cleanup_optimizers=cleanup_optimizers,
        )

        def add_append_list(l1, l2):
            l = copy.copy(l1)
            for idx, nb in enumerate(l2):
                if idx < len(l):
                    l[idx] += nb
                else:
                    l.append(nb)
            return l

        loop_timing = add_append_list(prof1[1], prof2[1])

        loop_process_count = list(prof1[2])
        global_sub_profs = []
        final_sub_profs = []
        cleanup_sub_profs = []

        for i in range(min(len(loop_process_count), len(prof2[2]))):
            process_count = loop_process_count[i]
            for process, count in prof2[2][i].items():
                if process in process_count:
                    process_count[process] += count
                else:
                    process_count[process] = count

            def merge(opts, attr, idx):
                tmp = []
                for opt in opts:
                    o1 = getattr(prof1[0], attr)
                    o2 = getattr(prof2[0], attr)
                    if opt in o1 and opt in o2:
                        p1 = prof1[idx][i][o1.index(opt)]
                        p2 = prof2[idx][i][o2.index(opt)]
                        m = None
                        if hasattr(opt, "merge_profile"):
                            m = opt.merge_profile(p1, p2)
                    elif opt in o1:
                        m = prof1[idx][i][o1.index(opt)]
                    else:
                        m = prof2[idx][i][o2.index(opt)]
                    tmp.append(m)
                return tmp

            global_sub_profs.append(merge(global_optimizers, "global_optimizers", 9))
            final_sub_profs.append(merge(final_optimizers, "final_optimizers", 10))
            cleanup_sub_profs.append(
                merge(cleanup_optimizers, "cleanup_optimizers", 11)
            )

        # Add the iteration done by only one of the profile.
        loop_process_count.extend(prof1[2][len(loop_process_count) :])
        global_sub_profs.extend(prof1[9][len(global_sub_profs) :])
        final_sub_profs.extend(prof1[10][len(final_sub_profs) :])
        cleanup_sub_profs.extend(prof1[11][len(cleanup_sub_profs) :])

        global_sub_profs.extend(prof2[9][len(loop_process_count) :])
        final_sub_profs.extend(prof2[10][len(loop_process_count) :])
        cleanup_sub_profs.extend(prof2[11][len(loop_process_count) :])

        max_nb_nodes = max(prof1[3], prof2[3])

        global_opt_timing = add_append_list(prof1[4], prof2[4])

        nb_nodes = add_append_list(prof1[5], prof2[5])

        time_opts = merge_dict(prof1[6], prof2[6])
        io_toposort_timing = add_append_list(prof1[7], prof2[7])
        assert (
            len(loop_timing)
            == len(global_opt_timing)
            == len(global_sub_profs)
            == len(io_toposort_timing)
            == len(nb_nodes)
        )
        assert len(loop_timing) == max(len(prof1[1]), len(prof2[1]))

        node_created = merge_dict(prof1[8], prof2[8])
        return (
            new_opt,
            loop_timing,
            loop_process_count,
            max_nb_nodes,
            global_opt_timing,
            nb_nodes,
            time_opts,
            io_toposort_timing,
            node_created,
            global_sub_profs,
            final_sub_profs,
            cleanup_sub_profs,
        )


def _check_chain(r, chain):
    """
    WRITEME

    """
    chain = list(reversed(chain))
    while chain:
        elem = chain.pop()
        if elem is None:
            if r.owner is not None:
                return False
        elif r.owner is None:
            return False
        elif isinstance(elem, Op):
            if not r.owner.op == elem:
                return False
        else:
            try:
                if issubclass(elem, Op) and not isinstance(r.owner.op, elem):
                    return False
            except TypeError:
                return False
        if chain:
            r = r.owner.inputs[chain.pop()]
    # print 'check_chain', _check_chain.n_calls
    # _check_chain.n_calls += 1

    # The return value will be used as a Boolean, but some Variables cannot
    # be used as Booleans (the results of comparisons, for instance)
    return r is not None


def check_chain(r, *chain):
    """
    WRITEME

    """
    if isinstance(r, Apply):
        r = r.outputs[0]
    return _check_chain(r, reduce(list.__iadd__, ([x, 0] for x in chain)))


def pre_greedy_local_optimizer(fgraph, optimizations, out):
    """Apply local optimizations to a graph.

    This function traverses the computation graph in the graph before the
    variable `out` but that are not in the `fgraph`. It applies
    `local_optimizations` to each variable on the traversed graph.

    XXX: This changes the nodes in a graph in-place!

    Its main use is to apply locally constant folding when generating
    the graph of the indices of a subtensor.

    Changes should not be applied to nodes that are in an `fgraph`,
    so we use `fgraph` to prevent that.

    Notes
    -----
    This doesn't do an equilibrium optimization, so, if there is optimization
    like `local_upcast_elemwise_constant_inputs` in the list that adds
    additional nodes to the inputs of the node, it might be necessary to call
    this function multiple times.

    Parameters
    ----------
    fgraph : FunctionGraph
        The graph used to avoid/filter nodes.
    optimizations : list of LocalOptimizer
        The list of local optimizations to apply
    out : Variable
        A `Variable` specifying the graph to optimize.

    """

    def local_recursive_function(list_opt, out, optimized_vars, depth):
        if not getattr(out, "owner", None):
            return [out], optimized_vars
        node = out.owner

        if node in fgraph.apply_nodes:
            return node.outputs, optimized_vars

        # Walk up the graph via the node's inputs
        for idx, inp in enumerate(node.inputs):
            if inp in optimized_vars:
                nw_in = optimized_vars[inp]
            else:
                if inp.owner:
                    outs, optimized_vars = local_recursive_function(
                        list_opt, inp, optimized_vars, depth + 1
                    )
                    for k, v in zip(inp.owner.outputs, outs):
                        optimized_vars[k] = v
                    nw_in = outs[inp.owner.outputs.index(inp)]

                else:
                    nw_in = inp
                    optimized_vars[inp] = inp

            # XXX: An in-place change
            node.inputs[idx] = nw_in

        # Apply the optimizations
        results = node.outputs
        for opt in list_opt:
            ret = opt.transform(fgraph, node)
            if ret is not False and ret is not None:
                assert len(ret) == len(node.outputs), opt
                for k, v in zip(node.outputs, ret):
                    optimized_vars[k] = v
                results = ret
                if ret[0].owner:
                    node = out.owner
                else:
                    break

        return results, optimized_vars

    if out.owner:
        out_index = out.owner.outputs.index(out)
    else:
        out_index = 0

    final_outs, optimized_nodes = local_recursive_function(optimizations, out, {}, 0)
    return final_outs[out_index]


def copy_stack_trace(from_var, to_var):
    """
    Copies the stack trace from one or more tensor variables to
    one or more tensor variables and returns the destination variables.

    Parameters
    ----------
    from_var
        Tensor variable or list of tensor variables to copy stack traces from.
    to_var
        Tensor variable or list of tensor variables to copy stack traces to.

    Notes
    -----
    The stacktrace is assumed to be of the form of a list of lists
    of tuples. Each tuple contains the filename, line number, function name
    and so on. Each list of tuples contains the truples belonging to a
    particular variable.

    """

    # Store stack traces from from_var
    tr = []
    if isinstance(from_var, Iterable) and not isinstance(from_var, Variable):
        # If from_var is a list, store concatenated stack traces
        for v in from_var:
            tr += getattr(v.tag, "trace", [])

    else:
        # If from_var is not a list, it must be a single tensor variable,
        # so just store that particular stack trace
        tr = getattr(from_var.tag, "trace", [])

    if tr and isinstance(tr[0], tuple):
        # There was one single stack trace, we encapsulate it in a list
        tr = [tr]

    # Copy over stack traces to to_var
    if isinstance(to_var, Iterable) and not isinstance(to_var, Variable):
        # Copy over stack traces from from_var to each variable in
        # to_var, including the stack_trace of the to_var before
        for v in to_var:
            v.tag.trace = getattr(v.tag, "trace", []) + tr
    else:
        # Copy over stack traces from from_var to each variable to
        # to_var, including the stack_trace of the to_var before
        to_var.tag.trace = getattr(to_var.tag, "trace", []) + tr
    return to_var


@contextlib.contextmanager
def inherit_stack_trace(from_var):
    """
    Contextmanager that copies the stack trace from one or more variable nodes to all
    variable nodes constructed in the body. new_nodes is the list of all the newly created
    variable nodes inside an optimization that is managed by graph.nodes_constructed().

    Parameters
    ----------
    from_var
        Variable node or a list of variable nodes to copy stack traces from.

    """
    with nodes_constructed() as new_nodes:
        yield
    copy_stack_trace(from_var, new_nodes)


def check_stack_trace(f_or_fgraph, ops_to_check="last", bug_print="raise"):
    """
    This function checks if the outputs of specific ops of a compiled graph
    have a stack.

    Parameters
    ----------
    f_or_fgraph: aesara.compile.function.types.Function or
          aesara.graph.fg.FunctionGraph
        The compiled function or the function graph to be analysed.
    ops_to_check: it can be of four different types:
          - classes or instances inheriting from aesara.graph.op.Op
          - tuple/list of classes or instances inheriting from aesara.graph.op.Op
          - string
          - function returning a boolean and taking as input an instance of
            aesara.graph.op.Op.
        - if ops_to_check is a string, it should be either 'last' or 'all'.
          'last' will check only the last op of the graph while 'all' will
          check all the ops of the graph.
        - if ops_to_check is an op or a tuple/list of ops, the function will
          check that all the outputs of their occurrences in the graph have a
          stack trace.
        - if ops_to_check is a function, it should take as input a
          aesara.graph.op.Op and return a boolean indicating if the input op should
          be checked or not.
    bug_print: string belonging to {'raise', 'warn', 'ignore'}
        You can specify the behaviour of the function when the specified
        ops_to_check are not in the graph of f_or_fgraph: it can either raise
        an exception, write a warning or simply ignore it.

    Returns
    -------
    boolean
        True if the outputs of the specified ops have a stack, False otherwise.

    """
    if isinstance(f_or_fgraph, aesara.compile.function.types.Function):
        fgraph = f_or_fgraph.maker.fgraph
    elif isinstance(f_or_fgraph, aesara.graph.fg.FunctionGraph):
        fgraph = f_or_fgraph
    else:
        raise ValueError("The type of f_or_fgraph is not supported")

    if isinstance(ops_to_check, Op) or (
        inspect.isclass(ops_to_check) and issubclass(ops_to_check, Op)
    ):
        ops_to_check = (ops_to_check,)

    # if ops_to_check is a string
    if isinstance(ops_to_check, str):
        if ops_to_check == "last":
            apply_nodes_to_check = [
                fgraph.outputs[i].owner for i in range(len(fgraph.outputs))
            ]
        elif ops_to_check == "all":
            apply_nodes_to_check = fgraph.apply_nodes
        else:
            raise ValueError("The string ops_to_check is not recognised")

    # if ops_to_check is a list/tuple of ops
    elif isinstance(ops_to_check, (tuple, list)):
        # Separate classes from instances in ops_to_check
        op_instances = []
        op_classes = []
        for obj in ops_to_check:
            if isinstance(obj, Op):
                op_instances.append(obj)
            else:
                op_classes.append(obj)
        op_classes = tuple(op_classes)

        apply_nodes_to_check = [
            node for node in fgraph.apply_nodes if node.op in ops_to_check
        ] + [
            node
            for node in fgraph.apply_nodes
            if isinstance(node.op, op_classes)
            or (
                hasattr(node.op, "scalar_op")
                and isinstance(node.op.scalar_op, op_classes)
            )
        ]

    # if ops_to_check is a function
    elif callable(ops_to_check):
        apply_nodes_to_check = [
            node for node in fgraph.apply_nodes if ops_to_check(node)
        ]

    else:
        raise ValueError("ops_to_check does not have the right type")

    if not apply_nodes_to_check:
        msg = (
            "Provided op instances/classes are not in the graph or the "
            "graph is empty"
        )
        if bug_print == "warn":
            warnings.warn(msg)
        elif bug_print == "raise":
            raise Exception(msg)
        elif bug_print == "ignore":
            pass
        else:
            raise ValueError("The string bug_print is not recognised")

    for node in apply_nodes_to_check:
        for output in node.outputs:
            if not hasattr(output.tag, "trace") or not output.tag.trace:
                return False

    return True


class CheckStackTraceFeature(Feature):
    def on_import(self, fgraph, node, reason):
        # In optdb we only register the CheckStackTraceOptimization when
        # config.check_stack_trace is not off but we also double check here.
        if config.check_stack_trace != "off" and not check_stack_trace(fgraph, "all"):
            if config.check_stack_trace == "raise":
                raise AssertionError(
                    "Empty stack trace! The optimization that inserted this variable is "
                    + str(reason)
                )
            elif config.check_stack_trace in ["log", "warn"]:
                apply_nodes_to_check = fgraph.apply_nodes
                for node in apply_nodes_to_check:
                    for output in node.outputs:
                        if not hasattr(output.tag, "trace") or not output.tag.trace:
                            output.tag.trace = [
                                [
                                    (
                                        "",
                                        0,
                                        "Empty stack trace! The optimization that"
                                        + "inserted this variable is "
                                        + str(reason),
                                        "",
                                    )
                                ]
                            ]
                if config.check_stack_trace == "warn":
                    warnings.warn(
                        "Empty stack trace! The optimization that inserted this variable is"
                        + str(reason)
                    )


class CheckStackTraceOptimization(GlobalOptimizer):
    """Optimizer that serves to add CheckStackTraceOptimization as an fgraph feature."""

    def add_requirements(self, fgraph):
        if not hasattr(fgraph, "CheckStackTraceFeature"):
            fgraph.attach_feature(CheckStackTraceFeature())

    def apply(self, fgraph):
        pass
