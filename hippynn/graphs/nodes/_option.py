"""

This module is experimental.

:meta private:
"""
# Can't import this high-level node type before importing graphs.nodes,
# because it requires .graphs, and .graphs requires .nodes!
from collections import OrderedDict


# TODO
## COMPLETELY UNTESTED work in progress
# This was partially implemented as a way to think about mixed boundary conditions,
# so the graph processing can be conditional on some variable about the system
# Caveat Emptor: does not track modules used in the criteria functions!
# Criteria functions should have no learnable parameters!

import torch

from .base import SingleNode, MultiNode, AutoKw, ExpandParents


class _OptionModule(torch.nn.Module):
    def __init__(self, criteria_dict):
        super().__init__()

        # Similar indirection to GraphModule; see construction of GraphModule.moddict.
        self.criteria_names = OrderedDict((fn, "option{}".format(i)) for i, fn in enumerate(criteria_dict))
        self.moddict = torch.nn.ModuleDict({self.criteria_names[fn]: mod for fn, mod in self.criteria_dict.items()})

    def forward(self, option_input, *other_inputs):
        for fn, fn_name in zip(self.criteria_names):
            if fn(option_input):
                return self.moddict[fn_name](*other_inputs)
        raise RuntimeError("No criteria satisfied for option {}".format(option_input))


class OptionalMixin(ExpandParents, AutoKw):
    _auto_module_class = _OptionModule

    @_parent_expander.match(SingleNode)
    def _expand0(self, option, *, criteria_map, one_option):
        option_parents = one_option.parents
        assert all(
            o.parents == option_parents for k, o in criteria_map
        ), "All possible options must be linked to the same parents!"
        return (option, *option_parents)

    def __init__(self, name, parents, criteria_map, module="auto", **kwargs):
        one_option = criteria_map[list(criteria_map.keys())[0]]

        parents = self.expand_parents(parents, criteria_map, one_option)
        self.module_kwargs = {"criteria_map": {k: v.torch_module for k, v in criteria_map}}
        self._input_names = parents

        super().__init__(name, parents, module=module, **kwargs)


class OptionalNode(OptionalMixin, SingleNode):
    def __init__(self, name, parents, criteria_map, module="auto", **kwargs):
        assert all(isinstance(v, SingleNode) for k, v in criteria_map), "OptionalNode option types must be all BaseNode"
        one_option = one_value_from_dict(criteria_map)

        idxstate = one_option._index_state
        assert all(
            o._index_state == idxstate for o in criteria_map.values()
        ), "Sub-options must provide the same index structure to be compatible"
        self._index_state = idxstate

        super().__init__(self, name, parents, module=module, **kwargs)


class OptionalMultiNode(OptionalMixin, MultiNode):
    def __init__(self, name, parents, criteria_map, module="auto", **kwargs):
        one_option = one_value_from_dict(criteria_map)
        assert all(
            isinstance(v, SingleNode) for k, v in criteria_map
        ), "OptionalMultiNode option types must be all MultiNode"

        onames = one_option._output_names
        assert all(
            o._output_names == onames for o in criteria_map.values()
        ), "Sub-options must provide the same outputs to be compatible"
        self._output_names = onames

        child_states = one_option._output_index_states
        assert all(
            o._output_index_states == child_states for o in criteria_map.values()
        ), "Sub-options must provide the same index structure to be compatible"
        self._output_index_states = child_states

        super().__init__(self, name, parents, module=module, **kwargs)


def make_option(name, option_node, criteria_map, module="auto"):
    one_option = one_value_from_dict(criteria_map)
    if isinstance(one_option, MultiNode):
        return OptionalMultiNode(name, (option_node,), criteria_map, module=module)
    if isinstance(one_option, SingleNode):
        return OptionalNode(name, (option_node,), criteria_map, module=module)
    raise ValueError("No optional node constructor matching type: {}".format(type(one_option)))


def one_value_from_dict(d):
    return d[list(d.keys()[0])]
