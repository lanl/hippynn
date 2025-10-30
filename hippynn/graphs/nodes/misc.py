"""
Nodes not otherwise categorized.
"""
from ..indextypes import IdxType
from .base import AutoNoKw, SingleNode, MultiNode, ExpandParents
from ...layers import algebra as algebra_modules


class ListNode(AutoNoKw, SingleNode):
    input_names = "features"
    output_names = "wrapped_features"
    auto_module_class = algebra_modules.ListMod

    def __init__(self, name, parents, module="auto"):
        super().__init__(name, parents, module=module)


class EnsembleTarget(ExpandParents, AutoNoKw, MultiNode):
    auto_module_class = algebra_modules.EnsembleTarget
    input_names = NotImplemented  # NotImplemented tells __init_subclass__ that we will fill this in later.
    output_names = "mean", "std", "all"

    parent_expander.get_main_outputs()
    parent_expander.require_compatible_idx_states()

    def __init__(self, name, parents, module="auto"):

        parents = self.expand_parents(parents)

        index_state = parents[0].index_state
        db_name = parents[0].db_name  # assumes that all are the same!

        self.output_index_states = (index_state,)*3
        self.input_names = [f"input_{i}" for i in range(len(parents))]

        super().__init__(name, parents, module=module)
        for c, out_name in zip(self.children, self.output_names):
            c.db_name = f'{db_name}_{out_name}'


def __getattr__(name: str):
    """
    Module-level getattr for finding of old functions.
    """
    if name == "StrainInducer":
        # Backwards compatibility for unpickling prior models
        from . import physics
        from ..._deprecations import warn_name_change
        warn_name_change(name, name, old_module=__name__, new_module=physics, stacklevel=2)
        return physics.StrainInducer
    
    raise AttributeError(f"module {__name__!r} has no attribute {name}")
