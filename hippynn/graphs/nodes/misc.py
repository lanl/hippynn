"""
Nodes not otherwise categorized.
"""
from ..indextypes import IdxType
from .base import AutoNoKw, SingleNode, MultiNode, ExpandParents
from ...layers import indexers as index_modules, algebra as algebra_modules


class StrainInducer(AutoNoKw, MultiNode):
    _input_names = "coordinates", "cell"
    _output_names = "strained_coordinates", "strained_cell", "strain"
    _output_index_states = NotImplemented
    _auto_module_class = index_modules.CellScaleInducer

    def __init__(self, name, parents, module="auto", **kwargs):
        position, cell = parents
        self._output_index_states = position._index_state, IdxType.NotFound, IdxType.NotFound
        super().__init__(name, parents, module=module, **kwargs)


class ListNode(AutoNoKw, SingleNode):
    _input_names = "features"
    _output_names = "wrapped_features"
    _auto_module_class = algebra_modules.ListMod

    def __init__(self, name, parents, module="auto"):
        super().__init__(name, parents, module=module)

class EnsembleTarget(MultiNode, ExpandParents, AutoNoKw):
    _auto_module_class = algebra_modules.EnsembleTarget
    _input_names = None # Overridden by constructor
    _output_names = "mean", "std", "all"

    _parent_expander.get_main_outputs()
    _parent_expander.require_compatible_idx_states()

    def __init__(self, name, parents, module="auto"):

        parents = self.expand_parents(parents)
        idx_state = parents[0]._index_state
        n_parents = len(parents)
        db_name = parents[0].db_name  # assumes that all are the same!

        self._output_index_states = (idx_state,)*3
        self._input_names = [f"input_{i}" for i in range(n_parents)]

        super().__init__(name, parents, module=module)
        for c, out_name in zip(self.children, self._output_names):
            c.db_name = f'{db_name}_{out_name}'
