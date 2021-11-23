"""
Nodes not otherwise categorized.
"""
from ..indextypes import IdxType
from .base import AutoNoKw, SingleNode, MultiNode
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
