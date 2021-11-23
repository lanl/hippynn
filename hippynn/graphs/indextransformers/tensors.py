"""
Functions for packing and unpacking tensor formats.
"""
from ..indextypes import register_index_transformer, IdxType

from ..nodes.indexers import QuadUnpackNode


@register_index_transformer(IdxType.QuadPack, IdxType.QuadMol)
def idx_QuadTriMol(node):
    parents = (node,)
    return parents, QuadUnpackNode
