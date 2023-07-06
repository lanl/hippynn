"""
Nodes for inputting information to the graph.
"""
import warnings
from .base import InputNode
from ..indextypes import IdxType
from .tags import Charges, Positions, Species, PairCache


class SpeciesNode(Species, InputNode):
    _index_state = IdxType.MolAtom
    input_type_str = "Species"


class PositionsNode(Positions, InputNode):
    _index_state = IdxType.MolAtom
    input_type_str = "Positions"


class CellNode(InputNode):
    _index_state = IdxType.Molecules
    input_type_str = "Cells"

class ForceNode(InputNode):
    _index_state = IdxType.MolAtom
    input_type_str = "Force"


class InputCharges(Charges, InputNode):
    _index_state = IdxType.MolAtom
    input_type_str = "InputCharges"


class Indices(InputNode):
    _index_state = IdxType.Molecules
    input_type_str = "Index"

    def __init__(self):
        super().__init__(db_name="indices")


class PairIndices(PairCache, InputNode):
    _index_state = IdxType.NotFound
    input_type_str = "PairIndices"


class SplitIndices(InputNode):
    _index_state = IdxType.Molecules
    input_type_str = "Index"

    def __init__(self):
        super().__init__(db_name="indices")
