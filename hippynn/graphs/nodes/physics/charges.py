


from ....layers import physics as physics_layers
from ...indextypes import IdxType, elementwise_compare_reduce, index_type_coercion
from ..base import (
    AutoKw,
    AutoNoKw,
    ExpandParents,
    MultiNode,
    SingleNode,
    Node,
    find_unique_relative,
)
from ..indexers import AtomIndexer, acquire_encoding_padding
from ..inputs import PositionsNode
from ..tags import Charges


class ChargeMomentNode(ExpandParents, AutoNoKw, SingleNode):
    input_names = "charges", "positions", "system_index", "n_systems"

    @parent_expander.matchlen(1)
    def expansion0(self, charges, *, purpose, **kwargs):
        return charges, find_unique_relative(charges, PositionsNode, why_desc=purpose)

    @parent_expander.match(Charges, PositionsNode)
    def expansion1(self, charges, positions, *, purpose, **kwargs):
        enc, pidxer = acquire_encoding_padding((charges, positions), species_set=None, purpose=purpose)
        return charges, positions, pidxer

    @parent_expander.match(Charges, PositionsNode, AtomIndexer)
    def expansion2(self, charges, positions, pdxer, **kwargs):
        return charges, positions, pdxer.system_index, pdxer.n_systems

    parent_expander.assertlen(4)
    parent_expander.get_main_outputs()
    parent_expander.require_idx_states(IdxType.Atoms, IdxType.Atoms, None, None)

    def __init__(self, name, parents, module="auto", **kwargs):
        parents = self.expand_parents(parents)
        super().__init__(name, parents, module=module, **kwargs)


class DipoleNode(ChargeMomentNode):
    """
    Compute the dipole of point charges.
    """

    auto_module_class = physics_layers.Dipole
    index_state = IdxType.Systems


class QuadrupoleNode(ChargeMomentNode):
    """
    Compute the traceless quadrupole of point charges.
    """

    auto_module_class = physics_layers.Quadrupole
    index_state = IdxType.QuadMol
