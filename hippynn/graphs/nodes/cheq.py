"""
Node for charge equilibration model.
"""
from ...layers import cheq as cheq_layers 

from .base import MultiNode, AutoKw, find_unique_relative, find_relatives, ExpandParents, Node
from ..indextypes import IdxType
from .networks import Network
from .targets import HChargeNode
from .inputs import PositionsNode, SpeciesNode, CellNode


class ChEQNode(ExpandParents, AutoKw, MultiNode):
    input_names = "species", "coordinates", "U", "chi"
    output_names = "charge", "coul_energy", "dipole", "out_U", "out_chi"
    output_index_states = (
        IdxType.SysAtom,
        IdxType.Systems,
        IdxType.Systems,
        IdxType.SysAtom,
        IdxType.SysAtom,
    )

    main_output = "charge"
    auto_module_class = cheq_layers.ChEQ

    @parent_expander.match(Network)
    def expand0(self, network, **kwargs):
        U = HChargeNode("ChEQ_U", network, module_kwargs=dict(first_is_interacting=False))
        chi = HChargeNode("ChEQ_chi", network, module_kwargs=dict(first_is_interacting=False))
        return U, chi

    @parent_expander.match(Network, Network)
    def expand1(self, network1, network2, **kwargs):
        U = HChargeNode("ChEQ_U", network1, module_kwargs=dict(first_is_interacting=False))
        chi = HChargeNode("ChEQ_chi", network2, module_kwargs=dict(first_is_interacting=False))
        return U, chi

    @parent_expander.match(HChargeNode, HChargeNode)
    def expand2(self, U, chi, **kwargs):
        positions = find_unique_relative([U, chi], PositionsNode)
        species = find_unique_relative([U, chi], SpeciesNode)

        return species, positions, U.main_output, chi.main_output

    @parent_expander.match(Node, Node, Node, Node)
    def warn_if_pbc_detected(self, *parents, **kwargs):
        try:
            cell_nodes = find_relatives(parents, CellNode)
        except:
            import warnings
            warnings.warn("Periodic boundaries were detected in the graph; " +\
                          "This ChEQ node computes using open boundary conditions only",
                          stacklevel=3
                          )
        return parents


    def __init__(self, name, parents, lower_bound=0.0, units={"energy": "eV", "length": "Angstrom"}, module="auto", **kwargs):
        parents = self.expand_parents(parents, **kwargs)
        self.module_kwargs = dict(lower_bound=lower_bound, units=units)
        super().__init__(name, parents, module=module, **kwargs)
