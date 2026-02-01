import torch
from ...layers import cheq as cheq_layers 

from .base import MultiNode, AutoKw, find_unique_relative, ExpandParents, SingleNode
from ..indextypes import IdxType
from .networks import Network
from .targets import HChargeNode, HBondNode
from .inputs import PositionsNode, SpeciesNode, CellNode
from .pairs import PairIndexer
from .indexers import PaddingIndexer


class ChEQNode(ExpandParents, AutoKw, MultiNode):
    _input_names = "species", "coordinates", "U", "chi"  # , "real_atoms"#, \
    # "pair_first", "pair_second", "pair_dist"
    _output_names = "charge", "coul_energy", "dipole", "out_U", "out_chi"
    _output_index_states = (
        IdxType.MolAtom,
        IdxType.Molecules,
        IdxType.Molecules,
        IdxType.MolAtom,
        IdxType.MolAtom,
    )  # (IdxType.Molecules, )*len(_output_names)

    _main_output = "charge"
    _auto_module_class = cheq_layers.ChEQ

    @_parent_expander.match(Network)
    def expand0(self, network, **kwargs):
        U = HChargeNode("ChEQ_U", network, module_kwargs=dict(first_is_interacting=False))
        chi = HChargeNode("ChEQ_chi", network, module_kwargs=dict(first_is_interacting=False))
        return U, chi

    @_parent_expander.match(Network, Network)
    def expand1(self, network1, network2, **kwargs):
        U = HChargeNode("ChEQ_U", network1, module_kwargs=dict(first_is_interacting=False))
        chi = HChargeNode("ChEQ_chi", network2, module_kwargs=dict(first_is_interacting=False))
        return U, chi

    @_parent_expander.match(HChargeNode, HChargeNode)
    def expand2(self, U, chi, **kwargs):
        positions = find_unique_relative([U, chi], PositionsNode)
        species = find_unique_relative([U, chi], SpeciesNode)
        # indexer = find_unique_relative([U, chi], PaddingIndexer)
        # pairs = find_unique_relative([U, chi], PairIndexer)

        return species, positions, U.main_output, chi.main_output  # , indexer.real_atoms, pairs.pair_first, \
        # pairs.pair_second, pairs.pair_dist

    def __init__(self, name, parents, lower_bound=0.0, units={"energy": "eV", "length": "Angstrom"}, module="auto", **kwargs):
        parents = self.expand_parents(parents, **kwargs)
        self.module_kwargs = dict(lower_bound=lower_bound, units=units)
        super().__init__(name, parents, module=module, **kwargs)
