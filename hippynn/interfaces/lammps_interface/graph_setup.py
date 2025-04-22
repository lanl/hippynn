import torch

from ... import IdxType, GraphModule
from ...graphs import get_subgraph, find_relatives, copy_subgraph, find_unique_relative, replace_node
from ...graphs.gops import check_link_consistency
from ...graphs.indextypes import index_type_coercion
from ...graphs.nodes.base import InputNode, AutoNoKw, SingleNode, ExpandParents, MultiNode
from ...graphs.nodes.indexers import PaddingIndexer
from ...graphs.nodes.inputs import SpeciesNode
from ...graphs.nodes.pairs import PairFilter
from ...graphs.nodes.physics import VecMag, GradientNode
from ...graphs.nodes.tags import PairIndexer, Encoder


def setup_LAMMPS_graph(energy):
    """

    :param energy: energy node for lammps interface
    :return: graph for computing from lammps MLIAP unified inputs.
    """
    required_nodes = [energy]

    why = "Generating LAMMPS Calculator interface"
    subgraph = get_subgraph(required_nodes)

    search_fn = lambda targ, sg: lambda n: n in sg and isinstance(n, targ)
    pair_indexers = find_relatives(required_nodes, search_fn(PairIndexer, subgraph), why_desc=why)

    new_required, new_subgraph = copy_subgraph(required_nodes, assume_inputed=pair_indexers)
    pair_indexers = find_relatives(new_required, search_fn(PairIndexer, new_subgraph), why_desc=why)

    species = find_unique_relative(new_required, search_fn(SpeciesNode, new_subgraph), why_desc=why)

    encoder = find_unique_relative(species, search_fn(Encoder, new_subgraph), why_desc=why)
    padding_indexer = find_unique_relative(species, search_fn(PaddingIndexer, new_subgraph), why_desc=why)
    inv_real_atoms = padding_indexer.inv_real_atoms

    species_set = torch.as_tensor(encoder.species_set).to(torch.int64)
    min_radius = max(p.dist_hard_max for p in pair_indexers)

    ###############################################################
    # Set up graph to accept external pair indices and shifts

    in_pair_first = InputNode("pair_first")
    in_pair_first._index_state = IdxType.Pair
    in_pair_second = InputNode("pair_second")
    in_pair_second._index_state = IdxType.Pair
    in_pair_coord = InputNode("pair_coord")
    in_pair_coord._index_state = IdxType.Pair
    in_nlocal = InputNode("nlocal")
    in_nlocal._index_state = IdxType.Scalar
    pair_dist = VecMag("pair_dist", in_pair_coord)
    mapped_pair_first = ReIndexAtomNode("pair_first_internal", (in_pair_first, inv_real_atoms))
    mapped_pair_second = ReIndexAtomNode("pair_second_internal", (in_pair_second, inv_real_atoms))

    new_inputs = [species, in_pair_first, in_pair_second, in_pair_coord, in_nlocal]

    # Construct Filters and replace the existing pair indexers with the
    # corresponding new (filtered) node that accepts external pairs of atoms
    for pi in pair_indexers:
        if pi.dist_hard_max == min_radius:
            replace_node(pi.pair_first, mapped_pair_first, disconnect_old=False)
            replace_node(pi.pair_second, mapped_pair_second, disconnect_old=False)
            replace_node(pi.pair_coord, in_pair_coord, disconnect_old=False)
            replace_node(pi.pair_dist, pair_dist, disconnect_old=False)
            pi.disconnect()
        else:
            mapped_node = PairFilter(
                "DistanceFilter-LAMMPS",
                (pair_dist, in_pair_first, in_pair_second, in_pair_coord),
                dist_hard_max=pi.dist_hard_max,
            )
            replace_node(pi.pair_first, mapped_node.pair_first, disconnect_old=False)
            replace_node(pi.pair_second, mapped_node.pair_second, disconnect_old=False)
            replace_node(pi.pair_coord, mapped_node.pair_coord, disconnect_old=False)
            replace_node(pi.pair_dist, mapped_node.pair_dist, disconnect_old=False)
            pi.disconnect()

    energy, *new_required = new_required
    try:
        atom_energies = energy.atom_energies
    except AttributeError:
        atom_energies = energy

    try:
        atom_energies = index_type_coercion(atom_energies, IdxType.Atoms)
    except ValueError:
        raise RuntimeError(
            "Could not build LAMMPS interface. Pass an object with index type IdxType.Atoms or "
            "an object with an `atom_energies` attribute."
        )

    local_atom_energy = LocalAtomEnergyNode("local_atom_energy", (atom_energies, in_nlocal))
    grad_rij = GradientNode("grad_rij", (local_atom_energy.total_local_energy, in_pair_coord), -1)

    implemented_nodes = local_atom_energy.local_atom_energies, local_atom_energy.total_local_energy, grad_rij

    check_link_consistency((*new_inputs, *implemented_nodes))
    mod = GraphModule(new_inputs, implemented_nodes)
    mod.eval()

    return min_radius / 2, species_set, mod


class ReIndexAtomMod(torch.nn.Module):
    def forward(self, raw_atom_index_array, inverse_real_atoms):
        return inverse_real_atoms[raw_atom_index_array]


class ReIndexAtomNode(AutoNoKw, SingleNode):
    _input_names = "raw_atom_index_array", "inverse_real_atoms"
    _main_output = "total_local_energy"
    _auto_module_class = ReIndexAtomMod

    def __init__(self, name, parents, module="auto", **kwargs):
        self._index_state = parents[0]._index_state
        super().__init__(name, parents, module=module, **kwargs)


class LocalAtomsEnergy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, all_atom_energies, nlocal):
        local_atom_energies = all_atom_energies[:nlocal]
        total_local_energy = torch.sum(local_atom_energies)
        return local_atom_energies, total_local_energy


class LocalAtomEnergyNode(AutoNoKw, ExpandParents, MultiNode):
    _input_names = "all_atom_energies", "nlocal"
    _output_names = "local_atom_energies", "total_local_energy"
    _main_output = "total_local_energy"
    _output_index_states = None, IdxType.Scalar
    _auto_module_class = LocalAtomsEnergy

    _parent_expander.assertlen(2)
    _parent_expander.get_main_outputs()
    _parent_expander.require_idx_states(IdxType.Atoms, IdxType.Scalar)

    def __init__(self, name, parents, module="auto", **kwargs):
        parents = self.expand_parents(parents)
        super().__init__(name, parents, module=module, **kwargs)
