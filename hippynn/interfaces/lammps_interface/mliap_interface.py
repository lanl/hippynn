"""
Interface for creating LAMMPS MLIAP Unified models.
"""
import pickle

import numpy as np
import torch
torch.set_default_dtype(torch.float32)

from lammps.mliap.mliap_unified_abc import MLIAPUnified

import hippynn
from hippynn.graphs import (find_relatives,find_unique_relative,
    get_subgraph, copy_subgraph, replace_node, IdxType,
    GraphModule)
from hippynn.graphs.gops import check_link_consistency
from hippynn.graphs.nodes.base import InputNode, MultiNode, AutoNoKw, ExpandParents
from hippynn.graphs.nodes.tags import Encoder, PairIndexer
from hippynn.graphs.nodes.physics import GradientNode, VecMag
from hippynn.graphs.nodes.inputs import SpeciesNode

class MLIAPInterface(MLIAPUnified):
    """
    Class for creating ML-IAP Unified model based on hippynn graphs.
    """
    def __init__(self, energy_node, element_types, ndescriptors=1, nparams=None,
                 model_device=torch.device("cpu")):
        """
        :param energy_node: Node for energy
        :param element_types: list of atomic symbols corresponding to element types
        :param ndescriptors: the number of lammps descriptors
        :param nparams: the number of lammps parameters
        :param model_device: the device to send torch data to (cpu or cuda)
        """
        super().__init__()
        self.element_types = element_types
        self.ndescriptors = ndescriptors
        self._model_device = model_device

        # Build the calculator
        self.rcutfac, self.species_set, self.graph = setup_LAMMPS_graph(energy_node)
        if nparams is None:
            nparams = sum(p.nelement() for p in self.graph.parameters())
        self.nparams = nparams
        self.graph.to(torch.float64)

    def compute_gradients(self, data):
        """
        Test compute_gradients.
        
        :param data: MLIAPData object (provided internally by lammps)
        """
    
    def compute_descriptors(self, data):
        """
        Test compute_descriptors.
        
        :param data: MLIAPData object (provided internally by lammps)
        """
    
    def compute_forces(self, data):
        """
        Test compute_forces.
        
        :param data: MLIAPData object (provided internally by lammps)
        """
        elems = torch.from_numpy(data.elems).type(torch.int64).reshape(1, data.ntotal)
        z_vals = self.species_set[elems+1]
        pair_i = torch.from_numpy(data.pair_i).type(torch.int64)
        pair_j = torch.from_numpy(data.pair_j).type(torch.int64)
        rij = torch.from_numpy(data.rij).type(torch.float64)
        nlocal = torch.tensor(data.nlistatoms)

        # note your sign for rij might need to be +1 or -1, depending on how your implementation works
        inputs = (inp.to(self._model_device) for inp in [z_vals, pair_i, pair_j, -rij, nlocal])

        f = torch.from_numpy(data.f)
        atom_energy, total_energy, fij = self.graph(*inputs)

        atom_energy = atom_energy.to(torch.device("cpu"))
        total_energy = total_energy.to(torch.device("cpu"))
        fij = fij.type(f.dtype).to(torch.device("cpu"))

        data.update_pair_forces(fij.detach().numpy())
        data.eatoms = atom_energy.squeeze(1).detach().numpy().astype(np.double)
        data.energy = total_energy.item()

    def __getstate__(self):
        self.species_set = self.species_set.to(torch.device("cpu"))
        self.graph.to(torch.device("cpu"))
        return self.__dict__
    
    def __setstate__(self, d):
        self.__dict__ = d
        self.species_set = self.species_set.to(self._model_device)
        self.graph.to(self._model_device)


def setup_LAMMPS_graph(energy):
    required_nodes = [energy]

    why = "Generating LAMMPS Calculator interface"
    subgraph = get_subgraph(required_nodes)

    search_fn = lambda targ,sg: lambda n: n in sg and isinstance(n,targ)
    pair_indexers = find_relatives(required_nodes, search_fn(PairIndexer, subgraph), why_desc=why)

    new_required, new_subgraph = copy_subgraph(required_nodes, assume_inputed=pair_indexers, tag="LAMMPS")
    pair_indexers = find_relatives(new_required, search_fn(PairIndexer, new_subgraph), why_desc=why)

    species = find_unique_relative(new_required, search_fn(SpeciesNode, new_subgraph),why_desc=why)

    encoder = find_unique_relative(species, search_fn(Encoder, new_subgraph), why_desc=why)
    species_set = torch.as_tensor(encoder.species_set).to(torch.int64)
    min_radius = max(p.dist_hard_max for p in pair_indexers)

    ###############################################################
    # Set up graph to accept external pair indices and shifts

    in_pair_first = InputNode("(LAMMPS)pair_first")
    in_pair_first._index_state = hippynn.graphs.IdxType.Pair
    in_pair_second = InputNode("(LAMMPS)pair_second")
    in_pair_second._index_state = hippynn.graphs.IdxType.Pair
    in_pair_coord = InputNode("(LAMMPS)pair_coord")
    in_pair_coord._index_state = hippynn.graphs.IdxType.Pair
    in_nlocal = InputNode("(LAMMPS)nlocal")
    in_nlocal._index_state = hippynn.graphs.IdxType.Scalar
    pair_dist = VecMag("(LAMMPS)pair_dist", in_pair_coord)

    new_inputs = [species,in_pair_first,in_pair_second,in_pair_coord,in_nlocal]

    for pi in pair_indexers:
        replace_node(pi.pair_first, in_pair_first, disconnect_old=False)
        replace_node(pi.pair_second, in_pair_second, disconnect_old=False)
        replace_node(pi.pair_coord, in_pair_coord, disconnect_old=False)
        replace_node(pi.pair_dist, pair_dist, disconnect_old=False)
        pi.disconnect()

    energy, *new_required = new_required
    local_atom_energy = LocalAtomEnergyNode("(LAMMPS)local_atom_energy", (energy.atom_energies, in_nlocal))
    grad_rij = GradientNode("(LAMMPS)grad_rij", (local_atom_energy.total_local_energy, in_pair_coord), -1)

    implemented_nodes = local_atom_energy.local_atom_energies, local_atom_energy.total_local_energy, grad_rij

    check_link_consistency((*new_inputs, *implemented_nodes))
    mod = GraphModule(new_inputs, implemented_nodes)
    mod.eval()

    return min_radius / 2, species_set, mod


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

    def __init__(self, name, parents, module='auto', **kwargs):
        parents = self.expand_parents(parents)
        super().__init__(name, parents, module=module, **kwargs)
