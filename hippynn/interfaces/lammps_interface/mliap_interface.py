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
from hippynn.graphs.indextypes import index_type_coercion
from hippynn.graphs.gops import check_link_consistency
from hippynn.graphs.nodes.base import InputNode, MultiNode, AutoNoKw, ExpandParents
from hippynn.graphs.nodes.tags import Encoder, PairIndexer
from hippynn.graphs.nodes.physics import GradientNode, VecMag
from hippynn.graphs.nodes.inputs import SpeciesNode

class MLIAPInterface(MLIAPUnified):
    """
    Class for creating ML-IAP Unified model based on hippynn graphs.
    """
    def __init__(self, energy_node, element_types, ndescriptors=1,
                 model_device=torch.device("cpu")):
        """
        :param energy_node: Node for energy
        :param element_types: list of atomic symbols corresponding to element types
        :param ndescriptors: the number of descriptors to report to LAMMPS
        :param model_device: the device to send torch data to (cpu or cuda)
        """
        super().__init__()
        self.element_types = element_types
        self.ndescriptors = ndescriptors
        self.model_device = model_device

        # Build the calculator
        self.rcutfac, self.species_set, self.graph = setup_LAMMPS_graph(energy_node)
        self.nparams = sum(p.nelement() for p in self.graph.parameters())
        self.graph.to(torch.float64)

    def compute_gradients(self, data):
        pass
    
    def compute_descriptors(self, data):
        pass
    
    def as_tensor(self,array):
        return torch.as_tensor(array,device=self.model_device)

    def compute_forces(self, data):
        """
        :param data: MLIAPData object (provided internally by lammps)
        :return None
        This function writes results to the input `data`.
        """
        elems = self.as_tensor(data.elems).type(torch.int64).reshape(1, data.ntotal)
        z_vals = self.species_set[elems+1]
        pair_i = self.as_tensor(data.pair_i).type(torch.int64)
        pair_j = self.as_tensor(data.pair_j).type(torch.int64)
        rij = self.as_tensor(data.rij).type(torch.float64)
        nlocal = self.as_tensor(data.nlistatoms) 
           
        # note your sign for rij might need to be +1 or -1, depending on how your implementation works
        inputs = [z_vals, pair_i, pair_j, -rij, nlocal]
        atom_energy, total_energy, fij = self.graph(*inputs)
        
        # Test if we are using lammps-kokkos or not. Is there a more clear way to do that?
        if isinstance(data.elems,np.ndarray):
            return_device = 'cpu'
        else:
            # Hope that kokkos device and pytorch device are the same (default cuda)
            return_device = elems.device
        
        atom_energy = atom_energy.squeeze(1).detach().to(return_device)
        total_energy = total_energy.detach().to(return_device)

        f = self.as_tensor(data.f)
        fij = fij.type(f.dtype).detach().to(return_device)
        
        if return_device=="cpu":
            fij = fij.numpy()
            data.eatoms = atom_energy.numpy().astype(np.double)
        else:
            eatoms = torch.as_tensor(data.eatoms,device=return_device)
            eatoms.copy_(atom_energy)
         
        data.update_pair_forces(fij)
        data.energy = total_energy.item()

    def __getstate__(self):
        self.species_set = self.species_set.to(torch.device("cpu"))
        self.graph.to(torch.device("cpu"))
        return self.__dict__.copy()
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.species_set = self.species_set.to(self.model_device)
        self.graph.to(self.model_device)


def setup_LAMMPS_graph(energy):
    """

    :param energy: energy node for lammps interface
    :return: graph for computing from lammps MLIAP unified inputs.
    """
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


    local_atom_energy = LocalAtomEnergyNode("(LAMMPS)local_atom_energy", (atom_energies, in_nlocal))
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
