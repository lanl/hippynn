"""
Interface for creating LAMMPS MLIAP Unified models.
"""
import pickle
import warnings

import numpy as np
import torch

from lammps.mliap.mliap_unified_abc import MLIAPUnified

import hippynn
from hippynn.tools import device_fallback
from hippynn.graphs import find_relatives, find_unique_relative, get_subgraph, copy_subgraph, replace_node, IdxType, GraphModule
from hippynn.graphs.indextypes import index_type_coercion
from hippynn.graphs.gops import check_link_consistency
from hippynn.graphs.nodes.base import InputNode, SingleNode, MultiNode, AutoNoKw, ExpandParents
from hippynn.graphs.nodes.tags import Encoder, PairIndexer
from hippynn.graphs.nodes.indexers import PaddingIndexer
from hippynn.graphs.nodes.physics import GradientNode, VecMag
from hippynn.graphs.nodes.inputs import SpeciesNode
from hippynn.graphs.nodes.pairs import PairFilter


class MLIAPInterface(MLIAPUnified):
    """
    Class for creating ML-IAP Unified model based on hippynn graphs.
    """

    def __init__(
        self,
        energy_node,
        element_types,
        ndescriptors=1,
        model_device=torch.device("cpu"),
        compute_dtype=torch.float32,
        energy_unit: float = None,
        distance_unit: float = None,
    ):
        """
        :param energy_node: Node for energy
        :param element_types: list of atomic symbols corresponding to element types
        :param ndescriptors: the number of descriptors to report to LAMMPS
        :param model_device: the device to send torch data to (cpu or cuda)
        :param energy_unit: If present, multiply the result by the given energy units.
            If your model was trained in Hartree and your lammps script will operate in eV,
            use en_unit = ase.units.Ha = 27.211386024367243
        :param distance_unit: If present, multi input distances by this much as well as dividing into output forces.
            If your model was trained to accept nm as input and lammps uses Angstroms,
            use dist_unit = ase.units.nm = 10.
        """
        super().__init__()
        if hippynn.settings.PYTORCH_GPU_MEM_FRAC < 1.0:
            torch.cuda.set_per_process_memory_fraction(hippynn.settings.PYTORCH_GPU_MEM_FRAC)
        self.element_types = element_types
        self.ndescriptors = ndescriptors
        self.model_device = model_device
        self.energy_unit = energy_unit
        self.distance_unit = distance_unit

        # Build the calculator
        self.rcutfac, self.species_set, self.graph = setup_LAMMPS_graph(energy_node)
        self.nparams = sum(p.nelement() for p in self.graph.parameters())
        self.compute_dtype = compute_dtype
        self.graph.to(compute_dtype)

    def compute_gradients(self, data):
        pass

    def compute_descriptors(self, data):
        pass

    def as_tensor(self, array):
        return torch.as_tensor(array, device=self.model_device)

    def empty_tensor(self, dimentions):
        return torch.empty(dimentions, device=self.model_device)

    def compute_forces(self, data):
        """
        :param data: MLIAPData object (provided internally by lammps)
        :return None
        This function writes results to the input `data`.
        """
        nlocal = self.as_tensor(data.nlistatoms)
        if nlocal.item() > 0:
            # If there are no local atoms, do nothing
            elems = self.as_tensor(data.elems).type(torch.int64).reshape(1, data.ntotal)
            z_vals = self.species_set[elems + 1]
            npairs = data.npairs

            if npairs > 0:
                pair_i = self.as_tensor(data.pair_i).type(torch.int64)
                pair_j = self.as_tensor(data.pair_j).type(torch.int64)
                rij = self.as_tensor(data.rij).type(self.compute_dtype)
            else:
                pair_i = self.empty_tensor(0).type(torch.int64)
                pair_j = self.empty_tensor(0).type(torch.int64)
                rij = self.empty_tensor([0, 3]).type(self.compute_dtype)

            if self.distance_unit is not None:
                rij = self.dist_unit * rij

            # note your sign for rij might need to be +1 or -1, depending on how your implementation works
            inputs = [z_vals, pair_i, pair_j, -rij, nlocal]
            atom_energy, total_energy, fij = self.graph(*inputs)
            # Test if we are using lammps-kokkos or not. Is there a more clear way to do that?
            using_kokkos = "kokkos" in data.__class__.__module__.lower()
            if using_kokkos:
                return_device = elems.device
            else:
                return_device = "cpu"

            # convert units
            if self.energy_unit is not None:
                atom_energy = self.en_unit * atom_energy
                total_energy = self.en_unit * total_energy
                fij = self.en_unit * fij

            if self.distance_unit is not None:
                fij = fij / self.dist_unit

            atom_energy = atom_energy.squeeze(1).detach().to(return_device)
            total_energy = total_energy.detach().to(return_device)

            f = self.as_tensor(data.f)
            fij = fij.type(f.dtype).detach().to(return_device)

            # hacky way to detect if we are in kokkos or not.

            if not using_kokkos:
                # write back to data.eatoms directly.
                fij = fij.numpy()
                data.eatoms = atom_energy.numpy().astype(np.double)
                if npairs > 0:
                    data.update_pair_forces(fij)
            else:
                # view to data.eatoms using pytorch, and write into the view.
                eatoms = torch.as_tensor(data.eatoms, device=return_device)
                eatoms.copy_(atom_energy)
                if npairs > 0:
                    if return_device == "cpu":
                        data.update_pair_forces_cpu(fij)
                    else:
                        data.update_pair_forces_gpu(fij)

            data.energy = total_energy.item()

    def __getstate__(self):
        self.species_set = self.species_set.to(torch.device("cpu"))
        self.graph.to(torch.device("cpu"))
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)
        try:
            torch.ones(0).to(self.model_device)
        except RuntimeError:
            fallback = device_fallback()
            warnings.warn(f"Model device ({self.model_device}) not found, falling back to f{fallback}")
            self.model_device = fallback

        if not hasattr(self, "en_unit"):
            self.en_unit = None
        if not hasattr(self, "dist_unit"):
            self.dist_unit = None

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
    in_pair_first._index_state = hippynn.graphs.IdxType.Pair
    in_pair_second = InputNode("pair_second")
    in_pair_second._index_state = hippynn.graphs.IdxType.Pair
    in_pair_coord = InputNode("pair_coord")
    in_pair_coord._index_state = hippynn.graphs.IdxType.Pair
    in_nlocal = InputNode("nlocal")
    in_nlocal._index_state = hippynn.graphs.IdxType.Scalar
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
