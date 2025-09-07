import math
import warnings
import itertools

import numpy as np
import torch

from hippynn.graphs import IdxType
from hippynn.graphs.nodes.base import ExpandParents, find_unique_relative
from hippynn.graphs.nodes.base.multi import MultiNode
from hippynn.graphs.nodes.base.definition_helpers import AutoKw
from hippynn.graphs.nodes.inputs import SpeciesNode, PositionsNode
from hippynn.graphs.nodes.tags import PairIndexer, AtomIndexer
from hippynn.layers import pairs

## Define repulsive potential node for hippynn
class RepulsivePotential(torch.nn.Module):
    def __init__(self, taper_point, strength, dr, perc):
        '''
        Let F(r) be the force between two particles of distance r generated
        by this potential. Then 
        F(taper_point)      = perc * strength 
        F(taper_point - dr) = strength

        Eg. If taper_point=3, strength=1, dr=0.5, and perc=0.01, then
        F(3)    = 0.01
        F(2.5)  = 1
        '''
        super().__init__()
        self.t = taper_point
        self.s = strength
        self.d = dr
        self.p = perc

        self.a = (1/self.d)*math.log(1/self.p)
        self.g = -1 * self.s * self.p * math.exp(self.a * self.t) / self.a

        self.summer = pairs.MolPairSummer()

    def forward(self, pair_dist, pair_first, mol_index, n_molecules, n_atoms_max):
        pair_energies = -1 * self.g * torch.exp(-1 * self.a  * pair_dist)
        atom_energies = torch.zeros((n_molecules * n_atoms_max, 1), device=pair_energies.device, dtype=pair_energies.dtype)
        atom_energies.index_add_(0, pair_first, pair_energies.unsqueeze(-1))
        mol_energy = self.summer(pair_energies, mol_index, n_molecules, pair_first)
        return mol_energy, atom_energies

class RepulsivePotentialNode(ExpandParents, AutoKw, MultiNode):
    input_names = "pair_dist", "pair_first", "mol_index", "n_molecules", "n_atoms_max"
    output_names = "mol_energy", "atom_energies",
    auto_module_class = RepulsivePotential
    output_index_states = IdxType.Systems, IdxType.Atoms,

    @parent_expander.match(PairIndexer, AtomIndexer)
    def expansion(self, pairfinder, pidxer, **kwargs):
        return pairfinder.pair_dist, pairfinder.pair_first, pidxer.mol_index, pidxer.n_molecules, pidxer.n_atoms_max

    def __init__(self, name, parents, taper_point, strength, dr, perc, module="auto"):
        self.module_kwargs = {
            "taper_point": taper_point,
            "strength": strength,
            "dr": dr,
            "perc": perc,
        }
        parents = self.expand_parents(parents, module="auto")
        super().__init__(name, parents, module=module)


# Newer version with support for multiple species
class RepulsivePotentialBySpecies(torch.nn.Module):
    def __init__(self, taper_point, strength, dr, perc):
        '''
        :param taper_point: 2D PyTorch tensor of shape (max(species), max(species)) where the entry at index [i, j] corresponds to the 
                             taper point for the repulsive potential between particles of species i and species j. 
        '''
        super().__init__()

        if not torch.allclose(taper_point, taper_point.T):
            warnings.warn("Non-symmetric 'taper_point' matrix will lead to inconsistent results due to arbitrary pair orderings.")

        self.t = taper_point
        self.s = strength
        self.d = dr
        self.p = perc

        self.a = (1/self.d)*math.log(1/self.p)
        self.g = -1 * self.s * self.p * torch.exp(self.a * self.t) / self.a

        self.summer = pairs.MolPairSummer()

    def forward(self, pair_dist, pair_first, pair_second, mol_index, n_molecules, n_atoms_max, spec):
        self.g = self.g.to(spec.device)
        idx1 = spec[pair_first]
        idx2 = spec[pair_second]
        pair_energies = -1 * self.g[idx1,idx2].reshape(pair_dist.shape) * torch.exp(-1 * self.a  * pair_dist)
        atom_energies = torch.zeros((n_molecules * n_atoms_max, 1), device=pair_energies.device, dtype=pair_energies.dtype)
        atom_energies.index_add_(0, pair_first, pair_energies.unsqueeze(-1))
        mol_energy = self.summer(pair_energies, mol_index, n_molecules, pair_first)
        return mol_energy, atom_energies,

class RepulsivePotentialBySpeciesNode(ExpandParents, AutoKw, MultiNode):
    input_names = "pair_dist", "pair_first", "pair_second", "mol_index", "n_molecules", "n_atoms_max", "spec"
    output_names = "mol_energy", "atom_energies",
    auto_module_class = RepulsivePotentialBySpecies
    output_index_states = IdxType.Systems, IdxType.Atoms,

    @parent_expander.match(PairIndexer, AtomIndexer)
    @parent_expander.match(PairIndexer, AtomIndexer, SpeciesNode)
    def expansion(self, pairfinder, pidxer, species=None, **kwargs):
        if species is None:
            species = find_unique_relative(pairfinder, SpeciesNode)
        return pairfinder.pair_dist, pairfinder.pair_first, pairfinder.pair_second, pidxer.mol_index, pidxer.n_molecules, pidxer.n_atoms_max, species.main_output
    
    @parent_expander.match(PositionsNode)
    @parent_expander.match(PositionsNode, SpeciesNode)
    def expansion(self, positions, species=None, **kwargs):
        pairfinder = find_unique_relative(positions, PairIndexer)
        pidxer = find_unique_relative(positions, AtomIndexer)
        if species is None:
            species = find_unique_relative(pairfinder, SpeciesNode)
        return pairfinder.pair_dist, pairfinder.pair_first, pairfinder.pair_second, pidxer.mol_index, pidxer.n_molecules, pidxer.n_atoms_max, species.main_output
    
    parent_expander.require_idx_states(None, None, None, None, None, None, IdxType.Atoms)

    def __init__(self, name, parents, taper_point, strength, dr, perc, module="auto"):
        self.module_kwargs = {
            "taper_point": taper_point,
            "strength": strength,
            "dr": dr,
            "perc": perc,
        }
        parents = self.expand_parents(parents, module="auto")
        super().__init__(name, parents, module=module)

def find_taper_point(unique_species, species_rdfs, rdf_bins, taper_point_fn):
    taper_point = torch.zeros((max(unique_species)+1, max(unique_species)+1))
    for type1, type2 in itertools.combinations_with_replacement(unique_species, 2):
        rdf_values = species_rdfs[f"rdf_values_{type1}_{type2}"]
        repulsive_potential_taper_point = taper_point_fn(rdf_bins, rdf_values)
        taper_point[type1, type2] = repulsive_potential_taper_point
        taper_point[type2, type1] = repulsive_potential_taper_point
    return taper_point
