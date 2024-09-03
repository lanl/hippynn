import math
import torch

from hippynn.graphs import IdxType
from hippynn.graphs.nodes.base import ExpandParents
from hippynn.graphs.nodes.base.definition_helpers import AutoKw
from hippynn.graphs.nodes.base.multi import MultiNode
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

    def forward(self, pair_dist, pair_first, mol_index, n_molecules):
        atom_energies = -1 * self.g * torch.exp(-1 * self.a  * pair_dist)
        mol_energies = self.summer(atom_energies, mol_index, n_molecules, pair_first)
        return mol_energies, atom_energies,

class RepulsivePotentialNode(ExpandParents, AutoKw, MultiNode):
    _input_names = "pair_dist", "pair_first", "mol_index", "n_molecules"
    _output_names = "mol_energies", "atom_energies",
    _auto_module_class = RepulsivePotential
    _output_index_states = IdxType.Molecules, IdxType.Pair,

    @_parent_expander.match(PairIndexer, AtomIndexer)
    def expansion(self, pairfinder, pidxer, **kwargs):
        return pairfinder.pair_dist, pairfinder.pair_first, pidxer.mol_index, pidxer.n_molecules

    def __init__(self, name, parents, taper_point, strength, dr, perc, module="auto"):
        self.module_kwargs = {
            "taper_point": taper_point,
            "strength": strength,
            "dr": dr,
            "perc": perc,
        }
        parents = self.expand_parents(parents, module="auto")
        super().__init__(name, parents, module=module)