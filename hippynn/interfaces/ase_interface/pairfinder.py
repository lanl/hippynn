import ase
import torch

from ...graphs.nodes.pairs import _DispatchNeighbors
from ...layers import pairs as pair_modules


def ASE_compute_neighbors(cutoff, positions, cell):
    positions = positions.detach().cpu().numpy()
    cell = cell.detach().cpu().numpy()

    nlist = ase.neighborlist.NeighborList(
        cutoff,
        skin=0.0,
        sorted=True,
        self_interaction=False,
        bothways=True,
        primitive=ase.neighborlist.NewPrimitiveNeighborList,
    )
    n_atoms = len(positions)
    symbols = ["H"] * n_atoms  # We don't actually care what atom types we use

    atoms = ase.Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
    nlist.update(atoms)
    pf = nlist.nl.pair_first
    ps = nlist.nl.pair_second
    offset = -nlist.nl.offset_vec  # Different conventions in ASE!
    return tuple(torch.from_numpy(x) for x in [pf, ps, offset])


class ASENeighbors(pair_modules.dispatch._DispatchNeighbors):
    def compute_one(self, positions, cell):
        return ASE_compute_neighbors(self.dist_hard_max, positions, cell)


class ASEPairNode(_DispatchNeighbors):
    _auto_module_class = ASENeighbors


from ...experiment import assembly

assembly._PAIRCACHE_COMPATIBLE_COMPUTERS.add(ASEPairNode)
