"""
Modules for analyzing pair-valued data
"""
import torch


class RDFBins(torch.nn.Module):
    def __init__(self, bins, species_set):
        super().__init__()
        if bins is None:
            raise TypeError("Bins must not be None!")
        bins = torch.as_tensor(bins)
        species_set = torch.as_tensor(species_set)
        self.register_buffer("bins", bins.to(torch.float))
        self.register_buffer("species_set", species_set.to(torch.int))

    def bin_info(self):
        # Note: widths don't make perfect sense for non-evenly-spaced bins.
        centers = (self.bins[1:] + self.bins[:-1]) / 2
        widths = self.bins[1:] - self.bins[:-1]
        return centers, widths

    def forward(self, pair_dist, pair_first, pair_second, one_hot, n_molecules):
        n_species = one_hot.shape[-1]
        n_bins = self.bins.shape[0] - 1

        rdf = torch.zeros((n_species, n_species, n_bins), dtype=pair_dist.dtype, device=pair_dist.device)
        for i in range(n_species):
            for j in range(n_species):
                mask = one_hot[:, i][pair_first] & one_hot[:, j][pair_second]
                maskpairs = pair_dist[mask]
                less = maskpairs.unsqueeze(-1) < self.bins.unsqueeze(0)
                less_counts = less.sum(dim=0)
                rdf[i, j] = less_counts[..., 1:] - less_counts[..., :-1]
        return (rdf / n_molecules).unsqueeze(0)


def min_dist_info(rij_list, j_list, mol_index, atom_index, inv_real_atoms, n_atoms_max, n_molecules):
    n_atoms = rij_list.shape[0]
    dev = rij_list.device

    if rij_list.shape[1] == 0:
        # empty neighbors list
        min_dist_mol = torch.zeros(n_molecules, dtype=rij_list.dtype, device=dev)
        min_dist_atom = torch.zeros(n_atoms, dtype=rij_list.dtype, device=dev)
        min_dist_mol_atom_locs = torch.zeros(n_molecules, dtype=torch.int64, device=dev)
        min_dist_atomneigh = torch.zeros((n_atoms), dtype=torch.int64, device=dev)
        return min_dist_mol, min_dist_mol_atom_locs, min_dist_atom, min_dist_atomneigh

    rmag_list = rij_list.norm(dim=2)
    maxr = rmag_list.max()

    rmaglist_new = rmag_list.clone()
    rmaglist_new[rmaglist_new == 0] = maxr
    min_dist_atom, where_min_dist_atom = rmaglist_new.min(dim=1)

    ara = torch.arange(n_atoms, dtype=where_min_dist_atom.dtype, device=dev)
    min_dist_atomneigh = j_list[ara, where_min_dist_atom]

    min_dist_molatom = torch.full((n_molecules, n_atoms_max), maxr, device=rmag_list.device, dtype=rmag_list.dtype)
    min_dist_molatom[mol_index, atom_index] = min_dist_atom
    min_dist_mol, where_min_dist_mol = min_dist_molatom.min(dim=1)

    atom1 = where_min_dist_mol

    atom1_batchloc = torch.arange(n_molecules, device=dev, dtype=torch.int64) * n_atoms_max + atom1
    atom1_atomloc = inv_real_atoms[atom1_batchloc]
    atom2 = atom_index[min_dist_atomneigh[atom1_atomloc]]

    min_dist_mol_atom_locs = torch.stack([atom1, atom2], dim=1)

    return min_dist_mol, min_dist_mol_atom_locs, min_dist_atom, min_dist_atomneigh


class MinDistModule(torch.nn.Module):
    def forward(self, rmag_list, j_list, mol_index, atom_index, inv_real_atoms, n_atoms_max, n_molecules):
        return min_dist_info(rmag_list, j_list, mol_index, atom_index, inv_real_atoms, n_atoms_max, n_molecules)
