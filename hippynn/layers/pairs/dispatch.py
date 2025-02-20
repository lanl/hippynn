"""
System-by-system pair finders
"""

from itertools import product
import numpy as np
import torch

from .open import PairMemory
from .periodic import filter_pairs

def wrap_points_np(coords, cell, inv_cell):
    # cell is (basis,cartesian)
    # inv is (cartesian,basis)
    projections = coords @ inv_cell
    wrapped_offset, fractional_coords = np.divmod(projections, 1)
    # wrapped_coords is (atoms,cartesian)
    wrapped_coords = fractional_coords @ cell
    return wrapped_coords, wrapped_offset.astype(np.int64)


def neighbor_list_np(cutoff, coords, cell):
    # cell is (basis,cartesian)
    # inv is (cartesian,basis)
    inv_cell = np.linalg.inv(cell)
    coords, wrapped_offset = wrap_points_np(coords, cell, inv_cell)

    eps = 1e-5
    inv_lengths = np.linalg.norm(inv_cell, axis=0)
    n_bounds = (np.floor(cutoff * inv_lengths + eps) + 1).astype(int)
    n1, n2, n3 = n_bounds

    drij = coords[:, np.newaxis] - coords[np.newaxis, :]
    wrap_offset_ij = wrapped_offset[:, np.newaxis] - wrapped_offset[np.newaxis, :]

    pair_first = []
    pair_second = []
    pair_image = []
    for i1 in range(-n1, n1 + 1):
        for i2 in range(-n2, n2 + 1):
            for i3 in range(-n3, n3 + 1):
                oi = (i1, i2, i3)
                oi_float = np.asarray(oi, dtype=coords.dtype)
                offset = np.dot(oi_float, cell)
                diff = drij + offset
                dist = np.linalg.norm(diff, axis=2)
                pf, ps = np.nonzero(dist < cutoff)

                if i1 == i2 == i3 == 0:
                    nonself = pf != ps
                    pf = pf[nonself]
                    ps = ps[nonself]

                if len(pf) > 0:
                    pi = np.repeat(np.asarray(oi)[np.newaxis, :], repeats=len(pf), axis=0)
                    # pi is wrapped image locations, need to convert back to absolute.
                    pi = pi - wrap_offset_ij[pf, ps]
                    pair_first.append(pf)
                    pair_second.append(ps)
                    pair_image.append(pi)

    pair_first = np.concatenate(pair_first)
    pair_second = np.concatenate(pair_second)
    pair_image = np.concatenate(pair_image)

    return pair_first, pair_second, pair_image


@torch.jit.script
def wrap_points_torch(coords, cell, inv_cell):
    # cell is (basis,cartesian)
    # inv is (cartesian,basis)
    projections = coords @ inv_cell
    fractional_coords = torch.remainder(projections, 1)
    wrapped_offset = torch.div(projections, 1, rounding_mode="floor")
    # wrapped_coords is (atoms,cartesian)
    wrapped_coords = fractional_coords @ cell
    return wrapped_coords, wrapped_offset.to(torch.int64)


@torch.jit.script
def compute_offset_range(inv_cell, cutoff: float):
    dev = inv_cell.device
    eps = 1e-5
    inv_lengths = torch.linalg.norm(inv_cell, dim=0)
    n_bounds = (torch.floor(cutoff * inv_lengths + eps) + 1).to(torch.int64)
    n1, n2, n3 = n_bounds.unbind()
    n1range = torch.arange(-n1, n1 + 1, device=dev)
    n2range = torch.arange(-n2, n2 + 1, device=dev)
    n3range = torch.arange(-n3, n3 + 1, device=dev)

    offset_range = torch.cartesian_prod(n1range, n2range, n3range)
    # shape: n_permutations, 3 (basis)
    return offset_range


@torch.jit.script
def neighbor_list_torch(cutoff: float, coords, cell):
    # cell is (basis,cartesian)
    # inv is (cartesian,basis)
    inv_cell = torch.linalg.inv(cell)
    coords, wrapped_offset = wrap_points_torch(coords, cell, inv_cell)

    offset_range = compute_offset_range(inv_cell, cutoff)
    # shape: n_permutations, 3 (basis)

    drij = torch.sub(coords.unsqueeze(1), coords.unsqueeze(0))
    # shape n_atoms, n_atoms, 3
    wrap_offset_ij = torch.sub(wrapped_offset.unsqueeze(1), wrapped_offset.unsqueeze(0))

    perm_offsets = offset_range.to(cell.dtype) @ cell
    # shape n_perms, 3 (cartesian)
    perm_offsets = perm_offsets.unsqueeze(1).unsqueeze(2)
    # shape n_perms, 1, 1, 3

    drij = drij.unsqueeze(0)
    # shape 1, n_atoms, n_atoms, 3

    diff = torch.add(drij, perm_offsets)

    dist = torch.linalg.norm(diff, dim=3)
    nonz = torch.nonzero(dist < cutoff)
    po, pf, ps = nonz.unbind(1)
    pi = offset_range[po]

    # Remove connections to self.
    other_image = pi.type(torch.bool).any(dim=1)  # same as (pi != 0).any(dim=1)
    other_particle = torch.ne(pf, ps)  # same as pf != ps
    nonself = torch.logical_or(other_particle, other_image)

    pf = pf[nonself]
    ps = ps[nonself]
    pi = pi[nonself]
    # pi is wrapped image locations, need to convert back to absolute.
    pi = pi - wrap_offset_ij[pf, ps]
    return pf, ps, pi

def neighbor_list_kdtree(cutoff, coords, cell):
    """
    Use KD Tree implementation from scipy.spatial to find pairs under periodic boundary conditions 
    with an orthorhombic cell.
    """
    # Dev note: Imports are cached, this will only be slow once.
    from scipy.spatial import KDTree

    # Verify that cell is orthorhombic
    cell_prod = cell @ cell.T
    if torch.count_nonzero(cell_prod - torch.diag(torch.diag(cell_prod))):
        raise ValueError("KD Tree search only works for orthorhombic cells.")
    
    # Verify that the cutoff is less half the shortest cell side length. Otherwise, it is possible to 
    # have multiple images of the same point be within the cutoff distance of another point. The
    # current algorithm is unable to handle this. 
    cell_side_lengths = torch.sqrt(torch.diag(cell_prod))
    if (cutoff >= cell_side_lengths/2).any():
        raise ValueError(f"Cutoff value ({cutoff}) must be less than half the shortest cell side length ({cell_side_lengths.min()}).")
    
    if torch.count_nonzero(cell - torch.diag(torch.diag(cell))):
        # Transform via isometry to a basis where cell is a diagonal matrix if it currently is not
        new_cell = torch.sqrt(cell_prod)
        new_coords = coords @ torch.linalg.inv(cell) @ new_cell
    else:
        new_cell = cell.clone()
        new_coords = coords.clone()

    new_coords = new_coords % torch.diag(new_cell)

    # The following three lines are included to prevent an extremely rare but not unseen edge 
    # case where the modulo operation returns a particle coordinate that is exactly equal to 
    # the corresponding cell length, causing KDTree to throw an error
    n_particles = new_coords.shape[0]
    tiled_cell = torch.tile(torch.diag(new_cell), (n_particles,)).reshape(n_particles, -1)
    new_coords = torch.where(new_coords == tiled_cell, 0, new_coords)

    new_coords = new_coords.detach().cpu().numpy()
    new_cell = torch.diag(new_cell).detach().cpu().numpy()

    tree = KDTree(
        data=new_coords, 
        boxsize=new_cell,
    )
    
    pairs = tree.query_pairs(r=cutoff, output_type='ndarray')
    pairs = torch.as_tensor(pairs, device=coords.device)
    pair_first, pair_second = torch.unbind(pairs, dim=1)

    # Wrap coordinates into cell and keep track of how they were translated
    inv_cell = torch.linalg.inv(cell)
    coords, wrapped_offset = wrap_points_torch(coords, cell, inv_cell)

    # Find difference vector between pairs without considering the MIC
    pair_diff = torch.sub(coords[pair_first], coords[pair_second])

    # Possible adjacent offset directions for images of the difference vector
    # More is not needed because of the restriction that the cutoff is less than the length of 
    # each side of the cell
    offset_range = torch.tensor(list(product([-1, 0, 1], repeat=3)), device=coords.device)

    # All adjacent offsets
    perm_offsets = offset_range.to(cell.dtype) @ cell
    
    # All adjacent offset images of the difference vector
    pair_diff = pair_diff.unsqueeze(1) + perm_offsets.unsqueeze(0)
    
    # L2 norm of offset images
    pair_diff = torch.linalg.norm(pair_diff, dim=2)

    # Index of shortest offset image
    pair_diff = torch.argmin(pair_diff, dim=1)

    # Offset direction corresponding to shortest offset image plus accounting for the wrapping done earlier
    pair_image = offset_range[pair_diff]
    pair_image -= (wrapped_offset[pair_first] - wrapped_offset[pair_second])

    # KDTree only returns each pair once (eg. (1,2) but not (2,1))
    doubled_pair_first = torch.concat((pair_first, pair_second))
    doubled_pair_second = torch.concat((pair_second, pair_first))
    doubled_pair_image = torch.concat((pair_image, -pair_image))

    return doubled_pair_first, doubled_pair_second, doubled_pair_image


class _DispatchNeighbors(torch.nn.Module):
    def __init__(self, dist_hard_max):
        super().__init__()
        self.dist_hard_max = dist_hard_max

        self.set_combinator(1)
        self.n_images = 1

    def set_combinator(self, n_images):
        self.n_images = n_images

    def compute_one(self, r, c):
        return NotImplemented

    def forward(self, coordinates, nonblank, real_atoms, inv_real_atoms, cell, mol_index, n_molecules, n_atoms_max):

        with torch.no_grad():
            dev = coordinates.device  # where to put the results.

            cell_list = cell.unbind(0)
            coord_list = coordinates.unbind(0)
            nb_list = nonblank.unbind(0)
            sys_inv_atoms = inv_real_atoms.reshape(n_molecules, n_atoms_max).unbind(0)
            nlist_data = []

            for (r, c, nb, sia) in zip(coord_list, cell_list, nb_list, sys_inv_atoms):
                # r: positions
                # c: cell
                # nb: nonblank
                # sia: system inv atom indices, these map from system atom numbers to batch atom numbers

                # Remove blank atoms from positions and indices of atoms.
                r = r[nb]
                sia = sia[nb]
                # dispatch to implementation
                outs = self.compute_one(r, c)
                pf, ps, of = outs

                # convert system-relative atom indices back to whole-batch atom indices
                pf = sia[pf]
                ps = sia[ps]

                # store
                nlist_data.append((pf, ps, of))

            # transpose lists into tensors for whole batch
            try:
                pair_first, pair_second, offsets = zip(*nlist_data)
                pair_first = torch.cat(pair_first).to(dev)
                pair_second = torch.cat(pair_second).to(dev)
                offsets = torch.cat(offsets).to(dev)
                max_images = offsets.abs().max()
                self.set_combinator(max_images)

            # Catch when no neighbors in this whole batch...  very rare?
            except (ValueError, RuntimeError):
                if pair_first.shape[0] != 0:
                    raise
                pair_first = torch.empty(0, dtype=torch.int64, device=dev)
                pair_second = torch.empty(0, dtype=torch.int64, device=dev)
                offsets = torch.empty((0, 3), dtype=torch.int64, device=dev)

            # Number the offsets
            n_off = self.n_images * 2 + 1
            o1, o2, o3 = (offsets + self.n_images).unbind(dim=1)
            offset_index = o3 + n_off * (o2 + n_off * o1)

            pair_mol = mol_index[pair_first]
            pair_cell = cell[pair_mol]
            pair_offsets = torch.bmm(offsets.unsqueeze(1).to(pair_cell.dtype), pair_cell).squeeze(1)

        # now calculate pair_dist, paircoord differentiably
        # print("Pairs found",pair_first.shape)
        coordflat = coordinates.reshape(n_molecules * n_atoms_max, 3)[real_atoms]
        paircoord = coordflat[pair_first] - coordflat[pair_second] + pair_offsets
        distflat = paircoord.norm(dim=1)

        return distflat, pair_first, pair_second, paircoord, offsets, offset_index


class NPNeighbors(_DispatchNeighbors):
    def compute_one(self, positions, cell):
        positions = positions.detach().cpu().numpy()
        cell = cell.detach().cpu().numpy()
        outputs = neighbor_list_np(self.dist_hard_max, positions, cell)
        return [torch.from_numpy(o) for o in outputs]


class TorchNeighbors(_DispatchNeighbors):
    def compute_one(self, positions, cell):
        with torch.no_grad():
            outputs = neighbor_list_torch(self.dist_hard_max, positions, cell)
        return outputs
    
class KDTreeNeighbors(_DispatchNeighbors):
    '''
    Node for finding pairs under periodic boundary conditions using Scipy's KD Tree algorithm. 
    Cell must be orthorhombic.
    '''

    def compute_one(self, positions, cell):
        with torch.no_grad():
            outputs = neighbor_list_kdtree(self.dist_hard_max, positions, cell)
        return outputs
    

class KDTreePairsMemory(PairMemory):
    '''
    Implementation of KDTreePairs with an added memory component.

    Stores current pair indices in memory and reuses them to compute the pair distances if no 
    particle has moved more than skin/2 since last pair calculation. Otherwise uses the
    _pair_indexer_class to recompute the pairs.

    Increasing the value of 'skin' will increase the number of pair distances computed at
    each step, but decrease the number of times new pairs must be computed. Skin should be 
    set to zero while training for fastest results.
    '''

    _pair_indexer_class = KDTreeNeighbors

    def forward(self, coordinates, nonblank, real_atoms, inv_real_atoms, cells, mol_index, n_molecules, n_atoms_max):
        if self.recalculation_needed(coordinates, cells):
            self.recalculations += 1

            inputs = (coordinates, nonblank, real_atoms, inv_real_atoms, cells, mol_index, n_molecules, n_atoms_max)
            outputs = self._pair_indexer(*inputs)
            distflat, pair_first, pair_second, paircoord, offsets, offset_index = outputs

            with torch.no_grad():
                pair_mol = mol_index[pair_first]
                pair_cell = cells[pair_mol]
                pair_offsets = torch.bmm(offsets.unsqueeze(1).to(pair_cell.dtype), pair_cell).squeeze(1)

            for name, var in [
                ("pair_first", pair_first),
                ("pair_second", pair_second),
                ("offsets", offsets),
                ("offset_index", offset_index),
                ("pair_offsets", pair_offsets),
                ("positions", coordinates),
                ("cells", cells),
                ]:
                self.__setattr__(name, var)

        else:
            self.reuses += 1

            coordflat = coordinates.reshape(n_molecules * n_atoms_max, 3)[real_atoms]
            paircoord = coordflat[self.pair_first] - coordflat[self.pair_second] + self.pair_offsets
            distflat = paircoord.norm(dim=1)

        # We filter the lists to only send forward relevant pairs (those with distance under cutoff), improving performance.   
        return filter_pairs(self.hard_dist_cutoff, distflat, self.pair_first, self.pair_second, paircoord, self.offsets, self.offset_index)
