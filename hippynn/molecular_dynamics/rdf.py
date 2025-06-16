# Calculate RDF and ADF
# Portions of this code were written with assistance from an LLM

from itertools import combinations_with_replacement

import numpy as np                                   
import torch

from .pbc_tools import find_mic, extract_multicell_diagonal
from ..layers.pairs.indexing import padded_neighlist
from ..tools import progress_bar

try:
    from numba import jit
except ImportError:
    # Dummy jit
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


def ensure_positions_cells_compatibility(positions, cells):
    n_frames, _, _ = positions.shape
    if cells is not None:
        if cells.shape == (3,) or cells.shape == (3,3):
            cells = np.tile(cells, (n_frames, 1))
        elif len(cells) != len(positions):
            raise ValueError(f"Number of frames in `positions` and `cells` do not match. Provided: {len(positions)}, {len(cells)}.")
    return positions, cells

def ensure_positions_species_compatibility(positions, species):
    n_frames, n_particles, _ = positions.shape
    if species is not None:
        if species.squeeze().shape == (n_particles,):
            species = np.tile(species, (n_frames, 1))
        elif species.shape[0] != positions.shape[0]:
            raise ValueError(f"Number of frames in `positions` and `species` do not match. Provided: {positions.shape[0]}, {species.shape[0]}.")
        elif species.shape[1] != positions.shape[1]:
            raise ValueError(f"Number of particles in `positions` and `species` do not match. Provided: {positions.shape[1]}, {species.shape[1]}.")
        species = species.reshape(n_frames, n_particles)
    return positions, species

def check_KDTree_compatibility(cells, cutoff):
    """`cells` must be collapsed to (3,) cell representations"""
    if (cutoff >= cells/2).any():
        raise ValueError(f"Cutoff value ({cutoff}) must be less than half the shortest cell side length ({cells.min()}).")

def get_KDTree_tree(positions, cell):
    """`cells` must be collapsed to (3,) cell representations"""

    # Dev note: Imports are cached, this will only be slow once.
    from scipy.spatial import KDTree

    positions = positions % cell # coordinates must be inside cell

    # The following three lines are included to prevent an extremely rare but not unseen edge 
    # case where the modulo operation returns a particle coordinate that is exactly equal to 
    # the corresponding cell length, causing KDTree to throw an error
    n_particles = positions.shape[0]
    tiled_cell = np.tile(cell, (n_particles, 1))
    positions = np.where(positions == tiled_cell, 0, positions)

    return KDTree(positions, boxsize=cell)

def calculate_rdf(positions: np.ndarray, cutoff: float, cells: np.ndarray = None, species: np.ndarray = None, n_bins: int = 300, lower_cutoff: float = 0):
    """Computes the RDF. If `species` is not provided, also computes species pair specific RDFS. 

    :param positions: Shape (n_frames, n_particles, 3).
    :type positions: np.ndarray
    :param cutoff: Largest pair distance considered.
    :type cutoff: float
    :param cells: Shape (n_frames, 3, 3) or None to if no PBC, defaults to None.
    :type cells: np.ndarray or None, optional
    :param species: Shape (n_frames, n_particles) or (n_particles,), defaults to None.
    :type species: np.ndarray or None, optional
    :param n_bins: Number of bins for RDF(s), defaults to 300.
    :type n_bins: int, optional
    :param lower_cutoff: Smallest pair distance considered, defaults to 0.
    :type lower_cutoff: float, optional

    :return: A tuple containng
        - **bin_centers** (*np.ndarray*): x-values for plotting RDF, shape (n_bins,).
        - **rdf** (*np.ndarray*): y-values for plotting RDF, shape (n_bins,).
        - **species_rdfs** (*dict(str, np.ndarray)*): Only included if `species` is provided. Dictionary where the keys are pairs "rdf_values_i-j" for each
          pair of species i and j, and the values are the RDF y-values for that species pair. The same x-values are used for all RDFs.
    :rtype: tuple
    """

    positions, cells = ensure_positions_cells_compatibility(positions, cells)
    positions, species = ensure_positions_species_compatibility(positions, species)

    if cells is not None: 
        cells = extract_multicell_diagonal(cells)

    check_KDTree_compatibility(cells, cutoff=cutoff)

    bins = np.linspace(lower_cutoff, cutoff, n_bins + 1)
    n_timesteps = len(positions)
    
    counts_running = np.zeros(n_bins)

    if species is not None:
        unique_species = np.unique(species)
        counts_running_species = {f"rdf_values_{i}-{j}": np.zeros(n_bins) for i, j in combinations_with_replacement(unique_species, 2)}

    for i in progress_bar(range(len(positions))):
        cell = (cells[i] if cells is not None else None)

        tree = get_KDTree_tree(positions[i], cell)
        tree_dict = tree.sparse_distance_matrix(tree, cutoff)

        pairs = np.array(list(tree_dict.keys()))
        dists = np.array(list(tree_dict.values()))

        pairs = pairs[np.where(dists > lower_cutoff)]
        dists = dists[np.where(dists > lower_cutoff)]

        counts, _ = np.histogram(dists, bins=bins)
        counts_running += counts

        if species is not None:
            for j, k in combinations_with_replacement(unique_species, 2):
                dists_spec = dists[(species[i][pairs[:,0]] == j) & (species[i][pairs[:,1]] == k)]
                counts, _ = np.histogram(dists_spec, bins=bins)
                counts_running_species[f'rdf_values_{j}-{k}'] += counts
                if j!=k:
                    dists_spec = dists[(species[i][pairs[:,0]] == k) & (species[i][pairs[:,1]] == j)]
                    counts, _ = np.histogram(dists_spec, bins=bins)
                    counts_running_species[f'rdf_values_{j}-{k}'] += counts

    # Overall values to normalize by
    avg_n_pairs = counts_running.sum() / n_timesteps
    sphere_vol = cutoff**3
    avg_density = avg_n_pairs / sphere_vol

    # Specific values to each bin/shell
    # Volume of shell with radii from 0 to cutoff and thickness cutoff/n_bins
    shell_vols = np.power(bins[1:], 3) - np.power(bins[:-1], 3)
    avg_counts = counts_running / n_timesteps
    shell_densities = avg_counts / shell_vols

    # RDF ratio
    rdf = shell_densities / avg_density

    if species is not None:
        species_rdfs = dict()
        for j, k in combinations_with_replacement(unique_species, 2):
            avg_n_pairs = counts_running_species[f'rdf_values_{j}-{k}'].sum() / n_timesteps
            avg_density = avg_n_pairs / sphere_vol

            avg_counts = counts_running_species[f'rdf_values_{j}-{k}'] / n_timesteps
            shell_densities = avg_counts / shell_vols

            species_rdfs[f'rdf_values_{j}-{k}'] = shell_densities / avg_density

    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    return (bin_centers, rdf) if species is None else (bin_centers, rdf, species_rdfs)

@jit(nopython=True)
def find_triples(jlist_pad, rijlist_pad):
    n_centers, max_pairs = jlist_pad.shape
    max_triples = max_pairs * (max_pairs + 1) // 2

    triples = np.zeros((n_centers, max_triples, 2), dtype=np.int64) - 1
    triples_vec = np.zeros((n_centers, max_triples, 2, 3), dtype=np.float64)
    for j in range(n_centers):
        idx = 0
        row = jlist_pad[j]
        row_vec = rijlist_pad[j]
        n_real = np.count_nonzero(row + 1)
        for k in range(n_real):
            for l in range(k+1, n_real):
                triples[j, idx, 0] = row[k]
                triples[j, idx, 1] = row[l]
                triples_vec[j, idx, 0] = row_vec[k]
                triples_vec[j, idx, 1] = row_vec[l]
                idx += 1

    return triples, triples_vec

def calculate_adf(positions, cutoffs, cells=None, species=None):
    """
    Computes ADFs for given cutoffs. If `species` is not provided, also computes species triple specific ADFs. 

    This code will be quite slow if numba is not available.

    :param positions: Shape (n_frames, n_particles, 3).
    :type positions: np.ndarray
    :param cutoff: Maximum arm lengths on angles to consider.
    :type cutoff: list[float]
    :param cells: Shape (n_frames, 3, 3) or (3, 3), no PBC if None, defaults to None.
    :type cells: np.ndarray or None, optional
    :param species: Shape (n_frames, n_particles) or (n_particles,), defaults to None.
    :type species: np.ndarray or None, optional

    :return: A dictionary with keys
        - **f"adf_values_all-all-all_cutoff_{cutoff}"** (*np.ndarray*): y-values for plotting ADF of all positions for each value `cutoff` in `cutoffs`, shape (180,).
        - **f"adf_values_{center}-{end1}-{end2}_cutoff_{cutoff}"** (*np.ndarray*): Available only if `species` was provided. y-values for plotting ADF of all angles 
          with center of type `center` and ends of species `end1` and `end2` for all possible combinations triples of species species, and for each value 
          `cutoff` in `cutoffs`, shape (180,).
    :rtype: dict
    """

    positions, cells = ensure_positions_cells_compatibility(positions, cells)
    positions, species = ensure_positions_species_compatibility(positions, species)

    if cells is not None: 
        cells = extract_multicell_diagonal(cells)

    try:
        iter(cutoffs)
    except TypeError:
        cutoffs = [cutoffs]

    check_KDTree_compatibility(cells, max(cutoff))

    adfs = dict()
    for cutoff in cutoffs:
        adfs[f"adf_values_all-all-all_cutoff_{cutoff}"] = np.zeros((180,))

    if species is not None:
        unique_species = np.unique(species)
        for center in unique_species:
            for end1, end2 in combinations_with_replacement(unique_species, 2):
                for cutoff in cutoffs:
                    adfs[f"adf_values_{center}-{end1}-{end2}_cutoff_{cutoff}"] = np.zeros((180,))
        

    for i in progress_bar(range(len(positions))):
        cell = (cells[i] if cells is not None else None)

        tree = get_KDTree_tree(positions[i], cell)
        pairs = tree.query_pairs(max(cutoffs), output_type='ndarray')

        if len(pairs) == 0:
            continue

        vecs = find_mic(positions[i][pairs[:,0]] - positions[i][pairs[:,1]], cell=cell)
        vecs = np.array(vecs)

        vecs = vecs

        pairs = np.concatenate((pairs, pairs[:,::-1]))
        vecs = np.concatenate((vecs, -vecs))

        jlist_pad, rijlist_pad = padded_neighlist(torch.as_tensor(pairs[:,0]), torch.as_tensor(pairs[:,1]), torch.as_tensor(vecs), torch.as_tensor(positions[i]))
        jlist_pad, rijlist_pad = np.asarray(jlist_pad), np.asarray(rijlist_pad)

        triples, triples_vec = find_triples(jlist_pad, rijlist_pad)

        triples_dists = np.linalg.norm(triples_vec, axis=-1)

        triples_vec = triples_vec / triples_dists[...,None]

        for cutoff in cutoffs:
            mask = (triples_dists[:, :, 0] <= cutoff) & (triples_dists[:, :, 1] <= cutoff)
            triples_vec_cutoff = triples_vec[mask]

            angles = np.arccos((triples_vec_cutoff[:,0] * triples_vec_cutoff[:,1]).sum(axis=-1)) / np.pi * 180 
            counts, _ = np.histogram(angles, bins=np.arange(0,181,dtype=angles.dtype))

            adfs[f"adf_values_all-all-all_cutoff_{cutoff}"] += counts

        if species is not None:
            for center in unique_species:
                mask1 = species[i] == center
                jlist_pad_center = jlist_pad[mask1]
                rijlist_pad_center = rijlist_pad[mask1]

                triples, triples_vec = find_triples(jlist_pad_center, rijlist_pad_center)

                for end1, end2 in combinations_with_replacement(unique_species, 2):
                    mask2 = (triples != -1)
                    triples_species = np.zeros_like(triples) - 1
                    triples_species[mask2] = species[i][triples[mask2]]  

                    target1 = (triples_species[:, :, 0] == end1) & (triples_species[:, :, 1] == end2)
                    target2 = (triples_species[:, :, 0] == end2) & (triples_species[:, :, 1] == end1)
                    mask3 = np.where(target1 | target2)                  

                    triples_ends = triples_vec[mask3]
                    triples_dists = np.linalg.norm(triples_ends, axis=-1)

                    triples_ends = triples_ends / triples_dists[...,None]

                    for cutoff in cutoffs:
                        mask4 = (triples_dists[:, 0] <= cutoff) & (triples_dists[:, 1] <= cutoff)
                        triples_ends_cutoff = triples_ends[mask4]

                        angles = np.arccos((triples_ends_cutoff[:,0] * triples_ends_cutoff[:,1]).sum(axis=-1)) / np.pi * 180 
                        counts, _ = np.histogram(angles, bins=np.arange(0,181,dtype=angles.dtype))

                        adfs[f"adf_values_{center}-{end1}-{end2}_cutoff_{cutoff}"] += counts

    for key, value in adfs.items():
        adfs[key] = value / value.sum()

    if species is not None:
        for center in unique_species:
            for end1, end2 in combinations_with_replacement(unique_species, 2):
                for cutoff in cutoffs:
                    adfs[f"adf_values_{center}-{end2}-{end1}_cutoff_{cutoff}"] = adfs[f"adf_values_{center}-{end1}-{end2}_cutoff_{cutoff}"]
        
    return adfs