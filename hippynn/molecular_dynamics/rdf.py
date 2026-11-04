# Calculate RDF and ADF
# Portions of this code were written with assistance from an LLM

from itertools import combinations_with_replacement

import numpy as np

from .pbc_tools import find_mic, extract_multicell_diagonal
from ..tools import progress_bar

def ensure_positions_cells_compatibility(positions, cells):
    n_frames, _, _ = positions.shape
    if cells is None:
        raise ValueError("Periodic boundary conditions are required: `cells` must not be None.")
    if cells.shape == (3,) or cells.shape == (3,3):
        cells = np.tile(cells, (n_frames, 1, 1))
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

def calculate_rdf(positions: np.ndarray, cutoff: float, cells: np.ndarray, species: np.ndarray = None, n_bins: int = 300, lower_cutoff: float = 0):
    """Computes the RDF. If `species` is provided, also computes species pair specific RDFS.

    :param positions: Shape (n_frames, n_particles, 3).
    :type positions: np.ndarray
    :param cutoff: Largest pair distance considered.
    :type cutoff: float
    :param cells: Shape (n_frames, 3, 3) or (3, 3). 
    :type cells: np.ndarray
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
    bin_width = (cutoff - lower_cutoff) / n_bins
    
    counts_running = np.zeros(n_bins)
    norm_running = 0.0  # sum over frames of N^2 / V

    if species is not None:
        unique_species = np.unique(species)
        n_species = len(unique_species)
        species_pairs = list(combinations_with_replacement(unique_species, 2))
        counts_running_species = {f"rdf_values_{j}-{k}": np.zeros(n_bins) for j, k in species_pairs}
        norm_running_species = {f"rdf_values_{j}-{k}": 0.0 for j, k in species_pairs}

    for i in progress_bar(range(len(positions))):
        cell = cells[i]
        volume = np.prod(cell)

        tree = get_KDTree_tree(positions[i], cell)
        pair_data = tree.sparse_distance_matrix(tree, cutoff, output_type="ndarray")

        keep = pair_data["v"] > lower_cutoff
        pair_first, pair_second, dists = pair_data["i"][keep], pair_data["j"][keep], pair_data["v"][keep]

        counts, _ = np.histogram(dists, bins=bins)
        counts_running += counts
        norm_running += len(positions[i]) ** 2 / volume

        if species is not None:
            # Assign each distance to its bin (the bins are uniform, so divide, then adjust at the edges to match np.histogram).
            bin_index = np.clip(((dists - lower_cutoff) / bin_width).astype(np.int64), 0, n_bins - 1)
            bin_index -= dists < bins[bin_index]
            bin_index += (dists >= bins[bin_index + 1]) & (bin_index < n_bins - 1)
            # Count distances by species pair and bin.
            species_index = np.searchsorted(unique_species, species[i])
            pair_code = species_index[pair_first] * n_species + species_index[pair_second]
            counts_all = np.bincount(pair_code * n_bins + bin_index, minlength=n_species**2 * n_bins).reshape(n_species, n_species, n_bins)
            for j, k in species_pairs:
                key = f"rdf_values_{j}-{k}"
                a, b = np.searchsorted(unique_species, [j, k])
                counts_running_species[key] += counts_all[a, b]
                norm_running_species[key] += (species[i] == j).sum() * (species[i] == k).sum() / volume

    # Volume of each spherical shell
    shell_vols = 4 / 3 * np.pi * (np.power(bins[1:], 3) - np.power(bins[:-1], 3))

    rdf = counts_running / (norm_running * shell_vols)

    if species is not None:
        species_rdfs = {
            key: counts_running_species[key] / (norm_running_species[key] * shell_vols) for key in counts_running_species
        }

    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    return (bin_centers, rdf) if species is None else (bin_centers, rdf, species_rdfs)

def calculate_adf(positions, cutoffs, cells, species=None):
    """
    Computes ADFs for given cutoffs. If `species` is not provided, also computes species triple specific ADFs. 

    :param positions: Shape (n_frames, n_particles, 3).
    :type positions: np.ndarray
    :param cutoffs: Maximum arm lengths on angles to consider.
    :type cutoffs: list[float]
    :param cells: Shape (n_frames, 3, 3) or (3, 3).
    :type cells: np.ndarray
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

    cells = extract_multicell_diagonal(cells)

    try:
        iter(cutoffs)
    except TypeError:
        cutoffs = [cutoffs]

    check_KDTree_compatibility(cells, max(cutoffs))

    adfs = dict()
    for cutoff in cutoffs:
        adfs[f"adf_values_all-all-all_cutoff_{cutoff}"] = np.zeros((180,))

    if species is not None:
        unique_species = np.unique(species)
        n_species = len(unique_species)
        for center in unique_species:
            for end1, end2 in combinations_with_replacement(unique_species, 2):
                for cutoff in cutoffs:
                    adfs[f"adf_values_{center}-{end1}-{end2}_cutoff_{cutoff}"] = np.zeros((180,))
        

    for i in progress_bar(range(len(positions))):
        cell = cells[i]

        # Find all pairs of particles within the largest cutoff.
        tree = get_KDTree_tree(positions[i], cell)
        pairs = tree.query_pairs(max(cutoffs), output_type='ndarray')

        if len(pairs) == 0:
            continue

        cell = np.diag(cell)
        vecs = find_mic(positions[i][pairs[:,1]] - positions[i][pairs[:,0]], cell=cell)

        # Build the neighbor list: list each pair in both directions, grouped by center, with the vector from center to neighbor.
        centers = np.concatenate((pairs[:, 0], pairs[:, 1]))
        neighbors = np.concatenate((pairs[:, 1], pairs[:, 0]))
        vecs = np.concatenate((vecs, -vecs))
        order = np.argsort(centers)
        centers, neighbors, vecs = centers[order], neighbors[order], vecs[order]

        # Enumerate the triples: for each center, every pair of its neighbors (n neighbors give n(n-1)/2 triples).
        n_neigh = np.bincount(centers, minlength=len(positions[i]))
        row_start = np.concatenate(([0], np.cumsum(n_neigh)[:-1]))
        if not (n_neigh >= 2).any():
            continue
        first, second, center_of_triple = [], [], []
        for c in range(len(n_neigh)):
            k, l = np.triu_indices(n_neigh[c], 1)
            first.append(row_start[c] + k)
            second.append(row_start[c] + l)
            center_of_triple.append(np.full(len(k), c))
        first, second, center_of_triple = np.concatenate(first), np.concatenate(second), np.concatenate(center_of_triple)

        # Calculate the angle between the two arms of each triple, and the arm lengths.
        v1, v2 = vecs[first].astype(np.float64, copy=False), vecs[second].astype(np.float64, copy=False)
        d1, d2 = np.linalg.norm(v1, axis=-1), np.linalg.norm(v2, axis=-1)
        cosines = np.clip((v1 / d1[:, None] * (v2 / d2[:, None])).sum(axis=-1), -1.0, 1.0)
        angles = np.arccos(cosines) / np.pi * 180
        # Assign each angle to a 1-degree bin (same adjustment at the edges as in the RDF).
        angle_bin = np.minimum(np.floor(angles).astype(np.int64), 179)
        angle_bin -= angles < angle_bin
        angle_bin += (angles >= angle_bin + 1) & (angle_bin < 179)

        longest_arm = np.maximum(d1, d2)

        # Label each triple by the species of its center and its two arms (the order of the arms does not matter).
        if species is not None:
            species_index = np.searchsorted(unique_species, species[i])
            center_index = species_index[center_of_triple]
            arm1, arm2 = species_index[neighbors[first]], species_index[neighbors[second]]
            lo, hi = np.minimum(arm1, arm2), np.maximum(arm1, arm2)
            species_key = (center_index * n_species + lo) * n_species + hi

        # Count the angles in each bin for every cutoff, for all triples and for each species combination (a triple counts if both arms are within the cutoff).
        for cutoff in cutoffs:
            inside = longest_arm <= cutoff
            adfs[f"adf_values_all-all-all_cutoff_{cutoff}"] += np.bincount(angle_bin[inside], minlength=180)
            if species is not None:
                counts = np.bincount(species_key[inside] * 180 + angle_bin[inside], minlength=n_species**3 * 180).reshape(n_species, n_species, n_species, 180)
                for center_idx, center in enumerate(unique_species):
                    for end1_idx, end2_idx in combinations_with_replacement(range(n_species), 2):
                        end1, end2 = unique_species[end1_idx], unique_species[end2_idx]
                        adfs[f"adf_values_{center}-{end1}-{end2}_cutoff_{cutoff}"] += counts[center_idx, end1_idx, end2_idx]

    for key, value in adfs.items():
        adfs[key] = value / value.sum()

    if species is not None:
        for center in unique_species:
            for end1, end2 in combinations_with_replacement(unique_species, 2):
                for cutoff in cutoffs:
                    adfs[f"adf_values_{center}-{end2}-{end1}_cutoff_{cutoff}"] = adfs[f"adf_values_{center}-{end1}-{end2}_cutoff_{cutoff}"]
        
    return adfs
