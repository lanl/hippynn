"""
Things to do before training, i.e. initialization of network and diagnostics.
"""

import warnings
import numpy as np
import torch

from .graphs import find_unique_relative, Predictor
from .graphs.nodes.base import _BaseNode
from .graphs.nodes.tags import Encoder
from .graphs.nodes.inputs import SpeciesNode, PositionsNode, CellNode, ForceNode
from .graphs.nodes.pairs import OpenPairIndexer, DynamicPeriodicPairs, MinDistNode
from .graphs.nodes.indexers import acquire_encoding_padding, SysMaxOfAtomsNode
from .graphs.nodes.physics import VecMag
from .networks.hipnn import compute_hipnn_e0


def hierarchical_energy_initialization(
    energy_module,
    database=None,
    trainable_after=False,
    decay_factor=1e-2,
    encoder=None,
    energy_name=None,
    species_name=None,
    peratom=False,
):
    """
    Computes values for the non-interacting energy using the training data.

    :param energy_module:   HEnergyNode or torch module for energy prediction
    :param database:        InterfaceDB object to get training data, required if model contains E0 term
    :param trainable_after: Determines if it should change .requires_grad attribute for the E0 parameters
    :param decay_factor:    change initialized weights of further energy layers by ``df**N`` for layer N
    :param encoder:         species encoder, can be auto-identified from energy node
    :param energy_name:     name for the energy variable, can be auto-identified from energy node
    :param species_name:    name for the species variable, can be auto-identified from energy node
    :param peratom:
    :return: None
    """

    if isinstance(energy_module, _BaseNode):
        if encoder is None:
            encoder = find_unique_relative(energy_module, Encoder, "Constructing E0 Values")
        if species_name is None:
            species_name = find_unique_relative(energy_module, SpeciesNode, "Constructing E0 Values").db_name
        if energy_name is None:
            energy_name = energy_module.main_output.db_name

        energy_module = energy_module.torch_module

    if isinstance(encoder, _BaseNode):
        encoder = encoder.torch_module

    # If model has E0 term, set its initial value using the database provided
    if not energy_module.first_is_interacting:
        if database is None:
            raise ValueError("Database must be provided if model includes E0 energy term.")
    
        train_data = database.splits["train"]

        z_vals = train_data[species_name]
        t_vals = train_data[energy_name]

        encoder.to(t_vals.device)
        eovals = compute_hipnn_e0(encoder, z_vals, t_vals, peratom=peratom)
        eo_layer = energy_module.layers[0]

        if not eo_layer.weight.data.shape[-1] == eovals.shape[-1]:
            raise ValueError("The shape of the computed E0 values does not match the shape expected by the model.")
        
        eo_layer.weight.data = eovals.reshape(1, -1)
        print("Computed E0 energies:", eovals)
        eo_layer.weight.data = eovals.expand_as(eo_layer.weight.data)
        eo_layer.weight.requires_grad_(trainable_after)
    
    # Decay layers E1, E2, etc... according to decay_factor
    for layer in energy_module.layers[1:]:
        layer.weight.data *= decay_factor
        if layer.bias is not None:
            layer.bias.data *= decay_factor
        decay_factor *= decay_factor

def set_e0_values(*args, **kwargs):
    warnings.warn("The function set_e0_values is depreciated. Please use the hierarchical_energy_initialization function instead.")
    return hierarchical_energy_initialization(*args, **kwargs)

def _setup_min_dist_graph(
    species_name,
    positions_name,
    dist_hard_max,
    species_set,
    cell_name=None,
    pair_finder_class="auto",
    device=None,
):
    species = SpeciesNode(db_name=species_name)
    positions = PositionsNode(db_name=positions_name)

    # Species set required to build encoder and atom indexer.
    acquire_encoding_padding(species, species_set=species_set)

    if cell_name is not None:

        cell = CellNode(db_name=cell_name)

        if pair_finder_class == "auto":
            pair_finder_class = DynamicPeriodicPairs
        pair_parents = (positions, species, cell)

    else:
        if pair_finder_class == "auto":
            pair_finder_class = OpenPairIndexer
        pair_parents = (positions, species)

    pair_finder = pair_finder_class("PairFinder", pair_parents, dist_hard_max=dist_hard_max)

    min_dist_mol = MinDistNode("MinDists", pair_finder).min_dist_mol
    pred = Predictor(pair_parents, [min_dist_mol], model_device=device, name="Minimum Distance Calculator")

    return pred, min_dist_mol


def calculate_min_dists(
    array_dict: dict,
    species_name: str,
    positions_name: str,
    dist_hard_max: float,
    cell_name: str = None,
    device: torch.device = None,
    pair_finder_class: _BaseNode = "auto",
    batch_size: int = 50,
):
    """
    Calculates the minimum distance found in each system in ``array_dict``.

    Example usage for unsplit data::

    >>> db = Database(...)
    >>> min_dists = calculate_min_dists(db.arr_dict,"Z","R",5.0)

    If the database has been split::

    >>> min_dists_train = calculate_min_dists(db.splits['train'],"Z","R",5.0)

    Example usage to prune out low-distance data::

    >>> db = Database(...)
    >>> dist_threshold = ...
    >>> min_dist = calculate_min_dists(db.arr_dict,"Z","R",5.0)
    >>> low_distance_system = min_dist < dist_threshold
    >>> db.arr_dict = {k:v[~low_distance_system] for k,v in db.arr_dict.items()}

    .. Note::
       The cutoff radius ``dist_hard_max`` should be set large enough such that each atom is expected
       to have at least one neighbor. If an atom has no neighbors, its min_dist will be set to the
       largest distance found in the current *batch*.
       If an entire system has no neighbors, the minimum distance will be set to zero.

    :param array_dict: dictionary mapping strings to tensors/numpy arrays
    :param species_name: dictionary key for species
    :param positions_name: dictionary key for positions
    :param dist_hard_max: maximum distance to search
    :param cell_name: dictionary key for cell (periodic boundary conditions.
     if the cell is not specified, open boundaries are used.
    :param pair_finder_class: if 'auto', choose automatically. elsewise build this kind of pair finder.
    :param device: Where to perform the computation.
    :param batch_size: batch size to perform evaluation over.
    :return:
    """
    # Check for required info before proceeding to more expensive stuff.
    if species_name not in array_dict:
        raise KeyError(f"Species key {species_name} not in dictionary.")

    if positions_name not in array_dict:
        raise KeyError(f"Positions key {positions_name} not in dictionary.")

    if cell_name is not None:
        if cell_name not in array_dict:
            raise KeyError(f"Cell key {cell_name} not in dictionary.")

    species_set = list(np.unique(array_dict[species_name]))

    if 0 not in species_set:
        species_set = np.concatenate([[0], species_set], axis=0)

    if not len(species_set) or set(species_set) == {0}:
        raise ValueError("Species set empty!")

    pred, node_key = _setup_min_dist_graph(
        species_name=species_name,
        positions_name=positions_name,
        dist_hard_max=dist_hard_max,
        species_set=species_set,
        cell_name=cell_name,
        pair_finder_class=pair_finder_class,
        device=device,
    )

    input_dict = {
        species_name: torch.as_tensor(array_dict[species_name]),
        positions_name: torch.as_tensor(array_dict[positions_name]),
    }
    if cell_name is not None:
        input_dict[cell_name] = torch.as_tensor(array_dict[cell_name])

    results = pred(**input_dict, batch_size=batch_size)[node_key]

    return results


def _setup_max_force_graph(
    species_name,
    force_name,
    species_set,
    device=None,
):
    species = SpeciesNode(db_name=species_name)
    force = ForceNode(db_name=force_name)
    # Species set required to build encoder and atom indexer.
    enc, pad = acquire_encoding_padding(species, species_set=species_set)
    forcemag = VecMag("FMag", (force, species))
    max_force = SysMaxOfAtomsNode("MaxForce", (forcemag, pad))

    pred = Predictor([species, force], [max_force], model_device=device, name="Max Force Calculator")

    return pred, max_force


def calculate_max_system_force(
    array_dict: dict,
    species_name: str,
    force_name: str,
    device: torch.device = None,
    batch_size: int = 50,
):
    """
    Calculates the maximum force magnitude in each system in ``array_dict``.

    Example usage for unsplit data::

    >>> db = Database(...)
    >>> max_force = calculate_max_system_force(db.arr_dict,"Z","F")

    If the database has been split::

    >>> max_force_train = calculate_max_system_force(db.splits['train'],"Z","F")

    Example usage to prune out high-force data::

    >>> db = Database(...)
    >>> force_threshold = ...
    >>> max_force = calculate_max_system_force(db.arr_dict,"Z","F")
    >>> high_force_system = max_force > force_threshold
    >>> db.arr_dict = {k:v[~high_force_system] for k,v in db.arr_dict.items()}

    :param array_dict: dictionary mapping strings to tensors/numpy arrays
    :param species_name: dictionary key for species
    :param force-name: dictionary key for positions
    :param device: Where to perform the computation.
    :param batch_size: batch size to perform evaluation over.
    :return:
    """
    # Check for required info before proceeding to more expensive stuff.
    if species_name not in array_dict:
        raise KeyError(f"Species key {species_name} not in dictionary.")

    if force_name not in array_dict:
        raise KeyError(f"Positions key {force_name} not in dictionary.")

    species_set = list(np.unique(array_dict[species_name]))
    if 0 not in species_set:
        species_set = np.concatenate([[0], species_set], axis=0)

    if not len(species_set) or set(species_set) == {0}:
        raise ValueError("Species set empty!")

    pred, node_key = _setup_max_force_graph(
        species_name=species_name,
        force_name=force_name,
        species_set=species_set,
        device=device,
    )

    input_dict = {
        species_name: torch.as_tensor(array_dict[species_name]),
        force_name: torch.as_tensor(array_dict[force_name]),
    }

    results = pred(**input_dict, batch_size=batch_size)[node_key]

    return results
