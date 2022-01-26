"""
Things to do before training, i.e. initialization of network and diagnostics.
"""

import numpy as np
import torch

from .graphs import find_unique_relative, Predictor
from .graphs.nodes.base import _BaseNode
from .graphs.nodes.tags import Encoder
from .graphs.nodes.inputs import SpeciesNode, PositionsNode, CellNode
from .graphs.nodes.pairs import OpenPairIndexer, DynamicPeriodicPairs, MinDistNode
from .graphs.nodes.indexers import acquire_encoding_padding
from .networks.hipnn import compute_hipnn_e0


def set_e0_values(
    energy_module,
    database,
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
    :param database:        InterfaceDB object to get training data
    :param trainable_after: Determines if it should change .requires_grad attribute for the E0 parameters.
    :param decay_factor:    change initialized weights of further energy layers by ``df**N`` for layer N
    :param network_module:  network for running the species encoding. Can be auto-identified from energy node
    :param energy_name:     name for the energy variable, can be auto-identified from energy node
    :param species_name:    name for the species variable, can be auto-identified from energy node
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

    train_data = database.splits["train"]

    z_vals = train_data[species_name]
    t_vals = train_data[energy_name]

    eovals = compute_hipnn_e0(encoder, z_vals, t_vals, peratom=peratom)
    eo_layer = energy_module.layers[0]
    print("Computed E0 energies:", eovals)
    eo_layer.weight.data = eovals.expand_as(eo_layer.weight.data)

    eo_layer.weight.requires_grad_(trainable_after)
    for layer in energy_module.layers[1:]:
        layer.weight.data *= decay_factor
        layer.bias.data *= decay_factor
        decay_factor *= decay_factor


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

    if not species_set or set(species_set) == {0}:
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
