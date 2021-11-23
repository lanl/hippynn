"""
initialization of network
"""

from .graphs import find_unique_relative
from .graphs.nodes.base import _BaseNode
from hippynn.graphs.nodes.tags import Encoder
from .graphs.nodes.inputs import SpeciesNode
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
    :param decay_factor:    change initialized weights of further energy layers by df**N for layer N
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
