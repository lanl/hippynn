import pytest

from hippynn.graphs import inputs, networks, targets, physics


@pytest.fixture()
def neural_network_node(network_parameters):
    species = inputs.SpeciesNode(db_name="species")
    positions = inputs.PositionsNode(db_name="coordinates")
    cell = inputs.CellNode(db_name="cell")

    network = networks.Hipnn("HIPNN", (species, positions, cell), module_kwargs=network_parameters, periodic=True)
    return network


@pytest.fixture
def network_parameters():
    return {
        "possible_species": [0, 13],
        "n_features": 10,
        "n_sensitivities": 20,
        "dist_soft_min": 1.25,
        "dist_soft_max": 7,
        "dist_hard_max": 7.5,
        "n_interaction_layers": 1,
        "n_atom_layers": 3,
        "sensitivity_type": "inverse",
        "resnet": True,
    }


@pytest.mark.parametrize(
    "net_class",
    [
        networks.Hipnn,
        networks.HipnnVec,
        networks.HipnnQuad,
        pytest.param(networks.HipHopnn, marks=pytest.mark.filterwarnings("ignore:.*Beta.*")),
    ],
)
def test_build_network(net_class, network_parameters):
    species = inputs.SpeciesNode(db_name="species")
    positions = inputs.PositionsNode(db_name="coordinates")
    cell = inputs.CellNode(db_name="cell")

    network = net_class("HIPNN", (species, positions, cell), module_kwargs=network_parameters, periodic=True)
    return


@pytest.mark.parametrize(
    "target_cls",
    [
        targets.HEnergyNode,
        targets.HChargeNode,
        targets.AtomizationEnergyNode,
    ],
)
def test_build_atom_target(target_cls, neural_network_node):
    target_node = target_cls("target", neural_network_node)
    return


def test_build_bonds(neural_network_node):
    bond_parameters = {
        "dist_soft_min": 0.9,
        "dist_soft_max": 5.0,
        "dist_hard_max": 5.5,
        "n_dist": 20,
    }
    bonds = targets.HBondNode("bonds", neural_network_node, module_kwargs=bond_parameters)
    return
