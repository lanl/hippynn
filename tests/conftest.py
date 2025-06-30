import pytest

import torch
import hippynn


@pytest.fixture
def network_parameters():
    return {
        "possible_species": [0, 1],
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


@pytest.fixture
def bond_parameters():
    bond_parameters = {
        "dist_soft_min": 0.8,
        "dist_soft_max": 5.0,
        "dist_hard_max": 5.5,
        "n_dist": 20,
    }
    return bond_parameters


@pytest.fixture()
def neural_network_node(network_parameters):
    from hippynn.graphs import inputs, networks

    species = inputs.SpeciesNode(db_name="species")
    positions = inputs.PositionsNode(db_name="coordinates")
    cell = inputs.CellNode(db_name="cell")

    network = networks.Hipnn("HIPNN", (species, positions, cell), module_kwargs=network_parameters, periodic=True)

    return network


@pytest.fixture()
def energy_model(neural_network_node):
    henergy = hippynn.targets.HEnergyNode("Energy", neural_network_node, db_name="T")
    return henergy


@pytest.fixture
def example_box():
    n_atom = 7
    batch_size = 5
    n_dim = 3
    l = 2
    z = torch.ones((batch_size, n_atom), dtype=torch.int64)
    r = l * torch.rand((batch_size, n_atom, n_dim), dtype=torch.float)
    c = l * torch.eye(n_dim, dtype=torch.float).unsqueeze(0).expand((batch_size, n_dim, n_dim))
    return {"species": z, "coordinates": r, "cell": c}  # must match names in neural_network_node
