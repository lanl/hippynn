import pytest

import torch
import hippynn


from pathlib import Path

MODEL_DIR = Path(__file__).parents[2] / "collected_models"


skip_if_no_models = pytest.mark.skipif(not MODEL_DIR.exists(), reason="test model resources not found")


ignore_relocation = pytest.mark.filterwarnings("ignore:.*HIPPYNN_DEPRECATION_WARNINGS=ignore*.")
ignore_weights_only_warning = pytest.mark.filterwarnings("ignore:.*weights_only=False*.")
ignore_cusp_warning = pytest.mark.filterwarnings("ignore:.*'cusp_reg' parameter*.")
ignore_sensitivity_warning = pytest.mark.filterwarnings("ignore:.*underneath sensitivity range*.")


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
def input_nodes():
    from hippynn.graphs import inputs

    species = inputs.SpeciesNode(db_name="species")
    positions = inputs.PositionsNode(db_name="coordinates")
    cell = inputs.CellNode(db_name="cell")

    return species, positions, cell


@pytest.fixture()
def neural_network_node(input_nodes, network_parameters):
    from hippynn.graphs import networks

    return networks.Hipnn("HIPNN", input_nodes, module_kwargs=network_parameters, periodic=True)


@pytest.fixture()
def energy_node(neural_network_node):
    henergy = hippynn.targets.HEnergyNode("Energy", neural_network_node, db_name="T")
    return henergy


@pytest.fixture()
def energy_graph(input_nodes, energy_node):
    from hippynn.graphs import GraphModule

    graph = GraphModule(required_inputs=input_nodes, nodes_to_compute=(energy_node,))

    return graph


@pytest.fixture
def example_box():
    n_atom = 7
    batch_size = 5
    n_dim = 3
    l = 2
    z = torch.ones((batch_size, n_atom), dtype=torch.int64)
    r = l * torch.rand((batch_size, n_atom, n_dim), dtype=torch.float)
    c = l * torch.eye(n_dim, dtype=torch.float).unsqueeze(0).expand((batch_size, n_dim, n_dim))
    return {"species": z, "coordinates": r, "cell": c}  # must match names in input_nodes
