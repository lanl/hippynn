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

@pytest.fixture
def example_database(example_box):
    from hippynn.databases.database import Database

    n_duplicates = 10
    duplicated_box = {}
    for k, v in example_box.items():
        repeats = [1 for _ in v.shape]
        repeats[0] = n_duplicates
        duplicated_box[k] = v.repeat(repeats)

    # Determine input names from the example box keys.
    input_names = list(example_box.keys())
    # Start with no targets; they will be added later.
    database = Database(
        arr_dict=duplicated_box,
        inputs=input_names,
        targets=[],
        seed=0,
        quiet=False,
    )
    return database

@pytest.fixture
def example_training_setup(energy_node, example_database):
    """Prepare training modules and a database.
    """
    from hippynn.graphs.nodes.loss import MAELoss
    from hippynn.experiment.assembly import assemble_for_training
    from hippynn.experiment.routines import SetupParams
    import torch

    loss_node = MAELoss.of_node(energy_node)
    training_modules, db_info = assemble_for_training(
        train_loss=loss_node, validation_losses={"val": loss_node}
    )

    # Add a random target column matching the expected shape.
    en_name = energy_node.main_output.db_name
    batch_size = example_database.arr_dict["species"].shape[0]
    example_database.arr_dict[en_name] = torch.randn(batch_size, 1)
    example_database.targets = [en_name]

    # Align the database with the inputs/targets expected by the training modules.
    example_database.align(**db_info)
    example_database.make_trainvalidtest_split(test_size=0.2, valid_size=0.2)

    setup_params = SetupParams(batch_size=1, max_epochs=1, stopping_key="val", learning_rate=0.001)
    return training_modules, example_database, setup_params


@pytest.fixture
def temporary_directory():

    import tempfile
    
    temp_name = "TEST_TMP_DIRECTORY"
    with tempfile.TemporaryDirectory(prefix=temp_name) as tempdir:
        with hippynn.active_directory(tempdir, create=False):
            yield tempdir   # test runs here.

