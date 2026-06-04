import pytest

import hippynn

from conftest import skip_if_no_models, MODEL_DIR
from conftest import ignore_cusp_warning, ignore_relocation, ignore_weights_only_warning, ignore_sensitivity_warning
import torch



def test_write_nodes(network_parameters):
    from hippynn.experiment.serialization.node_serializer import serialize, _populate_registry
    _populate_registry()

    from hippynn.graphs.nodes import inputs, networks, targets
    # model inputs
    species = inputs.SpeciesNode(db_name="Z")
    positions = inputs.PositionsNode(db_name="R")

    # Model computations
    network = networks.Hipnn("HIPNN", (species, positions), module_kwargs=network_parameters)
    henergy = targets.HEnergyNode("HEnergy", network)
    molecule_energy = henergy.system_energy
    molecule_energy.db_name = "T"

    graph = hippynn.GraphModule([species,positions],[molecule_energy])
    output,tensor_data = serialize(graph)

    print(output)
    return output_tensor_data

@pytest.fixture
def written_nodes(network_parameters):
    output = test_write_nodes(network_parameters)
    return output

def test_read_nodes(written_nodes):
    from hippynn.experiment.serialization.node_serializer import deserialize_nodes
    nodes, tensor_data = written_nodes
    output = deserialize_nodes(written_nodes)
    pass


@ignore_relocation
@ignore_weights_only_warning
@ignore_cusp_warning
@skip_if_no_models
def test_load_old():
    from hippynn.tools import active_directory
    from hippynn.experiment import load_checkpoint_from_cwd

    location = "./quad0_b512_int1_p5_GPU0_seed363144"
    with active_directory(MODEL_DIR / location, create=False):
        check = load_checkpoint_from_cwd(map_location="cpu")

    return


@ignore_weights_only_warning
@ignore_cusp_warning
@ignore_relocation
@ignore_sensitivity_warning
@skip_if_no_models
def test_run_old(example_box):
    from hippynn.tools import active_directory
    from hippynn.experiment import load_checkpoint_from_cwd
    from hippynn.graphs import Predictor

    location = "./quad0_b512_int1_p5_GPU0_seed363144"

    with active_directory(MODEL_DIR / location, create=False):
        check = load_checkpoint_from_cwd(map_location="cpu")

    model = check["training_modules"].model
    predictor = Predictor.from_graph(model)

    renamed_box = dict(
        Z=example_box["species"],
        R=example_box["coordinates"],
    )

    outputs = predictor(**renamed_box)

    return


@ignore_weights_only_warning
@ignore_cusp_warning
@ignore_relocation
@skip_if_no_models
def test_validate_old():

    from optimizer.test_configs import c2h6_config

    floatX = torch.float64
    # fmt: off
    expected_T = torch.tensor([[-566.04203682388606466702],
        [-633.75044483854321697436],
        [-683.59081489815389431897],
        [-706.40015799352158865076],
        [-713.39539090423795641982],
        [-711.64985486498540012690],
        [-705.40931757941268642753],
        [-694.82413721139664630755],
        [-686.65307032058126424090],
        [-677.45483365599477565411],
        [-571.35603221853000377450],
        [-471.07485521951696227916],
        [-485.98350140870610402999],
        [-526.74108660103354395687],
        [-525.06023249181453138590]], dtype=floatX)
    # fmt: on
    torch.set_printoptions(precision=24)
    
    from hippynn.tools import active_directory
    from hippynn.experiment import load_checkpoint_from_cwd
    from hippynn.graphs import Predictor

    location = "./quad0_b512_int1_p5_GPU0_seed363144"

    with active_directory(MODEL_DIR / location, create=False):
        check = load_checkpoint_from_cwd(map_location="cpu")

    model = check["training_modules"].model
    predictor = Predictor.from_graph(model).to(floatX)

    renamed_box = dict(
        Z=c2h6_config["Z"],
        R=c2h6_config["R"].to(floatX),
    )
    outputs = predictor(**renamed_box)
    out_energy = outputs["T"]

    absdiff = (out_energy - expected_T).abs()
    maxreldiff = (absdiff / expected_T).max()
    print(f"({maxreldiff=})")

    assert maxreldiff < 1.e-12, f"Output of network changed. (relative difference = {maxreldiff}, absdiff = {absdiff})"

    return

