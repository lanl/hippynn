import pytest

import hippynn
import ase

from conftest import skip_if_no_models, MODEL_DIR
from conftest import ignore_cusp_warning, ignore_relocation, ignore_weights_only_warning


def skip_if_no_lammps(func):
    try:
        import lammps
    except ImportError:  # missing lammps doesn't raise ImportError per se.  ModuleNotFoundError.
        # Something went wrong importing!
        wrapper = pytest.mark.skip(reason="lammps is not installed")
    else:
        # Importing
        wrapper = lambda f: f
    return wrapper(func)


def test_build_training_modules(energy_node):

    mae = hippynn.loss.MAELoss.of_node(energy_node)

    validation_losses = {"MAE": mae}

    training_modules, db_info = hippynn.experiment.assemble_for_training(mae, validation_losses)


# Mark as xfail because lammps python installations require manual steps.
@skip_if_no_lammps
def test_build_lammps_interface(energy_node):
    from hippynn.interfaces.lammps_interface import MLIAPInterface

    interface = MLIAPInterface(energy_node, element_types=[1])


def test_build_ase_interface(energy_node):
    from hippynn.interfaces.ase_interface import HippynnCalculator

    calc = HippynnCalculator(energy_node)


@ignore_weights_only_warning
@ignore_cusp_warning
@ignore_relocation
@skip_if_no_models
def test_build_ensemble_auto():

    from hippynn.graphs import make_ensemble

    location = str((MODEL_DIR / "quad*").resolve())
    import warnings

    ensemble_graph, ensemble_info = make_ensemble(location)

    assert ensemble_info == ({"Z": 16, "R": 16}, {"T": 16, "Grad": 16})


NODE_TO_ENSEMBLIZE = "HEnergy.atom_energies"


@pytest.fixture
def example_ensemble_graphs():
    from hippynn.graphs.ensemble import get_graphs

    location = str((MODEL_DIR / "quad*").resolve())
    graphs = get_graphs(location)

    nodes = [g.unique_node_from_name(NODE_TO_ENSEMBLIZE) for g in graphs]
    for n in nodes:
        n.db_name = "AE"

    return graphs


@ignore_weights_only_warning
@ignore_cusp_warning
@ignore_relocation
@skip_if_no_models
def test_build_ensemble_nodes(example_ensemble_graphs):
    from hippynn.graphs import make_ensemble

    nodes = [g.unique_node_from_name(NODE_TO_ENSEMBLIZE) for g in example_ensemble_graphs]

    ensemble_graph, ensemble_info = make_ensemble(nodes)

    assert ensemble_info == ({"Z": 16, "R": 16}, {"AE": 16})


@ignore_weights_only_warning
@ignore_cusp_warning
@ignore_relocation
@skip_if_no_models
def test_build_ensemble_target_graphs(example_ensemble_graphs):
    from hippynn.graphs import make_ensemble

    ensemble_graph, ensemble_info = make_ensemble(example_ensemble_graphs, targets=[NODE_TO_ENSEMBLIZE])

    assert ensemble_info == ({"Z": 16, "R": 16}, {NODE_TO_ENSEMBLIZE: 16})


@ignore_weights_only_warning
@ignore_cusp_warning
@ignore_relocation
@skip_if_no_models
def test_build_ensemble_target_nodes(example_ensemble_graphs):
    from hippynn.graphs import make_ensemble

    # Deliberately get a graph that is not exactly the right one.
    nodes = [g.unique_node_from_name("HEnergy") for g in example_ensemble_graphs]

    ensemble_graph, ensemble_info = make_ensemble(nodes, targets=[NODE_TO_ENSEMBLIZE])

    assert ensemble_info == ({"Z": 16, "R": 16}, {NODE_TO_ENSEMBLIZE: 16})

    pass
