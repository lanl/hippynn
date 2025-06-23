import pytest

import torch
import hippynn


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


@pytest.fixture
def example_all_target_nodes(neural_network_node, bond_parameters):
    from hippynn.graphs import inputs, targets, physics, GraphModule

    network = neural_network_node
    henergy = targets.HEnergyNode("E", network)
    aenergy = targets.AtomizationEnergyNode("T", network)
    hcharge = targets.HChargeNode("C", network)
    bonds = targets.HBondNode("B", network, module_kwargs=bond_parameters)

    dipole = physics.DipoleNode("dipole", hcharge)
    quadrupole = physics.QuadrupoleNode("quadrupole", hcharge)

    from hippynn.graphs import find_unique_relative

    positions = find_unique_relative(network, inputs.PositionsNode)

    force_h = physics.GradientNode("F_E", (henergy, positions), sign=-1)
    force_a = physics.GradientNode("F_T", (aenergy, positions), sign=-1)

    all_targets = [henergy, aenergy, hcharge, bonds, dipole, quadrupole, force_h, force_a]
    quadrupole._index_state = hippynn.graphs.IdxType.Molecules  # hack to avoid db_form conversion

    return all_targets


sensitivity_warning_supression = pytest.mark.filterwarnings("ignore:.*underneath sensitivity range*.")


@sensitivity_warning_supression
def test_run_targets(example_all_target_nodes, example_box):
    from hippynn.graphs import GraphModule, find_relatives, inputs

    targets = example_all_target_nodes

    graph_inputs = find_relatives(targets, inputs.InputNode)

    graph = GraphModule(graph_inputs, targets)

    input_values = [example_box[node.db_name] for node in graph_inputs]

    output_values = graph(*input_values)

    return


def test_build_predictor(example_all_target_nodes):

    from hippynn.graphs import GraphModule, Predictor, find_relatives, inputs

    # some children of the targets may not yet have suitable IdxType labels.
    targets = [t.main_output for t in example_all_target_nodes]
    graph_inputs = find_relatives(targets, inputs.InputNode)
    graph = GraphModule(graph_inputs, targets)
    predictor = Predictor.from_graph(graph)


@sensitivity_warning_supression
def test_run_predictor(example_all_target_nodes, example_box):
    from hippynn.graphs import GraphModule, Predictor, find_relatives, inputs

    # some children of the targets may not yet have suitable IdxType labels.
    targets = [t.main_output for t in example_all_target_nodes]
    graph_inputs = find_relatives(targets, inputs.InputNode)
    graph = GraphModule(graph_inputs, targets)
    predictor = Predictor.from_graph(graph)

    outputs = predictor(**example_box)
