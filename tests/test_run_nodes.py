import pytest

import torch
import hippynn
from conftest import ignore_sensitivity_warning


@pytest.fixture
def example_all_target_nodes(neural_network_node, bond_parameters):
    from hippynn.graphs import inputs, targets, physics

    network = neural_network_node
    henergy = targets.HEnergyNode("E", network)
    aenergy = targets.AtomizationEnergyNode("T", network)
    hcharge = targets.HChargeNode("C", network)
    
    
    bonds = targets.HBondNode("B", network, module_kwargs=bond_parameters)

    dipole = physics.DipoleNode("dipole", hcharge)
    quadrupole = physics.QuadrupoleNode("quadrupole", hcharge)
    cheq = physics.ChEQNode("c", network).dipole

    from hippynn.graphs import find_unique_relative

    positions = find_unique_relative(network, inputs.PositionsNode)

    force_h = physics.GradientNode("F_E", (henergy, positions), sign=-1)
    force_a = physics.GradientNode("F_T", (aenergy, positions), sign=-1)
    stress_s, force_s = physics.setup_stressforce_nodes(henergy)
    

    all_targets = [henergy, aenergy, hcharge, bonds, dipole, quadrupole, cheq, force_h, force_a, force_s, stress_s]
    quadrupole.index_state = hippynn.graphs.IdxType.Systems  # hack to avoid db_form conversion

    return all_targets


@pytest.mark.parametrize("operation", ["add", "sub", "mul", "truediv", "pow"])
def test_node_algebra(operation):
    from hippynn.graphs.nodes.base import ValueNode
    from hippynn.graphs import GraphModule
    import operator

    func = getattr(operator, operation)
    a = 2
    b = 3
    out = func(a, b)

    a_val = ValueNode(2)
    b_val = ValueNode(3)
    out_val = func(a_val, b_val)

    graph = GraphModule([], [out_val])
    out_check = graph()[0]

    assert out == out_check, f"Values not equal! {out} {operation} {out_check}"
    return


@ignore_sensitivity_warning
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


@ignore_sensitivity_warning
def test_run_predictor(example_all_target_nodes, example_box):
    from hippynn.graphs import GraphModule, Predictor, find_relatives, inputs

    # some children of the targets may not yet have suitable IdxType labels.
    targets = [t.main_output for t in example_all_target_nodes]
    graph_inputs = find_relatives(targets, inputs.InputNode)
    graph = GraphModule(graph_inputs, targets)
    predictor = Predictor.from_graph(graph)

    outputs = predictor(**example_box)


@ignore_sensitivity_warning
def test_atomization_conversion(example_box, neural_network_node):
    from hippynn.graphs import targets, base

    energy = targets.AtomizationEnergyNode("HEnergy", neural_network_node, db_name="T")

    hen_equivalent = energy.create_henergy_equivalent()

    input_nodes = energy.find_relatives(base.InputNode)
    model = hippynn.GraphModule(input_nodes, [energy.system_energy, hen_equivalent.system_energy])

    args = [example_box[node.db_name] for node in input_nodes]
    
    # convert dtypes for better precision
    args = [a.to(torch.float64 if a.dtype.is_floating_point else a.dtype) for a in args]
    model = model.to(torch.float64)

    en_1, en_2 = model(*args)

    assert torch.allclose(en_1, en_2)

    return
