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


@pytest.fixture
def explicit_edge_network_params():
    return {
        "possible_species": [0, 1],
        "n_features": 4,
        "n_sensitivities": 6,
        "dist_soft_min": 0.5,
        "dist_soft_max": 2.0,
        "dist_hard_max": 1.0,
        "n_interaction_layers": 1,
        "n_atom_layers": 1,
        "sensitivity_type": "inverse",
        "resnet": True,
    }


@pytest.fixture
def explicit_edge_input_nodes():
    from hippynn.graphs import inputs

    species = inputs.SpeciesNode(db_name="Z")
    positions = inputs.PositionsNode(db_name="R")
    edge_indices = inputs.PredefinedEdgeIndicesNode(db_name="edge_indices")

    return species, positions, edge_indices


@pytest.fixture
def explicit_edge_network(explicit_edge_input_nodes, explicit_edge_network_params):
    from hippynn.graphs import networks

    return networks.Hipnn("HIPNN", explicit_edge_input_nodes, module_kwargs=dict(explicit_edge_network_params))


@pytest.fixture
def explicit_edge_box():
    z = torch.tensor(
        [
            [1, 1, 1, 0],
            [1, 1, 0, 0],
        ],
        dtype=torch.long,
    )
    r = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    edges = torch.tensor(
        [
            [[0, 2, -1], [1, 0, -1]],
            [[1, -1, -1], [0, -1, -1]],
        ],
        dtype=torch.long,
    )

    return z, r, edges


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


def test_build_network_from_explicit_edges(explicit_edge_network):
    from hippynn.graphs import find_unique_relative
    from hippynn.graphs.nodes import pairs
    from hippynn.layers.hiplayers import NoCutoff

    pairfinder = find_unique_relative(explicit_edge_network, pairs.PredefinedEdgePairIndexer)

    assert isinstance(pairfinder, pairs.PredefinedEdgePairIndexer)
    assert pairfinder.name == "PredefinedEdgePairIndexer"
    assert isinstance(explicit_edge_network.torch_module.sensitivity_layers[0].cutoff, NoCutoff)


def test_explicit_edges_convert_to_directed_pair_tensors_for_multiple_frames(
    explicit_edge_input_nodes, explicit_edge_network, explicit_edge_box
):
    from hippynn.graphs import GraphModule, find_unique_relative
    from hippynn.graphs.nodes import pairs

    pairfinder = find_unique_relative(explicit_edge_network, pairs.PredefinedEdgePairIndexer)
    graph = GraphModule(
        explicit_edge_input_nodes,
        [pairfinder.pair_first, pairfinder.pair_second, pairfinder.pair_dist, pairfinder.pair_coord],
    )

    pair_first, pair_second, pair_dist, pair_coord = graph(*explicit_edge_box)

    assert torch.equal(pair_first, torch.tensor([0, 2, 4]))
    assert torch.equal(pair_second, torch.tensor([1, 0, 3]))
    assert torch.allclose(pair_dist, torch.tensor([1.0, 3.0, 2.0]))
    assert torch.allclose(
        pair_coord,
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [-3.0, 0.0, 0.0],
                [0.0, -2.0, 0.0],
            ]
        ),
    )


def test_pair_cacher_sparse_cache_feeds_pair_uncacher():
    from hippynn.layers.pairs import PairCacher, PairUncacher

    coordinates = torch.tensor([[[0.1, 0.0, 0.0], [9.9, 0.0, 0.0]]])
    cell = torch.tensor([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]])
    real_atoms = torch.tensor([0, 1])
    inv_real_atoms = torch.tensor([0, 1])
    system_index = torch.tensor([0, 0])
    pair_first = torch.tensor([0])
    pair_second = torch.tensor([1])
    cell_offsets = torch.tensor([[1, 0, 0]])
    offset_index = torch.tensor([0])

    sparse_cache = PairCacher()(
        pair_first,
        pair_second,
        cell_offsets,
        offset_index,
        real_atoms,
        system_index,
        1,
        2,
    )
    pair_dist, cached_first, cached_second, pair_coord, cached_offsets, cached_offset_index = PairUncacher()(
        sparse_cache, coordinates, cell, real_atoms, inv_real_atoms, 2, 1
    )

    assert sparse_cache.is_sparse
    assert torch.equal(cached_first, pair_first)
    assert torch.equal(cached_second, pair_second)
    assert torch.equal(cached_offsets, cell_offsets)
    assert torch.equal(cached_offset_index, offset_index)
    assert torch.allclose(pair_dist, torch.tensor([0.2]), atol=1e-6)
    assert torch.allclose(pair_coord, torch.tensor([[0.2, 0.0, 0.0]]), atol=1e-6)


def test_pair_uncacher_reads_dense_predefined_edges():
    from hippynn.layers.pairs import PairUncacher

    coordinates = torch.tensor([[[0.1, 0.0, 0.0], [9.9, 0.0, 0.0]]])
    cell = torch.tensor([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]])
    real_atoms = torch.tensor([0, 1])
    inv_real_atoms = torch.tensor([0, 1])
    edge_indices = torch.tensor([[[0], [1], [1], [0], [0]]])

    pair_dist, pair_first, pair_second, pair_coord, cell_offsets, offset_index = PairUncacher()(
        edge_indices, coordinates, cell, real_atoms, inv_real_atoms, 2, 1
    )

    assert torch.equal(pair_first, torch.tensor([0]))
    assert torch.equal(pair_second, torch.tensor([1]))
    assert torch.equal(cell_offsets, torch.tensor([[1, 0, 0]]))
    assert offset_index is None
    assert torch.allclose(pair_dist, torch.tensor([0.2]), atol=1e-6)
    assert torch.allclose(pair_coord, torch.tensor([[0.2, 0.0, 0.0]]), atol=1e-6)


def test_periodic_predefined_edges_build_from_network_inputs(explicit_edge_network_params):
    from hippynn.graphs import GraphModule, find_unique_relative, inputs, networks
    from hippynn.graphs.nodes import pairs

    species = inputs.SpeciesNode(db_name="Z")
    positions = inputs.PositionsNode(db_name="R")
    cell = inputs.CellNode(db_name="cell")
    edge_indices = inputs.PredefinedEdgeIndicesNode(db_name="edge_indices")
    network = networks.Hipnn(
        "PeriodicPredefinedHIPNN",
        (species, positions, cell, edge_indices),
        module_kwargs=dict(explicit_edge_network_params),
    )
    pairfinder = find_unique_relative(network, pairs.PredefinedEdgePairIndexer)
    graph = GraphModule(
        [species, positions, cell, edge_indices],
        [pairfinder.pair_first, pairfinder.pair_second, pairfinder.pair_dist, pairfinder.pair_coord],
    )

    z = torch.tensor([[1, 1]], dtype=torch.long)
    r = torch.tensor([[[0.1, 0.0, 0.0], [9.9, 0.0, 0.0]]])
    c = torch.tensor([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]])
    edges = torch.tensor([[[0], [1], [1], [0], [0]]], dtype=torch.long)

    pair_first, pair_second, pair_dist, pair_coord = graph(z, r, c, edges)

    assert torch.equal(pair_first, torch.tensor([0]))
    assert torch.equal(pair_second, torch.tensor([1]))
    assert torch.allclose(pair_dist, torch.tensor([0.2]), atol=1e-6)
    assert torch.allclose(pair_coord, torch.tensor([[0.2, 0.0, 0.0]]), atol=1e-6)


def test_explicit_edges_disable_sensitivity_cutoff_but_radial_path_does_not(
    explicit_edge_input_nodes, explicit_edge_network_params
):
    from hippynn.graphs import networks
    from hippynn.layers.hiplayers import CosCutoff, NoCutoff

    species, positions, _edge_indices = explicit_edge_input_nodes
    explicit_network = networks.Hipnn(
        "PredefinedEdgeHIPNN", explicit_edge_input_nodes, module_kwargs=dict(explicit_edge_network_params)
    )
    explicit_cutoff = explicit_network.torch_module.sensitivity_layers[0].cutoff

    radial_network = networks.Hipnn(
        "RadialHIPNN", (species, positions), module_kwargs=dict(explicit_edge_network_params)
    )
    radial_cutoff = radial_network.torch_module.sensitivity_layers[0].cutoff

    explicit_cosine_params = dict(explicit_edge_network_params)
    explicit_cosine_params["cutoff_type"] = CosCutoff
    explicit_cosine_network = networks.Hipnn(
        "PredefinedEdgeCosineHIPNN", explicit_edge_input_nodes, module_kwargs=explicit_cosine_params
    )
    explicit_cosine_cutoff = explicit_cosine_network.torch_module.sensitivity_layers[0].cutoff

    long_dist = torch.tensor([2.0])
    assert isinstance(explicit_cutoff, NoCutoff)
    assert isinstance(radial_cutoff, CosCutoff)
    assert isinstance(explicit_cosine_cutoff, CosCutoff)
    assert torch.equal(explicit_cutoff(long_dist), torch.ones_like(long_dist))
    assert torch.equal(radial_cutoff(long_dist), torch.zeros_like(long_dist))
    assert torch.equal(explicit_cosine_cutoff(long_dist), torch.zeros_like(long_dist))


def test_expanded_explicit_edge_parents_disable_sensitivity_cutoff(
    explicit_edge_input_nodes, explicit_edge_network_params
):
    from hippynn.graphs import networks
    from hippynn.layers.hiplayers import CosCutoff, NoCutoff

    explicit_network = networks.Hipnn(
        "PredefinedEdgeHIPNN", explicit_edge_input_nodes, module_kwargs=dict(explicit_edge_network_params)
    )
    expanded_network = networks.Hipnn(
        "ExpandedPredefinedEdgeHIPNN", explicit_network.parents, module_kwargs=dict(explicit_edge_network_params)
    )

    explicit_cosine_params = dict(explicit_edge_network_params)
    explicit_cosine_params["cutoff_type"] = CosCutoff
    expanded_cosine_network = networks.Hipnn(
        "ExpandedPredefinedEdgeCosineHIPNN", explicit_network.parents, module_kwargs=explicit_cosine_params
    )

    expanded_cutoff = expanded_network.torch_module.sensitivity_layers[0].cutoff
    expanded_cosine_cutoff = expanded_cosine_network.torch_module.sensitivity_layers[0].cutoff

    assert isinstance(expanded_cutoff, NoCutoff)
    assert isinstance(expanded_cosine_cutoff, CosCutoff)
