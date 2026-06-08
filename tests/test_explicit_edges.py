import torch

from hippynn.graphs import GraphModule, find_unique_relative, inputs, networks
from hippynn.graphs.nodes import pairs
from hippynn.layers.hiplayers import CosCutoff, NoCutoff


def _edge_network_params():
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


def test_build_network_from_explicit_edges():
    species = inputs.SpeciesNode(db_name="Z")
    positions = inputs.PositionsNode(db_name="R")
    edge_indices = inputs.PreDefinedEdgeIndicesNode(db_name="edge_indices")

    network = networks.Hipnn("HIPNN", (species, positions, edge_indices), module_kwargs=_edge_network_params())
    pairfinder = find_unique_relative(network, pairs.PreDefinedEdgePairIndexer)

    assert isinstance(pairfinder, pairs.PreDefinedEdgePairIndexer)
    assert pairfinder.name == "PreDefinedEdgePairIndexer"
    assert isinstance(network.torch_module.sensitivity_layers[0].cutoff, NoCutoff)


def test_explicit_edges_convert_to_directed_pair_tensors_for_multiple_frames():
    species = inputs.SpeciesNode(db_name="Z")
    positions = inputs.PositionsNode(db_name="R")
    edge_indices = inputs.PreDefinedEdgeIndicesNode(db_name="edge_indices")

    network = networks.Hipnn("HIPNN", (species, positions, edge_indices), module_kwargs=_edge_network_params())
    pairfinder = find_unique_relative(network, pairs.PreDefinedEdgePairIndexer)
    graph = GraphModule(
        [species, positions, edge_indices],
        [pairfinder.pair_first, pairfinder.pair_second, pairfinder.pair_dist, pairfinder.pair_coord],
    )

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

    pair_first, pair_second, pair_dist, pair_coord = graph(z, r, edges)

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


def test_explicit_edges_disable_sensitivity_cutoff_but_radial_path_does_not():
    species = inputs.SpeciesNode(db_name="Z")
    positions = inputs.PositionsNode(db_name="R")
    edge_indices = inputs.PreDefinedEdgeIndicesNode(db_name="edge_indices")

    explicit_network = networks.Hipnn(
        "PreDefinedEdgeHIPNN", (species, positions, edge_indices), module_kwargs=_edge_network_params()
    )
    explicit_cutoff = explicit_network.torch_module.sensitivity_layers[0].cutoff

    radial_network = networks.Hipnn("RadialHIPNN", (species, positions), module_kwargs=_edge_network_params())
    radial_cutoff = radial_network.torch_module.sensitivity_layers[0].cutoff

    long_dist = torch.tensor([2.0])
    assert isinstance(explicit_cutoff, NoCutoff)
    assert isinstance(radial_cutoff, CosCutoff)
    assert torch.equal(explicit_cutoff(long_dist), torch.ones_like(long_dist))
    assert torch.equal(radial_cutoff(long_dist), torch.zeros_like(long_dist))
