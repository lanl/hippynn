import pytest

import torch
import hippynn
from conftest import ignore_sensitivity_warning
import inspect

from hippynn.graphs.nodes.pairs import (
    PeriodicPairIndexer,
    PeriodicPairIndexerMemory,
    SparsePairIndexer,
    NumpyDynamicPairs,
    KDTreePairs,
    KDTreePairsMemory,
)
from hippynn.interfaces.ase_interface import ASEPairNode

from test_sparse_neighbors import _torch_device, canonicalize_outs


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("cutoff,supercell", [(1.8, 1), (3.99, 2)])
@pytest.mark.parametrize(
    "pairfinder_cls",
    [PeriodicPairIndexer, PeriodicPairIndexerMemory, SparsePairIndexer, NumpyDynamicPairs, ASEPairNode, KDTreePairs, KDTreePairsMemory],
)
def test_consistency(pairfinder_cls, dtype, cutoff, supercell):

    from hippynn.graphs import GraphModule
    from hippynn.graphs.nodes.inputs import SpeciesNode, PositionsNode, CellNode
    from hippynn.graphs.nodes.indexers import acquire_encoding_padding
    from hippynn.graphs.nodes.pairs import PeriodicPairIndexer
    import ase
    import ase.build

    # ase.build.
    device = _torch_device()

    # Set up input nodes
    sp = SpeciesNode("Z")
    pos = PositionsNode("R")
    cell = CellNode("C")

    # cutoff = 3.9
    pos.requires_grad = True
    # Set up and compile calculation
    enc, pidxer = acquire_encoding_padding(sp, species_set=[0, 1])
    input_nodes = (pos, enc, pidxer, cell)
    ref = PeriodicPairIndexer("ref", input_nodes, dist_hard_max=cutoff)

    if pairfinder_cls in [PeriodicPairIndexerMemory, KDTreePairsMemory]:
        skin = 0.01
        got = pairfinder_cls("got", input_nodes, dist_hard_max=cutoff, skin=skin)
    else:
        got = pairfinder_cls("got", input_nodes, dist_hard_max=cutoff)
    out_nodes = [*ref.children, *got.children]
    computer = GraphModule([sp, pos, cell], out_nodes)
    computer.to(device)

    # build a molecule and convert it to a batch of one.
    a = ase.build.molecule("biphenyl", vacuum=0.5)
    import numpy as np

    a = ase.build.make_supercell(a, supercell * np.eye(3))

    positions = torch.as_tensor(a.get_positions(), dtype=dtype, device=device).unsqueeze(0)
    n_atoms = positions.shape[1]
    nonblank = torch.ones((1, n_atoms), dtype=torch.bool, device=device)
    cells = torch.as_tensor(a.get_cell().array, dtype=dtype, device=device).unsqueeze(0)
    species_tensor = nonblank.to(torch.long)

    # Run calculation
    outputs = computer(species_tensor, positions, cells)
    outputs = {k.name: v for k, v in zip(out_nodes, outputs)}

    keys = ["pair_first", "pair_second", "cell_offsets", "pair_dist", "pair_coord"]

    refs = [outputs[f"ref.{k}"] for k in keys]
    gots = [outputs[f"got.{k}"] for k in keys]
    ref_discrete, ref_dist, ref_disp = canonicalize_outs(*refs)
    got_discrete, got_dist, got_disp = canonicalize_outs(*gots)

    got_has_grad = got_dist.requires_grad, got_disp.requires_grad
    assert all(got_has_grad)

    print("n_pairs:", ref_dist.shape[0], end=" ")  # if pytest is run in -s, we will show number of pairs.
    assert ref_dist.shape == got_dist.shape, "different number of pairs"
    assert torch.equal(ref_discrete, got_discrete), "different discrete outputs"

    # These tols can be adjusted because right now essentially all algorithms recompute using the discrete indices and the same pytorch calls.
    rtol = 1e-13 if dtype == torch.float64 else 1e-6
    atol = 0

    is_close = torch.isclose(ref_dist, got_dist, atol=atol, rtol=rtol)
    bad_rows = torch.where(~is_close)[0]
    bad_rows = torch.unique(bad_rows)
    if bad_rows.any():
        print(f"Different distances detected: {bad_rows=}\n values:{torch.stack([ref_dist,got_dist],dim=1)[bad_rows][:1]}")
        raise ValueError("Different distances detected! {bad_rows=}")
    assert torch.allclose(ref_dist, got_dist, atol=atol, rtol=rtol)

    is_close = torch.isclose(ref_disp, got_disp, atol=atol, rtol=rtol)
    bad_rows = torch.where(~is_close)[0]
    bad_rows = torch.unique(bad_rows)
    if bad_rows.any():
        print(f"Different displacements detected: {bad_rows=}\n values:{torch.stack([ref_disp,got_disp],dim=2)[bad_rows][:1]}")
        raise ValueError(f"Different displacements detected! {bad_rows=}")
    assert torch.allclose(ref_disp, got_disp, atol=atol, rtol=rtol)
