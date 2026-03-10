"""
This file was written with assistance from an LLM.
"""

import random
from typing import Tuple

import pytest
import torch

# Under test
from hippynn.layers.pairs.csr_pairs.neighbor_algorithm import (
    normalize_atoms,
    build_image_offsets,
    build_image_atoms,
    voxelize_images,
    voxel_adjacency,
    calc_neighbors,
    expand_pairs,
    build_initial_data,
)
from hippynn.layers.pairs.csr_pairs.csrtable import CSRTable

from test_csrtable import device_dtype_pairs


def _torch_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@pytest.fixture(scope="module")
def rng_seed():
    return 1337


@pytest.fixture(scope="module")
def device():
    return _torch_device()


@pytest.fixture(scope="function", autouse=True)
def seeded(rng_seed):
    random.seed(rng_seed)
    torch.manual_seed(rng_seed)


def random_triclinic_cells(n_systems: int, device, dtype=torch.float32) -> torch.Tensor:
    """
    Generate 'nice' triclinic cells with reasonable aspect ratios.
    """
    # Build lower-triangular with positive diagonals, add off-diagonals.
    a = torch.rand(n_systems, 3, device=device, dtype=dtype) * 4.0 + 2.0  # lengths ~[2,6]
    # Angles: avoid degeneracy; skew but not extreme
    alpha = torch.rand(n_systems, device=device, dtype=dtype) * 0.6 + 1.2  # ~[1.2,1.8] rad
    beta = torch.rand(n_systems, device=device, dtype=dtype) * 0.6 + 1.2
    gamma = torch.rand(n_systems, device=device, dtype=dtype) * 0.6 + 1.2
    # Build cell vectors via standard formula
    # a vector along x
    ax = torch.stack([a[:, 0], torch.zeros_like(a[:, 0]), torch.zeros_like(a[:, 0])], dim=1)
    # b vector in xy plane
    bx = a[:, 1] * torch.cos(gamma)
    by = a[:, 1] * torch.sin(gamma)
    b = torch.stack([bx, by, torch.zeros_like(by)], dim=1)
    # c vector in 3D
    cx = a[:, 2] * torch.cos(beta)
    cy = a[:, 2] * (torch.cos(alpha) - torch.cos(beta) * torch.cos(gamma)) / torch.sin(gamma)
    cz_sq = a[:, 2] ** 2 - cx**2 - cy**2
    cz = torch.sqrt(torch.clamp(cz_sq, min=1e-6))
    c = torch.stack([cx, cy, cz], dim=1)

    H = torch.zeros((n_systems, 3, 3), device=device, dtype=dtype)
    H[:, 0, :] = ax
    H[:, 1, :] = b
    H[:, 2, :] = c
    return H


def random_batch(n_systems: int, n_atoms_max: int, device, dtype=torch.float32):
    cells = random_triclinic_cells(n_systems, device, dtype)
    # make some atoms per system (randomly fewer than N_max)
    counts = torch.randint(low=1, high=n_atoms_max + 1, size=(n_systems,), device=device)
    nonblank = torch.ones((n_systems, n_atoms_max), dtype=torch.bool, device=device)
    # for s in range(n_systems):
    #    if counts[s] > 0:
    #        nonblank[s, : counts[s]] = True
    # random Cartesian positions (not wrapped) somewhat larger than cell to force wrapping
    positions = torch.randn((n_systems, n_atoms_max, 3), device=device, dtype=dtype) * 3.0
    return positions, nonblank, cells


def make_cell(kind: str, dtype, device) -> torch.Tensor:
    """Return H with shape [1,3,3] in (basis, cartesian) convention."""
    if kind == "cubic":
        # isotropic box, non-identity to avoid masking bugs
        H = torch.diag(torch.tensor([2.0, 2.0, 2.0], dtype=dtype, device=device))
    elif kind == "orthorhombic":
        # diagonal but anisotropic
        H = torch.diag(torch.tensor([2.10, 1.70, 1.25], dtype=dtype, device=device))
    elif kind == "triclinic":
        # lower-triangular-ish with off-diagonals (skewed)
        # fmt: off
        H = torch.tensor(
            [[2.10, 0.20, 0.10],
             [0.00, 1.70, 0.30],
             [0.00, 0.10, 1.25]],
            dtype=dtype, device=device
        )
        # fmt: on
    else:
        raise ValueError(f"bad cell kind {kind!r}")
    return H.unsqueeze(0)  # [S=1,3,3]


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("cell_kind", ["cubic", "orthorhombic", "triclinic"])
def test_normalize_atoms_parametric(cell_kind, dtype):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H = make_cell(cell_kind, dtype, device)  # [1,3,3]

    # Fractional coords (some <0, >1) chosen to exercise wrapping
    # fmt: off
    f = torch.tensor(
        [[-0.20,  0.10,  0.90],
         [ 1.30, -0.70,  2.10],
         [ 0.05,  1.25, -0.40]],
        dtype=dtype, device=device
    )  # [N=3,3]
    # fmt: on
    
    
    # Build minimal CSR inputs expected by normalize_atoms
    systemCSR = CSRTable.from_counts(
        counts=torch.ones(1, dtype=torch.long, device=device),
        row_data={"cells": H},
    )

    # Cartesian by definition: r = f @ H  (H is (basis, cart))
    H = H[0]
    # Hinv = torch.linalg.inv(H)
    r = f @ H  # [N,3]

    atomCSR = CSRTable.from_counts(
        counts=torch.tensor([f.shape[0]], dtype=torch.long, device=device),
    )
    atomCSR["raw_positions"] = r

    # Call under test
    atomCSR, systemCSR = normalize_atoms(atomCSR, systemCSR)

    # Expected results
    k0 = torch.floor(f).to(torch.long)  # [N,3]
    f_wrapped = f - k0.to(dtype)  # [N,3]
    r_expected = f_wrapped @ H  # [N,3]

    # Tolerances
    atol = 1e-10 if dtype == torch.float64 else 5e-6
    rtol = 0.0

    # Checks
    assert torch.equal(atomCSR["offsets"], k0), "offsets mismatch (floor(f))"
    assert torch.allclose(atomCSR["positions"], r_expected, atol=atol, rtol=rtol), "wrapped positions mismatch"
    # Round-trip: (f_wrapped + k0) @ H == r
    r_recon = (f_wrapped + k0.to(dtype)) @ H
    assert torch.allclose(r_recon, r, atol=atol, rtol=rtol), "reconstruction mismatch"


def canonicalize_outs(idsA: torch.Tensor, idsB: torch.Tensor, k: torch.Tensor, dist, disp):

    discrete = torch.stack([idsA, idsB, *k.unbind(1)], dim=1)
    order = lexsort_torch(discrete.unbind(1))
    return map(lambda x: x[order], [discrete, dist, disp])


@torch.no_grad()
def brute_force_pairs(
    positions: torch.Tensor,  # [S, N_max, 3]  (Cartesian, unwrapped)
    nonblank: torch.Tensor,  # [S, N_max]     (bool)
    cells: torch.Tensor,  # [S, 3, 3]      (real-space basis; columns are axes)
    cutoff: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    import math

    """
    Very simple brute force (Cartesian only):
      • enumerate k in a per-axis cube: |kx|<=ceil(cutoff/||ax||), same for y,z
      • compute disp_cart = (r_j + k @ A^T) - r_i   (no wrapping, no inv(A))
      • keep d2 <= cutoff^2 (+eps), drop (i==j & k==0), keep same-atom with nonzero k
      • return (idA, idB, rel_k[int32/64], dist[float32])
    """
    device, p_dtype = positions.device, positions.dtype
    S, N_max, _ = positions.shape

    idsA_all, idsB_all, k_all, disp_all, d_all = [], [], [], [], []

    gids = torch.zeros(nonblank.shape, device=nonblank.device, dtype=torch.long)
    gids[nonblank] = torch.arange(nonblank.sum(), device=nonblank.device, dtype=torch.long)

    for s in range(S):
        R = positions[s]
        A = cells[s]
        # normalize coordinates
        A_inv = torch.linalg.inv(A)
        frac = torch.remainder(R @ A_inv, 1)
        shifts = torch.divide(R @ A_inv, 1.0, rounding_mode="floor").to(torch.int)
        R = frac @ A

        IDlist = gids[s].tolist()
        sigma_min = torch.linalg.svdvals(A).min()
        k_max = int(math.ceil(cutoff / sigma_min)) + 1

        krange = list(range(-k_max, k_max + 1))
        import itertools

        k_vals = list(itertools.product(krange, repeat=3))
        k_vals = torch.as_tensor(k_vals, device=A.device, dtype=A.dtype)
        if len(k_vals) > 1e5:
            raise ValueError("too many combinations! {len(k_vals)=}")
        # print(f"{k_max=}")
        for k_shift in k_vals.unbind(0):
            dr = k_shift @ A

            for i in range(N_max):
                if not nonblank[s, i]:
                    continue
                xi = R[i]
                si = shifts[i]
                for j in range(N_max):
                    if not nonblank[s, j]:
                        continue

                    xj = R[j]
                    sj = shifts[j]
                    r = xi - xj + dr
                    d = torch.linalg.norm(r).item()

                    if d < cutoff:

                        if i == j and (k_shift == 0).all():
                            # skip self-connection
                            continue

                        gidA = IDlist[i]
                        gidB = IDlist[j]
                        k_shift_mod = k_shift + sj - si
                        k_shift_mod = k_shift_mod.tolist()
                        r = r.tolist()

                        # print("Got ", gidA, gidB, k_shift_mod)#, d, si.tolist(),sj.tolist())#, dr.tolist(), r, d)

                        idsA_all.append(gidA)
                        idsB_all.append(gidB)

                        k_all.append(k_shift_mod)
                        disp_all.append(r)
                        d_all.append(d)

    if not idsA_all:
        empty_l = torch.empty(0, dtype=torch.long, device=device)
        empty_k = torch.empty((0, 3), dtype=torch.long, device=device)
        empty_f = torch.empty(0, dtype=positions.dtype, device=device)
        return empty_l, empty_l, empty_k, empty_f, empty_k.to(positions.dtype)

    idA = torch.as_tensor(idsA_all, device=A.device, dtype=torch.long)
    idB = torch.as_tensor(idsB_all, device=A.device, dtype=torch.long)
    rel_k = torch.as_tensor(k_all, device=A.device, dtype=torch.long)
    disp = torch.as_tensor(disp_all, device=A.device, dtype=R.dtype)

    dist = torch.as_tensor(d_all, device=A.device, dtype=R.dtype)

    return idA, idB, rel_k, dist, disp


def lexsort_torch(keys: list[torch.Tensor]) -> torch.Tensor:
    """
    Lexicographic order of equal-length 1D tensors (last key highest priority),
    equivalent to numpy.lexsort but stays on the tensor's device.
    """
    if list(keys) == []:
        raise ValueError("keys must be non-empty")
    n = keys[0].numel()
    dev = keys[0].device
    for k in keys:
        if k.ndim != 1 or k.numel() != n or k.device != dev:
            raise ValueError("all keys must be 1D, same length, same device")

    idx = torch.arange(n, device=dev)
    # sort from lowest to highest priority using stable argsort
    for k in reversed(keys):
        try:
            order = torch.argsort(k[idx], stable=True)
        except TypeError:
            order = torch.argsort(k[idx])  # fallback if stable not available
        idx = idx[order]
    return idx


@pytest.mark.parametrize("n_systems,n_atoms_max", [(2, 3), (2, 16), (4, 12)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_neighbors_vs_bruteforce(n_systems, n_atoms_max, dtype, device):
    cutoff = 0.8
    positions, nonblank, cells = random_batch(n_systems, n_atoms_max, device, dtype=dtype)

    with torch.no_grad():
        idA, idB, sys, rel_k, got_dist, got_disp = calc_neighbors(
            positions=positions,
            nonblank=nonblank,
            cells=cells,
            cutoff=cutoff,
            use_full_stencil=True,
        )

    refA, refB, refK, ref_dist, ref_disp = brute_force_pairs(positions, nonblank, cells, cutoff)

    ref_discrete, ref_dist, ref_disp = canonicalize_outs(refA, refB, refK, ref_dist, ref_disp)
    got_discrete, got_dist, got_disp = canonicalize_outs(idA, idB, rel_k, got_dist, got_disp)
    assert ref_discrete.shape == got_discrete.shape
    assert torch.equal(ref_discrete, got_discrete)
    atol = 1e-6
    assert torch.allclose(ref_dist, got_dist, atol=atol, rtol=0)


@pytest.mark.parametrize("n_systems,n_atoms_max", [(2, 10), (1,0), (0,10)])
@pytest.mark.parametrize("device_dtype", device_dtype_pairs)
def test_empty(device_dtype, n_systems, n_atoms_max):
    device, dtype = device_dtype

    cutoff = 2.0
    positions = torch.zeros((n_systems, n_atoms_max, 3), dtype=dtype, device=device)
    nonblank = torch.zeros((n_systems, n_atoms_max), dtype=torch.bool, device=device)
    # Make system 0 empty, system 1 with 1 atom at origin
    # nonblank[1, 0] = True
    cells = random_triclinic_cells(n_systems, device, dtype)

    try:
        with torch.no_grad():
            idA, idB, sys, rel_k, dist, disp = calc_neighbors(positions, nonblank, cells, cutoff)
    except:        
        if torch.device(device).type == "mps":
            pytest.xfail("mps does not work on all empty systems")
        else:
            raise

    # With only one atom total, there should be no pairs
    assert idA.numel() == 0, "IdA nonempty"
    assert idB.numel() == 0, "IdB nonempty"
    assert rel_k.numel() == 0, "rel_K nonempty"
    assert dist.numel() == 0, "dist nonempty"
    assert disp.numel() == 0, "disp nonempty"


@pytest.mark.parametrize("n_systems,n_atoms_max", [(2, 10)])
def test_single_lone_atom(device, n_systems, n_atoms_max):
    dtype = torch.float32
    cutoff = 0.1
    positions = torch.zeros((n_systems, n_atoms_max, 3), dtype=dtype, device=device)
    nonblank = torch.zeros((n_systems, n_atoms_max), dtype=torch.bool, device=device)
    # Make system 0 empty, system 1 with 1 atom at origin
    nonblank[1, 0] = True
    cells = random_triclinic_cells(n_systems, device, dtype)

    with torch.no_grad():
        idA, idB, sys, rel_k, dist, disp = calc_neighbors(positions, nonblank, cells, cutoff)

    # With only one atom total, there should be no pairs
    assert idA.numel() == 0, "IdA nonempty"
    assert idB.numel() == 0, "IdB nonempty"
    assert rel_k.numel() == 0, "rel_K nonempty"
    assert dist.numel() == 0, "dist nonempty"
    assert disp.numel() == 0, "disp nonempty"


hippynn = None
try:
    import hippynn  # type: ignore
except ImportError:
    hippynn = False


def matches_hippynn(positions, nonblank, cells, cutoff, device):

    ## compares against hippynn PeriodicPairIndexer

    from hippynn.graphs import GraphModule
    from hippynn.graphs.nodes.inputs import SpeciesNode, PositionsNode, CellNode
    from hippynn.graphs.nodes.indexers import acquire_encoding_padding
    from hippynn.graphs.nodes.pairs import PeriodicPairIndexer

    # Set up input nodes
    sp = SpeciesNode("Z")
    pos = PositionsNode("R")
    cell = CellNode("C")

    # Set up and compile calculation
    enc, pidxer = acquire_encoding_padding(sp, species_set=[0, 1])
    pairfinder = PeriodicPairIndexer("pair_finder", (pos, enc, pidxer, cell), dist_hard_max=cutoff)
    computer = GraphModule([sp, pos, cell], [*pairfinder.children])
    computer.to(device)

    # Get some random inputs
    species_tensor = nonblank.to(torch.long)

    # Run calculation
    outputs = computer(species_tensor, positions, cells)

    # Print outputs
    output_as_dict = {c.name: o for c, o in zip(pairfinder.children, outputs)}

    # def minmax(tensor):
    #     if tensor.numel()==0:
    #         return None,None
    #     else:
    #         return tensor.min(), tensor.max()

    # Build our neighbors
    with torch.no_grad():
        idA, idB, sys, rel_k, got_dist, got_disp = calc_neighbors(positions, nonblank, cells, cutoff)
        # for v in [idA, idB, rel_k, dist]:
        #     print(v.shape, v.dtype, *minmax(v))

    # for k, v in output_as_dict.items():
    #     print(k, v.shape, v.dtype, *minmax(v))

    ref_first, ref_second, ref_shift, ref_dist, ref_disp = [
        output_as_dict[f"pair_finder.{k}"] for k in ["pair_first", "pair_second", "cell_offsets", "pair_dist", "pair_coord"]
    ]

    ref_discrete, ref_dist, ref_disp = canonicalize_outs(ref_first, ref_second, ref_shift, ref_dist, ref_disp)
    got_discrete, got_dist, got_disp = canonicalize_outs(idA, idB, rel_k, got_dist, got_disp)
    # assert ref_discrete.shape==got_discrete.shape
    assert ref_dist.shape == got_dist.shape, "different number of pairs"
    assert torch.equal(ref_discrete, got_discrete), "different discrete outputs"
    atol = 1e-5
    is_close = torch.isclose(ref_dist, got_dist, atol=atol, rtol=0)
    bad_rows = torch.where(~is_close)[0]
    if bad_rows.any():
        raise ValueError(f"Different distances detected: {bad_rows=}\n values:{torch.stack([ref_dist,got_dist],dim=1)[bad_rows]}")
    assert torch.allclose(ref_dist, got_dist, atol=atol, rtol=0)

    print("n_pairs:", ref_dist.shape[0], end=" ")  # if pytest is run in -s, we will show number of pairs.
    assert ref_dist.shape == got_dist.shape, "different number of pairs"
    assert torch.equal(ref_discrete, got_discrete), "different discrete outputs"
    atol = 1e-5
    is_close = torch.isclose(ref_dist, got_dist, atol=atol, rtol=0)
    bad_rows = torch.where(~is_close)[0]
    bad_rows = torch.unique(bad_rows)
    if bad_rows.any():
        print(f"Different distances detected: {bad_rows=}\n values:{torch.stack([ref_dist,got_dist],dim=1)[bad_rows][:1]}")
        raise ValueError("Different distances detected! {bad_rows=}")
    assert torch.allclose(ref_dist, got_dist, atol=atol, rtol=0)

    is_close = torch.isclose(ref_disp, got_disp, atol=atol, rtol=0)
    bad_rows = torch.where(~is_close)[0]
    bad_rows = torch.unique(bad_rows)
    if bad_rows.any():
        print(f"Different displacements detected: {bad_rows=}\n values:{torch.stack([ref_disp,got_disp],dim=2)[bad_rows][:1]}")
        raise ValueError(f"Different displacements detected! {bad_rows=}")
    assert torch.allclose(ref_disp, got_disp, atol=atol, rtol=0)


@pytest.mark.parametrize("n_atoms,cutoff", [(1, 1.05), (1, 2.01), (1, 3.01), (2, 1.05), (2, 2.01), (2, 3.01), (100, 1.01), (200, 3.01)])
def test_matches_hippynn_cubic(n_atoms, cutoff, device):

    # note for developers: due to rounding behavior, don't test the cutoff == box length exactly.

    # use shrink factor so that distance outputs do not look like the rel_k outputs
    shrink_factor = 1.0
    dtype = torch.float32
    cutoff = shrink_factor * float(cutoff)
    ndim = 3
    n_systems = 1

    nonblank = torch.ones((n_systems, n_atoms), dtype=torch.bool)
    positions = shrink_factor * torch.linspace(0, 1, n_atoms, dtype=dtype)
    positions = positions.unsqueeze(0).unsqueeze(2).expand(n_systems, n_atoms, ndim)
    # positions = shrink_factor/2*torch.ones((1,1,3),dtype=torch.float32)
    cells = shrink_factor * torch.eye(ndim, dtype=dtype).unsqueeze(0).expand(n_systems, ndim, ndim)
    # positions, nonblank, cells = random_batch(n_systems, n_atoms_max, device, dtype)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    matches_hippynn(positions, nonblank, cells, cutoff, device)


@pytest.mark.skipif(not hippynn, reason="hippynn not installed")
@pytest.mark.parametrize(
    "n_systems,n_atoms_max,cutoff", [(1, 1, 2.0), (1, 2, 3.0), (20, 5, 0.3), (3, 20, 0.3), (3, 100, 0.3), (100, 10, 0.5), (20, 20, 3.01)]
)
def test_matches_hippynn_random(n_systems, n_atoms_max, cutoff, device):

    dtype = torch.float32
    cutoff = float(cutoff)
    ndim = 3
    positions, nonblank, cells = random_batch(n_systems, n_atoms_max, device, dtype)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    matches_hippynn(positions, nonblank, cells, cutoff, device)


@pytest.mark.parametrize("cutoff_factor", [1.0, 2.0, 3.0])
@pytest.mark.parametrize("cell_kind", ["cubic", "orthorhombic", "triclinic"])
def test_no_duplicate_pairs(cell_kind, cutoff_factor, device):
    """
    Broader smoke: random positions in cubic cells—still no duplicates of (idA,idB,rel_k).
    """

    dtype = torch.float32
    S = 1
    N = 2

    H = make_cell(cell_kind, dtype=dtype, device=device)
    cutoff = cutoff_factor * H.max()
    R = torch.linspace(0, H.max(), N, device=device, dtype=dtype).unsqueeze(0).unsqueeze(2).expand(1, N, 3)

    nonblank = torch.ones((S, N), dtype=torch.bool, device=device)

    with torch.no_grad():
        idA, idB, sys, rel_k, dist, disp = calc_neighbors(
            positions=R,
            nonblank=nonblank,
            cells=H,
            cutoff=float(cutoff),
            use_full_stencil=True,
        )

    key = torch.stack([idA, idB, *rel_k.unbind(-1)], dim=1)  # .to(torch.long)
    values, counts = torch.unique(key, dim=0, return_counts=True)
    if counts.numel() != 0 and (mc := counts.max().item()) > 1:
        where_dup = counts > 1
        duplicate_counts = torch.stack([*values.unbind(-1), counts], dim=-1)[where_dup]
        n_dup = where_dup.sum()
        if n_dup < 30:
            print("Duplicate [*values,count] pairs:")
            print(duplicate_counts)
        else:
            print("More than 30 duplicated entries")
        raise ValueError(f"Duplicate (idA,idB,rel_k) rows detected. Worst duplication: {mc}. Count: {n_dup}")



@pytest.mark.parametrize("device_dtype", device_dtype_pairs)
def test_device_dtype_combos(device_dtype: tuple[str]):
    device_name, dtype = device_dtype
    device = torch.device(device_name)

    # Minimal, well-behaved inputs:
    #  - 1 system
    #  - 2 atoms close enough to be neighbors
    positions = torch.zeros((1, 2, 3), dtype=dtype, device=device)
    positions[0, 0] = torch.tensor([0.10, 0.10, 0.10], dtype=dtype, device=device)
    positions[0, 1] = torch.tensor([0.20, 0.10, 0.10], dtype=dtype, device=device)

    nonblank = torch.tensor([[True, True]], dtype=torch.bool, device=device)
    cells = torch.eye(3, dtype=dtype, device=device).unsqueeze(0) * 5.0  # roomy box
    cutoff = 0.3

    idA, idB, sys, rel_k, dist, disp = calc_neighbors(
        positions=positions,
        nonblank=nonblank,
        cells=cells,
        cutoff=cutoff,
        use_full_stencil=True,
    )

    P = idA.shape[0]

    # Device placement and basic interface contracts
    for t in (idA, idB, rel_k, dist, disp):
        assert t.device.type == device.type
        if (i1 := t.device.index) and (i2 := device.index):
            assert i1 == i2
        assert t.shape[0] == (P)
        if t.data_ptr in (dist.data_ptr, disp.data_ptr):
            assert t.dtype == dtype
        else:
            assert t.dtype == torch.long

        if t.data_ptr in (rel_k.data_ptr, disp.data_ptr):
            assert t.ndim == 2
            assert t.shape[1] == 3
        else:
            assert t.ndim == 1

    assert P != 0
    assert idB.shape == (P,)
    assert rel_k.shape == (P, 3)
    assert dist.shape == (P,)
    assert disp.shape == (P, 3)

    assert idA.dtype == torch.long
    assert idB.dtype == torch.long
    assert rel_k.dtype == torch.long
    assert dist.dtype == dtype
    assert disp.dtype == dtype


def test_neighbor_counts_on_toy():
    """
    1 system, 2 atoms straddling the x-boundary.
    Choose L=1 and cutoff=0.6 so:
      - images along x are still relevant (true separation = 0.1 < cutoff)
      - voxel size == cutoff (typical), bbox is padded by +/-cutoff
      - expected grid: ceil((Δ+2*cutoff)/cutoff) per axis
          x: Δx≈0.1 → ceil((0.1+1.2)/0.6)=ceil(2.166..)=3
          y: Δy=0   → ceil((0+1.2)/0.6)=ceil(2.0)=2
          z: Δz=0   → 2
        ⇒ voxel_grid_shape == [3, 2, 2]
    """
    device = torch.device("cpu")
    dtype = torch.float32
    L = 1.0
    cutoff = 0.6

    # Two atoms across the x-boundary with 0.1 wrap-around separation
    pos = torch.zeros((1, 2, 3), dtype=dtype, device=device)
    pos[0, 0] = torch.tensor([0.95, 0.50, 0.50], dtype=dtype, device=device)
    pos[0, 1] = torch.tensor([0.05, 0.50, 0.50], dtype=dtype, device=device)

    nonblank = torch.tensor([[True, True]], dtype=torch.bool, device=device)
    cells = torch.diag(torch.tensor([L, L, L], dtype=dtype, device=device)).unsqueeze(0)

    # 0) Build atom/system CSRs and wrap atoms into [0,1) with integer offsets
    atomCSR, systemCSR = build_initial_data(pos, nonblank, cells, cutoff)

    atomCSR, systemCSR = normalize_atoms(atomCSR, systemCSR)

    # 1) Build image shifts (per-system)
    imageCSR = build_image_offsets(systemCSR, use_full_stencil=True)
    assert imageCSR.nnz == 27, "wrong number of images"
    assert imageCSR.nrows == 1, "image: wrong number of systems"

    # 2) Expand atoms × images (per-system)
    image_atomCSR = build_image_atoms(atomCSR, imageCSR, systemCSR)
    # sanity: should carry fields we later need
    for key in ("positions", "atom_gid", "offsets", "is_primary", "system"):
        assert key in image_atomCSR.data

    # 2) Image atoms (each image carries both atoms)
    assert image_atomCSR.nrows == 1, "image atoms: wrong number of systems"
    assert image_atomCSR.nnz == 2 * 2, "image atoms: wrong number of atoms"

    # 3) Voxelization → assert explicit grid shape [3,2,2]
    voxel_atomCSR, voxelCSR, sys_with_grid = voxelize_images(image_atomCSR, systemCSR)

    assert voxel_atomCSR.nrows == 2, "wrong number of voxels"  # 1 voxels
    assert voxel_atomCSR.nnz == image_atomCSR.nnz, "voxel atoms: wrong number of atoms"

    grid = sys_with_grid["voxel_grid_shape"]  # [1,3] long
    expected_grid = torch.tensor([[2, 1, 1]], dtype=torch.long, device=grid.device)
    assert torch.equal(grid, expected_grid), "wrong voxel grid shape"

    # 4) Adjacency sanity (≤ 27 edges per active voxel)
    vA, vB = voxel_adjacency(voxelCSR, sys_with_grid)
    assert vA.shape == vB.shape, "pairs don't match"
    assert vA.numel() == 4, "wrong number of voxel pairs "  # 2 self plus both voxels to each other.

    # 5) Final pairs: two ordered pairs, both at distance 0.1
    idA, idB, sys, rel_k, dist, disp = calc_neighbors(
        positions=pos,
        nonblank=nonblank,
        cells=cells,
        cutoff=cutoff,
        use_full_stencil=True,
    )
    assert dist.numel() == 2
    assert torch.allclose(dist, torch.full_like(dist, 0.1), atol=1e-6)


@pytest.mark.parametrize(
    "A",
    [
        # fmt: off
        # Sheared in xy, short z
        torch.tensor([[1.0, 0.4, 0.0],
                      [0.0, 1.2, 0.0],
                      [0.0, 0.0, 0.3]], dtype=torch.float64),
        # Rotated/anisotropic basis
        torch.tensor([[0.8,  0.6, 0.0],
                      [-0.6, 0.8, 0.0],
                      [0.1,  0.1, 1.5]], dtype=torch.float64),
        # fmt: on
    ],
)
def test_triclinic_bruteforce(A: torch.Tensor):
    device = torch.device("cpu")
    dtype = torch.float64
    cutoff = 0.6123

    # 1 system, 2 atoms roughly along the first basis direction
    positions = torch.tensor([[[0.05, 0.10, 0.10], [0.55, 0.10, 0.10]]], dtype=dtype, device=device)  # [1,2,3] Cartesian
    nonblank = torch.tensor([[True, True]], dtype=torch.bool, device=device)  # [1,2]
    cells = A.unsqueeze(0).to(dtype=dtype, device=device)  # [1,3,3]

    # Brute force (reference)
    idA_b, idB_b, k_b, dist_ref, d_ref = brute_force_pairs(positions, nonblank, cells, cutoff)

    # Pipeline under test
    idA, idB, sys, k, d, got_disp = calc_neighbors(
        positions=positions,
        nonblank=nonblank,
        cells=cells,
        cutoff=cutoff,
        use_full_stencil=True,
    )

    ref_discrete, ref_dist, ref_disp = canonicalize_outs(idA_b, idB_b, k_b, dist_ref, d_ref)
    got_discrete, got_dist, got_disp = canonicalize_outs(idA, idB, k, d, got_disp)
    assert ref_discrete.shape == got_discrete.shape
    assert torch.equal(ref_discrete, got_discrete)
    atol = 1e-6
    assert torch.allclose(ref_dist, got_dist, atol=atol, rtol=0)
