"""
This file was written with assistance from an LLM.


Neighbor list construction over triclinic periodic cells using PyTorch + CSR.

This module builds neighbor pairs for batched molecular/atomistic systems with
triclinic cells. It uses a light CSR “algebra” (see :mod:`csrtable`) to stage
row-wise joins and filtering:

Pipeline (high level)
---------------------
1) :func:`build_initial_data` — wrap raw inputs into CSRs for systems/atoms.
2) :func:`normalize_atoms` — wrap positions to fractional [0,1)^3 and record integer image offsets.
3) :func:`build_image_offsets` — compute per-system image-shift stencils large enough for ``cutoff``.
4) :func:`build_image_atoms` — cartesian-expand primary atoms × image shifts.
5) :func:`voxelize_images` — voxel grid per system with edge ≈ ``cutoff`` and assign image-atoms to voxels.
6) :func:`voxel_adjacency` — build voxel–voxel edges via a fixed stencil (includes self-edges).
7) :func:`expand_pairs` — expand voxel edges into candidate atom pairs.
8) :func:`prune_pairs` — filter by cutoff; drop same-atom self-edges for primary images.
9) :func:`calc_neighbors` — orchestration entry point that returns final pairs.

All tensor operations are batched and device/dtype agnostic. Index tensors are
``torch.long``. Distances/displacements remain differentiable w.r.t. input positions.
"""

import torch
from typing import Tuple

from .csrtable import CSRTable, row_and_offset, find_indices


def build_initial_data(positions, nonblank, cells, cutoff):
    """Package raw inputs (positions/mask/cells/cutoff) into CSR containers.

    Parameters
    ----------
    positions : Tensor, shape ``[S, A, 3]``, float
        Cartesian positions per system (padded to a common ``A``). Can be
        ``float32``/``float64``. Unused (masked) atoms are ignored via ``nonblank``.
    nonblank : BoolTensor, shape ``[S, A]``
        Mask of real atoms per system. ``True`` entries become CSR rows' entries.
    cells : Tensor, shape ``[S, 3, 3]``, float
        Triclinic cell matrices (row- or column-major consistent with einsum ops).
    cutoff : Tensor ``[S]`` or ``float``
        Radial cutoff per system; a scalar is broadcast to all systems.

    Returns
    -------
    atomCSR : CSRTable
        Rows = systems; entries = real atoms. Carries at least:
        ``"raw_positions" : [nnz,3]``, ``"atom_gid" : [nnz]`` (global id in the
        padded array), and any fields required downstream.
    systemCSR : CSRTable
        One entry per system (``counts = 1``). Carries:
        ``"cells" : [S,3,3]`` and ``"cutoff" : [S]``.

    Notes
    -----
    * ``atom_gid`` preserves a mapping back to the original padded indices.
    * No heavy validation is performed here.

    Complexity
    ----------
    O(S*A) to build the mask-based CSR.
    """

    device = positions.device
    coord_dtype = positions.dtype

    n_sys = cells.shape[0]
    assert n_sys == positions.shape[0], f"number of systems not identical ({n_sys} vs {positions.shape[0]})"

    if isinstance(cutoff, float):
        cutoff = torch.as_tensor(cutoff, dtype=coord_dtype, device=device)
    if cutoff.ndim == 0:
        cutoff = cutoff.expand(n_sys)
    assert cutoff.shape[0] == n_sys
    sys_counts = torch.ones(n_sys, device=device, dtype=torch.long)
    systemCSR = CSRTable.from_counts(sys_counts, row_data={"cells": cells, "cutoff": cutoff})
    atomCSR = CSRTable.from_mask(nonblank, data={"raw_positions": positions})
    # global atom id, including padding, to carry on for tracking.
    atomCSR["atom_gid"] = torch.arange(atomCSR.nnz, dtype=torch.long, device=device)

    return atomCSR, systemCSR


def normalize_atoms(
    atomCSR: CSRTable,  # [n_systems, n_atoms_max, 3]
    systemCSR: CSRTable,  # [n_systems, n_atoms_max] bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Wrap atoms into the primary triclinic cell and record image offsets.

    For each atom with Cartesian position ``x`` and cell ``H``, compute
    fractional ``f = H^{-1} x``, integer image ``k = floor(f)``, wrapped
    fractional ``f' = f - k`` in ``[0,1)^3``, and wrapped Cartesian
    ``x' = H f'``. Store these plus ``H^{-1}`` for later use.

    Parameters
    ----------
    atomCSR : CSRTable
        Must carry ``"raw_positions" : [nnz,3]`` and rows aligned to systems.
    systemCSR : CSRTable
        Must carry ``"cells" : [S,3,3]``. ``cells`` are inverted in-place and
        stored as ``"cells_inv"``.

    Returns
    -------
    atomCSR : CSRTable
        With added fields:
        ``"positions" : [nnz,3]`` (wrapped Cartesian),
        ``"offsets" : [nnz,3]`` (integer images, ``long``).
    systemCSR : CSRTable
        With added field ``"cells_inv" : [S,3,3]``.

    Determinism
    -----------
    Purely functional given inputs; no randomness.

    Autograd
    --------
    ``positions`` remain differentiable w.r.t. input positions; ``offsets``/``cells_inv`` are non-differentiable.
    """

    positions = atomCSR["raw_positions"]
    dtype = positions.dtype

    # Batched fractional coords and wrapping (triclinic)
    cells = systemCSR["cells"]
    systemCSR["cells_inv"] = cells_inv = torch.linalg.inv(cells)  # [n_systems,3,3]

    sys_ID = atomCSR["rows"]

    frac = torch.einsum("ax, axb->ab", positions, cells_inv[sys_ID])  # [n_atoms,3]
    offsets = torch.floor(frac).to(torch.long)  # integer images
    frac_wrapped = frac - offsets.to(dtype)  # [n_systems,n_atoms_max,3] in [0,1)
    positions_wrapped = torch.einsum("ab, abx->ax", frac_wrapped, cells[sys_ID])  # back to Cartesian

    atomCSR["positions"] = positions_wrapped
    atomCSR["offsets"] = offsets

    # recon = (frac_wrapped + offsets.to(frac_wrapped.dtype)) @ systemCSR["cells"][sys_ID]
    # assert (recon - positions).abs().max() <= (1e-5 if positions.dtype==torch.float32 else 1e-10), "Failed reconstruction!!"

    return atomCSR, systemCSR


def build_image_offsets(systemCSR: CSRTable, use_full_stencil=True) -> CSRTable:
    """Compute per-system periodic image offsets sufficient for ``cutoff``.

    The stencil size per axis is
    ``n_rep[k] = ceil(cutoff * ||(H^{-1})_{k,:}||_2)`` with a symmetric range
    ``{-n_rep[k], ..., 0, ..., +n_rep[k]}``. The Cartesian shift for an integer
    offset ``k`` is ``-k @ H`` (so that adding it moves an image back into the
    primary cell).

    Parameters
    ----------
    systemCSR : CSRTable
        Rows = systems; requires ``"cells" : [S,3,3]`` and ``"cutoff" : [S]``.
        If present, ``"cells_inv"`` is reused; otherwise it is computed.
    use_full_stencil : bool, default ``True``
        If ``True`` use the full symmetric 3D product. (Half-stencils or sparse
        variants can be added later.)

    Returns
    -------
    imageCSR : CSRTable
        Rows = systems; entries = image shifts for that system. Carries:
        ``"image_offsets" : [nnz,3]`` (long offsets),
        ``"shift" : [nnz,3]`` (Cartesian shift to apply),
        ``"is_primary" : [nnz]`` (True for ``k == (0,0,0)``).

    Complexity
    ----------
    O(S) to compute per-axis replication counts; O(total images) to materialize.
    """

    if not use_full_stencil:
        # TODO: Optional: Implement different stencils.
        # # Calculate nonzero delta entries
        # save dsome code which does this.
        # nonzero = all_deltas.abs().to(torch.bool)
        # # A mask where the first nonzero nonzero element is.
        # first_nonzero_mask = nonzero & (nonzero.to(torch.long).cumsum(dim=1)==1) # 2D mask of first nonzero values for each row.
        # # Extract VALUE of first nonzero mask; mask out first values, and then clamp+bool to make indicator positive ones.
        # # Run "any" to reduce and find if the first value was positive or not.
        # first_nonzero_positive = (all_deltas*first_nonzero_mask).clamp(min=0).to(torch.bool).any(dim=1)
        # one_way_deltas = first_nonzero_positive | (all_deltas==0).all(dim=1)
        # deltas = all_deltas[one_way_deltas]

        raise ValueError("Not implemented!")

    cells = systemCSR["cells"]
    n_systems = cells.shape[0]
    device = cells.device
    coord_dtype = cells.dtype

    systemCSR["reciprocol_norms"] = row_norms = torch.linalg.norm(systemCSR["cells_inv"], dim=1)

    n_rep_per_axis = torch.ceil(systemCSR["cutoff"].unsqueeze(-1) * row_norms).to(torch.long).clamp(min=1)

    # replication counts for periodic boundaries
    per_axis_counts = 2 * n_rep_per_axis + 1
    counts = per_axis_counts.prod(dim=1)

    # build the CSR for the repetitions, data fields will be populated
    # after explicit construction.
    imageCSR = CSRTable.from_counts(counts=counts)

    system, t = row_and_offset(imageCSR.starts)

    # unravel the index of each shift in the overall count
    ax_s, ay_s, az_s = per_axis_counts[system].unbind(1)
    ayz = ay_s * az_s
    ix = t // ayz
    rem = t % ayz
    iy = rem // az_s
    iz = rem % az_s

    # re-center the shifts
    kx = ix - n_rep_per_axis[system, 0]
    ky = iy - n_rep_per_axis[system, 1]
    kz = iz - n_rep_per_axis[system, 2]
    image_offsets = torch.stack((kx, ky, kz), dim=1)  # [nnz,3] long

    shift = -torch.einsum("pk,pkj->pj", image_offsets.to(coord_dtype), cells[system])
    is_primary = (image_offsets == 0).all(dim=1)

    imageCSR["image_offsets"] = image_offsets
    imageCSR["shift"] = shift
    imageCSR["is_primary"] = is_primary
    return imageCSR


def calculate_aabb(positions: torch.Tensor, system_id: torch.Tensor, n_systems: int):
    """Compute per-system axis-aligned bounding boxes (AABB).

    Parameters
    ----------
    positions : Tensor, shape ``[M, 3]`` (subset of atoms/images)
        Cartesian coordinates to bound.
    system_id : LongTensor, shape ``[M]``
        Owning system id for each position.
    n_systems : int
        Total number of systems.

    Returns
    -------
    mins : Tensor, shape ``[S, 3]``
        Per-system minima; systems without entries return zeros.
    maxs : Tensor, shape ``[S, 3]``
        Per-system maxima; systems without entries return zeros.

    Notes
    -----
    Uses ``scatter_reduce_`` with ``amin``/``amax``. Handles empty systems by
    emitting ``[0,0,0]`` for both min/max.
    """

    dtype = positions.dtype
    device = positions.device
    mins = torch.full((n_systems, 3), float("inf"), device=device, dtype=dtype)
    maxs = torch.full((n_systems, 3), float("-inf"), device=device, dtype=dtype)
    sys3 = system_id.unsqueeze(1).expand(-1, 3)  # [M,3] # each column xyz reduced in parallel
    # include_self=True works with ±inf init and covers empty systems safely
    mins.scatter_reduce_(0, sys3, positions, reduce="amin", include_self=True)
    maxs.scatter_reduce_(0, sys3, positions, reduce="amax", include_self=True)

    ## maybe not needed at all?
    # If a system has no entries, give it a span of [0,0]
    system_present = torch.zeros(n_systems, dtype=torch.bool, device=device)
    system_present[system_id] = True
    empty_mask = ~system_present
    if empty_mask.any():
        mins = mins.clone()
        maxs = maxs.clone()
        mins[empty_mask] = 0
        maxs[empty_mask] = 0

    return mins, maxs


def build_image_atoms(
    atomCSR: CSRTable,  # rows are systems, cols are atoms in system
    imageCSR: CSRTable,  # rows are displaced images of systems
    systemCSR: CSRTable,  # rows are systems
) -> CSRTable:
    """Expand primary atoms across image offsets (row-wise outer join).

    Parameters
    ----------
    atomCSR : CSRTable
        Rows = systems; requires fields
        ``"positions":[N,3]``, ``"atom_gid":[N]``, ``"offsets":[N,3]`` (base offsets).
    imageCSR : CSRTable
        Rows = systems; requires fields
        ``"image_offsets":[I,3]``, ``"shift":[I,3]``, ``"is_primary":[I]``.
    systemCSR : CSRTable
        Rows = systems; requires ``"cells"`` and ``"cutoff"`` (used for culling).

    Returns
    -------
    image_atomCSR : CSRTable
        Rows = systems; entries = atoms×images. Carries:
        ``"positions":[M,3]`` (shifted),
        ``"atom_gid":[M]``,
        ``"offsets":[M,3]`` (total offset for that image),
        ``"is_primary":[M]``,
        ``"system":[M]`` (system id per entry).

    Pruning
    -------
    A coarse AABB check removes far-away images that cannot generate neighbors
    within ``cutoff`` (per system).

    Complexity
    ----------
    O(N*I) per system in the worst case before pruning; typically much less after AABB culling.
    """

    # ---------- Outer (row-wise Cartesian join): atomCSR × imageCSR ----------
    image_atomCSR = atomCSR.outer(
        imageCSR,
        operations={
            # fmt: off
            # Shift positions using the shift of the image
            ("positions", "shift", "positions"):     lambda p, s: p + s,  # floatX
            # Offsets shift like positions
            ("offsets", "image_offsets", "offsets"): lambda ka, kd: ka + kd,  # long
            # Retain atom id
            ("atom_gid", None, "atom_gid"):          lambda f: f,  # long
            # Inherit primary images from image_cellCSR
            (None, "is_primary", "is_primary"):      lambda ip: ip,  # bool
            # Inherit system number from the atomCSR
            ("rows", None, "system"):                lambda r: r,  # long
            # fmt: on
        },
    )

    primary_positions = atomCSR["positions"]
    primary_systems = atomCSR["rows"]
    n_systems = systemCSR["rows"].shape[0]

    mins, maxs = calculate_aabb(primary_positions, primary_systems, n_systems)
    # broadcast back to images
    image_system = image_atomCSR["system"]
    image_positions = image_atomCSR["positions"]

    mins = mins[image_system]
    maxs = maxs[image_system]

    gap_1 = image_positions - maxs  # positive if image-atom is on right side of box
    gap_2 = mins - image_positions  # positive of image-atom is on left side of box

    # smallest positive gap wins, if both are negative, the particle is in the box range for that axis.
    # total_gap = gap_2.clamp(min=gap_1).clamp(min=0) # alternate calculation
    total_gap = torch.maximum(torch.maximum(gap_1, gap_2), torch.zeros_like(gap_1))
    dist_to_box = total_gap.norm(dim=-1)

    image_cutoffs = systemCSR["cutoff"][image_system]

    eps = 1e-5
    close_enough_to_aabb = dist_to_box < image_cutoffs * (1 + eps)

    # Drop images which cannot be relevant
    image_atomCSR = image_atomCSR.filter_mask(close_enough_to_aabb)

    return image_atomCSR


def voxelize_images(
    image_atomCSR: CSRTable,  # nnz-aligned fields: "positions","atom_gid","offsets","is_primary","system"
    systemCSR: CSRTable,
) -> Tuple[CSRTable, CSRTable, torch.Tensor]:
    """Partition image-atoms into near-cubic voxels of edge ≈ cutoff.

    Parameters
    ----------
    image_atomCSR : CSRTable
        Must carry (at least) ``"positions","atom_gid","offsets","is_primary","system"``.
    systemCSR : CSRTable
        Must carry per-system ``"cutoff"``. Outputs grid metadata into this CSR.

    Returns
    -------
    voxel_atomCSR : CSRTable
        Rows = global voxel ids; entries = image-atoms in that voxel. Carries the
        same fields, plus a per-entry ``"voxel_gid"`` if helpful downstream.
    voxelCSR : CSRTable
        Rows = systems; one entry per voxel. Carries:
        ``"v":[V,3]`` (local voxel coords),
        ``"m":[V,3]`` (per-system voxel grid shape),
        ``"s":[V,3]`` (local strides),
        ``"voxel_gid":[V]``,
        ``"is_primary_voxel":[V]`` (voxel contains at least one primary image-atom).
    systemCSR : CSRTable
        Augmented with per-system fields, e.g.:
        ``"voxel_grid_shape":[S,3]``, ``"voxel_origin":[S,3]``.

    Notes
    -----
    * A small skin factor inflates the voxel edge slightly to avoid numerical
      misses on boundaries.
    * Local voxel ids are unraveled/raveled with per-system strides.

    Complexity
    ----------
    O(M) to assign atoms to voxels; O(V) to materialize per-system voxel metadata.
    """
    # ---- shapes / device / dtype up-front ----
    cutoff = systemCSR["cutoff"]
    dtype = cutoff.dtype
    device = cutoff.device
    n_systems = systemCSR["rows"].shape[0]

    # calculate bounding boxes per system
    positions = image_atomCSR["positions"]  # [M,3]
    system_id = image_atomCSR["system"]  # [M]
    mins, maxs = calculate_aabb(positions, system_id, n_systems)

    # calculate grid shape per system
    span = maxs - mins
    per_system_grid_shape = torch.ceil(span / cutoff.unsqueeze(-1)).to(torch.long).clamp(min=1)  # [n_systems,3]
    systemCSR["voxel_grid_shape"] = per_system_grid_shape

    # setup voxel coordinate system.
    voxels_per_system = per_system_grid_shape.prod(dim=1)  # [n_systems]
    # print("Voxels per system", voxels_per_system)

    systemCSR["n_voxels_per_system"] = voxels_per_system
    # calculate voxel id offsets for each system
    voxel_offset = torch.zeros(n_systems+1, device=device,dtype=torch.long)
    torch.cumsum(voxels_per_system,0,out=voxel_offset[1:])
    voxel_offset = voxel_offset[:-1]
    systemCSR["voxel_offset"] = voxel_offset

    mx, my, mz = per_system_grid_shape.unbind(1)
    # We index the voxels in C-order:
    per_system_voxel_strides = torch.stack([mz * my, mz, torch.ones_like(mz)], dim=-1)
    # If you want to switch to Fortran order:
    # per_system_voxel_strides = torch.stack([torch.ones_like(mx), mx, mx * my], dim=-1)
    # (either order is fine.)

    systemCSR["voxel_strides"] = per_system_voxel_strides


    # Now calculate which voxel each image-atom falls into.

    origin_per_entry = mins[system_id]  # [M,3]
    dims_per_entry = per_system_grid_shape[system_id]  # [M,3]

    # tiny percentage extra in voxel size to ensure
    # all pairs can be found regardless of numerical noise
    # also should handle rightmost voxel coordinate being in range.
    skin_ratio = 1e-4
    cutoff_with_skin = cutoff * (1 + skin_ratio)

    voxel_nondim_coords = (positions - origin_per_entry) / cutoff_with_skin[system_id].unsqueeze(-1)
    # clamp_min here handles if the leftmost point in the box encounters numerical noise.
    voxel_indices = torch.floor(voxel_nondim_coords).to(torch.long).clamp_min(0)

    assert (voxel_indices >= 0).all() and (voxel_indices < dims_per_entry).all(), "Voxel index invalid."

    per_imageatom_strides = per_system_voxel_strides[system_id]
    # could be a dot product but not usually efficient for very dot index:
    local_voxel_linear_id = (voxel_indices * per_imageatom_strides).sum(dim=-1)  # [M]

    # Re-index atoms into a voxel-aligned CSR (was in system-aligned)
    global_voxel_id_per_entry = voxel_offset[system_id] + local_voxel_linear_id  # [M]

    occupied_voxel_ids, voxel_index_per_atom = torch.unique(global_voxel_id_per_entry, return_inverse=True, sorted=True)
    n_voxels = occupied_voxel_ids.shape[0]
    voxel_arange = torch.arange(n_voxels,device=device,dtype=torch.long)

    data = {k: image_atomCSR[k] for k in ("positions", "atom_gid", "offsets", "is_primary", "system")}
    data["voxel_id"] = global_voxel_id_per_entry
    voxel_atomCSR = CSRTable.from_coo(
        rows=voxel_index_per_atom,
        cols=torch.arange(image_atomCSR.nnz, dtype=torch.long, device=device), # each atom has its own column; we're just grouping rows.
        data=data,
        #nrows=voxelCSR.nnz,
        reorder=True,
    )

    
    # Note: the below scheme for populating voxel information looks like a race condition.
    # Actually we're just writing the atom data back to the voxel table, 
    # instead of re-computing all the voxel info based on the GID.
    
    occupied_voxel_sys = torch.full((n_voxels,),-1, dtype=torch.long, device=device)
    occupied_voxel_sys[voxel_index_per_atom] = system_id 
    occupied_voxel_stride = torch.full((n_voxels,3),-1, dtype=torch.long, device=device)
    occupied_voxel_stride[voxel_index_per_atom] = per_imageatom_strides
    occupied_voxel_dims = torch.full((n_voxels,3),-1, dtype=torch.long, device=device)
    occupied_voxel_dims[voxel_index_per_atom] = dims_per_entry
    
    occupied_voxel_local_id = torch.full((n_voxels,),-1, dtype=torch.long, device=device)
    occupied_voxel_local_id[voxel_index_per_atom] = local_voxel_linear_id
    occupied_voxel_coord = torch.full((n_voxels,3),-1, dtype=torch.long, device=device)
    occupied_voxel_coord[voxel_index_per_atom] = voxel_indices

    # The other strategy (could be implemented if needed for perforamnce)
    # would be to unwrap the voxel offset and local id, then
    # use offset to find the system -> strides, dims
    # use local id -> coord
    # actually that is irrelevant as each index is set to the same value;
    # all atoms belonging to a given voxel also belong to a given system. 

    # Conversion to bool cannot be done before reduce because 
    # cuda backend cannot reduce bools; reduce using bool if available,
    # but with int on cuda.
    
    accum_dtype = torch.long if device.type == "cuda" else torch.bool

    # Note that this data is not constant on atoms so we must do a 
    # safe reduction using max.
    primary_vox_mask = torch.zeros(n_voxels, dtype=accum_dtype, device=device)
     
    primary_vox_mask.scatter_reduce_(
        0,
        voxel_index_per_atom,
        image_atomCSR["is_primary"].to(accum_dtype),
        reduce="amax",
        include_self=False,
    )
    
    primary_vox_mask = primary_vox_mask.to(torch.bool)
    
    # ^^^
    # Note for future: if you are tempted to get "is_priamry" from
    # the voxel_atomCSR then make sure to reduce on the rows of that CSR,
    # as building it re-orders the atoms and puts the data in a different order than
    # the array voxel_index_per_atom
    
    data={
        "voxel_gid":occupied_voxel_ids,
        "v":occupied_voxel_coord,
        "s":occupied_voxel_stride,
        "m":occupied_voxel_dims,
        "is_primary_voxel":primary_vox_mask,
        }
    voxelCSR = CSRTable.from_coo(rows=occupied_voxel_sys, cols=voxel_arange,data=data)

    
    return voxel_atomCSR, voxelCSR, systemCSR


def voxel_adjacency(
    voxelCSR: CSRTable,
    systemCSR: CSRTable,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build voxel–voxel edges using a fixed stencil (includes self-edges).

    Assumptions
    -----------
    * PBC are handled by explicit image expansion; only in-bounds checks are needed.
    * Keep an edge if at least one endpoint voxel is primary (cheap pruning).
    * The stencil can be full (e.g., 3×3×3) or reduced; current code emits a
      symmetric set including (0,0,0).

    Parameters
    ----------
    voxelCSR : CSRTable
        Rows = systems; entries = voxels with fields
        ``"v","m","s","voxel_gid","is_primary_voxel"``.
    systemCSR : CSRTable
        Provides per-system voxel grid shapes.

    Returns
    -------
    first_global : LongTensor, shape ``[E]``
        Global voxel ids for the first endpoint.
    second_global : LongTensor, shape ``[E]``
        Global voxel ids for the second endpoint.

    Complexity
    ----------
    O(V * stencil_size), typically small constant factor per voxel.
    """
    device = voxelCSR.cols.device
    per_system_grid_shape = systemCSR["voxel_grid_shape"]

    n_systems = per_system_grid_shape.shape[0]

    # ---- Stencil: packed [S,3]; include (0,0,0) so intra-voxel pairs are produced ----
    base = torch.tensor([-1, 0, 1], device=device, dtype=torch.long)

    deltas = torch.cartesian_prod(base, base, base)  # [27,3]
    S = deltas.shape[0]

    # Same stencil is applied to each system.
    stencilCSR = CSRTable.from_counts(
        counts=torch.full((n_systems,), S, dtype=torch.long, device=device),
        col_data={"delta": deltas},  # packed [S,3]
    )

    # since the first particle is always primary, don't compute shifts from non-primary voxels,
    # only from primary ones. This will reduce excess rows in the pairing table.
    primary_voxelCSR = voxelCSR.filter_mask(voxelCSR["is_primary_voxel"])

    # ---- OUTER: voxelCSR × stencilCSR -> candidate neighbors per voxel ----
    cand = primary_voxelCSR.outer(
        stencilCSR,
        operations={
            # fmt: off
            ("v",   "delta", "v2"):  lambda v, d: v + d,  # neighbor coords [E,3]
            #("v",   None, "v1"):           lambda v:v,          # source voxel indices
            ("voxel_gid", None,    "gid1"):   lambda gid: gid,     # source voxel gid
            ("rows",      None,    "system"):  lambda r: r,         # system id
            ("m",      None,    "m"):          lambda m: m,         # voxel dims [E,3]
            #(None,"delta", "d"): lambda d: d, # keep delta
            ("s", None, "s"):                  lambda s:s           # strides for each voxel.       
            # fmt: on
        },
    )

    
    # Drop neighbors voxels that are not in bounds.
    # ;PBC is not required since image atoms are construted explictly)
    # ;;If you don't do this you will get spurious duplicate instances of the second voxel as well as
    # ;;invalid/corrupt voxel gids)
    v2 = cand["v2"]
    m = cand["m"]
    in_bounds = ((v2 >= 0) & (v2 < m)).all(dim=1)
    cand = cand.filter_mask(in_bounds)

    second_local = (cand["s"] * cand["v2"]).sum(dim=-1)  # reconstruct the per-system voxel ID numbers

    # reconstruct the global voxel ID numbers
    first_global = cand["gid1"]
    second_global = systemCSR["voxel_offset"][cand["system"]] + second_local

    # drop locations where the second voxel index is unpopulated.
    gids = voxelCSR["voxel_gid"]
    where_second_nonempty, second_indices = find_indices(second_global, gids)
    first_global = first_global[where_second_nonempty]    

    # re-find the voxel index for the first voxel in the pair, as well.
    where_first_valid, first_indices = find_indices(first_global, gids)
    assert where_first_valid.shape[0] == first_global.shape[0], f"Not all first voxels were valid!"

    return first_indices, second_indices


def expand_pairs(voxel_atomCSR, first_vox, second_vox):
    """Expand voxel edges to candidate atom pairs.

    Parameters
    ----------
    voxel_atomCSR : CSRTable
        Rows = global voxel ids; contains per-entry fields for image-atoms.
    first_vox : LongTensor, shape ``[E]``
        Global voxel ids (left endpoints).
    second_vox : LongTensor, shape ``[E]``
        Global voxel ids (right endpoints).

    Returns
    -------
    pairs : CSRTable
        Rows = edges (one row per ``(first_vox[i], second_vox[i])``), entries =
        Cartesian product of atoms from the two voxels. Carries at least:
        ``"idA","idB","posA","posB","rel_k","is_primary_B"`` and any additional
        payload required for pruning.

    Notes
    -----
    Implemented via :meth:`CSRTable.expand_pairings`/``outer``-style joins for
    the specified voxel pairings.
    """

    primary_voxelCSR = voxel_atomCSR.filter_mask(voxel_atomCSR["is_primary"])

    # 6) build candidate pairs (new API: single CSR with mapped payloads)
    pairs = CSRTable.expand_pairings(
        left_csr=primary_voxelCSR,
        right_csr=voxel_atomCSR,
        left_rows=first_vox,
        right_rows=second_vox,
        operations={
            # fmt: off
            ("positions", None, "posA"):                lambda x:x,
            (None, "positions", "posB"):                lambda x:x,
            ("positions", "positions", "displacement"): lambda x,y:y-x,
            ("atom_gid",   None,        "idA"):         lambda x: x,
            (None,         "atom_gid",  "idB"):         lambda y: y,
            ("offsets",    "offsets",  "offsets"):      lambda a,b:b-a,
            (None,         "is_primary", "primB"):      lambda y: y,
            ("system",     None,        "system"):      lambda x: x,
            # fmt: on
        },
    )

    pairs["distance"] = pairs["displacement"].norm(dim=-1)

    return pairs


def prune_pairs(pairs):
    """Apply geometric/pruning predicates to candidate pairs.

    Predicates
    ----------
    1) Radial cutoff: ``||posB - posA|| < cutoff`` (per-system).
    2) Self-edge removal: when ``idA == idB``, drop if ``is_primary_B`` is True
       (i.e., the primary image would create a trivial self-connection).

    Parameters
    ----------
    pairs : CSRTable
        Must carry fields:
        ``"idA","idB","posA","posB","rel_k","cutoff","is_primary_B"`` (names may vary
        slightly depending on upstream ops, but semantics match above).

    Returns
    -------
    idA : LongTensor, shape ``[K]``
    idB : LongTensor, shape ``[K]``
    rel_k : LongTensor, shape ``[K,3]``
    dist : Tensor, shape ``[K]``
    disp : Tensor, shape ``[K,3]``

    Autograd
    --------
    ``dist`` and ``disp`` are differentiable; discrete tensors are not.

    Complexity
    ----------
    O(E_pairs) with simple boolean masking and a single norm.
    """
    pass


def calc_neighbors(
    positions: torch.Tensor,  # [n_systems, n_atoms_max, 3] float
    nonblank: torch.Tensor,  # [n_systems, n_atoms_max] bool
    cells: torch.Tensor,  # [n_systems, 3, 3] float
    cutoff: float,
    use_full_stencil: bool = True,
    return_displacements=True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full neighbor-list pipeline (batched triclinic PBC).

    Pipeline Structure:
      1) normalize_atoms      -> wrapped positions + base image offsets (integer)
      2) build_image_offsets  -> per-system dynamic image CSR with ("image_offsets","shift","is_primary")
      3) build_image_atoms    -> CSR over systems with ("positions","atom_gid","offsets","is_primary","system")
      4) voxelize_images      -> (voxel_atomCSR, voxelCSR, per_system_grid_shape)
      5) voxel_adjancency     -> (first_voxel_id, second_voxel_id), includes (0,0,0)
      6) expand edges -> pairs via CSRTable.expand_cartesian_pairings on voxel_atomCSR
      7) prune: distance < cutoff, drop self-connections

    Parameters
    ----------
    positions : Tensor, shape ``[S, A, 3]``
        Cartesian positions per system, padded to ``A``.
    nonblank : BoolTensor, shape ``[S, A]``
        Mask indicating valid atoms.
    cells : Tensor, shape ``[S, 3, 3]``
        Triclinic cell matrices per system.
    cutoff : Tensor ``[S]`` or ``float``
        Per-system radial cutoff; a scalar is broadcast.
    use_full_stencil : bool, default ``True``
        Controls the image offset stencil in :func:`build_image_offsets`.
    return_displacements: bool, default ``True``
        If true, return distances and displacments as well as discrete indices.

    Returns
    -------
    idA : LongTensor, shape ``[K]``
        Source atom global ids (within the padded input).
    idB : LongTensor, shape ``[K]``
        Target atom global ids.
    rel_k : LongTensor, shape ``[K, 3]``
        Integer image offset for ``B`` relative to ``A``.
    dist : Tensor, shape ``[K]``
        Euclidean distances ``||posB - posA||``.
    disp : Tensor, shape ``[K, 3]``
        Displacements ``posB - posA`` in Cartesian coordinates.

    Guarantees
    ----------
    * Deterministic ordering given fixed inputs and a fixed CSR ordering policy.
    * Includes cross-image neighbors due to explicit image expansion.
    * Excludes trivial self-edges as described in :func:`prune_pairs`.

    Complexity
    ----------
    Roughly linear in the number of candidate pairs after voxel pruning; memory
    peaks during pair expansion and can be controlled by voxel granularity.

    Examples
    --------
    >>> idA, idB, k, dist, disp = calc_neighbors(positions, nonblank, cells, cutoff)

    """

    ##TODO Check call sites for cutoff variable which used to use float, now use per-system tensor.
    # Write test for per-system cutoffs
    device = positions.device

    ### TODO: use integer cell transforms to reduce skew during neighbor calculations!
    ### Remember to map back at the end?

    # shape tests: (Not testing this can cause silent errors)
    assert positions.shape[0] == nonblank.shape[0], "mismatched batch size"
    assert positions.shape[0] == cells.shape[0], "mismatched batch size"
    assert positions.shape[1] == nonblank.shape[1], "mismatched atom size"

    # 0) set up data structures
    atomCSR, systemCSR = build_initial_data(positions, nonblank, cells, cutoff)

    # 1) wrap + base offsets

    atomCSR, systemCSR = normalize_atoms(atomCSR, systemCSR)

    # 2) dynamic image offsets per system (nnz-aligned fields on returned CSR)
    # TODO: accept mixed boundary conditions.
    imageCSR = build_image_offsets(systemCSR, use_full_stencil=use_full_stencil)

    # 3) expand primary atoms to images (joins atoms × imageCSR)
    image_atomsCSR = build_image_atoms(atomCSR, imageCSR, systemCSR)

    voxel_atomCSR, voxelCSR, systemCSR = voxelize_images(image_atomsCSR, systemCSR)

    # Construct pairs of voxels where:
    # 1) they are adjvacent to each other (or same as each other)
    # 2) one of them contains primary atoms
    # TODO Add logic to only perform this on filled voxels. Or maybe scale voxels
    # per system so that we don't end up with many empty ones.
    # Something to address bad scaling when systems get very sparse!
    # (# of voxels scales like size of space, not size of particles)
    first_vox, second_vox = voxel_adjacency(voxelCSR, systemCSR)

    with torch.no_grad():
        pairs = expand_pairs(voxel_atomCSR=voxel_atomCSR, first_vox=first_vox, second_vox=second_vox)

    posA = pairs["posA"]  # [P,3]
    posB = pairs["posB"]  # [P,3]
    prim_B = pairs["primB"]  # [P]
    idA = pairs["idA"].to(torch.long)
    idB = pairs["idB"].to(torch.long)
    rel_k = pairs["offsets"]
    sys = pairs["system"]

    with torch.no_grad():

        pair_cutoffs = systemCSR["cutoff"][sys]
        keep = pairs["distance"] <= pair_cutoffs
        diff_atoms = idA != idB
        # For same atom pairs, we need to drop the self-connection,
        # so if B is the primary (same position as A) then we drop it.
        keep &= diff_atoms | ~prim_B

    # keep_ratio = keep.to(torch.long).sum().item()/keep.numel()
    # print(f"{keep_ratio=}")

    idA = idA[keep]
    idB = idB[keep]
    posA = posA[keep]
    posB = posB[keep]
    rel_k = rel_k[keep]
    sys = sys[keep]

    outs = (idA, idB, sys, rel_k)

    if return_displacements:
        # Only recalculate if the calling code asks for it.
        # (Layer wrapper does not; tests do)
        disp = posA - posB
        dist = torch.norm(disp, dim=1)
        outs = *outs, dist, disp

    return outs
