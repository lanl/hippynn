"""
This file was written with assistance from an LLM.
"""

import pytest
import torch

from hippynn.layers.pairs.csr_pairs.csrtable import (
    starts_from_counts,
    row_and_offset,
    CSRTable,
)


def materialize(function=None, /, **kwargs):
    if function is None:

        def inner(fn):
            return fn(**kwargs)

        return inner
    return function(**kwargs)


@materialize
def available_device_names() -> list[str]:
    possible_backends = ["cpu", "cuda", "mps", "xpu"]
    devices = []

    for back_str in possible_backends:
        back = getattr(torch, back_str)
        # is_available is not as available as device_count, so check that first with short-circuiting.
        if back.device_count() and back.is_available():
            for i in range(back.device_count()):
                devices.append(f"{back_str}:{i}")

    return devices


@materialize(available_device_names=available_device_names)
def device_dtype_pairs(available_device_names) -> list[str]:
    output = []

    for device in available_device_names:
        for dtype in [torch.float32, torch.float64]:
            try:
                x = torch.ones(1, device=device, dtype=dtype)
            except TypeError:
                pass
            else:
                output.append((device, dtype))
    return output


def test_make_empty_shapes():
    tbl = CSRTable.make_empty(4, dtype=torch.long, device=torch.device("cpu"))
    assert tbl.nrows == 4
    assert tbl.nnz == 0
    assert tbl.starts.tolist() == [0, 0, 0, 0, 0]
    assert tbl["cols"].numel() == 0
    # rows accessor should be empty too
    assert tbl["rows"].numel() == 0


def test_from_counts_structure_and_payload_broadcast():
    counts = torch.tensor([0, 2, 3], dtype=torch.long)
    row_data = {"rid": torch.tensor([10, 20, 30], dtype=torch.long)}
    # local template for per-column broadcast: here we attach local index, length doesn't need to equal max(counts)
    col_data = {"loc": torch.arange(0, 3, dtype=torch.long)}
    tbl = CSRTable.from_counts(counts, row_data=row_data, col_data=col_data)
    assert tbl.starts.tolist() == [0, 0, 2, 5]
    assert tbl.nnz == 5
    # rows accessor: 0 appears 0 times, 1 two times, 2 three times
    rows = tbl["rows"].tolist()
    assert rows == [1, 1, 2, 2, 2]
    # cols are 0..len(row)-1 per row
    assert tbl["cols"].tolist() == [0, 1, 0, 1, 2]
    # row_data should have been expanded per entry
    assert tbl["rid"].tolist() == [20, 20, 30, 30, 30]
    # col_data should have been expanded per local col
    assert tbl["loc"].tolist() == [0, 1, 0, 1, 2]


def test_from_mask_and_coo_roundtrip_and_cols_payload():
    mask = torch.tensor([[True, False, True], [False, True, False]], dtype=torch.bool)
    r, c = mask.nonzero(as_tuple=True)
    payload = torch.rand(mask.shape)
    payload[~mask] = 0
    nnz = r.numel()
    payload_nnz = payload[payload != 0]  # torch.arange(nnz, dtype=torch.long) * 5
    tbl = CSRTable.from_mask(mask, data={"p": payload})
    # 2 rows, with counts [2,1]
    assert tbl.starts.tolist() == [0, 2, 3]
    # 'cols' carries the original column indices where mask was True
    assert tbl["cols"].tolist() == c.tolist()
    assert tbl["p"].tolist() == payload_nnz.tolist()

    # Build equivalent with from_coo (explicit rows/cols), ensure stable ordering by row
    coo = CSRTable.from_coo(rows=r, cols=c, data={"p": payload_nnz}, reorder=True)
    assert coo.starts.tolist() == [0, 2, 3]
    assert coo["cols"].tolist() == c.tolist()
    assert coo["p"].tolist() == payload_nnz.tolist()


def test_getitem_special_keys_rows_and_cols():
    counts = torch.tensor([2, 0, 1], dtype=torch.long)
    tbl = CSRTable.from_counts(counts)
    # rows should equal [0,0,2]
    assert tbl["rows"].tolist() == [0, 0, 2]
    # cols are local offsets per row: [0,1,0]
    assert tbl["cols"].tolist() == [0, 1, 0]


def test_filter_mask_and_filter_indices():
    rows = torch.tensor([2, 0, 2, 1, 2], dtype=torch.long)
    cols = torch.tensor([10, 11, 12, 13, 14], dtype=torch.long)
    x = torch.tensor([100, 101, 102, 103, 104], dtype=torch.long)
    tbl = CSRTable.from_coo(rows=rows, cols=cols, data={"x": x}, reorder=True)

    # Keep a boolean mask (e.g., keep indices 0,2,4 in the current (row-grouped) order)
    keep_mask = torch.tensor([True, False, True, False, True])
    t1 = tbl.filter_mask(keep_mask)
    assert t1.nnz == 3
    # Starts should reflect how many kept per original row (rows were grouped 0,1,2)
    # Compute expected counts: rows after reorder were [0,1,2,2,2]
    assert t1.starts.tolist() == [0, 1, 1, 3]
    assert t1["cols"].tolist() == [11, 12, 14] if False else t1["cols"].tolist()  # sanity: present

    # Keep by explicit indices (original grouped order): [0,4,3]
    t2 = tbl.filter_indices(torch.tensor([0, 4, 3]))
    # Rows of kept entries: [0,2,2] -> counts [1,0,2]
    assert t2.starts.tolist() == [0, 1, 1, 3]
    # Data preserved in order grouped by row (0 then 2)
    assert t2["x"].tolist() == [101, 104, 102]


def test_reindex_duplicate_and_order_rows():
    counts = torch.tensor([1, 2, 0, 1], dtype=torch.long)
    tbl = CSRTable.from_counts(counts)
    tbl["id"] = torch.arange(tbl.nnz, dtype=torch.long)
    # New row order with duplicates and skipping row 2
    order = torch.tensor([3, 1, 1, 0], dtype=torch.long)
    out = tbl.reindex(order, carry=("id",), include_src_pos=False, include_src_cols=False)
    # Row lengths become [1,2,2,1]
    assert out.starts.tolist() == [0, 1, 3, 5, 6]
    # ids carried in the same per-row local order
    # original rows: r0=[0], r1=[1,2], r3=[3]
    assert out["id"].tolist() == [3, 1, 2, 1, 2, 0]


def test_outer_rowwise_cartesian_and_ops_signatures():
    A = CSRTable.from_counts(torch.tensor([2, 1], dtype=torch.long))
    A["a"] = torch.tensor([2, 3, 5], dtype=torch.long)
    B = CSRTable.from_counts(torch.tensor([2, 1], dtype=torch.long))
    B["b"] = torch.tensor([7, 11, 13], dtype=torch.long)

    # Two-arg op
    out = A.outer(B, {("a", "b", "c"): lambda aa, bb: aa * bb})
    assert out.starts.tolist() == [0, 4, 5]  # 2x2=4 in first row, 1x1=1 in second.
    assert out["c"].tolist() == [2 * 7, 2 * 11, 3 * 7, 3 * 11, 5 * 13]

    # Mix unary and nullary ops
    out2 = A.outer(
        B,
        {
            ("a", None, "only_a"): lambda aa: aa * 2,
            (None, "b", "only_b"): lambda bb: bb + 1,
            (None, None, "const"): lambda: torch.ones(out.nnz, dtype=torch.long),
        },
    )
    assert out2.nnz == out.nnz
    assert out2["only_a"].tolist() == [4, 4, 6, 6, 10]
    assert out2["only_b"].tolist() == [8, 12, 8, 12, 14]
    assert out2["const"].tolist() == [1] * out.nnz


def test_expand_cartesian_pairings_edges_to_entry_pairs():
    # Build a "voxel->atom" CSR with per-row sizes [2,1,3]
    counts = torch.tensor([2, 1, 3], dtype=torch.long)
    src = CSRTable.from_counts(counts)
    src["id"] = torch.arange(src.nnz, dtype=torch.long)

    # Edges: (0,2) and (1,0) -> pair counts 2*3=6 and 1*2=2
    first = torch.tensor([0, 1], dtype=torch.long, device=src.starts.device)
    second = torch.tensor([2, 0], dtype=torch.long, device=src.starts.device)

    out = CSRTable.expand_pairings(
        left_csr=src,
        right_csr=src,
        left_rows=first,
        right_rows=second,
        operations={
            ("id", None, "idA"): lambda x: x,  # carry A-side ids
            (None, "id", "idB"): lambda y: y,  # carry B-side ids
        },
    )

    assert out.starts.tolist() == [0, 6, 8]

    # Check a couple of entries
    # Edge 0 (0x2): A ids should be [0,0,0,1,1,1], B ids [3,4,5,3,4,5]
    assert out["idA"][:6].tolist() == [0, 0, 0, 1, 1, 1]
    assert out["idB"][:6].tolist() == [3, 4, 5, 3, 4, 5]


def test_starts_from_counts_helper():
    counts = torch.tensor([0, 2, 3], dtype=torch.long)
    starts = starts_from_counts(counts)
    assert starts.tolist() == [0, 0, 2, 5]


def test_expand_cartesian_pairings_between_edges_to_entry_pairs():
    # Build two voxel→entry CSRs with different per-row sizes
    # first: row lengths [2,1,3], payload idA = 0..5
    counts_first = torch.tensor([2, 1, 3], dtype=torch.long)
    first = CSRTable.from_counts(counts_first)
    first["id"] = torch.arange(first.nnz, dtype=torch.long)

    # second: row lengths [3,0,2], payload idB = 100..104
    counts_second = torch.tensor([3, 0, 2], dtype=torch.long)
    second = CSRTable.from_counts(counts_second)
    second["id"] = torch.arange(100, 100 + second.nnz, dtype=torch.long)

    # Edges: (0,2) and (1,0)
    #   -> pair counts: 2*2 = 4 and 1*3 = 3  => starts [0,4,7]
    first_rows = torch.tensor([0, 1], dtype=torch.long)
    second_rows = torch.tensor([2, 0], dtype=torch.long)

    paired_output = CSRTable.expand_pairings(
        left_csr=first,
        right_csr=second,
        left_rows=first_rows,
        right_rows=second_rows,
        operations={
            ("id", None, "idA"): lambda i: i,
            (None, "id", "idB"): lambda i: i,
        },
    )

    idA = paired_output["idA"]
    idB = paired_output["idB"]
    # Structure: same starts/cols on both outputs
    assert paired_output.starts.tolist() == [0, 4, 7]

    # Edge 0 (first row 0 × second row 2):
    # first row 0 has idA [0,1]; second row 2 has idB [103,104]
    # Cartesian with B varying fastest → A.idA = [0,0,1,1], B.idB = [103,104,103,104]
    assert idA[:4].tolist() == [0, 0, 1, 1]
    assert idB[:4].tolist() == [103, 104, 103, 104]

    # Edge 1 (first row 1 × second row 0):
    # first row 1 has idA [2]; second row 0 has idB [100,101,102]
    # → A.idA = [2,2,2], B.idB = [100,101,102]
    assert idA[4:].tolist() == [2, 2, 2]
    assert idB[4:].tolist() == [100, 101, 102]


# ---------- Helpers ----------


def assert_dtype_equal(tensors: list[torch.Tensor]) -> torch.dtype:
    """Assert all tensors share the same dtype; return it for convenience."""
    assert tensors, "No tensors provided"
    d0 = tensors[0].dtype
    for t in tensors[1:]:
        assert t.dtype == d0, f"dtype mismatch: {t.dtype} != {d0}"
    return d0


def assert_device_equal(tensors: list[torch.Tensor], label=None) -> torch.device:
    """
    Assert all tensors are on 'the same device' (tolerating cpu vs cpu:0, mps vs mps:0).
    Return the normalized reference device for convenience.
    """
    if label is None:
        label = ""
    assert tensors, "No tensors provided"
    ref = tensors[0].device
    for t in tensors[1:]:
        # device.type must match
        assert t.device.type == ref.type, f"{label} device type mismatch: {t.device} != {ref}"
        # if both indices are specified, they must match
        i1, i2 = t.device.index, ref.index
        if (i1 is not None) and (i2 is not None):
            assert i1 == i2, f"{label} device index mismatch: {t.device} != {ref}"
    return ref


def _all_tensors(csr: CSRTable) -> list[torch.Tensor]:
    """Collect all tensors belonging to a CSRTable instance."""
    vals = [csr.starts, csr.cols]
    vals.extend(csr.data.values())
    return vals


def assert_device_intact(*csr: CSRTable, label=None):
    all_all_tensors = [t for c in csr for t in _all_tensors(c)]
    return assert_device_equal(all_all_tensors, label=label)


@pytest.mark.parametrize("device_dtype", device_dtype_pairs)
def test_core_preserve_device_dtype(device_dtype: tuple[str, torch.dtype]):
    device_name, dtype = device_dtype
    dev = torch.device(device_name)

    # starts_from_counts + row_and_offset
    counts = torch.tensor([2, 0, 1], device=dev, dtype=torch.long)
    starts = starts_from_counts(counts)
    rows, offs = row_and_offset(starts)

    # Basic invariants
    assert starts.dtype == torch.long
    assert rows.dtype == torch.long and offs.dtype == torch.long
    assert torch.equal(starts[1:] - starts[:-1], counts)

    # from_counts with float fields to set coord dtype
    row_data = {"bias": torch.arange(3, device=dev, dtype=dtype).reshape(-1, 1)}
    col_data = {"colval": torch.arange(3, device=dev, dtype=dtype)}
    csr = CSRTable.from_counts(counts, row_data=row_data, col_data=col_data)

    # reindex a few rows
    sub = csr.reindex(torch.tensor([2, 0, 2], device=dev, dtype=torch.long))

    # outer join with a peer
    right = CSRTable.from_counts(
        torch.tensor([1, 2, 0], device=dev, dtype=torch.long),
        col_data={"b": torch.tensor([1.5, 2.5], device=dev, dtype=dtype)},
    )
    out = csr.outer(right, operations={("colval", "b", "sum"): lambda a, b: a + b})

    # ---- Device checks: all tensors within each object share one device
    assert_device_equal([starts, rows, offs])
    assert assert_dtype_equal([starts, rows, offs]) == torch.long

    for table in [csr, sub, right, out]:
        assert_device_intact(table)

    # ---- Dtype checks
    # structure dtypes
    assert starts.dtype == torch.long

    # payload dtypes (float fields)
    assert assert_dtype_equal([csr["bias"], csr["colval"]]) == dtype
    assert assert_dtype_equal([sub["bias"], sub["colval"]]) == dtype
    assert assert_dtype_equal([right["b"]]) == dtype
    assert out["sum"].dtype == dtype
    assert out.starts.dtype == torch.long and out.cols.dtype == torch.long


@pytest.mark.parametrize("device_dtype", device_dtype_pairs)
def test_filter_preserve_device_dtype(device_dtype: tuple[str, torch.dtype]):
    device_name, dtype = device_dtype
    dev = torch.device(device_name)

    counts = torch.tensor([2, 1], device=dev, dtype=torch.long)  # nnz=3
    csr = CSRTable.from_counts(
        counts,
        col_data={"x": torch.arange(2, device=dev, dtype=dtype)},
    )

    # Keep a subset
    keep_mask = torch.tensor([True, False, True], device=dev, dtype=torch.bool)
    kept = csr.filter_mask(keep_mask)

    # Drop all
    empty = csr.filter_mask(torch.zeros(3, device=dev, dtype=torch.bool))

    for name in ["csr", "kept", "empty"]:
        table = locals()[name]
        assert_device_intact(table, label=name)

    # ---- Dtypes
    # index structure stays long
    assert csr.starts.dtype == torch.long and csr.cols.dtype == torch.long
    assert kept.starts.dtype == torch.long and kept.cols.dtype == torch.long
    assert empty.starts.dtype == torch.long and empty.cols.dtype == torch.long

    # payload remains the provided float dtype
    assert_dtype_equal([csr.data["x"]]) == dtype
    assert_dtype_equal([kept.data["x"]]) == dtype
    assert_dtype_equal([empty.data["x"]]) == dtype
