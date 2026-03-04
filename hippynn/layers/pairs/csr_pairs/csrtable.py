"""
This file was written with assistance from an LLM.

Light-weight CSRTable container built on PyTorch tensors and utilities.

This module defines:
- :func:`starts_from_counts` — fast CSR row-pointer construction.
- :func:`row_and_offset` — decode flat indices to (row, in-row offset).
- :class:`CSRTable` — a minimal, device/dtype-agnostic CSR container with
  constructors, filtering, row reindexing, and row-wise Cartesian joins.

Conventions
-----------
* All index tensors are ``torch.long``.
* Shapes are given in brackets, e.g. ``[R]`` for rows, ``[nnz]`` for entries.
* ``device``/``dtype`` follow the tensors you pass in; no implicit moves/casts.
* We avoid heavyweight validation in hot paths; use lightweight invariants below.

Invariants
----------
Given a CSRTable with ``starts: [R+1]``, ``cols: [nnz]``, and ``data``:

* ``starts`` is non-decreasing with ``starts[0] == 0`` and ``starts[-1] == nnz``.
* For each row ``r``, ``row_len[r] = starts[r+1] - starts[r] >= 0``.
* Every ``v`` in ``data`` has length ``nnz`` and aligns with ``cols`` order.
"""

import torch
from typing import Dict, Tuple, Optional


def starts_from_counts(counts: torch.Tensor) -> torch.Tensor:
    """Build a CSR row-pointer (`starts`) from per-row counts.

    Parameters
    ----------
    counts : LongTensor, shape ``[R]``
        Number of entries in each row.

    Returns
    -------
    starts : LongTensor, shape ``[R+1]``
        CSR row pointer with ``starts[0] == 0`` and
        ``torch.diff(starts) == counts``. The final element equals ``nnz``.

    Notes
    -----
    * ``counts`` must be non-negative. No explicit check is performed.
    * ``device`` and ``dtype`` are inherited from ``counts`` (cast to ``long`` if needed).

    Examples
    --------
    >>> counts = torch.tensor([2, 0, 3], dtype=torch.long)
    >>> starts = starts_from_counts(counts)
    >>> starts
    tensor([0, 2, 2, 5])

    Complexity
    ----------
    O(R) time, O(1) extra memory.
    """
    starts = torch.empty(counts.numel() + 1, dtype=torch.long, device=counts.device)
    starts[0] = 0
    starts[1:] = counts.cumsum(0)
    return starts


def row_and_offset(starts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decode the flattened CSR value domain into row ids and in-row offsets.

    Parameters
    ----------
    starts : LongTensor, shape ``[R+1]``
        CSR row pointer. Must satisfy the standard CSR invariants.

    Returns
    -------
    rows_of_entry : LongTensor, shape ``[nnz]``
        For each entry position ``t ∈ [0, nnz)``, the owning row id.
    offset_in_row : LongTensor, shape ``[nnz]``
        For each entry position ``t``, the offset within its row
        (i.e., ``0..row_len[row)-1``).

    Notes
    -----
    If ``nnz == 0``, both outputs are empty tensors on the same device.

    Examples
    --------
    >>> starts = torch.tensor([0, 2, 2, 5], dtype=torch.long)
    >>> rows, offs = row_and_offset(starts)
    >>> rows, offs
    (tensor([0, 0, 2, 2, 2]), tensor([0, 1, 0, 1, 2]))

    Complexity
    ----------
    O(nnz) time, O(1) extra memory.
    """
    device = starts.device
    nnz = int(starts[-1].item())
    t = torch.arange(nnz, dtype=torch.long, device=device)
    counts = torch.diff(starts)
    n_rows = starts.numel() - 1
    rows = torch.repeat_interleave(torch.arange(n_rows, device=device, dtype=torch.long), counts)
    offs = t - starts[rows]
    return rows, offs


class CSRTable:
    """A minimal CSR container allowing for multiple payload arrays..

    Attributes
    ----------
    starts : LongTensor, shape ``[R+1]``
        Row pointer. ``starts[r+1] - starts[r]`` is the length of row ``r``.
    cols : LongTensor, shape ``[nnz]``
        Per-entry "column" payload. Its interpretation is user-defined:
        it can be a global column id, or a local index ``0..row_len-1``.
    data : dict[str, Tensor]
        Additional per-entry fields, each of length ``nnz`` and aligned to ``cols``.

    Design
    ------
    The class favors simple, tensorized transformations:
    * No heavyweight validation on construction.
    * ``filter_*`` and ``reindex`` preserve per-row stability as documented.
    * ``outer`` performs row-wise Cartesian products with user-supplied ops.

    See Also
    --------
    starts_from_counts, row_and_offset
    """

    def __init__(
        self,
        starts: torch.Tensor,  # LongTensor [R+1]
        cols: torch.Tensor,  # LongTensor [nnz]
        data: Optional[Dict[str, torch.Tensor]] = None,
        reorder: bool = True,  # (only meaningful in from_coo)
    ) -> None:
        """Construct a CSRTable.

        Parameters
        ----------
        starts : LongTensor, shape ``[R+1]``
            CSR row pointer; must be non-decreasing with ``starts[0] == 0``.
        cols : LongTensor, shape ``[nnz]``
            Per-entry column payload; arbitrary semantics.
        data : dict[str, Tensor], optional
            Per-entry fields; each tensor must have shape ``[nnz]``.
        reorder : bool, default ``True``
            Ignored for direct construction. Kept for API parity with
            :meth:`from_coo` (where it controls sorting).

        Notes
        -----
        * This initializer does not validate or coerce devices/dtypes.
        * Use a separate ``validate(strict=False)`` (not provided here) if you
          need explicit invariant checks in debug contexts.
        """
        self.starts = starts
        self.cols = cols
        self.data: Dict[str, torch.Tensor] = {} if data is None else data

        # infer a single device and a single "coord" dtype (first floating dtype wins;
        # if none are floating, fall back to the first tensor's dtype)
        device: torch.device | None = None
        dtype: torch.dtype | None = None  # for floating point only.

        if not self.data:
            example_tensor = cols
        else:
            for example_tensor in self.data.values():
                if example_tensor.is_floating_point():
                    break
        self.dtype = example_tensor.dtype
        self.device = example_tensor.device

    @property
    def nnz(self):
        """Total number of entries (non-zeros) across all rows.

        Returns
        -------
        int
            Equal to ``cols.numel()`` and ``starts[-1]``.
        """
        return self.cols.numel()

    @property
    def nrows(self):
        """Number of rows in the table (``R``).

        Returns
        -------
        int
            Equal to ``starts.numel() - 1``.
        """
        return self.starts.numel() - 1

    # constructors

    @classmethod
    def make_empty(cls, n_rows: int, *, dtype: torch.dtype, device: torch.device) -> "CSRTable":
        """Create an empty CSRTable with ``n_rows`` rows and zero entries.

        Parameters
        ----------
        n_rows : int
            Number of rows.
        dtype : torch.dtype
            Target dtype for the empty ``cols`` tensor.
        device : torch.device
            Target device for all tensors.

        Returns
        -------
        CSRTable
            A well-formed empty table with ``starts == [0]* (n_rows+1)``.

        Examples
        --------
        >>> CSRTable.make_empty(3, dtype=torch.long, device=torch.device("cpu")).starts
        tensor([0, 0, 0, 0])
        """
        starts = torch.zeros(n_rows + 1, dtype=torch.long, device=device)
        cols = torch.empty(0, dtype=torch.long, device=device)
        return cls(starts=starts, cols=cols, data={}, reorder=False)

    @classmethod
    def from_coo(
        cls,
        rows: torch.Tensor,  # LongTensor [nnz]
        cols: torch.Tensor,  # LongTensor [nnz]
        data: Optional[Dict[str, torch.Tensor]] = None,
        reorder: bool = True,
        nrows=None,
    ) -> "CSRTable":
        """Build a CSRTable from COO-style row/col indices.

        Parameters
        ----------
        rows : LongTensor, shape ``[nnz]``
            Row id for each entry.
        cols : LongTensor, shape ``[nnz]``
            Column payload for each entry (can be global column ids).
        data : dict[str, Tensor], optional
            Additional per-entry fields; each must be length ``nnz``.
        reorder : bool, default ``True``
            If ``True``, rows are grouped and entries are made stable within-row
            (lexicographic by ``(row, original_position)``). If ``False``,
            current order is assumed already grouped by row.
        nrows : int, optional
            If provided, fixes the number of rows; otherwise inferred as
            ``rows.max()+1`` (or ``0`` if empty).

        Returns
        -------
        CSRTable
            A table whose ``starts`` encodes per-row cardinalities and whose
            ``cols`` contains the provided per-entry payloads.

        Notes
        -----
        * All inputs must live on the same device.
        * No deduplication is performed; repeated (row, col) pairs are allowed.

        Complexity
        ----------
        O(nnz log nnz) if ``reorder=True`` (stable sort by row),
        otherwise O(nnz).
        """
        rows = rows.to(torch.long)
        cols = cols.to(torch.long)

        nnz = rows.numel()
        # --- New: validate data sizes up front ---
        d0 = {}
        if data:
            for k, v in data.items():
                if v.shape[0] != nnz:
                    raise ValueError(
                        f"from_coo: data['{k}'] length {v.shape[0]} != nnz {nnz}. Ensure payloads are already aligned to entries."
                    )
                d0[k] = v

        if reorder:
            # Group by rows (stable); we don't require an intra-row sort by cols.
            order = torch.argsort(rows, stable=True)
            rows_sorted = rows[order]
            cols_sorted = cols[order]
            data_sorted = {k: v[order] for k, v in (data or {}).items()}
        else:
            rows_sorted = rows
            cols_sorted = cols
            data_sorted = dict(data or {})

        if nrows is None:
            minlength = int(rows_sorted.max().item()) + 1 if rows_sorted.numel() else 0
        else:
            minlength = nrows
        counts = torch.bincount(rows_sorted, minlength=minlength)
        starts = starts_from_counts(counts)

        return cls(starts=starts, cols=cols_sorted, data=data_sorted, reorder=False)

    @classmethod
    def from_counts(
        cls,
        counts: torch.Tensor,  # LongTensor [R]
        row_data: Optional[Dict[str, torch.Tensor]] = None,  # each [R,(...)] → expands by row id
        col_data: Optional[Dict[str, torch.Tensor]] = None,  # each [.,C] → expands by col id
    ) -> "CSRTable":
        """Create a CSR with locally indexed columns from per-row counts.

        Parameters
        ----------
        counts : LongTensor, shape ``[R]``
            Number of entries per row.
        row_data : dict[str, Tensor], optional
            Per-row fields to expand to length ``nnz`` using row indexing.
        col_data : dict[str, Tensor], optional
            Per-column (local) fields to expand using the local column index
            ``0..counts[r]-1`` for each row.

        Returns
        -------
        CSRTable
            With ``cols`` equal to the local in-row index and ``data`` containing
            expanded fields from ``row_data``/``col_data``.

        Examples
        --------
        >>> counts = torch.tensor([2, 1], dtype=torch.long)
        >>> row_feat = torch.tensor([[10.0],[20.0]])
        >>> col_feat = torch.arange(2)  # local 0..C-1
        >>> t = CSRTable.from_counts(counts, row_data={"rf": row_feat}, col_data={"cf": col_feat})
        >>> t.cols
        tensor([0, 1, 0])
        """
        starts = starts_from_counts(counts.to(torch.long))  # [R+1]
        rows_of_entry, local_col = row_and_offset(starts)  # [nnz], [nnz]

        data: Dict[str, torch.Tensor] = {}
        if row_data:
            for k, v in row_data.items():
                data[k] = v[rows_of_entry]
        if col_data:
            for k, v in col_data.items():
                data[k] = v[local_col]

        return cls(starts=starts, cols=local_col, data=data, reorder=False)

    @classmethod
    def from_mask(
        cls,
        mask: torch.Tensor,  # BoolTensor [R, C]
        data: Optional[Dict[str, torch.Tensor]] = None,
    ) -> "CSRTable":
        """Build a CSR from a boolean mask.

        Parameters
        ----------
        mask : BoolTensor, shape ``[R, C]``
            For each ``True`` at ``(r, c)``, one entry is emitted in row ``r`` with
            column payload ``c``.
        data : dict[str, Tensor], optional
            Per-entry fields aligned to ``True`` positions. If provided, each tensor
            must have shape ``[R, C, ...]`` and will be gathered at ``True`` indices.

        Returns
        -------
        CSRTable
            Grouped by row with stable within-row ordering (scan order).

        Notes
        -----
        * This constructor is convenient when you have a dense predicate.
        * If ``mask`` is all-false, an empty table with ``R`` rows is returned.

        Complexity
        ----------
        O(R*C) for the boolean scan; storage scales with ``nnz``.
        """
        idx = mask.nonzero(as_tuple=False)  # shape nnz, 2

        rows, cols = idx.unbind(-1)  # if there are more indices what do we do?
        data = {k: v[rows, cols] for k, v in data.items()}
        # Reorder is false because we have just acquired this in sorted order.
        return cls.from_coo(rows, cols, data=data, reorder=False)

    def __getitem__(self, key: str) -> torch.Tensor:
        """Return a derived field or a stored per-entry tensor.

        Parameters
        ----------
        key : {"rows", "cols"} | str
            * ``"rows"`` — derived LongTensor ``[nnz]`` with row id per entry.
            * ``"cols"`` — the stored per-entry column payload.
            * other string — returned from ``self.data[key]``.

        Returns
        -------
        Tensor
            The requested tensor on the same device as the table.

        Raises
        ------
        KeyError
            If ``key`` is not present in ``data`` and is not a special key.
        """
        if key == "rows":
            rows_of_entry, _ = row_and_offset(self.starts)
            return rows_of_entry
        if key == "cols":
            return self.cols
        return self.data[key]

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        """Set or replace a per-entry field in ``data``.

        Parameters
        ----------
        key : str
            Name of the field to store. Special keys ``"rows"`` and ``"cols"``
            are disallowed.
        value : Tensor, shape ``[nnz, ...]``
            Must align with the current CSR order.

        Raises
        ------
        KeyError
            If attempting to set ``"rows"`` or ``"cols"``.
        """
        if key in ("rows", "cols"):
            raise KeyError(f"Invalid key for data: {key}")
        self.data[key] = value

    def filter_mask(self, keep_mask: torch.Tensor) -> "CSRTable":
        """Filter entries by a boolean mask while recomputing per-row pointers.

        Parameters
        ----------
        keep_mask : BoolTensor, shape ``[nnz]``
            Entries with ``True`` are kept.

        Returns
        -------
        CSRTable
            A new table with the same number of rows and with row-wise stable
            ordering among kept entries.

        Notes
        -----
        * Stability is per-row: relative order among kept entries from the same row
          matches their original order.
        * If all entries are dropped, returns an empty table with the same row count.
        """
        keep_mask = keep_mask.to(torch.bool)

        # Filter values
        cols_new = self.cols[keep_mask]
        data_new = {k: v[keep_mask] for k, v in self.data.items()}

        # Recompute per-row counts from kept entries
        rows_of_entry = self["rows"]
        rows_kept = rows_of_entry[keep_mask]
        R = self.starts.numel() - 1
        counts_new = torch.bincount(rows_kept, minlength=R)
        starts_new = starts_from_counts(counts_new)

        # Group kept entries by row (stable)
        order = torch.argsort(rows_kept, stable=True)
        cols_new = cols_new[order]
        data_new = {k: v[order] for k, v in data_new.items()}

        return CSRTable(starts=starts_new, cols=cols_new, data=data_new, reorder=False)

    def filter_indices(self, keep_indices: torch.Tensor) -> "CSRTable":
        """Filter entries by absolute positions in the current value domain.

        Parameters
        ----------
        keep_indices : LongTensor, shape ``[K]``
            Absolute positions ``0..nnz-1`` to keep.

        Returns
        -------
        CSRTable
            A new table with recomputed ``starts`` and per-row stable ordering
            (stability is defined by original positions within each row).

        Notes
        -----
        If ``K == 0``, returns an empty table with the same row count.
        """
        keep_indices = keep_indices.to(torch.long)
        if keep_indices.numel() == 0:
            R = self.starts.numel() - 1
            return CSRTable(
                starts=torch.zeros(R + 1, dtype=torch.long),
                cols=torch.empty(0, dtype=torch.long),
                data={k: v[:0] for k, v in self.data.items()},
                reorder=False,
            )

        cols_new = self.cols[keep_indices]
        data_new = {k: v[keep_indices] for k, v in self.data.items()}

        rows_of_entry = self["rows"]
        rows_kept = rows_of_entry[keep_indices]
        R = self.starts.numel() - 1
        counts_new = torch.bincount(rows_kept, minlength=R)
        starts_new = starts_from_counts(counts_new)

        N = keep_indices.numel()
        orig_pos = torch.arange(N, device=rows_kept.device, dtype=torch.long)
        # composite key: (row, original_position) → guarantees stable-by-row
        key = rows_kept * (N + 1) + orig_pos
        order = torch.argsort(key)
        cols_new = cols_new[order]
        data_new = {k: v[order] for k, v in data_new.items()}

        return CSRTable(starts=starts_new, cols=cols_new, data=data_new, reorder=False)

    def reindex(
        self,
        row_ids: torch.Tensor,  # LongTensor [E]
        carry: Optional[Tuple[str, ...]] = None,  # None → carry all fields
        include_src_pos: bool = False,
        include_src_cols: bool = False,
    ) -> "CSRTable":
        """Materialize a new CSR by selecting rows in the given order.

        Parameters
        ----------
        row_ids : LongTensor, shape ``[E]``
            Row ids to take from this table. Duplicates are allowed; unspecified
            rows are dropped. Output row count equals ``E``.
        carry : tuple[str, ...] or None, optional
            Names of ``data`` fields to carry over. ``None`` carries all fields.
        include_src_pos : bool, default ``False``
            If ``True``, adds ``data["src_pos"]`` with absolute source positions.
        include_src_cols : bool, default ``False``
            If ``True``, adds ``data["src_col"]`` with source ``cols`` values.

        Returns
        -------
        CSRTable
            New table with rows arranged to match ``row_ids`` and with ``cols``
            set to local in-row indices ``0..len(row)-1``.

        Stability
        ---------
        Within each emitted row, the original per-row order is preserved.

        Examples
        --------
        >>> t2 = t.reindex(torch.tensor([3, 1, 3]), include_src_pos=True)
        """
        row_ids = row_ids.to(torch.long)
        src_starts = self.starts
        row_beg = src_starts[row_ids]
        row_end = src_starts[row_ids + 1]
        row_len = row_end - row_beg

        out_starts = starts_from_counts(row_len)
        nnz_out = int(out_starts[-1].item())

        # Absolute positions in source value domain for each output entry
        local_off = torch.arange(nnz_out, dtype=torch.long, device=out_starts.device) - torch.repeat_interleave(out_starts[:-1], row_len)
        src_pos = torch.repeat_interleave(row_beg, row_len) + local_off

        # Decide fields to carry
        if carry is None:
            carry = tuple(self.data.keys())

        data = {k: self.data[k][src_pos] for k in carry}
        if include_src_pos:
            data["src_pos"] = src_pos
        if include_src_cols:
            data["src_col"] = self.cols[src_pos]

        out_cols = local_off  # local 0..len(row)-1
        return CSRTable(starts=out_starts, cols=out_cols, data=data, reorder=False)

    def outer(
        self,
        other: "CSRTable",
        operations: Dict[Tuple[Optional[str], Optional[str], str], callable],
    ) -> "CSRTable":
        """Row-wise Cartesian join between two CSR tables.

        For each row ``r``, forms the Cartesian product ``self[r] × other[r]`` and
        builds output fields by applying user-provided operations.

        Parameters
        ----------
        other : CSRTable
            Right-hand side table; must have the same number of rows.
        operations : dict[tuple[key_left, key_right, out_key], callable]
            Operation spec:
            * ``(None, None, out)`` → ``fn()`` (nullary)
            * ``(key, None, out)`` → ``fn(self[key])`` (unary, left)
            * ``(None, key, out)`` → ``fn(other[key])`` (unary, right)
            * ``(ka, kb, out)`` → ``fn(self[ka], other[kb])`` (binary)

            Each callable is applied to the expanded positions; it must be
            broadcasting-compatible or indexable per expanded pair.

        Returns
        -------
        CSRTable
            Output with the same row count and ``nnz[r] = len(self[r]) * len(other[r])``.

        Notes
        -----
        * The output ``cols`` encodes the right-hand local index (or may be a
          local index depending on implementation); rely on output ``data``
          for payloads you need downstream.
        * If a row is empty on either side, the corresponding output row is empty.
        """
        # Note, if this arrange construct is somehow costly (doesn't seem like it would be)
        # then this can be rewritten to avoid it.
        rows = torch.arange(self.nrows, device=self.starts.device, dtype=torch.long)
        return CSRTable.expand_pairings(self, other, rows, rows, operations)

    @staticmethod
    def expand_pairings(
        left_csr: "CSRTable",
        right_csr: "CSRTable",
        left_rows: torch.Tensor,
        right_rows: torch.Tensor,
        operations: Dict[Tuple[Optional[str], Optional[str], str], callable],
    ) -> "CSRTable":
        """Expand specific row pairings of two CSR tables (generalized join).

        Parameters
        ----------
        left_csr : CSRTable
            Left-hand table.
        right_csr : CSRTable
            Right-hand table.
        left_rows : LongTensor, shape ``[P]``
            Row ids in ``left_csr`` participating in pairings.
        right_rows : LongTensor, shape ``[P]``
            Row ids in ``right_csr``; ``left_rows[i]`` is paired with ``right_rows[i]``.
        operations : dict
            Same contract as in :meth:`outer`.

        Returns
        -------
        CSRTable
            Output CSR with ``R = P`` rows (one row per pairing) and with entries
            equal to the Cartesian product of the paired source rows.

        Notes
        -----
        * Use this when you have a precomputed mapping between rows of two tables
        (e.g., voxel adjacency), not necessarily one-to-one by index.
        * Stability is per paired row based on the original per-row orders.
        """

        ##TODO: change how `operations` API works to allow for more complex, multi-argument operations.

        A_st = left_csr.starts
        B_st = right_csr.starts

        # Sanity (rows must have same length E)
        if left_rows.numel() != right_rows.numel():
            raise ValueError(f"first_rows and second_rows must have identical length. {right_rows.numel()=} ; {left_rows.numel()=}")

        # shapes: number of rows:
        begA, endA = A_st[left_rows], A_st[left_rows + 1]
        begB, endB = B_st[right_rows], B_st[right_rows + 1]
        lenA = endA - begA
        lenB = endB - begB
        # Build row pointer for combination
        counts = lenA * lenB
        out_starts = starts_from_counts(counts)

        # shapes: nnz
        rows_of_out, cols_of_out = row_and_offset(out_starts)
        lenB_per_out_row = lenB[rows_of_out]
        # mod out the local counter by number of B items from that row. (B is faster varying)
        cols_A = torch.div(cols_of_out, lenB_per_out_row, rounding_mode="floor")
        cols_B = torch.remainder(cols_of_out, lenB_per_out_row)

        # Flat position indices
        posA = begA[rows_of_out] + cols_A  # [P]
        posB = begB[rows_of_out] + cols_B  # [P]

        # Build outputs per requested operation
        out_data: Dict[str, torch.Tensor] = {}
        for (ka, kb, kout), fn in operations.items():
            if ka is None and kb is None:
                out_data[kout] = fn()
            elif kb is None:
                v1 = left_csr[ka][posA]
                out_data[kout] = fn(v1)
            elif ka is None:
                v2 = right_csr[kb][posB]
                out_data[kout] = fn(v2)
            else:
                v1 = left_csr[ka][posA]
                v2 = right_csr[kb][posB]
                out_data[kout] = fn(v1, v2)

        return CSRTable(starts=out_starts, cols=cols_of_out, data=out_data, reorder=False)


def find_indices(query_vals: torch.Tensor, keys: torch.Tensor) -> Tuple[torch.Tensor]:
    """
    Map query values to their indices in keys, or -1 if not found.

    This function allows you to look at the CSR rows for a table and see if they exist in another csr table.

    Args:
        keys: [V] long tensor, strictly increasing 
        query_vals: [...] long tensor
    
    Returns:
        Tuple: (locations, indices)
        locations: indices of query values which are in the key.
        indices: Index of the key corresponding to those query values.
    
    """
    
    #Example:
    #    keys = [1, 5, 10, 15, 20]
    #    query_vals = [5, 7, 10, 25, 1]
    
    n = keys.shape[0]  # Number of keys
    # n = 5

    # Step 1: Binary search to find insertion points
    idx = torch.bucketize(query_vals, keys, right=False)  # long
    # idx = [1, 2, 2, 5, 0]
        
    # Step 2: Clamp indices to prevent out-of-bounds access
    idx_clamped = idx.clamp_max(n - 1)  # long
    # idx_clamped = [1, 2, 2, 4, 0]
    hit = (query_vals == keys[idx_clamped])
    # hit = [True, False, True, False, True]
    
    locations = torch.nonzero(hit,as_tuple=True)[0]
    # [0, 2, 4]
    indices = idx_clamped[locations]
    # [1, 2, 0]
    return locations, indices