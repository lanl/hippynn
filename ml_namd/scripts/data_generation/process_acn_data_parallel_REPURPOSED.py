#!/usr/bin/env python3
"""
ACN (Acetonitrile) Multi-State Molecular Dynamics Data Processing Pipeline - PARALLEL VERSION

This script provides parallelized processing of molecular dynamics simulation data
using multiprocessing. It is designed for production use with datasets ranging from
100K to millions of data points.

Key Features:
    - Parallel processing using multiprocessing Pool
    - Configurable number of worker processes (default: 8)
    - Smart chunk sizing to ensure no data loss
    - Data integrity validation after processing
    - Memory-efficient processing with numpy memory mapping
    - Progress tracking with tqdm
    - Comprehensive logging

Processing Steps (parallelized where beneficial):
    1. Load input files (I/O bound - not parallelized)
    2. Process energies (parallelized by sample chunks)
    3. Process dipoles (parallelized by sample chunks)
    4. Process forces (parallelized by sample chunks)
    5. Process NACRs (parallelized by sample chunks)
    6. Validate data integrity
    7. Chunk and save outputs

Usage:
    python process_acn_data_parallel.py --input-dir ./data --output-dir ./output
    python process_acn_data_parallel.py --input-dir . --output-dir . --n-workers 16
    python process_acn_data_parallel.py --input-dir . --output-dir . --n-states 10 --n-workers 8

Author: ACN/NEXMD Workflow
Version: 2.0.0 (Parallel)
Date: November 2025
"""

from __future__ import annotations

# Developer Information
# ---------------------
__developer__ = "Ajay Khanna"
__place__ = "LANL/Sergei Lab"
__date__ = "Sep-01-2026"
__project__ = "ml_namd"

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

# =============================================================================
# CONSTANTS
# =============================================================================

# REPURPOSED 2026-07-11 for excited-state (S1/S2/S3) oracle generation, n_states=11.
# Number of electronic states (1 ground + 10 excited)  [was 21]
N_STATES: int = 11

# Number of atoms in acetylacetone molecule
N_ATOMS: int = 15

# Number of NAC couplings between excited states: n_excited*(n_excited-1)//2 = 10*9/2  [was 190]
N_COUPLINGS: int = 45

# Reference energy for shifted calculations (eV) — DEFAULT; override per tautomer via --e0-opt.
#   enol -1391.422970121047 | keto -1391.451136550848
OPTIMIZED_ENERGY: float = -1391.422970121047  # S0 Optimized Energy via PySEQM (enol default)

# Reference forces for shifted calculations (15 atoms x 3 coordinates)
OPTIMIZED_FORCES: NDArray[np.float64] = np.array(
    [
        [6.09767223e-06, 5.16182851e-03, -2.84626353e-03],
        [-1.60408038e-03, -6.34445308e-03, -4.02516198e-03],
        [8.61501106e-04, 1.41074545e-03, 3.63932576e-03],
        [9.43439996e-04, -1.25869440e-03, 2.24689762e-03],
        [-5.77735975e-04, 1.30392930e-03, 2.29329916e-03],
        [1.69995215e-03, -3.84671492e-03, -2.14119582e-03],
        [-1.57438019e-03, 3.45538870e-03, -2.05595949e-03],
        [8.23418914e-04, -5.44891936e-04, 2.63386403e-03],
        [-5.40436639e-04, 6.30419613e-04, 2.73540986e-03],
        [-4.38205970e-04, -1.04667110e-03, -1.58481075e-04],
        [7.23256249e-04, 3.02980139e-04, -1.37934934e-03],
        [1.75435752e-03, 1.78403184e-04, 3.27739926e-04],
        [-7.28629288e-04, -3.00717455e-04, -1.41729874e-03],
        [4.28048873e-04, 1.05461220e-03, -1.18574971e-04],
        [-1.77660404e-03, -1.56164206e-04, 2.65748578e-04],
    ],
    dtype=np.float64,
)  # S0 Optimized Forces via PySEQM

# Chunking configuration for output
CHOP_START: int = 10000
CHOP_END: int = 100001  # exclusive (max chunk = 100K)
CHOP_STEP: int = 10000

# Parallel processing configuration
DEFAULT_N_WORKERS: int = 8
# Minimum samples per worker chunk (to avoid overhead for small chunks)
MIN_CHUNK_SIZE: int = 1000
# Maximum samples per worker chunk (for memory efficiency)
MAX_CHUNK_SIZE: int = 25000

# Input file specifications (required files - excluding configurable coordinate files)
INPUT_FILES: Dict[str, Tuple[str | int, ...]] = {
    "energies_all_states.npy": ("N", N_STATES),
    "total_energy.npy": ("N", 1),
    "transition_dipoles.npy": ("N", N_STATES - 1, 3),
    "gs_dipoles.npy": ("N", 3),
    "forces_all_states.npy": ("N", N_STATES, N_ATOMS, 3),
    "nacrs_all_states.npy": (N_COUPLINGS, "N", N_ATOMS, 3),
}

# Default coordinate input files (can be overridden via CLI)
DEFAULT_COORDS_R_FILE: str = "molecule_R.npy"  # Atomic coordinates (N, N_ATOMS, 3)
DEFAULT_COORDS_Z_FILE: str = "molecule_Z.npy"  # Atomic numbers (N, N_ATOMS)

# Optional file for atomic reordering (if forces/NACRs are sorted by Z)
OPTIONAL_SORTED_Z_FILE: str = "sorted_Z_indices.npy"


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class ParallelConfig:
    """Configuration for the parallel data processing pipeline.

    Attributes:
        input_dir: Directory containing input .npy files
        output_dir: Base directory for output files
        log_file: Path to log file
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        chop_start: First chunk size for output
        chop_end: Maximum chunk size (exclusive)
        chop_step: Increment between chunk sizes
        split_nacr: Whether to save individual NACR state-pair files
        n_states: Number of excited states to process
        n_workers: Number of parallel worker processes
        validate: Whether to run data integrity validation
    """

    input_dir: Path
    output_dir: Path
    log_file: Path
    log_level: str = "INFO"
    chop_start: int = CHOP_START
    chop_end: int = CHOP_END
    chop_step: int = CHOP_STEP
    split_nacr: bool = False
    n_states: int = N_STATES - 1  # Default: 20 excited states (21 total)
    n_workers: int = DEFAULT_N_WORKERS
    validate: bool = True
    sorted_z_file: Optional[Path] = None  # Optional file for atomic reordering
    coords_r_file: str = DEFAULT_COORDS_R_FILE  # Input file for atomic coordinates
    coords_z_file: str = DEFAULT_COORDS_Z_FILE  # Input file for atomic numbers
    e0_opt: float = OPTIMIZED_ENERGY  # per-tautomer S0 opt energy (--e0-opt); shifts acn_sE
    tautomer: str = "enol"  # provenance tag for the manifest

    @property
    def total_states(self) -> int:
        """Total number of states (ground + excited)."""
        return self.n_states + 1

    @property
    def n_couplings(self) -> int:
        """Number of NACR couplings for the selected states."""
        return self.n_states * (self.n_states - 1) // 2

    def get_chop_sizes(self) -> NDArray[np.int64]:
        """Get array of chunk sizes for output."""
        sizes = np.arange(self.chop_start, self.chop_end, self.chop_step)
        max_size = self.chop_end - 1 if self.chop_end > 0 else 0
        if len(sizes) == 0 or sizes[-1] < max_size:
            sizes = np.append(sizes, max_size)
        return sizes

    def get_chunk_dirs(self) -> List[str]:
        """Get list of chunk directory names."""
        return [f"{size // 1000}K" for size in self.get_chop_sizes()]


@dataclass
class ChunkInfo:
    """Information about a processing chunk.

    Attributes:
        chunk_id: Unique identifier for the chunk
        start_idx: Starting sample index (inclusive)
        end_idx: Ending sample index (exclusive)
        n_samples: Number of samples in this chunk
    """

    chunk_id: int
    start_idx: int
    end_idx: int

    @property
    def n_samples(self) -> int:
        return self.end_idx - self.start_idx


@dataclass
class ValidationResult:
    """Result of data integrity validation.

    Attributes:
        is_valid: Whether all validations passed
        total_samples: Expected total sample count
        actual_samples: Actual sample count in output
        checksum_match: Whether input/output checksums match
        shape_valid: Whether all output shapes are correct
        errors: List of validation error messages
    """

    is_valid: bool
    total_samples: int
    actual_samples: int
    checksum_match: bool
    shape_valid: bool
    errors: List[str] = field(default_factory=list)


# =============================================================================
# LOGGING SETUP
# =============================================================================


def setup_logging(log_file: Path, log_level: str) -> logging.Logger:
    """Configure dual logging to console and file."""
    logger = logging.getLogger("acn_parallel_processor")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(funcName)s - %(message)s"
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_format = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    return logger


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def print_header(logger: logging.Logger) -> None:
    """Print the pipeline header."""
    header = """
================================================================================
ACN Data Processing Pipeline - PARALLEL VERSION
================================================================================
"""
    logger.info(header)


def print_table(
    title: str, headers: List[str], rows: List[List[str]], logger: logging.Logger
) -> None:
    """Print a formatted table."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    separator = "├" + "┼".join("─" * (w + 2) for w in col_widths) + "┤"
    top_border = "┌" + "┬".join("─" * (w + 2) for w in col_widths) + "┐"
    bottom_border = "└" + "┴".join("─" * (w + 2) for w in col_widths) + "┘"

    def format_row(cells: List[str]) -> str:
        return "│" + "│".join(f" {c:<{w}} " for c, w in zip(cells, col_widths)) + "│"

    logger.info(f"\n[{title}]")
    logger.info(top_border)
    logger.info(format_row(headers))
    logger.info(separator)
    for row in rows:
        logger.info(format_row(row))
    logger.info(bottom_border)


def format_shape(shape: Tuple[int, ...]) -> str:
    """Format a shape tuple as a string."""
    return f"({', '.join(str(s) for s in shape)})"


def format_time(seconds: float) -> str:
    """Format seconds as a human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def compute_array_checksum(arr: NDArray) -> str:
    """Compute MD5 checksum of an array for validation."""
    return hashlib.md5(arr.tobytes()).hexdigest()


def get_nacr_state_pairs(n_excited_states: int = N_STATES - 1) -> List[Tuple[int, int]]:
    """Generate list of NACR state pairs (i, j) where 1 <= i < j <= n_excited_states."""
    pairs = []
    total_states = n_excited_states + 1
    for i in range(1, total_states):
        for j in range(i + 1, total_states):
            pairs.append((i, j))
    return pairs


def get_nacr_indices_for_subset(n_excited_states: int) -> List[int]:
    """Get indices of NACR couplings to extract for a subset of states."""
    full_pairs = get_nacr_state_pairs(N_STATES - 1)
    full_pair_to_idx = {pair: idx for idx, pair in enumerate(full_pairs)}
    subset_pairs = get_nacr_state_pairs(n_excited_states)
    indices = [full_pair_to_idx[pair] for pair in subset_pairs]
    return indices


# =============================================================================
# CHUNK MANAGEMENT
# =============================================================================


def calculate_optimal_chunks(
    n_samples: int,
    n_workers: int,
    min_chunk: int = MIN_CHUNK_SIZE,
    max_chunk: int = MAX_CHUNK_SIZE,
) -> List[ChunkInfo]:
    """Calculate optimal chunk distribution ensuring no data loss.

    This function ensures:
    1. All samples are covered (no gaps)
    2. Chunks are evenly distributed across workers
    3. Any remainder samples are distributed to early chunks
    4. Chunk sizes respect min/max constraints

    Args:
        n_samples: Total number of samples
        n_workers: Number of worker processes
        min_chunk: Minimum samples per chunk
        max_chunk: Maximum samples per chunk

    Returns:
        List of ChunkInfo objects covering all samples
    """
    # Calculate base chunk size
    base_chunk_size = max(min_chunk, n_samples // n_workers)
    base_chunk_size = min(base_chunk_size, max_chunk)

    # Calculate number of chunks needed
    n_chunks = max(1, (n_samples + base_chunk_size - 1) // base_chunk_size)

    # Recalculate chunk size to distribute evenly
    chunk_size = n_samples // n_chunks
    remainder = n_samples % n_chunks

    chunks = []
    current_idx = 0

    for i in range(n_chunks):
        # Add one extra sample to first 'remainder' chunks
        extra = 1 if i < remainder else 0
        this_chunk_size = chunk_size + extra
        end_idx = current_idx + this_chunk_size

        chunks.append(
            ChunkInfo(
                chunk_id=i,
                start_idx=current_idx,
                end_idx=end_idx,
            )
        )
        current_idx = end_idx

    # Verify no data loss
    total_covered = sum(c.n_samples for c in chunks)
    assert (
        total_covered == n_samples
    ), f"Data loss detected: {total_covered} != {n_samples}"

    # Verify no gaps
    for i in range(1, len(chunks)):
        assert (
            chunks[i].start_idx == chunks[i - 1].end_idx
        ), f"Gap detected at chunk {i}"

    return chunks


# =============================================================================
# PARALLEL PROCESSING WORKER FUNCTIONS
# =============================================================================


def process_energies_chunk(
    chunk: ChunkInfo,
    energies: NDArray[np.float64],
    total_energy: NDArray[np.float64],
    n_total_states: int,
    e0_opt: float,
) -> Tuple[int, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Process energy data for a single chunk.

    Returns:
        Tuple of (chunk_id, acn_S, acn_sE, acn_dE) for this chunk
    """
    start, end = chunk.start_idx, chunk.end_idx

    # Slice to selected states
    e_chunk = energies[start:end, :n_total_states]
    te_chunk = total_energy[start:end]

    # Raw energies
    acn_S = e_chunk.copy()

    # Shifted energies (per-tautomer E0_opt threaded in; spawn workers don't see globals)
    acn_sE = e_chunk - e0_opt

    # Delta energies
    acn_dE = e_chunk - te_chunk

    return (chunk.chunk_id, acn_S, acn_sE, acn_dE)


def process_dipoles_chunk(
    chunk: ChunkInfo,
    gs_dipoles: NDArray[np.float64],
    transition_dipoles: NDArray[np.float64],
    n_excited_states: int,
) -> Tuple[int, NDArray[np.float64]]:
    """Process dipole data for a single chunk.

    Returns:
        Tuple of (chunk_id, acn_D) for this chunk
    """
    start, end = chunk.start_idx, chunk.end_idx

    # Ground state dipoles: (chunk_size, 3) -> (chunk_size, 1, 3)
    gs_expanded = gs_dipoles[start:end, np.newaxis, :]

    # Transition dipoles: slice to selected excited states
    trans_chunk = transition_dipoles[start:end, :n_excited_states, :]

    # Concatenate
    acn_D = np.concatenate([gs_expanded, trans_chunk], axis=1)

    return (chunk.chunk_id, acn_D)


def process_forces_chunk(
    chunk: ChunkInfo,
    forces: NDArray[np.float64],
    sort_indices: Optional[NDArray[np.int64]],
    n_total_states: int,
) -> Tuple[int, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Process force data for a single chunk.

    Args:
        chunk: Chunk information with start/end indices
        forces: Force array (n_samples, n_states, n_atoms, 3)
        sort_indices: Optional sorted Z indices for atomic reordering.
                      If None, forces are assumed to be in correct order.
        n_total_states: Number of states to process

    Returns:
        Tuple of (chunk_id, acn_F, acn_sF, acn_dF) for this chunk
    """
    start, end = chunk.start_idx, chunk.end_idx

    # Slice to selected states
    f_chunk = forces[start:end, :n_total_states, :, :]

    if sort_indices is not None:
        # Compute unsort indices to restore original atomic order
        si_chunk = sort_indices[start:end]
        unsort_indices = np.argsort(si_chunk, axis=1)
        unsort_indices_4d = unsort_indices[:, None, :, None]

        # Restore atomic order
        acn_F = np.take_along_axis(f_chunk, unsort_indices_4d, axis=2)

        # Shifted forces (apply unsorting after shift)
        shifted_raw = f_chunk - OPTIMIZED_FORCES[None, None, :, :]
        acn_sF = np.take_along_axis(shifted_raw, unsort_indices_4d, axis=2)
    else:
        # No reordering needed - data is already in correct order
        acn_F = f_chunk.copy()
        acn_sF = f_chunk - OPTIMIZED_FORCES[None, None, :, :]

    # Delta forces (difference from ground state)
    ground_state_forces = acn_F[:, 0:1, :, :]
    acn_dF = acn_F - ground_state_forces

    return (chunk.chunk_id, acn_F, acn_sF, acn_dF)


def process_nacrs_chunk(
    chunk: ChunkInfo,
    nacrs: NDArray[np.float64],
    sort_indices: Optional[NDArray[np.int64]],
    n_excited_states: int,
    nacr_indices: Optional[List[int]] = None,
) -> Tuple[int, NDArray[np.float64]]:
    """Process NACR data for a single chunk.

    Note: NACRs have shape (n_couplings, N, 15, 3), so we slice on axis 1.

    Args:
        chunk: Chunk information with start/end indices
        nacrs: NACR array (n_couplings, n_samples, n_atoms, 3)
        sort_indices: Optional sorted Z indices for atomic reordering.
                      If None, NACRs are assumed to be in correct order.
        n_excited_states: Number of excited states
        nacr_indices: Optional list of coupling indices to extract

    Returns:
        Tuple of (chunk_id, acn_NACR) for this chunk
    """
    start, end = chunk.start_idx, chunk.end_idx

    # Extract relevant couplings if subset
    if nacr_indices is not None:
        nacrs_subset = nacrs[nacr_indices, start:end, :, :]
    else:
        nacrs_subset = nacrs[:, start:end, :, :]

    # Transpose: (n_couplings, chunk_size, 15, 3) -> (chunk_size, n_couplings, 15, 3)
    nacrs_transposed = np.moveaxis(nacrs_subset, 0, 1)

    if sort_indices is not None:
        # Compute unsort indices to restore original atomic order
        si_chunk = sort_indices[start:end]
        unsort_indices = np.argsort(si_chunk, axis=1)
        unsort_indices_4d = unsort_indices[:, None, :, None]

        # Restore atomic order
        acn_NACR = np.take_along_axis(nacrs_transposed, unsort_indices_4d, axis=2)
    else:
        # No reordering needed - data is already in correct order
        acn_NACR = nacrs_transposed.copy()

    return (chunk.chunk_id, acn_NACR)


def process_denacr_chunk(
    chunk: ChunkInfo,
    acn_NACR: NDArray[np.float64],
    energies: NDArray[np.float64],
    nacr_pairs: List[Tuple[int, int]],
) -> Tuple[int, NDArray[np.float64]]:
    """Process scaled NACR (dENACR) data for a single chunk.

    Computes dENACR = NACR * (E_j - E_i) based on Hellmann-Feynman theorem.
    For NACR_ij = <psi_i|nabla_R H|psi_j> / (E_i - E_j), we recover the
    derivative coupling by multiplying by the energy gap (E_j - E_i).

    Args:
        chunk: Chunk information with start/end indices
        acn_NACR: Processed NACR array (n_samples, n_couplings, n_atoms, 3)
        energies: Energy array (n_samples, n_states) - column 0 is ground state,
                  columns 1+ are excited states S1, S2, ...
        nacr_pairs: List of (i, j) state pairs where i < j (1-indexed for excited states)

    Returns:
        Tuple of (chunk_id, acn_dENACR) for this chunk
    """
    start, end = chunk.start_idx, chunk.end_idx
    n_chunk = end - start
    n_couplings = len(nacr_pairs)
    n_atoms = acn_NACR.shape[2]

    # Get NACR chunk
    nacr_chunk = acn_NACR[start:end]  # (chunk_size, n_couplings, n_atoms, 3)

    # Get energies chunk
    e_chunk = energies[start:end]  # (chunk_size, n_states)

    # Pre-allocate dENACR
    acn_dENACR = np.empty((n_chunk, n_couplings, n_atoms, 3), dtype=np.float64)

    # For each coupling pair, compute dENACR = NACR * (E_j - E_i)
    for pair_idx, (state_i, state_j) in enumerate(nacr_pairs):
        # State indices in energy array: state_i and state_j are 1-indexed (S1=1, S2=2, ...)
        # In energies array: column 0 = ground, column 1 = S1, column 2 = S2, etc.
        # So state_i maps to column state_i, state_j maps to column state_j
        E_i = e_chunk[:, state_i]  # (chunk_size,)
        E_j = e_chunk[:, state_j]  # (chunk_size,)

        # Energy gap: E_j - E_i (positive since j > i and typically E_j > E_i)
        delta_E = E_j - E_i  # (chunk_size,)

        # Scale NACR by energy gap
        # nacr_chunk[:, pair_idx] has shape (chunk_size, n_atoms, 3)
        # delta_E needs to be broadcast: (chunk_size,) -> (chunk_size, 1, 1)
        acn_dENACR[:, pair_idx, :, :] = (
            nacr_chunk[:, pair_idx, :, :] * delta_E[:, None, None]
        )

    return (chunk.chunk_id, acn_dENACR)


# =============================================================================
# PARALLEL PROCESSING ORCHESTRATION
# =============================================================================


def run_parallel_processing(
    energies: NDArray[np.float64],
    total_energy: NDArray[np.float64],
    gs_dipoles: NDArray[np.float64],
    transition_dipoles: NDArray[np.float64],
    forces: NDArray[np.float64],
    sort_indices: Optional[NDArray[np.int64]],
    nacrs: NDArray[np.float64],
    config: ParallelConfig,
    logger: logging.Logger,
) -> Dict[str, NDArray[np.float64]]:
    """Run all processing steps in parallel.

    Args:
        energies: Energy array (n_samples, n_states)
        total_energy: Total energy array (n_samples, 1)
        gs_dipoles: Ground state dipoles (n_samples, 3)
        transition_dipoles: Transition dipoles (n_samples, n_states-1, 3)
        forces: Forces array (n_samples, n_states, n_atoms, 3)
        sort_indices: Optional sorted Z indices for atomic reordering.
                      If None, forces/NACRs are assumed to be in correct order.
        nacrs: NACR array (n_couplings, n_samples, n_atoms, 3)
        config: Processing configuration
        logger: Logger instance

    Returns:
        Dictionary of output arrays
    """
    n_samples = energies.shape[0]
    n_workers = config.n_workers

    logger.info("\n[PARALLEL PROCESSING]")
    logger.info(f"  Total samples: {n_samples:,}")
    logger.info(f"  Worker processes: {n_workers}")
    logger.info(
        f"  Processing {config.total_states} states (ground + {config.n_states} excited)"
    )
    logger.info(f"  NACR couplings: {config.n_couplings}")
    logger.info(
        f"  Atomic reordering: {'enabled' if sort_indices is not None else 'disabled'}"
    )

    # Calculate optimal chunks
    chunks = calculate_optimal_chunks(n_samples, n_workers)
    logger.info(f"  Processing chunks: {len(chunks)}")
    logger.info(
        f"  Chunk sizes: {chunks[0].n_samples:,} to {chunks[-1].n_samples:,} samples"
    )

    # Prepare NACR indices for subset if needed
    nacr_indices = None
    if config.n_states < N_STATES - 1:
        nacr_indices = get_nacr_indices_for_subset(config.n_states)

    # Pre-allocate output arrays
    acn_S = np.empty((n_samples, config.total_states), dtype=np.float64)
    acn_sE = np.empty((n_samples, config.total_states), dtype=np.float64)
    acn_dE = np.empty((n_samples, config.total_states), dtype=np.float64)
    acn_D = np.empty((n_samples, config.total_states, 3), dtype=np.float64)
    acn_F = np.empty((n_samples, config.total_states, N_ATOMS, 3), dtype=np.float64)
    acn_sF = np.empty((n_samples, config.total_states, N_ATOMS, 3), dtype=np.float64)
    acn_dF = np.empty((n_samples, config.total_states, N_ATOMS, 3), dtype=np.float64)
    acn_NACR = np.empty((n_samples, config.n_couplings, N_ATOMS, 3), dtype=np.float64)
    acn_dENACR = np.empty((n_samples, config.n_couplings, N_ATOMS, 3), dtype=np.float64)

    # Get NACR state pairs for dENACR computation
    nacr_pairs = get_nacr_state_pairs(config.n_states)

    # Track processed samples for validation
    processed_samples = set()

    with mp.Pool(processes=n_workers) as pool:
        # Process Energies
        logger.info("\n  Step 1/5: Processing energies...")
        start_time = time.time()

        energy_func = partial(
            process_energies_chunk,
            energies=energies,
            total_energy=total_energy,
            n_total_states=config.total_states,
            e0_opt=config.e0_opt,
        )

        for chunk_id, s_chunk, se_chunk, de_chunk in tqdm(
            pool.imap_unordered(energy_func, chunks),
            total=len(chunks),
            desc="    Energies",
            unit="chunk",
        ):
            chunk = chunks[chunk_id]
            start, end = chunk.start_idx, chunk.end_idx
            acn_S[start:end] = s_chunk
            acn_sE[start:end] = se_chunk
            acn_dE[start:end] = de_chunk
            processed_samples.update(range(start, end))

        elapsed = time.time() - start_time
        logger.info(f"    ✓ Energies processed ({elapsed:.2f}s)")

        # Verify all samples processed
        if len(processed_samples) != n_samples:
            raise RuntimeError(
                f"Energy processing incomplete: {len(processed_samples)}/{n_samples} samples"
            )
        processed_samples.clear()

        # Process Dipoles
        logger.info("  Step 2/5: Processing dipoles...")
        start_time = time.time()

        dipole_func = partial(
            process_dipoles_chunk,
            gs_dipoles=gs_dipoles,
            transition_dipoles=transition_dipoles,
            n_excited_states=config.n_states,
        )

        for chunk_id, d_chunk in tqdm(
            pool.imap_unordered(dipole_func, chunks),
            total=len(chunks),
            desc="    Dipoles",
            unit="chunk",
        ):
            chunk = chunks[chunk_id]
            start, end = chunk.start_idx, chunk.end_idx
            acn_D[start:end] = d_chunk
            processed_samples.update(range(start, end))

        elapsed = time.time() - start_time
        logger.info(f"    ✓ Dipoles processed ({elapsed:.2f}s)")

        if len(processed_samples) != n_samples:
            raise RuntimeError(
                f"Dipole processing incomplete: {len(processed_samples)}/{n_samples} samples"
            )
        processed_samples.clear()

        # Process Forces
        logger.info("  Step 3/5: Processing forces...")
        start_time = time.time()

        force_func = partial(
            process_forces_chunk,
            forces=forces,
            sort_indices=sort_indices,
            n_total_states=config.total_states,
        )

        for chunk_id, f_chunk, sf_chunk, df_chunk in tqdm(
            pool.imap_unordered(force_func, chunks),
            total=len(chunks),
            desc="    Forces",
            unit="chunk",
        ):
            chunk = chunks[chunk_id]
            start, end = chunk.start_idx, chunk.end_idx
            acn_F[start:end] = f_chunk
            acn_sF[start:end] = sf_chunk
            acn_dF[start:end] = df_chunk
            processed_samples.update(range(start, end))

        elapsed = time.time() - start_time
        logger.info(f"    ✓ Forces processed ({elapsed:.2f}s)")

        if len(processed_samples) != n_samples:
            raise RuntimeError(
                f"Force processing incomplete: {len(processed_samples)}/{n_samples} samples"
            )
        processed_samples.clear()

        # Process NACRs
        logger.info("  Step 4/5: Processing NACRs...")
        start_time = time.time()

        nacr_func = partial(
            process_nacrs_chunk,
            nacrs=nacrs,
            sort_indices=sort_indices,
            n_excited_states=config.n_states,
            nacr_indices=nacr_indices,
        )

        for chunk_id, nacr_chunk in tqdm(
            pool.imap_unordered(nacr_func, chunks),
            total=len(chunks),
            desc="    NACRs",
            unit="chunk",
        ):
            chunk = chunks[chunk_id]
            start, end = chunk.start_idx, chunk.end_idx
            acn_NACR[start:end] = nacr_chunk
            processed_samples.update(range(start, end))

        elapsed = time.time() - start_time
        logger.info(f"    ✓ NACRs processed ({elapsed:.2f}s)")

        if len(processed_samples) != n_samples:
            raise RuntimeError(
                f"NACR processing incomplete: {len(processed_samples)}/{n_samples} samples"
            )
        processed_samples.clear()

        # Process dENACR (scaled NACRs)
        logger.info("  Step 5/5: Processing dENACR (scaled NACRs)...")
        start_time = time.time()

        denacr_func = partial(
            process_denacr_chunk,
            acn_NACR=acn_NACR,
            energies=energies,
            nacr_pairs=nacr_pairs,
        )

        for chunk_id, denacr_chunk in tqdm(
            pool.imap_unordered(denacr_func, chunks),
            total=len(chunks),
            desc="    dENACR",
            unit="chunk",
        ):
            chunk = chunks[chunk_id]
            start, end = chunk.start_idx, chunk.end_idx
            acn_dENACR[start:end] = denacr_chunk
            processed_samples.update(range(start, end))

        elapsed = time.time() - start_time
        logger.info(f"    ✓ dENACR processed ({elapsed:.2f}s)")

        if len(processed_samples) != n_samples:
            raise RuntimeError(
                f"dENACR processing incomplete: {len(processed_samples)}/{n_samples} samples"
            )

    return {
        "acn_S.npy": acn_S,
        "acn_sE.npy": acn_sE,
        "acn_dE.npy": acn_dE,
        "acn_D.npy": acn_D,
        "acn_F.npy": acn_F,
        "acn_sF.npy": acn_sF,
        "acn_dF.npy": acn_dF,
        "acn_NACR.npy": acn_NACR,
        "acn_dENACR.npy": acn_dENACR,
    }


# =============================================================================
# DATA VALIDATION
# =============================================================================


def validate_data_integrity(
    input_energies: NDArray[np.float64],
    input_total_energy: NDArray[np.float64],
    output_data: Dict[str, NDArray[np.float64]],
    config: ParallelConfig,
    logger: logging.Logger,
) -> ValidationResult:
    """Validate data integrity after processing.

    Performs:
    1. Sample count validation
    2. Shape validation
    3. Checksum validation for reversible operations
    4. NaN/Inf detection

    Args:
        input_energies: Original energy data
        input_total_energy: Original total energy data
        output_data: Processed output arrays
        config: Processing configuration
        logger: Logger instance

    Returns:
        ValidationResult with validation status and details
    """
    logger.info("\n[DATA VALIDATION]")

    errors = []
    n_samples = input_energies.shape[0]
    n_total = config.total_states
    n_couplings = config.n_couplings

    # Expected shapes
    expected_shapes = {
        "acn_S.npy": (n_samples, n_total),
        "acn_sE.npy": (n_samples, n_total),
        "acn_dE.npy": (n_samples, n_total),
        "acn_D.npy": (n_samples, n_total, 3),
        "acn_F.npy": (n_samples, n_total, N_ATOMS, 3),
        "acn_sF.npy": (n_samples, n_total, N_ATOMS, 3),
        "acn_dF.npy": (n_samples, n_total, N_ATOMS, 3),
        "acn_NACR.npy": (n_samples, n_couplings, N_ATOMS, 3),
        "acn_dENACR.npy": (n_samples, n_couplings, N_ATOMS, 3),
        "acn_R.npy": (n_samples, N_ATOMS, 3),
        "acn_Z.npy": (n_samples, N_ATOMS),
    }

    # Shape validation
    shape_valid = True
    for name, expected in expected_shapes.items():
        actual = output_data[name].shape
        if actual != expected:
            shape_valid = False
            errors.append(
                f"Shape mismatch for {name}: expected {expected}, got {actual}"
            )
            logger.error(f"  ✗ {name}: shape {actual} != {expected}")
        else:
            logger.debug(f"  ✓ {name}: shape {actual} OK")

    # Sample count validation
    actual_samples = output_data["acn_S.npy"].shape[0]
    sample_count_valid = actual_samples == n_samples

    if not sample_count_valid:
        errors.append(
            f"Sample count mismatch: expected {n_samples}, got {actual_samples}"
        )

    # Checksum validation for acn_S (should match input energies sliced)
    input_slice = input_energies[:, :n_total]
    input_checksum = compute_array_checksum(input_slice)
    output_checksum = compute_array_checksum(output_data["acn_S.npy"])
    checksum_match = input_checksum == output_checksum

    if not checksum_match:
        errors.append("Checksum mismatch for acn_S (energies)")
        logger.warning("  ⚠ Energy checksum mismatch - investigating...")
        # Check if values match even if checksum differs (floating point issues)
        if np.allclose(input_slice, output_data["acn_S.npy"]):
            logger.info("    Values match within tolerance (floating point)")
            checksum_match = True
            errors.pop()  # Remove the error
    else:
        logger.info("  ✓ Energy checksum validation passed")

    # Validate shifted energies computation
    expected_sE = input_energies[:, :n_total] - config.e0_opt
    if not np.allclose(expected_sE, output_data["acn_sE.npy"]):
        errors.append("Shifted energy computation incorrect")
        logger.error("  ✗ Shifted energy validation failed")
    else:
        logger.info("  ✓ Shifted energy validation passed")

    # Validate delta energies computation
    expected_dE = input_energies[:, :n_total] - input_total_energy
    if not np.allclose(expected_dE, output_data["acn_dE.npy"]):
        errors.append("Delta energy computation incorrect")
        logger.error("  ✗ Delta energy validation failed")
    else:
        logger.info("  ✓ Delta energy validation passed")

    # NaN/Inf detection
    for name, arr in output_data.items():
        if np.any(np.isnan(arr)):
            errors.append(f"NaN values detected in {name}")
            logger.error(f"  ✗ {name}: contains NaN values")
        if np.any(np.isinf(arr)):
            errors.append(f"Inf values detected in {name}")
            logger.error(f"  ✗ {name}: contains Inf values")

    is_valid = len(errors) == 0

    if is_valid:
        logger.info("\n  ✓ All validation checks passed!")
    else:
        logger.error(f"\n  ✗ Validation failed with {len(errors)} error(s)")

    return ValidationResult(
        is_valid=is_valid,
        total_samples=n_samples,
        actual_samples=actual_samples,
        checksum_match=checksum_match,
        shape_valid=shape_valid,
        errors=errors,
    )


# =============================================================================
# SAVING FUNCTIONS
# =============================================================================


def save_chunked_outputs(
    output_data: Dict[str, NDArray[np.float64]],
    config: ParallelConfig,
    logger: logging.Logger,
) -> None:
    """Save processed data in chunks to size-based directories."""
    chop_sizes = config.get_chop_sizes()
    chunk_dirs = config.get_chunk_dirs()

    # Print output dimensions
    table_rows = [
        [name.replace(".npy", ""), format_shape(arr.shape)]
        for name, arr in output_data.items()
    ]
    print_table("OUTPUT DIMENSIONS", ["Array", "Full Shape"], table_rows, logger)

    # Get NACR state pairs
    nacr_pairs = get_nacr_state_pairs(config.n_states)
    max_state = config.n_states
    logger.info(
        f"\n  NACR state pairs: {len(nacr_pairs)} pairs (S1-S2 through S{max_state-1}-S{max_state})"
    )

    # Reshape dENACR arrays: (N, n_couplings, 15, 3) -> (N, n_couplings, 45)
    logger.info("\n[RESHAPING dENACR ARRAYS]")
    original_shape = output_data["acn_dENACR.npy"].shape
    n_samples_full = original_shape[0]
    n_couplings = original_shape[1]
    logger.info(
        f"  Reshaping acn_dENACR.npy: {format_shape(original_shape)} -> ({n_samples_full}, {n_couplings}, {N_ATOMS * 3})"
    )
    output_data["acn_dENACR.npy"] = output_data["acn_dENACR.npy"].reshape(
        n_samples_full, n_couplings, N_ATOMS * 3
    )
    logger.info(
        f"  ✓ acn_dENACR.npy reshaped to {format_shape(output_data['acn_dENACR.npy'].shape)}"
    )

    logger.info("\n[CHUNKING PROGRESS]")

    # Create all directories
    for chunk_dir in chunk_dirs:
        dir_path = config.output_dir / chunk_dir
        dir_path.mkdir(parents=True, exist_ok=True)

    # Save chunks
    for chop_size, chunk_dir in tqdm(
        zip(chop_sizes, chunk_dirs),
        total=len(chop_sizes),
        desc="Saving chunks",
        unit="chunk",
    ):
        output_path = config.output_dir / chunk_dir

        # Save unified arrays
        for filename, arr in output_data.items():
            chunked_arr = arr[:chop_size]
            np.save(output_path / filename, chunked_arr)
            logger.debug(
                f"Saved {output_path / filename}: {format_shape(chunked_arr.shape)}"
            )

        # Save state-wise arrays
        for state_idx in range(config.total_states):
            np.save(
                output_path / f"acn_S{state_idx}.npy",
                output_data["acn_S.npy"][:chop_size, state_idx],
            )
            np.save(
                output_path / f"acn_sE{state_idx}.npy",
                output_data["acn_sE.npy"][:chop_size, state_idx],
            )
            np.save(
                output_path / f"acn_dE{state_idx}.npy",
                output_data["acn_dE.npy"][:chop_size, state_idx],
            )
            np.save(
                output_path / f"acn_D{state_idx}.npy",
                output_data["acn_D.npy"][:chop_size, state_idx, :],
            )
            np.save(
                output_path / f"acn_F{state_idx}.npy",
                output_data["acn_F.npy"][:chop_size, state_idx, :, :],
            )
            np.save(
                output_path / f"acn_sF{state_idx}.npy",
                output_data["acn_sF.npy"][:chop_size, state_idx, :, :],
            )
            np.save(
                output_path / f"acn_dF{state_idx}.npy",
                output_data["acn_dF.npy"][:chop_size, state_idx, :, :],
            )

        # Save NACR and dENACR state-pair arrays if requested
        if config.split_nacr:
            for pair_idx, (state_i, state_j) in enumerate(nacr_pairs):
                # Save NACR pair
                nacr_pair_data = output_data["acn_NACR.npy"][:chop_size, pair_idx, :, :]
                np.save(output_path / f"acn_NACR{state_i}{state_j}.npy", nacr_pair_data)
                # Save dENACR pair - already reshaped to (N, n_couplings, 45), extract and keep as (N, 45)
                denacr_pair_data = output_data["acn_dENACR.npy"][
                    :chop_size, pair_idx, :
                ]
                np.save(
                    output_path / f"acn_dENACR{state_i}{state_j}.npy", denacr_pair_data
                )

    # Summary
    unified_files = len(output_data)
    state_files = config.total_states * 7
    nacr_pair_files = (
        len(nacr_pairs) * 2 if config.split_nacr else 0
    )  # NACR + dENACR pairs
    total_files = unified_files + state_files + nacr_pair_files

    logger.info(
        f"\n  ✓ Saved {len(chop_sizes)} chunks ({chunk_dirs[0]} to {chunk_dirs[-1]})"
    )
    logger.info(f"  ✓ {total_files} files per chunk:")
    logger.info(f"      - {unified_files} unified arrays")
    logger.info(
        f"      - {state_files} state-wise arrays ({config.total_states} states × 7 properties)"
    )
    if config.split_nacr:
        logger.info(
            f"      - {nacr_pair_files} NACR/dENACR state-pair arrays ({len(nacr_pairs)} pairs × 2)"
        )

    # Print final shapes of all created arrays
    logger.info("\n[FINAL ARRAY SHAPES]")
    logger.info("  Unified arrays:")
    for name, arr in output_data.items():
        logger.info(f"    {name}: {format_shape(arr.shape)}")

    if config.split_nacr:
        logger.info("\n  State-pair arrays (per chunk):")
        for pair_idx, (state_i, state_j) in enumerate(nacr_pairs):
            # NACR pair shape
            nacr_shape = (chop_sizes[-1], N_ATOMS, 3)
            logger.info(
                f"    acn_NACR{state_i}{state_j}.npy: {format_shape(nacr_shape)}"
            )
            # dENACR pair shape (flattened)
            denacr_shape = (chop_sizes[-1], N_ATOMS * 3)
            logger.info(
                f"    acn_dENACR{state_i}{state_j}.npy: {format_shape(denacr_shape)}"
            )


# =============================================================================
# DATA LOADING
# =============================================================================


def load_input_data(config: ParallelConfig, logger: logging.Logger) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    Optional[NDArray[np.int64]],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
    int,
]:
    """Load all input files.

    Returns:
        Tuple of (energies, total_energy, gs_dipoles, transition_dipoles,
                  forces, sort_indices, nacrs, acn_R, acn_Z, n_samples)
        Note: sort_indices is None if --sorted-z was not provided.
    """
    logger.info("\n[LOADING INPUT FILES]")

    # Build complete list of required files (including configurable coordinate files)
    required_files = list(INPUT_FILES.keys()) + [
        config.coords_r_file,
        config.coords_z_file,
    ]

    # Check for missing required files
    missing = []
    for filename in required_files:
        if not (config.input_dir / filename).exists():
            missing.append(filename)

    if missing:
        raise FileNotFoundError(f"Missing required input files: {missing}")

    # Load required files
    loaded = {}
    table_rows = []

    for filename in tqdm(required_files, desc="Loading files", unit="file"):
        filepath = config.input_dir / filename
        data = np.load(filepath)
        loaded[filename] = data
        table_rows.append([filename, format_shape(data.shape)])
        logger.debug(f"Loaded {filename}: {data.shape}")

    # Load optional sorted_Z_indices file if specified
    sort_indices: Optional[NDArray[np.int64]] = None
    if config.sorted_z_file is not None:
        if not config.sorted_z_file.exists():
            raise FileNotFoundError(
                f"Sorted Z indices file not found: {config.sorted_z_file}"
            )
        sort_indices = np.load(config.sorted_z_file).astype(np.int64)
        table_rows.append([config.sorted_z_file.name, format_shape(sort_indices.shape)])
        logger.debug(f"Loaded {config.sorted_z_file.name}: {sort_indices.shape}")

    print_table("INPUT DIMENSIONS", ["File", "Shape"], table_rows, logger)

    n_samples = loaded["energies_all_states.npy"].shape[0]
    n_files_loaded = len(INPUT_FILES) + (1 if sort_indices is not None else 0)
    logger.info(f"\n  ✓ All {n_files_loaded} input files loaded")
    logger.info(f"  ✓ Detected {n_samples:,} samples/timesteps")
    if sort_indices is not None:
        logger.info(
            f"  ✓ Atomic reordering enabled (using {config.sorted_z_file.name})"
        )
    else:
        logger.info(
            "  ✓ Atomic reordering disabled (forces/NACRs assumed in correct order)"
        )

    return (
        loaded["energies_all_states.npy"].astype(np.float64),
        loaded["total_energy.npy"].astype(np.float64),
        loaded["gs_dipoles.npy"].astype(np.float64),
        loaded["transition_dipoles.npy"].astype(np.float64),
        loaded["forces_all_states.npy"].astype(np.float64),
        sort_indices,
        loaded["nacrs_all_states.npy"].astype(np.float64),
        loaded[config.coords_r_file].astype(np.float64),
        loaded[config.coords_z_file].astype(np.int64),
        n_samples,
    )


# =============================================================================
# ARGUMENT PARSING
# =============================================================================


def parse_arguments() -> ParallelConfig:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ACN Multi-State MD Data Processing Pipeline - PARALLEL VERSION",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_acn_data_parallel.py --input-dir ./data --output-dir ./output
  python process_acn_data_parallel.py --input-dir . --output-dir . --n-workers 16
  python process_acn_data_parallel.py --n-states 10 --n-workers 8 --no-validate
  python process_acn_data_parallel.py --sorted-z sorted_Z_indices.npy  # Enable atomic reordering

Performance Tips:
  - Use n_workers equal to number of CPU cores for optimal performance
  - For very large datasets (>1M samples), consider increasing MAX_CHUNK_SIZE
  - Use --no-validate to skip validation for faster processing
        """,
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("."),
        help="Directory containing input .npy files (default: current directory)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Base directory for output files (default: current directory)",
    )

    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("processing_parallel.log"),
        help="Path to log file (default: processing_parallel.log)",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    parser.add_argument(
        "--chop-start",
        type=int,
        default=CHOP_START,
        help=f"First chunk size (default: {CHOP_START})",
    )

    parser.add_argument(
        "--chop-end",
        type=int,
        default=CHOP_END,
        help=f"Maximum chunk size, exclusive (default: {CHOP_END})",
    )

    parser.add_argument(
        "--chop-step",
        type=int,
        default=CHOP_STEP,
        help=f"Increment between chunk sizes (default: {CHOP_STEP})",
    )

    parser.add_argument(
        "--split-nacr",
        action="store_true",
        default=False,
        help="Save individual NACR state-pair files",
    )

    parser.add_argument(
        "--n-states",
        type=int,
        default=N_STATES - 1,
        help=f"Number of excited states to process (default: {N_STATES - 1})",
    )

    parser.add_argument(
        "--n-workers",
        type=int,
        default=DEFAULT_N_WORKERS,
        help=f"Number of parallel worker processes (default: {DEFAULT_N_WORKERS})",
    )

    parser.add_argument(
        "--no-validate",
        action="store_true",
        default=False,
        help="Skip data integrity validation (faster but less safe)",
    )

    parser.add_argument(
        "--sorted-z",
        type=Path,
        default=None,
        metavar="FILE",
        help="Optional file with sorted Z indices for atomic reordering. "
        "If provided, forces and NACRs will be reordered from Z-sorted "
        "order back to original molecular order. If not provided, data "
        "is assumed to already be in the correct atomic order.",
    )

    parser.add_argument(
        "--coords-r",
        type=str,
        default=DEFAULT_COORDS_R_FILE,
        metavar="FILE",
        help=f"Input file for atomic coordinates (N, {N_ATOMS}, 3). "
        f"(default: {DEFAULT_COORDS_R_FILE})",
    )

    parser.add_argument(
        "--coords-z",
        type=str,
        default=DEFAULT_COORDS_Z_FILE,
        metavar="FILE",
        help=f"Input file for atomic numbers (N, {N_ATOMS}). "
        f"(default: {DEFAULT_COORDS_Z_FILE})",
    )

    parser.add_argument(
        "--e0-opt",
        type=float,
        default=OPTIMIZED_ENERGY,
        metavar="EV",
        help="Per-tautomer S0-optimized energy (eV) subtracted to form acn_sE. "
        "enol -1391.422970121047 | keto -1391.451136550848. Logged to manifest.",
    )
    parser.add_argument(
        "--tautomer",
        type=str,
        choices=["enol", "keto"],
        default="enol",
        help="Provenance tag written to generation_manifest.json.",
    )

    args = parser.parse_args()

    return ParallelConfig(
        input_dir=args.input_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        log_file=args.log_file.resolve(),
        log_level=args.log_level,
        chop_start=args.chop_start,
        chop_end=args.chop_end,
        chop_step=args.chop_step,
        split_nacr=args.split_nacr,
        n_states=args.n_states,
        n_workers=args.n_workers,
        validate=not args.no_validate,
        sorted_z_file=args.sorted_z.resolve() if args.sorted_z else None,
        coords_r_file=args.coords_r,
        coords_z_file=args.coords_z,
        e0_opt=args.e0_opt,
        tautomer=args.tautomer,
    )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def main() -> int:
    """Main entry point."""
    total_start_time = time.time()

    # Parse arguments
    config = parse_arguments()

    # Setup logging
    logger = setup_logging(config.log_file, config.log_level)

    try:
        print_header(logger)

        # Log configuration
        logger.info(f"  Input directory:  {config.input_dir}")
        logger.info(f"  Output directory: {config.output_dir}")
        logger.info(f"  Workers:          {config.n_workers}")
        logger.info(
            f"  States:           {config.total_states} (ground + {config.n_states} excited)"
        )
        logger.info(
            f"  Validation:       {'enabled' if config.validate else 'disabled'}"
        )
        logger.info(
            f"  Atomic reorder:   {'enabled (' + config.sorted_z_file.name + ')' if config.sorted_z_file else 'disabled'}"
        )

        # Load input data
        (
            energies,
            total_energy,
            gs_dipoles,
            transition_dipoles,
            forces,
            sort_indices,
            nacrs,
            acn_R,
            acn_Z,
            n_samples,
        ) = load_input_data(config, logger)

        # Run parallel processing
        output_data = run_parallel_processing(
            energies=energies,
            total_energy=total_energy,
            gs_dipoles=gs_dipoles,
            transition_dipoles=transition_dipoles,
            forces=forces,
            sort_indices=sort_indices,
            nacrs=nacrs,
            config=config,
            logger=logger,
        )

        # Add acn_R and acn_Z to output data (passthrough, no processing needed)
        output_data["acn_R.npy"] = acn_R
        output_data["acn_Z.npy"] = acn_Z

        # Validate if enabled
        if config.validate:
            validation = validate_data_integrity(
                input_energies=energies,
                input_total_energy=total_energy,
                output_data=output_data,
                config=config,
                logger=logger,
            )

            if not validation.is_valid:
                logger.error("Data validation failed!")
                for error in validation.errors:
                    logger.error(f"  - {error}")
                return 1

        # Save outputs
        save_chunked_outputs(output_data, config, logger)

        # Reproducibility manifest (per user request: log which E0_opt/tautomer was used)
        manifest = {
            "tautomer": config.tautomer,
            "e0_opt_eV": config.e0_opt,
            "n_states_total": config.total_states,
            "n_excited": config.n_states,
            "n_couplings": config.n_couplings,
            "n_samples": int(n_samples),
            "input_dir": str(config.input_dir),
            "output_dir": str(config.output_dir),
            "coords_r_file": config.coords_r_file,
            "coords_z_file": config.coords_z_file,
            "sorted_z_reorder": config.sorted_z_file is not None,
            "script": os.path.basename(__file__),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(config.output_dir / "generation_manifest.json", "w") as mf:
            json.dump(manifest, mf, indent=2)
        logger.info(f"\n  ✓ Wrote generation_manifest.json (tautomer={config.tautomer}, e0_opt={config.e0_opt})")

        # Summary
        total_elapsed = time.time() - total_start_time
        chunk_dirs = config.get_chunk_dirs()

        logger.info("\n[SUMMARY]")
        logger.info(f"  Total processing time: {format_time(total_elapsed)}")
        logger.info(f"  Samples processed:     {n_samples:,}")
        logger.info(
            f"  Chunks created:        {len(chunk_dirs)} ({chunk_dirs[0]} to {chunk_dirs[-1]})"
        )
        logger.info(f"  Output directory:      {config.output_dir}")
        logger.info(f"  Log file:              {config.log_file}")

        # Performance stats
        samples_per_second = n_samples / total_elapsed
        logger.info(f"\n  Performance: {samples_per_second:,.0f} samples/second")

        logger.info(
            """
================================================================================
Processing complete!
================================================================================"""
        )

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    # Required for multiprocessing on some platforms
    mp.set_start_method("spawn", force=True)
    sys.exit(main())
