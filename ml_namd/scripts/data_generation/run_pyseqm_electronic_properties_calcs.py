#!/usr/bin/env python3
"""
SEQM Electronic Structure Calculation Script
=============================================

A production-ready, multi-GPU parallel script for computing electronic structure
properties using the SEQM (Semi-Empirical Quantum Mechanics) library.

This script computes the following properties for molecular configurations:
    - Ground state energies and forces
    - Excited state energies (CIS) and forces
    - Ground state dipole moments
    - Excited state dipole moments (relaxed)
    - Transition dipole moments (ground → excited)
    - Non-adiabatic coupling vectors (NACRs) between excited states

Features:
    - Multi-GPU parallelization using shared memory
    - Batched processing for memory efficiency
    - Real-time progress monitoring
    - Comprehensive error handling and validation
    - Automatic dimension inference based on molecular system
    - PEP 8 compliant with full type hints

Input Requirements:
    - coords.npy: Shape (N, A, 3) - Atomic coordinates in Angstroms
        N = number of molecular configurations
        A = number of atoms per molecule
    - species.npy: Shape (N, A) - Atomic numbers (e.g., 6=C, 7=N, 1=H)

Output Arrays:
    - energies: (N, n_states) - All electronic state energies in eV
    - forces: (N, n_states, A, 3) - Gradients for all states in eV/Angstrom
    - gs_dipoles: (N, 3) - Ground state dipole moments in Debye
    - ex_dipoles: (N, n_excited, 3) - Excited state dipoles in Debye
    - trans_dipoles: (N, n_excited, 3) - Transition dipoles (S0→Sn) in Debye
    - nacrs: (n_couplings, N, A, 3) - Non-adiabatic coupling vectors

Usage Examples:
    # Basic usage with 4 GPUs
    python run_grads_energies_tdipoles_pyseqm.py \\
        --coords coords.npy \\
        --species species.npy \\
        --gpus 0 1 2 3

    # Custom batch size and 21 states (ground + 20 excited)
    python run_grads_energies_tdipoles_pyseqm.py \\
        --coords coords.npy \\
        --species species.npy \\
        --gpus 0 1 \\
        --batch 250 \\
        --n_states 21

Author: Ajay Khanna
Version: 2.0.0
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
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from multiprocessing import shared_memory
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.multiprocessing as mp

# --- SEQM imports ---
from seqm.ElectronicStructure import Electronic_Structure
from seqm.Molecule import Molecule
from seqm.seqm_functions.constants import Constants

if TYPE_CHECKING:
    from multiprocessing.context import SpawnProcess

    from numpy.typing import NDArray


__version__ = "2.0.0"
__author__ = "Ajay Khanna"


# =============================================================================
# Constants
# =============================================================================

SUPPORTED_METHODS = ("AM1", "PM3", "MNDO")
DEFAULT_BATCH_SIZE = 500
DEFAULT_N_STATES = 11
DEFAULT_SCF_EPS = 1e-10
DEFAULT_CIS_TOL = 1e-8


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class SEQMConfig:
    """Configuration for SEQM electronic structure calculations.

    This dataclass encapsulates all parameters needed for SEQM calculations,
    providing validation and conversion to the SEQM parameter dictionary format.

    Attributes:
        method: Semi-empirical method to use. Supported: 'AM1', 'PM3', 'MNDO'.
        scf_eps: SCF convergence threshold (energy difference between iterations).
        scf_converger: SCF convergence algorithm parameters. Default: [2].
        n_states: Total number of electronic states (1 ground + n-1 excited).
        cis_tol: CIS (Configuration Interaction Singles) convergence tolerance.
        do_all_forces: Whether to compute analytical gradients for all states.
        do_all_nac: Whether to compute all non-adiabatic coupling vectors.

    Example:
        >>> config = SEQMConfig(method="AM1", n_states=11)
        >>> params = config.to_seqm_params()
    """

    method: str = "AM1"
    scf_eps: float = DEFAULT_SCF_EPS
    scf_converger: list[int] = field(default_factory=lambda: [2])
    n_states: int = DEFAULT_N_STATES
    cis_tol: float = DEFAULT_CIS_TOL
    do_all_forces: bool = True
    do_all_nac: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.method not in SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported method '{self.method}'. "
                f"Supported: {SUPPORTED_METHODS}"
            )
        if self.n_states < 1:
            raise ValueError(f"n_states must be >= 1, got {self.n_states}")
        if self.scf_eps <= 0:
            raise ValueError(f"scf_eps must be > 0, got {self.scf_eps}")
        if self.cis_tol <= 0:
            raise ValueError(f"cis_tol must be > 0, got {self.cis_tol}")

    def to_seqm_params(self) -> dict[str, Any]:
        """Convert configuration to SEQM parameter dictionary.

        Returns:
            Dictionary compatible with SEQM Molecule and Electronic_Structure.
        """
        return {
            "method": self.method,
            "scf_eps": self.scf_eps,
            "scf_converger": self.scf_converger,
            "excited_states": {
                "n_states": self.n_states - 1,  # SEQM wants number of excited states
                "cis_tol": self.cis_tol,
            },
            "analytical_gradient": [True],
            "do_all_forces": self.do_all_forces,
            "do_all_nac": self.do_all_nac,
        }


@dataclass
class OutputDimensions:
    """Automatic dimension calculation for all output arrays.

    All dimensions are computed based on:
        - N: Number of molecular configurations
        - A: Number of atoms per molecule
        - n_states: Total number of electronic states

    The NAC coupling count follows the formula:
        n_couplings = (n_states - 1) * (n_states - 2) // 2
    This is because NACs are computed between excited states only (not ground state).

    Attributes:
        n_configs: Number of molecular configurations (N).
        n_atoms: Number of atoms per molecule (A).
        n_states: Total number of electronic states.
        n_excited: Number of excited states (n_states - 1).
        n_couplings: Number of NAC pairs between excited states.

    Example:
        >>> dims = OutputDimensions(n_configs=1000, n_atoms=15, n_states=21)
        >>> print(dims.forces_shape)
        (1000, 21, 15, 3)
        >>> print(dims.n_couplings)
        190
    """

    n_configs: int
    n_atoms: int
    n_states: int
    n_excited: int = field(init=False)
    n_couplings: int = field(init=False)

    def __post_init__(self) -> None:
        """Calculate derived dimensions."""
        self.n_excited = self.n_states - 1
        # NACs are computed between excited states only (not ground state)
        # For n excited states, we have n*(n-1)/2 unique pairs
        self.n_couplings = self.n_excited * (self.n_excited - 1) // 2

    @property
    def energies_shape(self) -> tuple[int, int]:
        """Shape for all state energies: (N, n_states)."""
        return (self.n_configs, self.n_states)

    @property
    def forces_shape(self) -> tuple[int, int, int, int]:
        """Shape for all state forces: (N, n_states, A, 3)."""
        return (self.n_configs, self.n_states, self.n_atoms, 3)

    @property
    def gs_dipole_shape(self) -> tuple[int, int]:
        """Shape for ground state dipole moments: (N, 3)."""
        return (self.n_configs, 3)

    @property
    def ex_dipole_shape(self) -> tuple[int, int, int]:
        """Shape for excited state dipole moments: (N, n_excited, 3)."""
        return (self.n_configs, self.n_excited, 3)

    @property
    def transition_dipole_shape(self) -> tuple[int, int, int]:
        """Shape for transition dipole moments (S0 → Sn): (N, n_excited, 3)."""
        return (self.n_configs, self.n_excited, 3)

    @property
    def total_energy_shape(self) -> tuple[int, int]:
        """Shape for total SCF energy: (N, 1)."""
        return (self.n_configs, 1)

    @property
    def nacr_shape(self) -> tuple[int, int, int, int]:
        """Shape for non-adiabatic coupling vectors: (n_couplings, N, A, 3)."""
        return (self.n_couplings, self.n_configs, self.n_atoms, 3)

    def get_nbytes(self, shape: tuple[int, ...], dtype: type = np.float64) -> int:
        """Calculate number of bytes for an array of given shape.

        Args:
            shape: Array shape as tuple of integers.
            dtype: NumPy dtype (default: np.float64).

        Returns:
            Total number of bytes required.
        """
        return int(np.prod(shape)) * np.dtype(dtype).itemsize

    def get_total_memory_gb(self) -> float:
        """Calculate total memory required for all output arrays in GB.

        Returns:
            Total memory in gigabytes.
        """
        total_bytes = sum(
            self.get_nbytes(shape)
            for shape in [
                self.energies_shape,
                self.forces_shape,
                self.gs_dipole_shape,
                self.ex_dipole_shape,
                self.transition_dipole_shape,
                self.total_energy_shape,
                self.nacr_shape,
            ]
        )
        return total_bytes / (1024**3)

    def print_summary(self) -> None:
        """Print a formatted summary of all output dimensions."""
        print("\n" + "=" * 65)
        print("OUTPUT DIMENSIONS SUMMARY")
        print("=" * 65)
        print(f"  Configurations (N):      {self.n_configs:>12,}")
        print(f"  Atoms per molecule (A):  {self.n_atoms:>12}")
        print(f"  Total states:            {self.n_states:>12}")
        print(f"  Excited states:          {self.n_excited:>12}")
        print(f"  NAC couplings:           {self.n_couplings:>12}")
        print("-" * 65)
        print("  Output array shapes:")
        print(f"    Energies (all states):     {str(self.energies_shape):>20}")
        print(f"    Forces (all states):       {str(self.forces_shape):>20}")
        print(f"    Ground state dipoles:      {str(self.gs_dipole_shape):>20}")
        print(f"    Excited state dipoles:     {str(self.ex_dipole_shape):>20}")
        print(f"    Transition dipoles:        {str(self.transition_dipole_shape):>20}")
        print(f"    Total SCF energy:          {str(self.total_energy_shape):>20}")
        print(f"    NACRs:                     {str(self.nacr_shape):>20}")
        print("-" * 65)
        print(f"  Total memory required:   {self.get_total_memory_gb():>12.2f} GB")
        print("=" * 65 + "\n")


# =============================================================================
# Utility Functions
# =============================================================================


def format_duration(seconds: float) -> str:
    """Format time duration into human-readable string.

    Args:
        seconds: Time duration in seconds.

    Returns:
        Formatted string (e.g., "1h 23m 45s", "45.2s", "2m 30.5s").

    Example:
        >>> format_duration(3665.5)
        '1h 1m 5s'
        >>> format_duration(45.234)
        '45.23s'
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.0f}s"


def split_contiguous(n_total: int, num_workers: int) -> list[tuple[int, int]]:
    """Split N items into contiguous ranges for parallel workers.

    Distributes items as evenly as possible, with earlier workers
    receiving one extra item if there's a remainder.

    Args:
        n_total: Total number of items to split.
        num_workers: Number of workers to distribute among.

    Returns:
        List of (start, end) tuples for each worker's range.

    Raises:
        ValueError: If num_workers <= 0 or n_total < 0.

    Example:
        >>> split_contiguous(10, 3)
        [(0, 4), (4, 7), (7, 10)]
        >>> split_contiguous(9, 3)
        [(0, 3), (3, 6), (6, 9)]
    """
    if num_workers <= 0:
        raise ValueError(f"num_workers must be positive, got {num_workers}")
    if n_total < 0:
        raise ValueError(f"n_total must be non-negative, got {n_total}")

    base_size = n_total // num_workers
    remainder = n_total % num_workers
    ranges: list[tuple[int, int]] = []
    start = 0

    for worker_idx in range(num_workers):
        # First 'remainder' workers get one extra item
        length = base_size + (1 if worker_idx < remainder else 0)
        ranges.append((start, start + length))
        start += length

    return ranges


def validate_inputs(
    coords_path: Path,
    species_path: Path,
) -> tuple[NDArray[np.float64], NDArray[np.int64], int, int]:
    """Validate and load input coordinate and species arrays.

    Performs comprehensive validation of input files including:
        - File existence
        - Array dimensionality
        - Shape consistency between coordinates and species
        - Coordinate array last dimension is 3 (x, y, z)

    Args:
        coords_path: Path to coordinates .npy file.
        species_path: Path to species .npy file.

    Returns:
        Tuple of (coords_mmap, species_mmap, n_configs, n_atoms).

    Raises:
        FileNotFoundError: If input files don't exist.
        ValueError: If array shapes are invalid or inconsistent.

    Example:
        >>> coords, species, n, a = validate_inputs(Path("coords.npy"), Path("species.npy"))
        >>> print(f"Loaded {n} configs with {a} atoms")
    """
    # Check file existence
    if not coords_path.exists():
        raise FileNotFoundError(f"Coordinates file not found: {coords_path}")
    if not species_path.exists():
        raise FileNotFoundError(f"Species file not found: {species_path}")

    # Load as memory-mapped arrays (read-only for efficiency)
    coords = np.load(coords_path, mmap_mode="r")
    species = np.load(species_path, mmap_mode="r")

    # Validate coordinates shape: must be (N, A, 3)
    if coords.ndim != 3:
        raise ValueError(
            f"Coordinates must be 3D array (N, A, 3), "
            f"got {coords.ndim}D with shape {coords.shape}"
        )
    if coords.shape[-1] != 3:
        raise ValueError(
            f"Coordinates last dimension must be 3 (x, y, z), "
            f"got {coords.shape[-1]}"
        )

    n_configs, n_atoms, _ = coords.shape

    # Validate species shape: must be (N, A) matching coordinates
    if species.ndim != 2:
        raise ValueError(
            f"Species must be 2D array (N, A), "
            f"got {species.ndim}D with shape {species.shape}"
        )
    if species.shape[0] != n_configs:
        raise ValueError(
            f"Species configurations ({species.shape[0]}) "
            f"doesn't match coordinates ({n_configs})"
        )
    if species.shape[1] != n_atoms:
        raise ValueError(
            f"Species atoms ({species.shape[1]}) "
            f"doesn't match coordinates ({n_atoms})"
        )

    return coords, species, n_configs, n_atoms


# =============================================================================
# Progress Display
# =============================================================================


def print_progress_bar(
    current: int,
    total: int,
    start_time: float,
    bar_length: int = 40,
    prefix: str = "Progress",
) -> None:
    """Print a dynamic progress bar to stdout with ETA.

    Creates a visual progress bar that updates in-place, showing:
        - Visual bar representation
        - Percentage complete
        - Items processed / total
        - Elapsed time
        - Estimated time remaining (ETA)

    Args:
        current: Current progress count.
        total: Total items to process.
        start_time: Start time from time.time().
        bar_length: Length of the progress bar in characters.
        prefix: Text to display before the bar.

    Example:
        Progress: |████████████░░░░░░░░░░░░░░░░░░| 40.0% [400/1000] Elapsed: 1m 20s ETA: 2m 0s
    """
    if total == 0:
        return

    fraction = current / total
    filled_length = int(bar_length * fraction)
    bar = "█" * filled_length + "░" * (bar_length - filled_length)
    percent = fraction * 100

    elapsed = time.time() - start_time

    if current > 0 and current < total:
        eta = elapsed * (total - current) / current
        eta_str = format_duration(eta)
    elif current >= total:
        eta_str = "0s"
    else:
        eta_str = "calculating..."

    # Use carriage return to update in place
    sys.stdout.write(
        f"\r  {prefix}: |{bar}| {percent:5.1f}% "
        f"[{current:,}/{total:,}] "
        f"Elapsed: {format_duration(elapsed)} ETA: {eta_str}    "
    )
    sys.stdout.flush()

    # Print newline when complete
    if current >= total:
        sys.stdout.write("\n")


# =============================================================================
# Shared Memory Management
# =============================================================================


@dataclass
class SharedMemoryManager:
    """Manager for shared memory buffers used in multi-process communication.

    Handles creation, cleanup, and access to shared memory blocks
    for inter-process communication of output arrays.

    Attributes:
        dims: OutputDimensions object specifying array shapes.
        buffers: Dictionary of SharedMemory objects by name.
        arrays: Dictionary of numpy array views by name.
    """

    dims: OutputDimensions
    buffers: dict[str, shared_memory.SharedMemory] = field(default_factory=dict)
    arrays: dict[str, NDArray[np.float64]] = field(default_factory=dict)

    def allocate(self) -> None:
        """Allocate all shared memory buffers.

        Creates shared memory for:
            - energies
            - forces
            - gs_dipole
            - ex_dipole
            - trans_dipole
            - total_energy
            - nacr

        Raises:
            MemoryError: If insufficient memory is available.
        """
        array_specs: dict[str, tuple[int, ...]] = {
            "energies": self.dims.energies_shape,
            "forces": self.dims.forces_shape,
            "gs_dipole": self.dims.gs_dipole_shape,
            "ex_dipole": self.dims.ex_dipole_shape,
            "trans_dipole": self.dims.transition_dipole_shape,
            "total_energy": self.dims.total_energy_shape,
            "nacr": self.dims.nacr_shape,
        }

        for name, shape in array_specs.items():
            nbytes = self.dims.get_nbytes(shape)
            self.buffers[name] = shared_memory.SharedMemory(create=True, size=nbytes)
            self.arrays[name] = np.ndarray(
                shape, dtype=np.float64, buffer=self.buffers[name].buf
            )
            # Initialize to NaN for debugging unwritten values
            self.arrays[name].fill(np.nan)

    def get_names(self) -> dict[str, str]:
        """Get dictionary mapping array names to shared memory names.

        Returns:
            Dictionary for passing to worker processes.
        """
        return {name: shm.name for name, shm in self.buffers.items()}

    def cleanup(self) -> None:
        """Clean up all shared memory buffers.

        Closes and unlinks all shared memory blocks.
        Safe to call multiple times.
        """
        for name, shm in list(self.buffers.items()):
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass  # Ignore errors during cleanup
        self.buffers.clear()
        self.arrays.clear()


# =============================================================================
# Worker Function
# =============================================================================


def worker(
    gpu_id: int,
    idx_range: tuple[int, int],
    coords_path: str,
    species_path: str,
    shm_names: dict[str, str],
    dims_tuple: tuple[int, int, int],
    batch_size: int,
    config: SEQMConfig,
    progress_shm_name: str,
) -> None:
    """GPU worker function for parallel SEQM calculations.

    Each worker processes a contiguous subset of molecular configurations
    on a dedicated GPU, writing results directly to shared memory.

    Args:
        gpu_id: CUDA device ID to use for this worker.
        idx_range: Tuple of (start_index, end_index) to process.
        coords_path: Path to coordinates .npy file.
        species_path: Path to species .npy file.
        shm_names: Dictionary mapping output names to shared memory names.
        dims_tuple: Tuple of (n_configs, n_atoms, n_states).
        batch_size: Number of configurations to process per batch.
        config: SEQMConfig object with calculation parameters.
        progress_shm_name: Shared memory name for progress counter.

    Raises:
        RuntimeError: If SEQM calculation fails.
    """
    # Set PyTorch defaults for numerical precision
    torch.set_default_dtype(torch.float64)

    # Select GPU device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # Unpack dimensions
    n_total, n_atoms, n_states = dims_tuple
    dims = OutputDimensions(n_total, n_atoms, n_states)

    # Load input data (memory-mapped, read-only)
    coords_mmap: NDArray[np.float64] = np.load(coords_path, mmap_mode="r")
    species_mmap: NDArray[np.int64] = np.load(species_path, mmap_mode="r")

    # Attach to shared memory blocks
    shm_handles: dict[str, shared_memory.SharedMemory] = {}
    for name, shm_name in shm_names.items():
        shm_handles[name] = shared_memory.SharedMemory(name=shm_name)

    # Create array views on shared memory
    E = np.ndarray(
        dims.energies_shape, dtype=np.float64, buffer=shm_handles["energies"].buf
    )
    F = np.ndarray(
        dims.forces_shape, dtype=np.float64, buffer=shm_handles["forces"].buf
    )
    GS_D = np.ndarray(
        dims.gs_dipole_shape, dtype=np.float64, buffer=shm_handles["gs_dipole"].buf
    )
    EX_D = np.ndarray(
        dims.ex_dipole_shape, dtype=np.float64, buffer=shm_handles["ex_dipole"].buf
    )
    TD = np.ndarray(
        dims.transition_dipole_shape,
        dtype=np.float64,
        buffer=shm_handles["trans_dipole"].buf,
    )
    Etot = np.ndarray(
        dims.total_energy_shape,
        dtype=np.float64,
        buffer=shm_handles["total_energy"].buf,
    )
    NACRs = np.ndarray(
        dims.nacr_shape, dtype=np.float64, buffer=shm_handles["nacr"].buf
    )

    # Progress tracking
    progress_shm = shared_memory.SharedMemory(name=progress_shm_name)
    progress_arr = np.ndarray((1,), dtype=np.int64, buffer=progress_shm.buf)

    # Get SEQM parameters
    params = config.to_seqm_params()

    # Process assigned range in batches
    start_idx, end_idx = idx_range

    for batch_start in range(start_idx, end_idx, batch_size):
        batch_end = min(batch_start + batch_size, end_idx)
        batch_len = batch_end - batch_start

        try:
            # Copy batch data to contiguous arrays
            coords_np = np.ascontiguousarray(coords_mmap[batch_start:batch_end]).copy()
            species_np = np.ascontiguousarray(
                species_mmap[batch_start:batch_end]
            ).copy()

            # Transfer to GPU with pinned memory for async transfer
            if device.type == "cuda":
                coords_tensor = torch.from_numpy(coords_np).pin_memory()
                species_tensor = torch.from_numpy(species_np).pin_memory()
                coords_tensor = coords_tensor.to(device, non_blocking=True)
                species_tensor = species_tensor.to(device, non_blocking=True)
            else:
                coords_tensor = torch.from_numpy(coords_np).to(device)
                species_tensor = torch.from_numpy(species_np).to(device)

            # Run SEQM electronic structure calculation
            const = Constants().to(device)
            mol = Molecule(const, params, coords_tensor, species_tensor).to(device)
            driver = Electronic_Structure(params).to(device)
            driver(mol)

            # Extract energies: ground state + excited states
            n_mol = mol.nmol
            all_energies = torch.empty(
                (n_mol, n_states), dtype=torch.float64, device=device
            )
            all_energies[:, 0] = mol.Etot  # Ground state = total SCF energy
            all_energies[:, 1:] = (
                mol.Etot.unsqueeze(1) + mol.cis_energies
            )  # Excited = SCF + CIS

            # Synchronize GPU before transferring to CPU
            if device.type == "cuda":
                torch.cuda.synchronize()

            # Write results to shared memory
            E[batch_start:batch_end, :] = all_energies.detach().cpu().numpy()
            F[batch_start:batch_end, :, :, :] = mol.all_forces.detach().cpu().numpy()
            GS_D[batch_start:batch_end, :] = mol.dipole.detach().cpu().numpy()
            EX_D[batch_start:batch_end, :, :] = (
                mol.all_cis_relaxed_diploles.detach().cpu().numpy()
            )
            TD[batch_start:batch_end, :, :] = (
                mol.transition_dipole.detach().cpu().numpy()
            )
            Etot[batch_start:batch_end, 0] = mol.Etot.detach().cpu().numpy()
            # NACRs have coupling index first: (n_couplings, batch, atoms, 3)
            NACRs[:, batch_start:batch_end, :, :] = mol.all_nac.detach().cpu().numpy()

            # Update progress counter (atomic operation)
            progress_arr[0] += batch_len

            # Free GPU memory
            del coords_tensor, species_tensor, const, mol, driver, all_energies
            if device.type == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            # Log error with context
            print(
                f"\n  ERROR in GPU {gpu_id} worker at batch "
                f"{batch_start}-{batch_end}: {type(e).__name__}: {e}"
            )
            raise

    # Close shared memory handles (don't unlink - parent will do that)
    for shm in shm_handles.values():
        shm.close()
    progress_shm.close()


# =============================================================================
# Main Function
# =============================================================================


def parse_arguments() -> argparse.Namespace:
    """Parse and validate command-line arguments.

    Returns:
        Namespace object containing all arguments.
    """
    parser = argparse.ArgumentParser(
        prog="run_grads_energies_tdipoles_pyseqm.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Version: {__version__}

Examples:
  # Basic usage with 4 GPUs
  python %(prog)s --coords coords.npy --species species.npy --gpus 0 1 2 3

  # Custom batch size and more excited states
  python %(prog)s --coords coords.npy --species species.npy --gpus 0 1 \\
      --batch 250 --n_states 21

  # Use PM3 method with tighter convergence
  python %(prog)s --coords coords.npy --species species.npy --gpus 0 \\
      --method PM3 --scf_eps 1e-12

For more information, see the module docstring or contact the authors.
        """,
    )

    # Required arguments group
    required = parser.add_argument_group("Required Arguments")
    required.add_argument(
        "--coords",
        type=Path,
        required=True,
        metavar="FILE",
        help=(
            "Path to coordinates .npy file. "
            "Expected shape: (N, A, 3) where N=configurations, A=atoms."
        ),
    )
    required.add_argument(
        "--species",
        type=Path,
        required=True,
        metavar="FILE",
        help=(
            "Path to atomic species .npy file. "
            "Expected shape: (N, A) with atomic numbers (e.g., 6=C, 7=N, 1=H)."
        ),
    )
    required.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        required=True,
        metavar="ID",
        help="GPU device IDs to use (e.g., 0 1 2 3). Work is split across GPUs.",
    )

    # Calculation parameters group
    calc = parser.add_argument_group("Calculation Parameters")
    calc.add_argument(
        "--batch",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        metavar="N",
        help=f"Batch size per GPU (default: {DEFAULT_BATCH_SIZE}). "
        "Reduce if running out of GPU memory.",
    )
    calc.add_argument(
        "--n_states",
        type=int,
        default=DEFAULT_N_STATES,
        metavar="N",
        help=f"Total electronic states including ground state (default: {DEFAULT_N_STATES}). "
        "E.g., 21 = 1 ground + 20 excited states.",
    )
    calc.add_argument(
        "--method",
        type=str,
        default="AM1",
        choices=SUPPORTED_METHODS,
        help="Semi-empirical method (default: AM1).",
    )
    calc.add_argument(
        "--scf_eps",
        type=float,
        default=DEFAULT_SCF_EPS,
        metavar="FLOAT",
        help=f"SCF convergence threshold (default: {DEFAULT_SCF_EPS}).",
    )
    calc.add_argument(
        "--cis_tol",
        type=float,
        default=DEFAULT_CIS_TOL,
        metavar="FLOAT",
        help=f"CIS convergence tolerance (default: {DEFAULT_CIS_TOL}).",
    )

    # Output files group
    output = parser.add_argument_group("Output Files")
    output.add_argument(
        "--out_e",
        type=Path,
        default=Path("energies_all_states.npy"),
        metavar="FILE",
        help="Output for all state energies (default: energies_all_states.npy).",
    )
    output.add_argument(
        "--out_f",
        type=Path,
        default=Path("forces_all_states.npy"),
        metavar="FILE",
        help="Output for all state forces (default: forces_all_states.npy).",
    )
    output.add_argument(
        "--out_gs_D",
        type=Path,
        default=Path("gs_dipoles.npy"),
        metavar="FILE",
        help="Output for ground state dipoles (default: gs_dipoles.npy).",
    )
    output.add_argument(
        "--out_ex_D",
        type=Path,
        default=Path("ex_dipoles_net.npy"),
        metavar="FILE",
        help="Output for excited state dipoles (default: ex_dipoles_net.npy).",
    )
    output.add_argument(
        "--out_td",
        type=Path,
        default=Path("transition_dipoles.npy"),
        metavar="FILE",
        help="Output for transition dipoles (default: transition_dipoles.npy).",
    )
    output.add_argument(
        "--out_etot",
        type=Path,
        default=Path("total_energy.npy"),
        metavar="FILE",
        help="Output for total SCF energy (default: total_energy.npy).",
    )
    output.add_argument(
        "--out_nacr",
        type=Path,
        default=Path("nacrs_all_states.npy"),
        metavar="FILE",
        help="Output for NACRs (default: nacrs_all_states.npy).",
    )

    # Version
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point for SEQM electronic structure calculations.

    Orchestrates the complete workflow:
        1. Parse and validate arguments
        2. Load and validate input data
        3. Allocate shared memory for outputs
        4. Launch GPU worker processes
        5. Monitor progress
        6. Save results to disk
        7. Clean up resources

    Returns:
        Exit code: 0 for success, non-zero for errors.
    """
    # ==========================================================================
    # Parse Arguments
    # ==========================================================================
    args = parse_arguments()

    # ==========================================================================
    # Start Timing
    # ==========================================================================
    start_datetime = datetime.now()
    start_time = time.time()

    print("\n" + "=" * 70)
    print("SEQM ELECTRONIC STRUCTURE CALCULATION")
    print("=" * 70)
    print(f"  Version:    {__version__}")
    print(f"  Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Method:     {args.method}")
    print(f"  GPUs:       {args.gpus}")
    print(f"  Batch size: {args.batch}")
    print(f"  States:     {args.n_states} (1 ground + {args.n_states - 1} excited)")
    print(f"  SCF eps:    {args.scf_eps}")
    print(f"  CIS tol:    {args.cis_tol}")

    # ==========================================================================
    # Validate Inputs
    # ==========================================================================
    print("\n" + "-" * 70)
    print("VALIDATING INPUTS")
    print("-" * 70)

    try:
        coords_mmap, species_mmap, n_configs, n_atoms = validate_inputs(
            args.coords, args.species
        )
        print(f"  ✓ Loaded {n_configs:,} configurations with {n_atoms} atoms each")
        print(f"  ✓ Coordinates file: {args.coords}")
        print(f"  ✓ Species file: {args.species}")
    except (FileNotFoundError, ValueError) as e:
        print(f"\n  ✗ ERROR: {e}")
        return 1

    # Validate GPU availability
    if torch.cuda.is_available():
        n_gpus_available = torch.cuda.device_count()
        invalid_gpus = [g for g in args.gpus if g >= n_gpus_available]
        if invalid_gpus:
            print(
                f"\n  ✗ ERROR: GPU(s) {invalid_gpus} requested but only "
                f"{n_gpus_available} available (0-{n_gpus_available - 1})"
            )
            return 1
        print(f"  ✓ Using {len(args.gpus)} GPU(s): {args.gpus}")
        for gpu_id in args.gpus:
            gpu_name = torch.cuda.get_device_name(gpu_id)
            gpu_mem = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9
            print(f"      GPU {gpu_id}: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("  ⚠ Warning: CUDA not available, will use CPU (very slow)")

    # ==========================================================================
    # Create Configuration and Dimensions
    # ==========================================================================
    try:
        config = SEQMConfig(
            method=args.method,
            scf_eps=args.scf_eps,
            n_states=args.n_states,
            cis_tol=args.cis_tol,
        )
    except ValueError as e:
        print(f"\n  ✗ Configuration error: {e}")
        return 1

    dims = OutputDimensions(n_configs, n_atoms, args.n_states)
    dims.print_summary()

    # ==========================================================================
    # Allocate Shared Memory
    # ==========================================================================
    print("-" * 70)
    print("ALLOCATING SHARED MEMORY")
    print("-" * 70)

    shm_manager = SharedMemoryManager(dims)

    try:
        shm_manager.allocate()
        print(f"  ✓ Allocated {dims.get_total_memory_gb():.2f} GB shared memory")
    except MemoryError as e:
        print(f"\n  ✗ ERROR: Insufficient memory - {e}")
        return 1
    except Exception as e:
        print(f"\n  ✗ ERROR allocating shared memory: {e}")
        shm_manager.cleanup()
        return 1

    # Progress counter in shared memory
    progress_shm = shared_memory.SharedMemory(create=True, size=8)
    progress_arr = np.ndarray((1,), dtype=np.int64, buffer=progress_shm.buf)
    progress_arr[0] = 0

    # ==========================================================================
    # Launch Worker Processes
    # ==========================================================================
    print("\n" + "-" * 70)
    print("RUNNING CALCULATIONS")
    print("-" * 70)

    # Split work across GPUs
    ranges = split_contiguous(n_configs, len(args.gpus))

    print("  Work distribution:")
    for gpu_id, (start, end) in zip(args.gpus, ranges):
        print(f"    GPU {gpu_id}: configs {start:>7,} - {end:>7,} ({end - start:,})")

    # Prepare shared memory names
    shm_names = shm_manager.get_names()
    dims_tuple = (n_configs, n_atoms, args.n_states)

    # Launch workers using spawn context (required for CUDA)
    ctx = mp.get_context("spawn")
    processes: list[SpawnProcess] = []

    print("\n  Starting workers...")
    calc_start_time = time.time()

    for gpu_id, idx_range in zip(args.gpus, ranges):
        p = ctx.Process(
            target=worker,
            args=(
                gpu_id,
                idx_range,
                str(args.coords.absolute()),
                str(args.species.absolute()),
                shm_names,
                dims_tuple,
                args.batch,
                config,
                progress_shm.name,
            ),
        )
        p.start()
        processes.append(p)

    # Monitor progress
    print()
    try:
        while any(p.is_alive() for p in processes):
            current = int(progress_arr[0])
            print_progress_bar(current, n_configs, calc_start_time)
            time.sleep(0.5)

        # Final progress update
        print_progress_bar(n_configs, n_configs, calc_start_time)

    except KeyboardInterrupt:
        print("\n\n  ⚠ Interrupted! Terminating workers...")
        for p in processes:
            p.terminate()
            p.join(timeout=5)
        shm_manager.cleanup()
        progress_shm.close()
        progress_shm.unlink()
        return 130  # Standard interrupt exit code

    # Wait for all processes and check exit codes
    exit_codes = []
    for p in processes:
        p.join()
        exit_codes.append(p.exitcode)

    # Check for worker errors
    if any(code != 0 for code in exit_codes):
        print("\n  ✗ ERROR: One or more workers failed:")
        for gpu_id, code in zip(args.gpus, exit_codes):
            status = "✓ OK" if code == 0 else f"✗ Exit code {code}"
            print(f"      GPU {gpu_id}: {status}")
        shm_manager.cleanup()
        progress_shm.close()
        progress_shm.unlink()
        return 1

    calc_time = time.time() - calc_start_time
    print(f"\n  ✓ Calculations completed in {format_duration(calc_time)}")
    print(f"    Throughput: {n_configs / calc_time:.1f} configurations/second")

    # ==========================================================================
    # Save Results
    # ==========================================================================
    print("\n" + "-" * 70)
    print("SAVING RESULTS")
    print("-" * 70)

    try:
        # Save all output arrays
        np.save(args.out_e, shm_manager.arrays["energies"])
        np.save(args.out_f, shm_manager.arrays["forces"])
        np.save(args.out_gs_D, shm_manager.arrays["gs_dipole"])
        np.save(args.out_td, shm_manager.arrays["trans_dipole"])
        np.save(args.out_etot, shm_manager.arrays["total_energy"])
        np.save(args.out_nacr, shm_manager.arrays["nacr"])

        # Compute net excited state dipoles (GS dipole + relaxed difference)
        gs_expanded = shm_manager.arrays["gs_dipole"][:, np.newaxis, :]  # (N, 1, 3)
        net_ex_dipoles = shm_manager.arrays["ex_dipole"] + gs_expanded  # (N, n_ex, 3)
        np.save(args.out_ex_D, net_ex_dipoles)

        print("  Saved output files:")
        print(f"    ✓ {str(args.out_e):<30} {shm_manager.arrays['energies'].shape}")
        print(f"    ✓ {str(args.out_f):<30} {shm_manager.arrays['forces'].shape}")
        print(f"    ✓ {str(args.out_gs_D):<30} {shm_manager.arrays['gs_dipole'].shape}")
        print(f"    ✓ {str(args.out_ex_D):<30} {net_ex_dipoles.shape}")
        print(
            f"    ✓ {str(args.out_td):<30} {shm_manager.arrays['trans_dipole'].shape}"
        )
        print(
            f"    ✓ {str(args.out_etot):<30} {shm_manager.arrays['total_energy'].shape}"
        )
        print(f"    ✓ {str(args.out_nacr):<30} {shm_manager.arrays['nacr'].shape}")

    except Exception as e:
        print(f"\n  ✗ ERROR saving results: {e}")
        shm_manager.cleanup()
        progress_shm.close()
        progress_shm.unlink()
        return 1

    # ==========================================================================
    # Cleanup
    # ==========================================================================
    shm_manager.cleanup()
    progress_shm.close()
    progress_shm.unlink()

    # ==========================================================================
    # Summary
    # ==========================================================================
    end_datetime = datetime.now()
    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"  Start time:   {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  End time:     {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Total time:   {format_duration(total_time)}")
    print(f"  Calc time:    {format_duration(calc_time)}")
    print(f"  Throughput:   {n_configs / calc_time:.1f} configs/sec")
    print(f"  Configs:      {n_configs:,}")
    print(f"  States:       {args.n_states}")
    print(f"  GPUs used:    {len(args.gpus)}")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
