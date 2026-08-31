#!/usr/bin/env python3
"""
Unified HIPPYNN Training Script for Ground-to-Excited State Property Prediction.

Supports training any combination of molecular properties via independent flags:
  --train_energies      Per-state energies (default: True)
  --train_forces        Per-state atomic forces via autodiff F = -dE/dR
  --train_tdipoles      Per-state transition dipoles via predicted charges D = sum(q*r)
  --train_scaled_nacrs  Pairwise scaled Non-Adiabatic Coupling Vectors (dENACRs)

Any combination of flags is valid.  Nodes that are structurally required by an
active flag are built automatically (e.g., energy nodes for force autodiff) and
their metrics are tracked for monitoring, but only flagged properties enter the
training loss.

State selection:
  Standard mode   -- N_STATES + INCLUDE_GROUND_STATE
  Target mode     -- TARGET_STATE_LEARNING = <int> trains only that state

Configuration precedence: CLI arguments > config file > built-in defaults.

Usage examples:
  python unified_training.py --train_energies --train_forces --data_dir /path/to/data
  python unified_training.py --config_file training.inp
  python unified_training.py --train_scaled_nacrs --no_train_energies --n_states 3
  python unified_training.py --train_energies --train_tdipoles --target_state 2

For more information on HIPPYNN, see:
  - HIPPYNN Documentation: https://hippynn.readthedocs.io/
  
 Developer Information
---------------
  __developer__: "Ajay Khanna"
  __place__: "LANL"
  __date__: "May-01-2026"
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import hippynn  # type: ignore[import]
from hippynn import plotting  # type: ignore[import]
from hippynn.experiment import setup_training, train_model  # type: ignore[import]
from hippynn.experiment.serialization import (  # type: ignore[import]
    load_checkpoint,
    load_checkpoint_from_cwd,
)
from hippynn.experiment.controllers import (  # type: ignore[import]
    PatienceController,
    RaiseBatchSizeOnPlateau,
)
from hippynn.graphs import IdxType, inputs, loss, networks, physics, targets  # type: ignore[import]
from hippynn.graphs.nodes import base  # type: ignore[import]
from hippynn.graphs.nodes.excited import (  # type: ignore[import]
    MAEPhaseLoss,
    MSEPhaseLoss,
    NACRNode,
)
from hippynn.graphs.indextypes.reduce_funcs import elementwise_compare_reduce  # type: ignore[import]

from direct_nacr_head import DirectScaledNACRNode
from direct_nacr_head_v2 import DirectScaledNACRV2Node, V2_MODES
from direct_nacr_head_v3 import DirectScaledNACRV3Node, V3_MODES

matplotlib.use("Agg")


# =============================================================================
# Built-in Defaults
# =============================================================================

DEFAULTS: Dict[str, Any] = {
    # --- Paths ---
    "data_dir": "",
    "output_dir": "",
    # --- Dataset ---
    "dataset_prefix": "acn_",
    "n_atoms": 15,
    "seed": 114514,
    "test_size": 0.1,
    "valid_size": 0.1,
    # --- Property flags (what enters the loss) ---
    "train_energies": True,
    "train_forces": False,
    "train_tdipoles": False,
    "train_scaled_nacrs": False,
    # --- Scaled NACR architecture ---
    "scaled_nacr_head": "charge",
    "direct_nacr_n_dist": None,
    "direct_nacr_edge_scale": 0.5,
    "direct_nacr_symmetric_coeffs": True,
    "direct_nacr_enforce_zero_sum": True,
    "direct_nacr_first_is_interacting": False,
    "direct_nacr_v2_mode": "gdv_pair_mixing_residual",
    "gdv_source": "file",
    "gdv_file_template": "{prefix}GDV{pair}.npy",
    "gap_file_template": "{prefix}GAP{pair}.npy",
    "direct_nacr_v2_edge_scale": 0.5,
    "direct_nacr_v2_symmetric_coeffs": True,
    "direct_nacr_v2_enforce_zero_sum": True,
    "direct_nacr_v2_first_is_interacting": False,
    "direct_nacr_v2_atom_feature_index": -1,
    "direct_nacr_v2_atom_gate_hidden": None,
    # --- State selection ---
    "n_states": 3,
    "include_ground_state": True,
    "target_state": None,
    "targeted_scaled_nacr_pairs": None,
    # --- Network ---
    "network_type": "hiphopnn",
    "possible_species": [0, 1, 6, 8],
    "n_features": 75,
    "n_sensitivities": 63,
    "dist_soft_min": 0.7,
    "dist_soft_max": 5.67638290906325,
    "dist_hard_max": 11.833504244685173,
    "n_interaction_layers": 2,
    "n_atom_layers": 3,
    "l_max": 3,
    "n_max": 4,
    # --- Training hyperparameters ---
    "batch_size": 64,
    "eval_batch_size": 256,
    "max_batch_size": 256,
    "max_epochs": 200,
    "learning_rate": 5e-4,
    "termination_patience": 50,
    "scheduler_patience": 16,
    "scheduler_factor": 0.2,
    "stopping_key": "Loss",
    "fraction_train_eval": 0.1,
    "plot_frequency": 50,
    # --- Global loss weights ---
    "energy_weight": 1.0,
    "force_weight": 1.0,
    "tdipole_weight": 1.0,
    "nacr_weight": 1.0,
    "joint_state_nacr_loss": False,
    "l2_weight": 2e-5,
    # --- Per-state loss weights ---
    "state_energy_weights": {},
    "state_force_weights": {},
    "state_tdipole_weights": {},
    "state_nacr_weights": {},
    # --- Phase-sign correction ---
    "phasesign_corrected_scaled_nacr": False,
    "phasesign_corrected_dipoles": False,
    # --- Physics ---
    "force_sign": -1,
    # --- Torch defaults ---
    "allow_tf32": True,
    "default_dtype": "float32",
    # --- Restart ---
    "restart": False,
    "restart_dir": "",
}


# =============================================================================
# TrainingConfig Dataclass
# =============================================================================


@dataclass
class TrainingConfig:
    """Resolved, validated configuration for training."""

    # Paths
    data_dir: str = ""
    output_dir: str = ""
    # Dataset
    dataset_prefix: str = "acn_"
    n_atoms: int = 15
    seed: int = 114514
    test_size: float = 0.1
    valid_size: float = 0.1
    # Property flags
    train_energies: bool = True
    train_forces: bool = False
    train_tdipoles: bool = False
    train_scaled_nacrs: bool = False
    # Scaled NACR architecture
    scaled_nacr_head: str = "charge"
    direct_nacr_n_dist: Optional[int] = None
    direct_nacr_edge_scale: float = 0.5
    direct_nacr_symmetric_coeffs: bool = True
    direct_nacr_enforce_zero_sum: bool = True
    direct_nacr_first_is_interacting: bool = False
    direct_nacr_v2_mode: str = "gdv_pair_mixing_residual"
    gdv_source: str = "file"
    gdv_file_template: str = "{prefix}GDV{pair}.npy"
    gap_file_template: str = "{prefix}GAP{pair}.npy"
    direct_nacr_v2_edge_scale: float = 0.5
    direct_nacr_v2_symmetric_coeffs: bool = True
    direct_nacr_v2_enforce_zero_sum: bool = True
    direct_nacr_v2_first_is_interacting: bool = False
    direct_nacr_v2_atom_feature_index: int = -1
    direct_nacr_v2_atom_gate_hidden: Optional[int] = None
    # State selection
    n_states: int = 3
    include_ground_state: bool = True
    target_state: Optional[int] = None
    targeted_scaled_nacr_pairs: Optional[List[Tuple[int, int]]] = None
    # Network
    network_type: str = "hiphopnn"
    possible_species: List[int] = field(default_factory=lambda: [0, 1, 6, 8])
    n_features: int = 75
    n_sensitivities: int = 63
    dist_soft_min: float = 0.7
    dist_soft_max: float = 5.67638290906325
    dist_hard_max: float = 11.833504244685173
    n_interaction_layers: int = 2
    n_atom_layers: int = 3
    l_max: int = 3
    n_max: int = 4
    # Training hyperparameters
    batch_size: int = 64
    eval_batch_size: int = 256
    max_batch_size: int = 256
    max_epochs: int = 200
    learning_rate: float = 5e-4
    termination_patience: int = 50
    scheduler_patience: int = 16
    scheduler_factor: float = 0.2
    stopping_key: str = "Loss"
    fraction_train_eval: float = 0.1
    plot_frequency: int = 50
    # Global loss weights
    energy_weight: float = 1.0
    force_weight: float = 1.0
    tdipole_weight: float = 1.0
    nacr_weight: float = 1.0
    joint_state_nacr_loss: bool = False
    l2_weight: float = 2e-5
    # Per-state loss weights
    state_energy_weights: Dict[int, float] = field(default_factory=dict)
    state_force_weights: Dict[int, float] = field(default_factory=dict)
    state_tdipole_weights: Dict[int, float] = field(default_factory=dict)
    state_nacr_weights: Dict[Tuple[int, int], float] = field(default_factory=dict)
    # Phase-sign correction
    phasesign_corrected_scaled_nacr: bool = False
    phasesign_corrected_dipoles: bool = False
    # Physics
    force_sign: int = -1
    # Torch
    allow_tf32: bool = True
    default_dtype: str = "float32"
    # Restart
    restart: bool = False
    restart_dir: str = ""

    # --- Derived: which node categories must be built ---

    @property
    def use_charge_nacr_head(self) -> bool:
        """Use the built-in charge/energy-gap NACR ansatz."""
        return self.train_scaled_nacrs and self.scaled_nacr_head == "charge"

    @property
    def use_direct_nacr_head(self) -> bool:
        """Use the local direct equivariant scaled-NACR vector head."""
        return (
            self.train_scaled_nacrs
            and self.scaled_nacr_head == "direct_vector"
        )

    @property
    def use_direct_nacr_v2_head(self) -> bool:
        """Use the local GDV-aware direct scaled-NACR vector head."""
        return (
            self.train_scaled_nacrs
            and self.scaled_nacr_head == "direct_vector_v2"
        )

    @property
    def use_direct_nacr_v3_head(self) -> bool:
        """Use the GDV+gap-aware direct scaled-NACR vector head (v2 + energy-gap gate)."""
        return (
            self.train_scaled_nacrs
            and self.scaled_nacr_head == "direct_vector_v3"
        )

    @property
    def build_energy_nodes(self) -> bool:
        """Energy nodes are needed for energy/force or charge-NACR paths."""
        return (
            self.train_energies
            or self.train_forces
            or self.use_charge_nacr_head
        )

    @property
    def build_force_nodes(self) -> bool:
        """Force nodes are needed only when training forces."""
        return self.train_forces

    @property
    def build_charge_nodes(self) -> bool:
        """Charge nodes are needed for dipoles or charge-NACR paths."""
        return self.train_tdipoles or self.use_charge_nacr_head

    @property
    def build_dipole_nodes(self) -> bool:
        """Dipole nodes are needed only when training dipoles."""
        return self.train_tdipoles

    @property
    def build_nacr_nodes(self) -> bool:
        """NACR nodes are needed only when training NACRs."""
        return self.train_scaled_nacrs

    @property
    def needs_position_grad(self) -> bool:
        """positions.requires_grad is needed for force or charge-NACR autodiff."""
        return self.train_forces or self.use_charge_nacr_head

    @property
    def network_params(self) -> Dict[str, Any]:
        """Assemble the network hyperparameter dict for build_network."""
        return {
            "possible_species": list(self.possible_species),
            "n_features": self.n_features,
            "n_sensitivities": self.n_sensitivities,
            "dist_soft_min": self.dist_soft_min,
            "dist_soft_max": self.dist_soft_max,
            "dist_hard_max": self.dist_hard_max,
            "n_interaction_layers": self.n_interaction_layers,
            "n_atom_layers": self.n_atom_layers,
            "l_max": self.l_max,
            "n_max": self.n_max,
        }

    @property
    def active_flags_tag(self) -> str:
        """Short tag string for the active property flags, e.g. 'E_F_D'."""
        parts: List[str] = []
        if self.train_energies:
            parts.append("E")
        if self.train_forces:
            parts.append("F")
        if self.train_tdipoles:
            parts.append("D")
        if self.train_scaled_nacrs:
            parts.append("N")
        return "_".join(parts) if parts else "NONE"


# =============================================================================
# Config File Parser
# =============================================================================


def parse_config_file(path: str) -> Dict[str, str]:
    """Parse a KEY = VALUE config file.

    Lines starting with ``#`` are ignored.  Returns raw string values
    keyed by lower-case key names.
    """
    config: Dict[str, str] = {}
    with open(path, "r") as fh:
        for lineno, raw_line in enumerate(fh, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                print(
                    f"WARNING: Ignoring malformed line {lineno} in "
                    f"{path}: {raw_line.rstrip()}"
                )
                continue
            key, _, val = line.partition("=")
            key = key.strip().lower()
            val = val.strip()
            if "  #" in val:
                val = val[: val.index("  #")].strip()
            config[key] = val
    return config


# =============================================================================
# Per-State Weight Parsing
# =============================================================================


def parse_state_weights_single(tokens: List[str]) -> Dict[int, float]:
    """Parse ``['S0:1.0', 'S1:2.0']`` -> ``{0: 1.0, 1: 2.0}``."""
    weights: Dict[int, float] = {}
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if ":" not in tok:
            raise ValueError(
                f"Invalid per-state weight token '{tok}'. "
                "Expected format: S<n>:<weight>"
            )
        key_str, val_str = tok.split(":", 1)
        key_str = key_str.strip().upper().lstrip("S")
        try:
            state_idx = int(key_str)
        except ValueError:
            raise ValueError(
                f"Invalid state index in weight token '{tok}'. "
                "Expected integer after 'S'."
            )
        try:
            weight = float(val_str)
        except ValueError:
            raise ValueError(
                f"Invalid weight value in token '{tok}'. Expected a float."
            )
        weights[state_idx] = weight
    return weights


def parse_state_weights_pair(tokens: List[str]) -> Dict[Tuple[int, int], float]:
    """Parse ``['12:1.0', '23:2.0']`` -> ``{(1,2): 1.0, (2,3): 2.0}``."""
    weights: Dict[Tuple[int, int], float] = {}
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if ":" not in tok:
            raise ValueError(
                f"Invalid NACR weight token '{tok}'. "
                "Expected format: <ij>:<weight>"
            )
        key_str, val_str = tok.split(":", 1)
        key_str = key_str.strip()
        if len(key_str) != 2 or not key_str.isdigit():
            raise ValueError(
                f"Invalid NACR pair key '{key_str}' in token '{tok}'. "
                "Expected two-digit pair index (e.g. 12, 23)."
            )
        i, j = int(key_str[0]), int(key_str[1])
        if i >= j:
            raise ValueError(
                f"NACR pair must have i < j, got {i}{j} in token '{tok}'."
            )
        try:
            weight = float(val_str)
        except ValueError:
            raise ValueError(
                f"Invalid weight value in NACR token '{tok}'. Expected a float."
            )
        weights[(i, j)] = weight
    return weights


# =============================================================================
# CLI Argument Parser
# =============================================================================


def parse_cli_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Define and parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=(
            "Unified HIPPYNN Training Script for Excited-State Properties.\n"
            "Select which properties enter the loss with --train_* flags."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Config file
    p.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Path to a KEY=VALUE configuration file.",
    )

    # ---- Property flags ----
    fg = p.add_argument_group(
        "Property Flags",
        "Select which properties contribute to the training loss. "
        "Any combination is valid.  Structural dependencies are resolved "
        "automatically (e.g., --train_forces builds energy nodes for "
        "autodiff).  Use --no_train_* to explicitly disable a flag.",
    )
    fg.add_argument(
        "--train_energies",
        action="store_true",
        default=None,
        help="Include per-state energies in the loss (default: True).",
    )
    fg.add_argument(
        "--no_train_energies",
        action="store_true",
        default=False,
        help="Exclude energies from the loss.",
    )
    fg.add_argument(
        "--train_forces",
        action="store_true",
        default=None,
        help="Include per-state forces in the loss.",
    )
    fg.add_argument(
        "--no_train_forces",
        action="store_true",
        default=False,
        help="Exclude forces from the loss.",
    )
    fg.add_argument(
        "--train_tdipoles",
        action="store_true",
        default=None,
        help="Include per-state dipoles in the loss.",
    )
    fg.add_argument(
        "--no_train_tdipoles",
        action="store_true",
        default=False,
        help="Exclude dipoles from the loss.",
    )
    fg.add_argument(
        "--train_scaled_nacrs",
        action="store_true",
        default=None,
        help="Include scaled NACRs (dENACRs) in the loss.",
    )
    fg.add_argument(
        "--no_train_scaled_nacrs",
        action="store_true",
        default=False,
        help="Exclude scaled NACRs from the loss.",
    )

    # ---- Scaled NACR architecture ----
    nag = p.add_argument_group("Scaled NACR Architecture")
    nag.add_argument(
        "--scaled_nacr_head",
        type=str,
        choices=("charge", "direct_vector", "direct_vector_v2", "direct_vector_v3"),
        default=None,
        help=(
            "Scaled NACR head: 'charge' uses the built-in HIPPYNN "
            "charge/energy-gap NACRNode; 'direct_vector' uses the local "
            "direct equivariant vector head; 'direct_vector_v2' uses the "
            "local oracle-GDV-aware vector head."
        ),
    )
    nag.add_argument("--direct_nacr_n_dist", type=int, default=None)
    nag.add_argument("--direct_nacr_edge_scale", type=float, default=None)
    nag.add_argument(
        "--direct_nacr_symmetric_coeffs",
        action="store_true",
        default=None,
    )
    nag.add_argument(
        "--no_direct_nacr_symmetric_coeffs",
        action="store_true",
        default=False,
    )
    nag.add_argument(
        "--direct_nacr_enforce_zero_sum",
        action="store_true",
        default=None,
    )
    nag.add_argument(
        "--no_direct_nacr_enforce_zero_sum",
        action="store_true",
        default=False,
    )
    nag.add_argument(
        "--direct_nacr_first_is_interacting",
        action="store_true",
        default=None,
    )
    nag.add_argument(
        "--no_direct_nacr_first_is_interacting",
        action="store_true",
        default=False,
    )
    nag.add_argument(
        "--direct_nacr_v2_mode",
        type=str,
        choices=V2_MODES,
        default=None,
    )
    nag.add_argument(
        "--gdv_source",
        type=str,
        choices=("file",),
        default=None,
    )
    nag.add_argument("--gdv_file_template", type=str, default=None)
    nag.add_argument("--gap_file_template", type=str, default=None)
    nag.add_argument(
        "--direct_nacr_v2_edge_scale",
        type=float,
        default=None,
    )
    nag.add_argument(
        "--direct_nacr_v2_symmetric_coeffs",
        action="store_true",
        default=None,
    )
    nag.add_argument(
        "--no_direct_nacr_v2_symmetric_coeffs",
        action="store_true",
        default=False,
    )
    nag.add_argument(
        "--direct_nacr_v2_enforce_zero_sum",
        action="store_true",
        default=None,
    )
    nag.add_argument(
        "--no_direct_nacr_v2_enforce_zero_sum",
        action="store_true",
        default=False,
    )
    nag.add_argument(
        "--direct_nacr_v2_first_is_interacting",
        action="store_true",
        default=None,
    )
    nag.add_argument(
        "--no_direct_nacr_v2_first_is_interacting",
        action="store_true",
        default=False,
    )
    nag.add_argument(
        "--direct_nacr_v2_atom_feature_index",
        type=int,
        default=None,
    )
    nag.add_argument(
        "--direct_nacr_v2_atom_gate_hidden",
        type=int,
        default=None,
    )

    # ---- State selection ----
    sg = p.add_argument_group("State Selection")
    sg.add_argument("--n_states", type=int, default=None,
                    help="Number of excited states (default: 3).")
    sg.add_argument("--include_ground_state", action="store_true",
                    default=None, help="Include ground state S0 (default: True).")
    sg.add_argument("--no_ground_state", action="store_true",
                    default=False, help="Exclude ground state S0.")
    sg.add_argument("--target_state", type=int, default=None,
                    help="TARGET_STATE_LEARNING: train only this state index.")
    sg.add_argument("--targeted_scaled_nacr_pairs", type=str, nargs="+",
                    default=None, help="Specific NACR pairs (e.g. 12 23 34).")

    # ---- Network ----
    ng = p.add_argument_group("Network Architecture")
    ng.add_argument("--network_type", type=str,
                    choices=("hipnn", "hiphopnn"), default=None)
    ng.add_argument("--possible_species", type=int, nargs="+", default=None)
    ng.add_argument("--n_features", type=int, default=None)
    ng.add_argument("--n_sensitivities", type=int, default=None)
    ng.add_argument("--n_interaction_layers", type=int, default=None)
    ng.add_argument("--n_atom_layers", type=int, default=None)
    ng.add_argument("--dist_soft_min", type=float, default=None)
    ng.add_argument("--dist_soft_max", type=float, default=None)
    ng.add_argument("--dist_hard_max", type=float, default=None)
    ng.add_argument("--l_max", type=int, default=None)
    ng.add_argument("--n_max", type=int, default=None)

    # ---- Training hyperparameters ----
    tg = p.add_argument_group("Training Hyperparameters")
    tg.add_argument("--batch_size", type=int, default=None)
    tg.add_argument("--eval_batch_size", type=int, default=None)
    tg.add_argument("--max_batch_size", type=int, default=None)
    tg.add_argument("--max_epochs", type=int, default=None)
    tg.add_argument("--learning_rate", type=float, default=None)
    tg.add_argument("--termination_patience", type=int, default=None)
    tg.add_argument("--scheduler_patience", type=int, default=None)
    tg.add_argument("--scheduler_factor", type=float, default=None)
    tg.add_argument("--stopping_key", type=str, default=None)
    tg.add_argument("--fraction_train_eval", type=float, default=None)
    tg.add_argument("--plot_frequency", type=int, default=None)

    # ---- Global loss weights ----
    wg = p.add_argument_group("Global Loss Weights")
    wg.add_argument("--energy_weight", type=float, default=None)
    wg.add_argument("--force_weight", type=float, default=None)
    wg.add_argument("--tdipole_weight", type=float, default=None)
    wg.add_argument("--nacr_weight", type=float, default=None)
    wg.add_argument("--joint_state_nacr_loss", action="store_true", default=None,
                    help="Tier-1 sign fix: joint per-state phase loss for dENACR (replaces per-pair phaseless).")
    wg.add_argument("--l2_weight", type=float, default=None)

    # ---- Per-state loss weights ----
    swg = p.add_argument_group("Per-State Loss Weights")
    swg.add_argument("--state_energy_weights", type=str, nargs="*",
                     default=None, help="e.g. S0:1.0 S1:2.0")
    swg.add_argument("--state_force_weights", type=str, nargs="*",
                     default=None, help="e.g. S0:1.0 S1:2.0")
    swg.add_argument("--state_tdipole_weights", type=str, nargs="*",
                     default=None, help="e.g. S1:1.0 S2:2.0")
    swg.add_argument("--state_nacr_weights", type=str, nargs="*",
                     default=None, help="e.g. 12:1.0 23:2.0")

    # ---- Phase-sign correction ----
    pcg = p.add_argument_group(
        "Phase-Sign Correction",
        "When training data has been phase-sign corrected, use standard "
        "MSE/MAE for the training loss instead of phase-less variants.  "
        "Phase-less metrics are always tracked as diagnostics.",
    )
    pcg.add_argument(
        "--phasesign_corrected_scaled_nacr",
        action="store_true",
        default=None,
        help="Scaled NACR data is phase-sign corrected; use standard MSE/MAE.",
    )
    pcg.add_argument(
        "--no_phasesign_corrected_scaled_nacr",
        action="store_true",
        default=False,
        help="Scaled NACR data is NOT phase-sign corrected (default).",
    )
    pcg.add_argument(
        "--phasesign_corrected_dipoles",
        action="store_true",
        default=None,
        help="Dipole data is phase-sign corrected; use standard MSE/MAE.",
    )
    pcg.add_argument(
        "--no_phasesign_corrected_dipoles",
        action="store_true",
        default=False,
        help="Dipole data is NOT phase-sign corrected (default).",
    )

    # ---- Data paths ----
    dg = p.add_argument_group("Data Configuration")
    dg.add_argument("--data_dir", type=str, default=None)
    dg.add_argument("--dataset_prefix", type=str, default=None)
    dg.add_argument("--n_atoms", type=int, default=None)
    dg.add_argument("--seed", type=int, default=None)
    dg.add_argument("--test_size", type=float, default=None)
    dg.add_argument("--valid_size", type=float, default=None)

    # ---- Output ----
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output directory name (auto-generated if blank).")

    # ---- Physics ----
    p.add_argument("--force_sign", type=int, default=None,
                   help="Sign for force gradient (default: -1).")

    # ---- Torch ----
    p.add_argument("--allow_tf32", action="store_true", default=None)
    p.add_argument("--no_tf32", action="store_true", default=False)
    p.add_argument("--default_dtype", type=str,
                   choices=("float32", "float64"), default=None)

    # ---- Restart ----
    rg = p.add_argument_group(
        "Restart",
        "Resume training from a previous checkpoint.  The script loads "
        "the saved model, optimizer, scheduler, and metric state, then "
        "continues training.  The database is reloaded from DATA_DIR.",
    )
    rg.add_argument(
        "--restart",
        action="store_true",
        default=None,
        help="Resume training from a checkpoint in the output directory.",
    )
    rg.add_argument(
        "--no_restart",
        action="store_true",
        default=False,
        help="Start fresh training (default).",
    )
    rg.add_argument(
        "--restart_dir",
        type=str,
        default=None,
        help="Explicit path to checkpoint directory.  Defaults to "
             "OUTPUT_DIR (or auto-generated name).",
    )

    return p.parse_args(argv)


# =============================================================================
# Configuration Merging & Resolution
# =============================================================================

# Alias mapping: config-file key -> internal DEFAULTS key
_CONFIG_KEY_ALIASES: Dict[str, str] = {
    "target_state_learning": "target_state",
}

_SINGLE_STATE_WEIGHT_KEYS = (
    "state_energy_weights",
    "state_force_weights",
    "state_tdipole_weights",
)
_PAIR_STATE_WEIGHT_KEYS = ("state_nacr_weights",)

_BOOL_TRUTHY = {"true", "1", "yes", "on"}
_BOOL_FALSY = {"false", "0", "no", "off", ""}

# Property flags and their CLI negation counterparts
_FLAG_NEGATION = {
    "train_energies": "no_train_energies",
    "train_forces": "no_train_forces",
    "train_tdipoles": "no_train_tdipoles",
    "train_scaled_nacrs": "no_train_scaled_nacrs",
    "phasesign_corrected_scaled_nacr": "no_phasesign_corrected_scaled_nacr",
    "phasesign_corrected_dipoles": "no_phasesign_corrected_dipoles",
    "direct_nacr_symmetric_coeffs": "no_direct_nacr_symmetric_coeffs",
    "direct_nacr_enforce_zero_sum": "no_direct_nacr_enforce_zero_sum",
    "direct_nacr_first_is_interacting": (
        "no_direct_nacr_first_is_interacting"
    ),
    "direct_nacr_v2_symmetric_coeffs": (
        "no_direct_nacr_v2_symmetric_coeffs"
    ),
    "direct_nacr_v2_enforce_zero_sum": (
        "no_direct_nacr_v2_enforce_zero_sum"
    ),
    "direct_nacr_v2_first_is_interacting": (
        "no_direct_nacr_v2_first_is_interacting"
    ),
    "restart": "no_restart",
}


def _coerce_config_value(key: str, raw: str, default_val: Any) -> Any:
    """Coerce a raw config-file string to the Python type implied by
    *default_val*."""
    if raw == "" or raw is None:
        return None
    if default_val is None and str(raw).lower() in {"none", "null"}:
        return None

    if key in _SINGLE_STATE_WEIGHT_KEYS:
        tokens = raw.split()
        return parse_state_weights_single(tokens) if tokens else {}
    if key in _PAIR_STATE_WEIGHT_KEYS:
        tokens = raw.split()
        return parse_state_weights_pair(tokens) if tokens else {}
    if key == "targeted_scaled_nacr_pairs":
        tokens = [t.strip().strip(",") for t in raw.replace(",", " ").split()]
        return tokens if tokens else None
    if key == "possible_species":
        return [int(x) for x in raw.split()]

    if isinstance(default_val, bool):
        low = raw.lower()
        if low in _BOOL_TRUTHY:
            return True
        if low in _BOOL_FALSY:
            return False
        raise ValueError(f"Cannot parse '{raw}' as bool for key '{key}'")
    if isinstance(default_val, int):
        return int(raw)
    if isinstance(default_val, float):
        return float(raw)
    if default_val is None:
        try:
            return int(raw)
        except ValueError:
            return raw
    return raw


def merge_and_resolve(
    cli: argparse.Namespace,
    file_config: Optional[Dict[str, str]],
) -> TrainingConfig:
    """Merge CLI args, config-file values, and defaults into a TrainingConfig.

    Precedence: CLI > config file > DEFAULTS.
    """
    resolved: Dict[str, Any] = {}
    overrides: List[str] = []

    for key, default_val in DEFAULTS.items():
        # --- Layer 1: default ---
        val = copy.deepcopy(default_val)

        # --- Layer 2: config file ---
        file_val = None
        if file_config:
            file_key = key
            if key not in file_config:
                for alias, target in _CONFIG_KEY_ALIASES.items():
                    if target == key and alias in file_config:
                        file_key = alias
                        break
            if file_key in file_config:
                raw = file_config[file_key]
                file_val = _coerce_config_value(key, raw, default_val)
                if file_val is not None:
                    val = file_val

        # --- Layer 3: CLI ---
        cli_val = getattr(cli, key, None)

        # Boolean flags with --no_* negation
        if key == "include_ground_state":
            if cli.no_ground_state:
                cli_val = False
            elif cli.include_ground_state is True:
                cli_val = True
            else:
                cli_val = None
        elif key == "allow_tf32":
            if cli.no_tf32:
                cli_val = False
            elif cli.allow_tf32 is True:
                cli_val = True
            else:
                cli_val = None
        elif key in _FLAG_NEGATION:
            neg_attr = _FLAG_NEGATION[key]
            pos_val = getattr(cli, key, None)
            neg_val = getattr(cli, neg_attr, False)
            if neg_val:
                cli_val = False
            elif pos_val is True:
                cli_val = True
            else:
                cli_val = None

        # Per-state weights from CLI need parsing
        if key in _SINGLE_STATE_WEIGHT_KEYS and cli_val is not None:
            cli_val = parse_state_weights_single(cli_val)
        if key in _PAIR_STATE_WEIGHT_KEYS and cli_val is not None:
            cli_val = parse_state_weights_pair(cli_val)

        if cli_val is not None:
            if file_val is not None and cli_val != file_val:
                overrides.append(
                    f"  CLI '{key}' overrides config value "
                    f"{file_val!r} -> {cli_val!r}"
                )
            val = cli_val

        resolved[key] = val

    if overrides:
        print("=" * 72)
        print("WARNING: CLI arguments override config file values:")
        for msg in overrides:
            print(msg)
        print("=" * 72)

    # --- Parse NACR pairs ---
    raw_pairs = resolved.get("targeted_scaled_nacr_pairs")
    nacr_pairs: Optional[List[Tuple[int, int]]] = None
    if raw_pairs is not None:
        if isinstance(raw_pairs, list) and len(raw_pairs) > 0:
            parsed: List[Tuple[int, int]] = []
            for p_str in raw_pairs:
                p_str = str(p_str).strip().strip(",")
                if not p_str:
                    continue
                if len(p_str) != 2 or not p_str.isdigit():
                    raise ValueError(
                        f"Invalid NACR pair '{p_str}'. "
                        "Expected two-digit string (e.g. 12, 23)."
                    )
                i, j = int(p_str[0]), int(p_str[1])
                if i >= j:
                    raise ValueError(
                        f"NACR pair must have i < j. Got: {p_str}."
                    )
                if i < 1:
                    raise ValueError(
                        f"NACR pair indices must be >= 1. Got: {p_str}."
                    )
                parsed.append((i, j))
            nacr_pairs = parsed if parsed else None

    # --- Parse target_state ---
    target_state = resolved.get("target_state")
    if target_state is not None:
        target_state = int(target_state)
        if target_state < 0:
            raise ValueError(
                f"TARGET_STATE_LEARNING must be >= 0, got {target_state}"
            )

    cfg = TrainingConfig(
        data_dir=str(resolved["data_dir"]),
        output_dir=str(resolved["output_dir"]),
        dataset_prefix=str(resolved["dataset_prefix"]),
        n_atoms=int(resolved["n_atoms"]),
        seed=int(resolved["seed"]),
        test_size=float(resolved["test_size"]),
        valid_size=float(resolved["valid_size"]),
        train_energies=bool(resolved["train_energies"]),
        train_forces=bool(resolved["train_forces"]),
        train_tdipoles=bool(resolved["train_tdipoles"]),
        train_scaled_nacrs=bool(resolved["train_scaled_nacrs"]),
        scaled_nacr_head=str(resolved["scaled_nacr_head"]).lower(),
        direct_nacr_n_dist=(
            None
            if resolved.get("direct_nacr_n_dist") is None
            else int(resolved["direct_nacr_n_dist"])
        ),
        direct_nacr_edge_scale=float(resolved["direct_nacr_edge_scale"]),
        direct_nacr_symmetric_coeffs=bool(
            resolved["direct_nacr_symmetric_coeffs"]
        ),
        direct_nacr_enforce_zero_sum=bool(
            resolved["direct_nacr_enforce_zero_sum"]
        ),
        direct_nacr_first_is_interacting=bool(
            resolved["direct_nacr_first_is_interacting"]
        ),
        direct_nacr_v2_mode=str(resolved["direct_nacr_v2_mode"]).lower(),
        gdv_source=str(resolved["gdv_source"]).lower(),
        gdv_file_template=str(resolved["gdv_file_template"]),
        gap_file_template=str(resolved["gap_file_template"]),
        direct_nacr_v2_edge_scale=float(
            resolved["direct_nacr_v2_edge_scale"]
        ),
        direct_nacr_v2_symmetric_coeffs=bool(
            resolved["direct_nacr_v2_symmetric_coeffs"]
        ),
        direct_nacr_v2_enforce_zero_sum=bool(
            resolved["direct_nacr_v2_enforce_zero_sum"]
        ),
        direct_nacr_v2_first_is_interacting=bool(
            resolved["direct_nacr_v2_first_is_interacting"]
        ),
        direct_nacr_v2_atom_feature_index=int(
            resolved["direct_nacr_v2_atom_feature_index"]
        ),
        direct_nacr_v2_atom_gate_hidden=(
            None
            if resolved.get("direct_nacr_v2_atom_gate_hidden") is None
            else int(resolved["direct_nacr_v2_atom_gate_hidden"])
        ),
        n_states=int(resolved["n_states"]),
        include_ground_state=bool(resolved["include_ground_state"]),
        target_state=target_state,
        targeted_scaled_nacr_pairs=nacr_pairs,
        network_type=str(resolved["network_type"]).lower(),
        possible_species=list(resolved["possible_species"]),
        n_features=int(resolved["n_features"]),
        n_sensitivities=int(resolved["n_sensitivities"]),
        dist_soft_min=float(resolved["dist_soft_min"]),
        dist_soft_max=float(resolved["dist_soft_max"]),
        dist_hard_max=float(resolved["dist_hard_max"]),
        n_interaction_layers=int(resolved["n_interaction_layers"]),
        n_atom_layers=int(resolved["n_atom_layers"]),
        l_max=int(resolved["l_max"]),
        n_max=int(resolved["n_max"]),
        batch_size=int(resolved["batch_size"]),
        eval_batch_size=int(resolved["eval_batch_size"]),
        max_batch_size=int(resolved["max_batch_size"]),
        max_epochs=int(resolved["max_epochs"]),
        learning_rate=float(resolved["learning_rate"]),
        termination_patience=int(resolved["termination_patience"]),
        scheduler_patience=int(resolved["scheduler_patience"]),
        scheduler_factor=float(resolved["scheduler_factor"]),
        stopping_key=str(resolved["stopping_key"]),
        fraction_train_eval=float(resolved["fraction_train_eval"]),
        plot_frequency=int(resolved["plot_frequency"]),
        energy_weight=float(resolved["energy_weight"]),
        force_weight=float(resolved["force_weight"]),
        tdipole_weight=float(resolved["tdipole_weight"]),
        nacr_weight=float(resolved["nacr_weight"]),
        joint_state_nacr_loss=bool(resolved.get("joint_state_nacr_loss", False)),
        l2_weight=float(resolved["l2_weight"]),
        state_energy_weights=resolved.get("state_energy_weights") or {},
        state_force_weights=resolved.get("state_force_weights") or {},
        state_tdipole_weights=resolved.get("state_tdipole_weights") or {},
        state_nacr_weights=resolved.get("state_nacr_weights") or {},
        phasesign_corrected_scaled_nacr=bool(
            resolved["phasesign_corrected_scaled_nacr"]
        ),
        phasesign_corrected_dipoles=bool(
            resolved["phasesign_corrected_dipoles"]
        ),
        force_sign=int(resolved["force_sign"]),
        allow_tf32=bool(resolved["allow_tf32"]),
        default_dtype=str(resolved.get("default_dtype", "float32")),
        restart=bool(resolved["restart"]),
        restart_dir=str(resolved.get("restart_dir") or ""),
    )
    return cfg


# =============================================================================
# Validation
# =============================================================================


def validate_config(cfg: TrainingConfig) -> None:
    """Validate the resolved configuration.  Exits on fatal errors."""
    if not any([
        cfg.train_energies,
        cfg.train_forces,
        cfg.train_tdipoles,
        cfg.train_scaled_nacrs,
    ]):
        print(
            "\nERROR: No training properties selected.  At least one of "
            "--train_energies, --train_forces, --train_tdipoles, "
            "--train_scaled_nacrs must be True."
        )
        sys.exit(1)

    if cfg.scaled_nacr_head not in (
        "charge",
        "direct_vector",
        "direct_vector_v2",
        "direct_vector_v3",
    ):
        print(f"\nERROR: Invalid scaled_nacr_head: {cfg.scaled_nacr_head}")
        print("  Expected: charge, direct_vector, direct_vector_v2, or direct_vector_v3")
        sys.exit(1)

    if cfg.use_direct_nacr_head and cfg.direct_nacr_edge_scale <= 0:
        print("\nERROR: direct_nacr_edge_scale must be positive.")
        sys.exit(1)

    if cfg.use_direct_nacr_v2_head:
        if cfg.direct_nacr_v2_mode not in V2_MODES:
            print(
                f"\nERROR: Invalid direct_nacr_v2_mode: "
                f"{cfg.direct_nacr_v2_mode}"
            )
            print(f"  Expected one of: {', '.join(V2_MODES)}")
            sys.exit(1)
        if cfg.gdv_source != "file":
            print("\nERROR: V2 currently supports only gdv_source='file'.")
            sys.exit(1)
        if not cfg.gdv_file_template:
            print("\nERROR: gdv_file_template must not be empty.")
            sys.exit(1)
        if cfg.direct_nacr_v2_edge_scale <= 0:
            print("\nERROR: direct_nacr_v2_edge_scale must be positive.")
            sys.exit(1)
        if (
            cfg.direct_nacr_v2_atom_gate_hidden is not None
            and cfg.direct_nacr_v2_atom_gate_hidden <= 0
        ):
            print(
                "\nERROR: direct_nacr_v2_atom_gate_hidden must be positive "
                "when provided."
            )
            sys.exit(1)

    if cfg.use_direct_nacr_v3_head:
        if cfg.direct_nacr_v2_mode not in V3_MODES:
            print(f"\nERROR: Invalid direct_nacr_v2_mode: {cfg.direct_nacr_v2_mode}")
            print(f"  Expected one of: {', '.join(V3_MODES)}")
            sys.exit(1)
        if cfg.gdv_source != "file":
            print("\nERROR: V3 currently supports only gdv_source='file'.")
            sys.exit(1)
        if not cfg.gdv_file_template:
            print("\nERROR: gdv_file_template must not be empty.")
            sys.exit(1)
        if not cfg.gap_file_template:
            print("\nERROR: gap_file_template must not be empty.")
            sys.exit(1)
        if cfg.direct_nacr_v2_edge_scale <= 0:
            print("\nERROR: direct_nacr_v2_edge_scale must be positive.")
            sys.exit(1)

    if not cfg.data_dir:
        print(
            "\nERROR: DATA_DIR is required.  Provide via "
            "--data_dir or config file."
        )
        sys.exit(1)
    if not os.path.isdir(cfg.data_dir):
        print(f"\nERROR: Data directory does not exist: {cfg.data_dir}")
        sys.exit(1)

    if cfg.target_state is not None:
        if not isinstance(cfg.target_state, int) or cfg.target_state < 0:
            print(
                f"\nERROR: TARGET_STATE_LEARNING must be a "
                f"non-negative integer, got {cfg.target_state}"
            )
            sys.exit(1)

    if cfg.network_type not in ("hipnn", "hiphopnn"):
        print(
            f"\nERROR: Invalid NETWORK_TYPE: '{cfg.network_type}'. "
            f"Expected 'hipnn' or 'hiphopnn'."
        )
        sys.exit(1)


# =============================================================================
# State Determination
# =============================================================================


def determine_states(
    cfg: TrainingConfig,
) -> Tuple[List[int], List[int], List[Tuple[int, int]]]:
    """Determine training_states, required_states, and nacr_pairs.

    Returns
    -------
    training_states : list[int]
        States that directly participate in training.
    required_states : list[int]
        All states whose nodes must be built (superset of training_states).
    nacr_pairs : list[tuple[int, int]]
        Pairs (i, j) with i < j for NACR nodes.
    """
    # --- training_states ---
    if cfg.target_state is not None:
        training_states = [cfg.target_state]
    else:
        state_start = 0 if cfg.include_ground_state else 1
        state_end = cfg.n_states + 1
        training_states = list(range(state_start, state_end))

    # --- nacr_pairs ---
    nacr_pairs: List[Tuple[int, int]] = []
    if cfg.build_nacr_nodes:
        if cfg.targeted_scaled_nacr_pairs is not None:
            nacr_pairs = list(cfg.targeted_scaled_nacr_pairs)
            if cfg.target_state is not None:
                ts = cfg.target_state
                filtered = [(i, j) for (i, j) in nacr_pairs if j == ts]
                if filtered != nacr_pairs:
                    print(
                        f"WARNING: TARGET_STATE_LEARNING={ts} takes "
                        f"priority over TARGETED_SCALED_NACR_PAIRS."
                    )
                    print(f"  Original pairs: {nacr_pairs}")
                    print(f"  Filtered pairs: {filtered}")
                nacr_pairs = filtered

        elif cfg.target_state is not None:
            ts = cfg.target_state
            nacr_pairs = [(i, ts) for i in range(1, ts)]

        else:
            nacr_pairs = [
                (i, j)
                for i in range(1, cfg.n_states + 1)
                for j in range(i + 1, cfg.n_states + 1)
            ]

    # --- Validate NACR config ---
    if cfg.build_nacr_nodes:
        if cfg.n_states < 2 and cfg.target_state is None:
            print("\nERROR: NACR training requires N_STATES >= 2.")
            print(
                "  N_STATES=1 provides only S1, insufficient to form "
                "any coupling pair.  Minimum NACR is dENACR12."
            )
            sys.exit(1)

        if cfg.target_state is not None and cfg.target_state < 2:
            if not cfg.targeted_scaled_nacr_pairs:
                print(
                    f"\nERROR: TARGET_STATE_LEARNING={cfg.target_state}"
                    f" with NACR training requires target_state >= 2."
                )
                sys.exit(1)

        if len(nacr_pairs) == 0:
            print("\nERROR: No valid NACR pairs could be constructed.")
            print(
                f"  N_STATES={cfg.n_states}, "
                f"TARGET_STATE_LEARNING={cfg.target_state}, "
                f"TARGETED_SCALED_NACR_PAIRS="
                f"{cfg.targeted_scaled_nacr_pairs}"
            )
            sys.exit(1)

        if (cfg.target_state is None
                and cfg.targeted_scaled_nacr_pairs is None):
            expected = cfg.n_states * (cfg.n_states - 1) // 2
            if len(nacr_pairs) != expected:
                print(
                    f"\nWARNING: Expected {expected} NACR pairs for "
                    f"N_STATES={cfg.n_states}, got {len(nacr_pairs)}."
                )

    # --- required_states ---
    req = set(training_states)
    for i, j in nacr_pairs:
        req.add(i)
        req.add(j)
    required_states = sorted(req)

    return training_states, required_states, nacr_pairs


# =============================================================================
# NACR Dataset Shape Validation
# =============================================================================


def validate_nacr_shapes(
    cfg: TrainingConfig, nacr_pairs: List[Tuple[int, int]]
) -> None:
    """Validate NACR dataset shapes: (N_FRAMES, N_ATOMS * 3)."""
    if not nacr_pairs or not cfg.data_dir:
        return
    expected_cols = cfg.n_atoms * 3
    for i, j in nacr_pairs:
        db_name = f"dENACR{i}{j}"
        fpath = os.path.join(
            cfg.data_dir, f"{cfg.dataset_prefix}{db_name}.npy"
        )
        if not os.path.isfile(fpath):
            continue
        arr = np.load(fpath)
        if arr.ndim != 2 or arr.shape[1] != expected_cols:
            print(f"\nERROR: Shape mismatch for '{db_name}'.")
            print(f"  File: {fpath}")
            print(f"  Expected: (N_FRAMES, {expected_cols})")
            print(f"  Actual:   {arr.shape}")
            sys.exit(1)
        print(f"  NACR shape validated: {db_name} = {arr.shape}  [OK]")


def gdv_db_name(i: int, j: int) -> str:
    """Return the database key for a pair-specific GDV input."""
    return f"GDV{i}{j}"


def gdv_file_path(cfg: TrainingConfig, i: int, j: int) -> str:
    """Resolve the configured GDV file path for a NACR pair."""
    pair = f"{i}{j}"
    fname = cfg.gdv_file_template.format(
        prefix=cfg.dataset_prefix,
        pair=pair,
        i=i,
        j=j,
    )
    if os.path.isabs(fname):
        return fname
    return os.path.join(cfg.data_dir, fname)


def validate_gdv_shapes(
    cfg: TrainingConfig, nacr_pairs: List[Tuple[int, int]]
) -> None:
    """Validate oracle-GDV input shapes: (N_FRAMES, N_ATOMS * 3)."""
    if not nacr_pairs or not (
        cfg.use_direct_nacr_v2_head or cfg.use_direct_nacr_v3_head
    ):
        return
    expected_cols = cfg.n_atoms * 3
    for i, j in nacr_pairs:
        db_name = gdv_db_name(i, j)
        fpath = gdv_file_path(cfg, i, j)
        if not os.path.isfile(fpath):
            print(f"\nERROR: Missing GDV input file for '{db_name}'.")
            print(f"  File: {fpath}")
            print(
                "  Generate it with diagnose_gdv_nacr.py "
                "--write_gdv_inputs or adjust GDV_FILE_TEMPLATE."
            )
            sys.exit(1)
        arr = np.load(fpath)
        if arr.ndim != 2 or arr.shape[1] != expected_cols:
            print(f"\nERROR: Shape mismatch for '{db_name}'.")
            print(f"  File: {fpath}")
            print(f"  Expected: (N_FRAMES, {expected_cols})")
            print(f"  Actual:   {arr.shape}")
            sys.exit(1)
        print(f"  GDV shape validated: {db_name} = {arr.shape}  [OK]")


def gap_db_name(i: int, j: int) -> str:
    """Return the database key for a pair-specific energy-gap input (v3)."""
    return f"GAP{i}{j}"


def gap_file_path(cfg: TrainingConfig, i: int, j: int) -> str:
    """Resolve the configured GAP file path for a NACR pair (v3)."""
    fname = cfg.gap_file_template.format(
        prefix=cfg.dataset_prefix, pair=f"{i}{j}", i=i, j=j
    )
    if os.path.isabs(fname):
        return fname
    return os.path.join(cfg.data_dir, fname)


def validate_gap_shapes(
    cfg: TrainingConfig, nacr_pairs: List[Tuple[int, int]]
) -> None:
    """Validate per-pair energy-gap input shapes: (N_FRAMES,) or (N_FRAMES, 1)."""
    if not nacr_pairs or not cfg.use_direct_nacr_v3_head:
        return
    for i, j in nacr_pairs:
        db_name = gap_db_name(i, j)
        fpath = gap_file_path(cfg, i, j)
        if not os.path.isfile(fpath):
            print(f"\nERROR: Missing GAP input file for '{db_name}'.")
            print(f"  File: {fpath}")
            print("  Generate it with gen_modelA_inputs.py (GAP{ij}=sE_j-sE_i).")
            sys.exit(1)
        arr = np.load(fpath)
        if arr.ndim not in (1, 2) or (arr.ndim == 2 and arr.shape[1] != 1):
            print(f"\nERROR: Shape mismatch for '{db_name}'.")
            print(f"  File: {fpath}")
            print(f"  Expected: (N_FRAMES,) or (N_FRAMES, 1)")
            print(f"  Actual:   {arr.shape}")
            sys.exit(1)
        print(f"  GAP shape validated: {db_name} = {arr.shape}  [OK]")


# =============================================================================
# Output Directory Name Generation
# =============================================================================


def generate_output_dirname(
    cfg: TrainingConfig,
    nacr_pairs: List[Tuple[int, int]],
) -> str:
    """Generate a descriptive output directory name from active flags."""
    parts = [cfg.active_flags_tag, "TRAINING_MODEL"]
    if cfg.target_state is not None:
        parts.append(f"TARGET_S{cfg.target_state}")
    if cfg.targeted_scaled_nacr_pairs and nacr_pairs:
        pair_strs = [f"{i}{j}" for i, j in nacr_pairs]
        parts.append("PAIRS_" + "_".join(pair_strs))
    if cfg.train_scaled_nacrs:
        parts.append(f"NHEAD_{cfg.scaled_nacr_head.upper()}")
        if cfg.use_direct_nacr_v2_head:
            parts.append(f"V2_{cfg.direct_nacr_v2_mode.upper()}")
        if cfg.use_direct_nacr_v3_head:
            parts.append(f"V3_{cfg.direct_nacr_v2_mode.upper()}")
    return "_".join(parts)


# =============================================================================
# DataDumper (CSV Export Plotter)
# =============================================================================


class DataDumper(plotting.plotters.Plotter):
    """Custom plotter: exports prediction-vs-target data as CSV."""

    def __init__(self, node: base.Node, saved: str = "data.csv"):
        if hasattr(node, "main_output"):
            node = node.main_output
        self.node_db_name = getattr(node, "db_name", "unknown")
        reduced_true, reduced_pred = elementwise_compare_reduce(
            node.true, node.pred
        )
        super().__init__(
            (reduced_pred, reduced_true),
            plt_fn=None,
            saved=saved,
            shown=False,
        )

    def make_plot(
        self, data_args: Tuple[torch.Tensor, torch.Tensor]
    ) -> plt.Figure:
        pred, true = data_args
        fig = plt.figure()

        def custom_saver(fname: str, **kwargs: Any) -> None:
            base_name, _ = os.path.splitext(fname)
            csv_fname = (
                base_name + ".csv"
                if not base_name.endswith(".csv")
                else fname
            )
            try:
                pred_np = pred.detach().cpu().numpy().flatten()
                true_np = true.detach().cpu().numpy().flatten()
                if np.isnan(pred_np).any():
                    print(f"WARNING: NaNs in predictions for {csv_fname}")
                df = pd.DataFrame(
                    {"Predicted": pred_np, "Actual": true_np}
                )
                print(f"Exporting CSV data to {csv_fname}")
                df.to_csv(csv_fname, index=False)
            except Exception as e:
                print(f"Error exporting CSV for {self.node_db_name}: {e}")

        fig.savefig = custom_saver
        return fig


# =============================================================================
# Network Builder
# =============================================================================


def build_network(
    species: inputs.SpeciesNode,
    positions: inputs.PositionsNode,
    network_params: Dict[str, Any],
    choice: int = 1,
    l_max: int = 3,
    n_max: int = 4,
) -> networks.NeuralNetworkNode:
    """Construct a HIPNN (choice=0) or HIPHopNN (choice=1) network."""
    params = copy.deepcopy(network_params or {})
    if choice not in (0, 1):
        raise ValueError(f"Unsupported network choice: {choice!r}")

    def _coerce(key, typ, default, min_v=0):
        val = typ(params.get(key, default))
        if val < min_v:
            raise ValueError(f"'{key}' must be >= {min_v}, got {val}")
        params[key] = val
        return val

    _coerce("n_features", int, 16, 1)
    _coerce("n_sensitivities", int, 1, 1)
    _coerce("n_interaction_layers", int, 1, 0)
    _coerce("n_atom_layers", int, 1, 0)
    _coerce("dist_soft_min", float, 0.5, 0.0)
    _coerce("dist_soft_max", float, 3.0, 0.0)
    _coerce("dist_hard_max", float, 5.0, 0.0)

    try:
        if choice == 0:
            return networks.Hipnn(
                "hipnn_model", (species, positions),
                module_kwargs=params,
            )
        hp = copy.deepcopy(params)
        hp["l_max"] = int(l_max) if l_max is not None else 0
        hp["n_max"] = int(n_max) if n_max is not None else 0
        return networks.HipHopnn(
            "hiphopnn_model", (species, positions),
            module_kwargs=hp,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Network build failed (choice={choice}): {exc}"
        ) from exc


# =============================================================================
# Graph Node Builder
# =============================================================================


def build_graph_nodes(
    cfg: TrainingConfig,
    network: networks.NeuralNetworkNode,
    species: inputs.SpeciesNode,
    positions: inputs.PositionsNode,
    required_states: List[int],
    nacr_pairs: List[Tuple[int, int]],
) -> Dict[str, Any]:
    """Build all required HIPPYNN graph nodes.

    Which categories are created is determined by ``cfg.build_*``
    properties (derived from the four training flags + dependencies).
    """
    energy_nodes: List[targets.HEnergyNode] = []
    energy_outputs: List[base.Node] = []
    force_nodes: List[physics.GradientNode] = []
    force_outputs: List[base.Node] = []
    charge_nodes: List[targets.HChargeNode] = []
    dipole_nodes: List[physics.DipoleNode] = []
    nacr_nodes: List[base.Node] = []
    gdv_input_nodes: Dict[Tuple[int, int], inputs.InputNode] = {}
    state_index_map: Dict[int, int] = {}

    for list_idx, state_idx in enumerate(required_states):
        state_index_map[state_idx] = list_idx

        if cfg.build_energy_nodes:
            e_name = f"sE{state_idx}"
            energy = targets.HEnergyNode(
                e_name, network, module_kwargs=None
            )
            energy.mol_energy.db_name = e_name
            energy_nodes.append(energy)
            energy_outputs.append(energy.mol_energy)
            print(f"  Created energy node: {e_name}")

        if cfg.build_force_nodes:
            f_name = f"F{state_idx}"
            force = physics.GradientNode(
                f_name,
                (energy_nodes[list_idx], positions),
                sign=cfg.force_sign,
                db_name=f_name,
            )
            force_nodes.append(force)
            force_outputs.append(force)
            print(
                f"  Created force node: {f_name} "
                f"(autodiff, sign={cfg.force_sign})"
            )

        if cfg.build_charge_nodes:
            q_name = f"Q{state_idx}"
            charge = targets.HChargeNode(
                q_name, network, module_kwargs=None
            )
            charge_nodes.append(charge)
            print(f"  Created charge node: {q_name}")

            if cfg.build_dipole_nodes:
                d_name = f"D{state_idx}"
                dipole = physics.DipoleNode(
                    d_name, (charge, positions), db_name=d_name
                )
                dipole_nodes.append(dipole)
                print(f"  Created dipole node: {d_name}")

    if cfg.build_nacr_nodes and nacr_pairs:
        if cfg.scaled_nacr_head == "charge":
            for i, j in nacr_pairs:
                idx_i = state_index_map[i]
                idx_j = state_index_map[j]
                nacr_name = f"dENACR{i}{j}"
                nacr_node = NACRNode(
                    nacr_name,
                    (
                        charge_nodes[idx_i],
                        charge_nodes[idx_j],
                        positions,
                        energy_nodes[idx_i],
                        energy_nodes[idx_j],
                    ),
                    db_name=nacr_name,
                )
                nacr_nodes.append(nacr_node)
                print(
                    f"  Created charge-based NACR node: "
                    f"{nacr_name} (S{i}-S{j})"
                )
        elif cfg.scaled_nacr_head == "direct_vector":
            for i, j in nacr_pairs:
                nacr_name = f"dENACR{i}{j}"
                nacr_node = DirectScaledNACRNode(
                    nacr_name,
                    (network, positions),
                    db_name=nacr_name,
                    module_kwargs={
                        "n_target": 1,
                        "n_dist": (
                            cfg.direct_nacr_n_dist
                            or cfg.n_sensitivities
                        ),
                        "dist_soft_min": cfg.dist_soft_min,
                        "dist_soft_max": cfg.dist_soft_max,
                        "dist_hard_max": cfg.dist_hard_max,
                        "edge_scale": cfg.direct_nacr_edge_scale,
                        "symmetric_coeffs": (
                            cfg.direct_nacr_symmetric_coeffs
                        ),
                        "enforce_zero_sum": (
                            cfg.direct_nacr_enforce_zero_sum
                        ),
                        "first_is_interacting": (
                            cfg.direct_nacr_first_is_interacting
                        ),
                    },
                )
                nacr_nodes.append(nacr_node)
                print(
                    f"  Created DIRECT equivariant scaled-NACR node: "
                    f"{nacr_name} (S{i}-S{j})"
                )
        elif cfg.scaled_nacr_head == "direct_vector_v2":
            for i, j in nacr_pairs:
                nacr_name = f"dENACR{i}{j}"
                gdv_name = gdv_db_name(i, j)
                gdv_input = inputs.InputNode(
                    gdv_name,
                    db_name=gdv_name,
                    index_state=IdxType.Systems,
                )
                gdv_input_nodes[(i, j)] = gdv_input
                nacr_node = DirectScaledNACRV2Node(
                    nacr_name,
                    (network, positions, gdv_input),
                    db_name=nacr_name,
                    module_kwargs={
                        "mode": cfg.direct_nacr_v2_mode,
                        "n_target": 1,
                        "n_dist": (
                            cfg.direct_nacr_n_dist
                            or cfg.n_sensitivities
                        ),
                        "dist_soft_min": cfg.dist_soft_min,
                        "dist_soft_max": cfg.dist_soft_max,
                        "dist_hard_max": cfg.dist_hard_max,
                        "edge_scale": cfg.direct_nacr_v2_edge_scale,
                        "symmetric_coeffs": (
                            cfg.direct_nacr_v2_symmetric_coeffs
                        ),
                        "enforce_zero_sum": (
                            cfg.direct_nacr_v2_enforce_zero_sum
                        ),
                        "first_is_interacting": (
                            cfg.direct_nacr_v2_first_is_interacting
                        ),
                        "atom_feature_index": (
                            cfg.direct_nacr_v2_atom_feature_index
                        ),
                        "atom_gate_hidden": (
                            cfg.direct_nacr_v2_atom_gate_hidden
                        ),
                    },
                )
                nacr_nodes.append(nacr_node)
                print(
                    f"  Created V2 GDV-aware scaled-NACR node: "
                    f"{nacr_name} (S{i}-S{j}), GDV input {gdv_name}, "
                    f"mode={cfg.direct_nacr_v2_mode}"
                )
        elif cfg.scaled_nacr_head == "direct_vector_v3":
            for i, j in nacr_pairs:
                nacr_name = f"dENACR{i}{j}"
                gdv_name = gdv_db_name(i, j)
                gap_name = gap_db_name(i, j)
                gdv_input = inputs.InputNode(
                    gdv_name, db_name=gdv_name, index_state=IdxType.Systems
                )
                gap_input = inputs.InputNode(
                    gap_name, db_name=gap_name, index_state=IdxType.Systems
                )
                gdv_input_nodes[(i, j)] = gdv_input
                nacr_node = DirectScaledNACRV3Node(
                    nacr_name,
                    (network, positions, gdv_input, gap_input),
                    db_name=nacr_name,
                    module_kwargs={
                        "mode": cfg.direct_nacr_v2_mode,
                        "n_target": 1,
                        "n_dist": (
                            cfg.direct_nacr_n_dist or cfg.n_sensitivities
                        ),
                        "dist_soft_min": cfg.dist_soft_min,
                        "dist_soft_max": cfg.dist_soft_max,
                        "dist_hard_max": cfg.dist_hard_max,
                        "edge_scale": cfg.direct_nacr_v2_edge_scale,
                        "symmetric_coeffs": (
                            cfg.direct_nacr_v2_symmetric_coeffs
                        ),
                        "enforce_zero_sum": (
                            cfg.direct_nacr_v2_enforce_zero_sum
                        ),
                        "first_is_interacting": (
                            cfg.direct_nacr_v2_first_is_interacting
                        ),
                        "atom_feature_index": (
                            cfg.direct_nacr_v2_atom_feature_index
                        ),
                        "atom_gate_hidden": (
                            cfg.direct_nacr_v2_atom_gate_hidden
                        ),
                    },
                )
                nacr_nodes.append(nacr_node)
                print(
                    f"  Created V3 GDV+gap-aware scaled-NACR node: "
                    f"{nacr_name} (S{i}-S{j}), GDV input {gdv_name}, "
                    f"GAP input {gap_name}, mode={cfg.direct_nacr_v2_mode}"
                )
        else:
            raise ValueError(
                f"Unsupported scaled_nacr_head: {cfg.scaled_nacr_head}"
            )

    return {
        "energy_nodes": energy_nodes,
        "energy_outputs": energy_outputs,
        "force_nodes": force_nodes,
        "force_outputs": force_outputs,
        "charge_nodes": charge_nodes,
        "dipole_nodes": dipole_nodes,
        "nacr_nodes": nacr_nodes,
        "gdv_input_nodes": gdv_input_nodes,
        "state_index_map": state_index_map,
    }


# =============================================================================
# Loss Function Builder
# =============================================================================


def build_loss(
    cfg: TrainingConfig,
    nodes: Dict[str, Any],
    network: networks.NeuralNetworkNode,
    required_states: List[int],
    nacr_pairs: List[Tuple[int, int]],
) -> Tuple[Any, Dict[str, Any]]:
    """Construct total_loss and validation_losses.

    For every *built* node category, RMSE/MAE metrics are always added
    to ``validation_losses`` for monitoring.  A term accumulates into
    ``total_loss`` **only** when the corresponding ``train_*`` flag is True.
    """
    validation_losses: Dict[str, Any] = {}
    total_loss: Optional[Any] = None

    def accum(term: Any, weight: float) -> None:
        nonlocal total_loss
        if total_loss is None:
            total_loss = weight * term
        else:
            total_loss = total_loss + weight * term

    # ---- Block A: Energy ----
    if cfg.build_energy_nodes:
        for li, si in enumerate(required_states):
            name = f"sE{si}"
            out = nodes["energy_outputs"][li]
            rmse = loss.MSELoss.of_node(out) ** 0.5
            mae = loss.MAELoss.of_node(out)
            sl = rmse + mae
            validation_losses[f"{name}-RMSE"] = rmse
            validation_losses[f"{name}-MAE"] = mae
            validation_losses[f"{name}-Loss"] = sl

            if cfg.train_energies:
                sw = cfg.state_energy_weights.get(si, 1.0)
                ew = cfg.energy_weight * sw
                accum(sl, ew)
                print(
                    f"  Energy loss: {name}  "
                    f"(global={cfg.energy_weight}, "
                    f"state={sw}, effective={ew})"
                )
            else:
                print(f"  Energy monitored (not in loss): {name}")

    # ---- Block B: Force ----
    if cfg.build_force_nodes:
        for li, si in enumerate(required_states):
            name = f"F{si}"
            out = nodes["force_outputs"][li]
            rmse = loss.MSELoss.of_node(out) ** 0.5
            mae = loss.MAELoss.of_node(out)
            sl = rmse + mae
            validation_losses[f"{name}-RMSE"] = rmse
            validation_losses[f"{name}-MAE"] = mae
            validation_losses[f"{name}-Loss"] = sl

            if cfg.train_forces:
                sw = cfg.state_force_weights.get(si, 1.0)
                ew = cfg.force_weight * sw
                accum(sl, ew)
                print(
                    f"  Force loss:  {name}  "
                    f"(global={cfg.force_weight}, "
                    f"state={sw}, effective={ew})"
                )
            else:
                print(f"  Force monitored (not in loss): {name}")

    # ---- Block C: Dipole ----
    if cfg.build_dipole_nodes:
        use_std_dipole = cfg.phasesign_corrected_dipoles
        for li, si in enumerate(required_states):
            name = f"D{si}"
            dnode = nodes["dipole_nodes"][li]

            # Always compute BOTH standard and phase-less metrics
            std_rmse = loss.MSELoss.of_node(dnode) ** 0.5
            std_mae = loss.MAELoss.of_node(dnode)
            phl_rmse = MSEPhaseLoss.of_node(dnode) ** 0.5
            phl_mae = MAEPhaseLoss.of_node(dnode)

            # Track both in validation_losses for comparison
            validation_losses[f"{name}-StdRMSE"] = std_rmse
            validation_losses[f"{name}-StdMAE"] = std_mae
            validation_losses[f"{name}-PhLessRMSE"] = phl_rmse
            validation_losses[f"{name}-PhLessMAE"] = phl_mae

            # Select which enters the loss (same normalization either way)
            if use_std_dipole:
                rmse, mae = std_rmse, std_mae
                loss_tag = "standard MSE/MAE (phase-corrected data)"
            else:
                rmse, mae = phl_rmse, phl_mae
                loss_tag = "phase-less MSE/MAE"

            tl = (rmse / np.sqrt(3)) + mae
            validation_losses[f"{name}-RMSE"] = rmse
            validation_losses[f"{name}-MAE"] = mae
            validation_losses[f"{name}-Loss"] = tl

            # Dipole magnitude (always standard — magnitude is sign-invariant)
            true_mag = physics.VecMag(f"TrueMag_{name}", dnode.true)
            pred_mag = physics.VecMag(f"PredMag_{name}", dnode.pred)
            mag_rmse = loss.MSELoss(true_mag, pred_mag) ** 0.5
            mag_mae = loss.MAELoss(true_mag, pred_mag)
            validation_losses[f"Mag_{name}-RMSE"] = mag_rmse
            validation_losses[f"Mag_{name}-MAE"] = mag_mae

            if cfg.train_tdipoles:
                sw = cfg.state_tdipole_weights.get(si, 1.0)
                ew = cfg.tdipole_weight * sw
                accum(tl, ew)
                accum(mag_rmse + mag_mae, ew)
                print(
                    f"  Dipole loss: {name}  "
                    f"(global={cfg.tdipole_weight}, "
                    f"state={sw}, effective={ew}, "
                    f"{loss_tag})"
                )
            else:
                print(f"  Dipole monitored (not in loss): {name}")

    # ---- Block D: NACR ----
    if cfg.build_nacr_nodes and nodes["nacr_nodes"]:
        use_std_nacr = cfg.phasesign_corrected_scaled_nacr
        # Tier-1 sign fix: joint per-state phase loss (min over s_i of ||d_ij^ML - s_i s_j d_ij^QC||^2).
        use_joint = bool(getattr(cfg, "joint_state_nacr_loss", False))
        if use_joint and cfg.train_scaled_nacrs:
            from joint_state_nacr_loss import make_joint_loss_node
            _jl = make_joint_loss_node(nodes["nacr_nodes"], nacr_pairs)
            validation_losses["dENACR-JointStatePhase"] = _jl
            accum(_jl, cfg.nacr_weight)
            print(f"  NACR loss:   JOINT per-state phase  (global={cfg.nacr_weight}, "
                  f"pairs={nacr_pairs}, per-pair phaseless kept as diagnostics)")
        for nnode, (i, j) in zip(nodes["nacr_nodes"], nacr_pairs):
            # Always compute BOTH standard and phase-less metrics
            std_rmse = loss.MSELoss.of_node(nnode) ** 0.5
            std_mae = loss.MAELoss.of_node(nnode)
            phl_rmse = MSEPhaseLoss.of_node(nnode) ** 0.5
            phl_mae = MAEPhaseLoss.of_node(nnode)

            # Track both in validation_losses for comparison
            validation_losses[f"{nnode.name}-StdRMSE"] = std_rmse
            validation_losses[f"{nnode.name}-StdMAE"] = std_mae
            validation_losses[f"{nnode.name}-PhLessRMSE"] = phl_rmse
            validation_losses[f"{nnode.name}-PhLessMAE"] = phl_mae

            # Select which enters the loss (same normalization either way)
            if use_std_nacr:
                rmse, mae = std_rmse, std_mae
                loss_tag = "standard MSE/MAE (phase-corrected data)"
            else:
                rmse, mae = phl_rmse, phl_mae
                loss_tag = "phase-less MSE/MAE"

            tl = (rmse / np.sqrt(3 * cfg.n_atoms)) + mae
            validation_losses[f"{nnode.name}-RMSE"] = rmse
            validation_losses[f"{nnode.name}-MAE"] = mae
            validation_losses[f"{nnode.name}-Loss"] = tl

            if cfg.train_scaled_nacrs and not use_joint:
                pw = cfg.state_nacr_weights.get((i, j), 1.0)
                ew = cfg.nacr_weight * pw
                accum(tl, ew)
                print(
                    f"  NACR loss:   {nnode.name}  "
                    f"(global={cfg.nacr_weight}, "
                    f"pair={pw}, effective={ew}, "
                    f"{loss_tag})"
                )
            else:
                print(
                    f"  NACR monitored (not in loss): {nnode.name}"
                )

    # ---- Guard ----
    if total_loss is None:
        raise RuntimeError(
            "No loss terms accumulated.  At least one of "
            "--train_energies, --train_forces, --train_tdipoles, "
            "--train_scaled_nacrs must be True and produce valid "
            "loss terms for the selected states."
        )

    # ---- Block E: L2 (always) ----
    l2_reg = loss.l2reg(network)
    validation_losses["L2"] = l2_reg
    validation_losses["Loss_wo_L2"] = total_loss
    final_loss = total_loss + cfg.l2_weight * l2_reg
    validation_losses["Loss"] = final_loss

    return final_loss, validation_losses


# =============================================================================
# Plotter Builder
# =============================================================================


def build_plotters(
    cfg: TrainingConfig,
    nodes: Dict[str, Any],
    network: networks.NeuralNetworkNode,
    required_states: List[int],
    nacr_pairs: List[Tuple[int, int]],
) -> plotting.PlotMaker:
    """Build PlotMaker with comparison plots and CSV exporters."""
    pl: List[Any] = []

    def add(node, prefix):
        pl.append(plotting.Hist2D.compare(
            node, saved=f"{prefix}.pdf", shown=False))
        pl.append(DataDumper(node, saved=f"{prefix}.csv"))

    if cfg.build_energy_nodes:
        for li, si in enumerate(required_states):
            add(nodes["energy_outputs"][li], f"Energy_sE{si}")

    if cfg.build_force_nodes:
        for li, si in enumerate(required_states):
            add(nodes["force_nodes"][li], f"Force_F{si}")

    if cfg.build_dipole_nodes:
        for li, si in enumerate(required_states):
            dn = nodes["dipole_nodes"][li]
            add(dn, f"Dipole_D{si}")
            tm = physics.VecMag(f"PlotTrueMag_D{si}", dn.true)
            pm = physics.VecMag(f"PlotPredMag_D{si}", dn.pred)
            pl.append(plotting.Hist2D(
                tm, pm,
                xlabel=f"True |D{si}|", ylabel=f"Pred |D{si}|",
                saved=f"DipoleMag_D{si}.pdf", shown=False,
            ))

    if cfg.build_nacr_nodes:
        for nn in nodes["nacr_nodes"]:
            add(nn, f"NACR_{nn.name}")

    for i in range(cfg.n_interaction_layers):
        pl.append(plotting.SensitivityPlot(
            network.torch_module.sensitivity_layers[i],
            saved=f"Sensitivity_{i}.pdf", shown=False,
        ))

    return plotting.PlotMaker(*pl, plot_every=cfg.plot_frequency)


# =============================================================================
# Pre-Training Summary
# =============================================================================


def _fmt_weights(label, gw, sw, keys, pfx="S"):
    items = []
    for k in keys:
        w = sw.get(k, 1.0)
        eff = gw * w
        ks = f"{k[0]}{k[1]}" if isinstance(k, tuple) else f"{pfx}{k}"
        items.append(f"{ks}: {gw}x{w}={eff:.4g}")
    if items:
        return [f"  Per-state {label} (effective): {', '.join(items)}"]
    return []


def print_pretraining_summary(
    cfg: TrainingConfig,
    training_states: List[int],
    required_states: List[int],
    nacr_pairs: List[Tuple[int, int]],
    output_dirname: str,
    cli_source: Optional[str],
) -> None:
    """Print comprehensive pre-training summary."""
    sep = "=" * 72
    print(f"\n{sep}")
    print("UNIFIED HIPPYNN TRAINING SCRIPT")
    print(sep)

    print("\n[Configuration Source]")
    print(f"  Config file: {cli_source or '(none)'}")
    print("  Precedence: CLI > config file > defaults")

    print("\n[Property Flags]")
    in_loss = []
    if cfg.train_energies:     in_loss.append("Energies")
    if cfg.train_forces:       in_loss.append("Forces")
    if cfg.train_tdipoles:     in_loss.append("Dipoles")
    if cfg.train_scaled_nacrs: in_loss.append("Scaled NACRs")
    print(f"  In loss:     {', '.join(in_loss)}")

    built_only = []
    if cfg.build_energy_nodes and not cfg.train_energies:
        built_only.append("Energy (structural dep.)")
    if cfg.build_charge_nodes and not cfg.train_tdipoles:
        built_only.append("Charge (structural dep.)")
    if built_only:
        print(f"  Built for dependencies (monitored): "
              f"{', '.join(built_only)}")
    print(f"  Active tag:  {cfg.active_flags_tag}")

    if cfg.train_scaled_nacrs:
        print("\n[NACR Architecture]")
        print(f"  Scaled NACR head: {cfg.scaled_nacr_head}")
        if cfg.use_direct_nacr_head:
            print("  Direct vector mode:")
            print("    Uses local direct equivariant vector head.")
            print("    Does not build charges for NACR.")
            print("    Does not build energies for NACR.")
            print("    Does not multiply by energy gaps.")
            print("    Target files are already scaled dENACRs.")
            print(
                f"    n_dist: "
                f"{cfg.direct_nacr_n_dist or cfg.n_sensitivities}"
            )
            print(f"    edge_scale: {cfg.direct_nacr_edge_scale}")
            print(
                f"    symmetric_coeffs: "
                f"{cfg.direct_nacr_symmetric_coeffs}"
            )
            print(
                f"    enforce_zero_sum: "
                f"{cfg.direct_nacr_enforce_zero_sum}"
            )
            print(
                f"    first_is_interacting: "
                f"{cfg.direct_nacr_first_is_interacting}"
            )
        if cfg.use_direct_nacr_v2_head:
            print("  Direct vector V2 mode:")
            print("    Uses local oracle-GDV-aware vector head.")
            print("    GDV is a reference input, not a loss target.")
            print("    Loss remains against already scaled dENACRs.")
            print("    Does not build charges for NACR.")
            print("    Does not build energies for NACR.")
            print("    Not yet deployable without reference or ML forces.")
            print(f"    v2_mode: {cfg.direct_nacr_v2_mode}")
            print(f"    gdv_source: {cfg.gdv_source}")
            print(f"    gdv_file_template: {cfg.gdv_file_template}")
            print(
                f"    n_dist: "
                f"{cfg.direct_nacr_n_dist or cfg.n_sensitivities}"
            )
            print(f"    edge_scale: {cfg.direct_nacr_v2_edge_scale}")
            print(
                f"    symmetric_coeffs: "
                f"{cfg.direct_nacr_v2_symmetric_coeffs}"
            )
            print(
                f"    enforce_zero_sum: "
                f"{cfg.direct_nacr_v2_enforce_zero_sum}"
            )
            print(
                f"    first_is_interacting: "
                f"{cfg.direct_nacr_v2_first_is_interacting}"
            )

    # --- Phase-Sign Correction ---
    if cfg.phasesign_corrected_dipoles or cfg.phasesign_corrected_scaled_nacr:
        print("\n[Phase-Sign Correction]")
        if cfg.phasesign_corrected_dipoles:
            print(
                "  Dipoles:     PHASE-SIGN CORRECTED dataset"
            )
            print(
                "               Training loss uses standard MSE/MAE."
            )
            print(
                "               Phase-less MSE/MAE tracked as "
                "diagnostic (PhLess* keys)."
            )
        else:
            if cfg.build_dipole_nodes:
                print(
                    "  Dipoles:     NOT phase-corrected "
                    "(using phase-less MSE/MAE in loss)"
                )
        if cfg.phasesign_corrected_scaled_nacr:
            print(
                "  Scaled NACR: PHASE-SIGN CORRECTED dataset"
            )
            print(
                "               Training loss uses standard MSE/MAE."
            )
            print(
                "               Phase-less MSE/MAE tracked as "
                "diagnostic (PhLess* keys)."
            )
        else:
            if cfg.build_nacr_nodes:
                print(
                    "  Scaled NACR: NOT phase-corrected "
                    "(using phase-less MSE/MAE in loss)"
                )

        print()
        print(
            "  ---- Phase-Sign Diagnostic Guide ----"
        )
        print(
            "  Both standard and phase-less MAE are tracked for every "
            "dipole/NACR node."
        )
        print(
            "  Compare them on the validation set:"
        )
        print()
        print(
            "    delta = |MAE_standard - MAE_phaseless| / MAE_standard"
        )
        print()
        print(
            "  If delta < 1%:"
        )
        print(
            "    Phase correction is consistent. The model is reliably "
            "predicting the correct sign."
        )
        print(
            "  If delta is large:"
        )
        print(
            "    The model frequently flips signs. Possible causes:"
        )
        print(
            "    (a) Residual phase inconsistencies in the training data "
            "for that pair/state."
        )
        print(
            "    (b) Regions near conical intersections where the "
            "scaled NACR changes sign"
        )
        print(
            "        rapidly and the model cannot resolve it."
        )
        print(
            "  Relevant validation keys (per node):"
        )
        print(
            "    <node>-StdMAE   = standard MAE"
        )
        print(
            "    <node>-PhLessMAE = phase-less MAE"
        )
        print(
            "  ------------------------------------------"
        )

    print("\n[State Selection]")
    if cfg.target_state is not None:
        print(f"  Mode: TARGET_STATE_LEARNING = {cfg.target_state}")
        print(f"  Training ONLY state S{cfg.target_state}")
    else:
        print("  Mode: Standard")
        if cfg.include_ground_state:
            print(f"  Training: S0 through S{cfg.n_states}")
        else:
            print(f"  Training: S1 through S{cfg.n_states} (no ground)")
    print(f"  Training states: {['S'+str(s) for s in training_states]}")
    print(f"  Required states: {['S'+str(s) for s in required_states]}")
    if nacr_pairs:
        print(f"  NACR pairs: {[f'dENACR{i}{j}' for i,j in nacr_pairs]}")
    else:
        print("  NACR pairs: (none)")

    print("\n[Dataset]")
    print(f"  Data directory:   {cfg.data_dir}")
    print(f"  Dataset prefix:   {cfg.dataset_prefix}")
    print(f"  N_ATOMS:          {cfg.n_atoms}")
    print(f"  Seed:             {cfg.seed}")
    print(f"  Test/Valid split:  {cfg.test_size}/{cfg.valid_size}")
    print("  Required files:")
    print(f"    Species:   {cfg.dataset_prefix}Z.npy")
    print(f"    Positions: {cfg.dataset_prefix}R.npy")
    if cfg.build_energy_nodes:
        fs = [f"{cfg.dataset_prefix}sE{s}.npy" for s in required_states]
        print(f"    Energies:  {', '.join(fs)}")
    if cfg.build_force_nodes:
        fs = [f"{cfg.dataset_prefix}F{s}.npy" for s in required_states]
        print(f"    Forces:    {', '.join(fs)}")
    if cfg.build_dipole_nodes:
        fs = [f"{cfg.dataset_prefix}D{s}.npy" for s in required_states]
        print(f"    Dipoles:   {', '.join(fs)}")
    if cfg.build_nacr_nodes and nacr_pairs:
        fs = [f"{cfg.dataset_prefix}dENACR{i}{j}.npy"
              for i,j in nacr_pairs]
        print(f"    NACRs:     {', '.join(fs)}")
    if cfg.use_direct_nacr_v2_head and cfg.gdv_source == "file":
        fs = [os.path.basename(gdv_file_path(cfg, i, j))
              for i, j in nacr_pairs]
        print(f"    GDV inputs:{', '.join(fs)}")

    print("\n[Network]")
    nt = "HIPHopNN" if cfg.network_type == "hiphopnn" else "HIPNN"
    print(f"  Type: {nt}")
    print(f"  {json.dumps(cfg.network_params, indent=4)}")

    print("\n[Training Hyperparameters]")
    print(f"  Batch size:           {cfg.batch_size}")
    print(f"  Eval batch size:      {cfg.eval_batch_size}")
    print(f"  Max batch size:       {cfg.max_batch_size}")
    print(f"  Max epochs:           {cfg.max_epochs}")
    print(f"  Learning rate:        {cfg.learning_rate}")
    print(f"  Termination patience: {cfg.termination_patience}")
    print(f"  Scheduler patience:   {cfg.scheduler_patience}")
    print(f"  Scheduler factor:     {cfg.scheduler_factor}")
    print(f"  Stopping key:         {cfg.stopping_key}")
    print(f"  Fraction train eval:  {cfg.fraction_train_eval}")
    print(f"  Plot frequency:       {cfg.plot_frequency}")

    print("\n[Loss Weights]")
    print(f"  Global:  Energy={cfg.energy_weight}, "
          f"Force={cfg.force_weight}, "
          f"Dipole={cfg.tdipole_weight}, "
          f"NACR={cfg.nacr_weight}, L2={cfg.l2_weight}")
    if cfg.state_energy_weights:
        for l in _fmt_weights("energy", cfg.energy_weight,
                              cfg.state_energy_weights, required_states):
            print(l)
    if cfg.state_force_weights:
        for l in _fmt_weights("force", cfg.force_weight,
                              cfg.state_force_weights, required_states):
            print(l)
    if cfg.state_tdipole_weights:
        for l in _fmt_weights("dipole", cfg.tdipole_weight,
                              cfg.state_tdipole_weights, required_states):
            print(l)
    if cfg.state_nacr_weights and nacr_pairs:
        for l in _fmt_weights("NACR", cfg.nacr_weight,
                              cfg.state_nacr_weights, nacr_pairs):
            print(l)

    print("\n[Physics]")
    print(f"  Force sign: {cfg.force_sign}  (F = {cfg.force_sign}*dE/dR)")

    print("\n[Torch Defaults]")
    print(f"  allow_tf32:     {cfg.allow_tf32}")
    print(f"  default_dtype:  {cfg.default_dtype}")

    print("\n[Output]")
    print(f"  Directory:  {output_dirname}/")
    print(f"  Summary:    training_summary.json")
    print(f"  Log:        training_log.txt")
    print(sep)
    print()


# =============================================================================
# Training Summary Saver
# =============================================================================


def save_training_summary(
    cfg: TrainingConfig,
    metric_tracker: Any,
    output_dirname: str,
) -> None:
    """Save training metrics and configuration to JSON."""
    summary = copy.deepcopy(cfg.network_params)
    if "possible_species" in summary:
        del summary["possible_species"]

    summary["active_flags"] = cfg.active_flags_tag
    summary["train_energies"] = cfg.train_energies
    summary["train_forces"] = cfg.train_forces
    summary["train_tdipoles"] = cfg.train_tdipoles
    summary["train_scaled_nacrs"] = cfg.train_scaled_nacrs
    summary["scaled_nacr_head"] = cfg.scaled_nacr_head
    summary["direct_nacr_n_dist"] = cfg.direct_nacr_n_dist
    summary["direct_nacr_edge_scale"] = cfg.direct_nacr_edge_scale
    summary["direct_nacr_symmetric_coeffs"] = (
        cfg.direct_nacr_symmetric_coeffs
    )
    summary["direct_nacr_enforce_zero_sum"] = (
        cfg.direct_nacr_enforce_zero_sum
    )
    summary["direct_nacr_first_is_interacting"] = (
        cfg.direct_nacr_first_is_interacting
    )
    summary["direct_nacr_v2_mode"] = cfg.direct_nacr_v2_mode
    summary["gdv_source"] = cfg.gdv_source
    summary["gdv_file_template"] = cfg.gdv_file_template
    summary["gap_file_template"] = cfg.gap_file_template
    summary["direct_nacr_v2_edge_scale"] = (
        cfg.direct_nacr_v2_edge_scale
    )
    summary["direct_nacr_v2_symmetric_coeffs"] = (
        cfg.direct_nacr_v2_symmetric_coeffs
    )
    summary["direct_nacr_v2_enforce_zero_sum"] = (
        cfg.direct_nacr_v2_enforce_zero_sum
    )
    summary["direct_nacr_v2_first_is_interacting"] = (
        cfg.direct_nacr_v2_first_is_interacting
    )
    summary["direct_nacr_v2_atom_feature_index"] = (
        cfg.direct_nacr_v2_atom_feature_index
    )
    summary["direct_nacr_v2_atom_gate_hidden"] = (
        cfg.direct_nacr_v2_atom_gate_hidden
    )
    summary["phasesign_corrected_scaled_nacr"] = (
        cfg.phasesign_corrected_scaled_nacr
    )
    summary["phasesign_corrected_dipoles"] = (
        cfg.phasesign_corrected_dipoles
    )
    summary["metric"] = metric_tracker.best_metric_values
    summary["avg_epoch_time"] = float(
        np.average(metric_tracker.epoch_times)
    )
    summary["Loss"] = metric_tracker.best_metric_values["valid"]["Loss"]

    with open("training_summary.json", "w") as out:
        json.dump(summary, out, indent=4)

    print(f"\n{'=' * 72}")
    print("Training Complete!")
    print(f"{'=' * 72}")
    best = metric_tracker.best_metric_values['valid']['Loss']
    print(f"Best validation loss: {best:.6f}")
    print(f"Total training time:  "
          f"{np.sum(metric_tracker.epoch_times):.2f} s")
    print(f"Average epoch time:   "
          f"{np.average(metric_tracker.epoch_times):.2f} s")
    print(f"Output directory:     {output_dirname}/")
    print(f"Active properties:    {cfg.active_flags_tag}")
    print(f"{'=' * 72}")


# =============================================================================
# Restart Summary
# =============================================================================


def print_restart_summary(
    cfg: TrainingConfig,
    checkpoint_dir: str,
    controller: Any,
    metric_tracker: Any,
) -> None:
    """Print checkpoint state summary before resuming training."""
    sep = "=" * 72
    print(f"\n{sep}")
    print("RESTART: RESUMING TRAINING FROM CHECKPOINT")
    print(sep)

    print(f"\n  Checkpoint directory:  {checkpoint_dir}")
    print(f"  Structure file:       experiment_structure.pt")
    print(f"  Checkpoint file:      best_checkpoint.pt")

    # Extract info from metric_tracker
    best_metrics = getattr(metric_tracker, "best_metric_values", None)
    epoch_times = getattr(metric_tracker, "epoch_times", [])
    epochs_completed = len(epoch_times)

    print(f"\n  Epochs completed:     {epochs_completed}")
    if epoch_times:
        print(
            f"  Total time so far:    {np.sum(epoch_times):.2f} s"
        )
        print(
            f"  Avg epoch time:       {np.average(epoch_times):.2f} s"
        )

    if best_metrics and "valid" in best_metrics:
        valid_metrics = best_metrics["valid"]
        if "Loss" in valid_metrics:
            print(
                f"  Best validation loss: "
                f"{valid_metrics['Loss']:.6f}"
            )

    # Controller state
    if hasattr(controller, "max_epochs"):
        print(f"\n  Controller max_epochs:          "
              f"{controller.max_epochs}")
    if hasattr(controller, "termination_patience"):
        print(f"  Controller termination_patience: "
              f"{controller.termination_patience}")
    if hasattr(controller, "batch_size"):
        print(f"  Current batch_size:             "
              f"{controller.batch_size}")

    # Optimizer learning rate
    if hasattr(controller, "optimizer"):
        for i, pg in enumerate(controller.optimizer.param_groups):
            lr = pg.get("lr", "N/A")
            print(f"  Optimizer LR (group {i}):        {lr}")

    # Overrides
    overrides = []
    if hasattr(controller, "max_epochs"):
        if cfg.max_epochs != controller.max_epochs:
            overrides.append(
                f"  max_epochs: {controller.max_epochs} -> "
                f"{cfg.max_epochs}"
            )
    if hasattr(controller, "termination_patience"):
        if cfg.termination_patience != controller.termination_patience:
            overrides.append(
                f"  termination_patience: "
                f"{controller.termination_patience} -> "
                f"{cfg.termination_patience}"
            )

    if overrides:
        print(f"\n  Parameter overrides for this restart:")
        for o in overrides:
            print(o)

    print(f"\n  Database will be reloaded from: {cfg.data_dir}")
    print(sep)
    print()


# =============================================================================
# Main
# =============================================================================


def main(argv: Optional[List[str]] = None) -> None:
    """Main entry point: parse -> validate -> build -> train -> save."""

    # 1. Parse
    cli = parse_cli_args(argv)
    file_config: Optional[Dict[str, str]] = None
    if cli.config_file:
        if not os.path.isfile(cli.config_file):
            print(f"ERROR: Config file not found: {cli.config_file}")
            sys.exit(1)
        file_config = parse_config_file(cli.config_file)
        print(f"Loaded config file: {cli.config_file}")

    cfg = merge_and_resolve(cli, file_config)

    # 2. Validate
    validate_config(cfg)

    # 3. Determine states (needed for output dirname even on restart)
    training_states, required_states, nacr_pairs = determine_states(cfg)

    # 4. Validate NACR shapes (skip on restart — already validated)
    if not cfg.restart:
        if cfg.build_nacr_nodes and nacr_pairs:
            print("\nValidating NACR dataset shapes...")
            validate_nacr_shapes(cfg, nacr_pairs)
        if (
            cfg.use_direct_nacr_v2_head or cfg.use_direct_nacr_v3_head
        ) and nacr_pairs:
            print("\nValidating GDV input shapes...")
            validate_gdv_shapes(cfg, nacr_pairs)
        if cfg.use_direct_nacr_v3_head and nacr_pairs:
            print("\nValidating GAP input shapes...")
            validate_gap_shapes(cfg, nacr_pairs)

    # 5. Output directory
    output_dirname = (
        cfg.output_dir if cfg.output_dir
        else generate_output_dirname(cfg, nacr_pairs)
    )

    # 6. Torch settings
    torch.backends.cuda.matmul.allow_tf32 = cfg.allow_tf32
    torch.backends.cudnn.allow_tf32 = cfg.allow_tf32
    dtype_map = {"float32": torch.float32, "float64": torch.float64}
    torch.set_default_dtype(
        dtype_map.get(cfg.default_dtype, torch.float32)
    )
    hippynn.settings.WARN_LOW_DISTANCES = False
    hippynn.settings.TRANSPARENT_PLOT = True

    # ================================================================
    # RESTART BRANCH
    # ================================================================
    if cfg.restart:
        checkpoint_dir = cfg.restart_dir if cfg.restart_dir else output_dirname

        # Validate checkpoint directory and files
        if not os.path.isdir(checkpoint_dir):
            print(
                f"\nERROR: Restart directory does not exist: "
                f"{checkpoint_dir}"
            )
            print(
                "  Ensure the output directory from the previous "
                "training run is accessible."
            )
            sys.exit(1)

        structure_file = os.path.join(
            checkpoint_dir, "experiment_structure.pt"
        )
        checkpoint_file = os.path.join(
            checkpoint_dir, "best_checkpoint.pt"
        )

        for fpath, label in [
            (structure_file, "Structure file"),
            (checkpoint_file, "Checkpoint file"),
        ]:
            if not os.path.isfile(fpath):
                print(f"\nERROR: {label} not found: {fpath}")
                print(
                    "  Cannot restart without both "
                    "experiment_structure.pt and best_checkpoint.pt."
                )
                sys.exit(1)

        with hippynn.tools.active_directory(checkpoint_dir):
            # Append to existing log with restart separator
            with hippynn.tools.log_terminal("training_log.txt", "at"):

                print("\n" + "=" * 72)
                print(f"RESTART AT {__import__('datetime').datetime.now()}")
                print("=" * 72)

                # Load checkpoint
                print("\nLoading checkpoint...")
                check = load_checkpoint_from_cwd(
                    model_device="auto",
                    restart_db=False,
                )

                training_modules = check["training_modules"]
                controller = check["controller"]
                metric_tracker = check["metric_tracker"]

                # Print restart summary
                print_restart_summary(
                    cfg, checkpoint_dir, controller, metric_tracker
                )

                # Apply overrides if provided.
                # PatienceController may expose max_epochs and
                # termination_patience as read-only properties.
                # We introspect __dict__ to find the backing attribute.
                def _override_attr(
                    obj: Any, name: str, new_val: Any
                ) -> bool:
                    """Try to set *name* on *obj*.  Falls back to common
                    private-attribute patterns if the public name is a
                    read-only property."""
                    # Try direct assignment first
                    try:
                        setattr(obj, name, new_val)
                        return True
                    except AttributeError:
                        pass
                    # Search for backing attribute in __dict__
                    candidates = [
                        f"_{name}",        # _max_epochs
                        f"{name}_",        # max_epochs_
                        f"__{name}",       # __max_epochs
                    ]
                    # Also check _ClassName__name (name-mangled)
                    cls_name = type(obj).__name__
                    candidates.append(f"_{cls_name}__{name}")
                    for cand in candidates:
                        if cand in obj.__dict__:
                            obj.__dict__[cand] = new_val
                            print(
                                f"  (Set via backing attribute "
                                f"'{cand}')"
                            )
                            return True
                    # Last resort: brute-force search __dict__ for
                    # keys containing the name
                    for k, v in obj.__dict__.items():
                        if name in k and isinstance(v, type(new_val)):
                            obj.__dict__[k] = new_val
                            print(
                                f"  (Set via matching attribute "
                                f"'{k}')"
                            )
                            return True
                    return False

                if hasattr(controller, "max_epochs"):
                    if cfg.max_epochs != controller.max_epochs:
                        print(
                            f"Overriding max_epochs: "
                            f"{controller.max_epochs} -> "
                            f"{cfg.max_epochs}"
                        )
                        if not _override_attr(
                            controller, "max_epochs", cfg.max_epochs
                        ):
                            print(
                                "  WARNING: Could not override "
                                "max_epochs on the controller. "
                                "Training will use the checkpoint "
                                "value."
                            )

                if hasattr(controller, "termination_patience"):
                    if (cfg.termination_patience
                            != controller.termination_patience):
                        print(
                            f"Overriding termination_patience: "
                            f"{controller.termination_patience} -> "
                            f"{cfg.termination_patience}"
                        )
                        if not _override_attr(
                            controller,
                            "termination_patience",
                            cfg.termination_patience,
                        ):
                            print(
                                "  WARNING: Could not override "
                                "termination_patience. Training "
                                "will use the checkpoint value."
                            )

                # Reload database
                db_info = training_modules.evaluator.db_info
                print(f"\nReloading database from: {cfg.data_dir}")
                database = hippynn.databases.DirectoryDatabase(
                    name=cfg.dataset_prefix,
                    directory=cfg.data_dir,
                    seed=cfg.seed,
                    test_size=cfg.test_size,
                    valid_size=cfg.valid_size,
                    **db_info,
                )

                # Resume training
                print("\nResuming training...\n")
                metric_tracker = train_model(
                    training_modules,
                    database,
                    controller,
                    metric_tracker,
                    callbacks=None,
                    batch_callbacks=None,
                )

            # Save updated summary
            save_training_summary(cfg, metric_tracker, checkpoint_dir)

        return

    # ================================================================
    # FRESH TRAINING BRANCH
    # ================================================================

    # Pre-training summary
    print_pretraining_summary(
        cfg, training_states, required_states, nacr_pairs,
        output_dirname, cli_source=cli.config_file,
    )

    # Build and train
    with hippynn.tools.active_directory(output_dirname):
        with hippynn.tools.log_terminal("training_log.txt", "wt"):

            species = inputs.SpeciesNode(db_name="Z")
            positions = inputs.PositionsNode(db_name="R")
            if cfg.needs_position_grad:
                positions.requires_grad = True

            net_choice = 0 if cfg.network_type == "hipnn" else 1
            network = build_network(
                species, positions, cfg.network_params,
                choice=net_choice, l_max=cfg.l_max, n_max=cfg.n_max,
            )
            print("  Network constructed successfully")

            print("\nBuilding graph nodes...")
            nodes = build_graph_nodes(
                cfg, network, species, positions,
                required_states, nacr_pairs,
            )

            print("\nBuilding loss function...")
            final_loss, validation_losses = build_loss(
                cfg, nodes, network, required_states, nacr_pairs,
            )

            print("\nBuilding plotters...")
            plotter = build_plotters(
                cfg, nodes, network, required_states, nacr_pairs,
            )

            training_modules, db_info = (
                hippynn.experiment.assemble_for_training(
                    final_loss, validation_losses, plot_maker=plotter,
                )
            )

            optimizer = torch.optim.AdamW(
                training_modules.model.parameters(),
                lr=cfg.learning_rate,
            )
            scheduler = RaiseBatchSizeOnPlateau(
                optimizer=optimizer,
                max_batch_size=cfg.max_batch_size,
                patience=cfg.scheduler_patience,
                factor=cfg.scheduler_factor,
            )
            controller = PatienceController(
                optimizer=optimizer,
                scheduler=scheduler,
                batch_size=cfg.batch_size,
                eval_batch_size=cfg.eval_batch_size,
                max_epochs=cfg.max_epochs,
                stopping_key=cfg.stopping_key,
                fraction_train_eval=cfg.fraction_train_eval,
                termination_patience=cfg.termination_patience,
            )

            print(f"\nLoading database from: {cfg.data_dir}")
            database = hippynn.databases.DirectoryDatabase(
                name=cfg.dataset_prefix,
                directory=cfg.data_dir,
                seed=cfg.seed,
                test_size=cfg.test_size,
                valid_size=cfg.valid_size,
                **db_info,
            )

            training_modules, controller, metric_tracker = (
                setup_training(
                    training_modules=training_modules,
                    setup_params=hippynn.experiment.SetupParams(
                        controller=controller
                    ),
                )
            )

            print("\nStarting training...\n")
            metric_tracker = train_model(
                training_modules, database, controller,
                metric_tracker, callbacks=None, batch_callbacks=None,
            )

        save_training_summary(cfg, metric_tracker, output_dirname)


if __name__ == "__main__":
    main()
