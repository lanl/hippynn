"""Shared paths and settings for the enol worked example.

Every step is a plain `python` command; a CUDA GPU is required for training and dynamics. This module
resolves all repository paths relative to its own location, so the example runs from wherever the
repository lives -- no machine-specific paths.
"""
import os
from pathlib import Path

# --- tautomer-specific settings (the ONLY difference between the enol and keto examples) ---
TAUT = "enol"
IC_PREFIX = "enol1ps"          # initial-condition files: ics/<IC_PREFIX>_{R0,V0,Z}.npy
NSTEPS = 20000                 # enol = 1 ps at dt = 0.05 fs
# ------------------------------------------------------------------------------------------

SEEDS = [42, 101726, 102593, 112793, 112891]       # the 5 split seeds (sigma_ML ensemble)
HEADS = ["both", "gdv_only", "geom_only"]          # "both" == the hybrid main-result head

EX = Path(__file__).resolve().parent               # ml_namd/examples/<taut>
ML_NAMD = EX.parent.parent                          # ml_namd/
SCRIPTS = ML_NAMD / "scripts"
TOOLS = SCRIPTS / "training" / "tools"
TRAINER = SCRIPTS / "training"
FSSH = SCRIPTS / "fssh_mlip"
ANALYSIS = SCRIPTS / "analysis"
DATAGEN = SCRIPTS / "data_generation"

DATA_ROOT = EX / "data"        # BALF pool downloaded to DATA_ROOT/<taut>/balf (see 00_download_data.md)
RUN_ROOT = EX / "runs"         # training + genA outputs
ICS = EX / "ics"               # <IC_PREFIX>_{R0,V0,Z}.npy
OUT = EX / "out"               # FSSH@MLIP trajectories + harvest


def pool_dir():
    return DATA_ROOT / TAUT / "balf"


def modelA_dir(seed):
    return RUN_ROOT / "model_A" / TAUT / "balf" / "stratified" / f"rnd_seed_{seed}" / "model"


def modelA_leaf(seed):
    return RUN_ROOT / "model_A" / TAUT / "balf" / "stratified" / f"rnd_seed_{seed}"


def genA_pool(seed):
    return RUN_ROOT / "model_B" / TAUT / "balf" / "stratified" / f"rnd_seed_{seed}" / "genA_pool"


def modelB_dir(seed, head):
    return RUN_ROOT / "model_B" / TAUT / "balf" / head / "stratified" / f"rnd_seed_{seed}" / "model"


def base_env():
    """Environment for the tools: config_gen reads the data/run roots; float64 needs kernels off."""
    e = dict(os.environ)
    e["HIPPYNN_USE_CUSTOM_KERNELS"] = "False"
    e["ML_NAMD_DATA_ROOT"] = str(DATA_ROOT)
    e["ML_NAMD_RUN_ROOT"] = str(RUN_ROOT)
    return e
