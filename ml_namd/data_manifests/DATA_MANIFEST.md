# Data manifest — schema and provenance

The datasets are **not stored in this repository**. They are distributed through the Harvard Dataverse
record and downloaded into each example before you run it:

> Khanna, Ajay, 2026, *Replication Data for: Machine Learning for Non Adiabatic Excited State Molecular
> Dynamics*, https://doi.org/10.7910/DVN/HCL8EK, Harvard Dataverse.

The published bundle ships the **base active-learning (BALF)** training pools and the initial conditions
for **both tautomers** (enol and keto). Download it, then place each tautomer's arrays under
`../examples/<taut>/data/` and the initial conditions under `../examples/<taut>/ics/` (see that example's
`00_download_data.md`). This file documents the array schema so you can verify a download.

## Shared conventions
- **Molecule:** acetylacetone (AcAc), 15 atoms. Atom order in every `acn_Z` is
  `[8, 8, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1]` (2 O, 5 C, 8 H), identical for enol and keto.
- **File prefix:** `acn_`. One `.npy` per array; the leading axis is the frame index.
- **State index `k`:** 0 = S0 (ground), 1.. = excited. `acn_sE{k}` = state energy, `acn_F{k}` = state force.
- **Pair index `ij`:** two-digit adiabatic state pair, e.g. `12` = S1-S2. Model B uses `12, 13, 23`.
- **Energy shift:** `sE = E_raw - E0`, with `E0(enol) = -1391.422970121047`, `E0(keto) = -1391.451136550848` (eV).
- **Units:** positions `R` in A, energies in eV, forces in eV/A, gaps `GAP` in eV, NACR in A^-1, GDV
  (gradient-difference vectors) in eV/A; `NACT = NACR . V . C` with `C = 2.41888425e-5`.

## 1. Model A training pool (energies + forces)
Ground-state-sampled geometries labeled with the reference electronic structure. enol uses the S0-S3
manifold (an S0-S5 superset is FSSH-compatible); keto uses S0-S5.

| Array | Shape | Dtype | Meaning |
|---|---|---|---|
| `acn_R` | (N, 15, 3) | float64 | atomic positions [A] |
| `acn_Z` | (N, 15) | int64 | atomic numbers (fixed order above) |
| `acn_V` | (N, 15, 3) | float64 | atomic velocities (when present) |
| `acn_sE0 .. acn_sE{K}` | (N,) | float64 | per-state shifted energies |
| `acn_F0 .. acn_F{K}` | (N, 15, 3) | float64 | per-state forces [eV/A] |

## 2. Model B training pool (scaled NACR / dENACR, BALF)
The base active-learning clean-geometry pool; coupling pairs `12, 13, 23`.

| Array | Shape | Dtype | Role |
|---|---|---|---|
| `acn_R`, `acn_Z` | (N,15,3), (N,15) | f64, i64 | geometry + species |
| `acn_dENACR12/13/23` | (N, 45) | float64 | **target** scaled NACR (flattened 15x3) per pair |
| `acn_GDV12/13/23` | (N, 45) | float32 | **input** gradient-difference vector per pair |
| `acn_GAP12/13/23` | (N,) | float64 | **input** state-pair energy gap [eV] |
| `acn_sE*`, `acn_F*` | (N,), (N,15,3) | float64 | state energies + forces (context) |

> At inference the GDV/GAP **inputs** come from Model A's predictions, written by the genA step
> (`../scripts/data_generation/gen_modelA_inputs.py`); the dENACR **target** is the reference coupling.
> The A-predicted GDV/GAP arrays must be isolated from any reference copies so the trainer reads the
> A-predicted inputs (see `../docs/model_b_usage_guide.md`).

## 3. FSSH reference + initial conditions
FSSH@PYSEQM reference trajectories (`tdc_method = hamiltonian_fd`, `decohere_on_hop = True`) and the
initial conditions that launch **both** FSSH streams (MLIP and reference) from identical starting points.
Per-trajectory HDF5 schema (F frames, S states):

| Path in h5 | Shape | Dtype | Meaning |
|---|---|---|---|
| `/data/excitation/state_energies` | (F, S) | float64 | adiabatic state energies per frame |
| `/data/nonadiabatic/NACT` | (F, S, S) | float64 | time-derivative coupling matrix |
| `/data/nonadiabatic/active_surface` | (F,) | int64 | active surface per frame (hopping) |
| `/data/nonadiabatic/electronic_amplitudes` | (F, S, 2) | float64 | complex amplitudes (re, im) |
| `/data/thermo/{Ek,Ep,T}` | (F,) | float64 | kinetic/potential energy, temperature |

Initial conditions are the matched per-IC `R0` / `V0` / `Z` arrays (the MLIP driver reads them directly;
the reference reads the same starting points). The per-IC seed is `422 + index`, shared by both streams;
atoms are in descending-Z order. Exact arrays, counts, and integrity hashes ship with the Dataverse
bundle.

## Notes
- Transition dipoles / oscillator strengths are not stored in the reference h5 (only the ground dipole);
  recover them via a single-point CIS on saved geometries if needed.
- The reference loses some trajectories to near-conical eigensolve failures the MLIP does not; always
  compare survivor-conditioned populations and report the differing survivor counts.
