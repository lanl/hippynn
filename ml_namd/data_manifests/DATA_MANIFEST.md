# Data manifest — enol-form (Model A, Model B, FSSH reference)

Schema + provenance for the three datasets this side-quest uses. **The data itself is not stored in
this repo** (per project decision); it lives on the laptop ground-zero drive `D:/ml_namd_project/`
(mirrored from the clusters). Profiled 2026-08-31 by header-only inspection (shapes/dtypes exact).

## Shared conventions
- **Molecule:** acetylacetone (AcAc), 15 atoms. Atom order in every `acn_Z` = `[8, 8, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1]` (2 O, 5 C, 8 H). Same order for enol and keto.
- **File prefix:** `acn_`. Per-array `.npy`, leading axis = frame index (sample).
- **State index `k`:** 0 = S0 ground state, 1..10 = excited. `acn_sE{k}` = state energy, `acn_F{k}` = state force. Model A here trains S0–S5 (k = 0..5); the arrays carry through S10.
- **Pair index `ij`:** two-digit adiabatic state pair, e.g. `12` = S1–S2. Model B uses the 3 pairs `12, 13, 23`.
- **Energy shift:** `sE = E_raw − E0_opt`, with `E0_opt(enol) = −1391.422970121047`, `E0_opt(keto) = −1391.451136550848` (eV).
- **Units (project convention):** positions `R` in Å, energies in eV, forces in eV/Å, energy gaps `GAP` in eV, NACR in Å⁻¹, GDV (gradient-difference vectors) in eV/Å; `NACT = NACR · V · C` with `C = 2.41888425e-5` (velocity-dotted time-derivative coupling).

---

## 1. Model A training set — enol S0-dynamics (E + F, S0–S5)
- **Purpose:** train Model A (energies + forces, multistate S0–S5) on ground-state MD geometries.
- **Location (canonical, laptop):** `D:/ml_namd_project/data/enol/{5k,50k,100k}/s0_dyn/` (50k profiled below; 100k has the same schema, `N = 100000`).
- **Origin:** PySEQM AM1/CIS single points on enol GS-MD frames (registry `Dataset_Registry_July_11_2026.txt`).
- **Size:** 316 `.npy` (full 10-state NACR/NACT/GDV/GAP/dipole payload; Model A needs only the E/F subset).

| Array | Shape | Dtype | Meaning (Model A subset) |
|---|---|---|---|
| `acn_R` | (50000, 15, 3) | float64 | atomic positions [Å] |
| `acn_Z` | (50000, 15) | int64 | atomic numbers (fixed order above) |
| `acn_V` | (50000, 15, 3) | float64 | atomic velocities |
| `acn_sE0 … acn_sE10` | (50000,) | float64 | per-state energies (shifted), k = 0..10 |
| `acn_F0 … acn_F10` | (50000, 15, 3) | float64 | per-state forces [eV/Å] |
| *(also present, for Model B / analysis)* | | | `acn_dENACR{ij}` (…,45) f64, `acn_GDV{ij}` (…,45) f32, `acn_GAP{ij}`, `acn_NACR{ij}` (…,15,3), `acn_NACT{ij}`, `acn_D*` dipoles |

---

## 2. Model B training set — enol BALF (clean geometries, dENACR pairs 12/13/23)
- **Purpose:** train Model B (scaled NACR / dENACR, v3 head) on the BALF (base active-learning) clean-geometry pool; consumes Model A's predicted gaps + GDVs at inference.
- **Location (canonical, laptop):** `D:/ml_namd_project/data_gen/enol_balf_galf/BALF/`
  - top level: curated raw arrays — `energies_all_states.npy`, `forces_all_states.npy`, `nacrs_all_states.npy`, `transition_dipoles.npy`, `gs_dipoles.npy`, `ex_dipoles_net.npy`, `total_energy.npy`, `acn_R.npy` (50000,15,3), `acn_Z.npy` (50000,15).
  - **training-ready S0–S3 pool:** `BALF/pool_s0s3/` — the per-pair Model-B inputs/targets below.
  - `BALF/extracted/50K/` + `generation_manifest.json` (generation provenance).
- **Cluster origin:** Chicoma `/lustre/scratch5/akhanna2/ml_namd_project/data_gen/enol_balf_galf/BALF/` (the keto analogue trained from `balf_galf_full/BALF/modelB_nf145_s0s5_10p`).

Model-B pool (`BALF/pool_s0s3/`) key arrays:

| Array | Shape | Dtype | Meaning |
|---|---|---|---|
| `acn_R`, `acn_Z` | (N,15,3), (N,15) | f64, i64 | geometry + species |
| `acn_dENACR12/13/23` | (N, 45) | float64 | **target** scaled NACR (flattened 15×3) per pair |
| `acn_GDV12/13/23` | (N, 45) | float32 | **input** gradient-difference vector per pair (Model A at inference) |
| `acn_GAP12/13/23` | (N,) | float64 | **input** state-pair energy gap (Model A at inference) |
| `acn_sE0..3`, `acn_F0..3` | (N,), (N,15,3) | float64 | S0–S3 energies + forces (context) |

---

## 3. FSSH reference + initial conditions — Vishikh's 1000-IC enol
- **Purpose:** FSSH@PYSEQM reference trajectories (init = S3, `hamiltonian_fd` TDC, `decohere_on_hop=True`, 1 ps) and the initial conditions for both the reference and the MLIP FSSH runs.
- **Location (canonical, laptop):** `D:/ml_namd_project/fssh_ref/enol_init3_hfd_1ps_darwin/`
  - `enol_init3_hfd_decohere/` — **1000** per-trajectory h5 files `enol_s3hfd_i{0000..0999}.0.h5`.
  - survivor extraction (`*_survivors_*.npy`): `R` (777, 501, 15, 3) f32, `Z` (15,) i16, `ic_index` (777,) i32, `has_nan` (777,) bool, `n_surv`/`NF`/`meta` scalars → **777 of 1000 ICs survive 1 ps**.
- **Origin:** Darwin `/vast/home/akhanna2/data/ml_project/ml_namd_project/fssh_ref_updated/` (venv_master, PySEQM master). Compared as a strict n3 view (N = 775) against keto in the apple-to-apple analysis.

Per-trajectory h5 schema (501 frames = 1 ps, 4 states S0–S3):

| Path in h5 | Shape | Dtype | Meaning |
|---|---|---|---|
| `/data/steps` | (501,) | int64 | MD step index |
| `/data/excitation/state_energies` | (501, 5) | float64 | adiabatic state energies per frame |
| `/data/excitation/active_state` | () | int64 | initial active surface (= 3, S3) |
| `/data/nonadiabatic/NACT` | (501, 4, 4) | float64 | time-derivative coupling matrix |
| `/data/nonadiabatic/active_surface` | (501,) | int64 | active surface per frame (hopping) |
| `/data/nonadiabatic/electronic_amplitudes` | (501, 4, 2) | float64 | complex amplitudes (last axis = re, im) |
| `/data/properties/ground_dipole` | (501, 3) | float64 | ground-state dipole |
| `/data/thermo/{Ek,Ep,T}` | (501,) | float64 | kinetic / potential energy, temperature |

---

## 4. Raw launch initial conditions — matched 1000-IC enol (R0 / V0 / Z)
- **Purpose:** the raw per-IC arrays that launch *both* FSSH streams so the two share identical starting points. The MLIP driver (`namd_composed_batch.py`) reads them directly as `--coords-npy R0 --vel-npy V0 --z-npy Z`; the FSSH@PYSEQM reference consumes the same ICs in `coords.xyz` + NEXMD `velocity.out` form (`run_fssh_ref.py`), with the raw `velocity.out` divided by 1000 (`VEL_DIV`, see `fssh_pyseqm_guide.md`).
- **Expected location (laptop, canonical):** `D:/ml_namd_project/fssh_ref/enol_init3_hfd_1ps_darwin/ics_matched/enol1ps_{R0,V0,Z}.npy`.
- **Cluster origin:** Darwin `/vast/home/akhanna2/data/ml_project/ml_namd_project/fssh_ref_updated/ics/sampled_enol/` (xyz/velocity form for the reference) and the matched `ics_matched/` npy form.
- **Expected shapes/dtypes** (per the 15-atom AcAc convention; 1000 ICs): `R0` (1000, 15, 3) f64 [Å], `V0` (1000, 15, 3) f64, `Z` (15,) or (1000, 15) int, species order `[8,8,6,6,6,6,6,1×8]`. Seed per index = `422 + idx` (shared by both FSSH streams).

| Array | Expected shape | Dtype | md5 |
|---|---|---|---|
| `enol1ps_R0.npy` | (1000, 15, 3) | float64 | PENDING (drive absent) |
| `enol1ps_V0.npy` | (1000, 15, 3) | float64 | PENDING (drive absent) |
| `enol1ps_Z.npy` | (15,) / (1000, 15) | int | PENDING (drive absent) |

**md5 status:** `[UNVERIFIED — external drive D: not mounted at 2026-08-31 profiling]`. The canonical
copies live only on D: (absent) and the clusters; no copy exists on C:. When D: is mounted, pin the
three md5s (PowerShell):

```powershell
Get-FileHash -Algorithm MD5 D:\ml_namd_project\fssh_ref\enol_init3_hfd_1ps_darwin\ics_matched\enol1ps_R0.npy,`
  D:\ml_namd_project\fssh_ref\enol_init3_hfd_1ps_darwin\ics_matched\enol1ps_V0.npy,`
  D:\ml_namd_project\fssh_ref\enol_init3_hfd_1ps_darwin\ics_matched\enol1ps_Z.npy | Format-Table Hash,Path
```

Confirm the path (the `ics_matched/` subdir is the expected mirror of the cluster `ics/` set; verify it
exists on the mounted drive) and the shapes above before recording, then replace each `PENDING` cell.

---

## Notes and open items
- The reference h5 store transition dipoles / oscillator strengths are **not** saved (only `ground_dipole`); recover via single-point CIS on saved geometries if needed.
- md5 checksums are intentionally omitted for the multi-GB training arrays (Sections 1–2); the small IC files (Section 4) carry md5 once D: is mounted and the arrays are pinned.
- Survivor-extraction arrays for the reference (Section 3) are the 777-of-1000 subset; the raw launch ICs (Section 4) are the full 1000 before any survival filtering.
