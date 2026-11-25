# NVE@MLIP guide: energy-conserving dynamics driven by Model A

NVE@MLIP runs microcanonical (constant energy) molecular dynamics with **Model A** (the E+F network)
as the potential-energy engine, on a chosen adiabatic state. Its purpose is validation: a good E+F
model must conserve total energy over long trajectories without the molecule falling apart. It is the
single-state precursor to the multistate FSSH@MLIP driver.

Scripts live in `../scripts/nve_mlip/`. Model A background is in `model_a_guide.md`.

Placeholders: `$PY` = the project Python (`../env/README.md`); `<ND>` = the NVE working directory;
`<MODEL>` = a trained Model A directory.

---

## 1. Validation philosophy: four checks, not one

Machine-learned energy conservation **alone is misleading**. A model can conserve a smooth but wrong
surface, or drift only because of the integrator rather than the PES. Every NVE run is judged on four
independent checks:

1. **Energy conservation** — total energy drift over the trajectory (`check_nve_drift.py`).
2. **Structural integrity** — no fragmentation, collision, or bond blow-up (`analyze_geom_integrity.py`).
3. **Timestep convergence** — the drift must vanish as the timestep shrinks (integrator error), not
   persist (which would indicate genuine PES roughness).
4. **Reference-QM agreement** — the ML energies/forces track AM1/CIS on sampled frames
   (`reference_qm_check.py`).

A run passes only when all four agree. Conservation with fragmentation, or conservation that does not
improve with a smaller timestep, is a failure.

## 2. The workflow

```
make_start_xyz.py        pull a start geometry (S0 frame, well-sampled)
        │
        ├─ S0 :  NVE directly from the frame
        │
        └─ S1–S3 : pyseqm_hippynn_optimize.py  (geomeTRIC-relax onto the S_k minimum) ─► optimized .xyz
        │
   make_nve_inputs.py / make_prod_inputs.py     write the per-run .input files
        │
   pyseqm_hippynn_md.py  ─►  NVE trajectory (h5 + xyz)
        │
   check_nve_drift.py + analyze_geom_integrity.py + reference_qm_check.py   validate
```

- **Start geometry.** `make_start_xyz.py` writes `geom/{t}_s0_start.xyz` from a well-sampled S0-MD frame
  (frame 25000). S0 runs NVE directly from that frame.
- **Excited-state start (S1–S3).** Starting an excited run from the S0-region frame is wrong; first
  relax onto the $S_k$ minimum with `pyseqm_hippynn_optimize.py` (geomeTRIC, using the *same*
  `HIPPYNN_ESDriver` as the runner so the PES is identical), then run NVE on that surface.
- **Inputs.** `make_nve_inputs.py` / `make_prod_inputs.py` generate self-contained `.input` files (one
  per cell), so a production step is `-i <file>` with no CLI overrides (reproducibility).

## 3. The runner and its `.input` contract

`pyseqm_hippynn_md.py` wraps PySEQM's `Molecular_Dynamics_Basic` (NVE) — or `_Langevin` (NVT) — around a
HIPPYNN E+F model for one state. The `.input` keys:

```
model_dir            trained Model A directory
xyz_file             start geometry (S0 frame, or the optimized excited .xyz)
state_index          adiabatic state 0..5 -> uses sE{k} / F{k}
energy_offset        E_gs_opt for the tautomer (added back to the shifted ML energy)
ensemble             nve  (or nvt, which adds `friction`)
temperature          init temperature [K] (Maxwell-Boltzmann seeding, then microcanonical)
timestep             fs  (production: 0.1)
nsteps               number of steps (100 ps at 0.1 fs = 1,000,000)
random_seed          keep fixed across compared runs (42)
output_prefix/dir, trajectory_interval, log_interval, device
```

Internally the runner **prunes** the graph to `GraphModule([Z, R], [sE_k, F_k])` (so oracle GDV input
nodes cannot break it), calls it under `torch.enable_grad()` with a detached clone of the coordinates,
casts to float64, and adds `energy_offset`. See `model_a_guide.md` (Section 4) for why each step is
needed.

## 4. Production settings (do not change without re-validating)

- **Timestep 0.1 fs** (not 0.5). 100 ps therefore = 1,000,000 steps.
- **Maxwell–Boltzmann initialization at 300 K, then microcanonical** (NVE); same `random_seed = 42`
  across runs you compare.
- **`energy_offset = E_gs_opt`:** enol `-1391.422970121047`, keto `-1391.451136550848` eV.
- **Model selection: per-dynamics matched multistate.** Each state uses a model trained on that state's
  own dynamics distribution (S0 → S0-dynamics model, S3 → S3-dynamics model, ...). Do not run an S3 NVE
  with an S0-distribution model — cross-distribution use looks like fragmentation but is a sampling
  mismatch, not a model failure.

## 5. Pass criteria (from the production grid)

The 100 ps S0–S3 × {enol, keto} grid (dt = 0.1 fs, seed 42, 10⁶ steps) passed 16/16: worst
`max|dEtot|` = 0.33 meV = **0.022 meV/atom**, i.e. ~45× under the 1 meV/atom working criterion; `|drift|`
≤ ~1.3 µeV/ps in every cell; all trajectories connected with no fragmentation over 100 ps. Use as the
bar: **max|dEtot| well under 1 meV/atom, drift ≤ a few µeV/ps, zero fragmentation, and ML–QM agreement
on sampled frames.**

## 6. The timestep lesson (why check 3 matters)

An early multistate S3 run at **dt = 0.5 fs drifted +4.68 meV/ps (FAIL)**; the same model at
**dt = 0.1 fs drifted +0.001 meV/ps (PASS)** with the molecule intact. The drift was **integrator
error, not PES roughness** — it vanished with a smaller timestep. This is exactly why timestep
convergence is a required check: a single-timestep "conservation failure" can be an integrator artifact,
and a single-timestep "pass" can hide roughness. Always confirm the trend with at least two timesteps.

## 7. Running it

Every step is a plain `python` command (a CUDA GPU is required); there is no scheduler. To fan many
cells across GPUs, adapt the generic template in `../optional_hpc/`.

- **Smoke:** build a short dt = 0.1 fs NVE with `../scripts/nve_mlip/make_nve_inputs.py`, run it with
  `../scripts/nve_mlip/pyseqm_hippynn_md.py`, and check conservation with
  `../scripts/nve_mlip/check_nve_drift.py` — exercises the runner + drift check on both models.
- **Multistate:** optimize S1/S2/S3 onto their PES with `../scripts/nve_mlip/pyseqm_hippynn_optimize.py`,
  then run the S0–S3 NVE cells (one per state × tautomer) with `../scripts/nve_mlip/pyseqm_hippynn_md.py`.
- **Production:** generate the 100 ps input-file grid for one tautomer with
  `../scripts/nve_mlip/make_prod_inputs.py`, then run each input with `../scripts/nve_mlip/pyseqm_hippynn_md.py`.

## 8. Gotchas

- **Energy offset must match the tautomer** (enol vs keto); a wrong offset shifts the total energy but,
  because it is constant, still "conserves" — so the drift check passes while the absolute energy is
  wrong. Cross-check with `reference_qm_check.py`.
- **`torch.enable_grad()`** is required for forces; **pass a detached clone** of the coordinates.
- **Species order** `[8,8,6,6,6,6,6,1×8]`, `read_xyz(sort=False)`.
- **Custom kernels off** for float64 (`HIPPYNN_USE_CUSTOM_KERNELS=False`).
- **geomeTRIC** must be installed in the environment for the excited-state optimizer.

## 9. Verify

1. Environment: `python ../env/verify_env.py`.
2. Build + run a short NVE (`make_nve_inputs.py` → `pyseqm_hippynn_md.py`), then confirm
   `check_nve_drift.py` reports ≈ 0 meV/ps at dt = 0.1 fs with no fragmentation.
3. For an excited state, confirm the optimizer produced a sensible $S_k$ minimum before the NVE.

See also: `model_a_guide.md` (the E+F model), `fssh_mlip_guide.md` (the multistate successor), and the
best-practices framework the four checks come from.
