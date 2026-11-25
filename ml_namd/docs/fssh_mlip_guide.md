# FSSH@MLIP guide: MLIP-driven fewest-switches surface hopping

FSSH@MLIP runs nonadiabatic surface-hopping dynamics with a trained MLIP as the electronic-structure
engine instead of an on-the-fly quantum-chemistry calculation. The classical propagation, hopping, and
decoherence machinery is PySEQM's; only the source of energies, forces, and couplings is replaced. This is
the deliverable the whole pipeline builds toward, and it is benchmarked against FSSH@PYSEQM (see
`fssh_pyseqm_guide.md`).

Scripts live in `../scripts/fssh_mlip/`. The physics of the coupling is in `model_b_math_background.md`;
Model A/B training is in `model_a_guide.md` / `model_b_usage_guide.md`.

Placeholders: `$PY` = the project Python (see `../env/README.md`); `<MODEL_A>` / `<MODEL_B>` = trained model
directories.

---

## 1. What it computes each step

Recap (full derivation in the math doc). At each geometry the engine needs, per state pair `ij`:

1. **energies** `E_k` and **forces** `F_k` -- from Model A (propagation + gaps + GDV);
2. the **scaled coupling** `D_ij` (dENACR) -- from Model B;
3. the **derivative coupling** `d_ij = D_ij / (E_j - E_i)`, using the **raw** gap;
4. the **time-derivative coupling** `sigma_ij = d_ij . v`, which drives the hop probability.

Signs are made temporally consistent by a per-state phase match against the previous step before the
coherences are propagated.

## 2. Architecture

Two driver flavors, both subclassing PySEQM's `SurfaceHoppingDynamics`:

- **Composed A->B (production).** `ComposedSurfaceHopping` (`hippynn_fssh_composed.py`) uses a frozen
  Model A for E/F and a separate Model B for the coupling. A thin adapter wraps `ModelBComposedPredictor`
  (in `enol_ef_nacr_developer_handoff/src/`), which internally computes `GAP_ij = E_j - E_i` and
  `GDV_ij = F_i - F_j` from Model A and feeds them to Model B. Batch runner: `namd_composed_batch.py`.
- **Joint single model (legacy).** `HiphopSurfaceHopping` (`hippynn_fssh.py`) uses one network that emits
  E, F, and NACR together, through `HiphopPredictorV3`. Batch runner: `namd_batch.py`.

Both share the engine in `hippynn_fssh.py`: `_compute_electronic_structure` queries the predictor, builds
the NAC dictionary as `D_ij / Delta_ij` (raw gap), applies `_phase_match`, and forms `mol.nac_dot`.
`_phase_match` is the **GPU-vectorized** per-state sign alignment (it selects the best gauge for all
trajectories in one batched operation, replacing the former per-molecule loop; bit-identical, unit-verified).

## 3. Configuration baked into the driver

The engine sets the PySEQM nonadiabatic block to:

```python
"nonadiabatic": {"tdc_method": "nac_dot_v", "decohere_on_hop": True, "detect_crossings": False}
```

- **`tdc_method = "nac_dot_v"`.** The MLIP has no CIS amplitudes, so the finite-difference overlap TDC
  (`hamiltonian_fd`, used by the FSSH@PYSEQM reference) is infeasible; the velocity-projected coupling
  `sigma = d . v` is used instead. This TDC choice is the one physical difference from the reference and is
  validated harmless at the ensemble level.
- **`decohere_on_hop = True`.** Decoherence is applied on hops; without it the quantum populations stay
  trapped on the initial surface (over-coherence), while the classical populations still relax.
- **`detect_crossings = False`** for the MLIP path.

Run-level choices are CLI arguments to the batch runner: `--states 0,1,2,3`, `--nacr-pairs 12,13,23`,
`--initial-state 3`, `--nsteps`, `--timestep 0.05`, `--temperature`, initial conditions.

## 4. Critical rules (do not violate)

- **Raw gap, no floor.** `d = D / Delta` divides by the raw gap with no floor and no clamp. An earlier
  `max(Delta, 1e-8)` inverted small MLIP gaps, exploded the coupling and the TDC, and produced amplitude
  NaNs that froze hopping. The physical near-CI gap is finite (~1e-2 eV).
- **Survivors-only ensembles; S0 is never plotted.** Populations are conditioned on the live ensemble; the
  ground state is excluded from the excited-state population panels.
- **Phase consistency is a training problem.** The driver's `_phase_match` aligns a consistent per-state
  gauge to the previous step but **cannot** repair a pair-inconsistent (pair-scramble) coupling; that must
  be fixed in Model B's training loss (see the joint per-state loss in `model_b_usage_guide.md`).
- **Velocity at t=0.** `nac_dot_v` needs velocities at the first step, so the engine seeds them during
  `initialize` before the first electronic-structure evaluation.
- **Per-IC seeding.** For an like-for-like comparison with the reference, seed per initial condition
  (`422 + global_index`), i.e. run one IC per process (`--start idx --count 1 --seed 422+idx`) or ensure
  the runner derives the per-IC seed -- not one fixed seed for a whole batch.

## 5. Running it

Batched composed sweep over N initial conditions:

```bash
$PY ../scripts/fssh_mlip/namd_composed_batch.py \
    --modelA-dir <MODEL_A>/model --modelB-dir <MODEL_B>/model \
    --coords-npy R0.npy --vel-npy V0.npy --z-npy Z.npy \
    --states 0,1,2,3 --nacr-pairs 12,13,23 --initial-state 3 \
    --nsteps 200000 --timestep 0.05 --count 125 --start 0 \
    --output-dir out --output-prefix run
```

- **Batching / OOM.** `--count 125` is the OOM-safe maximum on a 40 GB GPU (the joint autograd path OOMs
  near 250). Startup is a tiny fraction of a 10 ps run, so larger batches buy almost nothing.
- **Multiple GPUs.** Speed comes from **chunk packing** (fanning `--start`/`--count` slices across GPUs),
  not intra-batch parallelism; a simple per-GPU loop (or the generic `../optional_hpc/` template as a job
  array) suffices. There is no cross-GPU communication.
- Unlike FSSH@PYSEQM there is no CIS eigensolve (the MLIP supplies E/F/NACR), so batched GPU propagation is
  safe and fast.

See `../examples/<tautomer>/04_run_ml_tsh.py` for a ready-to-run per-tautomer command.

## 6. Outputs, harvest, and the like-for-like comparison

Each trajectory writes a PySEQM h5 with `data/nonadiabatic/{active_surface, NACT, electronic_amplitudes}`
and `data/excitation/state_energies`. Harvest the ensemble with `../scripts/analysis/harvest_fssh_full.py`
(survivor-conditional live-ensemble normalization) into a compact `.npz`, then compare against the
FSSH@PYSEQM reference harvested the same way. Match the manifold (states/pairs), the initial state, the
timestep, and the survivor conditioning; report both survivor-conditioned and original-N normalizations.

Classical populations come from `active_surface`; quantum `|c|^2` from `electronic_amplitudes` (excited-only,
renormalized, S0 as the norm deficit). See the harvester header for the exact conventions.

## 7. Composed vs joint -- which to use

Use the **composed A->B** driver for production: it lets Model A (E/F) and Model B (coupling) be trained,
frozen, and swapped independently. The **joint** driver is retained for the single-model line and
comparison. Both produce the same h5 schema and are harvested identically.

## 8. Gotchas

- **Unpickle dependencies.** Loading the models rebuilds the graphs, so `direct_nacr_head_v3` (and
  `joint_state_nacr_loss` for joint-trained Model B) must be importable before `torch.load`. The engine
  prepends the training modules + the predictor bundle to `sys.path`; keep `enol_ef_nacr_developer_handoff/src/`
  next to the driver.
- **Species order** `[8,8,6,6,6,6,6,1x8]`; the ICs' Z must match.
- **Pair order** `12,13,23` must agree across the model, the `--nacr-pairs` argument, and the harvest.
- **Custom kernels off** (float64): `HIPPYNN_USE_CUSTOM_KERNELS=False`.
- **Termination.** The engine forces all-converged CIS flags (the MLIP never runs SCF/CIS), so the gap-stop
  fires on the S0-S1 gap; trajectories terminate at the ground-state funnel.

## 9. Verify

1. Environment: `python ../env/verify_env.py`.
2. Smoke: run the composed batch runner with a small `--count` (e.g. 5 ICs, 500 steps). A pass is: clean
   model load, h5 written, hops firing, and **zero amplitude NaN**.
3. Short production chunk + harvest, and eyeball the survivor-conditional populations against the reference
   before launching the full sweep.

See also: `model_b_math_background.md` (the coupling), `fssh_pyseqm_guide.md` (the reference), and
`slurm_job_guide.md` (running the pipeline).
