# Validation report

Accuracy evidence for the pipeline, stage by stage, benchmarked against the AM1/CIS FSSH@PYSEQM reference.

> **The population numbers below are PROVISIONAL.** They track the manuscript's Table I but are not yet
> frozen; confirm them against the locked manuscript set before quoting. The acceptance test
> (`../tests/acceptance/check_populations.py`) certifies a run only once its targets file is marked
> `locked: true`. Regenerate the numbers from a fresh run with `../scripts/analysis/harvest_topo.py` /
> the example `05_harvest.py`.

The headline: the main result is the **gap-stratified, hybrid (both-branch) Model B, 5-seed ensemble**, for
**both tautomers**, compared against the Langevin reference ensemble. Model A (energies/forces) and NVE
energy conservation are validated for both tautomers.

---

## 1. Status at a glance

| Stage | Metric | Result | Status |
|---|---|---|---|
| Model A energies/forces | parity vs AM1/CIS | correlation ~1.0; gaps ~0.99 | validated |
| Model A gap at hop frames | MAE vs CIS/AM1 | ~7 meV, compression ~1.0 | validated |
| NVE energy conservation | max\|dE_tot\|, drift | ~0.02 meV/atom; tiny drift over 1e6 steps (16/16 cells) | validated |
| FSSH@MLIP populations (main) | hybrid head, 5-seed mean +/- sigma_ML | see Table I (provisional) | reported |

## 2. Model A: energies and forces
Model A reproduces the AM1/CIS energies and forces with parity correlations near unity and state-gap
correlations ~0.99, and it reconstructs the gradient-difference vectors Model B consumes. Critically for
FSSH, the gap **at the geometries where hops occur** is accurate (MAE ~7 meV, compression ~1.0 vs the
reference) -- the model does not systematically compress the near-degenerate gap. Larger capacity does not
improve the hop-frame gap, so it is well learned at the production network size. Validated for both tautomers.

## 3. NVE energy conservation
The production NVE grid (S0-S3 x {enol, keto} x {single-state, multistate}, dt = 0.1 fs, 1e6 steps) passed
**16/16**. Worst-case `max|dE_tot|` ~0.02 meV/atom, far under the 1 meV/atom working criterion; the drift is
tiny and all trajectories stayed connected. The four-check protocol (conservation + integrity + timestep
convergence + reference-QM) is in `nve_at_mlip_guide.md`. Validated for both tautomers.

## 4. FSSH@MLIP populations vs the reference (Table I -- PROVISIONAL)
Main result = the stratified **hybrid** (both-branch) Model B, averaged over the 5 split seeds
(sigma_ML = sample std, ddof=1), survivor-conditioned, compared to the Langevin reference ensemble.
Enol is reported at 1 ps, keto at 10 ps.

| Tautomer | State | Classical ML (mean +/- sigma_ML) | Classical Ref (mean +/- sigma_Ref) |
|---|---|---|---|
| enol (1 ps) | S1 | ~1.000 | ~1.000 |
| keto (10 ps) | S1 | ~0.710 +/- 0.048 | ~0.612 +/- 0.016 |
| keto (10 ps) | S2 | ~0.105 +/- 0.006 | ~0.088 +/- 0.016 |
| keto (10 ps) | S3 | ~0.186 +/- 0.048 | ~0.300 +/- 0.007 |

(Provisional, to be confirmed against the frozen Table I; the machine-readable targets, including the enol
per-state values, live in `../tests/acceptance/targets_{enol,keto}.json`.) Always report both the
survivor-conditioned and the original-N normalizations, and compare apple-to-apple (same ICs, manifold,
initial state, timestep, survivor conditioning).

## 5. Static vs dynamics (why the hybrid head is the main result)
The coupling head that is **worst** on the static dENACR MAE is not the one that best reproduces the
*dynamics*. An earlier deployed keto configuration used the gdv-only head; the current main result for both
tautomers is the **hybrid** (both-branch) head. Do not resurrect the superseded gdv-only keto number as the
headline -- the static-MAE table and the dynamical populations rank the heads differently, which is the
scientific point. The gdv-only and geom-only heads are retained as ablations.

## 6. Caveats when comparing to the reference
- **TDC method differs by construction.** The reference uses `hamiltonian_fd`; the MLIP uses `nac_dot_v`
  (no CIS amplitudes). Validated harmless at the ensemble level; note it.
- **Survival asymmetry.** The reference loses trajectories to near-conical eigensolve deaths the MLIP does
  not; compare survivor-conditional populations and report the differing survivor counts.
- **Per-IC seeding.** Each initial condition is seeded `base + global_index` (one draw per step), matching
  the reference (see `fssh_mlip_guide.md`).
- **Guarded harvest.** Populations are harvested with the fragmentation guard (`harvest_topo.py`); report
  the health yield.

## 7. Bottom line
- Model A (E+F) and NVE conservation: **validated**, both tautomers.
- FSSH@MLIP main result: the stratified hybrid 5-seed ensemble, both tautomers, numbers **provisional**
  pending the frozen Table I; certify a run with `../tests/acceptance/check_populations.py` once its
  targets are locked.

See also: `inference_benchmark_report.md`, `known_limitations.md`, `model_b_math_background.md`, and the
FSSH guides.
