# Validation report

Accuracy evidence for the pipeline, stage by stage, benchmarked against reference AM1/CIS. The honest
summary: **the enol-form FSSH@MLIP reproduces the FSSH@PYSEQM reference; the keto form over-relaxes and
is an active investigation** with a well-characterized cause. Model A (energies/forces) and NVE energy
conservation are validated for both tautomers.

Numbers below are representative results from the campaign, not a live re-run; regenerate them with the
harvest utilities (`../scripts/analysis/harvest_fssh_full.py`) against a fresh run before quoting in a
manuscript.

---

## 1. Status at a glance

| Stage | Metric | Result | Status |
|---|---|---|---|
| Model A energies/forces | parity vs AM1/CIS | correlation ≈ 1.0; gaps ≈ 0.99 | validated |
| Model A gap at hop frames | MAE vs true CIS/AM1 | ~7 meV, compression ~1.0 | validated |
| NVE energy conservation | max\|dE_tot\|, drift | 0.022 meV/atom, ≤ ~1.3 µeV/ps (16/16 cells) | validated |
| Model B coupling (honest) | hop-frame S1–S2 \|NACT\| ratio | ~1.17 (enol/keto phaseless) | mild bias |
| FSSH@MLIP populations — **enol** | classical/quantum S1 vs reference | S1 ≈ 1.0 classical / ~0.96 quantum vs ref 0.959 | validated |
| FSSH@MLIP populations — **keto** | survivor-cond. S1 vs reference | S1 ~0.77–0.83 vs ref 0.61 | **over-relaxes (open)** |

## 2. Model A: energies and forces

Model A reproduces the AM1/CIS energies and forces it is trained on with parity correlations near unity
and state-gap correlations ≈ 0.99, and it reconstructs the gradient-difference vectors Model B consumes.
Critically for FSSH, the **gap at the geometries where hops actually occur** (the near-degeneracy hop
frames) is accurate: MAE ~7 meV with compression ~1.0 versus the true CIS/AM1 reference — i.e. the model
does **not** systematically compress the near-CI gap. (An apparent "2× gap compression" seen earlier was
an S0-dynamics artifact, not the hop regime.) Larger network capacity (nf172 vs nf145) does not improve
the gap, so the gap is well learned at nf145.

## 3. NVE energy conservation

The 100 ps production NVE grid — S0–S3 × {enol, keto} × {single-state, multistate}, dt = 0.1 fs,
seed 42, 10⁶ steps — passed **16/16**. Worst-case `max|dE_tot|` = 0.33 meV = **0.022 meV/atom**, about
**45× under** the 1 meV/atom working criterion; `|drift|` ≤ ~1.3 µeV/ps in every cell; all trajectories
stayed connected with no fragmentation over 100 ps. Both per-state single models and the multistate
model conserve at production length. The four-check protocol (conservation + integrity + timestep
convergence + reference-QM) is described in `nve_at_mlip_guide.md`; the timestep check was decisive — a
dt = 0.5 fs "failure" (+4.68 meV/ps) became a pass (+0.001 meV/ps) at dt = 0.1 fs, showing the drift was
integrator error, not PES roughness.

## 4. Model B: coupling magnitude at hop frames

The load-bearing coupling metric is the median ratio of the MLIP time-derivative-coupling magnitude to
the reference at the frames where hops occur. For the honest phaseless Model B the **S1–S2 ratio is
~1.17** (both BALF and GALF sampling) — a mild ~16% over-estimate, consistent across the 3-pair and
4-pair models. The other pairs (13, 23) are near or under 1.0.

A caution surfaced during this analysis: the joint (sign-consistent) loss, in its pure sum-of-squares
form, **inflated** the S1–S2 magnitude to ~1.73, because a pure-L2 objective over-weights the large
near-CI couplings relative to the phaseless RMSE+MAE blend. That inflation is a loss-form artifact, not
the inherent model bias — match the loss form when comparing sign strategies (see
`model_b_math_background.md`, Section 5).

## 5. FSSH@MLIP populations vs the FSSH@PYSEQM reference

Populations are compared apple-to-apple: same initial conditions, manifold, initial surface, timestep,
and survivor conditioning, harvested identically for both.

**Enol — validated.** With the NaN-floor fix and decoherence on, both the composed A→B and joint drivers
match the reference for BALF and GALF sampling: classical S1 ≈ 1.000, quantum |c|² ≈ 0.96 against the
1000-IC reference's 0.959. Earlier reports of enol "under-relaxation" were a gap-floor artifact and are
retracted. Enol is the working demonstration of end-to-end FSSH@MLIP.

**Keto — over-relaxes (open).** For keto S0–S3 (init3, 10 ps), survivor-conditional final classical
populations: reference (ns3, 821 survivors) **S1 = 0.608 / S3 = 0.292**; phaseless composed
**S1 = 0.833 (BALF) / 0.774 (GALF)**; joint-loss composed **S1 = 0.771 / 0.753**. The MLIP over-drives
the S2→S1 funnel, piling population onto S1 and draining S3. The Tier-1 sign-consistent loss moves S1 in
the right direction (0.833 → 0.771) but does not close the ~0.17 gap.

## 6. Keto over-relaxation: what is understood

The residual keto discrepancy has been attributed through controlled tests:

- **Not survival.** Re-harvesting over the reference's exact surviving initial conditions changes S1 by
  ~0.005 — the reference's eigensolve deaths are not the over-relaxers.
- **Not the gap.** Model A's near-CI gap is accurate (Section 2); the excess is in the Model B coupling.
- **Not Model A capacity.** nf172 does not improve the gap.
- **Coupling magnitude + sign.** S1–S2 is 99.3% of keto hops; the honest phaseless coupling over-estimates
  its magnitude by ~16%, and the per-pair phaseless sign resolution can be pair-inconsistent (a
  "pair-scramble") that the driver's per-state phase match cannot repair. Both push populations the same
  way. Enol tolerates both because its S1–S2 gap is well separated; keto does not, because of its
  near-degenerate carbonyl pair.

The indicated fix is a Model B that is sign-consistent **without** the joint loss's magnitude inflation
(a structural per-state phase carrier), together with active-learning calibration of the S1–S2 coupling
magnitude at the near-CI seam. This is the open line of work.

## 7. Caveats when comparing to the reference

- **TDC method differs by construction.** The reference uses `hamiltonian_fd`; the MLIP uses `nac_dot_v`
  (it has no CIS amplitudes). Validated harmless at the ensemble level, but note it.
- **Survival asymmetry.** The reference loses trajectories to near-CI eigensolve deaths that the MLIP
  does not; always compare survivor-conditional populations and report the differing survivor counts.
- **Tautomer scope.** This repository's datasets and the validated pipeline target the **enol** form; keto
  is the harder benchmark and the current research frontier.
- **Configuration coverage** predicts where the MLIP is extrapolating (enol FSSH samples ~70% out of the
  adiabatic training distribution, keto ~8%) but is a non-causal proxy — the causal quantity is the
  coupling calibration at hop frames, not coverage of rare geometry.

## 8. Bottom line

- Model A (E+F) and NVE energy conservation: **validated**, both tautomers.
- FSSH@MLIP end-to-end: **validated for enol** against the AM1/CIS reference.
- FSSH@MLIP for keto: **over-relaxes ~2× on S1**, with a characterized cause (coupling-magnitude +
  sign at the near-degenerate S1–S2 seam) and an identified path (structural sign consistency +
  coupling-magnitude active learning). Tracked as the open problem.

See also: `inference_benchmark_report.md` (throughput), `known_limitations.md` (open issues),
`model_b_math_background.md` (the coupling), and the FSSH guides.
