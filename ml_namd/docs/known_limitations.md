# Known limitations and unresolved questions

Open issues in the science and the engineering, with pointers to where each is discussed. This is a
living list; resolved items move to the changelog.

---

## Scientific / accuracy

1. **Keto S1-S2 coupling calibration.** The near-degenerate keto S1-S2 carbonyl pair (the large majority
   of keto hops) is the hard case: the coupling magnitude is mildly over-estimated and the per-pair
   phaseless sign can be pair-inconsistent. See `validation_report.md`.

2. **Sign consistency without magnitude inflation.** The joint per-state phase loss fixes the sign but, in
   its pure sum-of-squares form, can inflate the S1-S2 coupling magnitude relative to the phaseless blend;
   match the loss form when comparing sign strategies (see `model_b_math_background.md`).

3. **S1-S2 coupling-magnitude active learning.** Enriching the coupling-magnitude labels at the near-CI
   seam is a natural next step.

4. **TDC method mismatch is validated at the ensemble level.** The MLIP uses `nac_dot_v` (no CIS
   amplitudes) while the reference uses `hamiltonian_fd`; populations agree at the ensemble level, but a
   per-frame TDC equivalence has not been established.

5. **Configuration coverage is a non-causal proxy.** OOD/coverage indicates where the MLIP extrapolates
   but does not predict FSSH accuracy; the causal quantity is the coupling calibration at hop frames.

## Reproducibility / packaging

6. **Vendored predictor bundle.** The FSSH@MLIP predictor bundle
   (`scripts/fssh_mlip/enol_ef_nacr_developer_handoff/src/`) is vendored so the composed driver is
   runnable-complete.

7. **Data-generation labeling fork not vendored.** The reference electronic-structure labeling uses a
   larger PySEQM fork, referenced on `PYTHONPATH` rather than included here.

8. **PySEQM commit.** The reference engine is pinned by commit (`e273f6d`); confirm your checkout matches.
   The reference method (`hamiltonian_fd` + `decohere_on_hop=True`) is fixed regardless.

9. **Datasets are external.** Training pools and initial conditions are distributed via the Harvard
   Dataverse record (see `data_manifests/DATA_MANIFEST.md`), not stored in the repository.

## Engineering / environment

10. **float64-only inference.** Inference runs in float64 with custom kernels disabled. A validated
    float32/mixed-precision path for the coupling could raise throughput but has not been tested for
    near-CI stability.

11. **HIPPYNN version for the node files.** The coupling-head node classes reference `IdxType.Systems`,
    absent in older HIPPYNN; use the pinned commit (see `installation_and_environment.md`).

12. **Amplitude-NaN class of failures.** Eliminated by the raw-gap-no-floor rule; any future change to the
    coupling reconstruction must re-check that small/negative MLIP gaps do not re-introduce the pole (see
    `fssh_mlip_guide.md`).
