# Known limitations and unresolved questions

Open issues in the science, the reproducibility record, and the engineering, with pointers to where each
is discussed. This is a living list; resolved items should be moved to the changelog.

---

## Scientific / accuracy

1. **Keto FSSH@MLIP over-relaxes (the headline open problem).** Keto S1 over-populates ~2× versus the
   AM1/CIS reference (S1 ~0.77–0.83 vs 0.61); enol is validated but keto is not. Cause is characterized
   (Section 6 of `validation_report.md`): coupling-magnitude over-estimate + sign inconsistency at the
   near-degenerate S1–S2 carbonyl pair, which is 99.3% of keto hops. The fix is identified but not built.

2. **Sign consistency without magnitude inflation is unsolved.** The joint per-state phase loss fixes the
   sign but, in its pure sum-of-squares form, inflates the S1–S2 coupling magnitude (~1.17 → ~1.73). A
   structural per-state phase-carrier head — sign guarantee without the magnitude side-effect — is the
   proposed remedy and has not been implemented. Whether the residual is the loss *form* or the *gauge*
   is being isolated by a matched-form factorial (arm A/B/C); the result was pending at last update.

3. **S1–S2 coupling-magnitude calibration (active learning) not done.** The honest ~16% S1–S2
   over-estimate at the near-CI seam calls for AL enrichment of the coupling-magnitude labels there; this
   is planned, not executed.

4. **TDC method mismatch is validated only at the ensemble level.** The MLIP uses `nac_dot_v` (it has no
   CIS amplitudes) while the reference uses `hamiltonian_fd`. Populations agree at the ensemble level, but
   a per-frame equivalence of the two TDCs has not been established.

5. **Tautomer scope.** Only the **enol** form is validated end-to-end. Keto is the harder benchmark and
   the current research frontier; other systems are out of scope for the validated pipeline.

6. **Configuration coverage is a non-causal proxy.** OOD/coverage indicates where the MLIP extrapolates
   but does not predict FSSH accuracy; the causal quantity is the coupling calibration at hop frames. Do
   not diagnose a run by coverage alone.

## Reproducibility / packaging

7. **Vendored predictor bundle is not refactored.** The FSSH@MLIP predictor bundle
   (`scripts/fssh_mlip/enol_ef_nacr_developer_handoff/src/`, 13 modules) is staged verbatim; it has not
   had the additive-documentation pass the owned scripts got. (The FSSH@PYSEQM worker `.sh` scripts
   **have** now had that pass — additive header/comments only, executable lines byte-identical — so they
   are no longer verbatim; see the changelog.)

8. **The PySEQM-ml_namd data-generation fork is not vendored.** It is a full PySEQM fork, referenced on
   `PYTHONPATH` by the data-gen slurms and documented, but not included in this repo (too large).

9. **Darwin PySEQM commit is unpinned** (cluster-only close-out staged). The Darwin `venv_master`
   `master_src` checkout had no `.git`, so its exact commit is unverified (project memory records
   `107f5b0`); the Chicoma master is pinned (`e273f6d`). A runnable content-match recipe (clone master,
   `diff -rq` the tree against `master_src` at each candidate commit) is staged in `../env/README.md`
   under "Pending capture"; run it on Darwin to resolve the pin. The reference method
   (`hamiltonian_fd` + `decohere_on_hop=True`) is fixed regardless, so this blocks exact pinning only.

10. **GPU driver version is not captured** (cluster-only close-out staged). The environment records were
    taken on login nodes without a GPU, so the driver version is absent. The exact one-shot command
    (`nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader`, run inside a GPU
    job on each cluster) is staged in `../env/README.md` under "Pending capture".

11. **Duplicated harvesters — RESOLVED.** The byte-identical `harvest_fssh_full.py` / `harvest_clean.py`
    copies were de-duplicated: the canonical shared versions now live in `scripts/analysis/` (with a
    README) and the `scripts/fssh_pyseqm/` duplicates were removed; both drivers reference the analysis
    copies. See the changelog.

12. **Raw launch ICs are cluster-only; reference survivor arrays md5-pinned** (resolved 2026-08-31 with
    `D:` mounted). A full-drive search confirmed the 1000-IC enol `R0/V0/Z` launch arrays are **not**
    mirrored to the laptop (no `R0.npy`/`V0.npy`/`enol1ps_*`, no `ics_matched/` dir on `D:`, none on
    `C:`); they live only on the clusters, so `data_manifests/DATA_MANIFEST.md` §4 now records that plus
    the cluster-side `md5sum` command. What **is** on `D:` — the reference survivor arrays (§3) — was
    md5-pinned (Python `hashlib.md5`; the laptop's FIPS policy blocks .NET MD5). Only the cluster-side
    launch-IC md5s remain, and they must be taken on the cluster, not the laptop.

## Engineering / environment

13. **float64-only inference.** Inference runs in float64 with custom kernels disabled (the graph is
    float64 for energy-conserving MD). A validated float32/mixed-precision path for the coupling could
    raise throughput but has not been tested for near-CI stability.

14. **Local environment cannot import the node files.** The laptop HIPPYNN is older than the cluster pin
    and lacks `IdxType.Systems`, so the head/node modules and any model using them import only in the
    cluster environment. Behavior verification and smokes must run there; only pure-torch pieces run
    locally.

15. **Amplitude-NaN class of failures.** These are eliminated by the raw-gap-no-floor rule, but any future
    change to the coupling reconstruction must re-check that small/negative MLIP gaps do not re-introduce
    the pole (see `fssh_mlip_guide.md`).

## Where these are tracked

The scientific items (1–6) are the substance of the ongoing FSSH@MLIP investigation and are documented in
`validation_report.md` and `model_b_math_background.md`. The packaging/engineering items (7–15) are the
deferred-cleanup list from the migration (`migration_refactoring_changelog.md`).
