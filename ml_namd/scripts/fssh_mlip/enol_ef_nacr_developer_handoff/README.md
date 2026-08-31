# FSSH@MLIP predictor bundle (vendored)

`src/` is the predictor bundle that the FSSH@MLIP drivers in `../` import at runtime. It is staged
here verbatim from the cluster driver directory so `hippynn_fssh.py` resolves its path assumption
`HANDOFF = <driver_dir>/enol_ef_nacr_developer_handoff` and adds `HANDOFF/src` to `sys.path`.

## What the drivers import from here
- `hiphop_predictor_v3.HiphopPredictorV3` — v3-capable predictor (auto-detects the GAP node; runs v2
  too); used by `hippynn_fssh.py` (the joint / single-model path).
- `modelb_composed_predictor.ModelBComposedPredictor` — the composed A→B predictor (frozen E+F Model A
  + decoupled NACR Model B); used by `hippynn_fssh_composed.py`.
- Supporting inference/helper modules: `hiphop_predictor.py`, `multistate_ef_inference_hiphop.py`,
  `multistate_ef_nacr_inference_hiphop.py`, plus data/test utilities (`generate_gdv_data.py`,
  `iid_nact_rval.py`, `ood_nact_bvj.py`, `smoke_v3_joint.py`).

## Duplication note (important)
This bundle also carries its OWN copies of `unified_training.py`, `direct_nacr_head.py`,
`direct_nacr_head_v2.py`, and `direct_nacr_head_v3.py`. These are older, pre-refactor copies retained
so the FSSH runtime imports resolve exactly as they did on the cluster (the pooled/joint model pickle
references training symbols via `__main__`). The CANONICAL, refactored versions of those training
modules live in `../../training/`. They are functionally equivalent (the refactor was
documentation-only, AST-verified), so behavior is identical.

Follow-up cleanup (deferred): de-duplicate by pointing the driver's `sys.path` at `../../training/`
instead of vendoring copies here, then remove the duplicated training modules from this bundle. Until
then, treat `../../training/` as canonical and this bundle as a runtime vendoring.

Provenance: `/lustre/scratch5/akhanna2/keto_mlip_10ps/driver/enol_ef_nacr_developer_handoff/src/`,
pulled 2026-08-30 (13 .py files). These predictor modules are NOT yet refactored (Phase 2 covered the
driver scripts in `../`); refactoring them additively is a candidate next step.
