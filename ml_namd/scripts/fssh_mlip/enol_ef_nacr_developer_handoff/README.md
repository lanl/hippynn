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

Provenance: `enol_ef_nacr_developer_handoff/src/`,
pulled 2026-08-30 (13 .py files).

Refactor status: the 9 predictor-specific modules (`hiphop_predictor`, `hiphop_predictor_v3`,
`modelb_composed_predictor`, `multistate_ef_inference_hiphop`, `multistate_ef_nacr_inference_hiphop`,
`generate_gdv_data`, `iid_nact_rval`, `ood_nact_bvj`, `smoke_v3_joint`) were given the standard
developer-header dunder block, verified AST-EQUIVALENT to their originals (logic unchanged). The four
training-module copies here (`unified_training.py`, `direct_nacr_head.py`, `_v2.py`, `_v3.py`) are
**left verbatim on purpose**: they are the older functional versions the runtime pickle was built
against (e.g. this `unified_training.py` is 2702 lines vs the refactored 2861-line canonical in
`../../training/`), so they must not be altered. Use `../../training/` as the canonical, refactored
source; these copies exist only so the FSSH runtime resolves its pickled references exactly as on the
cluster.
