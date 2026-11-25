# Migration and refactoring changelog

How the MLIP-for-NAMD code was packaged into this repository and prepared for public, generic use.
Guiding constraint: **preserve numerical and scientific behavior exactly** while improving clarity,
documentation, reproducibility, and environment-independence. The upstream HIPPYNN library tree is
unmodified; all project code lives under `ml_namd/`.

## Packaging
- A standalone copy of `lanl/hippynn` with all project code isolated under a top-level `ml_namd/` folder,
  so the upstream tree stays untouched and a future upstream merge is clean.
- The canonical pipeline scripts (training, data generation, FSSH@MLIP, FSSH@PYSEQM, NVE@MLIP, analysis)
  were staged, then refactored additively: full docstrings, a developer header, type hints, and intent
  comments, **preserving public names, CLI/config keys, `acn_` array names, and all numerics**. This is
  mandatory because HIPPYNN node/loss class names and module filenames are pickled into
  `experiment_structure.pt` at training time and resolved when a model is loaded for FSSH; renaming any of
  them would break existing trained checkpoints.

## Behavior-preservation checks
- **Pickled node/loss files** (the dENACR heads and the joint loss): seed-matched instantiation of the
  original and refactored class, then `state_dict` and `forward()` compared on random inputs -- required
  max |delta| = 0.
- **Other Python files:** an AST-equivalence check (parse both, strip docstrings/annotations/metadata,
  compare the normalized AST) -- logically identical.

## Vendored runtime dependencies
- The FSSH@MLIP **predictor bundle** (`scripts/fssh_mlip/enol_ef_nacr_developer_handoff/src/`) is vendored
  so the composed driver is runnable-complete. The separate PySEQM data-generation/labeling fork is larger
  and is referenced on `PYTHONPATH` rather than vendored.

## Generic / public-use pass
- Author- and environment-specific details were removed so the repository is independent of any one
  machine: batch-scheduler scripts were replaced by plain-`python` example steps plus a single generic
  template under `optional_hpc/`; raw environment captures were replaced by a curated `env/requirements.txt`
  with a generic `env/README.md` and `verify_env.py`; dataset locations now point to the Harvard Dataverse
  record (see `data_manifests/DATA_MANIFEST.md`). A `tools/check_no_cluster_info.py` gate enforces this.

## Documentation
Under `ml_namd/docs/`: the Model B math background + usage guide; Model A, NVE@MLIP, FSSH@MLIP, and
FSSH@PYSEQM guides; the installation/environment guide; the running guide; validation and
inference-benchmark reports; this changelog; and `known_limitations.md`. Plus the top-level `README.md`,
per-group READMEs, and the data manifest.
