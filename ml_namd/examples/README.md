# Worked examples

Two complete, self-contained examples that reproduce the pipeline end to end for each acetylacetone
tautomer:

- [`enol/`](enol/README.md) -- the default, clean case (1 ps dynamics).
- [`keto/`](keto/README.md) -- the harder, near-degenerate case (10 ps dynamics).

Each example runs the full chain as plain `python` steps (no job scheduler required; a **CUDA GPU is
required** for training and dynamics):

```
00_download_data.md   fetch this tautomer's BALF pool + initial conditions from the Harvard Dataverse DOI
01_make_split.py      (optional) preview the gap-stratified split's near-CI balance
02_train_model_A.py   train Model A (E+F), then run genA -> the A-predicted GDV/GAP pool for Model B
03_train_model_B.py   train Model B (decoupled dENACR head), reusing Model A's exact split
04_run_ml_tsh.py      composed Model A + Model B FSSH@MLIP (per-IC seeded), after a train-A==deploy-A check
05_harvest.py         guarded harvest of the 5-seed ensemble -> mean +/- sigma_ML populations
```

The data are distributed via the Harvard Dataverse record (DOI `10.7910/DVN/HCL8EK`); see each example's
`00_download_data.md`. Recreate the environment first with `../docs/installation_and_environment.md`.

The two examples share identical step scripts; only `_common.py` differs (tautomer, initial-condition
prefix, and dynamics length). The main result uses the **`both`** (hybrid) Model B head with a 5-seed
ensemble; `gdv_only` and `geom_only` are the ablation heads.
