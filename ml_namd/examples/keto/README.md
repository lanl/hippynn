# Keto example

Reproduce the full pipeline for the **keto** tautomer end to end. A CUDA GPU is required for training and
dynamics; no job scheduler is needed (each step is a plain `python` command). Recreate the environment
first via `../../docs/installation_and_environment.md`.

Keto is the harder, near-degenerate case (its S1-S2 carbonyl pair is near-degenerate), run to **10 ps**.
The pipeline is identical to the enol example; only the tautomer, initial-condition prefix, and dynamics
length differ.

## 0. Download the data
See [`00_download_data.md`](00_download_data.md): fetch the keto BALF pool and the 10 ps initial conditions
from the Harvard Dataverse DOI and place them under `data/keto/balf/` and `ics/`.

## Run (one seed shown; repeat steps 2-4 for all 5 seeds to get the sigma_ML ensemble)

```bash
python 01_make_split.py   --seed 42                 # optional: preview the stratified split
python 02_train_model_A.py --seed 42                # Model A (E+F) + genA pool
python 03_train_model_B.py --seed 42 --head both    # Model B (hybrid head)
python 04_run_ml_tsh.py    --seed 42 --head both    # composed ML-TSH (per-IC seeded)
# ... repeat 02-04 for seeds 101726 102593 112793 112891 ...
python 05_harvest.py       --head both              # 5-seed mean +/- sigma_ML populations
```

The main result is the **`both`** (hybrid) head over the 5 seeds [42, 101726, 102593, 112793, 112891];
`gdv_only` / `geom_only` are the ablation heads. Keto dynamics run to **10 ps** (longer than enol).

## Layout
- `01..05_*.py` -- the ordered pipeline steps (overview in [`../README.md`](../README.md)).
- `_common.py` -- paths + keto settings (tautomer, IC prefix, dynamics length).
- `data/`, `ics/`, `runs/`, `out/` -- NOT in git: `data/` and `ics/` come from the Dataverse DOI; `runs/`
  and `out/` are created by the steps.
