# Installation and environment guide

Step-by-step setup for the environments this project uses. The authoritative, captured pins live in
`../env/` (full records in `../env/records/`, per-environment `pip freeze` in `../env/requirements/`,
and a checker at `../env/verify_env.py`). This guide turns those records into a reproducible install.

CUDA-enabled scientific stacks are **not** reproducible from `pip install -r` alone: the CUDA toolkit,
the matching PyTorch wheel, and the editable HIPPYNN/PySEQM commits must all line up. Follow the order
below.

---

## 1. Which environment for which task

| Environment | Used for | Where | Manager |
|---|---|---|---|
| `venv_master` (Chicoma) | Model A/B **training**, **FSSH@MLIP** (composed), inference | Chicoma | venv on conda `mlip_md_dyn` |
| `venv_master` (Darwin) | **FSSH@PYSEQM** reference | Darwin | venv |
| `hiphopnn_env` (Darwin) | HIPPYNN training on Darwin | Darwin | conda |
| local analysis env | plotting, AST checks, light analysis on the laptop | laptop | conda |

Common pins across the cluster environments (captured 2026-08-31):

- Python **3.13.5**
- PyTorch **2.7.1+cu118** (CUDA build **11.8**)
- NumPy 2.5.0 (Chicoma) / 2.4.1 (Darwin)
- HIPPYNN **editable**, git commit **`d260cd5`**
- PySEQM **editable**, git commit **`e273f6d`** (Chicoma master)

Always confirm exact versions against `../env/records/<env>.txt`; treat the numbers here as the summary.

## 2. Prerequisites

- An NVIDIA GPU with a driver new enough for CUDA 11.8 runtime (the toolkit ships with the wheel; only
  the driver is external). GPU model/driver is not in the captured records because they were taken on
  login nodes — capture it on a GPU node with
  `nvidia-smi --query-gpu=name,driver_version --format=csv`.
- `conda` (or `micromamba`) and `git`.
- Source checkouts of HIPPYNN and PySEQM (this repo is itself the HIPPYNN checkout; PySEQM is separate).

## 3. Recreate the training / FSSH@MLIP environment (the primary env)

This mirrors the Chicoma `venv_master`. Substitute your own paths.

**3.1 Create the base with the recorded Python.**

```bash
conda create -n mlip_md_dyn python=3.13.5 -y
conda activate mlip_md_dyn
```

(The Chicoma layout is a `venv` created `--system-site-packages` on top of this conda env; a plain conda
env works too. Whichever you choose, keep it isolated.)

**3.2 Install the matching CUDA PyTorch wheel FIRST, from the cu118 index.**

```bash
pip install torch==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

Do this before anything else so the rest of the stack resolves against the correct torch build.

**3.3 Install the remaining pip packages** from the captured freeze (drop the `torch` line already
installed in 3.2 to avoid a CPU-wheel downgrade):

```bash
grep -v -i '^torch==' ../env/requirements/chicoma_venv_master.requirements.txt > /tmp/reqs.txt
pip install -r /tmp/reqs.txt
```

**3.4 Install HIPPYNN editable at the pinned commit.** This repository is the HIPPYNN checkout; check
out the pinned commit (or your working branch) and install editable:

```bash
# from the repo root
git checkout d260cd5        # or your ml_namd branch, which is based on it
pip install -e .
```

If you keep HIPPYNN elsewhere, `git clone`, `git checkout d260cd5`, `pip install -e .` there instead.

**3.5 Install PySEQM editable at the pinned commit.**

```bash
git clone https://github.com/lanl/PYSEQM.git pyseqm && cd pyseqm
git checkout e273f6d        # the Chicoma master pin
pip install -e .
cd -
```

**3.6 Verify.**

```bash
python ../env/verify_env.py --env chicoma_venv_master
```

A pass confirms Python/torch/CUDA/numpy match the pins and that `torch`, `numpy`, `hippynn`, and `seqm`
import. Then run a **training smoke** (2 epochs on a small dataset) and an **FSSH smoke**
(`../scripts/slurm/keto_s0s3_jointB_smoke.slurm`, 5 ICs) before trusting the environment for real work.
Only after both pass should any existing environment file be replaced.

## 4. The Darwin environments

Same PyTorch/HIPPYNN pins; the differences are the manager and the roles.

- **`venv_master` (FSSH@PYSEQM reference).** A `venv` on Darwin (RHEL 8.10). Recreate as in Section 3 but
  from `../env/requirements/darwin_venv_master.requirements.txt` (NumPy 2.4.1). PySEQM here is a master
  checkout used only for the reference driver (`../scripts/fssh_pyseqm/run_fssh_ref.py`); the captured
  `master_src` had no `.git`, so pin its commit explicitly when you rebuild.
- **`hiphopnn_env` (Darwin training).** A conda env for HIPPYNN training on Darwin; recreate from
  `../env/requirements/darwin_hiphopnn_env.requirements.txt`.

## 5. Local analysis environment (laptop)

For plotting, dataset inspection, and the AST-equivalence checks used during refactoring, a light conda
env with `numpy`, `h5py`, `matplotlib`, `torch` (CPU is fine), and a HIPPYNN checkout is enough. Note
the local HIPPYNN may be an **older** build than the cluster pin — this matters for one thing:

> The hippynn graph-node classes in `../scripts/training/direct_nacr_head_v3.py` reference
> `IdxType.Systems`, which is absent in older HIPPYNN. Importing those node files (or a model that uses
> them) therefore fails locally and must be done in the cluster environment. Pure-torch pieces (e.g.
> the loss module in `joint_state_nacr_loss.py`, whose hippynn node is guarded) do run locally.

## 6. HPC / SLURM notes

- The SLURM scripts activate their environment explicitly. The NVE and data-generation slurms
  `source` a conda profile and `conda activate .../mlip_md_dyn`; the FSSH slurms call the `venv_master`
  interpreter directly. Keep those activation lines pointing at your environment.
- Threading is pinned for CPU FSSH@PYSEQM (`OMP_NUM_THREADS=1`, one FSSH trajectory per physical core);
  do not oversubscribe.
- `HIPPYNN_USE_CUSTOM_KERNELS=False` is set for float64 MD/inference — the triton kernels are not
  float64 paths and the systems are tiny (15 atoms), so the dense fallback costs nothing.
- The data-generation slurms add the **PySEQM-ml_namd fork** to `PYTHONPATH` (a separate, larger fork
  used only for labeling; not vendored in this repo). Point `PYTHONPATH` at your copy.

## 7. Environment variables

| Variable | Value | Why |
|---|---|---|
| `HIPPYNN_USE_CUSTOM_KERNELS` | `False` | float64 graphs; avoids the non-float64 triton kernels |
| `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` | `1` | one FSSH@PYSEQM trajectory per core (no oversubscription) |
| `OMP_PROC_BIND`, `OMP_PLACES` | `close`, `cores` | pin threads to cores |
| `PYTHONUTF8`, `LC_ALL`, `LANG` | `1`, `C.UTF-8` | stable text encoding on the clusters |
| `PYTHONPATH` | `.../PYSEQM-ml_namd` | data-generation fork (labeling only) |

## 8. Troubleshooting

- **`pip install -r` gives a CPU torch.** You installed the requirements before the CUDA wheel, or did
  not drop the `torch==` line in 3.3. Reinstall torch from the cu118 index (3.2).
- **`AttributeError: type object 'IdxType' has no attribute 'Systems'`.** Your HIPPYNN is older than the
  pin; use the cluster environment for the node files (Section 5).
- **`ModuleNotFoundError` when loading a trained Model B.** The pickled graph references
  `direct_nacr_head_v3` (and `joint_state_nacr_loss` for joint-trained models); ensure both are on
  `sys.path` before `torch.load` (the FSSH driver's bootstrap does this).
- **`Can't initialize NVML` on a login node.** Expected — there is no GPU on the login node;
  `torch.cuda.is_available()` is False there. Run GPU work in a job.
- **Amplitude NaNs / frozen hopping in FSSH@MLIP.** Almost always a gap floor; the coupling division
  must use the raw gap with no floor (see `../scripts/fssh_mlip/hippynn_fssh.py`).

See also: `../env/README.md` (the captured records) and `model_b_usage_guide.md` (the training +
inference workflow that runs in these environments).
