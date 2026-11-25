# Installation and environment guide

Step-by-step setup for the Python environment the pipeline uses. The pinned versions and a checker live
in `../env/` (`requirements.txt`, `verify_env.py`).

CUDA-enabled scientific stacks are **not** reproducible from `pip install -r` alone: the CUDA toolkit,
the matching PyTorch wheel, and the editable HIPPYNN/PySEQM commits must all line up. Follow the order
below.

---

## 1. Pinned stack
- Python **3.13.5**
- PyTorch **2.7.1+cu118** (CUDA build **11.8**)
- NumPy 2.x (minor version environment-dependent)
- HIPPYNN **editable**, git commit **`d260cd5`**
- PySEQM **editable**, git commit **`e273f6d`**

## 2. Prerequisites
- An NVIDIA GPU with a driver new enough for the CUDA 11.8 runtime (the toolkit ships with the wheel;
  only the driver is external). Check it on a GPU host with
  `nvidia-smi --query-gpu=name,driver_version --format=csv`.
- `conda` (or `micromamba`) and `git`.
- Source checkouts of HIPPYNN (this repository is itself the HIPPYNN checkout) and PySEQM.

## 3. Recreate the environment

**3.1 Base interpreter.**

```bash
conda create -n ml_namd python=3.13.5 -y
conda activate ml_namd
```

**3.2 Install the CUDA PyTorch wheel FIRST (cu118 index).**

```bash
pip install torch==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

Do this before anything else so the rest of the stack resolves against the correct torch build.

**3.3 Install the remaining pinned packages.**

```bash
pip install -r ../env/requirements.txt
```

**3.4 Install HIPPYNN editable at the pinned commit.** This repository is the HIPPYNN checkout:

```bash
# from the repository root
git checkout d260cd5        # or your working branch, which is based on it
pip install -e .
```

If you keep HIPPYNN elsewhere, `git clone`, `git checkout d260cd5`, `pip install -e .` there instead.

**3.5 Install PySEQM editable at the pinned commit.**

```bash
git clone https://github.com/lanl/PYSEQM.git pyseqm && cd pyseqm
git checkout e273f6d
pip install -e .
cd -
```

**3.6 Verify.**

```bash
python ../env/verify_env.py
```

A pass confirms Python/torch/CUDA match the pins and that `torch`, `numpy`, `hippynn`, and `seqm` import.
Then run a short **training smoke** and an **FSSH smoke** (the example steps accept a small count) before
trusting the environment for real work.

## 4. Environment variables

| Variable | Value | Why |
|---|---|---|
| `HIPPYNN_USE_CUSTOM_KERNELS` | `False` | float64 graphs; the triton kernels are not float64 paths and the systems are tiny (15 atoms) |
| `OMP_NUM_THREADS` (and `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`) | `1` | one FSSH@PYSEQM reference trajectory per physical core (no oversubscription) |

## 5. Notes
- **Older HIPPYNN cannot import the coupling-head node files.** The graph-node classes in
  `../scripts/training/direct_nacr_head_v3.py` reference `IdxType.Systems`, which is absent in older
  HIPPYNN; use the pinned commit. Pure-torch pieces (e.g. the loss in `joint_state_nacr_loss.py`, whose
  hippynn node is guarded) import anywhere.
- **A GPU is required for training and dynamics**, not for inspection. A machine without a GPU can run
  only CPU smokes and analysis.
- If you use the separate PySEQM data-generation / labeling fork, put it on `PYTHONPATH` (it is a larger
  fork used only for generating training labels; it is not part of this repository).

## 6. Troubleshooting
- **`pip install -r` gives a CPU torch.** You installed the requirements before the CUDA wheel; reinstall
  torch from the cu118 index (3.2).
- **`AttributeError: type object 'IdxType' has no attribute 'Systems'`.** Your HIPPYNN is older than the
  pin; use the pinned commit (3.4).
- **`ModuleNotFoundError` when loading a trained Model B.** The pickled graph references
  `direct_nacr_head_v3` (and `joint_state_nacr_loss` for joint-trained models); ensure both are on
  `sys.path` before `torch.load` (the FSSH driver's bootstrap does this).
- **`Can't initialize NVML` / `cuda_available=False` on a headless host.** No GPU in that shell; run GPU
  work on a GPU node.
- **Amplitude NaNs / frozen hopping in FSSH@MLIP.** Almost always a gap floor; the coupling division must
  use the raw gap with no floor (see `../scripts/fssh_mlip/hippynn_fssh.py`).

See also: `../env/README.md` and `model_b_usage_guide.md`.
