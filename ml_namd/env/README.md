# Environment reproducibility

The ml_namd pipeline needs a specific CUDA-enabled scientific Python stack. This folder records the
pinned versions and provides a checker. It is deliberately environment-independent: recreate it on any
machine that has an NVIDIA GPU.

## Files
- `requirements.txt` — the key pinned dependencies (install **after** the CUDA PyTorch wheel; see below).
- `verify_env.py` — checks the active interpreter against the pins and confirms the load-bearing imports
  (`torch` + CUDA build, `numpy`, `hippynn`, `seqm`).

## Pinned stack
- Python **3.13.5**
- PyTorch **2.7.1+cu118** (CUDA build **11.8**)
- NumPy 2.x (minor version is environment-dependent; informational)
- HIPPYNN **editable**, git commit **`d260cd5`**
- PySEQM **editable**, git commit **`e273f6d`** (the master used by the FSSH@PYSEQM reference engine)

A CUDA-enabled stack is **not** reproducible from `pip install -r` alone: the CUDA toolkit, the matching
PyTorch wheel, and the editable HIPPYNN/PySEQM commits must all line up. Follow the ordered steps in
`../docs/installation_and_environment.md`.

## Recreate + verify (summary)
1. Create a clean environment with Python 3.13.5 (conda / micromamba / venv).
2. Install PyTorch first from the CUDA 11.8 wheel index.
3. `pip install -r requirements.txt`.
4. Install HIPPYNN and PySEQM editable at the pinned commits.
5. `python verify_env.py` — treat the environment as reproduced only after this passes **and** a short
   training smoke + FSSH smoke complete.

Set `HIPPYNN_USE_CUSTOM_KERNELS=False` for the float64 MD/inference path.
