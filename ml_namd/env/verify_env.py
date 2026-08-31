#!/usr/bin/env python3
"""Verify that the active Python environment matches the pinned reproducibility record.

Scientific purpose:
    The MLIP-for-NAMD pipeline (Model A/B training, NVE@MLIP, FSSH@MLIP, FSSH@PYSEQM) depends on a
    specific CUDA-enabled scientific stack. This script checks the running interpreter against the
    pinned key versions captured in ``ml_namd/env/`` and confirms the load-bearing imports succeed,
    so a regenerated environment can be accepted (or rejected) deterministically.

Usage:
    python verify_env.py [--env {chicoma_venv_master,darwin_venv_master,darwin_hiphopnn_env}]

Exit status:
    0  all pinned checks passed
    1  one or more checks failed (details printed)

Notes:
    - GPU/driver is reported when available but is NOT a hard requirement here (login nodes lack a GPU).
    - HIPPYNN and PYSEQM are pinned by git commit in the records, not by the PyPI version string; this
      script checks importability and prints the reported versions for a human to compare to the record.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass, field

# Pinned expectations shared by all three recorded environments (2026-08-31 capture).
PINNED_PYTHON: str = "3.13.5"
PINNED_TORCH: str = "2.7.1+cu118"
PINNED_TORCH_CUDA: str = "11.8"

# Per-environment values that legitimately differ (numpy minor version).
ENV_NUMPY: dict[str, str] = {
    "chicoma_venv_master": "2.5.0",
    "darwin_venv_master": "2.4.1",
    "darwin_hiphopnn_env": "2.4.1",
}


@dataclass
class CheckResult:
    """Outcome of a single environment check."""

    name: str
    ok: bool
    detail: str


def _check_python() -> CheckResult:
    running = ".".join(str(n) for n in sys.version_info[:3])
    return CheckResult("python", running == PINNED_PYTHON, f"running={running} pinned={PINNED_PYTHON}")


def _check_module_version(module_name: str, expected: str | None) -> CheckResult:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # importability is itself the check
        return CheckResult(module_name, False, f"import failed: {type(exc).__name__}: {exc}")
    version = getattr(module, "__version__", "?")
    if expected is None:
        return CheckResult(module_name, True, f"version={version} (informational)")
    return CheckResult(module_name, version == expected, f"version={version} pinned={expected}")


def _check_torch_cuda() -> list[CheckResult]:
    results: list[CheckResult] = []
    try:
        import torch
    except Exception as exc:
        return [CheckResult("torch", False, f"import failed: {type(exc).__name__}: {exc}")]
    results.append(CheckResult("torch", torch.__version__ == PINNED_TORCH,
                               f"version={torch.__version__} pinned={PINNED_TORCH}"))
    results.append(CheckResult("torch.cuda_build", torch.version.cuda == PINNED_TORCH_CUDA,
                               f"cuda_build={torch.version.cuda} pinned={PINNED_TORCH_CUDA}"))
    # GPU availability is reported, never required (login nodes have no GPU).
    available = torch.cuda.is_available()
    detail = f"cuda_available={available}"
    if available:
        detail += f", device_count={torch.cuda.device_count()}"
    results.append(CheckResult("torch.gpu_runtime", True, detail + " (informational)"))
    return results


def run_checks(env_label: str) -> list[CheckResult]:
    """Run every pinned check for the named environment and return the results in order."""
    numpy_expected = ENV_NUMPY.get(env_label)
    results: list[CheckResult] = [_check_python()]
    results.extend(_check_torch_cuda())
    results.append(_check_module_version("numpy", numpy_expected))
    # HIPPYNN + PYSEQM: importability only; commit comparison is a human step against the record.
    results.append(_check_module_version("hippynn", None))
    results.append(_check_module_version("seqm", None))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=sorted(ENV_NUMPY), default="chicoma_venv_master",
                        help="Which recorded environment to check numpy against (default: chicoma_venv_master).")
    args = parser.parse_args()

    print(f"Verifying against pinned record for: {args.env}")
    print(f"Interpreter: {sys.executable}\n")

    results = run_checks(args.env)
    width = max(len(r.name) for r in results)
    all_ok = True
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        all_ok = all_ok and result.ok
        print(f"  [{status}] {result.name.ljust(width)}  {result.detail}")

    print("\nRESULT:", "PASS — environment matches pins" if all_ok else "FAIL — see rows above")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
