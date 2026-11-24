#!/usr/bin/env python3
"""Verify that the active Python environment matches the project's pinned stack.

Scientific purpose:
    The ml_namd pipeline (Model A/B training, genA, NVE@MLIP, FSSH@MLIP, and the FSSH@PYSEQM
    reference) depends on a specific CUDA-enabled scientific stack. This script checks the running
    interpreter against the pinned key versions and confirms the load-bearing imports succeed, so a
    recreated environment can be accepted or rejected deterministically.

Usage:
    python verify_env.py

Exit status:
    0  all pinned checks passed
    1  one or more checks failed (details printed)

Notes:
    - A GPU is reported when available but is NOT required here (a headless/login shell may lack one).
    - HIPPYNN and PYSEQM are pinned by git commit (see requirements.txt and the install guide), not by a
      PyPI version string; this script checks importability and prints the reported versions.
"""
from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass

PINNED_PYTHON: str = "3.13.5"
PINNED_TORCH: str = "2.7.1+cu118"
PINNED_TORCH_CUDA: str = "11.8"


@dataclass
class CheckResult:
    """Outcome of a single environment check."""

    name: str
    ok: bool
    detail: str


def _check_python() -> CheckResult:
    running = ".".join(str(n) for n in sys.version_info[:3])
    return CheckResult("python", running == PINNED_PYTHON, f"running={running} pinned={PINNED_PYTHON}")


def _check_module(module_name: str, expected: str | None = None) -> CheckResult:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # importability is itself the check
        return CheckResult(module_name, False, f"import failed: {type(exc).__name__}: {exc}")
    version = getattr(module, "__version__", "?")
    if expected is None:
        return CheckResult(module_name, True, f"version={version} (informational)")
    return CheckResult(module_name, version == expected, f"version={version} pinned={expected}")


def _check_torch_cuda() -> list[CheckResult]:
    try:
        import torch
    except Exception as exc:
        return [CheckResult("torch", False, f"import failed: {type(exc).__name__}: {exc}")]
    results = [
        CheckResult("torch", torch.__version__ == PINNED_TORCH,
                    f"version={torch.__version__} pinned={PINNED_TORCH}"),
        CheckResult("torch.cuda_build", torch.version.cuda == PINNED_TORCH_CUDA,
                    f"cuda_build={torch.version.cuda} pinned={PINNED_TORCH_CUDA}"),
    ]
    available = torch.cuda.is_available()
    detail = f"cuda_available={available}"
    if available:
        detail += f", device_count={torch.cuda.device_count()}"
    results.append(CheckResult("torch.gpu_runtime", True, detail + " (informational)"))
    return results


def run_checks() -> list[CheckResult]:
    """Run every pinned check and return the results in order."""
    results: list[CheckResult] = [_check_python()]
    results.extend(_check_torch_cuda())
    results.append(_check_module("numpy"))     # minor version is environment-dependent
    results.append(_check_module("hippynn"))   # importability; commit pin is a human check
    results.append(_check_module("seqm"))
    return results


def main() -> int:
    print(f"Interpreter: {sys.executable}\n")
    results = run_checks()
    width = max(len(r.name) for r in results)
    all_ok = True
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        all_ok = all_ok and result.ok
        print(f"  [{status}] {result.name.ljust(width)}  {result.detail}")
    print("\nRESULT:", "PASS -- environment matches pins" if all_ok else "FAIL -- see rows above")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
