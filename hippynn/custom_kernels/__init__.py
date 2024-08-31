"""
Custom Kernels for hip-nn interaction sum.

This module provides implementations in pytorch, numba, and cupy.

Pytorch implementations take extra memory, but launch faster than numba kernels.

Numba kernels use far less memory, but do come with some launching overhead on GPUs.

Cupy kernels only work on the GPU, but are faster than numba.
Cupy kernels require numba for CPU operations.
"""
import warnings
from typing import Union

from .. import settings
from . import autograd_wrapper, env_pytorch

CUSTOM_KERNELS_AVAILABLE = []
try:
    import numba

    CUSTOM_KERNELS_AVAILABLE.append("numba")
except ImportError:
    pass

try:
    import cupy

    if "numba" not in CUSTOM_KERNELS_AVAILABLE:
        warnings.warn("Cupy was found, but numba was not. Cupy custom kernels not available.")
    else:
        CUSTOM_KERNELS_AVAILABLE.append("cupy")
except ImportError:
    pass

try:
    import triton
    import torch 
    device_capability = torch.cuda.get_device_capability()
    if device_capability[0] > 6:
        CUSTOM_KERNELS_AVAILABLE.append("triton")
    else:
        warnings.warn(
            f"Triton found but not supported by GPU's compute capability: {device_capability}"
        )
except ImportError:
    pass

        
except ImportError:
    pass

if not CUSTOM_KERNELS_AVAILABLE:
    warnings.warn(
        "Triton, cupy and numba are not available: Custom kernels will be disabled and performance maybe be degraded.")

CUSTOM_KERNELS_ACTIVE = False

envsum, sensesum, featsum = None, None, None


def _check_numba():
    import numba.cuda
    import torch

    if not numba.cuda.is_available():
        if torch.cuda.is_available():
            warnings.warn("numba.cuda.is_available() returned False: Custom kernels will fail on GPU tensors.")
        return True
    else:
        # atexit.register(numba.cuda.close)
        # Dev note for the future: Do not attempt the above `atexit` call!
        # Causes segfault on program exit on some systems.
        # Probably due to both numba and torch trying to finalize the GPU.
        # Leaving this note here in case anyone is tempted to try it in the future.
        # (At one point this was the right strategy...)
        return False


def _check_cupy():
    import cupy
    import numba
    import torch

    if not cupy.cuda.is_available():
        if torch.cuda.is_available():
            warnings.warn("cupy.cuda.is_available() returned False: Custom kernels will fail on GPU tensors.")
    

def set_custom_kernels(active: Union[bool, str] = True):
    """
    Activate or deactivate custom kernels for interaction.

    :param active: If true, set custom kernels to the best available. If False, turn them off and default to pytorch.
       If "triton", "numba" or "cupy", use those implementations explicitly. If "auto", use best available.
    :return: None
    """
    global envsum, sensesum, featsum, CUSTOM_KERNELS_ACTIVE

    if isinstance(active, str):
        active = active.lower()

    if active not in [True, False, "triton", "numba", "cupy", "pytorch", "auto"]:
        raise ValueError(f"Unrecognized custom kernel implementation: {active}")

    active_map = {"auto": True, "pytorch": False}
    if not CUSTOM_KERNELS_AVAILABLE:
        if active == "auto" or active == "pytorch":
            active = False
        elif active:
            raise RuntimeError(
                "Triton, numba and cupy were not found. Custom kernels are not available, but they were required by library settings.")
    else:
        active = active_map.get(active, active)

    # Handle fallback to pytorch kernels.
    if not active:
        envsum = env_pytorch.envsum
        sensesum = env_pytorch.sensesum
        featsum = env_pytorch.featsum
        CUSTOM_KERNELS_ACTIVE = False
        return

    # Select custom kernel implementation
    if not CUSTOM_KERNELS_AVAILABLE:
        raise RuntimeError("Numba was not found. Custom kernels are not available.")

    if active is True:
        if "triton" in CUSTOM_KERNELS_AVAILABLE:
            active = "triton"
        elif "cupy" in CUSTOM_KERNELS_AVAILABLE:
            active = "cupy"
        else:
            active = "numba"

    if active not in CUSTOM_KERNELS_AVAILABLE:
        raise RuntimeError(f"Unavailable custom kernel implementation: {active}")

    if active == "triton":
        from .env_triton import envsum as triton_envsum, sensesum as triton_sensesum, featsum as triton_featsum

        envsum, sensesum, featsum = autograd_wrapper.wrap_envops(triton_envsum, triton_sensesum, triton_featsum)
    elif active == "cupy":
        _check_numba()
        _check_cupy()
        from .env_cupy import cupy_envsum, cupy_featsum, cupy_sensesum

        envsum, sensesum, featsum = autograd_wrapper.wrap_envops(cupy_envsum, cupy_sensesum, cupy_featsum)
    elif active == "numba":
        _check_numba()
        from .env_numba import new_envsum, new_featsum, new_sensesum

        envsum, sensesum, featsum = autograd_wrapper.wrap_envops(new_envsum, new_sensesum, new_featsum)

    else:
        # We shouldn't get here except possibly mid-development, but just in case:
        # if you add a custom kernel implementation remember to add to this
        # dispatch block.
        raise ValueError(f"Unknown Implementation: {active}")

    CUSTOM_KERNELS_ACTIVE = active


try_custom_kernels = settings.USE_CUSTOM_KERNELS
set_custom_kernels(try_custom_kernels)
del try_custom_kernels
