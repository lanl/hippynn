"""
Custom Kernels for hip-nn interaction sum.

This module provides implementations in pytorch and numba.

Pytorch implementations take extra memory, but launch faster than numba kernels.

Numba kernels use far less memory, but do come with some launching overhead on GPUs.
"""
import warnings
from .. import settings

from . import autograd_wrapper, env_pytorch

try:
    import numba

    _CUSTOM_KERNELS_AVAILABLE = True
except ImportError:
    _CUSTOM_KERNELS_AVAILABLE = False
    warnings.warn("Numba not available: Custom Kernels will be disabled.")

_CUSTOM_KERNELS_ACTIVE = False
_INITIALIZED = False

envsum, sensesum, featsum = None, None, None


def _initialize():
    global _INITIALIZED
    import numba.cuda
    import torch

    if not numba.cuda.is_available():
        if torch.cuda.is_available():
            warnings.warn("numba.cuda.is_available() returned False: Custom kernels will fail on GPU tensors.")
    else:
        # atexit.register(numba.cuda.close)
        # Note: Do not attempt the above `atexit` call!
        # Causes segfault on program exit on some systems.
        # Probably due to both numba and torch trying to finalize the GPU.
        # Leaving this note here in case anyone is tempted to try it in the future.
        pass

    _INITIALIZED = True


def set_custom_kernels(active: bool = True):
    """
    Activate or deactivate custom kernels for interaction.

    :param active: If true, set custom kernels on. If False, turn them off and default to pytorch.
    :return: None
    """
    global envsum, sensesum, featsum, _CUSTOM_KERNELS_ACTIVE

    if active:
        if not _CUSTOM_KERNELS_AVAILABLE:
            raise RuntimeError("Numba was not found on module import -- custom kernels are not available.")
        if not _INITIALIZED:
            _initialize()

        from .env_numba import new_envsum, new_sensesum, new_featsum

        (
            envsum,
            sensesum,
            featsum,
        ) = autograd_wrapper.wrap_envops(new_envsum, new_sensesum, new_featsum)

    else:
        envsum = env_pytorch.envsum
        sensesum = env_pytorch.sensesum
        featsum = env_pytorch.featsum

    _CUSTOM_KERNELS_ACTIVE = active


if settings.USE_CUSTOM_KERNELS == "auto":
    try_custom_kernels = _CUSTOM_KERNELS_AVAILABLE
else:
    try_custom_kernels = settings.USE_CUSTOM_KERNELS

set_custom_kernels(try_custom_kernels)
