"""
Custom Kernels for hip-nn interaction sum.

This module provides implementations in pytorch, numba, cupy, and triton.

Pytorch implementations take extra memory, but launch faster than numba kernels.
Numba kernels use far less memory, but do come with some launching overhead on GPUs.
Cupy kernels only work on the GPU, but are faster than numba.
Cupy kernels require numba for CPU operations.
Triton custom kernels only work on the GPU, and are generaly faster than CUPY.
Triton kernels revert to numba or pytorch as available on module import.

On import, this module attempts to set the custom kernels as specified by the
user in hippynn.settings.

.. py:data:: CUSTOM_KERNELS_AVAILABLE
   :type: list[str]

   The set of custom kernels available, based on currently installed packages and hardware.

.. py:data:: CUSTOM_KERNELS_ACTIVE
   :type: str

   The currently active implementation of custom kernels.

"""
import warnings
from typing import Union
import torch
from .. import settings
from . import autograd_wrapper, env_pytorch


class CustomKernelError(Exception):
    pass


def populate_custom_kernel_availability():
    """
    Check available imports and populate the list of available custom kernels.

    This function changes the global variable custom_kernels.CUSTOM_KERNELS_AVAILABLE

    :return:
    """

    # check order for kernels is numba, cupy, triton.
    global CUSTOM_KERNELS_AVAILABLE

    CUSTOM_KERNELS_AVAILABLE = []

    try:
        import numba

        CUSTOM_KERNELS_AVAILABLE.append("numba")
    except ImportError:
        pass

    if torch.cuda.is_available():
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
            if torch.cuda.is_available():
                device_capability = torch.cuda.get_device_capability()
                if device_capability[0] > 6:
                    CUSTOM_KERNELS_AVAILABLE.append("triton")
                else:
                    warnings.warn(
                        f"Triton found but not supported by GPU's compute capability: {device_capability}"
                    )
        except ImportError:
            pass


    if not CUSTOM_KERNELS_AVAILABLE:
        warnings.warn(
            "Triton, cupy and numba are not available: Custom kernels will be disabled and performance maybe be degraded.")
    return CUSTOM_KERNELS_AVAILABLE

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
    return

def set_custom_kernels(active: Union[bool, str] = True):
    """
    Activate or deactivate custom kernels for interaction.

    This function changes the global variables:
        - custom_kernels.envsum
        - custom_kernels.sensum
        - custom_kernels.featsum
        - custom_kernels.CUSTOM_KERNELS_ACTIVE

    :param active: If true, set custom kernels to the best available. If False, turn them off and default to pytorch.
       If "triton", "numba" or "cupy", use those implementations explicitly. If "auto", use best available.
    :return: None
    """
    global envsum, sensesum, featsum, CUSTOM_KERNELS_ACTIVE

    if isinstance(active, str):
        active = active.lower()

    if active not in _POSSIBLE_CUSTOM_KERNELS:
        raise CustomKernelError(f"Unrecognized custom kernel implementation: {active}")

    if not CUSTOM_KERNELS_AVAILABLE:
        if active in ("auto", "pytorch"):  # These are equivalent to "false" when custom kernels are not available.
            active = False
        elif active:
            # The user explicitly set a custom kernel implementation or just True.
            raise CustomKernelError(
                "Triton, numba and cupy were not found." +
                f"Custom kernels are not available, but they were required by library setting: {active}")
    else:
        # If custom kernels are available, then "auto" and "pytorch" revert to bool values.
        active_map = {"auto": True, "pytorch": False}
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
        raise CustomKernelError("Numba was not found. Custom kernels are not available.")

    if active is True:
        if "triton" in CUSTOM_KERNELS_AVAILABLE:
            active = "triton"
        elif "cupy" in CUSTOM_KERNELS_AVAILABLE:
            active = "cupy"
        else:
            active = "numba"

    if active not in CUSTOM_KERNELS_AVAILABLE:
        raise CustomKernelError(f"Unavailable custom kernel implementation: {active}")

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
        raise CustomKernelError(f"Unknown Implementation: {active}")

    CUSTOM_KERNELS_ACTIVE = active
    return

CUSTOM_KERNELS_AVAILABLE = []

_POSSIBLE_CUSTOM_KERNELS = [True, False, "triton", "numba", "cupy", "pytorch", "auto"]

try_custom_kernels = settings.USE_CUSTOM_KERNELS

CUSTOM_KERNELS_ACTIVE = None

envsum, sensesum, featsum = None, None, None

try:
    populate_custom_kernel_availability()
    set_custom_kernels(try_custom_kernels)
except CustomKernelError as eee:
    raise
except Exception as ee:
    warnings.warn(f"Custom kernels are disabled due to an expected error:\n\t{ee}", stacklevel=2)
    del ee

    envsum = env_pytorch.envsum
    sensesum = env_pytorch.sensesum
    featsum = env_pytorch.featsum
    CUSTOM_KERNELS_ACTIVE = False

del try_custom_kernels
