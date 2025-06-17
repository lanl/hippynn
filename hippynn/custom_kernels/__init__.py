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

See the :doc:`/user_guide/ckernels` section of the documentation for more information.

Depending on your available packages, you may have the following options:

* "pytorch": dense pytorch operations.
* "sparse": sparse pytorch operations. Can be faster than pure pytorch for large
  enough systems, and will not require as much memory.
  May require latest pytorch version. Cannot cover all circumstances; should error
  if encountering a result that this implementation cannot cover..
* "numba": numba implementation of custom kernels, beats pytorch-based kernels.
* "cupy": cupy implementation of custom kernels, better than numba
* "triton": triton-based custom kernels, uses auto-tuning and the triton compiler.
  This is usually the best option.

The available items are stored in the variable :data:`hippynn.custom_kernels.CUSTOM_KERNELS_AVAILABLE`.
The active implementation is stored in :data:`hippynn.custom_kernels.CUSTOM_KERNELS_ACTIVE`.

For more information, see :doc:`/user_guide/ckernels`

"""
# Dev notes:
# For a new implmentation, make a new MessagePassingKernels object in your implementation file.
# This will register your implementation with the system.
# Then do the following:
#   - Add your implementation name to _POSSIBLE_CUSTOM_KERNELS
#   - Add an import block for the file to the populate_custom_kernels() function.
# If your custom kernel has constraints on what devices or hardware configurations are possible,
# the have your module raise an ImportError. You may also want to use
# warnings.warn to warn the user of why the implementation is not available.
import warnings
from typing import Union
import torch
from .. import settings
from .registry import MessagePassingKernels
from . import env_pytorch



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

    # Look for CPU-capable kernels.
    try:
        import numba
        from . import env_numba
        from . import env_atomic
    except ImportError:
        pass

    try:
        from . import env_sparse
    except ImportError:
        pass

    # these kernels only work if torch can get to the GPU.
    if torch.cuda.is_available():
        try:
            import cupy
            from . import env_cupy
        except ImportError:
            pass

        try:
            import triton
            # Note: Device capability check is located in env_triton now.
            from . import env_triton
        except ImportError:
            pass

    CUSTOM_KERNELS_AVAILABLE = list(MessagePassingKernels.get_available_implementations())
    return CUSTOM_KERNELS_AVAILABLE


def set_custom_kernels(active: Union[bool, str] = True) -> str:
    """
    Activate or deactivate custom kernels for interaction.

    This function changes the global variables:
        - :func:`hippynn.custom_kernels.envsum`
        - :func:`hippynn.custom_kernels.sensesum`
        - :func:`hippynn.custom_kernels.featsum`

    Special non-implementation-name values are:
        - True: - Use the best GPU kernel from recommended implementations, error if none are available.
        - False: - equivalent to "pytorch"
        - "auto": - Equivalently to True if recommended is available, else equivalent to "pytorch"

    :param active: implementation name to activate
    :return: active, actual implementation selected.
    """
    populate_custom_kernel_availability()

    global envsum, sensesum, featsum, CUSTOM_KERNELS_ACTIVE

    if active is False:
        active = "pytorch"

    if isinstance(active, str):
        active = active.lower()

    if active not in _POSSIBLE_CUSTOM_KERNELS:
        warnings.warn(f"Using non-standard custom kernel implementation: {active}")

    # Our goal is that this if-block is to handle the cases for values in the range of
    # [True, "auto"] and turn them into the suitable actual implementation.
    if any((impl in CUSTOM_KERNELS_AVAILABLE) for impl in _RECOMMENDED_CUSTOM_KERNELS):
        # If recommended custom kernels are available,
        # then True reverts to "auto" and "False" reverts to "pytorch".
        if active is True:
            active = "auto"
        if active == "auto":
            for impl_case in _RECOMMENDED_CUSTOM_KERNELS:
                if impl_case in CUSTOM_KERNELS_AVAILABLE:
                    active = impl_case
                    break  # exit the for loop, we found the best choice.
    else:
        # In this case, no recommended kernel is available.
        # Use pytorch if active=="auto", and error if active==True.
        if active == "auto":
            warnings.warn(
                "triton, cupy and numba are not available: "
                "Custom kernels will be disabled and performance may be degraded.\n"
                "To silence this warning, set HIPPYNN_USE_CUSTOM_KERNELS=False", stacklevel=2)
            active = "pytorch"
        elif active is True:
            # The user explicitly set a custom kernel implementation to true, but no recommended ones.
            raise CustomKernelError(
                "Triton, numba and cupy were not found. " +
                f"Recommended custom kernels are not available, "
                f"but they were required by library setting: {active}")

    # Ok, finally set the implementation. Note that get_implementation
    # will error with type CustomKernelError if not found.
    kernel_implementation = MessagePassingKernels.get_implementation(active)
    envsum = kernel_implementation.envsum
    sensesum = kernel_implementation.sensesum
    featsum = kernel_implementation.featsum
    CUSTOM_KERNELS_ACTIVE = active

    return active


CUSTOM_KERNELS_AVAILABLE = []  #: List of available kernel implementations based on currently installed packages..

_POSSIBLE_CUSTOM_KERNELS = (
    True,
    False,
    "triton",
    "numba",
    "cupy",
    "pytorch",
    "sparse",
    "auto",  # This means, if possible, use order in _RECOMMENDED_CUSTOM_KERNELS below.
)

# These are in order of preference! If you change the order, you change the default for "auto".
_RECOMMENDED_CUSTOM_KERNELS = (
    "triton",
    "numba",
    "cupy",
)

try_custom_kernels = settings.USE_CUSTOM_KERNELS

CUSTOM_KERNELS_ACTIVE = None  #: Which custom kernel implementation is currently active.

envsum = None  #: See :func:`hippynn.custom_kernels.env_pytorch.envsum` for more information.
sensesum = None  #: See :func:`hippynn.custom_kernels.env_pytorch.sensesum` for more information.
featsum = None  #: See :func:`hippynn.custom_kernels.env_pytorch.featsum` for more information.

try:
    set_custom_kernels(try_custom_kernels)
except CustomKernelError as eee:
    raise  # We re-raise custom kernel releated errors.
except Exception as ee:
    warnings.warn(f"Custom kernels are disabled due to an unexpected error:\n"
                  f"\t{ee}", stacklevel=2)
    del ee
    # Since we don't know what caused the error in the above,
    # let's not re-call the function.
    envsum = env_pytorch.envsum
    sensesum = env_pytorch.sensesum
    featsum = env_pytorch.featsum
    CUSTOM_KERNELS_ACTIVE = False

del try_custom_kernels
