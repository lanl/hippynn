"""
This module implements a version of converting from pytorch tensors
to numba DeviceNDArrays that skips much of the indirection that takes place
in the numba implementation.

Note: This is not entirely API safe as numba has
not exposed all of these functions directly.
"""
import numpy as np

from numba.cuda.cudadrv import devices, devicearray

try:
    from numba.cuda.api_util import prepare_shape_strides_dtype
except ImportError:
    # older versions of numba
    from numba.cuda.api import _prepare_shape_strides_dtype as prepare_shape_strides_dtype

from numba.cuda.cudadrv import driver
from numba.core import config

require_context = devices.require_context
current_context = devices.get_context
gpus = devices.gpus

import torch

typedict = {
    torch.complex64: "<c8",
    torch.complex128: "<c16",
    torch.float16: "<f2",
    torch.float32: "<f4",
    torch.float64: "<f8",
    torch.uint8: "|u1",
    torch.int8: "|i1",
    torch.int16: "<i2",
    torch.int32: "<i4",
    torch.int64: "<i8",
}
typedict = {k: np.dtype(typestr) for k, typestr in typedict.items()}


@require_context
def batch_convert_torch_to_numba(*tensors, typedict=typedict):
    out = []

    for tensor in tensors:
        # v2 array interface only

        # CUDA devices are little-endian and tensors are stored in native byte
        # order. 1-byte entries are endian-agnostic.
        dtype = typedict[tensor.dtype]

        shape = tuple(tensor.shape)
        if tensor.is_contiguous():
            # __cuda_array_interface__ v2 requires the strides to be omitted
            # (either not set or set to None) for C-contiguous arrays.
            strides = None
        else:
            itemsize = tensor.storage().element_size()
            strides = tuple(s * itemsize for s in tensor.stride())
        data_ptr = tensor.data_ptr() if tensor.numel() > 0 else 0
        data = (data_ptr, False)  # read-only is false

        shape = shape
        strides = strides

        shape, strides, dtype = prepare_shape_strides_dtype(shape, strides, dtype, order="C")
        size = driver.memory_size_from_info(shape, strides, dtype.itemsize)
        devptr = driver.get_devptr_for_active_ctx(data_ptr)
        data = driver.MemoryPointer(current_context(), devptr, size=size, owner=tensor)
        stream = 0  # No "Numba default stream", not the CUDA default stream
        da = devicearray.DeviceNDArray(shape=shape, strides=strides, dtype=dtype, gpu_data=data, stream=stream)
        out.append(da)

    return out
