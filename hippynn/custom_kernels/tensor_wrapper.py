"""
Tools for converting between torch and numba-compatible arrays
"""
import functools

import numba
import torch
from .fast_convert import batch_convert_torch_to_numba


def via_numpy(func):
    """
    Decorator for piping a function through
    numpy arrays, and then giving the result back to torch.
    """
    #  Dev note: A bit of non-rigorous testing showed that this adds overhead
    #  on the order of microseconds.

    @functools.wraps(func)
    def wrapped(*args):
        args = [a.data.numpy() for a in args]
        result = func(*args)
        if not isinstance(result, (tuple, list)):
            return torch.as_tensor(result)
        return tuple(torch.as_tensor(r) for r in result)

    return wrapped


def _numba_gpu_not_found(*args, **kwargs):
    raise RuntimeError(
        "Error: Numba not configured to run on GPU.\n"
        "numba.cuda.is_available() returned False; numba was not able to find a GPU.\n"
        "Verify that your numba installation is able to find cuda toolkit, as this \n"
        "error condition likely indicates that torch can find the GPU, but numba can't.\n"
        "Alternatively, disable custom kernels."
    )


class NumbaCompatibleTensorFunction:
    def __init__(self):
        if numba.cuda.is_available():
            self.kernel64 = None
            self.kernel32 = None
        else:
            self.kernel64 = _numba_gpu_not_found
            self.kernel32 = _numba_gpu_not_found

    def __call__(self, *args, **kwargs):
        shapes = [x.shape for x in args]
        dev = args[0].device

        if dev.type == "cpu":
            return self.cpu_kernel(*args)
        else:  # GPU
            dtype = args[0].dtype
            launch_bounds = self.launch_bounds(*shapes)
            out = torch.zeros(self.out_shape(*shapes), device=dev, dtype=dtype)
            args = *args, out
            args = [a.detach() for a in args]
            with numba.cuda.gpus[dev.index]:
                numba_args = batch_convert_torch_to_numba(*args)
                if dtype == torch.float64:
                    if self.kernel64 is None:
                        self.kernel64 = self.make_kernel(numba.float64)
                    self.kernel64[launch_bounds](*numba_args)
                elif dtype == torch.float32:
                    if self.kernel32 is None:
                        self.kernel32 = self.make_kernel(numba.float32)
                    self.kernel32[launch_bounds](*numba_args)
                else:
                    raise ValueError("Bad dtype: {}".format(dtype))
        return args[-1]

    def make_kernel(self):
        return NotImplemented

    def out_shape(self, *shapes):
        return NotImplemented

    def launch_bounds(self, *shapes):
        return NotImplemented

    def cpu_kernel(self):
        return NotImplemented
