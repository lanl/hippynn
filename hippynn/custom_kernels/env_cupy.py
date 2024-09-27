"""
Cupy implementation of envsum custom kernels for GPU.
"""
import warnings
import torch
import cupy

if not cupy.cuda.is_available():
    if torch.cuda.is_available():
        warnings.warn("Cupy is installed but cupy.cuda.is_available() returned False. "
                      "Custom kernels will most likely fail on GPU tensors. ")

# If numba is available, this implementation will default to numba on CPU. If not, use vanilla pytorch.
try:
    from .env_numba import new_envsum as envsum_alternative, new_sensesum as sensesum_alternative, new_featsum as featsum_alternative
except ImportError:
    # Load backup implementation for CPU tensors.
    from .env_pytorch import envsum as envsum_alternative, sensesum as sensesum_alternative, featsum as featsum_alternative

from .env_numba import WrappedEnvsum, WrappedSensesum, WrappedFeatsum
from .utils import resort_pairs_cached

from hippynn.custom_kernels import MessagePassingKernels

CUPY_KERNEL_CODE = r"""
extern "C" __global__
void cupy_envsum(
  const FLOATX* sense,
  const FLOATX* feat,
  const long* psecond,
  const long* atom1_ids,
  const long* atom1_starts,
  FLOATX* env,
  long n_nu,
  long n_feat,
  long n_interact
  ) {
  
  int fidx = blockIdx.y*blockDim.x + threadIdx.x;
  int i = blockIdx.x*blockDim.y + threadIdx.y;

  if (i < n_interact){
    long pair_start = atom1_starts[i];
    long pair_end = atom1_starts[i + 1];
    long pfidx = atom1_ids[i];
    for (long nidx=0; nidx<n_nu; nidx++){
      FLOATX out = 0.0;
      for (long pidx=pair_start; pidx<pair_end; pidx++){
        long psidx = psecond[pidx];
        long where_s = pidx * n_nu + nidx;
        long where_f = psidx * n_feat + fidx;
        out += sense[where_s] * feat[where_f];
      }
      long where_e = n_nu * n_feat * pfidx + n_feat * nidx + fidx;
      env[where_e] = out;
    }
  }
}

extern "C" __global__
void cupy_sensesum(
  const FLOATX* env,
  const FLOATX* feat,
  const long* pfirst,
  const long* psecond,
  FLOATX* sense,
  long n_pairs,
  long n_nu,
  long n_feat
) {
  
  long pidx = blockIdx.x*blockDim.x+threadIdx.x;
  long nidx = blockIdx.y*blockDim.y+threadIdx.y;

  if (pidx < n_pairs){
    long pfidx = pfirst[pidx];
    long psidx = psecond[pidx];
    FLOATX tmp = 0.0;
    for (long fidx=0; fidx<n_feat; fidx++){
      long where_e = n_nu*n_feat*pfidx + n_feat*nidx + fidx;
      long where_f = n_feat*psidx + fidx;
      tmp += env[where_e]*feat[where_f];
    }
    long where_s = n_nu*pidx + nidx;
    sense[where_s] = tmp;
  }
}

extern "C" __global__
void cupy_featsum(
  const FLOATX* env,
  const FLOATX* sense,
  const long* pfirst,
  const long* atom2_ids,
  const long* atom2_starts,
  FLOATX* feat,
  long n_nu,
  long n_feat,
  long n_interact
) {
  
  long fidx = blockIdx.y*blockDim.x+threadIdx.x;
  long aidx = blockIdx.x*blockDim.y+threadIdx.y;

  if (aidx < n_interact and fidx < n_feat){
    long pair_start = atom2_starts[aidx];
    long pair_end = atom2_starts[aidx + 1];
    long psidx = atom2_ids[aidx];
    FLOATX tmp = 0.0;
    for (long pidx=pair_start; pidx<pair_end; pidx++){
      long pfidx = pfirst[pidx];
      for (long nidx=0; nidx < n_nu; nidx++){
        long where_e = n_nu*n_feat*pfidx + n_feat*nidx + fidx;
        long where_s = pidx*n_nu+nidx;
        tmp += env[where_e] * sense[where_s];
      }
    }
    long where_f = psidx*n_feat + fidx;
    feat[where_f] = tmp;
  }
}
"""

_CUPY_MODULES = {dtype: cupy.RawModule(code=CUPY_KERNEL_CODE.replace("FLOATX", dtype)) for dtype in ("float", "double")}


def _cupy_gpu_not_found(*args, **kwargs):
    raise RuntimeError(
        "Error: CuPy could not find the GPU."
        "Verify that your numba installation is able to find cuda toolkit, as this \n"
        "error condition likely indicates that torch can find the GPU, but cupy can't.\n"
        "Alternatively, disable custom kernels."
    )


class CupyGPUKernel:
    _cupy_name = None

    def __init__(self):
        if not cupy.cuda.is_available():
            self.kernel32 = _cupy_gpu_not_found
            self.kernel64 = _cupy_gpu_not_found
        else:
            self.kernel32 = _CUPY_MODULES["float"].get_function(self._cupy_name)
            self.kernel64 = _CUPY_MODULES["double"].get_function(self._cupy_name)

    def __call__(self, dtype, BPG, TPB, array_args, shape_args):

        out_array = array_args[-1]
        array_args = [cupy.asarray(a.detach().contiguous()).ravel() for a in array_args]
        args = (*array_args, *shape_args)

        if dtype == torch.float32:
            self.kernel32(BPG, TPB, args)
        elif dtype == torch.float64:
            self.kernel64(BPG, TPB, args)
        else:
            raise ValueError("Bad dtype: {}".format(dtype))

        return out_array


class CupyEnvsum(CupyGPUKernel):
    _cupy_name = "cupy_envsum"

    def __call__(self, sense, feat, pfirst, psecond):
        dev = sense.device
        if dev.type == "cpu":
            return envsum_alternative(sense, feat, pfirst, psecond)

        psecond_hold = psecond
        argsort, atom1_ids, atom1_starts, pfirst, (sense, psecond) = resort_pairs_cached(pfirst, [sense, psecond])
        resort_pairs_cached(psecond_hold, [])

        n_pairs, n_nu = sense.shape
        n_atoms, n_feat = feat.shape
        (n_interact,) = atom1_ids.shape
        dtype = sense.dtype

        env_out = torch.zeros(
            (
                n_atoms,
                n_nu,
                n_feat,
            ),
            device=dev,
            dtype=dtype,
        )
        array_args = sense, feat, psecond, atom1_ids, atom1_starts, env_out
        shape_args = n_nu, n_feat, n_interact

        if n_feat > 512:
            raise ValueError(f"Cupy GPU custom kernels are not compatible with feature sizes greater than 512 (got {n_feat})")

        TPB_MAX = 512
        TPB_X = n_feat
        TPB_Y = TPB_MAX // n_feat
        TPB = (TPB_X, TPB_Y)
        BPG_X = (n_interact + TPB_Y - 1) // TPB_Y
        BPG_Y = 1
        BPG = (BPG_X, BPG_Y)

        args = *array_args, *shape_args
        return super().__call__(dtype, BPG, TPB, array_args, shape_args)


class CupySensesum(CupyGPUKernel):
    _cupy_name = "cupy_sensesum"

    def __call__(self, env, feat, pfirst, psecond):
        dev = env.device
        if dev.type == "cpu":
            return sensesum_alternative(env, feat, pfirst, psecond)

        (n_pairs,) = pfirst.shape
        n_atoms, n_nu, n_feat = env.shape
        dtype = env.dtype

        sense_out = torch.zeros((n_pairs, n_nu), device=dev, dtype=dtype)
        array_args = env, feat, pfirst, psecond, sense_out
        shape_args = n_pairs, n_nu, n_feat

        if n_nu > 512:
            raise ValueError(f"Cupy GPU custom kernels are not compatible with sensitivity sizes greater than 512 (got {n_nu})")

        TPB_MAX = 512
        TPB_Y = n_nu
        TPB_X = TPB_MAX // TPB_Y
        TPB = (TPB_X, TPB_Y)
        BPG_X = (n_pairs + TPB_X - 1) // TPB_X
        BPG = (BPG_X, 1)

        return super().__call__(dtype, BPG, TPB, array_args, shape_args)


class CupyFeatsum(CupyGPUKernel):
    _cupy_name = "cupy_featsum"

    def __call__(self, env, sense, pfirst, psecond):
        dev = env.device
        if dev.type == "cpu":
            return featsum_alternative(env, sense, pfirst, psecond)

        pfirst_hold = pfirst
        argsort, atom2_ids, atom2_starts, psecond, (sense, pfirst) = resort_pairs_cached(psecond, [sense, pfirst])
        resort_pairs_cached(pfirst_hold, [])

        (n_pairs,) = pfirst.shape
        n_atoms, n_nu, n_feat = env.shape
        (n_interact,) = atom2_ids.shape
        dtype = env.dtype

        feat_out = torch.zeros((n_atoms, n_feat), device=dev, dtype=dtype)
        array_args = env, sense, pfirst, atom2_ids, atom2_starts, feat_out
        shape_args = n_nu, n_feat, n_interact

        if n_feat > 512:
            raise ValueError(f"Cupy GPU custom kernels are not compatible with feature sizes greater than 512 (got {n_feat})")

        TPB_max = 512
        if n_feat > 32:
            TPB_x = ((n_feat + 31) // 32) * 32
            TPB_y = TPB_max // TPB_x
            BPG_y = 1
        else:
            # If there aren't enough features to really fill a warp, stride them across the blocks instead.
            TPB_x = 1
            TPB_y = TPB_max
            BPG_y = n_feat
        TPB = (TPB_x, TPB_y)
        BPG_x = (n_atoms + TPB_y - 1) // TPB_y
        BPG = (BPG_x, BPG_y)

        return super().__call__(dtype, BPG, TPB, array_args, shape_args)


cupy_envsum = CupyEnvsum()
cupy_sensesum = CupySensesum()
cupy_featsum = CupyFeatsum()

cupy_kernels = MessagePassingKernels(
    "cupy",
    cupy_envsum,
    cupy_sensesum,
    cupy_featsum,
)
