"""
Atomic-operation version of custom kernels.

These kernels are not recommended for actualy use; they exist
for benchmarking purposes.
"""
import functools

import torch
import numba
import numba.cuda
import numpy as np

from .tensor_wrapper import NumbaCompatibleTensorFunction, via_numpy
from .registry import MessagePassingKernels


# conventions:
# pidx  : index of pair
# pfidx : index of first atom in pair (receiver)
# psidx : index of second atom in pair (sender)
# fidx  : index of feature
# nidx  : index of sensitivity (nu)

# Kernel which sums sensitivities and features to get environment.
# Numpy core signature: (p,n),(a,f),(p),(p),(a,n,f)
class WrappedEnvsum(NumbaCompatibleTensorFunction):
    def out_shape(self, sense_shape, feat_shape, *other_shapes):
        n_pair, n_nu = sense_shape
        n_atom, n_feature = feat_shape
        return n_atom, n_nu, n_feature

    def launch_bounds(self, sense_shape, *other_shapes):
        n_pairs, n_nu = sense_shape
        TPB = 1024
        BPG = (n_pairs + TPB - 1) // TPB
        return BPG, TPB

    @staticmethod
    def make_kernel(KERNEL_DTYPE):
        sig = "void({DTYPE}[:,:,],{DTYPE}[:,:],int64[:],int64[:],{DTYPE}[:,:,:])".format(DTYPE=KERNEL_DTYPE)
        @numba.cuda.jit(sig)
        def kernel(sens, feat, pfirst, psecond, env):
            n_pairs, n_nu = sens.shape
            n_atom, n_feat = feat.shape

            pidx, nidx, fidx = numba.cuda.grid(3)

            if pidx < n_pairs:
                pfidx = pfirst[pidx]
                psidx = psecond[pidx]
                for nidx in range(n_nu):
                    s = sens[pidx, nidx]
                    if abs(s) > 1e-10:
                        for fidx in range(n_feat):
                            out = s * feat[psidx, fidx]
                            numba.cuda.atomic.add(env, (pfidx, nidx, fidx), out)
        return kernel

    @staticmethod
    @via_numpy
    @numba.jit(parallel=True)
    def cpu_kernel(sens, feat, pfirst, psecond):
        n_pairs, n_nu = sens.shape
        n_atom, n_feat = feat.shape
        env_features = np.zeros((n_atom, n_nu, n_feat), dtype=sens.dtype)
        for nidx in numba.prange(n_nu):
            for pidx, (pfidx, psidx) in enumerate(zip(pfirst, psecond)):
                s = sens[pidx, nidx]
                if abs(s) > 1e-10:
                    for fidx in range(n_feat):
                        env_features[pfidx, nidx, fidx] += s * feat[psidx, fidx]
        return env_features


# Kernel which sums environment and features to get sensitivity
#Numpy core signature: (a,n,f),(a,f),(p),(p),(p,n),
class WrappedSensesum(NumbaCompatibleTensorFunction):
    def out_shape(self, env_shape, feat_shape, pfirst_shape, psecond_shape):
        n_pair, = pfirst_shape
        n_atom, n_nu, n_feat = env_shape
        return n_pair, n_nu

    def launch_bounds(self, env_shape, feat_shape, pfirst_shape, psecond_shape):
        n_pairs, = pfirst_shape
        TPB = 1024
        BPG = (n_pairs + TPB - 1) // TPB
        return BPG, TPB

    @staticmethod
    def make_kernel(KERNEL_DTYPE):
        sig = "void({DTYPE}[:,:,:],{DTYPE}[:,:],int64[:], int64[:],{DTYPE}[:,:])".format(DTYPE=KERNEL_DTYPE)
        @numba.cuda.jit(sig)
        def kernel(env, feat, pfirst, psecond, sense):
            n_pairs, = pfirst.shape
            n_atom, n_nu, n_feat = env.shape
            pidx, nidx, fidx = numba.cuda.grid(3)

            if pidx < n_pairs:
                pfidx = pfirst[pidx]
                psidx = psecond[pidx]
                for nidx in range(n_nu):
                    tmp = 0.
                    for fidx in range(n_feat):
                        tmp += env[pfidx, nidx, fidx] * feat[psidx, fidx]
                    numba.cuda.atomic.add(sense, (pidx, nidx), tmp)

        return kernel

    @staticmethod
    @via_numpy
    @numba.jit(parallel=True)
    def cpu_kernel(env, feat, pfirst, psecond):
        n_atom, n_nu, n_feat = env.shape
        n_pairs, = pfirst.shape
        sense = np.zeros((n_pairs, n_nu), dtype=env.dtype)
        for nidx in numba.prange(n_nu):
            for pidx in numba.prange(n_pairs):
                pfidx = pfirst[pidx]
                psidx = psecond[pidx]
                for fidx in range(n_feat):
                     sense[pidx, nidx]+=env[pfidx, nidx, fidx] * feat[psidx, fidx]
        return sense

# Kernel which sums environment and sensitivity to get features
#Numpy core signature: (a,n,f),(p,n),(p),(p),(a,f),
class WrappedFeatsum(NumbaCompatibleTensorFunction):
    def out_shape(self, env_shape, sense_shape, pfirst_shape, psecond_shape):
        n_atom, n_nu, n_feature = env_shape
        return n_atom, n_feature

    def launch_bounds(self, env_shape, sense_shape, pfirst_shape, psecond_shape):
        n_pairs, n_nu = sense_shape
        TPB = 1024
        BPG = (n_pairs + TPB - 1) // TPB
        return BPG, TPB

    @staticmethod
    def make_kernel(KERNEL_DTYPE):
        sig = "void({DTYPE}[:,:,:],{DTYPE}[:,:],int64[:],int64[:],{DTYPE}[:,:])".format(DTYPE=KERNEL_DTYPE)

        @numba.cuda.jit(sig)
        def kernel(env, sense, pfirst, psecond, feat):
            n_pairs, n_nu = sense.shape
            n_atom, n_feat = feat.shape

            pidx, nidx, fidx = numba.cuda.grid(3)

            if pidx < n_pairs:
                pfidx = pfirst[pidx]
                psidx = psecond[pidx]
                for fidx in range(n_feat):
                    tmp = 0
                    for nidx in range(n_nu):
                        tmp += env[pfidx, nidx, fidx] * sense[pidx, nidx]
                    numba.cuda.atomic.add(feat, (psidx, fidx), tmp)

        return kernel

    @staticmethod
    @via_numpy
    @numba.jit(parallel=False)
    def cpu_kernel(env, sens, pfirst, psecond):
        n_atom, n_nu, n_feat = env.shape
        n_pairs, = pfirst.shape
        feat = np.zeros((n_atom, n_feat), dtype=sens.dtype)

        for pidx, (pfidx, psidx) in enumerate(zip(pfirst, psecond)):
            for nidx in range(n_nu):
                for fidx in range(n_feat):
                    feat[psidx, fidx] += env[pfidx, nidx, fidx] * sens[pidx, nidx]
        return feat

atomic_envsum = WrappedEnvsum()
atomic_sensesum = WrappedSensesum()
atomic_featsum = WrappedFeatsum()

numba_kernels = MessagePassingKernels(
    "_numba_atomic",
    atomic_envsum,
    atomic_sensesum,
    atomic_featsum,
)
