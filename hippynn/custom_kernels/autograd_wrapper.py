"""
Wraps non-pytorch implementations for use with pytorch autograd.
"""
import torch.autograd

import threading
from contextlib import contextmanager

_DEVICE_CONTEXT_LOCK = threading.Lock()
_DEVICE_TIMEOUT = 30  # if custom kernels have locked for 10s, throw an error


@contextmanager
def _lock_device(tensor):
    """
    This function locks the torch.cuda.device, which affects how
    triton and cupy try to launch their kernels.
    :param tensor:
    :return:
    """

    acquired = _DEVICE_CONTEXT_LOCK.acquire(timeout=_DEVICE_TIMEOUT)

    if not acquired:
        raise TimeoutError(f"Custom kernel device-lock appears deadlocked. (exceeded timeout {_DEVICE_CONTEXT_LOCK})")
    try:
        # Developer note: device_of is safe to CPU tensors, but torch.cuda.device is not!
        with torch.cuda.device_of(tensor):
            yield
    finally:
        _DEVICE_CONTEXT_LOCK.release()


def wrap_envops(envsum_impl, sensesum_impl, featsum_impl):
    """
    :param envsum_impl:  non-autograd implementation of envsum
    :param sensesum_impl:  non-autograd implementation of sensesum
    :param featsum_impl:  non-autograd implementtation of featsum
    :return:
    """

    class AGEnvsum(torch.autograd.Function):
        @staticmethod
        def forward(ctx, sense, feat, pfirst, psecond):
            ctx.save_for_backward(sense, feat, pfirst, psecond)
            if pfirst.shape[0] == 0:
                n_pair, n_nu = sense.shape
                n_atom, n_feat = feat.shape
                if n_pair != 0 or psecond.shape[0] != 0:
                    raise ValueError("Inconsistent shapes for envsum.")
                return torch.zeros((n_atom, n_nu, n_feat), dtype=feat.dtype, device=feat.device)
            with _lock_device(feat):
                env = envsum_impl(sense, feat, pfirst, psecond)
            return env

        @staticmethod
        def backward(ctx, grad_output):
            (
                sense,
                feat,
                pfirst,
                psecond,
            ) = ctx.saved_tensors
            need_gradsense, need_gradfeat, *_ = ctx.needs_input_grad

            need_gradsense = need_gradsense or None
            need_gradfeat = need_gradfeat or None
            grad_sense = need_gradsense and sensesum(grad_output, feat, pfirst, psecond)
            grad_feat = need_gradfeat and featsum(grad_output, sense, pfirst, psecond)

            return grad_sense, grad_feat, None, None

    class AGSensesum(torch.autograd.Function):
        @staticmethod
        def forward(ctx, env, feat, pfirst, psecond):
            ctx.save_for_backward(env, feat, pfirst, psecond)
            if pfirst.shape[0] == 0:
                n_atom0, n_nu, n_feat0 = env.shape
                n_atom1, n_feat1 = feat.shape
                if psecond.shape[0] != 0 or n_atom0 != n_atom1 or n_feat0 != n_feat1:
                    raise ValueError("Inconsistent shapes for sensesum")
                return torch.zeros((0, n_nu), dtype=feat.dtype, device=feat.device)
            with _lock_device(feat):
                sense = sensesum_impl(env, feat, pfirst, psecond)
            return sense

        @staticmethod
        def backward(ctx, grad_output):
            env, feat, pfirst, psecond = ctx.saved_tensors
            needs_gradenv, needs_gradfeat, *_ = ctx.needs_input_grad
            needs_gradenv = needs_gradenv or None
            needs_gradfeat = needs_gradfeat or None
            gradenv = needs_gradenv and envsum(grad_output, feat, pfirst, psecond)
            gradfeat = needs_gradfeat and featsum(env, grad_output, pfirst, psecond)
            return gradenv, gradfeat, None, None

    class AGFeatsum(torch.autograd.Function):
        @staticmethod
        def forward(ctx, env, sense, pfirst, psecond):
            ctx.save_for_backward(env, sense, pfirst, psecond)
            if pfirst.shape[0] == 0:
                n_atom, n_nu0, n_feat = env.shape
                n_pair, n_nu1 = sense.shape
                if psecond.shape[0] != 0 or n_nu0 != n_nu1:
                    raise ValueError("Inconsistent shapes for featsum")
                return torch.zeros((n_atom, n_feat), dtype=env.dtype, device=env.device)
            with _lock_device(env):
                feat = featsum_impl(env, sense, pfirst, psecond)
            return feat

        @staticmethod
        def backward(ctx, grad_output):
            env, sense, pfirst, psecond = ctx.saved_tensors
            needs_gradenv, needs_gradsense, *_ = ctx.needs_input_grad
            needs_gradenv = needs_gradenv or None
            needs_gradsense = needs_gradsense or None
            gradgenv = needs_gradenv and envsum(sense, grad_output, pfirst, psecond)
            gradsense = needs_gradsense and sensesum(env, grad_output, pfirst, psecond)
            return gradgenv, gradsense, None, None

    envsum = AGEnvsum.apply
    sensesum = AGSensesum.apply
    featsum = AGFeatsum.apply

    return envsum, sensesum, featsum


