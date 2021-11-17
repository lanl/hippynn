"""
Wraps non-pytorch implementations for use with pytorch autograd.
"""
import torch.autograd


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
            env = envsum_impl(sense, feat, pfirst, psecond)
            return env

        @staticmethod
        def backward(ctx, grad_output):
            sense, feat, pfirst, psecond, = ctx.saved_tensors
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
            sense = sensesum_impl(env, feat, pfirst, psecond)
            return sense

        @staticmethod
        def backward(ctx, grad_output):
            env, feat, pfirst, psecond = ctx.saved_tensors
            needs_gradenv, needs_gradfeat, *_ = ctx.needs_input_grad
            needs_gradenv = needs_gradenv or None
            needs_gradfeat = needs_gradfeat or None
            gradenv =  needs_gradenv and envsum(grad_output, feat, pfirst, psecond)
            gradfeat = needs_gradfeat and featsum(env, grad_output, pfirst, psecond)
            return gradenv, gradfeat, None, None


    class AGFeatsum(torch.autograd.Function):
        @staticmethod
        def forward(ctx, env, sense, pfirst, psecond):
            ctx.save_for_backward(env, sense, pfirst, psecond)
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