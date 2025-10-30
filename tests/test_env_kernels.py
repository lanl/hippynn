from types import SimpleNamespace

import pytest

import torch
import hippynn

from hippynn.custom_kernels import CUSTOM_KERNELS_AVAILABLE, _RECOMMENDED_CUSTOM_KERNELS

CUDA_STATUSES = [False]
if torch.cuda.is_available():
    CUDA_STATUSES.append(True)

from hippynn.custom_kernels.env_pytorch import envsum, sensesum, featsum
from hippynn.custom_kernels.registry import MessagePassingKernels


@pytest.mark.filterwarnings("ignore:.*Fake error*.")
@pytest.mark.filterwarnings("ignore:.*Compilation errored*.")
def test_register_bad_compile():
    def bad_compile(impl_function):
        raise ValueError("Fake error for testing purposes!")

    result = MessagePassingKernels(
        "bad_compiler_implementation",
        envsum_impl=envsum,
        sensesum_impl=sensesum,
        featsum_impl=featsum,
        compiler=bad_compile,
    )

    assert result is None


@pytest.fixture()
def default_envtest_args():
    args = dict(
        implementation="pytorch",
        accelerator="cuda",
        compare_against="_pytorch_raw",
        no_correctness=False,
        no_speed=True,
        no_gpu=False,
        no_cpu=False,
        n_large=0,
        seed=0,
    )
    args = SimpleNamespace(**args)
    return args


@pytest.mark.xfail(strict=False)
@pytest.mark.parametrize("implementation", _RECOMMENDED_CUSTOM_KERNELS)
def test_available_implementation(implementation):
    assert implementation in CUSTOM_KERNELS_AVAILABLE


ignore_not_importable = pytest.mark.filterwarnings("ignore:.*implementation not importable*.")


@ignore_not_importable
def test_meta_envsum(default_envtest_args):
    """
    Test that the envsum tester fails if the envsum implementation is broken.
    """

    args = default_envtest_args

    def bad_envsum_impl(*args):
        "Like real envsum, but worse!"
        real_out = envsum(*args)
        fake_out = real_out + 1e-4 * torch.rand_like(real_out)
        return fake_out

    bad_kernels = MessagePassingKernels(
        "bad_kernels",
        envsum_impl=bad_envsum_impl,
        sensesum_impl=sensesum,
        featsum_impl=featsum,
        compiler=None,
    )
    args.no_gpu = True
    args.no_cpu = False
    args.implementation = "bad_kernels"
    import hippynn.custom_kernels.test_env as test_env

    with pytest.raises(RuntimeError) as e:
        test_env.main(args)
    assert "Failed during envsum" in str(e.value)


@ignore_not_importable
@pytest.mark.parametrize("cuda_status", CUDA_STATUSES)
@pytest.mark.parametrize("implementation", CUSTOM_KERNELS_AVAILABLE.copy())
def test_envsum_kernel(implementation, cuda_status, default_envtest_args):
    args = default_envtest_args
    args.no_gpu = not cuda_status
    args.no_cpu = cuda_status

    import hippynn.custom_kernels.test_env as test_env

    test_env.main(args)
