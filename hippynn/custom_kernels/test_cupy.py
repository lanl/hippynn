
import torch
import numba
import env_cupy
from hippynn.custom_kernels.env_tests import Envops_tester
from hippynn.custom_kernels.env_tests import TEST_MEGA_PARAMS, TEST_LARGE_PARAMS, TEST_MEDIUM_PARAMS, TEST_SMALL_PARAMS

def main():
    tester = Envops_tester(
        env_cupy.cupy_envsum,
        env_cupy.cupy_sensesum,
        env_cupy.cupy_featsum,
    )
    # % time

    if torch.cuda.is_available():
        print("Running GPU tests")
        meminfo = numba.cuda.current_context().get_memory_info()
        use_large_gpu = meminfo.free > 2 ** 31

        if use_large_gpu:
            tester.check_correctness(device=torch.device("cuda"))
            print("-" * 80)
            print("Mega systems:", TEST_MEGA_PARAMS)
            tester.check_speed(n_repetitions=20,data_size=TEST_MEGA_PARAMS, device=torch.device("cuda"), compare_against="Numba")
            print("-" * 80)
            print("Large systems:", TEST_LARGE_PARAMS)
            tester.check_speed(n_repetitions=20, device=torch.device("cuda"), compare_against="Numba")
        else:
            print("Numba indicates less than 2GB free GPU memory -- skipping large system test")
            tester.check_correctness(device=torch.device("cuda"), n_large=0)

        print("-" * 80)
        print("Medium systems:", TEST_MEDIUM_PARAMS)
        tester.check_speed(
            n_repetitions=100, data_size=TEST_MEDIUM_PARAMS, device=torch.device("cuda"), compare_against="Numba"
        )
        print("-" * 80)
        print("Small systems:", TEST_SMALL_PARAMS)
        tester.check_speed(
            n_repetitions=100, data_size=TEST_SMALL_PARAMS, device=torch.device("cuda"), compare_against="Numba"
        )

    else:
        print("Cuda not available, not running GPU tests.")

    print("Running CPU tests")
    tester.check_correctness()
    print("-" * 80)
    print("Large systems:", TEST_LARGE_PARAMS)
    tester.check_speed(n_repetitions=10, compare_against="Numba")
    print("-" * 80)
    print("Medium systems:", TEST_MEDIUM_PARAMS)
    tester.check_speed(n_repetitions=100, data_size=TEST_MEDIUM_PARAMS, compare_against="Numba")
    print("-" * 80)
    print("Small systems:", TEST_SMALL_PARAMS)
    tester.check_speed(n_repetitions=100, compare_against="Numba", data_size=TEST_SMALL_PARAMS)

if __name__ == "__main__":
    main()
