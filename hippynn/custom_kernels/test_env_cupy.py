import torch
import numba
from . import env_cupy
from .test_env_numba import Envops_tester, main
from .test_env_numba import TEST_MEGA_PARAMS, TEST_LARGE_PARAMS, TEST_MEDIUM_PARAMS, TEST_SMALL_PARAMS

if __name__ == "__main__":
    main(
        env_cupy.cupy_envsum,
        env_cupy.cupy_sensesum,
        env_cupy.cupy_featsum,
    )
