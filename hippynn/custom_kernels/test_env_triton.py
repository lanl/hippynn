import torch
from .env_triton import envsum, sensesum, featsum
from .test_env_numba import Envops_tester, main, get_simulated_data
from .test_env_numba import TEST_MEGA_PARAMS, TEST_LARGE_PARAMS, TEST_MEDIUM_PARAMS, TEST_SMALL_PARAMS
from .utils import resort_pairs_cached

if __name__ == "__main__":

    main(
        envsum,
        sensesum,
        featsum,
    )
