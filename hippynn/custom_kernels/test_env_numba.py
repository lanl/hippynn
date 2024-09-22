
from .test_env import main
from . import env_numba

if __name__ == "__main__":
    main(
        env_numba.new_envsum,
        env_numba.new_sensesum,
        env_numba.new_featsum,
    )
