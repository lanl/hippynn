
from . import env_triton
from .test_env import main

if __name__ == "__main__":

    main(
        env_triton.envsum,
        env_triton.sensesum,
        env_triton.featsum,
    )
