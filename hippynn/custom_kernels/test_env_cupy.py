
from . import env_cupy
from .test_env import main


if __name__ == "__main__":
    main(
        env_cupy.cupy_envsum,
        env_cupy.cupy_sensesum,
        env_cupy.cupy_featsum,
    )
