
from . import env_sparse
from .test_env import main

if __name__ == "__main__":
    main(env_sparse.envsum,
         env_sparse.sensesum,
         env_sparse.featsum,
    )

