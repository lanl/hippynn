"""

Organized datasets for training and prediction.

.. Note::
   Databases constructed from disk (i.e. anything besides the base ``Database`` class)
   will load floating point data in the format (float32 or float64)
   specified via the ``torch.get_default_dtype()`` function. Use ``torch.set_default_dtype()``
   to control this behavior.

"""
from .database import Database
from .ondisk import DirectoryDatabase, NPZDatabase
from .utils import auto_detect_key

__all__ = ["Database", "DirectoryDatabase", "NPZDatabase", "auto_detect_key"]

try:
    import ase
except ImportError:
    pass
else:
    del ase
    from ..interfaces.ase_interface import AseDatabase, AseDatabaseIterable
    from .SNAPJson import SNAPDirectoryDatabase
    from .utils import load_database, write_extxyz
    __all__ += ["AseDatabase", "AseDatabaseIterable", "SNAPDirectoryDatabase", "load_database", "write_extxyz"]

    try:
        import h5py
    except ImportError:
        pass
    else:
        del h5py
        from .h5_pyanitools import PyAniFileDB, PyAniDirectoryDB
        __all__ += ["PyAniFileDB", "PyAniDirectoryDB"]
