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
has_ase = False
has_h5 = False

try: 
    import ase
    has_ase = True
    import h5py
    has_h5 = True
except ImportError:
    pass

if has_ase:
    from ..interfaces.ase_interface import AseDatabase
    from .SNAPJson import SNAPDirectoryDatabase
    if has_h5:
        from .h5_pyanitools import PyAniFileDB, PyAniDirectoryDB

all_list = ["Database", "DirectoryDatabase", "NPZDatabase"]

if has_ase:
    all_list += ["AseDatabase", "SNAPDirectoryDatabase"]
    if has_h5:
        all_list += ["PyAniFileDB", "PyAniDirectoryDB"]
__all__ = all_list
