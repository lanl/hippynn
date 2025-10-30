"""
Enum for index states.
"""
import enum
import warnings
from ..._deprecations import warn_name_change

# fmt: off
class IdxType(enum.Enum):
    Scalar      = "Scalar"
    Systems     = "Systems"
    Molecules   = Systems         # deprecated name
    SysAtom     = "SysAtom"       # rectangular padded array of atom-like data
    MolAtom     = SysAtom         # deprecated name
    Atoms       = "Atoms"
    Pairs       = "Pairs"
    Pair        = Pairs           # deprecated name
    SysAtomAtom = "SysAtomAtom"   # rectangular padded array of bond-like data
    MolAtomAtom = SysAtomAtom     # deprecated name
    QuadMol     = "QuadMol"
    QuadPack    = "QuadPack"      # packed 6-vec of quadrupole, upper triangle
    Unlabeled   = "Unlabeled"     # Does not conform to an existing indexing scheme; do not coerce.
    NotFound    = Unlabeled       
    
    def __repr__(self): return f"<{self.__class__.__name__}.{self.name}>"

    @classmethod
    def _missing_(cls, value):
        if value == "NOT FOUND":
            output = output = cls.Unlabeled
        else:
            output = cls.__members__.get(value, None)
        if output is not None:
            warn_name_change(value, output.value, old_module=__name__, new_module=__name__)
        return output

# fmt: on



