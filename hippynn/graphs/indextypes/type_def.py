"""
Enum for index states.
"""
import enum

@enum.unique
class IdxType(enum.Enum):
    Scalar      = "Scalar"
    MolAtom     = "MolAtom"       #rectangular padded array of atom-like data
    Molecules   = "Molecules"
    Atoms       = "Atoms"
    Pair        = "Pair"
    MolAtomAtom = "MolAtomAtom"   #rectangular padded array of bond-like data
    QuadMol     = "QuadMol"
    QuadPack    = "QuadPack"      #packed 6-vec of quadrupole upper triangle
    NotFound    = "NOT FOUND"
    def __repr__(self): return '<%s.%s>' % (self.__class__.__name__, self.name)