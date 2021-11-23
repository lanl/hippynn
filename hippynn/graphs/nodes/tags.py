"""
Tags for nodes to inherit from, allowing tagging of different kinds
of information.
This file should not depend on any actual nodes.
"""
from ..indextypes import IdxType


class Encoder:
    _output_names = "encoding", "nonblank"
    species_set = NotImplemented


class AtomIndexer:
    pass


class Network:
    pass


class PairIndexer:
    _output_names = "pair_dist", "pair_first", "pair_second", "pair_coord"
    _output_index_states = (IdxType.Pair,) * len(_output_names)


class PairCache:
    pass


class HAtomRegressor:
    pass


class Charges:
    pass


class Positions:
    pass


class Species:
    pass


class Energies:
    pass
