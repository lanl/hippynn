"""
Nodes for finding and manipulating pairs and distances.
"""

from .base.node_functions import NodeNotFound
from .base import AutoNoKw, AutoKw, ExpandParents, SingleNode, MultiNode, find_unique_relative, _BaseNode
from .indexers import PaddingIndexer, acquire_encoding_padding, OneHotEncoder
from .tags import Encoder, PairIndexer, AtomIndexer, PairCache
from .inputs import PositionsNode, CellNode, SpeciesNode
from ..indextypes import IdxType
from ...layers import pairs as pairs_modules


class OpenPairIndexer(ExpandParents, PairIndexer, MultiNode):
    _input_names = "coordinates", "nonblank", "real_atoms", "inv_real_atoms"
    _auto_module_class = pairs_modules.OpenPairIndexer

    @_parent_expander.match(PositionsNode, SpeciesNode)
    def expand0(self, pos, spec, *, purpose, **kwargs):
        enc = find_unique_relative(spec, Encoder, why_desc=purpose)
        padidx = find_unique_relative(spec, PaddingIndexer, why_desc=purpose)
        return pos, enc, padidx

    @_parent_expander.match(PositionsNode, Encoder, PaddingIndexer)
    def expand0(self, pos, encode, indexer, **kwargs):
        return pos, encode.nonblank, indexer.real_atoms, indexer.inv_real_atoms

    _parent_expander.assertlen(4)
    _parent_expander.get_main_outputs()
    _parent_expander.require_idx_states(IdxType.MolAtom, None, None, None)

    def __init__(self, name, parents, dist_hard_max, module="auto", **kwargs):
        self.dist_hard_max = dist_hard_max
        parents = self.expand_parents(parents)
        super().__init__(name, parents, module=module, **kwargs)

    def auto_module(self):
        return self._auto_module_class(self.dist_hard_max)


class PeriodicPairOutputs:
    _output_names = "pair_dist", "pair_first", "pair_second", "pair_coord", "cell_offsets", "offset_index"
    _output_index_states = (IdxType.Pair,) * len(_output_names)


class PeriodicPairIndexer(ExpandParents, AutoKw, PeriodicPairOutputs, PairIndexer, MultiNode):
    _input_names = "coordinates", "nonblank", "real_atoms", "inv_real_atoms", "cell"
    _auto_module_class = pairs_modules.PeriodicPairIndexer

    @_parent_expander.match(PositionsNode, SpeciesNode, CellNode)
    def expand0(self, pos, spec, cell, *, purpose, **kwargs):
        enc, padidx = acquire_encoding_padding(spec, species_set=None)
        return pos, enc, padidx, cell

    @_parent_expander.match(PositionsNode, Encoder, PaddingIndexer, CellNode)
    def expand1(self, pos, encode, indexer, cell, **kwargs):
        return pos, encode.nonblank, indexer.real_atoms, indexer.inv_real_atoms, cell

    _parent_expander.assertlen(5)
    _parent_expander.get_main_outputs()

    def __init__(self, name, parents, dist_hard_max, module="auto", module_kwargs=None, **kwargs):
        if module_kwargs is None:
            module_kwargs = {}
        self.module_kwargs = {"hard_dist_cutoff": dist_hard_max, **module_kwargs}
        self.dist_hard_max = self.module_kwargs["hard_dist_cutoff"]

        parents = self.expand_parents(parents)
        super().__init__(name, parents, module=module, **kwargs)

class Memory:
    @property
    def skin(self):
        return self.torch_module.skin
    
    @skin.setter
    def skin(self, skin):
        self.torch_module.skin = skin

    @property
    def reuse_percentage(self):
        return self.torch_module.reuse_percentage
    
    def reset_reuse_percentage(self):
        self.torch_module.reset_reuse_percentage()

class PeriodicPairIndexerMemory(PeriodicPairIndexer, Memory):
    '''
    Implementation of PeriodicPairIndexer with additional memory component.

    Stores current pair indices in memory and reuses them to compute the pair distances if no 
    particle has moved more than skin/2 since last pair calculation. Otherwise uses the
    _pair_indexer_class to recompute the pairs.

    Increasing the value of 'skin' will increase the number of pair distances computed at
    each step, but decrease the number of times new pairs must be computed. Skin should be 
    set to zero while training for fastest results.
    '''
    
    _auto_module_class = pairs_modules.periodic.PeriodicPairIndexerMemory

    def __init__(self, name, parents, dist_hard_max, skin, module="auto", module_kwargs=None, **kwargs):
        if module_kwargs is None:
            module_kwargs = {}
        self.module_kwargs = {"skin": skin, **module_kwargs}

        super().__init__(name, parents, dist_hard_max, module=module, module_kwargs=self.module_kwargs, **kwargs)


class ExternalNeighborIndexer(ExpandParents, PairIndexer, AutoKw, MultiNode):
    _input_names = "coordinates", "real_atoms", "shifts", "cell", "ext_pair_first", "ext_pair_second"
    _auto_module_class = pairs_modules.ExternalNeighbors

    _parent_expander.get_main_outputs()
    _parent_expander.assertlen(len(_input_names))
    _parent_expander.require_idx_states(IdxType.MolAtom, IdxType.MolAtom, None, None, None, None)

    def __init__(self, name, parents, hard_dist_cutoff, module="auto", **kwargs):
        self.module_kwargs = {"hard_dist_cutoff": hard_dist_cutoff}
        super().__init__(name, parents, module=module, **kwargs)


# Pair reindexer to re-use existing pairs
class PairReIndexer(ExpandParents, AutoNoKw, SingleNode):
    """
    For re-using index information to convert
    from IdxType.MolAtomAtom -> IdxType.Pairs
    """

    _input_names = "pair_features", "molecule_index", "atom_index", "pair_first", "pair_second"
    _auto_module_class = pairs_modules.PairReIndexer
    _index_state = IdxType.Pair

    @_parent_expander.match(_BaseNode)
    def expand0(self, pair_features):
        pad_idx = find_unique_relative(pair_features, PaddingIndexer)
        pair_idx = find_unique_relative(pair_features, PairIndexer)
        return pair_features, pad_idx, pair_idx

    @_parent_expander.match(_BaseNode, PaddingIndexer, PairIndexer)
    def expand1(self, pair_features, pad_idx, pair_idx):
        return (
            pair_features.main_output,
            pad_idx.molecule_index,
            pad_idx.atom_index,
            pair_idx.pair_first,
            pair_idx.pair_second,
        )

    _parent_expander.assertlen(5)
    _parent_expander.get_main_outputs()

    def __init__(self, name, parents, module="auto", **kwargs):
        super().__init__(name, parents, module=module, **kwargs)


# Pair deindexer to convert pair features back to padded form


class PairDeIndexer(ExpandParents, AutoNoKw, SingleNode):
    """
    For converting from IdxType.Pairs to IdxType.MolAtomAtom
    (Padded form)
    """

    _input_names = (
        "pair_features",
        "molecule_index",
        "atom_index",
        "n_molecules",
        "n_atoms_max" "pair_first",
        "pair_second",
    )
    _auto_module_class = pairs_modules.PairDeIndexer
    _index_state = IdxType.MolAtomAtom

    @_parent_expander.match(_BaseNode)
    def expand0(self, pair_features):
        pad_idx = find_unique_relative(pair_features, PaddingIndexer)
        pair_idx = find_unique_relative(pair_features, PairIndexer)
        return pair_features, pad_idx, pair_idx

    @_parent_expander.match(_BaseNode, PaddingIndexer, PairIndexer)
    def expand1(self, pair_features, pad_idx, pair_idx):
        return (
            pair_features.main_output,
            pad_idx.molecule_index,
            pad_idx.atom_index,
            pad_idx.n_molecules,
            pad_idx.n_atoms_max,
            pair_idx.pair_first,
            pair_idx.pair_second,
        )

    def __init__(self, name, parents, module="auto", **kwargs):
        super().__init__(name, parents, module=module, **kwargs)


class PairCacher(ExpandParents, AutoKw, PairCache, SingleNode):
    _input_names = (
        "pair_first",
        "pair_second",
        "cell_offsets",
        "offset_index",
        "real_atoms",
        "mol_index",
        "n_atoms_max",
        "n_molecules",
    )
    _auto_module_class = pairs_modules.PairCacher
    _index_state = IdxType.NotFound

    @_parent_expander.match(PairIndexer)
    def expand0(self, pair_indexer, *args, purpose, **kwargs):
        atomidx = find_unique_relative(pair_indexer, AtomIndexer)
        if "n_images" not in self.module_kwargs:
            self.module_kwargs["n_images"] = pair_indexer.torch_module.n_images
        return pair_indexer, atomidx

    @_parent_expander.match(PairIndexer, AtomIndexer)
    def expand1(self, pair_indexer, atomidx, *args, purpose, **kwargs):
        mi = atomidx.mol_index
        nam = atomidx.n_atoms_max
        n_molecules = atomidx.n_molecules
        ra = atomidx.real_atoms
        pf = pair_indexer.pair_first
        ps = pair_indexer.pair_second
        po = pair_indexer.cell_offsets
        poi = pair_indexer.offset_index
        return pf, ps, po, poi, ra, mi, n_molecules, nam

    _parent_expander.assertlen(8)
    _parent_expander.require_idx_states(IdxType.Pair, IdxType.Pair, None, None, None, None, None, None)

    def __init__(self, name, parents, module="auto", module_kwargs=None, **kwargs):
        self.module_kwargs = module_kwargs or {}
        parents = self.expand_parents(parents)
        super().__init__(name, parents, module=module, **kwargs)


class PairUncacher(ExpandParents, AutoNoKw, PairIndexer, MultiNode):
    _input_names = "sparsepairs", "coordinates", "cells", "real_atoms", "inv_real_atoms", "n_atoms_max", "n_molecules"
    _output_names = "pair_dist", "pair_first", "pair_second", "pair_coord", "cell_offsets", "offset_index"
    _output_index_states = (IdxType.Pair,) * len(_output_names)
    _auto_module_class = pairs_modules.PairUncacher
    _index_state = IdxType.NotFound

    @_parent_expander.match(PairCache)
    def expand0(self, sparse, *args, purpose, **kwargs):
        pos = find_unique_relative(sparse, PositionsNode)
        cell = find_unique_relative(sparse, CellNode)
        atomidx = find_unique_relative(sparse, AtomIndexer)
        return sparse, pos, cell, atomidx

    @_parent_expander.match(PairCache, PositionsNode, CellNode, AtomIndexer)
    @_parent_expander.match(_BaseNode, _BaseNode, _BaseNode, AtomIndexer) # Less constrained version
    def expand1(self, sp, r, c, atomidx, *args, purpose, **kwargs):
        ira = atomidx.inv_real_atoms
        nam = atomidx.n_atoms_max
        n_molecules = atomidx.n_molecules
        ra = atomidx.real_atoms
        return sp, r, c, ra, ira, nam, n_molecules

    _parent_expander.assertlen(7)

    def __init__(self, name, parents, dist_hard_max, module="auto", **kwargs):
        self.dist_hard_max = dist_hard_max
        parents = self.expand_parents(parents)
        super().__init__(name, parents, module=module, **kwargs)


class RDFBins(ExpandParents, AutoKw, SingleNode):
    _input_names = "pair_dists", "pair_first", "pair_second", "one_hot", "n_molecules"
    _index_state = None
    _auto_module_class = pairs_modules.RDFBins

    @_parent_expander.match(PositionsNode, SpeciesNode, CellNode)
    def expand0(self, pos, spec, cell, *, purpose, dist_hard_max=None, **kwargs):
        """
        Build a default Periodic Pair indexer.
        """
        pairs = PeriodicPairIndexer("Period Pairs", (pos, spec, cell), dist_hard_max=dist_hard_max)
        return pairs,

    @_parent_expander.match(PositionsNode, SpeciesNode)
    def expand1(self, pos, spec, *, purpose, dist_hard_max=None, **kwargs):
        """
        Builds an open pair indexer.
        """
        pairs = OpenPairIndexer("Period Pairs", (pos, spec), dist_hard_max=dist_hard_max)
        return pairs,

    @_parent_expander.match(PairIndexer)
    def expand2(self, pairs, *, purpose, **kwargs):
        """
        Get the encoding and padding associated with a pair indexer.
        """
        enc = find_unique_relative(pairs, OneHotEncoder)
        pad = find_unique_relative(pairs, PaddingIndexer)
        return pairs, enc, pad

    @_parent_expander.match(PairIndexer, OneHotEncoder, PaddingIndexer)
    def expand3(self, pairs, one_hot, pad, *, purpose, **kwargs):
        """
        Expanded the needed children of pairs, encoder, and padding indexer.
        """
        self.module_kwargs["species_set"] = one_hot.species_set
        return pairs.pair_dist, pairs.pair_first, pairs.pair_second, one_hot.encoding, pad.n_molecules

    _parent_expander.require_idx_states(IdxType.Pair, IdxType.Pair, IdxType.Pair, IdxType.Atoms, None)
    _parent_expander.assertlen(5)

    def __init__(self, name, parents, module="auto", bins=None, module_kwargs=None, **kwargs):
        if module_kwargs is None:
            module_kwargs = {}
        self.module_kwargs = {"bins": bins, **module_kwargs}
        parents = self.expand_parents(parents, dist_hard_max=max(bins))
        super().__init__(name, parents, module=module, **kwargs)


class _DispatchNeighbors(ExpandParents, AutoKw, PeriodicPairOutputs, PairIndexer, MultiNode):
    """
    Superclass for nodes that compute neighbors for systems one at a time.
    These should be capable of searching all feasible neighbors (no limit on number of images)
    """

    _input_names = (
        "coordinates",
        "nonblank",
        "real_atoms",
        "inv_real_atoms",
        "cell",
        "mol_index",
        "n_molecules",
        "n_atoms_max",
    )
    # Needs auto_module_class or explicit module

    @_parent_expander.match(PositionsNode, SpeciesNode, CellNode)
    def expand0(self, pos, spec, cell, **kwargs):
        """
        Acquire padding and encoding.
        """
        enc, padidx = acquire_encoding_padding(spec, species_set=None)
        return pos, enc, padidx, cell

    @_parent_expander.match(PositionsNode, Encoder, PaddingIndexer, CellNode)
    def expand1(self, pos, encode, indexer, cell, **kwargs):
        """
        Expand needed child nodes of encoder and padding indexer.
        """

        return (
            pos,
            encode.nonblank,
            indexer.real_atoms,
            indexer.inv_real_atoms,
            cell,
            indexer.mol_index,
            indexer.n_molecules,
            indexer.n_atoms_max,
        )

    _parent_expander.assertlen(8)
    _parent_expander.get_main_outputs()
    _parent_expander.require_idx_states(IdxType.MolAtom, None, None, None, None, None, None, None)

    def __init__(self, name, parents, dist_hard_max, module="auto", module_kwargs=None, **kwargs):
        self.dist_hard_max = dist_hard_max
        parents = self.expand_parents(parents)

        if module_kwargs is None:
            module_kwargs = {}
        self.module_kwargs = {"dist_hard_max": dist_hard_max, **module_kwargs}

        super().__init__(name, parents, module=module, **kwargs)


class NumpyDynamicPairs(_DispatchNeighbors):
    _auto_module_class = pairs_modules.NPNeighbors


class DynamicPeriodicPairs(_DispatchNeighbors):
    """
    Node for finding pairs in arbitrary periodic boundary conditions.
    Note: This will often be slower than PeriodicPairIndexer, but more general.
    If the speed is a concern, consider precomputing pairs with experiment.assembly.precompute_pairs
    """

    _auto_module_class = pairs_modules.TorchNeighbors

class KDTreePairs(_DispatchNeighbors):
    '''
    Node for finding pairs under periodic boundary conditions using Scipy's KD Tree algorithm. 
    Cell must be orthorhombic.
    '''
    _auto_module_class = pairs_modules.dispatch.KDTreeNeighbors

class KDTreePairsMemory(_DispatchNeighbors, Memory):
    '''
    Implementation of KDTreePairs with an added memory component.

    Stores current pair indices in memory and reuses them to compute the pair distances if no 
    particle has moved more than skin/2 since last pair calculation. Otherwise uses the
    _pair_indexer_class to recompute the pairs.

    Increasing the value of 'skin' will increase the number of pair distances computed at
    each step, but decrease the number of times new pairs must be computed. Skin should be 
    set to zero while training for fastest results.
    '''
    _auto_module_class = pairs_modules.dispatch.KDTreePairsMemory

    def __init__(self, name, parents, dist_hard_max, skin, module="auto", module_kwargs=None, **kwargs):
        if module_kwargs is None:
            module_kwargs = {}
        module_kwargs = {"skin": skin, **module_kwargs}

        super().__init__(name, parents, dist_hard_max, module=module, module_kwargs=module_kwargs, **kwargs)

class PaddedNeighborNode(ExpandParents, AutoNoKw, MultiNode):
    _input_names = "pair_first", "pair_second", "pair_coord"
    _output_names = (
        "j_list",
        "rij_list",
    )
    _output_index_states = IdxType.Atoms, IdxType.Atoms
    _auto_module_class = pairs_modules.PaddedNeighModule

    @_parent_expander.match(PairIndexer)
    def expand0(self, pair_finder, **kwargs):
        try:
            # Typically, the first atom tensor will come from
            # the output of the atom indexer, so look for that first.
            pad = pair_finder.find_unique_relative(AtomIndexer)
            atom_array = pad.indexed_features
        except NodeNotFound:
            # Fall back to finding -any- atom-indexed tensor.
            atom_arrays = pair_finder.find_relatives(
                lambda node: hasattr(node, "_index_state") and node._index_state == IdxType.Atoms
            )
            atom_array = atom_arrays.pop()

        return pair_finder.pair_first, pair_finder.pair_second, pair_finder.pair_coord, atom_array

    _parent_expander.assertlen(4)
    _parent_expander.get_main_outputs()
    _parent_expander.require_idx_states(IdxType.Pair, IdxType.Pair, IdxType.Pair, IdxType.Atoms)

    def __init__(self, name, parents, module="auto", **kwargs):
        parents = self.expand_parents(parents)
        super().__init__(name, parents, module=module, **kwargs)


class MinDistNode(ExpandParents, AutoNoKw, MultiNode):
    _input_names = "rij_list", "j_list", "mol_index", "atom_index", "inv_real_atoms", "n_atoms_max", "n_molecules"
    _output_names = "min_dist_mol", "mol_locs", "min_dist_atom", "atom_pairlocs"
    _output_index_states = IdxType.Molecules, IdxType.Molecules, IdxType.Atoms, IdxType.Atoms
    _auto_module_class = pairs_modules.MinDistModule

    @_parent_expander.match(PairIndexer)
    def expand0(self, pair_finder, **kwargs):

        try:
            neigh_list = pair_finder.find_unique_relative(PaddedNeighborNode)
        except NodeNotFound:
            neigh_list = PaddedNeighborNode("NeighList", pair_finder)

        return (neigh_list,)

    @_parent_expander.match(PaddedNeighborNode)
    def expand1(self, neigh_list, **kwargs):
        pad = neigh_list.find_unique_relative(AtomIndexer)
        return neigh_list, pad

    @_parent_expander.match(PaddedNeighborNode, AtomIndexer)
    def expand2(self, neigh_list, pad_idxer, **kwargs):
        return (
            neigh_list.rij_list,
            neigh_list.j_list,
            pad_idxer.mol_index,
            pad_idxer.atom_index,
            pad_idxer.inv_real_atoms,
            pad_idxer.n_atoms_max,
            pad_idxer.n_molecules,
        )

    _parent_expander.assertlen(7)
    _parent_expander.get_main_outputs()
    _parent_expander.require_idx_states(IdxType.Atoms, IdxType.Atoms, None, None, None, None, None)

    def __init__(self, name, parents, module="auto", **kwargs):
        parents = self.expand_parents(parents)
        super().__init__(name, parents, module=module, **kwargs)


# Graph Nodes for Filter Pair Indexer. Works with PeriodicPairs, OpenPairs, External Neighbors.
class PairFilter(AutoKw, PairIndexer, ExpandParents, MultiNode):
    _auto_module_class = pairs_modules.FilterDistance

    @_parent_expander.match(PairIndexer)
    def expand0(self, pair_indexer, purpose):

        # During graph construction, every node is connected to its current set of parents. 
        # It is possible that pair_indexer.children can contain itself; an un-initialized PairFilter. 
        # Only initialized PairIndexers are extracted here. 
        parents = [c for c in pair_indexer.children if hasattr(c, "_index_state")]

        # Validate that nothing unexpected has happened.
        # Hopefully this can't fail, but if we update the pair API or someone customizes this aspect of the
        # library, this should catch any problems.
        idx_states = set(c._index_state for c in parents)

        if len(idx_states) != 1:
            raise TypeError(f"Input contains mixed index states: {idx_states}. Input states should only consist of index type pair.")
        idx_state = idx_states.pop()
        if idx_state != IdxType.Pair:
            raise TypeError(f"Index state for inputs was {idx_state}, needs to be index type pair.")
        # Validation complete.
        self._output_names = tuple(f"out_{name}" for name in pair_indexer._output_names)
        self._input_names = tuple(f"in_{name}" for name in pair_indexer._output_names)
        self._output_index_states = (IdxType.Pair,)*len(parents)

        return parents

    def __init__(self, name, parents, dist_hard_max, module="auto", **kwargs):
        self.module_kwargs = {"hard_dist_cutoff": dist_hard_max}  # passes to PairIndexer superclass
        self.dist_hard_max = dist_hard_max
        parents = self.expand_parents(parents)
        super().__init__(name, parents, module=module, **kwargs)
