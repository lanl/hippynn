"""
Nodes for finding and manipulating pairs and distances.
"""

from .base.node_functions import NodeNotFound
from .base import AutoNoKw, AutoKw, ExpandParents, SingleNode, MultiNode, find_unique_relative, Node
from .indexers import PaddingIndexer, acquire_encoding_padding, OneHotEncoder
from .tags import Encoder, PairIndexer, AtomIndexer, PairCache
from .inputs import PositionsNode, CellNode, SpeciesNode
from ..indextypes import IdxType
from ...layers import pairs as pairs_modules


class OpenPairIndexer(AutoKw, ExpandParents, PairIndexer, MultiNode):
    input_names = "coordinates", "nonblank", "real_atoms", "inv_real_atoms"
    auto_module_class = pairs_modules.OpenPairIndexer
    auto_module_kwargs = {
        "hard_dist_cutoff": "dist_hard_max",
    }

    @parent_expander.match(PositionsNode, SpeciesNode)
    def expand0(self, pos, spec, *, purpose, **kwargs):
        enc = find_unique_relative(spec, Encoder, why_desc=purpose)
        padidx = find_unique_relative(spec, PaddingIndexer, why_desc=purpose)
        return pos, enc, padidx

    @parent_expander.match(PositionsNode, Encoder, PaddingIndexer)
    def expand0(self, pos, encode, indexer, **kwargs):
        return pos, encode.nonblank, indexer.real_atoms, indexer.inv_real_atoms

    parent_expander.assertlen(4)
    parent_expander.get_main_outputs()
    parent_expander.require_idx_states(IdxType.SysAtom, None, None, None)

    def __init__(self, name, parents, dist_hard_max, **kwargs):
        self.dist_hard_max = dist_hard_max
        super().__init__(name, parents, dist_hard_max=dist_hard_max, **kwargs)



class PeriodicPairOutputs:
    output_names = "pair_dist", "pair_first", "pair_second", "pair_coord", "cell_offsets", "offset_index"
    output_index_states = (IdxType.Pairs,) * len(output_names)


class PeriodicPairIndexer(AutoKw, ExpandParents,  PeriodicPairOutputs, PairIndexer, MultiNode):
    input_names = "coordinates", "nonblank", "real_atoms", "inv_real_atoms", "cell"
    auto_module_class = pairs_modules.PeriodicPairIndexer
    auto_module_kwargs = {
        "hard_dist_cutoff": "dist_hard_max",
    }

    @parent_expander.match(PositionsNode, SpeciesNode, CellNode)
    def expand0(self, pos, spec, cell, *, purpose, **kwargs):
        enc, padidx = acquire_encoding_padding(spec, species_set=None)
        return pos, enc, padidx, cell

    @parent_expander.match(PositionsNode, Encoder, PaddingIndexer, CellNode)
    def expand1(self, pos, encode, indexer, cell, **kwargs):
        return pos, encode.nonblank, indexer.real_atoms, indexer.inv_real_atoms, cell

    parent_expander.assertlen(5)
    parent_expander.get_main_outputs()
    def __init__(self, name, parents, dist_hard_max, **kwargs):
        self.dist_hard_max = dist_hard_max
        super().__init__(name, parents, dist_hard_max=dist_hard_max, **kwargs)

class SparsePairIndexer(PeriodicPairIndexer):
    auto_module_class = pairs_modules.SparsePairIndexer

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
    
    auto_module_class = pairs_modules.periodic.PeriodicPairIndexerMemory

    def __init__(self, name, parents, dist_hard_max, skin, module="auto", module_kwargs=None, **kwargs):
        if module_kwargs is None:
            module_kwargs = {}
        self.module_kwargs = {"skin": skin, **module_kwargs}

        super().__init__(name, parents, dist_hard_max, module=module, module_kwargs=self.module_kwargs, **kwargs)


class ExternalNeighborIndexer(AutoKw, ExpandParents, PairIndexer,  MultiNode):
    input_names = "coordinates", "real_atoms", "shifts", "cell", "ext_pair_first", "ext_pair_second"
    auto_module_class = pairs_modules.ExternalNeighbors
    auto_module_kwargs = "hard_dist_cutoff",

    parent_expander.get_main_outputs()
    parent_expander.assertlen(len(input_names))
    parent_expander.require_idx_states(IdxType.SysAtom, None, None, None, None, None)


# Pair reindexer to re-use existing pairs
class PairReIndexer(ExpandParents, AutoNoKw, SingleNode):
    """
    For re-using index information to convert
    from IdxType.SysAtomAtom -> IdxType.Pairs
    """

    input_names = "pair_features", "system_index", "atom_index", "pair_first", "pair_second"
    auto_module_class = pairs_modules.PairReIndexer
    index_state = IdxType.Pairs

    @parent_expander.match(Node)
    def expand0(self, pair_features):
        pad_idx = find_unique_relative(pair_features, PaddingIndexer)
        pair_idx = find_unique_relative(pair_features, PairIndexer)
        return pair_features, pad_idx, pair_idx

    @parent_expander.match(Node, PaddingIndexer, PairIndexer)
    def expand1(self, pair_features, pad_idx, pair_idx):
        return (
            pair_features.main_output,
            pad_idx.system_index,
            pad_idx.atom_index,
            pair_idx.pair_first,
            pair_idx.pair_second,
        )

    parent_expander.assertlen(5)
    parent_expander.get_main_outputs()

    def __init__(self, name, parents, module="auto", **kwargs):
        super().__init__(name, parents, module=module, **kwargs)


# Pair deindexer to convert pair features back to padded form


class PairDeIndexer(ExpandParents, AutoNoKw, SingleNode):
    """
    For converting from IdxType.Pairs to IdxType.SysAtomAtom
    (Padded form)
    """

    input_names = (
        "pair_features",
        "system_index",
        "atom_index",
        "n_systems",
        "n_atoms_max" "pair_first",
        "pair_second",
    )
    auto_module_class = pairs_modules.PairDeIndexer
    index_state = IdxType.SysAtomAtom

    @parent_expander.match(Node)
    def expand0(self, pair_features):
        pad_idx = find_unique_relative(pair_features, PaddingIndexer)
        pair_idx = find_unique_relative(pair_features, PairIndexer)
        return pair_features, pad_idx, pair_idx

    @parent_expander.match(Node, PaddingIndexer, PairIndexer)
    def expand1(self, pair_features, pad_idx, pair_idx):
        return (
            pair_features.main_output,
            pad_idx.system_index,
            pad_idx.atom_index,
            pad_idx.n_systems,
            pad_idx.n_atoms_max,
            pair_idx.pair_first,
            pair_idx.pair_second,
        )

    def __init__(self, name, parents, module="auto", **kwargs):
        super().__init__(name, parents, module=module, **kwargs)


class PairCacher(AutoKw, ExpandParents, PairCache, SingleNode):
    input_names = (
        "pair_first",
        "pair_second",
        "cell_offsets",
        "offset_index",
        "real_atoms",
        "system_index",
        "n_atoms_max",
        "n_systems",
    )
    auto_module_class = pairs_modules.PairCacher
    index_state = IdxType.Unlabeled

    @parent_expander.match(PairIndexer)
    def expand0(self, pair_indexer, *args, purpose, **kwargs):
        atomidx = find_unique_relative(pair_indexer, AtomIndexer)
        if "n_images" not in self.module_kwargs:
            self.module_kwargs["n_images"] = pair_indexer.torch_module.n_images
        return pair_indexer, atomidx

    @parent_expander.match(PairIndexer, AtomIndexer)
    def expand1(self, pair_indexer, atomidx, *args, purpose, **kwargs):
        mi = atomidx.system_index
        nam = atomidx.n_atoms_max
        n_systems = atomidx.n_systems
        ra = atomidx.real_atoms
        pf = pair_indexer.pair_first
        ps = pair_indexer.pair_second
        po = pair_indexer.cell_offsets
        poi = pair_indexer.offset_index
        return pf, ps, po, poi, ra, mi, n_systems, nam

    parent_expander.assertlen(8)
    parent_expander.require_idx_states(IdxType.Pairs, IdxType.Pairs, None, None, None, None, None, None)


class PairUncacher(ExpandParents, AutoNoKw, PairIndexer, MultiNode):
    input_names = "sparsepairs", "coordinates", "cells", "real_atoms", "inv_real_atoms", "n_atoms_max", "n_systems"
    output_names = "pair_dist", "pair_first", "pair_second", "pair_coord", "cell_offsets", "offset_index"
    output_index_states = (IdxType.Pairs,) * len(output_names)
    auto_module_class = pairs_modules.PairUncacher
    auto_module_kwargs = "dist_hard_max",
    index_state = IdxType.Unlabeled

    @parent_expander.match(PairCache)
    def expand0(self, sparse, *args, purpose, **kwargs):
        pos = find_unique_relative(sparse, PositionsNode)
        cell = find_unique_relative(sparse, CellNode)
        atomidx = find_unique_relative(sparse, AtomIndexer)
        return sparse, pos, cell, atomidx

    @parent_expander.match(PairCache, PositionsNode, CellNode, AtomIndexer)
    @parent_expander.match(Node, Node, Node, AtomIndexer) # Less constrained version
    def expand1(self, sp, r, c, atomidx, *args, purpose, **kwargs):
        ira = atomidx.inv_real_atoms
        nam = atomidx.n_atoms_max
        n_systems = atomidx.n_systems
        ra = atomidx.real_atoms
        return sp, r, c, ra, ira, nam, n_systems

    parent_expander.assertlen(7)

    def __init__(self, name, parents, dist_hard_max, **kwargs):
        self.dist_hard_max = dist_hard_max
        super().__init__(name, parents, dist_hard_max=dist_hard_max, **kwargs)


class RDFBins(AutoKw, ExpandParents, SingleNode):
    input_names = "pair_dists", "pair_first", "pair_second", "one_hot", "n_systems"
    index_state = IdxType.Scalar # Computes over whole batch.
    auto_module_class = pairs_modules.RDFBins
    auto_module_kwargs = "bins",
    parent_expansion_kwargs = "dist_hard_max",

    @parent_expander.match(PositionsNode, SpeciesNode, CellNode)
    def expand0(self, pos, spec, cell, *, purpose, dist_hard_max=None, **kwargs):
        """
        Build a default Periodic Pair indexer.
        """
        pairs = PeriodicPairIndexer("Period Pairs", (pos, spec, cell), dist_hard_max=dist_hard_max)
        return pairs,

    @parent_expander.match(PositionsNode, SpeciesNode)
    def expand1(self, pos, spec, *, purpose, dist_hard_max=None, **kwargs):
        """
        Builds an open pair indexer.
        """
        pairs = OpenPairIndexer("Period Pairs", (pos, spec), dist_hard_max=dist_hard_max)
        return pairs,

    @parent_expander.match(PairIndexer)
    def expand2(self, pairs, *, purpose, **kwargs):
        """
        Get the encoding and padding associated with a pair indexer.
        """
        enc = find_unique_relative(pairs, OneHotEncoder)
        pad = find_unique_relative(pairs, PaddingIndexer)
        return pairs, enc, pad

    @parent_expander.match(PairIndexer, OneHotEncoder, PaddingIndexer)
    def expand3(self, pairs, one_hot, pad, *, purpose, **kwargs):
        """
        Expanded the needed children of pairs, encoder, and padding indexer.
        """
        self.module_kwargs["species_set"] = one_hot.species_set
        return pairs.pair_dist, pairs.pair_first, pairs.pair_second, one_hot.encoding, pad.n_systems

    parent_expander.require_idx_states(IdxType.Pairs, IdxType.Pairs, IdxType.Pairs, IdxType.Atoms, None)
    parent_expander.assertlen(5)

    def __init__(self, name, parents, bins=None, **kwargs):
        dist_hard_max = max(bins)
        super().__init__(name, parents, bins=bins, dist_hard_max=dist_hard_max, **kwargs)


class _DispatchNeighbors(AutoKw, ExpandParents, PeriodicPairOutputs, PairIndexer, MultiNode):
    """
    Superclass for nodes that compute neighbors for systems one at a time.
    These should be capable of searching all feasible neighbors (no limit on number of images)
    """

    input_names = (
        "coordinates",
        "nonblank",
        "real_atoms",
        "inv_real_atoms",
        "cell",
        "system_index",
        "n_systems",
        "n_atoms_max",
    )
    auto_module_kwargs = "dist_hard_max",
    

    @parent_expander.match(PositionsNode, SpeciesNode, CellNode)
    def expand0(self, pos, spec, cell, **kwargs):
        """
        Acquire padding and encoding.
        """
        enc, padidx = acquire_encoding_padding(spec, species_set=None)
        return pos, enc, padidx, cell

    @parent_expander.match(PositionsNode, Encoder, PaddingIndexer, CellNode)
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
            indexer.system_index,
            indexer.n_systems,
            indexer.n_atoms_max,
        )

    parent_expander.assertlen(8)
    parent_expander.get_main_outputs()
    parent_expander.require_idx_states(IdxType.SysAtom, None, None, None, None, None, None, None)

    def __init__(self, name, parents, dist_hard_max, **kwargs):
        self.dist_hard_max = dist_hard_max
        super().__init__(name, parents, dist_hard_max=dist_hard_max, **kwargs)


class NumpyDynamicPairs(_DispatchNeighbors):
    auto_module_class = pairs_modules.NPNeighbors


class DynamicPeriodicPairs(_DispatchNeighbors):
    """
    Node for finding pairs in arbitrary periodic boundary conditions.
    Note: This will often be slower than PeriodicPairIndexer, but more general.
    If the speed is a concern, consider precomputing pairs with experiment.assembly.precompute_pairs
    """

    auto_module_class = pairs_modules.TorchNeighbors

class KDTreePairs(_DispatchNeighbors):
    '''
    Node for finding pairs under periodic boundary conditions using Scipy's KD Tree algorithm. 
    Cell must be orthorhombic.
    '''
    auto_module_class = pairs_modules.dispatch.KDTreeNeighbors

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
    auto_module_class = pairs_modules.dispatch.KDTreePairsMemory
    auto_module_kwargs = "dist_hard_max",

    def __init__(self, name, parents, dist_hard_max, skin, module="auto", module_kwargs=None, **kwargs):
        if module_kwargs is None:
            module_kwargs = {}
        module_kwargs = {"skin": skin, **module_kwargs}

        super().__init__(name, parents, dist_hard_max, module=module, module_kwargs=module_kwargs, **kwargs)

class PaddedNeighborNode(ExpandParents, AutoNoKw, MultiNode):
    input_names = "pair_first", "pair_second", "pair_coord"
    output_names = (
        "j_list",
        "rij_list",
    )
    output_index_states = IdxType.Atoms, IdxType.Atoms
    auto_module_class = pairs_modules.PaddedNeighModule

    @parent_expander.match(PairIndexer)
    def expand0(self, pair_finder, **kwargs):
        try:
            # Typically, the first atom tensor will come from
            # the output of the atom indexer, so look for that first.
            pad = pair_finder.find_unique_relative(AtomIndexer)
            atom_array = pad.indexed_features
        except NodeNotFound:
            # Fall back to finding -any- atom-indexed tensor.
            atom_arrays = pair_finder.find_relatives(
                lambda node: hasattr(node, "index_state") and node.index_state == IdxType.Atoms
            )
            atom_array = atom_arrays.pop()

        return pair_finder.pair_first, pair_finder.pair_second, pair_finder.pair_coord, atom_array

    parent_expander.assertlen(4)
    parent_expander.get_main_outputs()
    parent_expander.require_idx_states(IdxType.Pairs, IdxType.Pairs, IdxType.Pairs, IdxType.Atoms)

    def __init__(self, name, parents, module="auto", **kwargs):
        parents = self.expand_parents(parents)
        super().__init__(name, parents, module=module, **kwargs)


class MinDistNode(ExpandParents, AutoNoKw, MultiNode):
    input_names = "rij_list", "j_list", "system_index", "atom_index", "inv_real_atoms", "n_atoms_max", "n_systems"
    output_names = "min_dist_mol", "mol_locs", "min_dist_atom", "atom_pairlocs"
    output_index_states = IdxType.Systems, IdxType.Systems, IdxType.Atoms, IdxType.Atoms
    auto_module_class = pairs_modules.MinDistModule

    @parent_expander.match(PairIndexer)
    def expand0(self, pair_finder, **kwargs):

        try:
            neigh_list = pair_finder.find_unique_relative(PaddedNeighborNode)
        except NodeNotFound:
            neigh_list = PaddedNeighborNode("NeighList", pair_finder)

        return (neigh_list,)

    @parent_expander.match(PaddedNeighborNode)
    def expand1(self, neigh_list, **kwargs):
        pad = neigh_list.find_unique_relative(AtomIndexer)
        return neigh_list, pad

    @parent_expander.match(PaddedNeighborNode, AtomIndexer)
    def expand2(self, neigh_list, pad_idxer, **kwargs):
        return (
            neigh_list.rij_list,
            neigh_list.j_list,
            pad_idxer.system_index,
            pad_idxer.atom_index,
            pad_idxer.inv_real_atoms,
            pad_idxer.n_atoms_max,
            pad_idxer.n_systems,
        )

    parent_expander.assertlen(7)
    parent_expander.get_main_outputs()
    parent_expander.require_idx_states(IdxType.Atoms, IdxType.Atoms, None, None, None, None, None)

    def __init__(self, name, parents, module="auto", **kwargs):
        parents = self.expand_parents(parents)
        super().__init__(name, parents, module=module, **kwargs)


# Graph Nodes for Filter Pair Indexer. Works with PeriodicPairs, OpenPairs, External Neighbors.
class PairFilter(AutoKw, PairIndexer, ExpandParents, MultiNode):
    auto_module_class = pairs_modules.FilterDistance

    @parent_expander.match(PairIndexer)
    def expand0(self, pair_indexer, purpose):

        # During graph construction, every node is connected to its current set of parents. 
        # It is possible that pair_indexer.children can contain itself; an un-initialized PairFilter. 
        # Only initialized PairIndexers are extracted here. 
        parents = [c for c in pair_indexer.children if hasattr(c, "index_state")]

        # Validate that nothing unexpected has happened.
        # Hopefully this can't fail, but if we update the pair API or someone customizes this aspect of the
        # library, this should catch any problems.
        idx_states = set(c.index_state for c in parents)

        if len(idx_states) != 1:
            raise TypeError(f"Input contains mixed index states: {idx_states}. Input states should only consist of index type pair.")
        idx_state = idx_states.pop()
        if idx_state != IdxType.Pairs:
            raise TypeError(f"Index state for inputs was {idx_state}, needs to be index type pair.")
        # Validation complete.
        self.output_names = tuple(f"out_{name}" for name in pair_indexer.output_names)
        self.input_names = tuple(f"in_{name}" for name in pair_indexer.output_names)
        self.output_index_states = (IdxType.Pairs,)*len(parents)

        return parents

    def __init__(self, name, parents, dist_hard_max, module="auto", **kwargs):
        self.module_kwargs = {"hard_dist_cutoff": dist_hard_max}  # passes to PairIndexer superclass
        self.dist_hard_max = dist_hard_max
        parents = self.expand_parents(parents)
        super().__init__(name, parents, module=module, **kwargs)
