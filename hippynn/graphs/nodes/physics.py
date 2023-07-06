"""
Nodes for physics transformations
"""
import warnings

from .base import SingleNode, MultiNode, AutoNoKw, AutoKw, ExpandParents, find_unique_relative, _BaseNode
from .base.node_functions import NodeNotFound
from .indexers import AtomIndexer, PaddingIndexer, acquire_encoding_padding
from .pairs import OpenPairIndexer
from .tags import Encoder, PairIndexer, Charges
from .inputs import PositionsNode, SpeciesNode

from ..indextypes import IdxType, index_type_coercion, elementwise_compare_reduce
from ...layers import indexers as index_layers
from ...layers import physics as physics_layers
from ...layers import pairs as pair_layers


class GradientNode(AutoKw, SingleNode):
    """
    Compute the gradient of a quantity.
    """

    _input_names = "energy", "coordinates"
    _auto_module_class = physics_layers.Gradient

    def __init__(self, name, parents, sign, module="auto", **kwargs):
        self.module_kwargs = {"sign": sign}
        energy, position = parents
        position.requires_grad = True
        parents = energy.main_output, position
        self.sign = sign
        self._index_state = position._index_state
        super().__init__(name, parents, module=module, **kwargs)


class StressForceNode(AutoNoKw, MultiNode):
    _input_names = "energy", "strain", "coordinates", "cell"
    _output_names = "forces", "stress"
    _auto_module_class = physics_layers.StressForce

    def __init__(self, name, parents, module="auto", **kwargs):
        energy, strain, coordinates, cell = parents
        coordinates.requires_grad = True
        parents = energy.main_output, strain, coordinates, cell
        self._output_index_states = coordinates._index_state, strain._index_state
        super().__init__(name, parents, module=module, **kwargs)


class ChargeMomentNode(ExpandParents, AutoNoKw, SingleNode):
    _input_names = "charges", "positions", "mol_index", "n_molecules"

    @_parent_expander.matchlen(1)
    def expansion0(self, charges, *, purpose, **kwargs):
        return charges, find_unique_relative(charges, PositionsNode, why_desc=purpose)

    @_parent_expander.match(Charges, PositionsNode)
    def expansion1(self, charges, positions, *, purpose, **kwargs):
        enc, pidxer = acquire_encoding_padding((charges, positions), species_set=None, purpose=purpose)
        return charges, positions, pidxer

    @_parent_expander.match(Charges, PositionsNode, AtomIndexer)
    def expansion2(self, charges, positions, pdxer, **kwargs):
        return charges, positions, pdxer.mol_index, pdxer.n_molecules

    _parent_expander.assertlen(4)
    _parent_expander.get_main_outputs()
    _parent_expander.require_idx_states(IdxType.Atoms, IdxType.Atoms, None, None)

    def __init__(self, name, parents, module="auto", **kwargs):
        parents = self.expand_parents(parents)
        super().__init__(name, parents, module=module, **kwargs)


class DipoleNode(ChargeMomentNode):
    """
    Compute the dipole of point charges.
    """

    _auto_module_class = physics_layers.Dipole
    _index_state = IdxType.Molecules


class QuadrupoleNode(ChargeMomentNode):
    """
    Compute the traceless quadrupole of point charges.
    """

    _auto_module_class = physics_layers.Quadrupole
    _index_state = IdxType.QuadMol


# Setup for Coulomb Energy and Screened Coulomb Energy is nearly the same, up to validating the pair finder.
class ChargePairSetup(ExpandParents):
    @staticmethod
    def _validate_pairfinder(pairfinder, cutoff_distance):
        # This method required by this ExpandParents setup.
        # Raises an error if the pairfinder is not satisfactory.
        return NotImplemented

    @_parent_expander.match(Charges)
    def expansion0(self, charges, *, purpose, **kwargs):
        try:
            pos_or_pair = find_unique_relative(charges, PairIndexer, why_desc=purpose)
        except NodeNotFound:
            pos_or_pair = find_unique_relative(charges, PositionsNode, why_desc=purpose)
        return charges, pos_or_pair

    @_parent_expander.match(Charges, PositionsNode)
    @_parent_expander.match(Charges, PairIndexer)
    def expansion1(self, charges, pos_or_pair, *, purpose, **kwargs):
        species = find_unique_relative((pos_or_pair, charges), SpeciesNode, why_desc=purpose)
        return charges, pos_or_pair, species

    @_parent_expander.match(Charges, SpeciesNode)
    def expansion1(self, charges, species, *, purpose, **kwargs):
        positions = find_unique_relative((charges, species), PositionsNode, why_desc=purpose)
        return charges, positions, species

    @_parent_expander.match(Charges, _BaseNode, SpeciesNode)
    def expansion2(self, charges, pos_or_pair, species, *, purpose, **kwargs):
        encoder, pidxer = acquire_encoding_padding(species, species_set=None, purpose=purpose)
        return charges, pos_or_pair, encoder, pidxer

    @_parent_expander.match(Charges, PositionsNode, Encoder, PaddingIndexer)
    def expansion3(self, charges, positions, encoder, pidxer, *, cutoff_distance, **kwargs):
        try:
            pairfinder = find_unique_relative((charges, positions, encoder, pidxer), PairIndexer)
        except NodeNotFound:
            warnings.warn("Boundary conditions not specified, Building open boundary conditions.")
            pairfinder = OpenPairIndexer("PairIndexer", (positions, encoder, pidxer), dist_hard_max=cutoff_distance)
        return charges, pairfinder, pidxer

    @_parent_expander.match(Charges, PairIndexer, AtomIndexer)
    def expansion4(self, charges, pairfinder, pidxer, **kwargs):
        self._validate_pairfinder(pairfinder, None)
        pf = pairfinder
        return charges, pf.pair_dist, pf.pair_first, pf.pair_second, pidxer.mol_index, pidxer.n_molecules

    _parent_expander.assertlen(6)
    _parent_expander.get_main_outputs()
    _parent_expander.require_idx_states(IdxType.Atoms, *(None,) * 5)


class CoulombEnergyNode(ChargePairSetup, AutoKw, MultiNode):
    _input_names = "charges", "pair_dist", "pair_first", "pair_second", "mol_index", "n_molecules"
    _output_names = "mol_energies", "atom_voltages"
    _output_index_states = IdxType.Molecules, IdxType.Atoms
    _main_output = "mol_energies"
    _auto_module_class = physics_layers.CoulombEnergy

    @staticmethod
    def _validate_pairfinder(pairfinder, cutoff_distance):
        if not isinstance(pairfinder, OpenPairIndexer):
            raise TypeError(
                "Closed boundary conditions detected.\n"
                "Coulomb energy module is not compatible with open boundary conditions."
            )

        if pairfinder.torch_module.hard_dist_cutoff is not None:
            raise ValueError(
                "dist_hard_max is set to a finite value,\n"
                "coulomb energy requires summing over the entire set of pairs"
            )

    def __init__(self, name, parents, energy_conversion, module="auto"):
        """
        Besides the normal 'name' and 'parents' arguments, this node requires an `energy_conversion` parameter.
        This corresponds to coulomb's constant k in the equation E = kqq/r.
        """
        self.module_kwargs = {"energy_conversion_factor": energy_conversion}
        parents = self.expand_parents(parents, cutoff_distance=None)
        self.energy_conversion = energy_conversion
        super().__init__(name, parents, module=module)


class ScreenedCoulombEnergyNode(ChargePairSetup, AutoKw, SingleNode):
    _input_names = "charges", "pair_dist", "pair_first", "pair_second", "mol_index", "n_molecules"
    _output_names = "mol_energies", "atom_voltages"
    _index_state = IdxType.Molecules
    _auto_module_class = physics_layers.ScreenedCoulombEnergy

    @staticmethod
    def _validate_pairfinder(pairfinder, cutoff_distance):
        existing_cutoff = pairfinder.torch_module.hard_dist_cutoff
        if existing_cutoff is not None and existing_cutoff < cutoff_distance:
            raise ValueError(
                f"Distance cutoff ({existing_cutoff}) is set to less than\n"
                f" pair finder distance ({cutoff_distance}). Increase the cutoff distance\n"
                f" for the pair_finder (named: {pairfinder.name})"
            )

    def __init__(self, name, parents, energy_conversion, cutoff_distance, screening=None, module="auto"):
        """
        Besides the normal 'name' and 'parents' arguments, this node requires an `energy_conversion` parameter.
        This corresponds to coulomb's constant k in the equation E = kqq/r.
        """

        if screening is None and module == "auto":
            raise ValueError(
                "To build this module automatically a screening module must\n"
                "be provided (e.g. layers.physiscs.QScreening(p_value=4))"
            )

        self.module_kwargs = {
            "energy_conversion_factor": energy_conversion,
            "radius": cutoff_distance,
            "screening": screening,
        }
        parents = self.expand_parents(
            parents,
            cutoff_distance=cutoff_distance,
        )
        self.energy_conversion = energy_conversion
        super().__init__(name, parents, module=module)


class VecMag(ExpandParents, AutoNoKw, SingleNode):
    _input_names = ("vector",)
    _auto_module_class = physics_layers.VecMag
    _index_state = IdxType.NotFound

    @_parent_expander.match(_BaseNode, _BaseNode)
    def expansion2(self, vector, helper, *, purpose, **kwargs):
        # This somewhat strange construction allows us to
        # find a padding indexer if the vector is detached from the padding indexer.
        vector, helper = elementwise_compare_reduce(vector, helper)
        return (vector,)

    _parent_expander.assertlen(1)
    _parent_expander.get_main_outputs()

    def __init__(self, name, parents, module="auto", _helper=None, **kwargs):
        parents = self.expand_parents(parents)
        self._index_state = parents[0]._index_state
        assert len(parents) == 1, "Improper number of parents for {}".format(self.__class__.__name__)
        super().__init__(name, parents, module=module, **kwargs)


class AtomToMolSummer(ExpandParents, AutoNoKw, SingleNode):
    _input_names = "features", "mol_index", "n_molecules"
    _auto_module_class = index_layers.MolSummer
    _index_state = IdxType.Molecules

    @_parent_expander.match(_BaseNode)
    def expansion0(self, features, **kwargs):
        pdxer = find_unique_relative(features, AtomIndexer, why_desc="Generating Molecular summer")
        return features, pdxer

    @_parent_expander.match(_BaseNode, AtomIndexer)
    def expansion1(self, features, pdxer, **kwargs):
        return features, pdxer.mol_index, pdxer.n_molecules

    _parent_expander.assertlen(3)
    _parent_expander.get_main_outputs()
    _parent_expander.require_idx_states(IdxType.Atoms, None, None)

    def __init__(self, name, parents, module="auto", **kwargs):
        parents = self.expand_parents(parents)
        super().__init__(name, parents, module=module, **kwargs)


# TODO: This seems broken for parent expanders, check the signature of the layer.
class BondToMolSummmer(ExpandParents, AutoNoKw, SingleNode):

    _input_names = "pairfeatures", "mol_index", "n_molecules", "pair_first"
    _auto_module_class = pair_layers.MolPairSummer
    _index_state = IdxType.Molecules

    @_parent_expander.match(_BaseNode)
    def expansion0(self, features, *, purpose, **kwargs):
        pdxer = find_unique_relative(features, AtomIndexer, why_desc=purpose)
        pair_idxer = find_unique_relative(features, PairIndexer, why_desc=purpose)
        return features, pdxer, pair_idxer

    @_parent_expander.match(_BaseNode, AtomIndexer, PairIndexer)
    def expansion1(self, features, pdxer, pair_idxer, **kwargs):
        return features, pdxer.mol_index, pdxer.n_molecules, pair_idxer.pair_first

    @_parent_expander.match(_BaseNode, _BaseNode, _BaseNode, _BaseNode, _BaseNode)
    def expansion2(self, features, mol_index, n_molecules, **kwargs):
        return index_type_coercion(features.main_output, IdxType.Pair), mol_index, n_molecules

    def __init__(self, name, parents, module="auto", **kwargs):
        parents = self.expand_parents(parents)
        super().__init__(name, parents, module=module, **kwargs)


class PerAtom(ExpandParents, AutoNoKw, SingleNode):
    _input_names = "features", "species"
    _index_state = IdxType.Molecules
    _auto_module_class = physics_layers.PerAtom

    @_parent_expander.match(_BaseNode)
    def expansion0(self, features, *, purpose, **kwargs):
        return features, find_unique_relative(features, SpeciesNode, purpose)

    @_parent_expander.match(_BaseNode, _BaseNode)
    def expansion1(self, features, species, **kwargs):
        features = features.main_output
        assert (
            features._index_state == IdxType.Molecules
        ), "Can only calculate Per Atom averages on Molecular quantities"
        return features, species

    def __init__(self, name, parents, module="auto", **kwargs):
        parents = self.expand_parents(parents)
        super().__init__(name, parents, module=module, **kwargs)
