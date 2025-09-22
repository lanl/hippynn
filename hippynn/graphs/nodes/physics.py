"""
Nodes for physics transformations
"""
import warnings

from ...layers import indexers as index_layers
from ...layers import pairs as pair_layers
from ...layers import physics as physics_layers
from ..indextypes import IdxType, elementwise_compare_reduce, index_type_coercion
from .base import (
    AutoKw,
    AutoNoKw,
    ExpandParents,
    MultiNode,
    SingleNode,
    Node,
    find_unique_relative,
)
from .base.node_functions import NodeNotFound
from .indexers import AtomIndexer, PaddingIndexer, acquire_encoding_padding
from .inputs import PositionsNode, SpeciesNode
from .pairs import OpenPairIndexer
from .tags import Charges, Encoder, Energies, PairIndexer
from ..._deprecations import _DeprecatedNamesMixin

class GradientNode(AutoKw, SingleNode):
    """
    Compute the gradient of a quantity.
    """

    input_names = "energy", "coordinates"
    auto_module_class = physics_layers.Gradient
    auto_module_kwargs = "sign",

    def __init__(self, name, parents, sign, **kwargs):
        energy, position = parents
        position.requires_grad = True
        parents = energy.main_output, position
        self.sign = sign
        self.index_state = position.index_state
        super().__init__(name, parents, sign=sign, **kwargs)
        
class MultiGradientNode(AutoKw, MultiNode):
    """
    Compute the gradient of a quantity.
    """

    auto_module_class = physics_layers.MultiGradient
    auto_module_kwargs = "signs",


    def __init__(self, name: str, molecular_energies_parent: Node, generalized_coordinates_parents: tuple[Node], signs: tuple[int], **kwargs):
        if isinstance(signs, int):
            signs = (signs,)

        self.signs = signs

        parents = molecular_energies_parent, *generalized_coordinates_parents

        for parent in generalized_coordinates_parents:
            parent.requires_grad = True        

        self.input_names = tuple((parent.name for parent in parents))
        self.output_names = tuple((parent.name + "_grad" for parent in generalized_coordinates_parents))
        self.output_index_states = tuple(parent.index_state for parent in generalized_coordinates_parents)

        super().__init__(name, parents, signs=signs, **kwargs)


class StressForceNode(AutoNoKw, MultiNode):
    input_names = "energy", "strain", "coordinates", "cell"
    output_names = "forces", "stress"
    auto_module_class = physics_layers.StressForce

    def __init__(self, name, parents, module="auto", **kwargs):
        energy, strain, coordinates, cell = parents
        coordinates.requires_grad = True
        parents = energy.main_output, strain, coordinates, cell
        self.output_index_states = coordinates.index_state, strain.index_state
        super().__init__(name, parents, module=module, **kwargs)


class ChargeMomentNode(ExpandParents, AutoNoKw, SingleNode):
    input_names = "charges", "positions", "system_index", "n_systems"

    @parent_expander.matchlen(1)
    def expansion0(self, charges, *, purpose, **kwargs):
        return charges, find_unique_relative(charges, PositionsNode, why_desc=purpose)

    @parent_expander.match(Charges, PositionsNode)
    def expansion1(self, charges, positions, *, purpose, **kwargs):
        enc, pidxer = acquire_encoding_padding((charges, positions), species_set=None, purpose=purpose)
        return charges, positions, pidxer

    @parent_expander.match(Charges, PositionsNode, AtomIndexer)
    def expansion2(self, charges, positions, pdxer, **kwargs):
        return charges, positions, pdxer.system_index, pdxer.n_systems

    parent_expander.assertlen(4)
    parent_expander.get_main_outputs()
    parent_expander.require_idx_states(IdxType.Atoms, IdxType.Atoms, None, None)

    def __init__(self, name, parents, module="auto", **kwargs):
        parents = self.expand_parents(parents)
        super().__init__(name, parents, module=module, **kwargs)


class DipoleNode(ChargeMomentNode):
    """
    Compute the dipole of point charges.
    """

    auto_module_class = physics_layers.Dipole
    index_state = IdxType.Systems


class QuadrupoleNode(ChargeMomentNode):
    """
    Compute the traceless quadrupole of point charges.
    """

    auto_module_class = physics_layers.Quadrupole
    index_state = IdxType.QuadMol


# Setup for Coulomb Energy and Screened Coulomb Energy is nearly the same, up to validating the pair finder.
class ChargePairSetup(ExpandParents):
    @staticmethod
    def _validate_pairfinder(pairfinder, cutoff_distance):
        # This method required by this ExpandParents setup.
        # Raises an error if the pairfinder is not satisfactory.
        return NotImplemented

    @parent_expander.match(Charges)
    def expansion0(self, charges, *, purpose, **kwargs):
        try:
            pos_or_pair = find_unique_relative(charges, PairIndexer, why_desc=purpose)
        except NodeNotFound:
            pos_or_pair = find_unique_relative(charges, PositionsNode, why_desc=purpose)
        return charges, pos_or_pair

    @parent_expander.match(Charges, PositionsNode)
    @parent_expander.match(Charges, PairIndexer)
    def expansion1(self, charges, pos_or_pair, *, purpose, **kwargs):
        species = find_unique_relative((pos_or_pair, charges), SpeciesNode, why_desc=purpose)
        return charges, pos_or_pair, species

    @parent_expander.match(Charges, SpeciesNode)
    def expansion1(self, charges, species, *, purpose, **kwargs):
        positions = find_unique_relative((charges, species), PositionsNode, why_desc=purpose)
        return charges, positions, species

    @parent_expander.match(Charges, Node, SpeciesNode)
    def expansion2(self, charges, pos_or_pair, species, *, purpose, **kwargs):
        encoder, pidxer = acquire_encoding_padding(species, species_set=None, purpose=purpose)
        return charges, pos_or_pair, pidxer

    @parent_expander.match(Charges, PositionsNode, PaddingIndexer)
    def expansion3(self, charges, positions, pidxer, *, cutoff_distance, **kwargs):
        try:
            pairfinder = find_unique_relative((charges, positions, pidxer), PairIndexer)
        except NodeNotFound:
            warnings.warn("Boundary conditions not specified, Building open boundary conditions.")
            encoder = find_unique_relative(pidxer, Encoder)
            pairfinder = OpenPairIndexer("PairIndexer", (positions, encoder, pidxer), dist_hard_max=cutoff_distance)
        return charges, pairfinder, pidxer

    @parent_expander.match(Charges, PairIndexer, AtomIndexer)
    def expansion4(self, charges, pairfinder, pidxer, *, cutoff_distance, **kwargs):
        self._validate_pairfinder(pairfinder, cutoff_distance)
        pf = pairfinder
        return charges, pf.pair_dist, pf.pair_first, pf.pair_second, pidxer.system_index, pidxer.n_systems

    parent_expander.assertlen(6)
    parent_expander.get_main_outputs()
    parent_expander.require_idx_states(IdxType.Atoms, *(None,) * 5)


class CoulombEnergyNode(AutoKw, ChargePairSetup, Energies,  MultiNode, _DeprecatedNamesMixin):
    """
    Besides the normal 'name' and 'parents' arguments, this node requires an `energy_conversion` parameter.
    This corresponds to coulomb's constant k in the equation E = kqq/r.
    """
    _DEPRECATED_NAMES = {"mol_energies": "system_energies"}
    input_names = "charges", "pair_dist", "pair_first", "pair_second", "system_index", "n_systems"
    output_names = "system_energies", "atom_energies", "atom_voltages"
    output_index_states = IdxType.Systems, IdxType.Atoms, IdxType.Atoms
    main_output_name = "system_energies"
    auto_module_class = physics_layers.CoulombEnergy
    auto_module_kwargs = "energy_conversion_factor"

    

    
    @staticmethod
    def _validate_pairfinder(pairfinder, cutoff_distance):
        if not isinstance(pairfinder, OpenPairIndexer):
            raise TypeError(
                "Closed boundary conditions detected.\n"
                "Coulomb energy module is not compatible with closed boundary conditions."
            )

        if pairfinder.torch_module.hard_dist_cutoff is not None:
            raise ValueError(
                "hard_dist_cutoff is set to a finite value,\n"
                "coulomb energy requires summing over the entire set of pairs"
            )



class ScreenedCoulombEnergyNode(AutoKw, ChargePairSetup, Energies, MultiNode, _DeprecatedNamesMixin):
    """
    Besides the normal 'name' and 'parents' arguments, this node requires an `energy_conversion` parameter.
    This corresponds to coulomb's constant k in the equation E = kqq/r.
    """
    _DEPRECATED_NAMES = {"mol_energies": "system_energies"}
    input_names = "charges", "pair_dist", "pair_first", "pair_second", "system_index", "n_systems"
    output_names = "system_energies", "atom_energies", "atom_voltages"
    output_index_states = IdxType.Systems, IdxType.Atoms, IdxType.Atoms
    main_output_name = "system_energies"
    auto_module_class = physics_layers.ScreenedCoulombEnergy
    auto_module_kwargs = {
        "energy_conversion_factor":"energy_conversion_factor",
        "radius": "cutoff_distance",
        "screening": "screening",
    }
    parent_expansion_kwargs = "cutoff_distance",

    @staticmethod
    def _validate_pairfinder(pairfinder, cutoff_distance):
        existing_cutoff = pairfinder.torch_module.hard_dist_cutoff
        if existing_cutoff is not None and existing_cutoff < cutoff_distance:
            raise ValueError(
                f"Distance cutoff ({existing_cutoff}) is set to less than\n"
                f" pair finder distance ({cutoff_distance}). Increase the cutoff distance\n"
                f" for the pair_finder (named: {pairfinder.name})"
            )

    def __init__(self, name, parents, energy_conversion_factor, cutoff_distance, screening=None, module="auto", **kwargs):
        
        if screening is None and module == "auto":
            raise ValueError(
                "To build this module automatically a screening module must\n"
                "be provided (e.g. layers.physiscs.QScreening(p_value=4))"
            )
        super().__init__(name, parents,
                        energy_conversion_factor=energy_conversion_factor,
                         cutoff_distance=cutoff_distance,
                         screening=screening,
                         module=module,
                         **kwargs)


class VecMag(ExpandParents, AutoNoKw, SingleNode):
    input_names = ("vector",)
    auto_module_class = physics_layers.VecMag
    index_state = IdxType.Unlabeled

    @parent_expander.match(Node, Node)
    def expansion2(self, vector, helper, *, purpose, **kwargs):
        # This somewhat strange construction allows us to
        # find a padding indexer if the vector is detached from the padding indexer.
        vector, helper = elementwise_compare_reduce(vector, helper)
        return (vector,)

    parent_expander.assertlen(1)
    parent_expander.get_main_outputs()

    def __init__(self, name, parents, module="auto", _helper=None, **kwargs):
        parents = self.expand_parents(parents)
        self.index_state = parents[0].index_state
        assert len(parents) == 1, "Improper number of parents for {}".format(self.__class__.__name__)
        super().__init__(name, parents, module=module, **kwargs)


class AtomToMolSummer(ExpandParents, AutoNoKw, SingleNode):
    input_names = "features", "system_index", "n_systems"
    auto_module_class = index_layers.MolSummer
    index_state = IdxType.Systems

    @parent_expander.match(Node)
    def expansion0(self, features, **kwargs):
        pdxer = find_unique_relative(features, AtomIndexer, why_desc="Generating Molecular summer")
        return features, pdxer

    @parent_expander.match(Node, AtomIndexer)
    def expansion1(self, features, pdxer, **kwargs):
        return features, pdxer.system_index, pdxer.n_systems

    parent_expander.assertlen(3)
    parent_expander.get_main_outputs()
    parent_expander.require_idx_states(IdxType.Atoms, None, None)

    def __init__(self, name, parents, module="auto", **kwargs):
        parents = self.expand_parents(parents)
        super().__init__(name, parents, module=module, **kwargs)


# TODO: This seems broken for parent expanders, check the signature of the layer.
class BondToMolSummmer(ExpandParents, AutoNoKw, SingleNode):
    input_names = "pairfeatures", "system_index", "n_systems", "pair_first"
    auto_module_class = pair_layers.MolPairSummer
    index_state = IdxType.Systems

    @parent_expander.match(Node)
    def expansion0(self, features, *, purpose, **kwargs):
        pdxer = find_unique_relative(features, AtomIndexer, why_desc=purpose)
        pair_idxer = find_unique_relative(features, PairIndexer, why_desc=purpose)
        return features, pdxer, pair_idxer

    @parent_expander.match(Node, AtomIndexer, PairIndexer)
    def expansion1(self, features, pdxer, pair_idxer, **kwargs):
        return features, pdxer.system_index, pdxer.n_systems, pair_idxer.pair_first

    @parent_expander.match(Node, Node, Node, Node, Node)
    def expansion2(self, features, system_index, n_systems, **kwargs):
        return index_type_coercion(features.main_output, IdxType.Pairs), system_index, n_systems

    def __init__(self, name, parents, module="auto", **kwargs):
        parents = self.expand_parents(parents)
        super().__init__(name, parents, module=module, **kwargs)


class PerAtom(ExpandParents, AutoNoKw, SingleNode):
    input_names = "features", "species"
    index_state = IdxType.Systems
    auto_module_class = physics_layers.PerAtom

    @parent_expander.match(Node)
    def expansion0(self, features, *, purpose, **kwargs):
        return features, find_unique_relative(features, SpeciesNode, purpose)

    @parent_expander.match(Node, Node)
    def expansion1(self, features, species, **kwargs):
        features = features.main_output
        assert (
            features.index_state == IdxType.Systems
        ), "Can only calculate Per Atom averages on Molecular quantities"
        return features, species

    def __init__(self, name, parents, module="auto", **kwargs):
        parents = self.expand_parents(parents)
        super().__init__(name, parents, module=module, **kwargs)


class CombineEnergyNode(AutoNoKw, Energies,  ExpandParents, MultiNode):
    """
    Combines Local atom energies from different Energy Nodes.
    """

    input_names = "input_atom_energy_1", "input_atom_energy_2", "system_index", "n_systems"
    output_names = "mol_energy", "atom_energies"
    main_output_name = "mol_energy"
    output_index_states = (
        IdxType.Systems,
        IdxType.Atoms,
    )
    auto_module_class = physics_layers.CombineEnergy

    @parent_expander.match(Node, Energies)
    def expansion0(self, energy_1, energy_2, **kwargs):
        return energy_1, energy_2.atom_energies

    @parent_expander.match(Energies, Node)
    def expansion0(self, energy_1, energy_2, **kwargs):
        return energy_1.atom_energies, energy_2

    @parent_expander.match(Node, Node)
    def expansion1(self, energy_1, energy_2, **kwargs):
        pdindexer = find_unique_relative([energy_1, energy_2], AtomIndexer, why_desc="Generating CombineEnergies")
        return energy_1, energy_2, pdindexer

    @parent_expander.match(Node, Node, PaddingIndexer)
    def expansion2(self, energy_1, energy_2, pdindexer, **kwargs):
        return energy_1, energy_2, pdindexer.system_index, pdindexer.n_systems

    parent_expander.assertlen(4)
    parent_expander.require_idx_states(IdxType.Atoms, IdxType.Atoms, None, None)



class StrainInducer(AutoNoKw, MultiNode):
    input_names = "coordinates", "cell"
    output_names = "strained_coordinates", "strained_cell", "strain"
    output_index_states = NotImplemented
    auto_module_class = physics_layers.CellScaleInducer

    def __init__(self, name, parents, module="auto", **kwargs):
        position, cell = parents
        self.output_index_states = position.index_state, IdxType.Unlabeled, IdxType.Unlabeled
        super().__init__(name, parents, module=module, **kwargs)

    
def setup_stressforce_nodes(energy_node, return_transformed_inputs=False, positions_node="auto", cell_node="auto", strain_node="auto"):
    """_summary_

    :param energy_node: the energy to differenitate
    :param return_transformed_inputs: If true, return the strained positions, strained cell, and strain
    :param position_node: defaults to "auto"
    :param cell_node: defaults to "auto"
    :param strain_node: defaults to "auto"

    Using "auto" will cause a failure if the corresponding node cannot be found or is ambiguous.

    :return: (forces, stress) or (forces, stress, strained_positions, strained_cell, strain) depending on return_transformed_inputs flag.
    """

    from .misc import StrainInducer

    from .tags import Positions
    from .inputs import CellNode
    
    if positions_node == "auto":
        positions_node = find_unique_relative(energy_node, Positions)

    if cell_node == "auto":
        cell_node = find_unique_relative(energy_node, CellNode)
    
    if strain_node == "auto":
        strain_node = StrainInducer("Strain_inducer", (positions_node, cell_node))
    
    strained_coords = strain_node.strained_coordinates
    strained_cell = strain_node.strained_cell
    strain = strain_node.strain

    from hippynn.graphs.gops import replace_node

    replace_node(positions_node, strained_coords)
    replace_node(cell_node, strained_cell)

    derivatives = StressForceNode("StressForceCalculator", (energy_node, strain, positions_node, cell_node))
    forces, stress = derivatives.forces, derivatives.stress

    if return_transformed_inputs:

        return stress, forces, strained_coords, strained_cell, strain
    
    else:
        return stress, forces
