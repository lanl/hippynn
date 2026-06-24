"""
Node for charge equilibration model.
"""
from ...._deprecations import _DeprecatedNamesMixin

from ....layers import physics as physics_layers 

from ..base import MultiNode, AutoKw, find_unique_relative, find_relatives, ExpandParents, Node
from ...indextypes import IdxType
from ..networks import Network
from ..targets import HChargeNode
from ..inputs import PositionsNode, SpeciesNode, CellNode
from ..tags import Charges, Energies, PairIndexer, AtomIndexer
from ..indexers import acquire_encoding_padding
from ..pairs import OpenPairIndexer



# Setup for Coulomb Energy and Screened Coulomb Energy is nearly the same, up to validating the pair finder.
class ChargePairSetup(ExpandParents):
    parent_expansion_kwargs = "_pe_cutoff_distance",
    

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

    @parent_expander.match(Charges, PositionsNode, AtomIndexer)
    def expansion3(self, charges, positions, pidxer, *, _pe_cutoff_distance, **kwargs):
        try:
            pairfinder = find_unique_relative((charges, positions, pidxer), PairIndexer)
        except NodeNotFound:
            warnings.warn("Boundary conditions not specified, Building open boundary conditions.")
            encoder = find_unique_relative(pidxer, Encoder)
            pairfinder = OpenPairIndexer("PairIndexer", (positions, encoder, pidxer), dist_hard_max=_pe_cutoff_distance)
        return charges, pairfinder, pidxer

    @parent_expander.match(Charges, PairIndexer, AtomIndexer)
    def expansion4(self, charges, pairfinder, pidxer, *, _pe_cutoff_distance, **kwargs):
        self._validate_pairfinder(pairfinder, _pe_cutoff_distance)
        
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
    auto_module_kwargs = "energy_conversion_factor",

    def __init__(self, name, parents, energy_conversion_factor, module="auto", **kwargs):
        
        super().__init__(name, parents,
                         energy_conversion_factor=energy_conversion_factor,
                         _pe_cutoff_distance=None,
                         module=module,
                         **kwargs)


    
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

    @staticmethod
    def _validate_pairfinder(pairfinder, cutoff_distance):
        existing_cutoff = pairfinder.torch_module.hard_dist_cutoff
        if existing_cutoff is not None and existing_cutoff < cutoff_distance:
            raise ValueError(
                f"Distance cutoff ({existing_cutoff}) is set to less than\n"
                f"pair finder distance ({cutoff_distance}). Increase the cutoff distance\n"
                f"for the pair_finder (named: {pairfinder.name})"
            )

    def __init__(self, name, parents, energy_conversion_factor, cutoff_distance, screening=None, module="auto", **kwargs):
        
        if screening is None and module == "auto":
            raise ValueError(
                "To build this module automatically a screening module must\n"
                "be provided (e.g. layers.physiscs.QScreening(p_value=4))"
            )
        
        
        # Dev Note: the _pe_cutoff_distance argument duplicates the cutoff_distance argument
        # because the AutoKw and ExpandParent mixins both consume their keywords.
        # Since both of them require the cutoff, we have it supplied with two different names.
        # Would be nice if the workflow didn't require this, but the workaround is not costly,
        # just confusing to find.

        super().__init__(name, parents,
                        energy_conversion_factor=energy_conversion_factor,
                         cutoff_distance=cutoff_distance,
                         _pe_cutoff_distance=cutoff_distance,
                         screening=screening,
                         module=module,
                         **kwargs)


class ChEQNode(ExpandParents, AutoKw, MultiNode):
    input_names = "species", "coordinates", "U", "chi"
    output_names = "charge", "coul_energy", "dipole", "out_U", "out_chi"
    output_index_states = (
        IdxType.SysAtom,
        IdxType.Systems,
        IdxType.Systems,
        IdxType.SysAtom,
        IdxType.SysAtom,
    )

    main_output = "charge"
    auto_module_class = physics_layers.ChEQ

    @parent_expander.match(Network)
    def expand0(self, network, **kwargs):
        U = HChargeNode("ChEQ_U", network, module_kwargs=dict(first_is_interacting=False))
        chi = HChargeNode("ChEQ_chi", network, module_kwargs=dict(first_is_interacting=False))
        return U, chi

    @parent_expander.match(Network, Network)
    def expand1(self, network1, network2, **kwargs):
        U = HChargeNode("ChEQ_U", network1, module_kwargs=dict(first_is_interacting=False))
        chi = HChargeNode("ChEQ_chi", network2, module_kwargs=dict(first_is_interacting=False))
        return U, chi

    @parent_expander.match(HChargeNode, HChargeNode)
    def expand2(self, U, chi, **kwargs):
        positions = find_unique_relative([U, chi], PositionsNode)
        species = find_unique_relative([U, chi], SpeciesNode)

        return species, positions, U.main_output, chi.main_output

    @parent_expander.match(Node, Node, Node, Node)
    def warn_if_pbc_detected(self, *parents, **kwargs):
        try:
            cell_nodes = find_relatives(parents, CellNode)
        except:
            import warnings
            warnings.warn("Periodic boundaries were detected in the graph; " +\
                          "This ChEQ node computes using open boundary conditions only",
                          stacklevel=3
                          )
        return parents


    def __init__(self, name, parents, lower_bound=0.0, units={"energy": "eV", "length": "Angstrom"}, module="auto", **kwargs):
        parents = self.expand_parents(parents, **kwargs)
        self.module_kwargs = dict(lower_bound=lower_bound, units=units)
        super().__init__(name, parents, module=module, **kwargs)
