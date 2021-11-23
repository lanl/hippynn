"""
Nodes for usage with hippynn node system.
"""
import torch

from .seqm_modules import Scale
from .seqm_modules import SEQM_Energy, SEQM_All
from .seqm_modules import SEQM_MolMask, AtomMask, SEQM_OrbitalMask
from .seqm_modules import SEQM_MaskOnMol, SEQM_MaskOnMolAtom, SEQM_MaskOnMolOrbital, SEQM_MaskOnMolOrbitalAtom
from hippynn.graphs.nodes.base import MultiNode, AutoKw, find_unique_relative, ExpandParents, SingleNode
from hippynn.graphs.indextypes import IdxType
from hippynn.graphs.nodes.networks import Network
from hippynn.graphs.nodes.targets import HChargeNode, HBondNode
from hippynn.graphs.nodes.inputs import PositionsNode, SpeciesNode


class ScaleNode(AutoKw, SingleNode):
    _input_names = "notconverged"
    _auto_module_class = Scale
    _index_state = IdxType.Scalar

    def __init__(self, name, parents, func=torch.sqrt, module="auto", **kwargs):
        self.module_kwargs = {"func": func}
        super().__init__(name, parents, module=module, **kwargs)


class SEQM_MolMaskNode(AutoKw, SingleNode):
    _input_names = "notconverged"
    _auto_module_class = SEQM_MolMask
    _index_state = IdxType.Molecules

    def __init__(self, name, parents, module="auto", **kwargs):
        parents = (parents,)
        self.module_kwargs = {}
        super().__init__(name, parents, module=module, **kwargs)


class AtomMaskNode(AutoKw, SingleNode):
    _input_names = "species"
    _auto_module_class = AtomMask
    _index_state = IdxType.Molecules

    def __init__(self, name, parents, module="auto", **kwargs):
        parents = (parents,)
        self.module_kwargs = {}
        super().__init__(name, parents, module=module, **kwargs)


class SEQM_OrbitalMaskNode(AutoKw, SingleNode):
    _input_names = "species"
    _auto_module_class = SEQM_OrbitalMask
    _index_state = IdxType.Molecules

    def __init__(self, name, parents, target_method, nOccVirt=None, module="auto", **kwargs):
        parents = (parents,)
        self.module_kwargs = {"target_method": target_method, "nOccVirt": nOccVirt}
        super().__init__(name, parents, module=module, **kwargs)


class SEQM_MaskOnMolNode(AutoKw, SingleNode):
    """
    used for quantity like total energy, Heat of formation
    with shape (molecules,)
    """

    _input_names = "var", "mol_mask"
    _auto_module_class = SEQM_MaskOnMol
    _index_state = IdxType.NotFound

    def __init__(self, name, parents, module="auto", **kwargs):
        self.module_kwargs = {}
        super().__init__(name, parents, module=module, **kwargs)


class SEQM_MaskOnMolAtomNode(AutoKw, SingleNode):
    """
    used for quantity like force
    with shape (molecules,atoms)
    """

    _input_names = "var", "mol_mask", "atom_mask"
    _auto_module_class = SEQM_MaskOnMolAtom
    _index_state = IdxType.NotFound

    def __init__(self, name, parents, module="auto", **kwargs):
        self.module_kwargs = {}
        super().__init__(name, parents, module=module, **kwargs)


class SEQM_MaskOnMolOrbitalNode(AutoKw, SingleNode):
    """
    used for quantity like orbital energy
    with shape (molecules,orbitals)
    """

    _input_names = "var", "mol_mask", "orbital_mask"
    _auto_module_class = SEQM_MaskOnMolOrbital
    _index_state = IdxType.NotFound

    def __init__(self, name, parents, module="auto", **kwargs):
        self.module_kwargs = {}
        super().__init__(name, parents, module=module, **kwargs)


class SEQM_MaskOnMolOrbitalAtomNode(AutoKw, SingleNode):
    """
    used for quantity like orbital charge densit
    with shape (molecules,orbitals, atoms)
    """

    _input_names = "var", "mol_mask", "orbital_mask", "atom_mask"
    _auto_module_class = SEQM_MaskOnMolOrbitalAtom
    _index_state = IdxType.NotFound

    def __init__(self, name, parents, module="auto", **kwargs):
        self.module_kwargs = {}
        super().__init__(name, parents, module=module, **kwargs)


class SEQM_EnergyNode(ExpandParents, AutoKw, MultiNode):
    _input_names = "par_atom", "Positions", "Species"
    _output_names = "mol_energy", "Etot_m_Eiso", "notconverged"
    _main_output = "mol_energy"
    _output_index_states = (IdxType.Molecules,) * len(_output_names)
    _auto_module_class = SEQM_Energy

    @_parent_expander.match(Network)
    def expand0(self, network, seqm_parameters, decay_factor=1.0e-2, **kwargs):

        n_target_peratom = len(seqm_parameters["learned"])

        par_atom = HChargeNode(
            "SEQM_Atom_Params", network, module_kwargs=dict(n_target=n_target_peratom, first_is_interacting=True)
        )

        with torch.no_grad():
            for layer in par_atom.torch_module.layers:
                layer.weight.data *= decay_factor
                layer.bias.data *= decay_factor

        positions = find_unique_relative(network, PositionsNode)
        species = find_unique_relative(network, SpeciesNode)

        return par_atom.main_output, positions, species

    _parent_expander.assertlen(3)
    _parent_expander.get_main_outputs()
    _parent_expander.require_idx_states(IdxType.Atoms, None, None)

    def __init__(self, name, parents, seqm_parameters, decay_factor=1.0e-2, module="auto", **kwargs):
        parents = self.expand_parents(parents, seqm_parameters=seqm_parameters, decay_factor=decay_factor, **kwargs)
        self.module_kwargs = dict(seqm_parameters=seqm_parameters)
        super().__init__(name, parents, module=module, **kwargs)


class SEQM_AllNode(SEQM_EnergyNode):
    _input_names = "par_atom", "Positions", "Species"
    _output_names = (
        "mol_energy",
        "Etot_m_Eiso",
        "orbital_energies",
        "single_particle_density_matrix",
        "electric_energy",
        "nuclear_energy",
        "isolated_atom_energy",
        "orbital_charges",
        "notconverged",
        "atomic_charge",
    )
    _main_output = "mol_energy"
    _output_index_states = (IdxType.Molecules,) * len(_output_names)
    _auto_module_class = SEQM_All
