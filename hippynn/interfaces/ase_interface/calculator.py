import functools

import numpy as np
import warnings
import torch

from ase.calculators.calculator import compare_atoms, PropertyNotImplementedError, Calculator

from hippynn.graphs import find_relatives, find_unique_relative, get_subgraph, copy_subgraph, replace_node, GraphModule
from hippynn.graphs.gops import check_link_consistency

from hippynn.graphs.nodes.base.node_functions import NodeOperationError, NodeNotFound
from hippynn.graphs.nodes.base import InputNode
from hippynn.graphs.nodes.tags import Encoder, AtomIndexer, PairIndexer, Energies, Charges
from hippynn.graphs.nodes.pairs import ExternalNeighborIndexer
from hippynn.graphs.nodes.misc import StrainInducer
from hippynn.graphs.nodes.physics import CoulombEnergyNode, DipoleNode, StressForceNode
from hippynn.graphs.nodes.pairs import PairFilter
from hippynn.graphs.nodes.targets import AtomizationEnergyNode
from hippynn.graphs.nodes.inputs import SpeciesNode, PositionsNode, CellNode


import ase.neighborlist

# TODO: implement neighbors using scipy.spatial.cKDTree?
# This works for orthorhombic boxes and is much faster than ASE...


def setup_ASE_graph(energy, charges=None, extra_properties=None, species_set=None,indexer=None):
    if isinstance(energy, AtomizationEnergyNode):
        energy = energy.create_henergy_equivalent()

    if charges is None:
        required_nodes = [energy]
    else:
        required_nodes = [energy, charges]

    if extra_properties is not None:
        extra_names = list(extra_properties.keys())
        for ename in extra_names:
            if not ename.isidentifier():
                raise ValueError("ASE properties must be a valid python identifier. (Got '{}')".format(ename))
        del ename
        required_nodes = required_nodes + list(extra_properties.values())

    why = "Generating ASE Calculator interface"
    subgraph = get_subgraph(required_nodes)

    ########################################
    # TODO: Implement Ewald, Wolf, or similar version of coulomb energy?
    # Better: figure out how to get voltages from external code and pass back as gradients for HIPNN backwards pass.
    ########################################

    ###############################################################
    # Get a new subgraph, and find the nodes we need to construct the calculator
    # Factory for seeking out nodes only in the subgraph and of a specific type
    search_fn = lambda targ, sg: lambda n: n in sg and isinstance(n, targ)

    try:
        pair_indexers = find_relatives(required_nodes, search_fn(PairIndexer, subgraph), why_desc=why)
    except NodeOperationError as ee:
        raise ValueError(
            "No Pair indexers found. Why build an ASE interface with no need for neighboring atoms?"
        ) from ee

    # The required nodes passed back are copies of the ones passed in.
    # We use assume_inputed to avoid grabbing pieces of the graph
    # that are only prerequisites for the pair indexer.
    new_required, new_subgraph = copy_subgraph(required_nodes, assume_inputed=pair_indexers)
    # We now need access to the copied indexers, rather than the originals
    pair_indexers = find_relatives(new_required, search_fn(PairIndexer, new_subgraph), why_desc=why)

    species = find_unique_relative(new_required, search_fn(SpeciesNode, new_subgraph), why_desc=why)
    positions = find_unique_relative(new_required, search_fn(PositionsNode, new_subgraph), why_desc=why)

    # TODO: is .clone necessary? Or good? Or torch.as_tensor instead?
    if species_set is None:
        encoder = find_unique_relative(species, search_fn(Encoder, new_subgraph), why_desc=why)
        species_set = torch.as_tensor(encoder.species_set).to(torch.int64)  # works with lists or tensors
    if indexer is None:
        indexer = find_unique_relative(species, search_fn(AtomIndexer, new_subgraph), why_desc=why)
    min_radius = max(p.dist_hard_max for p in pair_indexers)
    ###############################################################

    ###############################################################
    # Set up graph to accept external pair indices and shifts

    in_shift = InputNode("shift_vector")
    in_cell = CellNode("cell")
    in_pair_first = InputNode("pair_first")
    in_pair_second = InputNode("pair_second")
    external_pairs = ExternalNeighborIndexer(
        "external_neighbors",
        (positions, indexer.real_atoms, in_shift, in_cell, in_pair_first, in_pair_second),
        hard_dist_cutoff=min_radius,
    )
    new_inputs = [species, positions, in_cell, in_pair_first, in_pair_second, in_shift]

    # Construct Filters
    # Replace the existing pair indexers with the corresponding new (filtered) node
    # that accepts external pairs of atoms:
    # (This is the primary reason we needed to copy the subgraph --)
    #  we don't want to break the original computation, and `replace_node` mutates graph connectivity
    for pi in pair_indexers:
        if pi.dist_hard_max == min_radius:
            mapped_node = external_pairs
        else:
            mapped_node = PairFilter(
                "DistanceFilter_external_neighbors",
                (external_pairs),
                dist_hard_max=pi.dist_hard_max, 
            )
        replace_node(pi, mapped_node, disconnect_old=True)
    ###############################################################

    ###############################################################
    # Set up gradient and, if possible, dipole properties.

    energy, *new_required = new_required

    cellscaleinducer = StrainInducer("Strain_inducer", (positions, in_cell))
    strain = cellscaleinducer.strain
    derivatives = StressForceNode("StressForceCalculator", (energy, strain, positions, in_cell))

    replace_node(positions, cellscaleinducer.strained_coordinates)
    replace_node(in_cell, cellscaleinducer.strained_cell)

    implemented_nodes = energy.main_output, derivatives.forces, derivatives.stress
    implemented_properties = ["potential_energy", "forces", "stress"]

    pbc_handler = PBCHandle(derivatives)

    if charges is not None:
        charges, *new_required = new_required
        dipole_moment = DipoleNode("Dipole", charges)
        implemented_nodes = *implemented_nodes, charges.main_output, dipole_moment
        implemented_properties = implemented_properties + ["charges", "dipole_moment"]

    #### Add other properties here:
    if extra_properties is not None:
        implemented_nodes = *implemented_nodes, *new_required
        implemented_properties = implemented_properties + extra_names

    ###############################################################

    # Finally, assemble the graph!
    check_link_consistency((*new_inputs, *implemented_nodes))
    mod = GraphModule(new_inputs, implemented_nodes)
    mod.eval()

    return min_radius, species_set, implemented_properties, mod, pbc_handler


class PBCHandle:
    def __init__(self, *nodes):
        self.modules = [n.torch_module for n in nodes]
        self._last = None

    def set(self, value):

        if isinstance(value, np.ndarray):
            if np.all(np.equal(self._last, value)):
                return
        else:
            if value == self._last:
                return

        self._last = value

        # Convert PBC arrays into simple True/False values (until we have determined how to use mixed PBC)
        if value is not True and value is not False:
            if all(v == True for v in value):
                value = True
            elif all(v == False for v in value):
                value = False
            else:
                raise ValueError("Unrecognized PBC condition: {}. Mixed PBC not yet supported.".format(value))
        for m in self.modules:
            m.pbc = value


# decorator for generating calculation methods; this code is also run for calls to
# get_property, but with the key filled in as a closure.
def _generate_calculation_method(key):

    cant_calculate = PropertyNotImplementedError("Property not implemented:'{}'".format(key))

    def method(self, atoms, allow_calculation=True, **kwargs):

        if key not in self.implemented_properties:
            raise cant_calculate

        if not allow_calculation:
            return self.results.get(key, None)

        if self.calculation_required(atoms, [key]):
            self.calculate()

        return self.results[key]

    return method


# UNUSED, was used for HippynnCaclulator.to()
# factory for forwarding pytorch methods to the calculator
def pass_to_pytorch(fn_name):
    wraps = getattr(torch.nn.Module, fn_name)

    @functools.wraps(wraps)
    def method(self, *args, **kwargs):
        getattr(self.module, fn_name)(*args, **kwargs)

    return method


class HippynnCalculator(Calculator): # Calculator inheritance required for ASE Mixing Calculator usage
    """
    ASE calculator based on hippynn graphs. Uses ASE neighbor lists. Not suitable for domain decomposition.
    """

    def __init__(self, energy, charges=None, skin=1.0, extra_properties=None, en_unit=None, dist_unit=None, species_set=None, indexer=None):
        """
        :param energy: Node for energy
        :param charges: Node for charges (optional)
        :param skin: Skin for neighbors list
        :param extra_properties: dictionary of names to nodes for additional nodes for the calculator to compute.
        :param name: identifying string
        :param en_unit: unit factor to use for energies -- set it to the same as the energy unit used
            in training. If not given, the model is assumed to output in kcal/mol!
        :param dist_unit: unit factor to use for distances -- set it to the same as the
            distance unit used in training. If not given, defaults to Angstrom
        """

        self.min_radius, self.species_set, self.implemented_properties, self.module, self.pbc = setup_ASE_graph(
            energy, charges=charges, extra_properties=extra_properties, species_set=species_set,indexer=indexer,
        )
        
        self.implemented_properties.append("energy") # Required for using mixing calculators in ASE
        
        self.atoms = None
        self._last_properties = None

        # get species set to determine length of cutoffs
        if skin < 0:
            raise ValueError("Negative skin radius not allowed.")
        self._cutoffs = self.min_radius  # The -1 is for 'blank atoms', not used in ase.
        self._skin = skin
        self.nl = None
        self.rebuild_neighbors()

        self._needs_calculation = True
        self.results = {}
        self.en_unit = en_unit if en_unit is not None else ase.units.kcal / ase.units.mol
        self.dist_unit = dist_unit if dist_unit is not None else ase.units.Angstrom
        self.device = torch.device("cpu")
        self.dtype = torch.get_default_dtype()
        if not hasattr(self, "name"):  # Older versions of ase do not set the name
            self.name = "Hippynn calculator"
        self.parameters = {} #Hack to work with ASE trajectory printing

    make = _generate_calculation_method
    get_potential_energy = make("potential_energy")
    get_energy = get_potential_energy
    get_potential_energies = make("potential_energies")
    get_energies = get_potential_energies
    get_free_energy = make("free_energy")
    get_forces = make("forces")
    get_charges = make("charges")
    get_dipole_moment = make("dipole_moment")
    get_dipole = get_dipole_moment
    get_stress = make("stress")
    get_stresses = make("stresses")
    get_magmom = make("magmom")
    get_magmoms = make("magmoms")
    del make

    def rebuild_neighbors(self):
        self.nl = ase.neighborlist.NeighborList(
            self._cutoffs+self._skin, #ASE neighbor list implementation is only safe up to cutoffs-skin
            skin=self._skin,
            sorted=True,
            self_interaction=False,
            bothways=True,
            primitive=ase.neighborlist.NewPrimitiveNeighborList,
        )

    # Dear ASE: This is not Java....
    def set_atoms(self, atoms):
        self.atoms = atoms.copy()
        self._needs_calculation = True

    def get_property(self, name, atoms, allow_calculation=True):
        # `or` defaults to last atoms received
        try:
            return getattr(self, "get_{}".format(name))(atoms or self.atoms, allow_calculation=allow_calculation)
        except AttributeError:
            raise PropertyNotImplementedError("Property not implemented:'{}'".format(name))

    def to(self, *args, **kwargs):
        self.module.to(*args, **kwargs)

        # The below section is complicated because the pytorch `to` method does not have a strict signature.
        allargs = *args, *kwargs.values()
        for a in allargs:
            # Attempt conversion of strings to dtype or device.
            try:
                a = torch.device(a)
            except (RuntimeError, TypeError):
                pass
            try:
                a = torch.dtype(a)
            except (RuntimeError, TypeError):
                pass

            if isinstance(a, torch.dtype):
                self.dtype = a
            if isinstance(a, torch.device):
                self.device = a
        return self

    def calculate(self, atoms=None, properties=None, system_changes=True):
        """
        Accepts 'properties' and 'system changes' but ignores them,
        purely for compatibility with ASE base calculator
        """
        self.atoms = self.atoms if atoms is None else atoms.copy()

        self.pbc.set(self.atoms.pbc)

        # Ase neighbor list raises ValueError if passed a new system with a different number of atoms.
        # If something like this happens, we just scrap that neighbors list and make a fresh one.
        try:
            self.nl.update(self.atoms)
        except ValueError:
            self.rebuild_neighbors()
            self.nl.update(self.atoms)

        # Get variables from atoms. Unsqueeze is to add batch axis.
        positions = torch.as_tensor(self.atoms.positions).unsqueeze(0)
        # Convert from ASE distance (angstrom) to whatever the network uses.
        positions = positions / self.dist_unit
        species = torch.as_tensor(self.atoms.numbers,dtype=torch.long).unsqueeze(0)
        cell = torch.as_tensor(self.atoms.cell.array).unsqueeze(0)
        # Get pair first and second from neighbors list
        pair_first = torch.as_tensor(self.nl.nl.pair_first,dtype=torch.long)
        pair_second = torch.as_tensor(self.nl.nl.pair_second,dtype=torch.long)
        pair_shiftvecs = torch.as_tensor(self.nl.nl.offset_vec,dtype=torch.long)

        # This order must be synchronized with function setup_ase_graph above
        inputs = species, positions, cell, pair_first, pair_second, pair_shiftvecs
        # Move to device, and convert to the type of float the model is using
        inputs = [
            inp.to(device=self.device, dtype=(self.dtype if torch.is_floating_point(inp) else None)) for inp in inputs
        ]

        # Run it all through pytorch!
        results = self.module(*inputs)

        self.results = {k: r.detach().cpu().numpy() for k, r in zip(self.implemented_properties, results)}

        # Convert units
        self.results["potential_energy"] = self.results["potential_energy"][0, 0] * self.en_unit
        self.results["energy"] = self.results["potential_energy"] # Required for using ASE mixing calculators, which assume potential energy is under the property "energy"
        self.results["forces"] = self.results["forces"][0] * (self.en_unit / self.dist_unit)
        # slightly opaque way to handle if pbc is 3-tuple or boolean.
        # Note: PBC handler forbids mixed BCs, so this check is enough.
        if all(self.atoms.pbc) if len(self.atoms.pbc) else self.atoms.pbc:
            stress_factor = self.en_unit / (self.dist_unit) ** 3
        else:
            stress_factor = self.en_unit
        self.results["stress"] = self.results["stress"][0] * stress_factor

        self._needs_calculation = False

    def calculation_required(self, atoms, properties=None, tol=1e-15):
        """
        Returns true if:
        1. A property in the list `properties` is not supported.
        2. Atoms are different from stored atoms. This proceeds by short-circuit:
        * if this check is passed, compare using ase `compare_atoms` and tolerance `tol`.
        * slight difference from ASE implementation: if `properties` is a string then it is wrapped in a list.

        """
        if isinstance(properties, str):
            properties = [properties]

        if any(prop not in self.implemented_properties for prop in properties):
            return True

        if bool(compare_atoms(self.atoms, atoms, tol=tol)):
            self._needs_calculation = True
            self.atoms = atoms.copy()

        return self._needs_calculation


def calculator_from_model(model, **kwargs):
    """
    Attempt to find the energy and charge nodes automatically.

    :param model: :class:`GraphModule` for the model.
    :param kwargs: passed to :class:`HippynnCalculator`

    .. Note::
       If your model has an energy node, but that is not the full energy for simulation, don't use this function.
       Similarly for charge.
    """
    possible_energy = find_relatives(model.nodes_to_compute, Energies)
    if len(possible_energy) != 1:
        raise ValueError("More than one energy node present, cannot auto-generate ASE interface")
    energy = possible_energy.pop()

    try:
        possible_charge = find_relatives(model.nodes_to_compute, Charges)
    except NodeNotFound as nnf:
        warnings.warn("No charge node found: ASE interface will not calculate charges or dipole moments.")
        charges = None
    else:
        if len(possible_charge) > 1:
            warnings.warn("More than one charge node present, charges left out of ASE interface")
            charges = None
        else:
            charges = possible_charge.pop()

    return HippynnCalculator(energy=energy, charges=charges, **kwargs)
