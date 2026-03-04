"""
Nodes for prediction of variables from network features.
"""

import torch

from .base import MultiNode, AutoKw, ExpandParents, find_unique_relative, Node
from .tags import AtomIndexer, Network, PairIndexer, HAtomRegressor, Charges, Energies, Encoder
from .indexers import PaddingIndexer
from ..indextypes import IdxType, index_type_coercion
from ...layers import targets as target_modules
from ..._deprecations import _DeprecatedNamesMixin


class HEnergyNode(Energies, HAtomRegressor, ExpandParents, AutoKw, MultiNode, _DeprecatedNamesMixin):
    """
    Predict a system-level scalar such as energy from a sum over local components.
    """
    _DEPRECATED_NAMES = {"mol_energy": "system_energy"}

    input_names = "hier_features", "system_index", "n_systems"
    output_names = "system_energy", "atom_energies", "energy_terms", "hierarchicality", "atom_hier", "mol_hier", "batch_hier"
    main_output_name = "system_energy"
    output_index_states = IdxType.Systems, IdxType.Atoms, None, IdxType.Systems, IdxType.Atoms, IdxType.Systems, IdxType.Scalar
    auto_module_class = target_modules.HEnergy
    auto_module_kwargs = "first_is_interacting", "feature_sizes"

    @parent_expander.match(Network)
    def expansion0(self, net, **kwargs):
        if "feature_sizes" not in self.module_kwargs:
            self.module_kwargs["feature_sizes"] = net.torch_module.feature_sizes
        pdindexer = find_unique_relative(net, AtomIndexer)
        return net, pdindexer.system_index, pdindexer.n_systems

    def __init__(self, name, parents, first_is_interacting=False, module_kwargs=None, **kwargs):
        """
        :param name:
        :param parents:
        :param first_is_interacting: If True, drop the first feature
         components (which do not interact and so are based only on initial features for the atom)
        :param module:
        :param module_kwargs: other module keywords to use in initialization.
        :param kwargs:
        """
        self.module_kwargs = module_kwargs or {}
        super().__init__(name, parents, first_is_interacting=first_is_interacting, **kwargs)


class HChargeNode(Charges, HAtomRegressor,  ExpandParents, AutoKw, MultiNode):
    """
    Predict an atom-level scalar such as charge from local features.
    """

    input_names = ("hier_features",)
    output_names = "atom_charges", "partial_sums", "charge_hierarchality"
    main_output_name = "atom_charges"
    output_index_states = IdxType.Atoms, None, IdxType.Atoms
    auto_module_class = target_modules.HCharge

    @parent_expander.match(Network)
    def expansion0(self, net, **kwargs):
        if "feature_sizes" not in self.module_kwargs:
            self.module_kwargs["feature_sizes"] = net.torch_module.feature_sizes
        return (net.main_output,)

    def __init__(self, name, parents, module="auto", module_kwargs=None, **kwargs):
        self.module_kwargs = module_kwargs or {}
        super().__init__(name, parents, module=module, **kwargs)


class LocalChargeEnergy(Energies, ExpandParents, AutoKw, HAtomRegressor, MultiNode, _DeprecatedNamesMixin):
    _DEPRECATED_NAMES = {"mol_energies": "system_energy"}
    input_names = "charges", "features", "system_index", "n_systems"
    output_names = "system_energy", "atom_energies"
    main_output_name = "system_energy"
    output_index_states = IdxType.Systems, IdxType.Atoms
    auto_module_class = target_modules.LocalChargeEnergy
    auto_module_kwargs = "first_is_interacting", "feature_sizes"


    @parent_expander.match(Node, Network)
    def expansion0(self, charge, network, **kwargs):
        if "feature_sizes" not in self.module_kwargs:
            self.module_kwargs["feature_sizes"] = network.torch_module.feature_sizes
        charge = index_type_coercion(charge.main_output, IdxType.Atoms)
        pdxer = find_unique_relative(network, PaddingIndexer)
        return charge, network.main_output, pdxer.system_index, pdxer.n_systems

    def __init__(self, name, parents, module_kwargs=None, **kwargs):
        self.module_kwargs = module_kwargs or {}
        super().__init__(name, parents, **kwargs)


class HBondNode(ExpandParents, AutoKw, MultiNode):
    """
    Predict an pair-level scalar such as bond order from local features on both atoms
    """

    auto_module_class = target_modules.HBondSymmetric
    output_names = "bonds", "bond_hierarchality"
    output_index_states = IdxType.Pairs, IdxType.Pairs
    input_names = "features", "pair_first", "pair_second", "pair_dist"
    main_output_name = "bonds"

    @parent_expander.match(Network)
    def expand0(self, net, *, purpose, **kwargs):
        if "feature_sizes" not in self.module_kwargs:
            self.module_kwargs["feature_sizes"] = net.torch_module.feature_sizes
        return net,

    @parent_expander.matchlen(1)
    def expand1(self, features, *, purpose, **kwargs):
        pairfinder = find_unique_relative(features, PairIndexer, why_desc=purpose)
        return features, pairfinder

    @parent_expander.matchlen(2)
    def expand2(self, features, pairfinder, **kwargs):
        return features.main_output, pairfinder.pair_first, pairfinder.pair_second, pairfinder.pair_dist

    def __init__(self, name, parents, module_kwargs=None, **kwargs):
        self.module_kwargs = module_kwargs or {}
        super().__init__(name, parents, **kwargs)


class AtomizationEnergyNode(Energies, HAtomRegressor, ExpandParents, AutoKw, MultiNode, _DeprecatedNamesMixin):
    _DEPRECATED_NAMES = {"mol_energy": "system_energy"}

    input_names = "hier_features", "vac_features", "encoding", "system_index", "n_systems"
    output_names = "system_energy", "partial_energies", "hierarchicality"
    output_index_states = IdxType.Systems, None, IdxType.Systems
    main_output_name = "system_energy"
    auto_module_class = target_modules.AtomizationEnergy
    auto_module_kwargs = "feature_sizes",

    @parent_expander.match(Network)
    def expansion1(self, net, **kwargs):
        from ..gops import vacuum_outputs
        encoder = find_unique_relative(net, Encoder)
        if "feature_sizes" not in self.module_kwargs:
            self.module_kwargs["feature_sizes"] = net.torch_module.feature_sizes
        pdindexer = find_unique_relative(net, AtomIndexer)
        species_set = net.torch_module.species_set[1:]
        vac_net, = vacuum_outputs([net], species_set=species_set)
        return net, vac_net, encoder, pdindexer

    @parent_expander.match(Network, Network, Encoder, AtomIndexer)
    def expansion0(self, net, vacuum_net, encoding, pdindexer, **kwargs):
        # Used to be built explicitly because of introduction of multiple encoders.
        # Now fixed by using constants when encoding on the vacuum side.
        # encatom = AtomReIndexer('Encoding[atoms]', (encoding.encoding, pdindexer))
        return net, vacuum_net, encoding.encoding, pdindexer.system_index, pdindexer.n_systems,

    parent_expander.assertlen(5)
    parent_expander.get_main_outputs()
    parent_expander.require_idx_states(None, None, IdxType.Atoms, None, None)

    def __init__(self, name, parents, module_kwargs=None, **kwargs):
        self.module_kwargs = module_kwargs or {}
        super().__init__(name, parents, **kwargs)

    def create_henergy_equivalent(self):
        from .. import copy_subgraph, Predictor

        # ignore vacuum encoding input
        net_parent, vac_net, _, system_index, n_systems = self.parents
        new_parents = (net_parent, system_index, n_systems)

        # copy-out subgraph for vacuum computations.
        new_graph, other_nodes = copy_subgraph(vac_net.feature_nodes, assume_inputed=[])

        # # This builds the graph for de-indexing atom to sysatom,
        # # without having a paddingindexer.
        # # Leaving this as reference in case it becomes necessary to do this later.
        # from .base.algebra import ValueNode
        # from .indexers import AtomDeIndexer
        # species_set = vac_net.torch_module.species_set
        # n_atom_types = len(species_set)-1
        # system_index = ValueNode(torch.arange(n_atom_types, dtype=torch.int64, device=species_set.device))
        # atom_index = ValueNode(torch.zeros(n_atom_types, dtype=torch.int64, device=species_set.device))
        # n_systems = ValueNode(n_atom_types, convert=False)
        # n_atoms_max = ValueNode(1, convert=False)
        # from ..indextypes.registry import assign_index_aliases
        # for n in new_graph:
        #     index_node = AtomDeIndexer(f"{n.name}[SysAtom]", (n, system_index, atom_index, n_systems, n_atoms_max))
        #     assign_index_aliases(n, index_node)

        # Alternative hacky way to solve the problem is much simpler.
        for n in new_graph:
            n.index_state = IdxType.SysAtom

        pred = Predictor(inputs=[], outputs=new_graph)
        outputs = pred()

        # Note that atomization ignores the first (raw vacuum) features of the network.
        # Note that we squeeze because each system has 1 atom.
        feature_list = [outputs[n.name].squeeze(1) for n in new_graph[1:]]
        weights = [x.weight for x in self.torch_module.layers]

        # Calculate the total bias terms per species.
        en_contributions = [w @ x.T for w, x in zip(weights, feature_list)]
        total_e0 = -sum(en_contributions)

        # Build HEnergyNode which matches the AtomizationEnergy module.
        feature_sizes = self.torch_module.feature_sizes
        feature_sizes = (total_e0.shape[1], *feature_sizes)
        henergy_mod = target_modules.HEnergy(feature_sizes=feature_sizes, first_is_interacting=False, n_target=1)

        en_0_layer = henergy_mod.layers[0]
        en_0_layer.weight = torch.nn.Parameter(total_e0)
        en_0_layer.bias = None

        for lay1, lay2 in zip(henergy_mod.layers[1:], self.torch_module.layers):
            lay1.weight = lay2.weight
            # lay1.bias.set_(torch.zeros_like(lay1.bias))
            lay1.bias = lay2.bias

        henergy_node = HEnergyNode(f"{self.name}[HEnergy]", new_parents, module=henergy_mod)

        return henergy_node
