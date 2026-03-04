""" "
Some tests for building and instantiating a custom class.

This file serves double-duty as it is referenced directly in
    docs/user_guide/custom_nodes.rst

The comment directives are used to specify where in the documentation these things are.
"""

import pytest


def test_create_simple_node_class(neural_network_node):

    # begin doc snippet
    from hippynn.graphs.nodes.base import SingleNode
    from hippynn.graphs import IdxType

    class FooNode(SingleNode):
        index_state = IdxType.Atoms

        def __init__(self, name, parents, module, **kwargs):
            super().__init__(name, parents, module=module, **kwargs)

    # end doc snippet

    # begin usage snippet
    from hippynn.layers.algebra import LambdaModule

    foo = FooNode("foo", (neural_network_node,), module=LambdaModule(lambda x: x))
    # end usage snippet
    pass


def test_create_simple_henergy_class(neural_network_node):

    # begin doc snippet
    import hippynn.layers.targets as target_modules

    from hippynn.graphs import IdxType
    from hippynn.graphs.nodes.base import MultiNode
    from hippynn.graphs.nodes.base.definition_helpers import AutoKw

    class SimpleHEnergyNode(AutoKw, MultiNode):
        input_names = "input_features", "system_index", "n_systems"
        output_names = "system_energies", "atom_energies", "energy_terms", "hierarchicality"
        main_output_name = "system_energies"
        output_index_states = IdxType.Molecules, IdxType.Atoms, None, IdxType.Molecules
        auto_module_class = target_modules.HEnergy

        def __init__(self, name, parents, module="auto", module_kwargs=None, **kwargs):
            self.module_kwargs = module_kwargs
            super().__init__(name, parents, module=module, **kwargs)

    # end doc snippet

    # begin usage snippet
    from hippynn.graphs.nodes.indexers import acquire_encoding_padding

    encoder, padding_indexer = acquire_encoding_padding(neural_network_node, None)

    parents = neural_network_node, padding_indexer.system_index, padding_indexer.n_systems
    module_kwargs = dict(feature_sizes=neural_network_node.torch_module.feature_sizes)

    energy = SimpleHEnergyNode("HEnergy", parents, module_kwargs=module_kwargs)
    # end usage snippet

    # Trigger some properties on the node:
    energy.main_output
    energy.true
    energy.pred
    energy.input_features
    energy.system_index
    energy.n_systems
    pass


def test_create_full_henergy(neural_network_node):

    # begin usage snippet
    from hippynn.graphs.nodes.targets import HEnergyNode

    energy = HEnergyNode("henergy", neural_network_node)
    # end usage snippet
    pass
