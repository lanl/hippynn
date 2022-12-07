Creating Custom Node Types
==========================

For applications of existing models or methods you shouldn't need
to create new *types* of nodes. However, if you want to extend
the capabilities of the library for your own application
(or as part of a contribution to this library) then you
may find yourself wanting to put a new type of node into
``hippynn``.

The very basics
---------------

The basic operation of creating a new hippynn node is not highly complex.
Let's assume we have a module FooModule that implements some pytorch operations,
and takes some keyword arguments in constructing that module.
A simple node could be built as follows::

    from hippynn.graphs.nodes.base import SingleNode
    from hippynn.graphs import IdxType
    class FooNode(SingleNode):
        _index_state = IdxType.Atom
        def __init__(self,name,parents,module,**kwargs):
            super().__init__(name,parents,module=module,**kwargs)

At a basic level, that's it. However, the parents of this node are completely unspecified;
there is no information about what tensors should go into the FooModule. Note that at this level,
the module itself is not created when building the node, and a suitable pytorch module
must be passed in.

A MultiNode
-----------

A slightly more complex example would be to use a ``MultiNode``, which is a torch
module that outputs several outputs. Specify the names of the outputs in the
``_output_names`` attribute as a tuple of strings. Additionally, you can
specify the ``IdxType`` of the outputs so that other nodes can recognize
what type of information is provided. Here is a stripped-down version of the
hierarchical energy regression target :class:`~hippynn.graphs.nodes.targets.HEnergyNode`::

    import hippynn.layers.targets as target_modules
    from hippynn.graphs.nodes import MultiNode
    from hippynn.graphs.nodes.base.definition_helpers import AutoKw

    class SimpleHEnergyNode(AutoKw, MultiNode):
        _input_names = "hier_features", "mol_index", "n_molecules"
        _output_names = "mol_energy", "atom_energies", "energy_terms", "hierarchicality"
        _main_output = "mol_energy"
        _output_index_states = IdxType.Molecules, IdxType.Atoms, None, IdxType.Molecules
        _auto_module_class = target_modules.HEnergy

        def __init__(self, name, parents, module='auto',module_kwargs=None,**kwargs):
            self.module_kwargs = module_kwargs
            super().__init__(name, parents, module=module, **kwargs)

Note that we have added the _input_names tuple as well, this attribute can be set on
SingleNode and MultiNode classes.

The ``_main_output`` attribute specifies what tensor to use by default when sending information
to a child node. This class also makes use of the ``AutoKw`` mix-in for defining a new module
using keyword arguments. These arguments will be passed to a new instance of the attribute
``auto_module_class``.

Parent expansion
----------------

The above example works, however, it 1) requires the user to find the appropriate
input nodes corresponding to ``hier_features``, ``mol_index``, ``n_molecules``, which are
required to run the underlying torch module.

The features will usually come from a network, and the molecule index and number of molecules
in a batch are processed by the padding indexer. We can use the ``ExpandParents`` class
to make invoking this node easier.

Let's take a look  at the full definition of :class:`~hippynn.graphs.nodes.targets.HEnergyNode`:

.. literalinclude:: ../../../hippynn/graphs/nodes/targets.py
   :pyobject: HEnergyNode

The parent classes ``Energies`` and ``HAtomRegressor`` do not add any methods, they
are simply mixin tags so that it is easy to find nodes based on their type. The key
additional superclass is ``ExpandParents``, which automatically provides the class with
a ``_parent_expander`` attribute that is an instance of a parent expander.
We then define a method called (arbitrarily) ``expansion0`` which is decorated by the parent
expander to be run when the form of the parents matches the given one, in this case,
a single parent with node type ``Network``. The function does two things.

1. It sets the value of the feature sizes for the underlying torch module based on those found
   in the network, if they have not already been defined.

2. It attempts to find a unique ``AtomIndexer`` object which is connected to the network node,
   and gets the outputs ``mol_index`` and ``n_molecules`` from that object.

A key aspect is that ``expansion0`` is only run if the parents match this form. If
a different form is found, the function is skipped. This way if we arise at a complex
model definition where there are multiple AtomIndexers or none whatsoever, but the inputs
to the node can be provided by some other route, we can always pass the fully specified
parents of the node, ``hier_features``, ``mol_index``, and ``n_molecules``.

Adding constraints to possible parents
--------------------------------------

Finally, it is possible to add additional information to the parent expander
to ensure that the final form of the parents is suitable for computation.

Let's take a look at the code for :class:`~hippynn.graphs.nodes.physics.ChargeMomentNode`:

.. literalinclude:: ../../../hippynn/graphs/nodes/physics.py
   :pyobject: ChargeMomentNode

This is the base class for the Dipole and Quadrupole Nodes. It uses several parent expansion functions:

1. ``@_parent_expander.match()``: Decorates a function to be used by the parent expansion
   if the type is matched. The returned values should be the new set of parents for the node.
   A function doesn't -have- to modify the set of parents.

2. ``_parent_expander.assertlen()``: Assert that there are a given number of parents for the node.

3. ``_parent_expander.get_main_outputs()``: If there are any MultiNodes in the parent set,
   replace them with their main outputs.

4. ``_parent_expander.require_idx_states()``: Throw an error if the index states of the parents
   do not match a specific form. Additionally, if the current index state can be converted to
   the needed index state, this conversion will automatically be applied using
   :func:`~hippynn.graphs.indextypes.index_type_coercion`.

A full list of available methods is  at the API documentation for the
:func:`~hippynn.graphs.nodes.base.definition_helpers.ParentExpander`.
These directives are executed when the node's ``expand_parents`` method is run, which
should be performed *before* calling to the ``super().__init__()`` method.
In combination, these directives allow for a powerful flexibility in building graphs
so that where possible, information is re-used or automatically generated in order
to simplify the syntax of invoking the node from a user perspective, but still allow
for a complete and unambiguous definition of node parents when in cases where it is
called for.


