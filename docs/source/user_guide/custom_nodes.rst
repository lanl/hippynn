
.. comment::

   This file is tied directly to the contents of the file:
      /tests/test_node_classes.py
   To edit what appears here, edit that file.
   Remember to re-run the tests to make sure the example code functions.

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
A simple node could be built as follows:

.. literalinclude:: ../../../tests/test_node_classes.py
   :language: python
   :pyobject: test_create_simple_node_class
   :dedent: 4
   :start-after: # begin doc snippet
   :end-before: # end doc snippet
      

At a basic level, that's it. However, the parents of this node are completely unspecified;
there is no information about what tensors should go into the FooModule. Note that at this level,
the module itself is not created when building the node, and a suitable pytorch module
must be passed in. We could thus create this node using something like:

.. literalinclude:: ../../../tests/test_node_classes.py
   :language: python
   :pyobject: test_create_simple_node_class
   :dedent: 4
   :start-after: # begin usage snippet
   :end-before: # end usage snippet


A MultiNode
-----------

A slightly more complex example would be to use a ``MultiNode``, which is a torch
module that outputs several outputs. Specify the names of the outputs in the
``output_names`` attribute as a tuple of strings. Additionally, you can
specify the ``IdxType`` of the outputs so that other nodes can recognize
what type of information is provided. Here is a stripped-down version of the
hierarchical energy regression target :class:`~hippynn.graphs.nodes.targets.HEnergyNode`:

.. literalinclude:: ../../../tests/test_node_classes.py
   :pyobject: test_create_simple_henergy_class
   :dedent: 4
   :start-after: # begin doc snippet
   :end-before: # end doc snippet


Note that we have added the ``input_names`` tuple as well, this attribute can be set on
both SingleNode and MultiNode classes.

The ``main_output_name`` attribute specifies what tensor to use by default when sending information
to a child node. This class also makes use of the ``AutoKw`` mix-in for defining a new module
using keyword arguments. These arguments will be passed to a new instance of the attribute
``auto_module_class``. To use this node, we now only need to supply the arguments for the pytorch module:

.. literalinclude:: ../../../tests/test_node_classes.py
   :language: python
   :pyobject: test_create_simple_henergy_class
   :dedent: 4
   :start-after: # begin usage snippet
   :end-before: # end usage snippet

However because the HEnergy nn.Module requires multiple input tensors,
it is a little bit of work to find the relevant metadata required. 

Parent expansion
----------------

The above example works, however, it 1) requires the user to find the appropriate
input nodes corresponding to ``input_features``, ``system_index``, ``n_systems``, which are
required to run the underlying torch module.

The features will usually come from a network, and the system index and number of systems
in a batch are processed by the padding indexer. We can thus use another feature of hippynn,
the ``ExpandParents`` class, to simplify construction of the node by making this logic an
optional step during the construction of the node.

Let's take a look  at the full definition of :class:`~hippynn.graphs.nodes.targets.HEnergyNode`:

.. literalinclude:: ../../../hippynn/graphs/nodes/targets.py
   :pyobject: HEnergyNode

The parent classes ``Energies`` and ``HAtomRegressor`` do not add any methods, they
are simply mixin tags so that it is easy to find nodes based on their type. The key
additional superclass is ``ExpandParents``, which automatically provides the class with
a ``parent_expander`` attribute that is an instance of a parent expander.
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

.. literalinclude:: ../../../tests/test_node_classes.py
   :language: python
   :pyobject: test_create_full_henergy
   :dedent: 4
   :start-after: # begin usage snippet
   :end-before: # end usage snippet

Adding constraints to possible parents
--------------------------------------

Finally, it is possible to add additional information to the parent expander
to ensure that the final form of the parents is suitable for computation.

Let's take a look at the code for :class:`~hippynn.graphs.nodes.physics.ChargeMomentNode`:

.. literalinclude:: ../../../hippynn/graphs/nodes/physics.py
   :pyobject: ChargeMomentNode

This is the base class for the Dipole and Quadrupole Nodes. It uses several parent expansion functions:

1. ``@parent_expander.match()``: Decorates a function to be used by the parent expansion
   if the type is matched. The returned values should be the new set of parents for the node.
   A function doesn't -have- to modify the set of parents.

2. ``parent_expander.assertlen()``: Assert that there are a given number of parents for the node.

3. ``parent_expander.get_main_outputs()``: If there are any MultiNodes in the parent set,
   replace them with their main outputs.

4. ``parent_expander.require_idx_states()``: Throw an error if the index states of the parents
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


