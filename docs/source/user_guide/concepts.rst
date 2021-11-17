hippynn Concepts
================


Layers/Networks
^^^^^^^^^^^^^^^

The Layers and Networks subpackages contain pure pytorch code.
Each ``torch.nn.Module`` returns a list of tensors


Nodes
^^^^^

A node is basically a pytorch module that is dressed up with some extra metadata.
More specifically, it is a container whose `.torch_module` implements the desired function.

The metadata assists in the creation of new nodes and graph structures.

- A node has ``parents`` that specify what tensors go into the node.

- As new nodes are created, the ``parents`` of that node get linked to their ``children``.

Nodes have an :class:`~hippynn.graphs.indextypes.type_def.IDXType` that specify the form of the data output by the node.
For example:
`Atom` specifies data on atoms in the system.
`Molecule` specifies data on an entire system.
`Pair` specifies data that connects pairs of atoms.

The inclusion of the index types helps determine:
1. Whether it is possible for nodes to be connected.
2. If it is not directly possible, but the two data types are compatible,
how to use index_transformers to transform the data from one index type to another.

A :class:`~hippynn.graphs.nodes.base.multi.MultiNode` is a node that itself produces several tensors.
This covers cases where a calculation can produce several useful tensors in tandem,
or when wrapping some modules already implemented.

:class:`~hippynn.graphs.nodes.base.base.SingleNode` and :class:`~hippynn.graphs.nodes.base.multi.MultiNode` are base
classes for subclassing.


Graphs
^^^^^^

A :class:`~hippynn.graphs.GraphModule` is a 'compiled' set of nodes; a ``torch.nn.Module`` that executes the graph.

GraphModules are used in a number of places within hippynn.


Experiment
^^^^^^^^^^

An experiment is a training run.



