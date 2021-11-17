hippynn Features
================


Modular set of pytorch layers for atomistic operations
----------------------------------------------------------
- Atomistic operations can be tricky to write in native pytorch.
  Most operations provided here support linear-scaling models.
- Model energy, force charge & charge moments, bond orders, and more!
- nn.Modules are written with minimal reference to the rest of the library;
  if you want to use them in your scripts without using the rest of the features
  provided here -- no problem!

API documentation for :mod:`~hippynn.layers`

Graph level API for simple and flexible construction of models from pytorch components.
---------------------------------------------------------------------------------------

- Build models based on the abstract physics/mathematics of the problem,
  without having to think about implementation details.
- Graph nodes support native python syntax, for example different forms of loss can be directly added.
- Link predicted values in the model with a database entry to compare predicted and true values
- IndexType logic records metadata about tensor structure, and provides
  automatic conversion to compatible structures when possible.
- Graph API is independent of module implementation.

API documentation for :mod:`~hippynn.graphs`

Plot level API for tracking your training.
----------------------------------------------------------
- Using the graph API, define quantities to evaluate before, during, or after training as
  figures using matplotlib.

API documentation for :mod:`~hippynn.plotting`


Training & Experiment API
----------------------------------------------------------
- Integrated with graph level API
- Pretty-printing loss metrics, generating plots periodically
- Callbacks and checkpointing

API documentation for :mod:`~hippynn.experiment`


Custom Kernels for fast execution
----------------------------------------------------------
- Certain operations are not efficiently written in pure pytorch, we provide
  alternative implementations with ``numba``
- These are directly linked in with pytorch Autograd -- use them like native pytorch functions.
- These provide advantages in memory footprint and speed
- Includes CPU and GPU execution for custom kernels

More information at :doc:`this page </user_guide/ckernels>`

Interfaces
----------------------------------------------------------
- ASE: Define `ase` calculators based on the graph-level API.
- PYSEQM: Use `pyseqm` calculations as nodes in a graph.

API documentation for :mod:`~hippynn.interfaces`