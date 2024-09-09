The hippynn python package - a modular library for atomistic machine learning with pytorch.
*******************************************************************************************

We aim to provide a powerful library for the training of atomistic
(or physical point-cloud) machine learning.
We want entry-level users to be able to efficiently train models
to millions of datapoints, and a modular structure for extensions and contribution.

While hippynn's development so-far has centered around the HIP-NN architecture, don't let that
discourage you if you are performing research with another model.
Get in touch, and let's work together to provide a high-quality implementation of your work,
either as a contribution or an interface extension to your own package.

Features:
=========
Modular set of pytorch layers for atomistic operations
----------------------------------------------------------
- Atomistic operations can be tricky to write in native pytorch.
  Most operations provided here support linear-scaling models.
- Model energy, force charge & charge moments, bond orders, and more!
- nn.Modules are written with minimal reference to the rest of the library;
  if you want to use them in your scripts without using the rest of the features
  provided here -- no problem!

Graph level API for simple and flexible construction of models from pytorch components.
---------------------------------------------------------------------------------------

- Build models based on the abstract physics/mathematics of the problem,
  without having to think about implementation details.
- Graph nodes support native python syntax, for example different forms of loss can be directly added.
- Link predicted values in the model with a database entry to compare predicted and true values
- IndexType logic records metadata about tensor structure, and provides
  automatic conversion to compatible structures when possible.
- Graph API is independent of module implementation.

Plot level API for tracking your training.
----------------------------------------------------------
- Using the graph API, define quantities to evaluate before, during, or after training as
  figures using matplotlib.

Training & Experiment API
----------------------------------------------------------
- Integrated with graph level API
- Pretty-printing loss metrics, generating plots periodically
- Callbacks and checkpointing

Custom Kernels for fast execution
----------------------------------------------------------
- Certain operations are not efficiently written in pure pytorch, we provide
  alternative implementations with ``numba`` and ``cupy``
- These are directly linked in with pytorch Autograd -- use them like native pytorch functions.
- These provide advantages in memory footprint and speed
- Includes CPU and GPU execution for custom kernels

Interfaces to other codes
----------------------------------------------------------
- ASE: Define ``ASE`` calculators based on the graph-level API.
- PYSEQM: Use ``PYSEQM`` calculations as nodes in a graph.
- LAMMPS: Use models from ``hippynn`` in LAMMPS via the MLIAP Unified interface.

Installation
============

- Clone this repository and navigate into it.

Dependencies using conda:
-------------------------

- Run ``conda install -c pytorch -c conda-forge --file conda_requirements.txt``

Dependencies using pip:
-----------------------
- Run ``pip install .``
- If you fee like tinkering, do an editable install: ``pip install -e .``
- You can install using all optional dependencies from pip with: ``pip install -e .[full]``

Notes
-----

- Install dependencies with pip from requirements.txt .
- Install dependencies with conda from conda_requirements.txt .
- If you don't want pip to install them, conda install from file before installing ``hippynn``.
  You may want to use -c pytorch for the pytorch channel.
  For ase and cupy, you probably want to use -c conda-forge.
- Optional dependencies are in optional_dependencies.txt

Documentation
=============

Please see https://lanl.github.io/hippynn/ for the latest documentation. You can also build
the documentation locally, see /docs/README.txt

Other things
============

We are currently under development. At the moment you should be prepared for breaking changes -- keep track
of what version you are using if you need to maintain consistency.

As we clean up the rough edges, we are preparing a manuscript.
If, in the mean time, you are using hippynn in your work, please cite this repository and the HIP-NN paper:

Lubbers, N., Smith, J. S., & Barros, K. (2018).
Hierarchical modeling of molecular energies using a deep neural network.
The Journal of chemical physics, 148(24), 241715.

See AUTHORS.txt for information on authors.

See LICENSE.txt for licensing information. hippynn is licensed under the BSD-3 license.
See COPYRIGHT.txt for copyright information.

Triad National Security, LLC (Triad) owns the copyright to hippynn, which it identifies as project number LA-CC-19-093.

Copyright 2019. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
Department of Energy/National Nuclear Security Administration. All rights in the program are
reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
Security Administration. The Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
derivative works, distribute copies to the public, perform publicly and display publicly, and to permit
others to do so.

