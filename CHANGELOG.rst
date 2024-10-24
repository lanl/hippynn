

Breaking changes:
-----------------

- ``set_e0_values`` has been renamed to ``hierarchical_energy_initialization``.
  The old name is still provided but deprecated, and will be removed.
- The argument ``restore_db`` has been renamed to ``restart_db``. The affected
  functions are ``load_checkpoint``, ``load_checkpoint_from_cwd``, and
  ``restore_checkpoint``.
- ``database.make_trainvalidtest_split`` now only takes keyword arguments to
  avoid confusions. Use ``make_trainvalidtest_split(test_size=a, valid_size=b)``
  instead of ``make_trainvalidtest_split(a, b)``.
- Invalid custom kernel specifications are now errors rather than warnings.
- Method of specifying units for custom MD algorithms has changed.

New Features:
-------------

- Added a new custom cuda kernel implementation using triton.
  These are highly performant and now the default implementation.
- Exporting any database to NPZ or H5 format after preprocessing can be done with a method call.
- Database states can be cached to disk to simplify the restarting of training.
- Added batch geometry optimizer features in order to optimize geometries
  in parallel on the GPU. Algorithms include FIRE, Newton-Raphson, and BFGS.
- Added experiment pytorch lightning trainer to provide for simple parallelized training.
- Added a molecular dynamics engine which includes the ability to batch over systems.
- Added examples pertaining to coarse graining.
- Added pair finders based on scipy KDTree for training to large systems.
- Added tool to drastically simplify creating ensemble models. The ensemblized graphs
  are compatible with molecular dynamics codes such ASE and LAMMPS.
- Added the ability to weight different systems/atoms/bonds in a loss function.
- Added new function to reload library settings.


Improvements:
-------------

- Eliminated dependency on pyanitools for loading ANI-style H5 datasets.
- SNAPjson format can now support an optional number of comment lines.
- Added unit conversion options to the LAMMPS interface.
- Improved performance of bond order regression.
- It is now possible to limit the memory usage of the MLIAP interface in LAMMPS
  using a library setting.
- Provide tunable regularization of HIP-NN-TS with an epsilon parameter, and
  set the default to use a better value for epsilon.
- Improved detection of valid custom kernel implementation.
- Improved computational efficiency of HIP-NN-TS network.
- ``StressForceNode`` now also works with batch size greater than 1.


Bug Fixes:
----------

- Fixed bug where custom kernels were not launching properly on non-default GPUs
- Fixed error when LAMMPS interface is in kokkos mode and the kokkos device was set to CPU.
- MLIAPInterface objects
- Fixed bug with RDF computer automatic initialization.

0.0.3
=======

Breaking changes:
-----------------

- The minimum python version has been increased to 3.9

New Features:
-------------

- Add nodes for non-adiabatic coupling vectors (NACR) and phase-less loss.
  See /examples/excited_states_azomethane.py.

- New MultiGradient node for computing more than one partial derivative
  using a single call to automatic differentiation.

Improvements:
-------------

- Multi-target dipole node now has a shape of (n_molecules, n_targets, 3).

- Add out-of-range warning to FuzzyHistogrammer.

- Create Memory parent class to remove redundancy.

- New setting TIMEPLOT_AUTOSCALING. If True (default), time plots with 
  log-scaling on the axes will only be produced if warranted by the data.
  If False, time plots with linear-scaling and log-scaling will be produced
  every time.

Bug Fixes:
----------

- Fix KDTreePairs issue caused by floating point precision limitations.

- Fix KDTreePairs issue with not moving tensors off GPU.

- Enable PairMemory nodes to handle batch size > 1.


0.0.2a2
=======

New Features:
-------------

- New FuzzyHistogrammer node for transforming scalar feature into a fuzzy/soft 
  histogram array

- New PeriodicPairIndexerMemory node which removes the need to recompute 
  pairs for each model evaluation in some instances, leading to speed improvements

- New KDTreePairs and KDTreePairsMemory nodes for computing pairs using linearly-
  scaling KD Tree algorithm. 

Improvements:
-------------

- ASE database loader added to read any ASE file or list of ASE files.

Bug Fixes:
----------
- Function 'gemerate_database_info' renamed to 'generate_database_info.'

- Fixed issue with class Predictor arising when multiple names for the same output node are provided.

- Fixed issue with MolPairSummer when the batch size and the feature size are both one.

0.0.2a1
=======

Improvements
------------

- Filtering scheme for pairfinders to avoid processing of unneeded data.


0.0.1
=====

Improvements:
-------------

- new "glue-on" method for damping coulomb energies locally

- Improve compatibility with ASE functions such as mixing calculators
  and trajectory saving.

0.0.1b4
=======

New Features:
-------------

- Added an interface to LAMMPS using the LAMMPS MLIAP UNIFIED pair style.
  see /examples/lammps/ and the documentation for more information.

Improvements:
-------------

- Add a setting to create plots with transparent backgrounds

- Improvements to documentation display

- Add an example for training to the Ani-1x dataset directly from
  the h5 file.

- ASE Calculator is now compatible with more ASE functions including
  mixing with other calculators.

- Cross-device restarting is now properly handled. Corresponding documentation
  has been added.

Bug Fixes:
----------

- Fixed a bug which expected files saved in a .pkl format,
  when in fact they are saved as .pt (pytorch) files.

- Fixed a bug in parsing of local settings file.

- Fixed a bug in parsing of settings through environmental variables.

- Fixed a false low distance warning when sensitivity functions are plotted.


0.0.1b3
=======

New Features:
-------------

- Cupy based interaction kernels are now available (GPU only). These
  kernels are typically higher performance than numba-based kernels,
  although overall gains will depend on many factors.
  To activate the kernels, install cupy.

Improvements:
-------------

- Sorted values of pair-lists handled by custom kernels are now cached.
  This drastically improves the ease of saturating the GPU by reducing
  the need for pair synchronization

- Numba GPU kernel overhead has been reduced by speeding up the time
  to convert between the torch and numba GPU array types.

- Misc. other improvements to reduce CPU/GPU synchronization needs.

- PyAnitools database is now more flexible and can read additional properties,
  for example parsing the COMP6 test set.


Bug Fixes:
----------

- small bugs in database loading


0.0.1b1
=======

New features:
-------------

- PeriodicPairIndexer can now handle arbitrary cells sizes with
  arbitrary boundary conditions, and is suitable for use in
  general training sets. As a result, it is no longer necessary to use
  DynamicPeriodicPairs, and caching pairs is less likely
  to bring performance improvements.

Improvements
------------

- The throughput of DynamicPeriodicPairs has been dramatically increased.

- If numba fails to find a GPU, a better error message is displayed.

- Loss broadcasting debugging can be changed with a new setting variable.

Bug fixes:
----------

- Fixed a bug where _DispatchNeighbors module incorrectly indexed atoms
  in the case where blank atoms did not appear after real ones.

- Fixed a bug where an ASE calculator couldn't be created when the
  training PairFinder is a subclass of _DispatchNeighbors

- Fixed a bug where an ASE calculator couldn't be created when the
  encoder and species indexer were generated using a python list
  for species.

- Fixed a bug with the ASE calculator failing in open boundary conditions.

- Fixed an incompatibility between our API and the pytorch API that
  prevented loading pytorch schedulers from a checkpoint.

0.0.1a2
=======

New features:
-------------

- New Pair test format, ``PaddedNeighborNode``:
    - This node can convert pair-style lists into a flat array of neighbors for
      each atom in the batch.
    - The output indices will be padded with index values of [-1] so that the array
      is rectangular, and the output difference vectors padded with vectors of 0.

- New function ``calculate_min_dists``, node ``MinDistNode``
    - This node can compute the minimum distance from atoms to other atoms,
      and aggregate this information over systems.
    - The primary utility is encapsulated in ``hippynn.pretraining.calculate_min_dists``.
      This function computers the minimum distance between any pair of atoms for each
      molecule in the dataset. This information can be useful for identifying
      data which is physically problematic or for setting the initial parameters for
      distance sensitivity in a network.

Improvements:
-------------

- Pyanitools database improvements
    - Can now specify the key value to use as the species array.
    - Species array can be either string valued, i.e. ``['C','H','H','H']``,
      or integer valued, i.e. ``[6,1,1,1]``. Previously only strings were accepted.

Bug fixes:
----------

- DynamicPeriodicPairs would find pairs in the wrong images in some cases, fixed.

- Scalar broadcasting of a node with a scalar, e.g. in algebraic operations, was broken, this is fixed.

- ``allow_unfound`` argument for databases was not working for some database formats.

- Anitools Databases were not filtering arrays, this is fixed.

0.0.1a
======
Initial public release.

