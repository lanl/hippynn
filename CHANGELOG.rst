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
      and aggregate this information over molecules.
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

- Scalar broadcasting of a node with a scalar, e.g. in alebgraic operations, was broken, this is fixed.

- ``allow_unfound`` argument for databases was not working for some database formats.

- Anitools Databases were not filtering arrays, this is fixed.

0.0.1a
======
Initial public release.

