Databases
=============


``hippynn`` :mod:`~hippynn.databases` wrap pytorch datasets.
They aim to provide for a simple interface for going from on-disk data to training,
as well as the capability to re-open the database (restart) without pickling all of the data.

Arrays can have arbitrary names, which should be assigned to the `db_name` of a node
to which they correspond.

The basic :class:`~hippynn.databases.Database` object takes a set of numpy arrays in
the following formats::

    system_variables.shape == (n_systems, *variable_shape)
    atom_variables.shape   == (n_systems, n_atoms_max, *variable_shape)
    bond_variables.shape   == (n_systems, n_atoms_max, n_atoms_max, *variable_shape)

e.g., for charges, your array should have a shape (n_systems, n_atoms_max,1).
for positions, your array should have a shape (n_systems, n_atoms_max,3).
For species, your array should have a shape (n_systems,n_atoms_max) -- this
is the one exception to the shapes given above.
Note that input of bond variables for periodic systems can be ill-defined
if there are multiple bonds between the same pairs of atoms. This is not yet
supported.

A note on *cell* variables. The shape of a cell variable should be specified as (n_systems,3,3), as described above.
It is important to know that there are two common conventions for the cell matrix itself; we use the convention that the basis index
comes first, and the cartesian index comes second. That is, similar to the ``ase`` package,
the element ``cell[sys,i,j]`` gives the ``j`` cartesian coordinate of cell vector ``i`` in system ``sys``. If you experience
massive errors while fitting to periodic boundary conditions, you may check the transposed version
of your cell data, or compute the RDF.

Predefined edges
-----------------------

When using :class:`~hippynn.graphs.nodes.inputs.PredefinedEdgeIndicesNode`,
the corresponding database array should have shape
``(n_systems, 2, num_edges)``. For each system, first array is the source of
directed edge and the second array is the target of the edge. The source and target 
rows are matched by the index.

For example, for a single system with the edges ``0 -> 1``, 
``0 -> 2``, ``1 -> 0``, and ``2 -> 0`` are stored as four columns:

    [[0, 0, 1, 2],
     [1, 2, 0, 0]]

Where:
- column ``0`` is ``[0, 1]``, so the edge is ``0 -> 1``.
- column ``1`` is ``[0, 2]``, so the edge is ``0 -> 2``.
- column ``2`` is ``[1, 0]``, so the edge is ``1 -> 0``.
- column ``3`` is ``[2, 0]``, so the edge is ``2 -> 0``.

Batched with a second two-atom system and padded to four edge columns, this 
is stored as::

    edge_indices = np.array(
        [
            [[0, 0, 1, 2],
             [1, 2, 0, 0]],
            [[0, 1, -1, -1],
             [1, 0, -1, -1]],
        ],
        dtype=np.int64,
    )

For periodic predefined edges, the array may instead have shape
``(n_systems, 5, num_edges)``. The first two rows are still the source and
target atom indices, and rows ``2:5`` give the integer cell offset vector for
each edge. These offsets are used together with the system cell to compute the
periodic displacement for the supplied edge.

The graph input should use the same database name as the stored array, for
example::

    edge_indices = inputs.PredefinedEdgeIndicesNode(db_name="edge_indices")
    network = networks.Hipnn("HIPNN", (species, positions, edge_indices), module_kwargs=network_params)

Supplying predefined edge indices makes hippynn build pairs from these
given edges instead of finding neighbors by radial distance. The pair list is
therefore not filtered by ``dist_hard_max``. By default, HIP-NN networks built
from predefined edges also use ``NoCutoff`` for the sensitivity cutoff, so
messages along the supplied edges are not zeroed by distance. To keep the usual
cosine sensitivity cutoff while using predefined edges, pass
``cutoff_type=CosCutoff`` in ``module_kwargs``.

For periodic predefined edges, the array may instead have shape
``(n_systems, 5, num_edges)``. The first two rows are still the source and
target atom indices, and rows ``2:5`` give the integer cell offset vector for
each edge.

Database Formats and notes
---------------------------

Numpy arrays on disk
........................

see :class:`hippynn.databases.NPZDatabase` (if arrays are stored
in a `.npz` dictionary) or :class:`hippynn.databases.DirectoryDatabase`
(if each array is in its own file).

Numpy arrays in memory
........................

Use the base :class:`hippynn.databases.Database` class directly to initialize
a database from a dictionary mapping db_names to numpy arrays.

pyanitools H5 files
........................

See :class:`hippynn.databases.PyAniFileDB` and see :class:`hippynn.databases.PyAniDirectoryDB`.

This format requires ``h5py`` and ``ase`` to be installed.

Snap JSON Format
........................

See :class:`hippynn.databases.SNAPDirectoryDatabase`. This format requires ``ase`` to be installed.

For more information on this format, see the FitSNAP_ software.

.. _FitSNAP: https://fitsnap.github.io

ASE Database
........................

If your training data is stored as ASE files of any type,
(.json,.db,.xyz,.traj ... etc.) it can be loaded directly
as a Database for hippynn.

The ASE database :class:`~hippynn.databases.AseDatabase` can be loaded with ASE installed.

See ~/examples/ase_db_example.py for a basic example utilizing the class.
