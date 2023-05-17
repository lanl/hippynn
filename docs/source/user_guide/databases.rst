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


ASE Objects Database handling
----------------------------------------------------------
If your training data is stored as ASE files of any type (.json,.db,.xyz,.traj ... etc.) it can be loaded directly 
a Database for hippynn.

The ASE database :class:`~hippynn.databases.AseDatabase` can be loaded with ASE installed.

See ~/examples/ase_db_example.py for a basic example utilzing the class.