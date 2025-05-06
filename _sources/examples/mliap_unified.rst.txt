LAMMPS interface
================

Hippynn models can be created from the LAMMPS ML-IAP Unified Abstract Base Class via
:class:`~hippynn.interfaces.lammps_interface.mliap_interface.MLIAPInterface`. These
models can used to calculate LAMMPS interatomic potentials.

To build a LAMMPS ML-IAP Unified model, you must pass the node associated with energy, a list of
species atomic symbols (whose order must agree with the order of the training hyperparameter
``possible_species``), and optionally the device to which to process torch data (e.g. ``"cpu"`` or ``"cuda"``).

Example::

    bundle = load_checkpoint_from_cwd(map_location="cpu", restart_db=False)
    model = bundle["training_modules"].model
    energy_node = model.node_from_name("HEnergy")
    unified = MLIAPInterface(energy_node, ["Al"], model_device=torch.device("cuda"))
    torch.save(unified, "mliap_unified_hippynn_Al_multilayer.pt")

After creating the Unified object, to perform a LAMMPS simulation you may ``pickle`` or
``torch.save`` it for use with a LAMMPS input script.
Example::

    pair_style	mliap unified mliap_unified_hippynn_Al.pt 0
    pair_coeff	* * Al

You may also load it directly into LAMMPS from the `mliappy` Python library.
Example::

    import lammps.mliap
    lammps.mliap.activate_mliappy(lmp)
    lmp.commands_string(before_loading)
    from lammps.mliap.mliap_unified_lj import MLIAPUnifiedLJ
    unified = MLIAPUnifiedLJ(["Ar"])
    lammps.mliap.load_unified(unified)
    lmp.commands_string(after_loading)

Note that you must call ``lammps.mliap.activate_mliappy()`` before loading the unified model.
Here ``before_loading`` would be a string of commands up to but not including
the ``pair_style mliap unified`` command in lammps, and ``after loading`` would be the commands
to run including the ``pair_style`` command and anything to run afterwards.


Using LAMMPS activation communication
-------------------------------------

When using multiple interaction layers, it is normally necessary to increase the size of the halo region
so that atoms from neighboring processors can be seen at a distance equal to the total interaction length
of the model. However, the environment flag HIPPYNN_COMM_FEATURES_LAMMPS allows for hippynn to message-pass
the activations and their gradients through LAMMPS's MPI infrastructure and thereby broadcast the neuron
values to the appropriate halo region, meaning that the halo only has to be as large as a single
interaction layer. This requires using the kokkos variant of LAMMPS-MLIAP.

Please note that this does not necessarily guarantee a speed increase. The trade-off of enabling this
feature is to use more communication per atom, but in reducing the number of atoms that need to be
communicated as well as the the number of atoms which the neural network needs to be applied to. This
does mean that the expected memory cost should go down when HIPPYNN_COMM_FEATURES_LAMMPS is active.
However, whether the simulation is overall faster or slower can depend on many other aspects of
the simulation setup, such as how it is partitioned across processors, what the structure of the
model is, and many factors in the underlying hardware.

For more information, see the `LAMMPS mliap documentation <lammps_mliap_docs_>`_ and
the `LAMMPS kokkos documentation <lammps_kokkos_docs_>`_

.. _lammps_mliap_docs: https://docs.lammps.org/pair_mliap.html
.. _lammps_kokkos_docs: https://docs.lammps.org/Speed_kokkos.html
