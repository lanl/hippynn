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
