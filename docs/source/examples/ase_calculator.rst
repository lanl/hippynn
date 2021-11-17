ASE Calculators
===============

Hippynn models can be used with `ase` to perform molecular dynamics or other tests.

To build an ASE calculator, you must pass the node associated with energy.
Example::

    from hippynn.interfaces.ase_interface import HippynnCalculator
    energy_node = model.node_from_name("energy")
    calc = HippynnCalculator(energy_node,en_unit=units.eV)
    calc.to(torch.float64)

Take note of the `en_unit` and `dist_unit` parameters for the calculator.
These parameters inform the calculator what units the model consumes and produces for energy and
for distance. If unspecified, the `en_unit` is kcal/mol, and the `dist_unit` is angstrom.
Whatever units your model uses, the output of the calculator will be in the `ase` unit system,
which has energy in eV and distance in Angstroms.

Given an ase atoms object, one can assign the calculator::

    atoms.calc = calc

And proceed to perform whatever simulation is desired.

The :class:`~hippynn.interfaces.ase_interface.HippynnCalculator` also supports a charge node for charge and dipole predictions,
and generates calculations for force and stress based on the energy using pytorch's
automatic differentiation capabilities.