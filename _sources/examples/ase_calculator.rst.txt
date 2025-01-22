ASE Calculators
===============

Hippynn models can be used with ``ase`` to perform molecular dynamics or other tests.

To build an ASE calculator, use the :class:`~hippynn.interfaces.ase_interface.HippynnCalculator` object.
You pass the node associated with energy.
Example::

    from hippynn.interfaces.ase_interface import HippynnCalculator
    energy_node = model.node_from_name("energy")
    calc = HippynnCalculator(energy_node,en_unit=units.eV)
    calc.to(torch.float64)

Take note of the ``en_unit`` and ``dist_unit`` parameters for the calculator.
These parameters inform the calculator what units the model consumes and produces for energy and
for distance. If unspecified, the ``en_unit`` is kcal/mol (different from ase default of eV!),
and the ``dist_unit`` is angstrom. Whatever units your model uses, the output of the calculator
will be in the ``ase`` unit system, which has energy in eV and distance in Angstroms.

If your model only contains one energy node, you can use the function
:func:`~hippynn.interfaces.ase_interface.calculator_from_model`, which automatically identifies
the energy node and from this creates the calculator. This function accepts keyword arguments that will
be passed to the calculator.

Given an ase ``atoms`` object, one can assign the calculator::

    atoms.calc = calc

And proceed to perform whatever simulation is desired.

The :class:`~hippynn.interfaces.ase_interface.HippynnCalculator` also supports a charge node for charge and dipole predictions,
and generates calculations for force and stress based on the energy using pytorch's
automatic differentiation capabilities.