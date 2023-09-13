Excited States Training
==============

hippynn now is able to predict excited-state energies, transition dipoles, and
the non-adiabatic coupling vectors (NACR) for a given molecule.

Multi-targets nodes are recommended due to efficiency and fewer recursive
layers.

For energies, the node can be constructed just like the ground-state
counterpart::

    energy = targets.HEnergyNode("E", network, module_kwargs={"n_target": n_states + 1})
    mol_energy = energy.mol_energy
    mol_energy.db_name = "E"

Note that a ``multi-target node`` is used here, defined by the keyword
``module_kwargs={"n_target": n_states + 1}``. Here, `n_states` is the number of
states in consideration. The extra state is for the ground state, which is often
useful. The database name is simply `E` with a shape of ``(n_molecules,
n_states+1)``.

Predicting the transition dipoles is also similar to the ground-state permanent
dipole::

    charge = targets.HChargeNode("Q", network, module_kwargs={"n_target": n_states})
    dipole = physics.DipoleNode("D", (charge, positions), db_name="D")

The database name is `D` with a shape of ``(n_molecules, n_states, 3)``.

For NACR, to avoid singularity problems, we enforcing the training of NACR*ΔE
instead::

    nacr = physics.NACRMultiStateNode(
        "ScaledNACR",
        (charge, positions, energy),
        db_name="ScaledNACR",
        module_kwargs={"n_target": n_states},
    )

For NACR between state `i` and `j`, :math:`\boldsymbol{d}_{ij}`, it is expressed
in the following way

.. math::
    \boldsymbol{d}_{ij}\Delta E_{ij} = \Delta E_{ij}\boldsymbol{q}_i \frac{\partial\boldsymbol{q}_j}{\partial\boldsymbol{R}}

:math:`E_{ij}` is energy difference between state `i` and `j`, which is
calculated internally in the NACR node based on the input of the ``energy``
node. :math:`\boldsymbol{R}` corresponding the ``positions`` node in the code.
:math:`\boldsymbol{q}_{i}` and :math:`\boldsymbol{q}_{j}` are the transition
atomic charges for state `i` and `j` contained in the ``charge`` node. This
charge node can be constructed from scratch or reused from the dipole
predictions. The database name is `ScaledNACR` with a shape of ``(n_molecules,
n_states*(n_states-1)/2, 3*n_atoms)``.

Due to the phase problem, when the loss function is constructed, the
`phase-less` version of MAE or RMSE should be used::

    energy_mae = loss.MAELoss.of_node(energy)
    dipole_mae = loss.MAEPhaseLoss.of_node(dipole)
    nacr_mae = loss.MAEPhaseLoss.of_node(nacr)

:func:`~hippynn.graphs.nodes.loss.MAEPhaseLoss` and
:func:`~hippynn.graphs.nodes.loss.MSEPhaseLoss` are the `phase-less` version MAE
and MSE, respectively, behaving exactly like the common version.

For a complete script, please take a look at ``examples/excited_states.py``.
