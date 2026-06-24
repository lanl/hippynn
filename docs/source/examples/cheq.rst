Charge Equilibration
=============================

`hippynn` has features for incorporating charge equilibration model (ChEQ) 
to include long-range interactions. 
These features can be found in :mod:`~hippynn.graphs.nodes.cheq`.

For a more detailed description, please see the paper [Li2025]_

The HIPNN-ChEQ model learns short-range energy (:math:`V^{S}_{i}`), electronegativity (:math:`\chi_{i}`), and Hubbard-U (:math:`u_{i}`)
indirectly through training with total energy, forces, and dipole. 

This is based on the second-order monopole charge equilibration model as following,

.. math::
   
   E({\bf R,q}) =  \sum_i V_i^{\rm S}({\bf R}) + \sum_i q_i \chi_i({\bf R}) + \frac{1}{2} \sum_i q_i^2 u_i({\bf R})  + \frac{1}{2} \sum_{ij}^{i \ne j} q_i \gamma_{ij} q_j 

Here index `i` runs over the atomic centers, :math:`i = 1,2,\ldots N`, :math:`{\bf q} = \{q_i\}` 
are the net partial charges on each atom, :math:`V_i^{\rm S}({\bf R})` are charge-independent,
short-range energy terms for each atom at position, :math:`{\bf R}_i`, and :math:`\chi_i({\bf R})` and :math:`u_i({\bf R})`
are the atomic electronegativities and chemical hardness (or Hubbard-U) parameters, respectively. 
We assume that these depend not only on the atom type but also on their local atomic environments. 
The charge-independent energy terms, :math:`V_i^{\rm S}({\bf R})`, as well as :math:`\chi_i({\bf R})` and :math:`u_i({\bf R})`
are parameterized using HIPNN that captures the local many-body interactions of each atom. This 
allows us to leverage HIPNN to parameterize their values based on reference data generated from
first-principles theory. 


We construct two HIPNN networks to predict the 1) short-range energy and 
2) electronegativity and Hubbard-U separately. (Note: it is not absolutely necessary to use two different networks,
 but we have found it to be effective--this way gradients are split to parameters in the same way as the energy is split.)

For the short-range energy, the node can be constructed using :class:`~hippynn.graphs.nodes.targets.HEnergyNode`::

    network1 = networks.Hipnn("HIPNN2", network.parents, module_kwargs = network1_params)
    henergy1 = targets.HEnergyNode("HEnergy",network1)

On the other hand, we build another HIP-NN network for the electronegativity and Hubbard-U predictions 
using :class:`~hippynn.graphs.nodes.cheq.ChEQNode`::

    network = networks.Hipnn("HIPNN", (species, positions), module_kwargs = network_params)
    henergy = ChEQNode("ChEQ", (network,), units={'energy':'kcal/mol', 'length':"Angstrom"}, lower_bound=0.01)

Then, we need to define the total energy, forces, and dipoles for training::

    dipole = henergy.dipole

    molecule_energy = henergy.coul_energy + henergy1.sys_energy 
    gradient = physics.GradientNode("Gradient", (molecule_energy, positions), sign=+1)

    molecule_energy.db_name="energies"
    gradient.db_name = "Grad"
    dipole.db_name = "dipole"

In order to obtain a HIPNN-ChEQ model which can run stable molecular dynamcis, we emphasize
the loss of forces and dipole more over the total energy::

    loss_error = 1.0 * (rmse_energy + mae_energy) + 100.0 * (rmse_grad + mae_grad) + 
                10000.0 * (rmse_dipole + mae_dipole)

For a complete script, please take a look at ``examples/cheq_ani2x_training.py``.

.. [Li2025] | Shadow Molecular Dynamics with a Machine Learned Flexible Charge Potential. 
            | Li et. al, 2025. https://pubs.acs.org/doi/10.1021/acs.jctc.5c00062 
