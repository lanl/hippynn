"""
    Unit tests for ASE Graph construction. 
"""

def ASE_FilterPair_Coulomb_Construct():    
    """ Construct ASE Calculator from a  HIPNN model with multiple PairIndexers and Coulomb Energy Node. 
    Ensures that graph construction using PairFilter does not break ASE graph. 

    Returns:
        status : True if ASE calculator can be built. 
        exception : Caught Exception if ASE calculator build fails.
    """

    import numpy as np

    import torch
    torch.set_default_dtype(torch.float32)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    # Hippynn imports. 
    import hippynn
    from hippynn.graphs import inputs, networks, targets, physics
    from hippynn.graphs.nodes import indexers, pairs 

    nacl_species_set = [0, 11, 17] 
    network_params = {
        "possible_species": nacl_species_set,   # Z values of the elements
        'n_features': 10,                     # Number of neurons at each layer
        "n_sensitivities": 8,                # Number of sensitivity functions in an interaction layer
        "dist_soft_min": 0.90,  # qm7 1.7  qm9 .85  AL100 .85
        "dist_soft_max": 6.5,  # qm7 10.  qm9 5.   AL100 5.
        "dist_hard_max": 7.5,  # qm7 15.  qm9 7.5  AL100 7.5
        "n_interaction_layers": 1,            # Number of interaction blocks
        "n_atom_layers": 1,                   # Number of atom layers in an interaction bloc
    }

    ### Define Models 
    from hippynn.graphs import inputs, networks, targets, physics

    # Input node (should be shared across both networks)
    species = inputs.SpeciesNode(db_name="Z")
    positions = inputs.PositionsNode(db_name="R")

    ### ACA network 
    network_aca = networks.Hipnn("aca", (species, positions), module_kwargs=network_params)

    # Targets
    hcharge = targets.HChargeNode("HCharge", network_aca)
    atom_charges = hcharge.atom_charges 
    molecule_charges = physics.AtomToMolSummer("molCharge", atom_charges)
    dipole = physics.DipoleNode("dipole", (hcharge, positions), db_name="dipole")

    # Manually define index nodes for coulomb energy only!
    # coulomb_r_max should be large enough to cover all atoms. 
    # Large coulomb_r_max can lead to very slow training. 
    coulomb_r_max = 100 # np.inf -> errors for periodic systems. 

    enc, padidxer = indexers.acquire_encoding_padding(
        species, 
        species_set = nacl_species_set
    )
    pairfinder = pairs.OpenPairIndexer(
        'OpenPairFinder', 
        (positions, enc, padidxer), 
        dist_hard_max=coulomb_r_max
    )

    # Coulomb Energy
    energy_conv = 14.397 # Coulomb-konst
    coulomb_energy = physics.CoulombEnergyNode(
        "cEnergy",
        # _input_names = "charges", "pair_dist", "pair_first", "pair_second", "mol_index", "n_molecules"
        (atom_charges,
            pairfinder.pair_dist, pairfinder.pair_first, pairfinder.pair_second,
            padidxer.mol_index, padidxer.n_molecules),
        energy_conversion=energy_conv, 
    )
    ### 

    # Energy Network. 
    network_energy = networks.Hipnn("hipnn", network_aca.parents, module_kwargs=network_params)    
    Henergy = targets.HEnergyNode("HEnergy", network_energy) 
    henergy = Henergy.mol_energy + coulomb_energy 
    force = physics.GradientNode("Grad", (henergy, positions), sign=+1)
    henergy.db_name = "energy" 
    force.db_name = "gradient"


    ### Define Loss Graph
    from hippynn.graphs import loss 

    # All the ACA losses 
    rmse_dipole =  loss.MSELoss.of_node(dipole) ** (1/2)
    mae_dipole = loss.MAELoss.of_node(dipole)
    rsq_dipole = loss.Rsq.of_node(dipole)
    loss_charge = loss.MeanSq.of_node(molecule_charges)**(1/2) # Regularizer for net-charge = 0 
    l2_reg_aca = loss.l2reg(network_aca)
    loss_reg_aca = 1e-4 * (l2_reg_aca) 
    loss_dipole = rmse_dipole + mae_dipole
    loss_aca = loss_charge + loss_dipole + loss_reg_aca

    # All Energy Loss
    mse_force = loss.MSELoss.of_node(force)
    rmse_force = mse_force ** (1 / 2)
    mae_force = loss.MAELoss.of_node(force)
    rsq_force =  loss.Rsq.of_node(force)
    rmse_energy = loss.MSELoss.of_node(henergy) ** (1 / 2)
    mae_energy = loss.MAELoss.of_node(henergy)
    rsq_energy =  loss.Rsq.of_node(henergy)
    loss_energy = (rmse_energy + mae_energy)
    loss_force = (rmse_force + mae_force)
    loss_hipnn = loss_energy + loss_force
    loss_train = loss_aca + loss_hipnn

    # Validation losses
    validation_losses = {
        "T-RMSE"      : rmse_energy,
        "T-MAE"       : mae_energy,
        "T-RSQ"       : rsq_energy,
        "F-RMSE"      : rmse_force,
        "F-MAE"       : mae_force,
        "F-RSQ"       : rsq_force,
        "D-RMSE"      : rmse_dipole,
        "D-MAE"       : mae_dipole,
        "D-RSQ"       : rsq_dipole,
        "C-Loss"      : loss_charge,
        "L2-reg"      : l2_reg_aca,
        "Loss_aca"    : loss_aca,
        "Loss_hipnn"  : loss_hipnn,
        "Loss"        : loss_train,
    }
    ###

    #Assemble Pytorch Model that can be trained. 
    training_modules, db_info = hippynn.experiment.assemble_for_training(
        loss_train,
        validation_losses,
    )

    # Extract the un-trained model    
    model = training_modules.model

    # Construct a Calculator from model. 
    from hippynn.interfaces.ase_interface import HippynnCalculator
    energy_node = model.node_from_name('add(HEnergy.mol_energy, cEnergy.mol_energies)')
    try:
        calc = HippynnCalculator(energy=energy_node)
        return True, None 
    except Exception as ex:
        return False, ex
        

if __name__ == "__main__":

    import unittest
    
    class ASE_FilterPICoulombTest(unittest.TestCase):
        def runTest(self):
            status, ex = ASE_FilterPair_Coulomb_Construct()
            self.assertEqual(status, True, "ASE Calculator Build Fails with exception : {}".format(ex))

    unittest.main()