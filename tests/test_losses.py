import pytest

import hippynn


@pytest.fixture()
def energy_force_nodes(network_parameters):
    from hippynn.graphs import inputs, networks, targets, physics

    # model inputs
    species = inputs.SpeciesNode(db_name="Z")
    positions = inputs.PositionsNode(db_name="R")

    # Model computations
    network = networks.Hipnn("HIPNN", (species, positions), module_kwargs=network_parameters)
    henergy = targets.HEnergyNode("HEnergy", network)
    # molecule_energy = henergy.mol_energy
    # molecule_energy.db_name = "en"
    forces = physics.GradientNode("force", (henergy, positions), sign=-1, db_name="f")

    return henergy, forces

def test_build_loss(energy_force_nodes):
    from hippynn.graphs.nodes import loss
    henergy, forces = energy_force_nodes

    mae = loss.MAELoss.of_node(henergy)
    rmse = loss.MSELoss.of_node(henergy)**(1/2)

    total = mae + rmse
    
    return



def test_weighted_loss_from_input(energy_force_nodes):
    from hippynn.graphs import inputs, loss, IdxType

    henergy, forces = energy_force_nodes

    en_mask = inputs.InputNode(db_name="en_mask", index_state=IdxType.Systems)
    force_mask = inputs.InputNode(db_name="f_mask", index_state=IdxType.SysAtom)

    mse_energy_weighted = loss.WeightedMSELoss.of_node(henergy, en_mask)
    mse_force_weighted = loss.WeightedMSELoss.of_node(forces, force_mask)


def test_weighted_loss_string(energy_force_nodes):
    from hippynn.graphs import loss

    henergy, forces = energy_force_nodes
    mse_energy_weighted2 = loss.WeightedMSELoss.of_node(henergy, "en_mask")
    mse_force_weighted2 = loss.WeightedMSELoss.of_node(forces, "f_mask")
