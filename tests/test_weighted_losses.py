import pytest

import hippynn


@pytest.fixture()
def weight_graph_variables(network_parameters):
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


def test_weighted_loss_from_input(weight_graph_variables):
    from hippynn.graphs import inputs, loss, IdxType

    henergy, forces = weight_graph_variables

    en_mask = inputs.InputNode(db_name="en_mask", index_state=IdxType.Molecules)
    force_mask = inputs.InputNode(db_name="f_mask", index_state=IdxType.SysAtom)

    mse_energy_weighted = loss.WeightedMSELoss.of_node(henergy, en_mask)
    mse_force_weighted = loss.WeightedMSELoss.of_node(forces, force_mask)


def test_weighted_loss_string(weight_graph_variables):
    from hippynn.graphs import loss

    henergy, forces = weight_graph_variables
    mse_energy_weighted2 = loss.WeightedMSELoss.of_node(henergy, "en_mask")
    mse_force_weighted2 = loss.WeightedMSELoss.of_node(forces, "f_mask")
