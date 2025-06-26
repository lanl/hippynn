import pytest

import hippynn
from hippynn.graphs import networks, targets, physics



@pytest.mark.parametrize(
    "net_class,",
    [
        networks.Hipnn,
        networks.HipnnVec,
        networks.HipnnQuad,
        pytest.param(networks.HipHopnn, marks=pytest.mark.filterwarnings("ignore:.*Beta.*")),
    ],
)
def test_build_network(net_class, network_parameters):
    from hippynn.graphs import inputs
    species = inputs.SpeciesNode(db_name="species")
    positions = inputs.PositionsNode(db_name="coordinates")
    cell = inputs.CellNode(db_name="cell")

    network = net_class("HIPNN", (species, positions, cell), module_kwargs=network_parameters, periodic=True)
    return


@pytest.mark.parametrize(
    "target_cls",
    [
        targets.HEnergyNode,
        targets.HChargeNode,
        targets.AtomizationEnergyNode,
    ],
)
def test_build_atom_target(target_cls, neural_network_node):
    target_node = target_cls("target", neural_network_node)
    return



def test_build_bonds(neural_network_node, bond_parameters):

    bonds = targets.HBondNode("bonds", neural_network_node, module_kwargs=bond_parameters)
    return


@pytest.mark.parametrize(
    "target_cls",
    [
        targets.HEnergyNode,
        targets.AtomizationEnergyNode,
    ],
)
def test_build_forces(target_cls, neural_network_node):
    energy = target_cls("energy", neural_network_node)
    from hippynn.graphs import inputs, physics
    from hippynn.graphs import find_unique_relative

    positions = find_unique_relative(energy, inputs.PositionsNode)
    force = physics.GradientNode("force", (energy, positions), sign=-1)


@pytest.mark.parametrize("moment_cls", [physics.DipoleNode, physics.QuadrupoleNode])
def test_build_charge_moment(moment_cls, neural_network_node):
    charge = targets.HChargeNode("charge", neural_network_node)

    moment = moment_cls("charge_moment", charge)

