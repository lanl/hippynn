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

def test_build_cheq(neural_network_node):

    cheq = physics.ChEQNode("ChEQ", (neural_network_node,), units={'energy':'kcal/mol', 'length':"Angstrom"}, lower_bound=0.01)

    return

@pytest.mark.filterwarnings("ignore:.*Wolf implementation currently uses exact derivative*.")
def test_build_coulomb(network_parameters):
    # requires open boundary so text fixture errors.
    from hippynn.graphs import inputs, networks, find_unique_relative
    from hippynn.graphs.nodes.tags import PairIndexer

    species = inputs.SpeciesNode(db_name="species")
    positions = inputs.PositionsNode(db_name="coordinates")
    neural_network_node = networks.Hipnn("HIPNN", (species, positions), module_kwargs=network_parameters, periodic=False)
    pairfinder = find_unique_relative(neural_network_node,PairIndexer)
    hcharge = targets.HChargeNode("Charges", neural_network_node)

    pairfinder.dist_hard_max = None
    pairfinder.torch_module.hard_dist_cutoff = None

    coula = physics.CoulombEnergyNode("coulomb energy",hcharge,energy_conversion_factor=1)

    coulb = physics.ScreenedCoulombEnergyNode(
        "coulomb energy", hcharge, energy_conversion_factor=1,
        cutoff_distance=10., screening=hippynn.layers.physics.WolfScreening(alpha=0.1)
    )

    combined = physics.CombineEnergyNode("Combined Energy", (coula, coulb))
    
    return
