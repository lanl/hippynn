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
    rmse = loss.MSELoss.of_node(henergy) ** (1 / 2)

    total = mae + rmse

    return


def test_loss_broadcast_guard():
    """
    Tests that inputs from a database will be wrapped with an unsqueeze.
    """
    import torch
    from hippynn.graphs.nodes import inputs, loss
    from hippynn.graphs import GraphModule

    a = inputs.InputNode(db_name="a")

    mae = loss.MAELoss.of_node(a)

    g = GraphModule([a.pred, a.true], [mae])

    true = torch.arange(5, dtype=torch.float)
    predicted = true.unsqueeze(1)

    out = g(predicted, true)
    out = out[0].item()
    assert out == 0.0, f"Should give zero, but gave {out}."

    # flipping the order (true has extra 1 in shape, but predicted does not) still gives a user warning.
    with pytest.warns(UserWarning) as recorder:
        out = g(true, predicted)

    message = recorder[0].message.args[0]

    assert "incorrect results due to broadcasting" in message

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


def test_weighted_huber_layer():
    import torch
    from hippynn.layers.algebra import WeightedHuberLoss

    pred = torch.tensor([0.0, 1.0, 4.0, -3.0])
    true = torch.tensor([0.5, 1.0, 0.0, 0.0])
    weights = torch.tensor([1.0, 2.0, 0.5, 1.5])

    elementwise = torch.nn.functional.huber_loss(pred, true, reduction="none")
    expected = ((weights / weights.mean()) * elementwise).mean()
    assert torch.allclose(WeightedHuberLoss()(pred, true, weights), expected)

    # Uniform weights reduce to the plain huber loss.
    uniform = torch.ones_like(pred)
    plain = torch.nn.functional.huber_loss(pred, true)
    assert torch.allclose(WeightedHuberLoss()(pred, true, uniform), plain)

    # delta must reach the computation: a residual of 2 is in the linear
    # regime for delta=1 (2 - 0.5 = 1.5) but quadratic for delta=5 (2**2 / 2 = 2.0).
    pred2 = torch.full((4,), 2.0)
    true2 = torch.zeros(4)
    assert torch.allclose(WeightedHuberLoss()(pred2, true2, uniform), torch.tensor(1.5))
    assert torch.allclose(WeightedHuberLoss(delta=5.0)(pred2, true2, uniform), torch.tensor(2.0))


def test_huber_node_delta():
    import torch
    from hippynn.graphs import GraphModule
    from hippynn.graphs.nodes import inputs, loss

    a = inputs.InputNode(db_name="a")
    default = loss.HuberLoss.of_node(a)
    wide = loss.HuberLoss.of_node(a, delta=5.0)
    g = GraphModule([a.pred, a.true], [default, wide])

    predicted = torch.full((4, 1), 2.0)
    true = torch.zeros(4, 1)
    out_default, out_wide = g(predicted, true)
    assert out_default.item() == pytest.approx(1.5)
    assert out_wide.item() == pytest.approx(2.0)

    repr(g)  # the delta-carrying module must stay printable


def test_weighted_huber_node():
    import torch
    from hippynn.graphs import GraphModule
    from hippynn.graphs.nodes import inputs, loss

    a = inputs.InputNode(db_name="a")
    w = inputs.InputNode(db_name="w")
    weighted = loss.WeightedHuberLoss.of_node(a, w, delta=2.0)
    g = GraphModule([a.pred, a.true, w.true], [weighted])

    predicted = torch.tensor([[3.0], [0.5], [-4.0], [1.0]])
    true = torch.zeros(4, 1)
    weights = torch.tensor([[1.0], [3.0], [0.5], [2.0]])

    elementwise = torch.nn.functional.huber_loss(predicted, true, reduction="none", delta=2.0)
    expected = ((weights / weights.mean()) * elementwise).mean()
    assert torch.allclose(g(predicted, true, weights)[0], expected)
