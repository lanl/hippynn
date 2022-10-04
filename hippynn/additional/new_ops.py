"""
Additional nodes and loss functions used for excited states training. 
"""
import torch

from ..graphs.nodes.base import AutoKw, ExpandParents, SingleNode
from ..graphs import loss


class NACR(torch.nn.Module):
    """
    Compute NAC vector * ΔE. Originally in hippynn.layers.physics.
    """

    def __init__(self):
        super().__init__()

    def forward(self, charges1, charges2, positions, energy1, energy2):
        dE = energy1 - energy2
        nacr = torch.autograd.grad(
            charges2, [positions], grad_outputs=[charges1, dE], create_graph=True
        )[0]
        return nacr


class NACRNode(ExpandParents, AutoKw, SingleNode):
    """
    Compute the non-adiabatic coupling vector multiplied by the energy difference
    between two states. Originally in hippynn.graphs.nodes.physics.
    """

    _input_names = "charges i", "charges j", "coordinates", "energy i", "energy j"
    # _auto_module_class = physics_layers.NACR
    _auto_module_class = NACR

    def __init__(self, name, parents, module="auto", **kwargs):
        """
        Automatically build the node for calculating NACR * ΔE between two states i
        and j.

        Args:
            name (str): name of the node
            parents (tuple): parents of the NACR node in the sequence of (charges i, \
                charges j, positions, energy i, energy j)
        """

        self.module_kwargs = {}
        charges1, charges2, positions, energy1, energy2 = parents
        positions.requires_grad = True
        self._index_state = positions._index_state
        parents = (
            charges1.main_output,
            charges2.main_output,
            positions,
            energy1.main_output,
            energy2.main_output,
        )
        super().__init__(name, parents, module=module, **kwargs)


# For loss functions with phases
def absolute_errors(predict, true):
    return torch.minimum(torch.abs(true - predict), torch.abs(true + predict))


def mae_with_phases(predict, true):
    errors = absolute_errors(predict, true)
    return torch.sum(errors) / predict.numel()


def mse_with_phases(predict, true):
    errors = absolute_errors(predict, true) ** 2
    return torch.sum(errors) / predict.numel()


class MAEPhaseLoss(loss._BaseCompareLoss, op=mae_with_phases):
    pass


class MSEPhaseLoss(loss._BaseCompareLoss, op=mse_with_phases):
    pass
