"""
Additional nodes and loss functions used for excited states training. 
"""
import torch

from ..graphs.nodes.base import SingleNode, AutoKw
from ..graphs import loss


class NACR(torch.nn.Module):
    """
    Compute NAC vector. Originally in hippynn.layers.physics.
    """

    def __init__(self):
        super().__init__()

    def forward(self, charges1, charges2, positions):
        nacr = torch.autograd.grad(
            charges1, [positions], grad_outputs=charges2, create_graph=True
        )[0]
        return nacr


class NACRNode(AutoKw, SingleNode):
    """
    Compute the non-adiabatic coupling vector between two states. Originally in
    hippynn.graphs.nodes.physics.
    """

    _input_names = "charge 1", "charge 2", "coordinates"
    # _auto_module_class = physics_layers.NACR
    _auto_module_class = NACR

    def __init__(self, name, parents, module="auto", **kwargs):
        self.module_kwargs = {}
        charges1, charges2, positions = parents
        positions.requires_grad = True
        parents = charges1.main_output, charges2.main_output, positions
        self._index_state = positions._index_state
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
