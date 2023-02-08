"""
Additional nodes and loss functions used for excited states training. 
"""
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from ..graphs import loss
from ..graphs.nodes.base import AutoKw, SingleNode
from ..graphs.indextypes import IdxType


class NACR(torch.nn.Module):
    """
    Compute NAC vector * ΔE. Originally in hippynn.layers.physics.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        charges1: Tensor,
        charges2: Tensor,
        positions: Tensor,
        energy1: Tensor,
        energy2: Tensor,
    ):
        dE = energy2 - energy1
        nacr = torch.autograd.grad(
            charges2, [positions], grad_outputs=[charges1], create_graph=True
        )[0].reshape(len(dE), -1)
        return nacr * dE


class NACRMultiState(torch.nn.Module):
    """
    Compute NAC vector * ΔE for all paris of states. Originally in hippynn.layers.physics.
    """

    def __init__(self, n_target=1):
        self.n_target = n_target
        super().__init__()

    def forward(self, charges: Tensor, positions: Tensor, energies: Tensor):
        # charges shape: n_molecules, n_atoms, n_targets
        # positions shape: n_molecules, n_atoms, 3
        # energies shape: n_molecules, n_targets
        # dE shape: n_molecules, n_targets, n_targets
        dE = energies.unsqueeze(1) - energies.unsqueeze(2)
        # take the upper triangle excluding the diagonal
        indices = torch.triu_indices(
            self.n_target, self.n_target, offset=1, device=dE.device
        )
        # dE shape: n_molecules, n_pairs
        # n_pairs = n_targets * (n_targets - 1) / 2
        dE = dE[..., indices[0], indices[1]]
        # compute q1 * dq2/dR
        nacr_ij = []
        for i, j in zip(*indices):
            nacr = torch.autograd.grad(
                charges[..., j],
                positions,
                grad_outputs=charges[..., i],
                create_graph=True,
            )[0]
            nacr_ij.append(nacr)
        # nacr shape: n_molecules, n_atoms, 3, n_pairs
        nacr = torch.stack(nacr_ij, dim=1)
        n_molecule, n_pairs, n_atoms, n_dims = nacr.shape
        nacr = nacr.reshape(n_molecule, n_pairs, n_atoms * n_dims)
        # multiply dE
        return nacr * dE.unsqueeze(2)


class NACRNode(AutoKw, SingleNode):
    """
    Compute the non-adiabatic coupling vector multiplied by the energy difference
    between two states. Originally in hippynn.graphs.nodes.physics.
    """

    _input_names = "charges i", "charges j", "coordinates", "energy i", "energy j"
    # _auto_module_class = physics_layers.NACR
    _auto_module_class = NACR

    def __init__(
        self, name: str, parents: Tuple, module="auto", module_kwargs=None, **kwargs
    ):
        """Automatically build the node for calculating NACR * ΔE between two states i
        and j.

        :param name: name of the node
        :type name: str
        :param parents: parents of the NACR node in the sequence of (charges i, \
                charges j, positions, energy i, energy j)
        :type parents: Tuple
        :param module: _description_, defaults to "auto"
        :type module: str, optional
        :param module_kwargs: keyword arguments passed to the corresponding layer,
            defaults to None
        :type module_kwargs: dict, optional
        """

        self.module_kwargs = {}
        if module_kwargs is not None:
            self.module_kwargs.update(module_kwargs)
        charges1, charges2, positions, energy1, energy2 = parents
        positions.requires_grad = True
        self._index_state = IdxType.Molecules
        # self._index_state = positions._index_state
        parents = (
            charges1.main_output,
            charges2.main_output,
            positions,
            energy1.main_output,
            energy2.main_output,
        )
        super().__init__(name, parents, module=module, **kwargs)


class NACRMultiStateNode(AutoKw, SingleNode):
    """
    Compute the non-adiabatic coupling vector multiplied by the energy difference
    between all pairs of states. Originally in hippynn.graphs.nodes.physics.
    """

    _input_names = "charges", "coordinates", "energies"
    # _auto_module_class = physics_layers.NACR
    _auto_module_class = NACRMultiState

    def __init__(self, name, parents, module="auto", module_kwargs=None, **kwargs):
        """Automatically build the node for calculating NACR * ΔE between all pairs of
        states.

        :param name: name of the node
        :type name: str
        :param parents: parents of the NACR node in the sequence of (charges, \
                positions, energies)
        :type parents: Tuple
        :param module: _description_, defaults to "auto"
        :type module: str, optional
        :param module_kwargs: keyword arguments passed to the corresponding layer,
            defaults to None
        :type module_kwargs: dict, optional
        """

        self.module_kwargs = {}
        if module_kwargs is not None:
            self.module_kwargs.update(module_kwargs)
        charges, positions, energies = parents
        positions.requires_grad = True
        self._index_state = IdxType.Molecules
        # self._index_state = positions._index_state
        parents = (
            charges.main_output,
            positions,
            energies.main_output,
        )
        super().__init__(name, parents, module=module, **kwargs)


# For loss functions with phases
def absolute_errors(predict: Tensor, true: Tensor):
    """Compute the absolute errors with phases between predicted and true values. In
    other words, prediction should be close to the absolute value of true, and the sign
    does not matter.

    :param predict: predicted values
    :type predict: torch.Tensor
    :param true: true values
    :type true: torch.Tensor
    :return: errors
    :rtype: torch.Tensor
    """

    return torch.minimum(torch.abs(true - predict), torch.abs(true + predict))


def mae_with_phases(predict: Tensor, true: Tensor):
    """MAE with phases

    :param predict: predicted values
    :type predict: torch.Tensor
    :param true: true values
    :type true: torch.Tensor
    :return: MAE with phases
    :rtype: torch.Tensor
    """

    errors = torch.minimum(
        torch.linalg.norm(true - predict, ord=1, dim=-1),
        torch.linalg.norm(true + predict, ord=1, dim=-1),
    )
    # errors = absolute_errors(predict, true)
    return torch.sum(errors) / predict.numel()


def mse_with_phases(predict: Tensor, true: Tensor):
    """MSE with phases

    :param predict: predicted values
    :type predict: torch.Tensor
    :param true: true values
    :type true: torch.Tensor
    :return: MSE with phases
    :rtype: torch.Tensor
    """

    errors = torch.minimum(
        torch.linalg.norm(true - predict, dim=-1),
        torch.linalg.norm(true + predict, dim=-1),
    )
    # errors = absolute_errors(predict, true) ** 2
    return torch.sum(errors**2) / predict.numel()


class MAEPhaseLoss(loss._BaseCompareLoss, op=mae_with_phases):
    pass


class MSEPhaseLoss(loss._BaseCompareLoss, op=mse_with_phases):
    pass
