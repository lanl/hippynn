import unittest

import numpy as np
import torch


class TestNACRLayers(unittest.TestCase):

    # random number of molecules, atoms, and states
    n_mol, n_atoms, n_states = np.random.randint(3, 8, 3)
    # random initial positions and charges
    positions = torch.rand(n_mol, n_atoms, 3)
    positions.requires_grad = True
    energies = torch.rand(n_mol, n_states)
    layer = torch.nn.Linear(3, n_states)
    charges = layer(positions)

    def setUp(self):
        from hippynn.additional import NACR, NACRMultiState

        self.NACR_layer = NACR()
        self.NACR_multi_layer = NACRMultiState(self.n_states)

    def test_multi_targets(self):
        indices = torch.triu_indices(self.n_states, self.n_states, offset=1, device=self.positions.device)
        nacr_singles = torch.empty(self.n_mol, self.n_atoms, 3, len(indices[0]))
        for i, (j, k) in enumerate(zip(*indices)):
            nacr_singles[..., i] = self.NACR_layer(
                self.charges[..., j],
                self.charges[..., k],
                self.positions,
                self.energies[:, j].unsqueeze(1),
                self.energies[:, k].unsqueeze(1),
            )
        nacr_multi = self.NACR_multi_layer(self.charges, self.positions, self.energies)
        self.assertTrue(torch.equal(nacr_singles, nacr_multi))

    def _numpy_implementation(self):
        # TODO: with a linear layer, it's possible to use its weights to implement an analytical gradient in numpy.
        pass
