import torch
from . import indexers
from torch import Tensor


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


class LocalEnergy(torch.nn.Module):
    def __init__(self, feature_sizes, first_is_interacting=False):

        super().__init__()
        self.first_is_interacting = first_is_interacting
        if first_is_interacting:
            feature_sizes = feature_sizes[1:]

        self.feature_sizes = feature_sizes

        self.summer = indexers.MolSummer()
        self.n_terms = len(feature_sizes)
        biases = (first_is_interacting, *(True for _ in range(self.n_terms - 1)))

        self.layers = torch.nn.ModuleList(torch.nn.Linear(nf, 1, bias=bias) for nf, bias in zip(feature_sizes, biases))
        self.players = torch.nn.ModuleList(torch.nn.Linear(nf, 1, bias=False) for nf in feature_sizes)
        self.ninf = float("-inf")

    def forward(self, all_features, mol_index, atom_index, n_molecules, n_atoms_max):
        """
        :param all_features: list of feature tensors
        :param mol_index: which molecule is the atom
        :param atom_index: which atom in the molecule is that atom
        :param n_molecules: total number of molecules in the batch
        :param n_atoms_max: maximum number of atoms in the batch
        :return: contributed_energy, atom_energy, atom_preenergy, prob, propensity
        """

        if self.first_is_interacting:
            all_features = all_features[1:]

        partial_preenergy = [lay(x) for x, lay in zip(all_features, self.layers)]
        atom_preenergy = sum(partial_preenergy)
        partial_potentials = [lay(x) for x, lay in zip(all_features, self.players)]
        propensity = sum(partial_potentials)  # Keep in mind that this has shape (natoms,1)

        # This segment does not need gradients, we are constructing the subtraction parameters for softmax
        # which results in a calculation that does not under or overflow; the result is most accurate this way
        # But actually invariant to the subtraction used, so it does not require a grad.
        # It's a standard SoftMax technique, however, the implementation is not built into pytorch for
        # the molecule/atom framework.
        with torch.autograd.no_grad():
            propensity_molatom = all_features[0].new_full((n_molecules, n_atoms_max, 1), self.ninf)
            propensity_molatom[mol_index, atom_index] = propensity
            propensity_norms = propensity_molatom.max(dim=1)[0]  # first element is max vals, 2nd is max position
            propensity_norm_atoms = propensity_norms[mol_index]

        propensity_normed = propensity - propensity_norm_atoms

        # Calculate probabilities with molecule version of softmax
        relative_prob = torch.exp(propensity_normed)
        z_factor_permol = self.summer(relative_prob, mol_index, n_molecules)
        atom_zfactor = z_factor_permol[mol_index]
        prob = relative_prob / atom_zfactor

        # Find molecular sum
        atom_energy = prob * atom_preenergy
        contributed_energy = self.summer(atom_energy, mol_index, n_molecules)

        return contributed_energy, atom_energy, atom_preenergy, prob, propensity
