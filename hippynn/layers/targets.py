"""
Layers for target prediction such as Energy and Charge
"""
import torch

from . import indexers
from . import hiplayers


class HEnergy(torch.nn.Module):
    """
    Predict a system-level scalar such as energy from a sum over local components.
    """

    def __init__(self, feature_sizes, first_is_interacting=False, n_target=1):
        """

        :param feature_sizes: size of features at each level
        :param first_is_interacting: whether to drop the first set of (non-interacting) features.
        :param n_target: number of items to regress to.
        """

        super().__init__()
        self.first_is_interacting = first_is_interacting
        if first_is_interacting:
            feature_sizes = feature_sizes[1:]

        self.feature_sizes = feature_sizes

        self.summer = indexers.MolSummer()
        self.n_terms = len(feature_sizes)
        biases = (first_is_interacting, *(True for _ in range(self.n_terms - 1)))

        self.layers = torch.nn.ModuleList(
            torch.nn.Linear(nf, n_target, bias=bias) for nf, bias in zip(feature_sizes, biases)
        )

    def forward(self, all_features, mol_index, n_molecules):
        """
        Pytorch Enforced Forward function

        :param: all_features a list of feature tensors:
        :param: mol_index the molecular index for atoms in the batch
        :param: total number of molecules in the batch
        :return: Total Energy
        """
        if self.first_is_interacting:
            all_features = all_features[1:]

        partial_energies = [lay(x) for x, lay in zip(all_features, self.layers)]
        partial_terms = [self.summer(x, mol_index, n_molecules) for x in partial_energies]
        partial_sums = [partial_terms[0]]
        z = partial_terms[0]
        for x in partial_terms[1:]:
            z = x + z
            partial_sums.append(z)

        total_atomen = sum(partial_energies)
        total_energies = self.summer(total_atomen, mol_index, n_molecules)

        if self.n_terms > 1:
            partial_esq = [torch.square(x) for x in partial_energies]
            partial_atom_hier = [x / (x + y) for x, y in zip(partial_esq[1:], partial_esq[:-1])]
            mol_hier = [self.summer(x, mol_index, n_molecules)/self.summer(x+y, mol_index, n_molecules)
                        for x,y in zip(partial_esq[1:], partial_esq[:-1])]
            mol_hier = sum(mol_hier)
            partial_batch_hier = [x.sum() / (x.sum() + y.sum()) for x, y in zip(partial_esq[1:], partial_esq[:-1])]
            batch_hier = sum(partial_batch_hier)
            total_atom_hier = sum(partial_atom_hier)
            total_hier = self.summer(total_atom_hier, mol_index, n_molecules)

        else:
            total_hier = torch.zeros_like(total_energies)
            mol_hier = torch.zeros_like(total_energies)
            total_atom_hier = torch.zeros_like(total_atomen)
            batch_hier = torch.zeros(1,dtype=total_energies.dtype,device=total_energies.device)

        return total_energies, total_atomen, partial_sums, total_hier, total_atom_hier, mol_hier, batch_hier


class HCharge(torch.nn.Module):
    """
    Predict an atom-level scalar such as charge from local features.
    """

    def __init__(self, feature_sizes, first_is_interacting=False, n_target=1):
        super().__init__()
        self.feature_sizes = feature_sizes

        self.n_terms = len(feature_sizes)
        self.n_target = n_target
        biases = (first_is_interacting, *(True for _ in range(self.n_terms - 1)))
        self.layers = torch.nn.ModuleList(
            torch.nn.Linear(nf, n_target, bias=bias) for nf, bias in zip(feature_sizes, biases)
        )

    def forward(self, all_features):
        """
        :param all_features a list of feature tensors:
        :return: charges, predicted charges summed at each layer, and charge hierarchcality
        """
        partial_charges = [lay(x) for x, lay in zip(all_features, self.layers)]
        partial_sums = [partial_charges[0]]
        z = partial_charges[0]
        for x in partial_charges[1:]:
            z = x + z
            partial_sums.append(z)

        total_charges = sum(partial_charges)

        if self.n_terms > 1:
            partial_esq = [x ** 2 for x in partial_charges]
            partial_atom_hier = [x / (x + y) for x, y in zip(partial_esq[1:], partial_esq[:-1])]
            charge_hier = sum(x for x in partial_atom_hier)
        else:
            charge_hier = None

        return total_charges, partial_sums, charge_hier


class LocalChargeEnergy(torch.nn.Module):
    def __init__(self, feature_sizes, first_is_interacting=False):
        super().__init__()
        self.feature_sizes = feature_sizes

        self.n_terms = len(feature_sizes)
        bias_state = (first_is_interacting, *(True for _ in range(self.n_terms - 1)))
        self.layers_lin = torch.nn.ModuleList(
            torch.nn.Linear(nf, 1, bias=bias) for nf, bias in zip(feature_sizes, bias_state)
        )
        self.layers_quad = torch.nn.ModuleList(
            torch.nn.Linear(nf, 1, bias=bias) for nf, bias in zip(feature_sizes, bias_state)
        )
        self.summer = indexers.MolSummer()

    def forward(self, charges, all_features, mol_index, n_molecules):

        partial_lin_terms = [lay(x) for x, lay in zip(all_features, self.layers_lin)]
        partial_quad_terms = [lay(x) for x, lay in zip(all_features, self.layers_lin)]

        total_lin = sum(partial_lin_terms)
        total_quad = sum(partial_quad_terms)

        atom_charge_energy = (total_quad * charges) ** 2 + total_lin
        molecule_charge_energy = self.summer(atom_charge_energy, mol_index, n_molecules)

        return molecule_charge_energy, atom_charge_energy


class HBondSymmetric(torch.nn.Module):
    def __init__(
        self,
        feature_sizes,
        n_dist,
        dist_soft_min,
        dist_soft_max,
        dist_hard_max,
        positive=False,
        symmetric=False,
        antisymmetric=False,
        sensitivity_type=hiplayers.InverseSensitivityModule,
        n_target=1,
        all_pairs=True,
    ):
        super().__init__()

        if symmetric and antisymmetric:
            raise ValueError("Bond-like prediction cannot be both symmetric and antisymmetric!")

        if antisymmetric and positive:
            raise ValueError("Bond-like prediction cannot be antisymmetric and positive!")

        self.sensitivity = sensitivity_type(n_dist, dist_soft_min, dist_soft_max, dist_hard_max)
        self.feature_sizes = feature_sizes
        self.n_terms = len(feature_sizes)

        self.symmetric = symmetric
        self.antisymmetric = antisymmetric
        self.positive = positive
        self.n_target = n_target
        self.n_dist = n_dist
        if positive:
            self.biases = torch.nn.ParameterList(torch.nn.Parameter(torch.zeros(n_target)) for _ in feature_sizes)
        else:
            self.biases = None

        self.weights = torch.nn.ParameterList(
            torch.nn.Parameter(torch.zeros(n_dist, n_target, nf, nf)) for nf in feature_sizes
        )
        for p in self.weights:
            torch.nn.init.xavier_normal_(p.data)

    def forward(self, all_features, pair_first, pair_second, pair_dist):

        weights = self.weights
        if self.symmetric:
            weights = [w + w.transpose(2, 3) for w in weights]
        if self.antisymmetric:
            weights = [w - w.transpose(2, 3) for w in weights]

        sense_vals = self.sensitivity(pair_dist)

        n_d, n_t, n_f, _ = self.weights[0].shape
        
        # NOTE: Old code left here for posterity. At the current moment (Aug 23), this code is far slower.
        # These are the contributions for each bond at a given sensitivity distance
        # bilinear takes shape (pair,feature1),(pair,feature2),(ndist*n_target,feature1,feature2)
        # and sums to pair,(ndist*n_target), which is reshaped.
        #partial_bond_dists = [
        #    torch.nn.functional.bilinear(
        #        f[pair_first],
        #        f[pair_second],
        #        w.reshape(self.n_dist * self.n_target, f.shape[-1], f.shape[-1]),
        #        bias=None,
        #    ).reshape(-1, self.n_dist, self.n_target)
        #    for f, w in zip(all_features, weights)
        #]
        # These are the contributions for each bond
        # multiply pair,ndist by pair,ndist,n_targets
        #partial_bonds = [(pbd * sense_vals.unsqueeze(2)).sum(dim=1) for pbd in partial_bond_dists]
        
        # NOTE: This code is faster now that pytorch as opt_einsum features built in.
        # Einsum implementation of combined partial_bond_dists and partial_bond operations..
        partial_bonds = [
            torch.einsum("bf,bg,bs,stfg->bt",f[pair_first],f[pair_second],sense_vals,w)
            for f,w in zip(all_features,weights)
        ]
        

        if self.positive:
            partial_bonds = [pb + b for pb, b in zip(partial_bonds, self.biases)]

        total_bonds = sum(partial_bonds)
        if self.positive:
            total_bonds = torch.nn.functional.softplus(total_bonds)

        if self.n_terms > 1:
            partial_bsq = [b ** 2 for b in partial_bonds]
            partial_hier = [x / (x + y) for x, y in zip(partial_bsq[1:], partial_bsq[:-1])]
            bond_hier = sum(partial_hier)
        else:
            bond_hier = torch.zeros_like(total_bonds)
        
        return total_bonds, bond_hier


class AtomizationEnergy(torch.nn.Module):
    def __init__(self, feature_sizes, decay_factor=0.01):
        super().__init__()

        # TODO: Add n_targets and alternative hierarchicality definitions
        # to this layer and node.
        feature_sizes = feature_sizes[1:]
        self.feature_sizes = feature_sizes


        self.summer = indexers.MolSummer()
        self.n_terms = len(feature_sizes)
        self.layers = torch.nn.ModuleList(torch.nn.Linear(nf, 1, bias=False)
                                          for nf in feature_sizes)

        for layer in self.layers:
            layer.weight.data *= decay_factor
            decay_factor *= decay_factor

        # This is hard-coded for this module, for compatibility with HEnergy
        # see how this operates in the `forward` method.
        self.first_is_interacting = True

    def forward(self, all_features, vacuum_features, encoding, mol_index, n_molecules):

        # Don't need the non-interacting features,
        # they contribute zero total for this layer, by definition the any component would cancel.
        # this is why we hardcode first_is_interacting in the `__init__` method.
        vacuum_features = vacuum_features[1:]
        all_features = all_features[1:]

        # get vacuum features for each atom and subtract them
        vacuum_energies = sum([lay(x) for x, lay in zip(vacuum_features, self.layers)])
        encoding = encoding.to(all_features[0].dtype)
        vacuum_features_eachatom = [encoding @ f for f in vacuum_features]
        renorm_features = [x - v for x, v in zip(all_features, vacuum_features_eachatom)]

        # Compute!
        en_terms = [lay(r) for r, lay in zip(renorm_features, self.layers)]
        total_atomen = sum(en_terms)
        total_energies = self.summer(total_atomen, mol_index, n_molecules)

        e0 = encoding @ vacuum_energies
        if self.n_terms > 1:
            partial_esq = [x ** 2 for x in en_terms]
            partial_atom_hier = [x / (x + y) for x, y in zip(partial_esq[1:], partial_esq[:-1])]
            total_atom_hier = sum(x for x in partial_atom_hier)
            total_hier = self.summer(total_atom_hier, mol_index, n_molecules)
        else:
            total_hier = torch.zeros_like(total_energies)

        return total_energies, en_terms, total_hier
