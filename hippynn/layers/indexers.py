"""
Layers for encoding, decoding, index states, besides pairs
"""

import warnings
import torch


class OneHotSpecies(torch.nn.Module):
    """
    Encodes species as one-hot map using `species_set`
    :param species_set iterable of species types. Typically this will be the Z-value of the elemeent.
    Note: 0 denotes a 'blank atom' used only for padding purposes

    :returns OneHotSpecies

    """

    def __init__(self, species_set):
        super().__init__()

        self.species_set = torch.as_tensor(species_set)
        self.n_species = self.species_set.shape[0]

        max_z = torch.max(self.species_set)
        zmap = torch.zeros((max_z + 1,), dtype=torch.long)
        zmap[self.species_set] = torch.arange(0, self.n_species, dtype=torch.long)

        self.species_map = torch.nn.Parameter(torch.as_tensor(zmap, dtype=torch.long), requires_grad=False)

    def forward(self, species):
        """
        :param species:
        :return: Initial one-hotted features, nonblank atoms
        """

        indices = self.species_map[species].to(species.device)
        onehot_species = torch.eye(self.n_species, dtype=torch.bool, device=species.device)[indices]
        nonblank = ~onehot_species[:, :, 0]
        initial_features = onehot_species[:, :, 1:]  # remove atoms that are 0 in the species map.

        return initial_features, nonblank


class PaddingIndexer(torch.nn.Module):
    """
    Hipnn's indexer

    Description:
        This indexer allows us to go from rectangular (mol,atom) representations
        To (flatatom) representations and a corresponding
        molecule index for those atoms.
        The 'real_index' allows us to take _values_ from a flattened rectangular
        representation (mol*atom)
        And select only the real ones.
        The 'inv_real_index', when indexed, converts a (mol*atom) index set into a (flatatom) index set

    """

    def forward(self, features, nonblank):
        """
        Pytorch Enforced Forward function

        :param features:
        :param nonblank:
        :return: real atoms, amd inverse atoms mappings
        """
        dev = features.device
        n_molecules, n_atoms_max = nonblank.shape
        n_fictitious_atoms = nonblank.shape[0] * nonblank.shape[1]
        # Just calculate the total number of atoms in the dataset

        flat_nonblank = nonblank.reshape(n_fictitious_atoms)
        # Flatten the nonblank n_mol x n_atoms nonblank matrix

        large_enough = torch.empty(nonblank.shape,device=dev,dtype=torch.int64)
        large_enough.resize_(0)
        real_atoms = torch.nonzero(flat_nonblank, as_tuple=False,out=large_enough)[:, 0]
        # print(real_atoms)
        # Grab the indexes of each real atom, give it an atom get an index back

        n_real_atoms = real_atoms.shape[0]
        # Count how many real atoms there are

        inv_real_atoms = torch.zeros((n_fictitious_atoms,), dtype=torch.long, device=dev)
        # Create a vector of 0's

        inv_real_atoms[real_atoms] = torch.arange(n_real_atoms, dtype=torch.long, device=dev)
        # Create the inverse real atom "function" give it an index get an atom back

        # Flatten incoming features to atom representation
        indexed_features = features.reshape(n_molecules * n_atoms_max, -1)[real_atoms]
        if indexed_features.ndimension() == 1:
            indexed_features = indexed_features.unsqueeze(1)

        # Get molecule index for atoms
        mol_index_shaped = (
            torch.arange(n_molecules, dtype=torch.long, device=dev).unsqueeze(1).expand(-1, n_atoms_max)
        )
        atom_index_shaped = (
            torch.arange(n_atoms_max, dtype=torch.long, device=dev).unsqueeze(0).expand(n_molecules, -1)
        )
        atom_index = atom_index_shaped.reshape(n_fictitious_atoms)[real_atoms]
        mol_index = mol_index_shaped.reshape(n_fictitious_atoms)[real_atoms]

        return indexed_features, real_atoms, inv_real_atoms, mol_index, atom_index, n_molecules, n_atoms_max


class AtomReIndexer(torch.nn.Module):
    def forward(self, molatom_thing, real_atoms):
        m, a, *rest = molatom_thing.shape
        out = molatom_thing.reshape(m * a, *rest)[real_atoms]
        if len(rest) == 0:
            out = out.unsqueeze(1)
        return out


class MolSummer(torch.nn.Module):
    """
    Molecule Summer

    Description:
        This sums (flatatom) things into (mol) things.
        It actually works similarly to the interaction layer
    """

    def forward(self, features, mol_index, n_molecules):
        featshape = (1,) if features.ndimension() == 1 else features.shape[1:]
        out_shape = (n_molecules, *featshape)
        result = torch.zeros(*out_shape, device=features.device, dtype=features.dtype)
        result.index_add_(0, mol_index, features)

        return result


class SysMaxOfAtoms(torch.nn.Module):
    """
    Take maximum over atom dimension.
    """
    def forward(self, features, mol_index, n_molecules):
        # Add feature dimension if not found
        if features.ndim == 1:
            featshape = (1,)
            features = features.unsqueeze(1)
        else:
            featshape = features.shape[1:]
        # Allocate result
        out_shape = (n_molecules, *featshape)
        result = torch.zeros(*out_shape, device=features.device, dtype=features.dtype)

        # Prepare index shape for scatter operation
        mi_expand = mol_index.reshape(-1, *(1,) * len(featshape))
        mi_expand = mi_expand.expand((-1, *featshape))

        # Perform calculation
        result.scatter_reduce_(0, mi_expand, features, reduce='amax', include_self=False)
        return result

class AtomDeIndexer(torch.nn.Module):
    def forward(self, features, mol_index, atom_index, n_molecules, n_atoms_max):
        featshape = 1 if features.ndimension() == 1 else features.shape[1:]
        out_shape = (n_molecules, n_atoms_max, *featshape)
        result = torch.zeros(*out_shape, device=features.device, dtype=features.dtype)
        result[mol_index, atom_index] = features
        return result


class CellScaleInducer(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pbc = False

    def forward(self, coordinates, cell):
        strain = torch.eye(
            coordinates.shape[2], dtype=coordinates.dtype, device=coordinates.device, requires_grad=True
        ).tile(coordinates.shape[0],1,1)
        strained_coordinates = torch.bmm(coordinates, strain)
        strained_cell = torch.bmm(cell, strain)
        return strained_coordinates, strained_cell, strain


class QuadPack(torch.nn.Module):
    """
    Converts quadrupoles flattened form to packed triangular, assumes symmmetric.
    Packed form: (molecule)   XX, YY, ZZ, XY, XZ, YZ
    Index                      0,  4,  8,  1,  2,  5
    Unpacked form: (molecule) XX, XY, XZ, YX, YY, YZ , ZX, ZY, ZZ
    Index                      00 01  02, 10, 11, 12,  20, 21, 22
    """

    def __init__(self):
        super().__init__()
        ind1 = [0, 1, 2, 0, 0, 1]
        ind2 = [0, 1, 2, 1, 2, 2]
        self.register_buffer("ind_1", torch.LongTensor(ind1))
        self.register_buffer("ind_2", torch.LongTensor(ind2))

    def forward(self, quadrupoles):
        return quadrupoles[:, self.ind1, self.ind2]


class QuadUnpack(torch.nn.Module):
    """
    Converts quadrupoles from packed triangular form to flattened molecule form.
    Packed form: (molecule)   XX, YY, ZZ, XY, XZ, YZ
    Index                      0,  1,  2,  3,  4,  5
    Unpacked form: (molecule) XX, XY, XZ, YX, YY, YZ , ZX, ZY, ZZ
    Index                      0,  3,  4,  3,  1,  5,  4,  5,  2
    """

    def __init__(self):
        super().__init__()
        indices = [0, 3, 4, 3, 1, 5, 4, 5, 2]
        self.register_buffer("index_permutation", torch.LongTensor(indices))

    def forward(self, packed_quadrupoles):
        return packed_quadrupoles[:, self.index_permutation]


class FilterBondsOneway(torch.nn.Module):
    def forward(self, bonds, pair_first, pair_second):
        # in seqm, only bonds with index first < second is used
        cond = pair_first < pair_second
        return bonds[cond]

class FuzzyHistogram(torch.nn.Module):
    """ 
    Transforms a scalar feature into a vectorized feature via 
    the fuzzy/soft histogram method.

    :param length: length of vectorized feature

    :returns FuzzyHistogram
    """

    def __init__(self, length, vmin, vmax):
        super().__init__()

        err_msg = "The value of 'length' must be a positive integer."
        if not isinstance(length, int):
            raise ValueError(err_msg)
        if length <= 0:
            raise ValueError(err_msg)

        if not (isinstance(vmin, (int,float)) and isinstance(vmax, (int,float))):
            raise ValueError("The values of 'vmin' and 'vmax' must be floating point numbers.")
        if vmin >= vmax:
            raise ValueError("The value of 'vmin' must be less than the value of 'vmax.'")

        self.bins = torch.nn.Parameter(torch.linspace(vmin, vmax, length), requires_grad=False)
        self.sigma = (vmax - vmin) / length

        self.vmin = vmin
        self.vmax = vmax

    def forward(self, values):
        # Warn user if provided values lie outside the range of the histogram bins
        values_out_of_range = (values < self.vmin) + (values > self.vmax)

        if values_out_of_range.sum() > 0:
            perc_out_of_range = values_out_of_range.float().mean()
            warnings.warn(
                "Values out of range for FuzzyHistogrammer\n"
                f"Number of values out of range: {values_out_of_range.sum()}\n"
                f"Percentage of values out of range: {perc_out_of_range * 100:.2f}%\n"
                f"Set range for FuzzyHistogrammer: ({self.vmin:.2f}, {self.vmax:.2f})\n"
                f"Range of values: ({values.min().item():.2f}, {values.max().item():.2f})"
            )

        if values.shape[-1] != 1:
            values = values[...,None]
        x = values - self.bins
        histo = torch.exp(-((x / self.sigma) ** 2) / 4)
        return torch.flatten(histo, end_dim=1)
    
class SpeciesIndexer(torch.nn.Module):
    def forward(self, values, onehot_encoding):
        n_species = onehot_encoding.shape[1]
        values_by_species = []
        for i in range(n_species):
            species_mask = onehot_encoding[:,i]
            species_values = values[species_mask]
            values_by_species.append(species_values)
        return values_by_species
