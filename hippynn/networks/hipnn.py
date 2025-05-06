"""
Implementation of HIPNN.
"""
import warnings
import torch

from typing import Union, List

from ..layers.hiplayers import (
    GaussianSensitivityModule,
    InverseSensitivityModule,
    InteractLayer,
    InteractLayerVec,
    InteractLayerQuad,
)
from ..layers.transform import ResNetWrapper


# computes E0 for the energy layer.
def compute_hipnn_e0(encoder, Z_Data, en_data, peratom=False, fit_dtype=torch.float64):
    """

    :param encoder: encoder of species to features (one-hot representation, probably)
    :param Z_Data: species data
    :param en_data: energy data
    :param peratom: whether energy is per-atom or total
    :return: energy per species as shape (n_features_encoded, 1)
    """

    original_dtype = en_data.dtype
    en_data = en_data.to(fit_dtype)
    x, nonblank = encoder(Z_Data)

    sums = x.sum(dim=1).to(en_data.dtype)
    if peratom:
        atom_counts = nonblank.sum(dim=1, keepdims=True).to(en_data.dtype)
        Z_matrix = sums / atom_counts
    else:
        Z_matrix = sums

    # Want to solve Z e = E  where Z is a composition matrix
    # Z^T Z e  = Z^T E is lower dimensional
    # e = (Z^T Z)^-1 Z^T E
    # shape n_species, n_species
    ZTZ = Z_matrix.T @ Z_matrix
    ZTZ_inv = torch.pinverse(ZTZ)
    # shape n_species, n_examples
    ZTZ_inv_Z = ZTZ_inv @ Z_matrix.T

    # shape (n_species, n_targets) (n_targets may be omitted)
    e_per_species = ZTZ_inv_Z @ en_data

    if e_per_species.ndim != 1:
        # if n_targets is included
        e_per_species = e_per_species.T

    e_per_species = e_per_species.to(original_dtype)
    return e_per_species


class Hipnn(torch.nn.Module):
    """
    Hipnn Main Module
    """

    _interaction_class = InteractLayer
    _interaction_kwargs = ()  # No extra arguments to regular interaction layer

    def __init__(
        self,
        n_features,
        n_sensitivities,
        dist_soft_min,
        dist_soft_max,
        dist_hard_max,
        n_atom_layers,
        n_interaction_layers=None,
        possible_species=None,
        n_input_features=None,
        sensitivity_type="inverse",
        resnet=True,
        activation=torch.nn.Softplus,
        **kwargs,
    ):
        """

        :param n_features: width of each layer
        :param n_sensitivities: number of sensitivity functions
        :param dist_soft_min: midpoint of first sensitivity function
        :param dist_soft_max: midpoint of last sensitivity function
        :param dist_hard_max: cutoff for cutoff function.
        :param n_interaction_layers: number of interaction blocks
        :param n_atom_layers: number of atom layers per interaction block.
        :param possible_species: list of species values in database, including 0 for none.
            For example, HCNO would be [0 1 6 7 8]
        :param n_input_features: number of input features to the model
        :param sensitivity_type: str or callable, type of sensitivity, default of
           'inverse' is what is in hip-nn original paper.
        :param resnet: bool or int, if int, size of internal resnet width
        :param activation: activation function or subclass of nn.module.

        Note: only one of possible_species or n_input_features is needed. If both are supplied,
        they must be consistent with each other.
        """

        super().__init__()

        if possible_species is not None:
            possible_species = torch.as_tensor(possible_species)
            if n_input_features is not None:
                if len(possible_species) - 1 != n_input_features:
                    raise ValueError("Species and input features are not consistent with each other.")
            else:
                n_input_features = len(possible_species) - 1
        else:
            if n_input_features is None:
                raise ValueError("Either n_input_features or possible_species must be set!")

        self.species_set = possible_species
        self.n_layers_per_block = n_atom_layers
        self.nf = n_features
        self.nf_in = n_input_features

        if isinstance(self.nf, int):
            if n_interaction_layers is None:
                raise ValueError("Must provide 'n_interaction_layers' if n_features is a single integer.")
            self.feature_sizes = (self.nf_in, *(self.nf for _ in range(n_interaction_layers)))
        else:
            if n_interaction_layers is not None:
                if len(self.nf) != n_interaction_layers:
                    raise ValueError("Number of interaction layers conflicts with feature sizes")
            else:
                n_interaction_layers = len(self.nf)

            self.feature_sizes = (self.nf_in, *self.nf)

        self.ni = n_interaction_layers
        self.resnet: Union[int, List[int]] = resnet

        if self.resnet not in (True, False):
            # resnet argument specifies a different size for the internal layers
            if isinstance(self.resnet, int):
                self.nf_middle = [self.resnet for _ in range(self.ni)]
            else:
                if len(self.resnet) != self.ni:
                    raise ValueError("Number of interaction layers conflicts with resnet size")
                self.nf_middle = self.resnet

        else:
            self.nf_middle = self.feature_sizes[1:]

        self.n_sensitivities = n_sensitivities
        self.dist_soft_max = dist_soft_max
        self.dist_soft_min = dist_soft_min
        self.dist_hard_max = dist_hard_max

        if not isinstance(activation, torch.nn.Module):
            activation = activation()
        self.activation = activation

        # Containers for main layers
        self.blocks = torch.nn.ModuleList()
        self.atomlayers = torch.nn.ModuleList()
        self.interactlayers = torch.nn.ModuleList()

        # determine sensitivity layer
        if sensitivity_type == "inverse":
            sensitivity_type = InverseSensitivityModule
        elif sensitivity_type == "linear":
            sensitivity_type = GaussianSensitivityModule
        elif callable(sensitivity_type):
            pass
        else:
            raise TypeError("Invalid sensitivity type:", sensitivity_type)

        interaction_kwargs = {k: kwargs.pop(k) for k in self._interaction_kwargs if k in kwargs}

        if len(kwargs) > 0:
            warnings.warn(f"Network initialized with unused {kwargs=}")

        # Finally, build the network!
        for in_size, out_size, middle_size in zip(self.feature_sizes[:-1], self.feature_sizes[1:], self.nf_middle):
            this_block = torch.nn.ModuleList()

            # Add interaction layer
            lay = self._interaction_class(
                in_size,
                middle_size,
                n_sensitivities,
                dist_soft_min,
                dist_soft_max,
                dist_hard_max,
                sensitivity_type,
                **interaction_kwargs,
            )
            if self.resnet:
                lay = ResNetWrapper(lay, in_size, middle_size, out_size, self.activation)
            this_block.append(lay)

            for j in range(n_atom_layers):
                # Add subsequent atom layers
                lay = torch.nn.Linear(out_size, middle_size)
                torch.nn.init.xavier_normal_(lay.weight.data)
                if self.resnet:
                    lay = ResNetWrapper(lay, out_size, middle_size, out_size, self.activation)
                this_block.append(lay)
            self.blocks.append(this_block)

    @property
    def interaction_layers(self):
        layers = [block[0] for block in self.blocks]
        if self.resnet:
            layers = [lay.base_layer for lay in layers]
        return layers

    @property
    def sensitivity_layers(self):
        return [il.sensitivity for il in self.interaction_layers]

    def regularization_params(self):
        params = []
        for block in self.blocks:
            for lay in block:
                if hasattr(lay, "regularization_params"):
                    params.extend(lay.regularization_params())
                else:
                    params.append(lay.weight)
        params = list(set(params))
        return params

    def forward(self, features, pair_first, pair_second, pair_dist):
        features = features.to(pair_dist.dtype)  # Convert one-hot features to floating point features.

        if pair_dist.ndim == 2:
            pair_dist = pair_dist.squeeze(dim=1)

        output_features = [features]

        for block in self.blocks:
            int_layer = block[0]
            atom_layers = block[1:]

            features = int_layer(features, pair_first, pair_second, pair_dist)
            if not self.resnet:
                features = self.activation(features)
            for lay in atom_layers:
                features = lay(features)
                if not self.resnet:
                    features = self.activation(features)
            output_features.append(features)

        return output_features


class HipnnVec(Hipnn):
    """
    HIP-NN-TS with l=1
    """

    _interaction_class = InteractLayerVec
    _interaction_kwargs = ("cusp_reg",)

    def __init__(self, *args, cusp_reg=1e-6, **kwargs):
        # cusp regularization for tensor sensitivity l>0. Defaults to 1e-6.
        super().__init__(*args, cusp_reg=cusp_reg, **kwargs)

    def forward(self, features, pair_first, pair_second, pair_dist, pair_coord):
        features = features.to(pair_dist.dtype)  # Convert one-hot features to floating point features.

        if pair_dist.ndim == 2:
            pair_dist = pair_dist.squeeze(dim=1)

        if pair_coord.ndim == 3:
            pair_coord = pair_coord.squeeze(dim=2)

        output_features = [features]

        for block in self.blocks:
            int_layer = block[0]
            atom_layers = block[1:]

            features = int_layer(features, pair_first, pair_second, pair_dist, pair_coord)
            if not self.resnet:
                features = self.activation(features)
            for lay in atom_layers:
                features = lay(features)
                if not self.resnet:
                    features = self.activation(features)
            output_features.append(features)

        return output_features


class HipnnQuad(HipnnVec):
    """
    HIP-NN-TS with l=2
    """

    _interaction_class = InteractLayerQuad
