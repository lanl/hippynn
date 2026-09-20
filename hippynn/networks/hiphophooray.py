import warnings
import torch

from ..custom_kernels import envsum
from .hipnn import Hipnn
from ..layers.hiplayers import HOPInteractionLayer, TensorExtractor
from ..layers.hiplayers import sensitivity as sensitivity_modules



from typing import NamedTuple
import torch
from jaxtyping import Float, Int64

from .tensors import cartesian_irreducible_mapping, pmaps, TensorExtractor

_MAPPINGS = [cartesian_irreducible_mapping(i).to(torch.get_default_dtype()) for i in range(1, 5)]
_INVARIANT_BASIS = [
    "i, i -> ", 
    "ij, ij -> ", 
    "ij, ik, jk -> ",
    "ijk, ijk -> ",
    "ijk, ijl, kmn, lmn -> ", 
    "i, ij, j -> ", 
    "i, ij, jk,k -> ", 
    "i, j, k, ijk -> ",
    "i, ijk, jkl, l ->", 
    "ij, ikl, klj ->", 
    "ij,ik, kml, lmj ->",
    "ij,kl,ijm,klm ->"]

_COVARIANT_BASIS = ["i -> i", 
                   "i, ij -> j",
                   #'ijk, jlm, ilm -> k',
                   ]

from typing import Annotated, NamedTuple

import torch
from jaxtyping import Float, Int64


class FeatureState(NamedTuple):
    pair_first: Annotated[
        Int64[torch.Tensor, "pairs"],
        "Index of the first atom in each pair",
    ]

    pair_second: Annotated[
        Int64[torch.Tensor, "pairs"],
        "Index of the second atom in each pair",
    ]

    pair_dist: Annotated[
        Float[torch.Tensor, "pairs"],
        "Euclidean distance between the atoms in each pair",
    ]

    pair_coord: Annotated[
        Float[torch.Tensor, "pairs 3"],
        "Displacement vector: position[pair_second] - position[pair_first]",
    ]

    rhat_compact_tensors: Annotated[
        Float[torch.Tensor, "pairs stf_components"],
        "STF tensor for each pair, flattened to its independent components",
    ]

    in_features_type0: Annotated[
        Float[torch.Tensor, "atoms channels"],
        "Optional: Type-0 per-atom features; invariant under rotations",
    ] | None = None

    in_features_type1: Annotated[
        Float[torch.Tensor, "atoms channels 3"],
        "Optional: Type-1 per-atom vector features; rotate with the input",
    ] | None = None


class HipHopHoorayNNModule(torch.nn.Module):
    _interaction_class = HOPInteractionLayer
    _interaction_kwargs = ("l_max", "n_max", "group_norm", "group_norm_eps")

    def __init__(self,
                 *args,
                 l_max=3,
                 n_max=4,
                 group_norm=True,
                 group_norm_eps=1e-5,
                 possible_species=None,
                 n_features=None,
                 sensitivity_module="InverseSensitivityModule",
                 n_sensitivities=20,
                 dist_soft_min=1.6, 
                 dist_soft_max=10.0,
                 dist_hard_max=12.5, 
                 n_interaction_layers=2, 
                 n_atom_layers=3):

        super().__init__()
        self.l_max = l_max
        self.n_max = n_max
        self.angular_tensor_extractor = TensorExtractor(l_max=l_max)

        if hasattr(sensitivity_modules, sensitivity_module):
            sensitivity_module = getattr(sensitivity_modules, sensitivity_module)
        else:
            raise ValueError(f"Invalid sensitivity module: {sensitivity_module!r}")
        self.sensitivity = sensitivity_module
        # Lifting layer 
        self.lifting_layer = CovariantLayer(
            inv_basis=_INVARIANT_BASIS,
            equi_basis=_COVARIANT_BASIS,
            l_max=l_max,
            nf_in=(len(possible_species)-1, len(possible_species)-1),
            nf_out=(n_features, n_features),
            sensitivity_module=self.sensitivity,
            n_sensitivities=n_sensitivities, 
            dist_soft_min=dist_soft_min,
            dist_soft_max=dist_soft_max,
            dist_hard_max=dist_hard_max, 
        )

            
    def extra_repr(self):
        return "Equivariant Prototype" 

    def forward(self, features, pair_first, pair_second, pair_dist, pair_coord):
        if pair_dist.ndim == 2:
            pair_dist = pair_dist.squeeze(dim=1)

        if pair_coord.ndim == 3:
            pair_coord = pair_coord.squeeze(dim=2)

        # Put warning if there is onehot and not learnable features.
        if features.dtype == torch.int64:
            warnings.warn(
                "Features are one-hot encoded. Consider using learnable features for better performance." \
                " Previous HippyNN models use linear layer in the beginning! Hurray mixed them at the end."
            )
        features = features.to(pair_dist.dtype)  # Convert one-hot features to floating point features.
        rhats = pair_coord / pair_dist.unsqueeze(1)
        angular_tensors = self.angular_tensor_extractor(rhats)

        # TODO: Slicing seems not necessary, but let's keep it for now.
        angular_tensors = angular_tensors[: self.l_max + 1]
        angular_tensors = torch.cat(angular_tensors, dim=-1)
        
        state = FeatureState(pair_coord=pair_coord, 
                             pair_dist=pair_dist,
                             pair_first=pair_first,
                             pair_second=pair_second,
                             rhat_compact_tensors=angular_tensors,
                             in_features_type0=features, 
                             in_features_type1=torch.randn([*features.shape, 3]))

        state = self.lifting_layer(state)


        #for block in self.blocks:
        #    int_layer = block[0]
        #    atom_layers = block[1:]

        #    features = int_layer(features, pair_first, pair_second, pair_dist, tensor_rhats)
        #    if not self.resnet:
        #        features = self.activation(features)
        #    for lay in atom_layers:
        #        features = lay(features)
        #        if not self.resnet:
        #            features = self.activation(features)
        #    output_features.append(features)

        #return output_features
        
        raise NotImplementedError(
            "Invariant/covariant feature generation has not been implemented yet."
        )

# Calculate contraction 
def contract_tensors(tensor_dict, einsum_str):
    """
    Contract a list of tensors according to the provided einsum string.

    Args:
        tensor_dict (dict): A dictionary mapping tensor names to torch.Tensor objects.
        einsum_str (str): The einsum string specifying the contraction.

    Returns:
        torch.Tensor: The result of the contraction.
    """
    inputs, output = einsum_str.replace(" ", "").split("->")
    terms = inputs.split(",")

    tensor_idx = [len(term) for term in terms]
    tensors = [tensor_dict[i] for i in tensor_idx]

    batched_einsum = ",".join("..." + term for term in terms)
    batched_einsum += "->..." + output

    return torch.einsum(batched_einsum, *tensors)

def _sum_compact_tensors(
    compact_tensors: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    """Aggregate per-point compact tensors over dimension 0."""
    return tuple(tensor.sum(dim=0) for tensor in compact_tensors)

def _compact_to_STF(compact_tensors):
    assert 1 <= len(compact_tensors) <= len(_MAPPINGS) + 1

    # Rank 0 requires no mapping.
    full_tensors = [compact_tensors[0]]

    for compact, mapping in zip(
        compact_tensors[1:],
        _MAPPINGS,
    ):
        assert compact.shape[-1] == mapping.shape[-1]

        # Contract the independent-component index.
        full = torch.tensordot(
            compact,
            mapping,
            dims=([-1], [-1]),
        )

        full_tensors.append(full)
    return tuple(full_tensors)

def _calc_basis(tensor_dict, inv_einsums, equ_einsums):
    scalar_outputs = []
    vector_outputs = []

    for einsum in inv_einsums: 
        result = contract_tensors(tensor_dict, einsum)
        scalar_outputs.append(result)

    for einsum in equ_einsums:
        result = contract_tensors(tensor_dict, einsum)
        vector_outputs.append(result)

    # Channel-last scalar layout: (..., channels)
    type0 = (
        torch.stack(scalar_outputs, dim=-1)
        if scalar_outputs
        else None
    )

    # Channel-before-Cartesian layout: (..., channels, 3)
    type1 = (
        torch.stack(vector_outputs, dim=-2)
        if vector_outputs
        else None
    )

    return type0, type1

def _vector_tensor_extractor(rhat_tensors, type1, radius=None):
        s, v, Q, T = rhat_tensors
        batch_size = type1.shape[0]

        #assert type1.shape == (batch_size, 3), f"Expected vectors shape (batch_size, 3), got {type1.shape}"
        #assert s.shape == (batch_size, 1), f"Expected s shape (batch_size, 1), got {s.shape}"
        #assert v.shape == (batch_size, 3), f"Expected v shape (batch_size, 3), got {v.shape}"
        #assert Q.shape == (batch_size, 3, 3), f"Expected Q shape (batch_size, 3, 3), got {Q.shape}"
        #assert T.shape == (batch_size, 3, 3, 3), f"Expected T shape (batch_size, 3, 3, 3), got {T.shape}"

        F1 = type1 * s

        F2 = torch.einsum(
            "bi,bj->bij",
            type1,
            v,
        )

        F3 = torch.einsum(
            "bi,bjk->bijk",
            type1,
            Q,
        )

        F4 = torch.einsum(
            "bi,bjkl->bijkl",
            type1,
            T,
        )

        return F1, F2, F3, F4

def _split_tensor_feature_by_orders(tensor_features, l_max, dim=-1): 
    if l_max == 0:
        return tensor_features
    elif l_max == 1:
        return tensor_features.split([1, 3], dim=dim)
    elif l_max == 2:
        return tensor_features.split([1, 3, 5], dim=dim)
    elif l_max == 3:
        return tensor_features.split([1, 3, 5, 7], dim=dim)
    else:
        raise ValueError(f"Invalid l_max:{l_max}")

class LiftingLayer(torch.nn.Module):
    def __init__(self, inv_basis, equi_basis, l_max) -> None:
        super().__init__()
        self.inv_basis = inv_basis
        self.equi_basis = equi_basis
        # This can be cached for the whole process and run with forward
        # we are missing radius functions and projection from inner state of node? 
        
    def forward(self, state: FeatureState) -> FeatureState:
        #assert invariants is None, f"Not implemented yet!"
        #assert radius is None, f"Not implemented yet!"
        rhat_compact_tensors = state.rhat_compact_tensors
        compact_tensors = _sum_compact_tensors(rhat_compact_tensors)
        tensors = _compact_to_STF(compact_tensors)

        # Calculate the basis functions
        type0, type1 = _calc_basis(
            tensors,
            self.inv_basis,
            self.equi_basis,
        )

        return state._replace(rhat_compact_tensors=rhat_compact_tensors, 
                              in_features_type0=type0,
                              type1=type1)

class CovariantLayer(torch.nn.Module):

    def __init__(self, 
                 inv_basis,
                 equi_basis,
                 l_max, 
                 sensitivity_module: sensitivity_modules.SensitivityModule,
                 n_sensitivities: int,
                 nf_in:  tuple[int, int],
                 nf_out: tuple[int, int], 
                 dist_soft_min: float,
                 dist_soft_max: float,
                 dist_hard_max: float):
        super().__init__()
        assert len(nf_in) == 2, f"Expected nf_in to be a tuple of length 2, got {len(nf_in)}"
        assert len(nf_out) == 2, f"Expected nf_out to be a tuple of length 2, got {len(nf_out)}"
        
        if l_max != 3: 
            raise NotImplementedError("Other than l_max=3, not implemented")  

        self.type0_in, self.type1_in = nf_in # Zero input channels for type0 or type1 is not allowed
        self.type0_out, self.type1_out = nf_out # Zero output channels for type0 or type1 is not allowed
        self.l_max = l_max
        self.nf_in = nf_in
        self.n_dist = n_sensitivities
        self.inv_basis = inv_basis
        self.equi_basis = equi_basis
        self.dist_soft_min = dist_soft_min
        self.dist_soft_max = dist_soft_max
        self.dist_hard_max = dist_hard_max
        self.lift_layer = LiftingLayer(inv_basis=inv_basis, equi_basis=equi_basis, l_max=l_max)
        # we are missing radius functions and projection from inner state of node? 
        # Sensitivity module
        self.sensitivity = sensitivity_module(n_sensitivities,
                                              dist_soft_min,
                                              dist_soft_max,
                                              dist_hard_max)
        #
        # Interaction weights
        self.int_weights_type0 = torch.nn.Parameter(torch.Tensor(self.n_dist * self.type0_in, self.type0_out))
        self.int_weights_type1 = torch.nn.Parameter(
            torch.empty(self.n_dist * self.type1_in, self.type1_out)
)
        torch.nn.init.xavier_normal_(self.int_weights_type0.data)
        torch.nn.init.xavier_normal_(self.int_weights_type1.data)

        # Mixing weights 
        # TODO: Fix the numbers here depending on input number of invariants and output number of features.
        type0_mixing_weights = torch.zeros(self.type0_out, len(_INVARIANT_BASIS) * 2, self.type0_out)
        type1_mixing_weights = torch.zeros(self.type1_out, len(_COVARIANT_BASIS) * 2, self.type1_out, 1)
        self.type0_mixing_weights = torch.nn.Parameter(type0_mixing_weights)
        self.type1_mixing_weights = torch.nn.Parameter(type1_mixing_weights)
        torch.nn.init.xavier_normal_(self.type0_mixing_weights.data)
        torch.nn.init.xavier_normal_(self.type1_mixing_weights.data)

        # Self-term and bias 
        # TODO: What if output and no in or otherway arround 
        if self.type0_in > 0 and self.type0_out > 0:
            self.selfint_type0 = torch.nn.Linear(self.type0_in, self.type0_out)
        if self.type1_in > 0 and self.type1_out > 0:
            self.selfint_type1 = torch.nn.Linear(self.type1_in, self.type1_out)

    def forward(self, state: FeatureState) -> FeatureState:
        rhat_compact_tensors, pair_dist = state.rhat_compact_tensors, state.pair_dist
        type0, type1 = state.in_features_type0, state.in_features_type1

        # Inner layers
        
        # input features that're invariant == type0, f(R(g)x) = f(x) 
        # input features that're covariant == type1, f(R(g)x) = R(g)f(x)  
    
        # Get shapes
        n_atoms_real = type0.shape[0] # number of atoms in all systems
        # Number of pairs and independent tensor components
        n_pair, n_tensor_comp = rhat_compact_tensors.shape[0], rhat_compact_tensors.shape[-1]

        # Sensititity function, shared along the tensors for now
        sense_scalar = self.sensitivity(pair_dist)
        # [n_pairs, STF, n_sensitivity_functions]
        sensitivity = sense_scalar.unsqueeze(1) * rhat_compact_tensors.unsqueeze(2)
        # Now it's time to do the contraction over vector x tensors
        # Align both vector tensors and scaler for using env sum. 
        sense_flat = sensitivity.reshape(n_pair, n_tensor_comp * self.n_dist)

        if (type0 is not None) and (type1 is not None):
            features = torch.cat([type0, type1.flatten(start_dim=1)], dim=1)  # (A, C0 + 3*C1)

        # env_features - [N_atoms, N_sensitivities x n_tensor_comp, channels_type0 x channels_type1]
        env_features = envsum(sense_flat,
                              features,
                              state.pair_first,
                              state.pair_second
                              )  

        
        # Note: After env sum, it's good time to normalize
        # [A, S x STF, Ch], [A, S x STF, Ch, 3] 
        env_type0, env_type1 = env_features.split([self.type0_in, 3 * self.type1_in], dim=-1)

        # Type0 Tensors 
        env_type0 = env_type0.reshape(n_atoms_real * n_tensor_comp, self.n_dist * self.type0_in)
        tensor_features_type0 = torch.mm(env_type0, self.int_weights_type0)
        tensor_features_type0 = tensor_features_type0.reshape(n_atoms_real, n_tensor_comp, self.type0_out)

        # This can be custom kernel, it would be nice to 
        s0, V0, Q0, T0 = _split_tensor_feature_by_orders(l_max=self.l_max, tensor_features=tensor_features_type0, dim=-2)

        V0 = torch.tensordot(V0, 
                             _MAPPINGS[0],
                             dims=([-2], [-1]))
        Q0 = torch.tensordot(Q0, 
                             _MAPPINGS[1], 
                             dims=([-2], [-1]))
        T0 = torch.tensordot(T0,
                             _MAPPINGS[2], 
                             dims=([-2], [-1]))

        type0_to_type0, type0_to_type1 = _calc_basis(tensor_dict={0: s0, 1:V0, 2: Q0, 3: T0}, 
                                                     inv_einsums=_INVARIANT_BASIS,
                                                     equ_einsums=_COVARIANT_BASIS)

        # Type1 Tensors
        env_type1 = env_type1.reshape(
        n_atoms_real * n_tensor_comp,
        self.n_dist * self.type1_in,
        3,
    )
        tensor_features_type1 = torch.einsum(
        "bfi,fo->boi", env_type1, self.int_weights_type1
    )
        tensor_features_type1 = tensor_features_type1.reshape(
        n_atoms_real, n_tensor_comp, self.type1_out, 3
    )
        s1, V1, Q1, T1 = _split_tensor_feature_by_orders(l_max=self.l_max, tensor_features=tensor_features_type1, dim=-3) 
        # NOTE: We can do the contraction in SFTs but the mapping is different than s0, V0, Q0, etc 
        # From
        # SFT for first 3 dimensions the last is the vector? 
        s1 = s1[:, 0]
        V1 = torch.tensordot(V1, 
                             _MAPPINGS[0],
                             dims=([-3], [-1]))
        Q1 = torch.tensordot(Q1, 
                             _MAPPINGS[1], 
                             dims=([-3], [-1]))
        T1 = torch.tensordot(T1,
                             _MAPPINGS[2], 
                             dims=([-3], [-1]))

        type1_to_type0, type1_to_type1 = _calc_basis(tensor_dict={0: None, 1: s1, 2:V1, 3: Q1, 4: T1}, 
                                                     inv_einsums=_INVARIANT_BASIS,
                                                     equ_einsums=_COVARIANT_BASIS)

        type0 = torch.concat([type0_to_type0, type1_to_type0], dim=-1)
        type1 = torch.concat([type0_to_type1, type1_to_type1], dim=-2)
        # Change the state type0, type1
        # TODO: Add norms, I think it should be done before the contractions and store magnitude invariants. 

        # TOOD: Add mixing 
        type0 = type0.reshape(n_atoms_real, self.type0_out * len(_INVARIANT_BASIS)*2)
        type0 = type0 @ self.type0_mixing_weights.reshape(-1, self.type0_out)
        type1 = type1.reshape(n_atoms_real, self.type1_out * len(_COVARIANT_BASIS)*2, 3)
        # Mix vector channels with shared scalar weights, preserving Cartesian components, this should be more efficient.
        type1 = torch.einsum(
            "aci,co->aoi", type1, self.type1_mixing_weights.reshape(-1, self.type1_out)
        )
        return type0, type1
