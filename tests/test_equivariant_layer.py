import torch
import numpy as np
import pytest
from hippynn.networks.hiphophooray import (
    FeatureState, CovariantLayer, _split_tensor_feature_by_orders, _MAPPINGS,
    _INVARIANT_BASIS, _COVARIANT_BASIS,
)
from hippynn.layers.hiplayers.sensitivity import InverseSensitivityModule
from hippynn.networks.tensors import TensorExtractor
from hippynn.custom_kernels import envsum
from scipy.spatial.transform import Rotation


DTYPE = torch.get_default_dtype()
RTOL = 1e-5
ATOL = 1e-5

def four_point_arrow_configuration(dtype=np.float32):
    """ Generate four points and vector as target

    Args:
        dtype (_type_, optional): Type of numpy arrays. Defaults to np.float32.

    Returns:
        _type_: [points, direction]
    """
    sqrt3 = np.sqrt(3.0)

    points = np.array([
        [ 1.0,          0.0,  0.0],   # unique tip
        [-0.5,  sqrt3 / 2.0,  0.0],   # tail
        [-0.5, -sqrt3 / 4.0,  0.75],  # tail
        [-0.5, -sqrt3 / 4.0, -0.75],  # tail
    ], dtype=dtype)

    tip = points[0]
    tail_center = points[1:].mean(axis=0)

    direction = tip - tail_center
    direction /= np.linalg.norm(direction)

    return points, direction

def get_state(points, in_features_type0, in_features_type1):
    """Get a FeatureState object from points and features."""
    n = points.shape[0]
    indices = torch.arange(n, device=points.device)

    pair_first = indices.repeat_interleave(n)
    pair_second = indices.repeat(n)

    mask = pair_first != pair_second
    pair_first = pair_first[mask]
    pair_second = pair_second[mask]

    pair_coord = points[pair_second] - points[pair_first]
    pair_dist = torch.linalg.norm(pair_coord, dim=-1)
    rhats = pair_coord / pair_dist.unsqueeze(-1)

    extractor = TensorExtractor(l_max=3)
    angular_tensors = extractor(rhats)

    state = FeatureState(
        pair_first=pair_first,
        pair_second=pair_second,
        pair_dist=pair_dist,
        pair_coord=pair_coord,
        rhat_compact_tensors=torch.cat(angular_tensors, dim=-1),
        in_features_type0=in_features_type0,
        in_features_type1=in_features_type1
    )
    return state

@pytest.fixture
def rotated_inputs():
    points, _ = four_point_arrow_configuration()
    points = torch.as_tensor(points, dtype=DTYPE)

    rotation = torch.as_tensor(
        Rotation.random(random_state=123).as_matrix(),
        dtype=DTYPE,
    )

    vectors = torch.randn(points.shape[0], 1, 3, dtype=DTYPE)
    in_features = torch.randn(points.shape[0], 5, dtype=DTYPE)

    state = get_state(points,
                      in_features_type0=in_features,
                      in_features_type1=vectors)
    rot_state = get_state(points @ rotation.T,
                          in_features_type0=in_features,
                          in_features_type1=vectors @ rotation.T)
    return state, rot_state, rotation


def test_inputs(rotated_inputs):
    state, rot_state, rotation = rotated_inputs
    assert torch.allclose(state.pair_first, rot_state.pair_first)
    assert torch.allclose(state.pair_second, rot_state.pair_second)
    assert torch.allclose(state.pair_dist, rot_state.pair_dist)
    assert torch.allclose(state.pair_coord @ rotation.T, rot_state.pair_coord)
    assert state.rhat_compact_tensors.shape == rot_state.rhat_compact_tensors.shape

    assert torch.allclose(state.in_features_type0, rot_state.in_features_type0)
    assert torch.allclose(state.in_features_type1 @ rotation.T, rot_state.in_features_type1)

    # Extract the tensors and check their properties
    compact_tensors = state.rhat_compact_tensors

    rot_compact_tensors = rot_state.rhat_compact_tensors
    env_features = envsum(compact_tensors,
                          torch.ones((4, 1)), 
                          state.pair_first,
                          state.pair_second)
    env_features_rot = envsum(rot_compact_tensors,
                              torch.ones((4, 1)),
                              rot_state.pair_first,
                              rot_state.pair_second)
    
    s, V, Q, T = _split_tensor_feature_by_orders(l_max=3, tensor_features=env_features, dim=-2)

    V = torch.tensordot(V, 
                            _MAPPINGS[0],
                            dims=([-2], [-1]))[:,0]
    Q = torch.tensordot(Q, 
                            _MAPPINGS[1], 
                            dims=([-2], [-1]))[:,0]
    T = torch.tensordot(T,
                            _MAPPINGS[2], 
                            dims=([-2], [-1]))[:,0]

    rot_s, rot_V, rot_Q, rot_T = _split_tensor_feature_by_orders(l_max=3, tensor_features=env_features_rot, dim=-2)

    rot_V = torch.tensordot(rot_V,
                            _MAPPINGS[0],
                            dims=([-2], [-1]))[:,0]
    rot_Q = torch.tensordot(rot_Q,
                            _MAPPINGS[1],
                            dims=([-2], [-1]))[:,0]
    rot_T = torch.tensordot(rot_T,
                            _MAPPINGS[2],
                            dims=([-2], [-1]))[:,0]
    # Test the tensors 
    assert torch.allclose(s, rot_s)
    # Rank 1: V_i = R_ij A_j
    assert torch.allclose(V @ rotation.T, rot_V)
    # Rank 2: Q_ij = R_ia R_jb Q_ab
    assert torch.allclose(rotation @ Q @ rotation.T, rot_Q)

    # Rank 3: B_ijk = R_ia R_jb R_kc A_abc
    corrected_T = torch.einsum('ia,jb,kc,habc->hijk',
                            rotation,
                            rotation,
                            rotation,  T)
    assert torch.allclose(rot_T, corrected_T)


def test_covariant_layer(rotated_inputs):
    state, rotated_state, rotation = rotated_inputs

    layer = CovariantLayer(
        inv_basis=_INVARIANT_BASIS,
        equi_basis=_COVARIANT_BASIS,
        l_max=3,
        nf_in=(state.in_features_type0.shape[-1], state.in_features_type1.shape[-2]),
        nf_out=(4, 4),
        sensitivity_module=InverseSensitivityModule,
        n_sensitivities=20,
        dist_soft_min=0.1,
        dist_soft_max=2,
        dist_hard_max=10.,
    )

    type0, type1 = layer(state)
    rot_type0, rot_type1 = layer(rotated_state)

    assert type0.shape == rot_type0.shape
    assert type1.shape == rot_type1.shape
    torch.testing.assert_close(type0, rot_type0, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(type1 @ rotation.T, rot_type1, rtol=RTOL, atol=ATOL)
