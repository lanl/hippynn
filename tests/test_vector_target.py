import torch

from hippynn.layers.targets import HVector


def _example_inputs(dtype=torch.float64):
    features = [
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [1.0, 1.0]], dtype=dtype),
        torch.tensor([[0.2], [0.7], [1.1], [-0.3]], dtype=dtype),
    ]
    positions = torch.tensor(
        [[1.0, 0.0, 0.0], [-0.5, 0.8, 0.0], [-0.5, -0.8, 0.0], [2.0, 1.0, -1.0]],
        dtype=dtype,
    )
    system_index = torch.tensor([0, 0, 0, 1])
    return features, positions, system_index, 2


def test_hvector_is_translation_invariant_and_rotation_equivariant():
    features, positions, system_index, n_systems = _example_inputs()
    layer = HVector(feature_sizes=(2, 1)).to(dtype=positions.dtype)

    angle = torch.tensor(0.73, dtype=positions.dtype)
    zero = torch.zeros((), dtype=positions.dtype)
    one = torch.ones((), dtype=positions.dtype)
    rotation = torch.stack(
        (
            torch.stack((angle.cos(), -angle.sin(), zero)),
            torch.stack((angle.sin(), angle.cos(), zero)),
            torch.stack((zero, zero, one)),
        )
    )
    translation = torch.tensor([4.0, -2.0, 1.5], dtype=positions.dtype)

    vector, atom_vectors, _ = layer(features, positions, system_index, n_systems)
    transformed_vector, transformed_atom_vectors, _ = layer(
        features, positions @ rotation.T + translation, system_index, n_systems
    )

    torch.testing.assert_close(transformed_vector, vector @ rotation.T)
    torch.testing.assert_close(transformed_atom_vectors, atom_vectors @ rotation.T)


def test_hvector_multiple_targets_shape():
    features, positions, system_index, n_systems = _example_inputs()
    layer = HVector(feature_sizes=(2, 1), n_target=2).to(dtype=positions.dtype)

    system_vectors, atom_vectors, vector_terms = layer(features, positions, system_index, n_systems)

    assert system_vectors.shape == (2, 2, 3)
    assert atom_vectors.shape == (4, 2, 3)
    assert len(vector_terms) == 2
    assert vector_terms[-1].shape == (2, 2, 3)
