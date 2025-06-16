import pytest

@pytest.fixture
def network_parameters():
    return {
        "possible_species": [0, 13],
        "n_features": 10,
        "n_sensitivities": 20,
        "dist_soft_min": 1.25,
        "dist_soft_max": 7,
        "dist_hard_max": 7.5,
        "n_interaction_layers": 1,
        "n_atom_layers": 3,
        "sensitivity_type": "inverse",
        "resnet": True,
    }
