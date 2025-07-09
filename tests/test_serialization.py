import pytest

import hippynn

from conftest import xfail_if_no_models, MODEL_DIR
from conftest import ignore_cusp_warning, ignore_relocation, ignore_weights_only_warning, ignore_sensitivity_warning
import torch


@ignore_relocation
@ignore_weights_only_warning
@ignore_cusp_warning
@xfail_if_no_models
def test_load_old():
    from hippynn.tools import active_directory
    from hippynn.experiment import load_checkpoint_from_cwd

    location = "./quad0_b512_int1_p5_GPU0_seed363144"
    with active_directory(MODEL_DIR / location, create=False):
        check = load_checkpoint_from_cwd(map_location="cpu")

    return


@ignore_weights_only_warning
@ignore_cusp_warning
@ignore_relocation
@ignore_sensitivity_warning
@xfail_if_no_models
def test_run_old(example_box):
    from hippynn.tools import active_directory
    from hippynn.experiment import load_checkpoint_from_cwd
    from hippynn.graphs import Predictor

    location = "./quad0_b512_int1_p5_GPU0_seed363144"

    with active_directory(MODEL_DIR / location, create=False):
        check = load_checkpoint_from_cwd(map_location="cpu")

    model = check["training_modules"].model
    predictor = Predictor.from_graph(model)

    renamed_box = dict(
        Z=example_box["species"],
        R=example_box["coordinates"],
    )

    outputs = predictor(**renamed_box)

    return

@ignore_weights_only_warning
@ignore_cusp_warning
@ignore_relocation
@xfail_if_no_models
def test_validate_old():

    from optimizer.test_configs import c2h6_config

    expected_T = torch.tensor([[-566.0421],
          [-633.7504],
          [-683.5908],
          [-706.4001],
          [-713.3954],
          [-711.6498],
          [-705.4093],
          [-694.8241],
          [-686.6531],
          [-677.4547],
          [-571.3558],
          [-471.0748],
          [-485.9834],
          [-526.7410],
          [-525.0599]])

    
    from hippynn.tools import active_directory
    from hippynn.experiment import load_checkpoint_from_cwd
    from hippynn.graphs import Predictor

    location = "./quad0_b512_int1_p5_GPU0_seed363144"

    with active_directory(MODEL_DIR / location, create=False):
        check = load_checkpoint_from_cwd(map_location="cpu")

    model = check["training_modules"].model
    predictor = Predictor.from_graph(model)

    renamed_box = dict(
        Z=c2h6_config["Z"],
        R=c2h6_config["R"],
    )

    outputs = predictor(**renamed_box)
    out_energy = outputs['T']

    maxreldiff = ((out_energy - expected_T)/expected_T).abs().max()

    assert maxreldiff < 1.2e-7, f"Output of network changed. (relative difference = {maxreldiff})"
    
    return
