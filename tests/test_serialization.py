import pytest

import hippynn

from test_run_nodes import sensitivity_warning_supression

from conftest import xfail_if_no_models, MODEL_DIR
from conftest import ignore_cusp_warning, ignore_relocation, ignore_weights_only_warning




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
@pytest.mark.filterwarnings("ignore:.*underneath sensitivity range*.")
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
