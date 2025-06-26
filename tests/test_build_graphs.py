import pytest

import hippynn
import ase

def xfail_if_no_lammps(func):
    try:
        import lammps
    except ImportError:  # missing lammps doesn't raise ImportError per se.  ModuleNotFoundError.
        # Something went wrong importing!
        wrapper = pytest.mark.xfail(strict=False)
    else:
        # Importing
        wrapper = lambda f: f
    return wrapper(func)

def test_build_training_modules(energy_model):

    mae = hippynn.loss.MAELoss.of_node(energy_model)

    validation_losses = {"MAE": mae}

    training_modules, db_info = hippynn.experiment.assemble_for_training(mae, validation_losses)


# Mark as xfail because lammps python installations require manual steps.
@xfail_if_no_lammps
def test_build_lammps_interface(energy_model):
    from hippynn.interfaces.lammps_interface import MLIAPInterface

    interface = MLIAPInterface(energy_model,element_types=[1])



def test_build_ase_interface(energy_model):
    from hippynn.interfaces.ase_interface import HippynnCalculator

    calc = HippynnCalculator(energy_model)
