import torch
torch.set_default_dtype(torch.float32)

from hippynn.experiment.serialization import load_checkpoint_from_cwd
from hippynn.tools import active_directory, device_fallback

from hippynn.interfaces.lammps_interface import MLIAPInterface


if __name__ == "__main__":
    # Load trained model
    try:
        with active_directory("../TEST_INP_MODEL", create=False):
            bundle = load_checkpoint_from_cwd(map_location="cpu")
    except FileNotFoundError:
        raise FileNotFoundError("Model not found, run lammps_train_model_InP.py first!")

    model = bundle["training_modules"].model
    energy_node = model.node_from_name("HEnergy")

    unified = MLIAPInterface(energy_node, ["In", "P"], model_device=device_fallback())
    torch.save(unified, "mliap_unified_hippynn_InP.pt")
