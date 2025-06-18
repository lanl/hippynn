import os
import glob

import torch
torch.set_default_dtype(torch.float32)

from hippynn.graphs import make_ensemble
from hippynn.tools import device_fallback
from hippynn.interfaces.lammps_interface import MLIAPInterface


if __name__ == "__main__":
    #Path to folder of models
    ENSEMBLE_FOLDER = "ALUMINUM_ENSEMBLES"
    MODEL_REGEX = "TEST_ALUMINUM_MODEL_*"
    model_form = os.path.join(ENSEMBLE_FOLDER, MODEL_REGEX)

    #Check model_form has models
    if len(glob.glob(model_form)) == 0:
      raise FileNotFoundError("Model not found, run ani_aluminum_example.py first!")
  
    # Load trained model as ensemble
    ensemble_graph, ensemble_info = make_ensemble(model_form)

    # Retrieve Ensemble Nodes
    ensemble_energy = ensemble_graph.node_from_name("ensemble_atomenergies")

    unified = MLIAPInterface(ensemble_energy.mean, ["Al"], model_device=device_fallback())
    torch.save(unified, "mliap_unified_hippynn_ensemble_Al.pt")
