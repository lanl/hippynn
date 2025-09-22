import torch
torch.set_default_dtype(torch.float32)

from hippynn.experiment.serialization import load_checkpoint_from_cwd
from hippynn.tools import active_directory, device_fallback

from hippynn.interfaces.lammps_interface import MLIAPInterface
from hippynn.graphs import make_ensemble # added

if __name__ == "__main__":
    # Load ensemble of trained models
    try:
        bundle = "../TEST_ALUMINUM_MODEL_*"
    except FileNotFoundError:
        raise FileNotFoundError("Model not found, run ani_aluminum_example.py first!")
    
    ensemble_graph, ensemble_info = make_ensemble(bundle) # make an ensemble of graphs
    ensemble_energy = ensemble_graph.node_from_name("ensemble_atomenergies") #create ensemble energy nodes
   
    extra_properties = {"energy_std": ensemble_energy.std} # create extra properties
    unified = MLIAPInterface(ensemble_energy, ["Al"],is_ensemble=True, extra_properties=None, model_device=device_fallback())
    torch.save(unified, "mliap_unified_hippynn_Al.pt")

