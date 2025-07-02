import torch
torch.set_default_dtype(torch.float32)

from hippynn.experiment.serialization import load_checkpoint_from_cwd
from hippynn.tools import active_directory, device_fallback

from hippynn.interfaces.lammps_interface import MLIAPInterface
from hippynn.graphs import make_ensemble # added

if __name__ == "__main__":
    # Load trained model
    try:
        #with active_directory("/vast/home/dshahi/test_al_model/TEST_ALUMINUM_MODEL", create=False):
        #with active_directory("/vast/home/dshahi/ensembles/ensemble-0206/model-0*/", create=False): #added
        #    bundle = load_checkpoint_from_cwd(map_location="cpu")
        bundle = "/vast/home/dshahi/aluminum/with_atom_energies_test/TEST_ALUMINUM_MODEL_*"
    except FileNotFoundError:
        raise FileNotFoundError("Model not found, run ani_aluminum_example.py first!")
    
    ensemble_graph, ensemble_info = make_ensemble(bundle) #added
    ensemble_energy = ensemble_graph.node_from_name("ensemble_atomenergies") #added
    ensemble_force = ensemble_graph.node_from_name("ensemble_force") #addedi
    ensemble_energy_all = ensemble_energy.all
   
    print("type ensemble energy:", type(ensemble_energy))
    extra_properties = {"energy_std": ensemble_energy.std}
    #unified = MLIAPInterface(ensemble_energy, ["H", "C", "N", "O", "P", "S", "Cl"], model_device=device_fallback())
    unified = MLIAPInterface(ensemble_energy, ["Al"],is_ensemble=True, extra_properties=extra_properties, model_device=device_fallback())
    #unified = MLIAPInterface(ensemble_energy.mean, ["Al"],is_ensemble=True, model_device=device_fallback())
    #unified = MLIAPInterface(energy_node, ["Al"], model_device=device_fallback())
    torch.save(unified, "mliap_unified_hippynn_Al.pt")
    print("Finished saving")

