import os
import torch
torch.set_default_dtype(torch.float32)

from hippynn.experiment.serialization import load_checkpoint_from_cwd
from hippynn.tools import active_directory, device_fallback

from hippynn.interfaces.lammps_interface import MLIAPInterface

def save_model_for_lammps(model_folder, species_names_ordered, mliap_filename = "mliap_unified_hippynn.pt"):
    with active_directory(model_folder, create=False):
        bundle = load_checkpoint_from_cwd(map_location='cpu')

        model = bundle["training_modules"].model

        henergy_node = model.node_from_name("HEnergy")
        repulse_node = model.node_from_name("repulse")

        atom_energies = henergy_node.atom_energies + repulse_node.atom_energies

        unified = MLIAPInterface(atom_energies, species_names_ordered, model_device=device_fallback())

        torch.save(unified, mliap_filename)
        print(f"LAMMPS ML-IAP saved at {os.path.abspath(os.path.join(model_folder, mliap_filename))}")

if __name__ == "__main__":
    save_model_for_lammps("model", species_names_ordered=["MeOH"])