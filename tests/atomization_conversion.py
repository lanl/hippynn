'''
To obtain the data files needed for this example, use the script process_QM7_data.py,
also located in this folder. The script contains further instructions for use.
'''

import torch

# Setup pytorch things
torch.set_default_dtype(torch.float64)

import hippynn

netname = "TEST_BAREBONES_SCRIPT"

# Hyperparameters for the network
# These are set deliberately small so that you can easily run the example on a laptop or similar.
network_params = {
    "possible_species": [0, 1, 6, 7, 8, 16],  # Z values of the elements in QM7
    "n_features": 20,  # Number of neurons at each layer
    "n_sensitivities": 20,  # Number of sensitivity functions in an interaction layer
    "dist_soft_min": 1.6,  # qm7 is in Bohr!
    "dist_soft_max": 10.0,
    "dist_hard_max": 12.5,
    "n_interaction_layers": 2,  # Number of interaction blocks
    "n_atom_layers": 3,  # Number of atom layers in an interaction block
}

# Define a model
from hippynn.graphs import inputs, networks, targets, physics

species = inputs.SpeciesNode(db_name="Z")
positions = inputs.PositionsNode(db_name="R")

network = networks.Hipnn("hipnn_model", (species, positions), module_kwargs=network_params)
# henergy = targets.HEnergyNode("HEnergy", network, db_name="T")
henergy = targets.AtomizationEnergyNode("HEnergy", network, db_name="T")

model = hippynn.GraphModule([species,positions], [henergy.mol_energy])

from hippynn import ase_interface as hai
import ase.units, ase.build

atoms = ase.build.molecule("H2O")

pos = torch.as_tensor(atoms.positions / ase.units.Bohr).unsqueeze(0).to(torch.get_default_dtype())
sp = torch.as_tensor(atoms.get_atomic_numbers()).unsqueeze(0)
pred = hippynn.Predictor.from_graph(model)
original_en = pred(Z=sp, R=pos)[henergy.mol_energy]
pred.graph.print_structure()
print("predictor output:", original_en)
calc = hai.calculator_from_model(model, dist_unit=ase.units.Bohr)

calc.module.print_structure()
atoms.calc = calc
ase_en = atoms.get_potential_energy() / (ase.units.kcal / ase.units.mol)
print("ASE Energy is:", ase_en)
print("Ratio:", ase_en/original_en)

if not torch.allclose(torch.as_tensor(ase_en), original_en):
    raise ValueError(f"Values do not match!: {ase_en},{original_en}")
