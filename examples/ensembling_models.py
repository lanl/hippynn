import torch
import hippynn

if torch.cuda.is_available():
    device = 0
else:
    device = 'cpu'

### Building the ensemble just requires calling one function call.
model_form = '../../collected_models/quad0_b512_p5_GPU*'
ensemble_graph, ensemble_info = hippynn.graphs.make_ensemble(model_form)

# Retrieve the ensemble node which has just been created.
# The name will be the prefix 'ensemble' followed by the db_name from the ensemble members.
ensemble_energy = ensemble_graph.node_from_name("ensemble_T")

### Building an ASE calculator for the ensemble

import ase.build

from hippynn.interfaces.ase_interface import HippynnCalculator

# The ensemble node has `mean`, `std`, and `all` outputs.
energy_node = ensemble_energy.mean
extra_properties = {"ens_predictions": ensemble_energy.all, "ens_std": ensemble_energy.std}
calc = HippynnCalculator(energy=energy_node, extra_properties=extra_properties)
calc.to(device)

# build something and attach the calculator
molecule = ase.build.molecule("CH4")
molecule.calc = calc

energy_value = molecule.get_potential_energy()  # Activate calculation to get results dict

print("Got energy", energy_value)
print("In units of kcal/mol", energy_value / (ase.units.kcal/ase.units.mol))

# All outputs from the ensemble members. Because the model was trained in kcal/mol, this is too.
# The name in the results dictionary comes from the key in the 'extra_properties' dictionary.
print("All predictions:", calc.results["ens_predictions"])


### Building a Predictor object for the ensemble
pred = hippynn.graphs.Predictor.from_graph(ensemble_graph)

# get batch-like inputs to the ensemble
z_vals = torch.as_tensor(molecule.get_atomic_numbers()).unsqueeze(0)
r_vals = torch.as_tensor(molecule.positions).unsqueeze(0)

pred.to(r_vals.dtype)
pred.to(device)
# Do some computation
output = pred(Z=z_vals, R=r_vals)
# Print the output of a node using the node or the db_name.
print(output[ensemble_energy.all])
print(output["T_all"])