"""
Script running an aluminum model with ASE.

This script is designed to match the 
LAMMPS script located at 
./lammps/in.mliap.unified.hippynn.Al

Before running this script, you must run 
`ani_aluminum_example_multilayer.py` to 
train the corresponding model.

Modified from ase MD example.
"""

# Imports
import numpy as np
import torch
import ase
import time

from hippynn.experiment.serialization import load_checkpoint_from_cwd
from hippynn.tools import active_directory
from hippynn.interfaces.ase_interface import HippynnCalculator
import ase.build
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.lattice.cubic import FaceCenteredCubic
from hippynn.graphs import make_ensemble

try:
    #with active_directory("/vast/home/dshahi/test_al_model/TEST_ALUMINUM_MODEL", create=False):
        #bundle = load_checkpoint_from_cwd(map_location="cpu")
        bundle = "/vast/home/dshahi/aluminum/with_atom_energies_test/TEST_ALUMINUM_MODEL_*" #added
except FileNotFoundError:
    raise FileNotFoundError("Model not found, run ani_aluminum_example.py first!")

#model = bundle["training_modules"].model
ensemble_graph, ensemble_info = make_ensemble(bundle) #added
ensemble_energy = ensemble_graph.node_from_name("ensemble_atomenergies") #added
ensemble_force = ensemble_graph.node_from_name("ensemble_force")

# Build the calculator
#energy_node = model.node_from_name("energy")

energy_node = ensemble_energy.mean #added
extra_properties = {"ens_predictions": ensemble_energy.all, "ens_std": ensemble_energy.std, "force_std":ensemble_force.std}
calc = HippynnCalculator(energy_node, extra_properties=extra_properties, en_unit=units.eV) #added
#calc = HippynnCalculator(energy_node, en_unit=units.eV)
'''# Load the files
try:
    with active_directory("TEST_ALUMINUM_MODEL_MULTILAYER", create=False):
        bundle = load_checkpoint_from_cwd(map_location='cpu')
except FileNotFoundError:
    raise FileNotFoundError("Model not found, run ani_aluminum_example_multilayer.py first!")

model = bundle["training_modules"].model


# Build the calculator
energy_node = model.node_from_name("energy")
calc = HippynnCalculator(energy_node, en_unit=units.eV)'''
calc.to(torch.float64)

if torch.cuda.is_available():
    calc.to(torch.device('cuda'))

# Build the atoms object
atoms = FaceCenteredCubic(directions=np.eye(3, dtype=int),
                          size=(1,1,1), symbol='Al', pbc=(True,True,True))
nrep = 3 # 4
reps = nrep*np.eye(3, dtype=int)
atoms = ase.build.make_supercell(atoms, reps, wrap=True)
atoms.rattle(0.1)
atoms.calc = calc
print("Number of atoms:", len(atoms))

# atoms.rattle(.1)
MaxwellBoltzmannDistribution(atoms, temperature_K=300)
dyn = VelocityVerlet(atoms, 0.5*units.fs)


forces = atoms.get_forces()
xyz = ase.io.write('positions.xyz', atoms, 'lammps-data')
atomic_numbers = atoms.get_atomic_numbers()
zipped = zip(atomic_numbers, forces)
print("zipped:",sorted(zipped, key=lambda x: x[0]))
zipped = sorted(zipped, key=lambda x: x[0])
ids = np.arange(1, forces.shape[0]+1).reshape(forces.shape[0], 1)
data = np.hstack((ids, forces))
sorted_data = data[data[:, 0].argsort()]
np.savetxt('forces_ase.xyz', sorted_data, fmt=['%d', '%.18e', '%.18e', '%.18e'])

# Simple tracker of the simulation progress, this is not needed to perform MD.
class Tracker():
    def __init__(self, dyn, atoms):
        self.last_call_time = time.time()
        self.last_call_steps = 0
        self.dyn = dyn
        self.atoms = atoms

    def update(self):
        now = time.time()
        diff = now-self.last_call_time
        diff_steps = dyn.nsteps - self.last_call_steps
        try:
            time_per_atom_step = diff/(len(atoms)*diff_steps)
        except ZeroDivisionError:
            time_per_atom_step = float('NaN')
        self.last_call_time = now
        self.last_call_steps = dyn.nsteps
        return time_per_atom_step

    def print(self):
        time_per_atom_step = self.update()
        """Function to print the potential, kinetic and total energy"""
        simtime = round(self.dyn.get_time() / (1000*units.fs), 3)
        print("Simulation time so far:",simtime,"ps")
        print("Performance:",round(1e6*time_per_atom_step,1)," microseconds/(atom-step)")
        epot = self.atoms.get_potential_energy() / len(self.atoms)
        ekin = self.atoms.get_kinetic_energy() / len(self.atoms)
        #forces = self.atoms.get_forces() # added
        stress = self.atoms.get_stress()
        print('Energy per atom: Epot = %.7feV  Ekin = %.7feV (T=%3.0fK)  '
              'Etot = %.7feV  Stress = %.7f' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin, stress[:3].sum()/3 / units.bar))
        #atomic_numbers = self.atoms.get_atomic_numbers()

        #print("forces:", forces) # added
        #print("type(forces)",type(forces)) # added
        #print("forces.shape",forces.shape) # added
        #positions = self.atoms.get_positions() # added
        '''xyz = ase.io.write('positions.xyz', self.atoms, 'lammps-data')
        zipped = zip(atomic_numbers, forces)
        print("zipped:",sorted(zipped, key=lambda x: x[0]))
        zipped = sorted(zipped, key=lambda x: x[0])
        #with open('zipped.txt', 'w') as f:
        #    for item in zipped:
        #        f.write(f"{item[0]},{item[1]}\n")
        print(f'FORCES: {type(forces)}')
        ids = np.arange(1, forces.shape[0]+1).reshape(forces.shape[0], 1)
        data = np.hstack((ids, forces))
        #sorted_data = np.argsort(data, axis=0)
        sorted_data = data[data[:, 0].argsort()]
        np.savetxt('forces_ase.xyz', sorted_data, fmt=['%d', '%.18e', '%.18e', '%.18e'])
        '''

'''pot_energ_all_atoms = atoms.get_potential_energy()/len(atoms)
print("pot_energ_all_atoms:", pot_energ_all_atoms)
energy_data = np.hstack((ids, pot_energ_all_atoms))
energy_sorted_data = energy_data[energy_data[:, 0].argsort()]
np.savetxt('energies_ase.xyz', energy_sorted_data, fmt=['%d', '%.18e'])
'''

eatoms = np.mean(atoms.calc.results['ens_predictions'], axis=1)[0,:,:]
eatoms_stdev = atoms.calc.results['ens_std'][0,:,:]
eatoms_data = np.hstack((ids, eatoms, eatoms_stdev))
sorted_eatoms_data = eatoms_data[eatoms_data[:,0].argsort()]
np.savetxt('eatoms_uq_ase.xyz', sorted_eatoms_data, fmt=['%d', '%.18e', '%.18e'])

# Now run the dynamics
tracker = Tracker(dyn, atoms)
tracker.print()
for i in range(20):  # Run 2 ps
    dyn.run(50)  # Run 20 fs
    tracker.print()
