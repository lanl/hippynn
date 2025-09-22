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
    bundle = "../TEST_ALUMINUM_MODEL_*" 
except FileNotFoundError:
    raise FileNotFoundError("Model not found, run ani_aluminum_example.py first!")

# Create ensemble prediction model
ensemble_graph, ensemble_info = make_ensemble(bundle) 
ensemble_energy = ensemble_graph.node_from_name("ensemble_atomenergies") 
ensemble_force = ensemble_graph.node_from_name("ensemble_force")

# Node rediction and calculator
energy_node = ensemble_energy.mean 
extra_properties = {"ens_predictions": ensemble_energy.all, "ens_std": ensemble_energy.std, "force_std":ensemble_force.std}
calc = HippynnCalculator(energy_node, extra_properties=extra_properties, en_unit=units.eV)
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

#Set initial conditions and dynamics
# atoms.rattle(.1)
MaxwellBoltzmannDistribution(atoms, temperature_K=300)
dyn = VelocityVerlet(atoms, 0.5*units.fs)

# Force calculation and output
forces = atoms.get_forces() #compute forces
xyz = ase.io.write('positions.xyz', atoms, 'lammps-data') # Write the atomic positions to LAMMPS-compatible file
atomic_numbers = atoms.get_atomic_numbers()
zipped = zip(atomic_numbers, forces) #Pair each atomic number with itsforce vector
zipped = sorted(zipped, key=lambda x: x[0]) #Sort the atom-force pairs by atomic number 
ids = np.arange(1, forces.shape[0]+1).reshape(forces.shape[0], 1)
data = np.hstack((ids, forces))
sorted_data = data[data[:, 0].argsort()]
np.savetxt('forces_ase.xyz', sorted_data, fmt=['%d', '%.18e', '%.18e', '%.18e']) #save force

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
        stress = self.atoms.get_stress()
        print('Energy per atom: Epot = %.7feV  Ekin = %.7feV (T=%3.0fK)  '
              'Etot = %.7feV  Stress = %.7f' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin, stress[:3].sum()/3 / units.bar))

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
