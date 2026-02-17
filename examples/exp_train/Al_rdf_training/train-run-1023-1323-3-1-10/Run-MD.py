import sys
import os
import subprocess
import numpy as np
from pathlib import Path

import hippynn
from hippynn.interfaces.ase_interface import HippynnCalculator
import torch
from copy import deepcopy

from hippynn.experiment.serialization import load_checkpoint_from_cwd
from hippynn.tools import active_directory
from hippynn.interfaces.ase_interface import HippynnCalculator


import ase
from ase.build import bulk
from ase.build import make_supercell
from ase.lattice.cubic import FaceCenteredCubic

from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase.md.verlet import VelocityVerlet
from ase.md.nptberendsen import NPTBerendsen
from ase.io.trajectory import Trajectory
from ase import units

from ase.optimize import BFGS,LBFGS,FIRE,GPMin
from ase.constraints import StrainFilter

from ase.md import MDLogger

from ase.io import read, write

from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from ASEtrackers import storeenergy,runRDF,genLog

#Args: HIPNNmodel, tmpDir, logFile, targTemp, targPres, trajFile
HIPNNmodel = sys.argv[1]
tmpDir = sys.argv[2]
logFile = sys.argv[3]
targTemp = float(sys.argv[4])
targPres = float(sys.argv[5]) #This is a float in ASE units. 1 atm is 1/0.986923*units.bar=6.324210830919189e-07
if len(sys.argv) > 6:
    trajFile = sys.argv[6]
else:
    trajFile = None
Path(tmpDir).mkdir(parents=True, exist_ok=True)

a2 = FaceCenteredCubic(size=(8,8,8),symbol='Al',pbc=(1,1,1),latticeconstant=4.0478)

with active_directory(HIPNNmodel):
    bundle = load_checkpoint_from_cwd(map_location="cpu", restore_db=False)
model = bundle["training_modules"].model
energy_node = model.node_from_name("energy")
calc = HippynnCalculator(energy_node, en_unit=units.eV)
calc.to(torch.float32)
calc.to(torch.device("cuda:0"))

np.linalg.norm(a2.get_positions()[1])

a2.calc = calc
sf = StrainFilter(a2)
opt=BFGS(sf)
opt.run(.005,steps=500)

a2.set_cell([[a2.cell[0,0],0,0],[0,a2.cell[1,1],0],[0,0,a2.cell[2,2]]])

if trajFile is not None:
    trajob = Trajectory(trajFile,mode='w',atoms=a2)

#a2.rattle(0.1)
MaxwellBoltzmannDistribution(a2, temperature_K=targTemp)
dyn = NPTBerendsen(a2,.5*units.fs,temperature_K=targTemp,pressure_au=targPres,taut=0.5e3 * units.fs,taup=1.0e3 * units.fs,compressibility_au=0.6)
#dyn.attach(MDLogger(dyn, a2, 'md.log', header=True, stress=True,peratom=True, mode="a"), interval=1000)
dyn.attach(genLog( a2, dyn, fileName=logFile, printHeader=True), interval=100)

#Fast equil        
dyn.run(20000)

dyn.set_taut(1.0e3 * units.fs)
dyn.set_taup(2.0e3 * units.fs)
#Slow equil
dyn.run(20000)

dyn.attach(runRDF(a2,dyn,radius=10.0,nBin=200,tmpDir=tmpDir), interval=1000)
if trajFile is not None:
    dyn.attach(trajob.write,interval=20)
#prod
dyn.run(50000)
