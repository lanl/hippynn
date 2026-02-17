import sys
import os
import subprocess
import numpy as np

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
from ase.md.nvtberendsen import NVTBerendsen
from ase.io.trajectory import Trajectory
from ase import units

from ase.optimize import BFGS,LBFGS,FIRE,GPMin
from ase.constraints import StrainFilter

from ase.md import MDLogger

from ase.io import read, write

from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from ASEtrackers import storeenergy,runRDF,genLog

#argv[1] is cuda device
HIPNNmodel = sys.argv[1]
tmpDir = sys.argv[2]
targTemp = float(sys.argv[3])

a2 = FaceCenteredCubic(size=(8,8,8),symbol='Al',pbc=(1,1,1),latticeconstant=4.0478)

with active_directory(HIPNNmodel):
    bundle = load_checkpoint_from_cwd(map_location="cpu", restore_db=False)
model = bundle["training_modules"].model
energy_node = model.node_from_name("energy")
calc = HippynnCalculator(energy_node, en_unit=units.eV)
calc.to(torch.float32)
calc.to(torch.device("cuda:0"))

print(a2.cell[0,0]/8)
print(a2.cell[2,2]/8)

np.linalg.norm(a2.get_positions()[1])

a2.calc = calc
sf = StrainFilter(a2)
opt=BFGS(sf)
opt.run(.005,steps=500)

print(a2.cell[0,0]/8)
print(a2.cell[2,2]/8)

a2.set_cell([[a2.cell[0,0],0,0],[0,a2.cell[1,1],0],[0,0,a2.cell[2,2]]])

#trajob = Trajectory('Al-traj.traj',mode='w',atoms=a2,properties=['energy','forces'])

#a2.rattle(0.1)
MaxwellBoltzmannDistribution(a2, temperature_K=targTemp)
dyn = NPT(a2,.5*units.fs,temperature_K=targTemp,externalstress=1/0.986923*units.bar,ttime=25*units.fs,pfactor=(75*units.fs)**2*(0.6),mask=[[1,0,0],[0,1,0],[0,0,1]])
#dyn.attach(MDLogger(dyn, a2, 'md.log', header=True, stress=True,peratom=True, mode="a"), interval=1000)
dyn.attach(genLog( a2, dyn, fileName='md-gen.log', printHeader=True), interval=1000)
        
dyn.run(40000)

dyn.attach(runRDF(a2,dyn,radius=15.0,nBin=150,tmpDir=tmpDir), interval=1000)

dyn.run(50000)
