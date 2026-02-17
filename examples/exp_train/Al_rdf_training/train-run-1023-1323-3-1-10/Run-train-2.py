import sys

from matplotlib import pyplot
import numpy as np
import os

import torch

import ase
from ase.build import bulk
from ase.build import make_supercell
from ase.visualize import view

from io import StringIO

import glob

from  FEFFtools import *

from scipy.optimize import minimize

from copy import deepcopy

import hippynn
from hippynn.interfaces.ase_interface import HippynnCalculator

from hippynn.experiment.serialization import load_checkpoint_from_cwd
from hippynn.tools import active_directory
from hippynn.interfaces.ase_interface import HippynnCalculator
from hippynn.experiment.assembly import precompute_pairs,assemble_for_training

from hippynn.graphs import loss

from hippynn.experiment.controllers import RaiseBatchSizeOnPlateau, PatienceController

from hippynn.experiment import SetupParams, setup_and_train

import math

#These are the optimization functions
#note they are lambda functions, so  whatever variables are set to at evaluation is what is used
boltzFac = lambda de : np.exp(-de/(ase.units.kB*targTemp))
lossFn = lambda x : np.linalg.norm(np.dot(simSpec,x)/np.sum(x)-targSpec)**2
lossFnD = lambda x : 2*np.dot((np.dot(simSpec,x)-simSpec.T*np.sum(x)),targSpec*np.sum(x)-np.dot(simSpec,x))/(np.sum(x)**3)
lossFnE = lambda de : lossFn(boltzFac(de))
lossFnED = lambda de : (-boltzFac(de)/(ase.units.kB*targTemp))*lossFnD(boltzFac(de))
lossRegE = lambda de : (1/de.shape[0])*np.linalg.norm(de+Ecur-EML)**2
lossRegED = lambda de : (2/de.shape[0])*(de+Ecur-EML)
lossRegExpE = lambda de : (1/de.shape[0])*np.sum(np.exp(np.abs(de+Ecur-EML)))
lossRegExpED = lambda de : (1/de.shape[0])*np.exp(np.abs(de+Ecur-EML))*np.sign(de+Ecur-EML)
lossRegQuadE = lambda de : (1/de.shape[0])*np.sum((de+Ecur-EML)**4)
lossRegQuadED = lambda de : (4/de.shape[0])*(de+Ecur-EML)**3
regFac = 2000000000.0
lossT = lambda x : (lossFnE(x) + regFac*lossRegQuadE(x))*1e8
lossTD = lambda x : (lossFnED(x) + regFac*lossRegQuadED(x))*1e8

#Inputs: curModelDir, DFTModelDir, newModelDir, mdDirStr, targTempStr
curModelDir = sys.argv[1]
MLModelDir = sys.argv[2] #Presently unused, 
newModelDir = sys.argv[3]
mdDirStr = sys.argv[4] #colon delminiated list of MD directories
targTempStr = sys.argv[5] #colon delminated list of temperatures

mdDirList = mdDirStr.split(':')
tempList = [int(curD) for curD in targTempStr.split(':')]
fractionThresh = 0.99

expData = {}
for temp in [943,1023,1123,1148,1158,1183,1198,1223,1273,1323]:
    expData[str(temp)] = np.genfromtxt('../data/Al_ref_{:s}K_Gr.txt'.format(str(temp)),delimiter=',')
targSpecData = [expData[str(curD)][(expData[str(curD)][:,0]>2.0)*(expData[str(curD)][:,0]<9.0),1] for curD in tempList]
targEnergyData = [expData[str(curD)][(expData[str(curD)][:,0]>2.0)*(expData[str(curD)][:,0]<9.0),0] for curD in tempList]
print('loading database')
#Load giant database
dirNX = []
dirNS = []
dirNSL = []
dirNDB = []
nSub = [len(glob.glob(curD+'/sub-*')) for curD in mdDirList]
print(nSub)
for curDI,curD in enumerate(mdDirList):
    tempNS = []
    tempNDB = []

    for curI in range(nSub[curDI]):
        print('loading spectrum data ' + curD + '/sub-{:d}/'.format(curI))
        (dir0X,dir0S,dir0SL,dir0DB) = loadFEFFdir(curD+'/sub-{:d}/'.format(curI),1.5,10)
        dirNX.append(deepcopy(dir0X))
        tempNS.append(deepcopy(dir0S))
        dirNSL.append(deepcopy(dir0SL))
        dirNDB.append(deepcopy(dir0DB))
        tempNDB.append(deepcopy(dir0DB))
    tempCell = np.concatenate([dir0DB['cell'] for dir0DB in tempNDB])
        
    nAtom = dirNDB[0]['species'].shape[1]    
    
    simEnergy = dirNX[0]
    simSpec = np.concatenate(tempNS)
    
    simSpec = simSpec[:,(simEnergy>2.0)*(simEnergy<9.0)]
    simEnergy = simEnergy[(simEnergy>2.0)*(simEnergy<9.0)]
    simSpec=simSpec.T

    rho = nAtom/(tempCell[:,0]*tempCell[:,1]*tempCell[:,2])
    rho =np.repeat(rho,nAtom)
    simSpec=simSpec/rho[np.newaxis,:]
    simSpec = simSpec/(2*math.pi*.1*simEnergy[:,np.newaxis]**2)
    
    dirNS.append(simSpec)
dirDB={}

dirDB['species'] = np.concatenate([dir0DB['species'] for dir0DB in dirNDB])
dirDB['coordinates'] = np.concatenate([dir0DB['coordinates'] for dir0DB in dirNDB])
nConf = dirDB['species'].shape[0]
#nAtom = dirDB['species'].shape[1]
tempCell = np.concatenate([dir0DB['cell'] for dir0DB in dirNDB])
dirDB['cell'] = np.zeros([nConf,3,3])
dirDB['cell'][:,0,0] = tempCell[:,0]
dirDB['cell'][:,1,1] = tempCell[:,1]
dirDB['cell'][:,2,2] = tempCell[:,2]
dirDB['atomenergies'] = np.zeros([nConf,nAtom,1])
dirDB['energy'] = np.zeros([nConf])
dirDB['force'] = np.zeros([nConf,nAtom,3])
dirDB["energyperatom"] = dirDB["energy"] / nAtom

#Save quantities
for curDI,curD in enumerate(dirNS):
    np.save(newModelDir+'-spec-{:02d}.npy'.format(curDI),curD)
np.savez(newModelDir+'-data.npz',**dirDB)

print('model loaded\n')
with active_directory(curModelDir):
    HIPNNbundleCur = load_checkpoint_from_cwd(map_location="cpu", restore_db=False)
HIPNNmodelCur = HIPNNbundleCur["training_modules"].model
HIPNNmodelCur.double()
predictorCur = hippynn.graphs.Predictor.from_graph(HIPNNmodelCur,model_device='cuda:0')

dirDBtrain = deepcopy(dirDB)
database = hippynn.databases.Database(dirDB,**HIPNNbundleCur["training_modules"].evaluator.db_info,seed=0)
database.split_the_rest('all')
resultsCur =predictorCur.apply_to_database(database,batch_size=1)['all']
Ecur = resultsCur['HEnergy.atom_energies'].cpu().detach().numpy().flatten()
np.save(newModelDir+'-Ecur.npy',Ecur)
Ecur=0

#dE0 = [np.zeros(curD.shape[1]) for curD in specData]

print('Starting Optimize Loop')
continueLoop = True
while continueLoop:
    opts = []
    lossSum = 0
    for curD in range(len(targSpecData)):
        dE0 = np.zeros(dirNS[curD].shape[1])

        #Set Variables used in lambda functions
        Ecur = 0
        EML = 0
        simSpec = dirNS[curD]
        targSpec = targSpecData[curD]
        targTemp = tempList[curD]
        
        opts.append(minimize(lossT,dE0,bounds=[(-.1,.1) for i in range(dE0.shape[0])],method='L-BFGS-B',jac=lossTD))
        lossSum += lossFnE(opts[-1]['x'])
    
    alldE = np.concatenate([curD['x'] for curD in opts])
    allSuc = [curD['success'] for curD in opts]
    
    curWeights = boltzFac(alldE)/np.sum(boltzFac(alldE))
    curNeff = (np.sum(curWeights)**2/np.dot(curWeights,curWeights))/(alldE.shape[0])
    print('Regularization Factor: ' + str(regFac))
    print('Neff:  ' + str(curNeff))
    print('Neff-J: ' + str(np.exp(-1*np.sum(curWeights*np.log(curWeights)))/(alldE.shape[0])))
    print('Min: ' + str(np.min(alldE)))
    print('Max: ' + str(np.max(alldE)))        
    print('Loss: ' + str(lossSum))
    print('Loss-Reg: ' + str(regFac*lossRegQuadE(alldE)))
    sys.stdout.flush()
    
    if all(allSuc) and curNeff>fractionThresh:
        outdE = deepcopy(alldE)
        regFac=regFac/2
    else:
        continueLoop=False

print('Solution Found')
curWeights = boltzFac(outdE)/np.sum(boltzFac(outdE))
print('Regularization Factor: ' + str(regFac*2))        
print('Neff:  ' + str((np.sum(curWeights)**2/np.dot(curWeights,curWeights))/(outdE.shape[0])))
print('Neff-J: ' + str(np.exp(-1*np.sum(curWeights*np.log(curWeights)))))
print('number of good spectra: ' + str(np.sum(outdE>0)))
print('Max x: ' + str(np.max(outdE)))
print('Min x: ' + str(np.min(outdE)))
print('Loss-Reg: ' + str(regFac*lossRegQuadE(outdE)))

np.save(newModelDir+'-sol.npy',outdE)

with hippynn.tools.active_directory(newModelDir):
    with hippynn.tools.log_terminal("training_log.txt", 'wt'):
        
        AEloss = loss.MAELoss.of_node(HIPNNmodelCur.node_from_name('HEnergy.atom_energies'))
        l2_reg = 1e-8*loss.l2reg(HIPNNmodelCur.node_from_name('HIPNN'))
        Tloss = AEloss + l2_reg
        
        validation_losses = {'atomic energy loss':AEloss,'l2 reg':l2_reg,'total loss':Tloss}
        	
        training_modules, db_info = assemble_for_training(Tloss, validation_losses)
        
        databaseTrain = hippynn.databases.Database(dirDBtrain,**db_info,seed=0)
        databaseTrain.arr_dict['atomenergies'] =np.array(resultsCur['atomenergies']+outdE.reshape(nConf,nAtom,1))
        databaseTrain.make_trainvalidtest_split(test_size=0.005,valid_size=0.1)
        
        optimizer = torch.optim.Adam(HIPNNmodelCur.parameters(), lr=1e-4)
        scheduler = RaiseBatchSizeOnPlateau(
            optimizer=optimizer,
            max_batch_size=1,
            patience=10,
            factor=0.5,
            )
        controller = PatienceController(
            optimizer=optimizer,
            scheduler=scheduler,
            batch_size=1,
            eval_batch_size=1,
            max_epochs=5,
            termination_patience=21,
            stopping_key='atomic energy loss',
            )
        experiment_params = SetupParams(controller=controller)
        
        print("Experiment Params:")
        print(experiment_params)
        
        setup_and_train(
                    training_modules=training_modules,
                    database=databaseTrain,
                    setup_params=experiment_params,
                    )
