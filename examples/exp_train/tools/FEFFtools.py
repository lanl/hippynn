from matplotlib import pyplot
import numpy as np
import os

import ase
from ase.build import bulk
from ase.build import make_supercell
from ase.visualize import view
from ase.md import analysis
from ase.io.trajectory import Trajectory

from io import StringIO

import glob

def readChi(fName):
    #headerNum = open(fName,'r').read().count('#') - 1
    comput = np.genfromtxt(fName)#,skip_header=headerNum)
    return(np.array([comput[:,0],comput[:,1]]))#*comput[:,0]**2]))
    
def compileAveChi(FNameList):
    specList = []
    minLength = 1000000
    for curSpec in FNameList:
        (xList,yList) = readChi(curSpec)
        specList.append(yList)
        if minLength>yList.shape[0]:
            minLength = yList.shape[0]
    specList = np.array([i[:minLength] for i in specList])
    AveSpec = np.mean(specList,axis=0)
    xList = xList[:minLength]
    #specList = np.array(specList)
    return(xList,AveSpec)
    
def processFEFFframe(curT,eMin,eMax):
    curAt = ase.io.read(curT + '/position.xyz')
    curF = open(curT + '/position.xyz','r')
    curF.readline()
    curAt.set_cell( [float(i) for i in curF.readline().split()])
    curF.close()
    nAtom = len(curAt)
    initF = True
    
    try: 
        #This directory has already been processed
        curArc = np.load(curT + '/atom-allSpec.npz')
    except:
       # This directory has not been processed,process directory
        for curChiI in range(nAtom):
            curChiF = curT + '/atom-{:06d}-chi.dat'.format(curChiI)
            if os.path.isfile(curChiF):
                (xList,yList) = readChi(curChiF)
                yList = yList[(xList>eMin)*(xList<eMax)]
                xList = xList[(xList>eMin)*(xList<eMax)]
                if initF:
                    initF=False
                    curArc = {'x':xList,'y':np.zeros([nAtom,len(xList)])}
                curArc['y'][curChiI] = yList
        np.savez(curT+'/atom-allSpec.npz',**curArc)
    return(curAt,curArc)
    

def loadFEFFdir(Tdir,eMin,eMax):
    initF=True
    timeDirList = glob.glob(Tdir + '/time-*')
    timeDirList.sort()
    nTime = len(timeDirList)
    for curTi,curT in enumerate(timeDirList):
        (curAt,curArc) = processFEFFframe(curT,eMin,eMax)
        if initF:
            initF=False
            nAtom = len(curAt)
            positions = np.zeros([nTime,nAtom,3])
            species = np.zeros([nTime,nAtom],dtype='int64')
            cell = -1*np.ones([nTime,3])
            xList = curArc['x']
            specList = np.zeros([nTime*nAtom,curArc['y'].shape[1]])
        positions[curTi] = curAt.get_positions()
        species[curTi] = curAt.get_atomic_numbers()
        cell[curTi] = np.diagonal(curAt.get_cell())
        specList[curTi*nAtom:(curTi+1)*nAtom] = curArc['y']
                
    arr_dict={'species':species,'coordinates':positions,'cell':cell}
    return(xList,specList,None,arr_dict)
        
def NcomputeDeriv(lambdaF,x,ind,disp=.001):
    tempP = x*1
    tempM = x*1
    tempP[ind] = x[ind] + disp
    tempM[ind] = x[ind] - disp
    return((lambdaF(tempP)-lambdaF(tempM))/(2*disp))
    
def computDiffTraj(trajFile,timestep):
    loadTraj = Trajectory(trajFile)
    A = analysis.DiffusionCoefficient(traj=loadTraj,timestep=timestep)
    D, std = A.get_diffusion_coefficients()
    np.savetxt(trajFile+'-diff.txt',np.array([D,std]).T,fmt='%16.8e',header="diffusion   stddv")

#lossFnNS = lambda x : np.linalg.norm(np.dot(simSpec,x)-targSpec)**2
#lossFnDNS = lambda x : 2*np.dot(simSpec.T,np.dot(simSpec,x)-targSpec)
#lossFn = lambda x : np.linalg.norm(np.dot(simSpec,x)/np.sum(x)-targSpec)**2
#lossFnD = lambda x : 2*np.dot((np.dot(simSpec,x)-simSpec.T*np.sum(x)),targSpec*np.sum(x)-np.dot(simSpec,x))/(np.sum(x)**3)
	
