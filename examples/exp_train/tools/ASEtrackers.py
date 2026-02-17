import ase
import weakref
from ase.parallel import world
from ase.utils import IOContext
from ase import units
import os
import numpy
import numpy as np
from copy import deepcopy
import subprocess
import multiprocessing
import time
import shutil
import sys
from FEFFtools import processFEFFframe
from ase.symbols import symbols2numbers

def writeXYZ(fileID,atoms):
    fileID.write(str(len(atoms)) + '\n')
    if any(atoms.get_pbc()):
        fileID.write('{:9.4f} {:9.4f} {:9.4f}\n'.format(*list(numpy.diag(atoms.get_cell()))))
    else:
        fileID.write('\n')
    c = atoms.get_positions()
    for J, i in zip(atoms,c):
        fileID.write('{:s} {:9.4f} {:9.4f} {:9.4f}\n'.format(J.symbol.ljust(3),i[0],i[1],i[2]))


class time_object():
    def __init__(self,init_time,timestep=1.0):
        self.ctime = init_time
        self.timestep=timestep
        
    def update_time(self,update_time):
        self.ctime = update_time
        
    def get_number_of_steps(self):
        return(self.ctime)
        
    def get_time(self):
        return(self.timestep*self.ctime)
            
        
class runFEFFLeg(IOContext):
    def __init__(self,atoms,d,radius=5.0,runFEFF=True,
            FEFFhead=" HOLE      1   1.0   * FYI: (Zn K edge @ 9659 eV, 2nd number is S0^2)\n CONTROL   1      1     1     1\n PRINT     1      0     0     0\n\n RMAX      5.0\n\n POTENTIALS\n     0     30     Zn        \n     1     30     Zn   \n\n   ATOMS\n",feffpath='/usr/projects/ml4chem/Programs/FEFF/feff85L',
            tmpDir='./'):
        self.atoms = atoms
        if hasattr(d, "get_time"):
            self.d = weakref.proxy(d)
        #elif isinstance(d,time_object):
        #    pass
        else:
            self.d = None
        self.radius=radius
        self.runFEFF=runFEFF
        self.FEFFhead=FEFFhead
        self.natoms = self.atoms.get_atomic_numbers().shape[0]
        self.nl = ase.neighborlist.NeighborList(self.radius,skin=0.0,sorted=False,self_interaction=True,bothways=True,primitive=ase.neighborlist.NewPrimitiveNeighborList)
        #self.nl = ase.neighborlist.build_neighbor_list(self.atoms,cutoffs=ase.neighborlist.natural_cutoffs(self.atoms,mult=self.radius))
        self.feffpath = feffpath
        self.tmpDir = tmpDir
    
    def __del__(self):
        self.close()
    
    def __call__(self):
        ctime = self.d.get_number_of_steps()
        self.nl.update(self.atoms)
        timedir = self.tmpDir + "/time-{:08d}".format(ctime)
        os.mkdir(timedir)
        xyzF = open(timedir + '/position.xyz','w')
        writeXYZ(xyzF,self.atoms)
        xyzF.close()
        for i in range(self.natoms):
            tempAtoms = deepcopy(self.atoms)
            tempAtoms.translate(numpy.diag(tempAtoms.get_cell())/2 - tempAtoms.get_positions()[i])
            tempAtoms.wrap()
            curneigh = (self.nl.get_neighbors(i))[0]
            tempPos = (tempAtoms.get_positions()[curneigh])
        
            curdir = timedir + '/atom-{:06d}'.format(i)
            os.mkdir(curdir)
            file = open(curdir + '/feff.inp','w')
            file.write(self.FEFFhead)
            #iterate through atoms to generate FEFF input. Center atom is last, so reverse to put first
            for writei in range(tempPos.shape[0]):
                file.write('  {:12.8f}   {:12.8f}   {:12.8f}   {:2d} \n'.format(tempPos[writei,0],tempPos[writei,1],tempPos[writei,2],(curneigh[writei]!=i)*1))
            file.close()
            if self.runFEFF:
                subprocess.Popen(self.feffpath + ";cp chi.dat ../atom-{ind:06d}-chi.dat; cd ../;rm -r atom-{ind:06d}".format(ind=i),shell=True,cwd=curdir)


class runFEFF(IOContext):
    def __init__(self,atoms,d,radius=5.0,runFEFF=True,
            FEFFhead=" HOLE      1   1.0   * FYI: (Zn K edge @ 9659 eV, 2nd number is S0^2)\n CONTROL   1      1     1     1\n PRINT     1      0     0     0\n\n RMAX      5.0\n\n POTENTIALS\n",
            feffpath='/usr/projects/ml4chem/Programs/FEFF/feff85L',
            tmpDir='./',rmDir=False,eMin=2,eMax=8,nPar=4):
        self.atoms = atoms
        if hasattr(d, "get_time"):
            self.d = weakref.proxy(d)
        #elif isinstance(d,time_object):
        #    pass
        else:
            self.d = None
        self.radius=radius
        self.runFEFF=runFEFF
        self.rmDir=rmDir
        self.FEFFhead=FEFFhead
        self.natoms = self.atoms.get_atomic_numbers().shape[0]
        self.nl = ase.neighborlist.NeighborList(self.radius,skin=0.0,sorted=False,self_interaction=False,bothways=True,primitive=ase.neighborlist.NewPrimitiveNeighborList)
        #self.nl = ase.neighborlist.build_neighbor_list(self.atoms,cutoffs=ase.neighborlist.natural_cutoffs(self.atoms,mult=self.radius))
        self.feffpath = feffpath
        self.tmpDir = tmpDir
        self.processList = []
        self.eMin = eMin
        self.eMax = eMax
        self.nPar=nPar
    
    def __del__(self):
        #Wait for jobs to finish
        liveList = [not proc.is_alive() for proc in self.processList]
        while not all(liveList):
            time.sleep(2)
            liveList = [not proc.is_alive() for proc in self.processList]
        self.close()
    
    def __call__(self):
        #To prevent overloading, ensure previous processes is done
        liveList = [proc.is_alive() for proc in self.processList]
        if len(self.processList)>0:
            while np.sum(liveList)>=self.nPar:
                time.sleep(2)
                liveList = [proc.is_alive() for proc in self.processList]
        self.processList.append(multiprocessing.Process(target=self.multiFunction))
        self.processList[-1].start()
    
    def multiFunction(self):
        ctime = self.d.get_number_of_steps()
        self.nl.update(self.atoms)
        timedir = self.tmpDir + "/time-{:08d}".format(ctime)
        os.mkdir(timedir)
        xyzF = open(timedir + '/position.xyz','w')
        writeXYZ(xyzF,self.atoms)
        xyzF.close()
        #Set of all elements in system for potentials
        potSet = set(self.atoms.get_chemical_symbols())
        for i in range(self.natoms):
            tempPos = ase.neighborlist.mic(self.atoms[self.nl.get_neighbors(i)[0]].get_positions() - self.atoms[i].position,self.atoms.get_cell(),self.atoms.get_pbc())
            #tempElem contains the chemical symbols for all neighbors
            tempElem = self.atoms[self.nl.get_neighbors(i)[0]].get_chemical_symbols()
            #tempAtoms = deepcopy(self.atoms)
            #tempAtoms.translate(numpy.diag(tempAtoms.get_cell())/2 - tempAtoms.get_positions()[i])
            #tempAtoms.wrap()
            #curneigh = (self.nl.get_neighbors(i))[0]
            #tempPos = (tempAtoms.get_positions()[curneigh])
        
            curdir = timedir + '/atom-{:06d}'.format(i)
            os.mkdir(curdir)
            file = open(curdir + '/feff.inp','w')
            file.write(self.FEFFhead)
            #Here is where we need to use potSet to write potentials
            #First write scattering atom (with 0 tag appended), then write all atoms
            file.write('0     '+str(self.atoms[i].number)+'     '+str(self.atoms[i].symbol)+'0       \n')
            #tempPot tracks which element is assigned to which potential
            tempPot = {}
            #Print each element in the potential header, index starts at 1 because non-scattering atoms
            for ipot, Zpot in enumerate(potSet): 
                file.write(str(ipot+1)+'     '+str(symbols2numbers(Zpot)[0])+'     '+Zpot+'       \n')
                tempPot[Zpot] = ipot+1

            file.write('\n   ATOMS\n')
            #tempPos is relative position to scattering atom
            file.write('  {:12.8f}   {:12.8f}   {:12.8f}   {:2d} \n'.format(0,0,0,0))
            for writei in range(tempPos.shape[0]):
                file.write('  {:12.8f}   {:12.8f}   {:12.8f}   {:2d} \n'.format(tempPos[writei,0],tempPos[writei,1],tempPos[writei,2],tempPot[tempElem[writei]]))#This needs to be modified to correspond with correct atom type
            file.close()
            if self.runFEFF:
                #subprocess.Popen(self.feffpath + ";cp chi.dat ../atom-{ind:06d}-chi.dat; cd ../;rm -r atom-{ind:06d}".format(ind=i),shell=True,cwd=curdir,stdout=subprocess.PIPE)
                subprocess.call(self.feffpath,shell=False,cwd=curdir,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                shutil.copy(curdir+'/chi.dat',timedir+'/atom-{ind:06d}-chi.dat'.format(ind=i))
                if self.rmDir:
                    shutil.rmtree(curdir)
        processFEFFframe(timedir,self.eMin,self.eMax)
        
class runRDF(IOContext):
    def __init__(self,atoms,d,radius=15.0,shift=None,nBin=150,tmpDir='./',nPar=4,saveAtomicRDF=True,saveAverageRDF=True):
        #nBin 
        self.atoms = atoms
        if hasattr(d, "get_time"):
            self.d = weakref.proxy(d)
        #elif isinstance(d,time_object):
        #    self.d = d
        else:
            self.d = None
        self.radius=radius
        self.natoms = self.atoms.get_atomic_numbers().shape[0]
        self.nBin = nBin
        self.tmpDir = tmpDir
        self.processList = []
        #If shift is 1/2 bin size 'x' represents a mid point of the binning
        #If shift is 0, x represents a maximum to the binned points
        if shift is None:
            self.shift = self.radius/(2*self.nBin)
        else:
            self.shift = shift
        self.nl = ase.neighborlist.NeighborList(self.radius+self.shift,skin=0.0,sorted=False,self_interaction=False,bothways=True,primitive=ase.neighborlist.NewPrimitiveNeighborList)
        self.nPar=nPar
        self.saveAverageRDF=saveAverageRDF
        self.saveAtomicRDF=saveAtomicRDF
            
    
    def __del__(self):
        #Wait for jobs to finish
        liveList = [not proc.is_alive() for proc in self.processList]
        while not all(liveList):
            time.sleep(2)
            liveList = [not proc.is_alive() for proc in self.processList]
        self.close()
    
    def __call__(self):
        #To prevent overloading, ensure previous processes is done
        liveList = [proc.is_alive() for proc in self.processList]
        if len(self.processList)>0:
            while np.sum(liveList)>=self.nPar:
                time.sleep(2)
                liveList = [proc.is_alive() for proc in self.processList]
        #ctime = self.d.get_number_of_steps()
        #timedir = self.tmpDir + "/time-{:08d}".format(ctime)
        #os.mkdir(timedir)
        #xyzF = open(timedir + '/position4.xyz','w')
        #writeXYZ(xyzF,self.atoms)
        #xyzF.close()
        self.processList.append(multiprocessing.Process(target=self.multiFunction))
        self.processList[-1].start()
    
    def multiFunction(self):
        ctime = self.d.get_number_of_steps()
        self.nl.update(self.atoms)
        timedir = self.tmpDir + "/time-{:08d}".format(ctime)
        os.mkdir(timedir)
        xyzF = open(timedir + '/position.xyz','w')
        writeXYZ(xyzF,self.atoms)
        xyzF.close()
        saveDict={}
        saveDict['x']=numpy.arange(self.radius/self.nBin,self.radius+self.radius/self.nBin,self.radius/self.nBin)
        saveDict['y']=numpy.zeros([self.natoms,self.nBin],dtype=int)
        for i in range(self.natoms):
            tempDist = numpy.linalg.norm(ase.neighborlist.mic(self.atoms[self.nl.get_neighbors(i)[0]].get_positions() - self.atoms[i].position,self.atoms.get_cell(),self.atoms.get_pbc()),axis=1) - self.shift
            numpy.add.at(saveDict['y'][i],(tempDist[tempDist<(self.radius)]*self.nBin/self.radius).astype(int),1)
        if self.saveAverageRDF:
            saveDict['yAve'] = numpy.mean(saveDict['y'],axis=0)
        if not self.saveAtomicRDF:
            del saveDict['y']
        numpy.savez(timedir+'/atom-allSpec.npz',**saveDict)
        #distMat = self.atoms.get_all_distances(mic=True)+np.eye(self.natoms)*np.max(self.atoms.get_cell())
        #numpy.save(timedir+'/distMat.npy',distMat)

class genLog(IOContext):
    def __init__(self,atoms,d,fileName='genLog.txt',printHeader=True,printExtra=False):
        self.atoms = atoms
        self.nAtoms = len(atoms)
        if hasattr(d, "get_time"):
            self.d = weakref.proxy(d)
        #elif isinstance(d,time_object):
        #    self.d = d
        else:
            self.d = None
        self.fID = open(fileName,'w')
        if printHeader:
            self.fID.write('Time (ps) | Temperature |                      Stress                     |      Cell Size       | Min Dist |    Pot En    |    Kin En    |  Clock  \n')
        self.fmt = '%9.3f | %11.3f | %8.1f %8.1f %8.1f %6.3f %6.3f %6.3f | %6.2f %6.2f %6.2f | %8.3f | %12.3f | %12.3f | %s  \n'
        self.printExtra=printExtra

    def __del__(self):
        self.fID.close()
        self.close()

    def __call__(self):
        outDat = (self.d.get_time() / (1000 * units.fs),self.atoms.get_temperature(),)
        outDat += tuple(self.atoms.get_stress(include_ideal_gas=True) / units.GPa)
        #outDat += tuple(np.diag(self.atoms.get_cell()))
        outDat += (self.atoms.cell[0,0],self.atoms.cell[1,1],self.atoms.cell[2,2],)
        outDat += (np.min(self.atoms.get_all_distances(mic=True)+np.eye(self.nAtoms)*np.max(self.atoms.get_cell())),)
        outDat += (self.atoms.get_potential_energy(),)
        outDat += (self.atoms.get_kinetic_energy(),)
        outDat += (time.strftime("%H:%M:%S"),)
        self.fID.write(self.fmt % outDat)
        self.fID.flush()
        if self.printExtra:
            ctime = self.d.get_number_of_steps()
            xyzF = open("time-{:08d}-position.xyz".format(ctime),'w')
            writeXYZ(xyzF,self.atoms)
            xyzF.close()
            distMat = self.atoms.get_all_distances(mic=True)+np.eye(self.nAtoms)*np.max(self.atoms.get_cell())
            targPair = distMat.argmin()
            xind = int(targPair/self.nAtoms)
            yind = targPair%self.nAtoms
            print('step: {:08d}'.format(ctime))
            print(xind)
            print(yind)
            print(self.atoms.get_cell())
            print(self.atoms.get_positions()[xind])
            print(self.atoms.get_positions()[yind])
            print(distMat[xind,yind])
            sys.stdout.flush()
            
            numpy.save('time-{:08d}-distMat.npy'.format(ctime),distMat)
        

#pseudocode        
#import multiprocessing as mp
#from time import sleep
#
#class A(object):
#    def __init__(self, *args, **kwargs):
#        # do other stuff
#        pass
#
#    def do_something(self, i):
#        sleep(0.2)
#        print('%s * %s = %s' % (i, i, i*i))
#
#    def run(self):
#        processes = []
#
#        for i in range(1000):
#            p = mp.Process(target=self.do_something, args=(i,))
#            processes.append(p)
#
#        [x.start() for x in processes]
#
#
#if __name__ == '__main__':
#    a = A()
#    a.run()

def storeenergy(a, d, mdcrd, traj):  # store a reference to atoms in the definition.
    """Function to print the potential, kinetic and total energy."""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)

    traj.write(str(d.get_number_of_steps()) + ' ' + str(ekin / (1.5 * units.kB)) + ' ' + str(epot) + ' ' + str(ekin) + ' ' + str(epot+ekin) + ' ' + '\n')
    
    writeXYZ(mdcrd,a)
    
    print('Step: %d Energy per atom: Epot = %.3f  Ekin = %.3f (T=%.3fK)  '
          'Etot = %.6f Vol = %.6f' % (d.get_number_of_steps(), epot, ekin, ekin / (1.5 * units.kB), epot + ekin, a.get_volume()))
