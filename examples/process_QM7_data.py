'''
This script extracts data from file the file qm7.mat, which can be 
downloaded from http://quantum-machine.org/datasets/, and saves the data
to .npy files formatted as Numpy arrays. A few adjustments are made so that
the .npy files are ready to be read by the example scripts for hippynn.

BEFORE RUNNING: Copy this file to a folder datasets/qm7_processed/, where datasets/ 
is at the same level as hippynn/. Ensure that the file qm7.mat is also present in 
that folder. Execute this script from that folder. 
'''

import scipy.io
import numpy as np

dataname = "qm7"
matobj = scipy.io.loadmat(dataname+'.mat')
arrays = []

for key, item in matobj.items():
    print("\nKey:", key)
    if type(item) is np.ndarray:
        # A few alterations are needed
        if key == 'T':
            item = item.T
        if key == 'Z':
            item = item.astype('int64')

        # Save array and print info
        np.save("data-"+dataname+key+'.npy', item)
        print("Shape: ", item.shape)
        print("Dtype: ", item.dtype)
        arrays.append(key)
    else:
        print("Item:\n", item)

# Check that everything was correctly saved
print("\nChecking saved files:")
for key in arrays:
    print("Loading", key)
    x = np.load("data-"+dataname+key+'.npy')
    print("Shape:", x.shape, "\n")
    
            
