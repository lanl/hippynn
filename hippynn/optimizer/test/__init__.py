import torch, os

cwd = os.path.dirname(__file__)

# Below are some test cases for the optimizer. 

# QM9b1 contain four conformers from the QM9 dataset. 
# they were all padded to size 29, they are already equilibrium conformers

# it seems this relative path does not work
QM9b1 = torch.load(os.path.join(cwd, 'qm9b1.pt'))

# C2H6 contain 15 ethane conformers with C-C bond elongated.
# 'E'/'F' is the B97-3c energy/forces calculated by ORCA4
C2H6 = torch.load(os.path.join(cwd, 'c2h6.pt'))

# to test the numerical stability of the optimizer, 
# I add meaningless 0 paddings to C2H6 to form this C2H6_padded
C2H6_padded = {
    'Z': torch.cat((C2H6['Z'].clone(), torch.zeros(C2H6['Z'].shape[0], 2)), dim=1), 
    'R': torch.cat((C2H6['R'].clone(), torch.zeros(C2H6['R'].shape[0], 2, 3)), dim=1),
    'E': C2H6['E'].clone(),
    'F': torch.cat((C2H6['F'].clone(), torch.zeros(C2H6['F'].shape[0], 2, 3)), dim=1),
}