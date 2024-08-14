import torch

# Below are some test cases for the optimizer.

from .test_configs import c2h6_config as C2H6, qm9b1_config as QM9b1

# C2H6 contain 15 ethane conformers with C-C bond elongated.
# 'E'/'F' is the B97-3c energy/forces calculated by ORCA4

# QM9b1 contain four conformers from the QM9 dataset.
# they were all padded to size 29, they are already equilibrium conformers

# To test the numerical stability of the optimizer,
# I add meaningless 0 paddings to C2H6 to form this C2H6_padded
C2H6_padded = {
    'Z': torch.cat((C2H6['Z'].clone(), torch.zeros(C2H6['Z'].shape[0], 2,dtype=torch.int64)), dim=1), 
    'R': torch.cat((C2H6['R'].clone(), torch.zeros(C2H6['R'].shape[0], 2, 3)), dim=1),
    'E': C2H6['E'].clone(),
    'F': torch.cat((C2H6['F'].clone(), torch.zeros(C2H6['F'].shape[0], 2, 3)), dim=1),
}
