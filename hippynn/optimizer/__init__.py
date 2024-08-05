from ..graphs import Predictor
from .BatchOptimizer import *
from .utils import *
from .test import *
from ase import Atoms

class HippyNNBatchOptimizer():
    def __init__(self, model, optimizer=BatchBFGSv3(), dump_traj=False, device='cpu', relocate_optimizer=False, force_key='F', force_sign=1.0):
        
        # assume model is a hippynn model, and there will be 'F' in its output
        ###TODO: Check if this is the best way to build a hipnn predictor
        self.device = device
        self.predictor = Predictor.from_graph(model, return_device=self.device, model_device=self.device)
        self.optimizer = optimizer
        if relocate_optimizer:
            self.optimizer.device = self.device
        self.dump_traj = dump_traj
        self.masks = None
        self.force_key = force_key
        self.force_sign = force_sign

        if dump_traj:
            print("Dump batch trajectories every step require copying \n \
                  and moving tensors to cpu, which is slow!")

    def __call__(self, Z, R, padding_number=0, prefix_list=None):

        self.masks = None

        self.optimizer.reset(coords=R, numbers=Z)
        
        while not self.optimizer.stop_signal:
            # dump a step before optimization to write the initial coordinates
            # this also enable wrapping the io part into a single function
            ###TODO: consider separate dump_step and dump_traj into two operations
            # dump_step write a single step to $batch_size xyz files 
            if self.dump_traj:
                self.dump_a_step(Z, self.optimizer.coords, padding_number, prefix_list)
            ret = self.predictor(Z=Z, R=self.optimizer.coords)
            # forces = -ret['Grad']
            # set this tunable, although I suggest to simply have 'F' in the model output
            forces = ret[self.force_key] * self.force_sign
            self.optimizer.coords = self.optimizer.coords.detach()
            self.optimizer(forces)
            
        # also return the model output from the last step
        # because the energies in it might be useful
        return self.optimizer.coords, ret

    def dump_a_step(self, numbers, coords, padding_number=0, prefix_list=None):
        N = coords.shape[0]
        if self.masks is None:
            self.debatched_numbers, debatched_coords, self.masks = debatch(numbers, coords, padding_number, return_mask=True)
        else:
            debatched_coords = debatch_coords(coords, self.masks)

        prefix_list = prefix_list or list(range(N))
        for n, c, prefix in zip(self.debatched_numbers, debatched_coords, prefix_list):
            atoms = Atoms(numbers=n.cpu(), positions=c.cpu())
            atoms.write('%s_traj.xyz'%(prefix), append=True)