from typing import Any
import torch
from torch.nn.functional import normalize


class GeometryOptimizer():

    def __init__(self, coords, max_steps=100, logfile=False):
        self.coords = coords
        self.max_steps = max_steps
        self.current_step = 0
        self.logfile = logfile

        self.stop_signal = False


    def __call__(self, *args: Any, **kwds: Any) -> Any:
        # __call__ call self.step() method if the current step is less than the max_steps
        self.step(*args, **kwds)
        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.stop_signal = True
    
    def _reset(self, coords):
        self.coords = coords
        self.current_step = 0
        self.stop_signal = False

    def _log(self, message=''):
        if self.logfile:
            with open(self.logfile, 'a') as f:
                f.write('step %d :\n'%(self.current_step))
                f.write(message)
                f.write('\n')
                #f.write('coordinates:\n')
                #f.write(str(self.coords))
                #f.write('\n')

    @staticmethod
    def fmax_criteria(forces, fmax=0.05):
        # not all optimizers use fmax as a criteria to stop
        # but it is a common criteria
        # forces are in the shape of (batch_size, n_atoms, 3)
        return forces.flatten(-2,-1).norm(p=2, dim=-1) < fmax
    
    @staticmethod
    def duq(t):
        # this is useful for batch calculation
        # (n,) -> (n,1,1)
        return t.unsqueeze(-1).unsqueeze(-1)
        

class BatchFIRE(GeometryOptimizer):
    # FIRE algorithm for batch of coordinates, forces will be input to step() function
    # assuming the input coordinates are in the shape of (batch_size, n_atoms, 3)
    # set dt for each molecule individually

    def __init__(self, coords=None, max_steps=100, dt=0.1, maxstep=0.2, dt_max=1.0, N_min=5, f_inc=1.1, f_dec=0.5, a_start=0.1, f_alpha=0.99, fmax=0.0, logfile=False):
        super().__init__(coords, max_steps)
        
        self.dt_start = dt   # save for reset
        self.maxstep = maxstep
        self.dt_max = dt_max
        self.N_min = N_min
        self.f_inc = f_inc
        self.f_dec = f_dec
        self.a_start = a_start
        self.f_alpha = f_alpha
        self.fmax = fmax
        self.logfile = logfile

        # enable initialization without coords
        if isinstance(coords, torch.Tensor):
            self.batch_size = coords.shape[0]
            self.dt = torch.ones(self.batch_size) * dt
            self.v = torch.zeros_like(self.coords)
            self.a = torch.ones(self.batch_size) * a_start
            # Nsteps is the number of steps where P became positive
            # NOT number of times step() function is called!
            self.Nsteps = torch.zeros_like(self.a)
        

    def step(self, forces):
        # forces: (batch_size, n_atoms, 3)
        
        # force projection on velocity
        ###TODO: from the FIRE paper, the projection seems to be per-atom, not per-molecule
        # but the implementation in ASE is per-molecule, not sure why they did this but I will follow their implementation

        fmax_mask = self.fmax_criteria(forces)
        # if all molecules have forces smaller than fmax, stop the optimization
        if fmax_mask.all():
            self.stop_signal = True
            return

        f_dot_v = torch.sum(forces * self.v, dim=(1,2))

        # update velocity first time
        #print(self.duq((1 - self.a)).shape, self.v.shape, self.duq(self.a).shape, normalize(forces, p=2, dim=2).shape, torch.norm(self.v, dim=(1,2)).shape)
        self.v = self.duq((1 - self.a)) * self.v \
                + self.duq(self.a) * normalize(forces, p=2, dim=2) * self.duq(torch.norm(self.v, dim=(1,2)))
    
        # increase dt where P is positive and N is larger than N_min
        positiveP_mask = (f_dot_v > 0)
        Nsteps_mask = (self.Nsteps > self.N_min)
        mask = positiveP_mask & Nsteps_mask
        self.dt = torch.clamp( (mask * self.f_inc + ~mask) * self.dt, max=self.dt_max)
        # decrease a where P is positive and N is larger than N_min
        self.a = self.a * self.f_alpha * mask + self.a * ~mask

        self.Nsteps += positiveP_mask

        # decrease dt where P is not positive
        self.dt = ( ~positiveP_mask * self.f_dec + positiveP_mask) * self.dt
        # set velocity to zero where P is not positive
        self.v = self.v * self.duq(positiveP_mask)
        # reset a to a_start where P is not positive
        self.a = self.a_start * ~positiveP_mask + self.a * positiveP_mask
        # reset Nsteps to zero where P is not positive
        self.Nsteps = self.Nsteps * positiveP_mask

        # update coordinates

        self.v += self.duq(self.dt) * forces
        dr = self.duq(self.dt) * self.v
        dr *= self.duq((self.maxstep / torch.norm(dr, dim=(1,2))).clamp(max=1.0))

        self.coords = self.coords + dr

    def reset(self, coords):
        self._reset(coords)
        self.batch_size = coords.shape[0]
        self.dt = torch.ones(self.batch_size) * self.dt_start
        self.v = torch.zeros_like(self.coords)
        self.a = torch.ones(self.batch_size) * self.a_start
        self.Nsteps = torch.zeros_like(self.a)
        
    def log(self, extra_message=''):
        message = 'dt: %s\n'%(str(self.dt))
        message += 'v: %s\n'%(str(self.v))
        message += 'a: %s\n'%(str(self.a))
        message += 'Nsteps: %s\n'%(str(self.Nsteps))
        message += extra_message
        print(message)
        self._log(message)