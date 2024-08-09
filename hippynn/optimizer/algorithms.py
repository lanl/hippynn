"""

Abstract optimizers that could be used with other force functions.

"""
import torch
from torch.nn.functional import normalize


class GeometryOptimizer:
    def __init__(self, coords, max_steps=100, logfile=False, device="cpu"):
        self.coords = coords
        self.max_steps = max_steps
        self.current_step = 0
        self.logfile = logfile
        self.device = device

        self.stop_signal = False

    def __call__(self, *args, **kwds):
        # __call__ call self.step() method if the current step is less than the max_steps
        self.step(*args, **kwds)
        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.stop_signal = True

    def _reset(self, coords):
        self.coords = coords
        self.current_step = 0
        self.stop_signal = False

    def _log(self, message=""):
        if self.logfile:
            with open(self.logfile, "a") as f:
                f.write("step %d :\n" % (self.current_step))
                f.write(message)
                f.write("\n")
                # f.write('coordinates:\n')
                # f.write(str(self.coords))
                # f.write('\n')

    @staticmethod
    def fmax_criteria(forces, fmax=0.05):
        # not all optimizers use fmax as a criteria to stop
        # but it is a common criteria
        # forces are in the shape of (batch_size, n_atoms, 3)
        return forces.flatten(-2, -1).norm(p=2, dim=-1) < fmax

    @staticmethod
    def duq(t):
        # this is useful for batch calculation
        # (n,) -> (n,1,1)
        return t.unsqueeze(-1).unsqueeze(-1)


class FIRE(GeometryOptimizer):
    # FIRE algorithm for batch of coordinates, forces will be input to step() function
    # assuming the input coordinates are in the shape of (batch_size, n_atoms, 3)
    # set dt for each molecule individually

    def __init__(
        self,
        coords=None,
        numbers=None,
        max_steps=100,
        dt=0.1,
        maxstep=0.2,
        dt_max=1.0,
        N_min=5,
        f_inc=1.1,
        f_dec=0.5,
        a_start=0.1,
        f_alpha=0.99,
        fmax=0.0,
        logfile=False,
        device="cpu",
    ):
        super().__init__(coords, max_steps, logfile, device)

        self.dt_start = dt  # save for reset
        self.maxstep = maxstep
        self.dt_max = dt_max
        self.N_min = N_min
        self.f_inc = f_inc
        self.f_dec = f_dec
        self.a_start = a_start
        self.f_alpha = f_alpha
        self.fmax = fmax

        # enable initialization without coords
        if isinstance(coords, torch.Tensor):
            self.reset(coords)

    def reset(self, coords, numbers=None):
        self._reset(coords)
        self.batch_size = coords.shape[0]
        self.dt = torch.ones(self.batch_size, device=self.device) * self.dt_start
        self.v = torch.zeros_like(self.coords, device=self.device)
        self.a = torch.ones(self.batch_size, device=self.device) * self.a_start
        # Nsteps is the number of steps where P became positive
        # NOT number of times step() function is called!
        self.Nsteps = torch.zeros_like(self.a, device=self.device)

    def step(self, forces):
        # forces: (batch_size, n_atoms, 3)

        # force projection on velocity
        ###TODO: from the FIRE paper, the projection seems to be per-atom, not per-molecule
        # but the implementation in ASE is per-molecule, not sure why they did this but I will follow their implementation

        fmax_mask = self.fmax_criteria(forces, self.fmax)
        # if all molecules have forces smaller than fmax, stop the optimization
        if fmax_mask.all():
            self.stop_signal = True
            return

        f_dot_v = torch.sum(forces * self.v, dim=(1, 2))

        # update velocity first time
        # print(self.duq((1 - self.a)).shape, self.v.shape, self.duq(self.a).shape, normalize(forces, p=2, dim=2).shape, torch.norm(self.v, dim=(1,2)).shape)
        self.v = self.duq((1 - self.a)) * self.v + self.duq(self.a) * normalize(forces, p=2, dim=2) * self.duq(
            torch.norm(self.v, dim=(1, 2))
        )

        # increase dt where P is positive and N is larger than N_min
        positiveP_mask = f_dot_v > 0
        Nsteps_mask = self.Nsteps > self.N_min
        mask = positiveP_mask & Nsteps_mask
        self.dt = torch.clamp((mask * self.f_inc + ~mask) * self.dt, max=self.dt_max)
        # decrease a where P is positive and N is larger than N_min
        self.a = self.a * self.f_alpha * mask + self.a * ~mask

        self.Nsteps += positiveP_mask

        # decrease dt where P is not positive
        self.dt = (~positiveP_mask * self.f_dec + positiveP_mask) * self.dt
        # set velocity to zero where P is not positive
        self.v = self.v * self.duq(positiveP_mask)
        # reset a to a_start where P is not positive
        self.a = self.a_start * ~positiveP_mask + self.a * positiveP_mask
        # reset Nsteps to zero where P is not positive
        self.Nsteps = self.Nsteps * positiveP_mask

        # update coordinates

        self.v += self.duq(self.dt) * forces
        dr = self.duq(self.dt) * self.v
        dr *= self.duq((self.maxstep / torch.norm(dr, dim=(1, 2))).clamp(max=1.0))

        self.coords = self.coords + dr

    def log(self, extra_message=""):
        message = "dt: %s\n" % (str(self.dt))
        message += "v: %s\n" % (str(self.v))
        message += "a: %s\n" % (str(self.a))
        message += "Nsteps: %s\n" % (str(self.Nsteps))
        message += extra_message
        print(message)
        self._log(message)


class NewtonRaphson(GeometryOptimizer):
    def __init__(self, coords=None, numbers=None, max_steps=100, etol=1e-3, fmax=0.05, logfile=False, device="cpu"):
        super().__init__(coords, max_steps, logfile, device)
        self.etol = etol
        self.fmax = fmax
        if isinstance(coords, torch.Tensor):
            self.reset(coords)

    def reset(self, coords, numbers=None):
        self._reset(coords)
        self.batch_size = coords.shape[0]
        self.last_e = torch.ones(self.batch_size, device=self.device) * torch.inf

    def step(self, forces, energies):
        # assuming energies is in the shape of (batch_size, 1)

        fmax_mask = self.fmax_criteria(forces, self.fmax)
        # if all molecules have forces smaller than fmax, stop the optimization
        if fmax_mask.all():
            self.stop_signal = True
            return

        etol_mask = self.etol_criteria(energies)
        # if all molecules have energies converged, stop the optimization
        if etol_mask.all():
            self.stop_signal = True
            return

        mask = ~fmax_mask & ~etol_mask

        # x_{n+1} = x_n - f(x_n)/f'(x_n)
        dr = energies / forces
        dr *= self.duq(mask)

        self.coords += dr

    def emax_criteria(self, energies):
        return (energies.flatten() - self.last_e).abs() < self.etol


class BFGSv1(GeometryOptimizer):
    """
    BFGS algorithm for batch of coordinates, forces will be input to step() function
    """

    # BFGS algorithm for batch of coordinates, forces will be input to step() function
    # assuming the input coordinates are in the shape of (batch_size, n_atoms, 3)
    # for convenience, notations follow the wikipedia page:
    # https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm

    ###TODO: BFGS may not be suitable for batch coordinates with zero-paddings
    # because the Hessian matrix will no longer be positive definite

    def __init__(self, coords=None, numbers=None, max_steps=100, H0=70.0, maxstep=0.2, fmax=0.0, logfile=False, device="cpu"):
        # hyperpamaters H0=70.0 & maxstep=0.2 are learned from ASE implementation
        super().__init__(coords, max_steps, logfile, device)
        self.H0 = H0  # initialize Hessian to diagonal matrices with H0
        self.maxstep = maxstep
        self.fmax = fmax

        # enable initialization without coords
        if isinstance(coords, torch.Tensor):
            self.reset(coords)

    def reset(self, coords, numbers=None):
        self._reset(coords)
        self.batch_size = self.coords.shape[0]
        self.n_atoms = self.coords.shape[1]

        self.B = None
        self.last_coords = None
        self.last_forces = None

    def step(self, forces):

        fmax_mask = self.fmax_criteria(forces, self.fmax)
        # if all molecules have forces smaller than fmax, stop the optimization
        if fmax_mask.all():
            self.stop_signal = True
            return

        # update Hessian
        flattened_forces = forces.reshape(self.batch_size, -1)
        self.update_B(flattened_forces)

        # get the step direction by updated Hessian
        eigenvalues, eigenvectors = torch.linalg.eigh(self.B)
        p = -torch.transpose(eigenvectors, 1, 2) @ (eigenvectors @ flattened_forces.unsqueeze(-1)) / torch.abs(eigenvalues.unsqueeze(-1))
        p = p.reshape(self.batch_size, self.n_atoms, 3)

        # update coords and forces
        self.last_coords = self.coords.clone().detach()
        self.last_forces = forces.clone().detach()

        # scale the steplength on each atom with the scaling factor
        # gotten from the maximum atomic steplength in a molecule
        # in this case we still move toward direction p

        maxsteplengths = torch.norm(p, dim=-1).max(dim=-1).values  # (batch_size,)
        scale = self.maxstep / maxsteplengths

        # final update movement
        dr = p * self.duq(scale)

        mask = ~fmax_mask  # may add more criteria here
        dr *= self.duq(mask)
        self.coords += dr

    def update_B(self, flattened_forces):
        if self.B is None:
            self.B = torch.eye(3 * self.n_atoms, device=self.device).repeat(self.batch_size, 1, 1) * self.H0
            return

        flattened_coords = self.coords.reshape(self.batch_size, -1)

        s = flattened_coords - self.last_coords.reshape(self.batch_size, -1)
        # Note that forces are -f'(x)
        y = -(flattened_forces - self.last_forces.reshape(self.batch_size, -1))

        alpha = 1.0 / (y.unsqueeze(1) @ s.unsqueeze(-1))
        beta = -1.0 / (s.unsqueeze(1) @ self.B @ s.unsqueeze(-1))

        self.B = (
            self.B
            + alpha * (y.unsqueeze(-1) @ y.unsqueeze(1))
            + beta * (self.B @ s.unsqueeze(-1)) @ s.unsqueeze(1) @ torch.transpose(self.B, 1, 2)
        )

    def log(self, extra_message=""):
        message = "Hessian: %s\n" % (str(self.H))
        message += extra_message
        print(message)
        self._log(message)


class BFGSv2(GeometryOptimizer):
    """
    batch BFGS algorithm with Moore-Penrose pseudoinverse
    """

    # BFGS algorithm for batch of coordinates, forces will be input to step() function
    # assuming the input coordinates are in the shape of (batch_size, n_atoms, 3)
    # for convenience, notations follow the wikipedia page:
    # https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm

    def __init__(self, coords=None, numbers=None, max_steps=100, H0=70.0, maxstep=0.2, fmax=0.0, logfile=False, device="cpu"):
        # hyperpamaters H0=70.0 & maxstep=0.2 are learned from ASE implementation
        super().__init__(coords, max_steps, logfile, device)
        self.H0 = H0  # initialize Hessian to diagonal matrices with H0
        self.maxstep = maxstep
        self.fmax = fmax

        # enable initialization without coords
        if isinstance(coords, torch.Tensor) and isinstance(numbers, torch.Tensor):
            self.reset(coords, numbers)

    def reset(self, coords, numbers):
        self._reset(coords)
        self.batch_size = self.coords.shape[0]
        self.n_atoms = self.coords.shape[1]

        self.last_coords = None
        self.last_forces = None

        t = numbers.unsqueeze(1).repeat(1, 3, 1).reshape(numbers.shape[0], -1).sort(descending=True)[0]
        t = t.to(torch.float32)  # cuda tensor calculation only works for float
        self.B = (t.unsqueeze(-1) @ t.unsqueeze(1)).to(bool)

    def step(self, forces):

        fmax_mask = self.fmax_criteria(forces, self.fmax)
        # if all molecules have forces smaller than fmax, stop the optimization
        if fmax_mask.all():
            self.stop_signal = True
            return

        # update Hessian
        flattened_forces = forces.reshape(self.batch_size, -1)
        self.update_B(flattened_forces)

        Binv = torch.linalg.pinv(self.B)
        p = Binv @ flattened_forces.unsqueeze(-1)
        p = p.reshape(self.batch_size, self.n_atoms, 3)

        # update coords and forces
        self.last_coords = self.coords.clone().detach()
        self.last_forces = forces.clone().detach()

        # scale the steplength on each atom with the scaling factor
        # gotten from the maximum atomic steplength in a molecule
        # in this case we still move toward direction p

        maxsteplengths = torch.norm(p, dim=-1).max(dim=-1).values  # (batch_size,)
        scale = self.maxstep / maxsteplengths

        # final update movement
        dr = p * self.duq(scale)

        mask = ~fmax_mask  # may add more criteria here
        dr *= self.duq(mask)
        self.coords += dr

    def update_B(self, flattened_forces):
        if self.B.dtype == torch.bool:  # the initial B mask is bool matrix
            self.B = torch.eye(3 * self.n_atoms, device=self.device).repeat(self.batch_size, 1, 1) * self.B.to(torch.float32) * self.H0
            return

        flattened_coords = self.coords.reshape(self.batch_size, -1)

        s = flattened_coords - self.last_coords.reshape(self.batch_size, -1)
        # Note that forces are -f'(x)
        y = -(flattened_forces - self.last_forces.reshape(self.batch_size, -1))

        alpha = 1.0 / (y.unsqueeze(1) @ s.unsqueeze(-1))
        beta = -1.0 / (s.unsqueeze(1) @ self.B @ s.unsqueeze(-1))

        self.B = (
            self.B
            + alpha * (y.unsqueeze(-1) @ y.unsqueeze(1))
            + beta * (self.B @ s.unsqueeze(-1)) @ s.unsqueeze(1) @ torch.transpose(self.B, 1, 2)
        )

    def log(self, extra_message=""):
        message = "Hessian: %s\n" % (str(self.H))
        message += extra_message
        print(message)
        self._log(message)


class BFGSv3(GeometryOptimizer):
    """
    BFGS algorithm for batch of coordinates, forces will be input to step() function

     In this version, instead of maintain the Hessian itself,
      we maintain inverse of Hessian and update it with Sherman–Morrison formula


    """

    # assuming the input coordinates are in the shape of (batch_size, n_atoms, 3)
    # for convenience, notations follow the wikipedia page:
    # https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm

    def __init__(self, coords=None, numbers=None, max_steps=100, H0=1.0 / 70, maxstep=0.2, fmax=0.0, logfile=False, device="cpu"):
        # hyperpamaters H0=70.0 & maxstep=0.2 are learned from ASE implementation
        super().__init__(coords, max_steps, logfile, device)
        self.H0 = H0  # initialize Hessian to diagonal matrices with H0
        self.maxstep = maxstep
        self.fmax = fmax

        # enable initialization without coords
        if isinstance(coords, torch.Tensor) and isinstance(numbers, torch.Tensor):
            self.reset(coords, numbers)

    def reset(self, coords, numbers):
        self._reset(coords)
        self.batch_size, self.n_atoms = self.coords.shape[:2]

        self.Binv = None
        self.last_coords = None
        self.last_forces = None

        # this is a way to get the initial B mask without for loop
        # it cost too much memory, need to verify if its really faster than for loop and padding
        t = numbers.unsqueeze(1).repeat(1, 3, 1).reshape(numbers.shape[0], -1).sort(descending=True)[0]
        t = t.to(torch.float32)
        self.Binv_mask = (t.unsqueeze(-1) @ t.unsqueeze(1)).to(bool)

    def step(self, forces):

        fmax_mask = self.fmax_criteria(forces, self.fmax)
        # if all molecules have forces smaller than fmax, stop the optimization
        if fmax_mask.all():
            self.stop_signal = True
            return

        # update Hessian
        flattened_forces = forces.reshape(self.batch_size, -1)
        self.update_Binv(flattened_forces)

        p = self.Binv @ flattened_forces.unsqueeze(-1)
        p = p.reshape(self.batch_size, self.n_atoms, 3)

        # print(p)

        # update coords and forces
        self.last_coords = self.coords.clone().detach()
        self.last_forces = forces.clone().detach()

        # scale the steplength on each atom with the scaling factor
        # gotten from the maximum atomic steplength in a molecule
        # in this case we still move toward direction p

        maxsteplengths = torch.norm(p, dim=-1).max(dim=-1).values  # (batch_size,)
        scale = self.maxstep / maxsteplengths

        # final update movement
        dr = p * self.duq(scale)

        mask = ~fmax_mask  # may add more criteria here
        dr *= self.duq(mask)
        self.coords += dr

    def update_Binv(self, flattened_forces):
        if self.Binv is None:  # the initial B mask is bool matrix
            self.Binv = torch.eye(3 * self.n_atoms, device=self.device).repeat(self.batch_size, 1, 1) * self.Binv_mask
            # after the first update, we no longer need to save the boolean Binv_mask, but we need to save
            # the "padded identity matrices", which has the same shape as Binv_mask, so I simply save it into self.Binv_mask
            self.Binv_mask = self.Binv.clone()
            self.Binv = self.Binv * self.H0
            return

        flattened_coords = self.coords.reshape(self.batch_size, -1)

        s = flattened_coords - self.last_coords.reshape(self.batch_size, -1)
        # Note that forces are -f'(x)
        y = -(flattened_forces - self.last_forces.reshape(self.batch_size, -1))

        alpha = 1.0 / (y.unsqueeze(1) @ s.unsqueeze(-1))

        # It seems that there is a way to avoid storing self.Binv_mask
        # but that requires more matrices multiplication
        # Note: self.Binv_mask is already the padded identity matrices
        self.Binv = (self.Binv_mask - alpha * (s.unsqueeze(-1) @ y.unsqueeze(1))) @ self.Binv @ (
            self.Binv_mask - alpha * (y.unsqueeze(-1) @ s.unsqueeze(1))
        ) + alpha * (s.unsqueeze(-1) @ s.unsqueeze(1))

    def log(self, extra_message=""):
        message = "Binv: %s\n" % (str(self.Binv))
        message += extra_message
        print(message)
        self._log(message)
