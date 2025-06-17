"""
Interface for creating LAMMPS MLIAP Unified models.
"""
# builtins
import warnings

# base requirements
import numpy as np
import torch

# lammps interface class
from lammps.mliap.mliap_unified_abc import MLIAPUnified

# hippynn things.
from ...tools import device_fallback
from ... import settings
from .graph_setup import setup_LAMMPS_graph


# specifically for LAMMPS comms
from ...networks.hipnn import Hipnn, HipnnVec, HipnnQuad


LAMMPS_COMM_MODULES = [Hipnn, HipnnVec, HipnnQuad]


class MLIAPInterface(MLIAPUnified):
    """
    Class for creating ML-IAP Unified model based on hippynn graphs.
    """

    def __init__(
        self,
        energy_node,
        element_types,
        ndescriptors=1,
        model_device=torch.device("cpu"),
        compute_dtype=torch.float32,
        energy_unit: float = None,
        distance_unit: float = None,
    ):
        """
        :param energy_node: Node for energy
        :param element_types: list of atomic symbols corresponding to element types
        :param ndescriptors: the number of descriptors to report to LAMMPS
        :param model_device: the device to send torch data to (cpu or cuda)
        :param energy_unit: If present, multiply the result by the given energy units.
            If your model was trained in Hartree and your lammps script will operate in eV,
            use energy_unit = ase.units.Ha = 27.211386024367243
        :param distance_unit: If present, multi input distances by this much as well as dividing into output forces.
            If your model was trained to accept nm as input and lammps uses Angstroms,
            use distance_unit = ase.units.nm = 10.
        """
        super().__init__()
        if settings.PYTORCH_GPU_MEM_FRAC < 1.0:
            torch.cuda.set_per_process_memory_fraction(settings.PYTORCH_GPU_MEM_FRAC)
        self.element_types = element_types
        self.ndescriptors = ndescriptors
        self.model_device = model_device
        self.energy_unit = energy_unit
        self.distance_unit = distance_unit

        # Build the calculator
        self.rcutfac, self.species_set, self.graph = setup_LAMMPS_graph(energy_node)
        self.nparams = sum(p.nelement() for p in self.graph.parameters())
        self.compute_dtype = compute_dtype
        self.graph.to(compute_dtype)

        self.clear_runtime_variables()

    def clear_runtime_variables(self):
        # Variables that will be populated at run time.
        self.mliap_data = None
        self.handles = []
        self.using_kokkos = None
        self.memory_set = None


    def compute_gradients(self, data):
        pass

    def compute_descriptors(self, data):
        pass

    def as_tensor(self, array):
        return torch.as_tensor(array, device=self.model_device)

    def empty_tensor(self, dimensions):
        return torch.empty(dimensions, device=self.model_device)

    def perform_setup(self):

        if self.memory_set is None:
            if settings.PYTORCH_GPU_MEM_FRAC < 1.0:
                torch.cuda.set_per_process_memory_fraction(settings.PYTORCH_GPU_MEM_FRAC)
            self.memory_set = True

        if self.using_kokkos is None:
            # Test if we are using lammps-kokkos or not. Is there a more clear way to do that?
            self.using_kokkos = "kokkos" in self.mliap_data.__class__.__module__.lower()

        if settings.COMM_FEATURES_LAMMPS:  # Setting the comm handles is selected
            if not self.handles:  # no handles have been installed yet
                if hasattr(self.mliap_data, "forward_exchange"):  # this is a compatible version of lammps
                    self.install_lammps_comm_hooks()
                else:
                    warnings.warn("Lammps feature communication was requested but not available in this "
                                  "version of lammps.")


    def install_lammps_comm_hooks(self):
        # Add pre-forward hooks and post-backwards hooks for message passing
        # comms to interaction layers in graph
        hipnn_modules = [n for n in self.graph.modules() if any([isinstance(n,m) for m in LAMMPS_COMM_MODULES]) ]

        handles = []
        for network in hipnn_modules:
            # The first interaction layer does not need the
            # forward_pre_hook nor the backward hook since the first
            # input layer are positions of ghosts which are already
            # known. Exchanging these inputs anyway should not affect
            # the answer.
            for module in network.interaction_layers[1:]:
                handle = module.register_forward_pre_hook(self.comms_forward_pre_hook)
                handles.append(handle)
                handle = module.register_full_backward_hook(self.comms_backward_hook)
                handles.append(handle)

        self.handles += handles

    def uninstall_hooks(self):
        for handle in self.handles:
            handle.remove()

    def comms_forward_pre_hook(self, module, args):  # -> None or modified input
        """
        :param module: Torch Module for Interaction Layer
        :param args: list of arguments intercepted from module's forward, features input missing features in ghosts
        :return modded_args: list of arguments passed on to forward now with ghost inputs
        Invokes blocking MPI exchanges within LAMMPS to retrieve ghost inputs
        """
        local_in_features, *other_args = args

        global_in_features = local_in_features  # We can modify in-place in the forward_pre_hook

        self.mliap_data.forward_exchange(local_in_features, global_in_features,
                                         local_in_features.shape[1])

        return global_in_features, *other_args

    def comms_backward_hook(self, module, grad_input, grad_output):
        """
        :param module: Torch Module for Interaction Layer
        :param grad_input: Gradient (of atom energies) with respect to inputs of forward, only features missing comms.
        :param grad_output: list of arguments intercepted from module's forward, missing ghost inputs
        :return modded_grad_input: Gradient with respect to inputs now with comms.
        Invokes blocking MPI exchanges within LAMMPS to retrieve ghost inputs

        The gradient with respect to the input features in grad_input contains
        contributions to gradients for other ranks within this rank's ghosts and is
        missing contributions to this rank's owned atoms on other ranks. After
        comms, modded_grad_input will have zero'ed out the ghosts (sent to other
        processors) and added gradient contributions from other rank's ghosts into
        this rank's owned atoms.
        """

        # Only the gradients with respect to the first input need changes
        local_grad_in_features, *rest_grad_in = grad_input

        # Can't modify in-place in backward_hook
        global_grad_in_features = local_grad_in_features.detach().clone()  # Copy entire tensor

        # Comms from ghosts into owned atoms is a "reverse" comms in LAMMPS
        self.mliap_data.reverse_exchange(local_grad_in_features, global_grad_in_features,
                                         local_grad_in_features.shape[1])

        return global_grad_in_features, *rest_grad_in

    def compute_forces(self, data):
        """
        :param data: MLIAPData object (provided internally by lammps)
        :return None
        This function writes results to the input `data`.
        """

        # If there are no local atoms, do nothing
        nlocal = self.as_tensor(data.nlistatoms)
        if nlocal.item() <= 0:
            return

        self.mliap_data = data  # hook data onto the (persistent) object for, e.g., comms hooks. This is needed!
        self.perform_setup()

        elems = self.as_tensor(data.elems).type(torch.int64).reshape(1, data.ntotal)
        z_vals = self.species_set[elems + 1]
        npairs = data.npairs

        if npairs > 0:
            pair_i = self.as_tensor(data.pair_i).type(torch.int64)
            pair_j = self.as_tensor(data.pair_j).type(torch.int64)
            rij = self.as_tensor(data.rij).type(self.compute_dtype)
        else:
            pair_i = self.empty_tensor(0).type(torch.int64)
            pair_j = self.empty_tensor(0).type(torch.int64)
            rij = self.empty_tensor([0, 3]).type(self.compute_dtype)

        if self.distance_unit is not None:
            rij = self.distance_unit * rij

        # note your sign for rij might need to be +1 or -1, depending on how your implementation works
        inputs = [z_vals, pair_i, pair_j, -rij, nlocal]
        atom_energy, total_energy, fij = self.graph(*inputs)



        # convert units
        if self.energy_unit is not None:
            atom_energy = self.energy_unit * atom_energy
            total_energy = self.energy_unit * total_energy
            fij = self.energy_unit * fij

        if self.distance_unit is not None:
            fij = fij / self.distance_unit

        # Write data back. Kokkos and non-kokkos interfaces have diverged, so slightly different paths.
        if self.using_kokkos:
            return_device = elems.device
        else:
            return_device = "cpu"

        atom_energy = atom_energy.squeeze(1).detach().to(return_device)
        total_energy = total_energy.detach().to(return_device)
        data.energy = total_energy.item()

        f = self.as_tensor(data.f)
        fij = fij.type(f.dtype).detach().to(return_device)

        if not self.using_kokkos:
            # write back to data.eatoms directly.
            fij = fij.numpy()
            data.eatoms = atom_energy.numpy().astype(np.double)
            if npairs > 0:
                data.update_pair_forces(fij)
        else:
            # view to data.eatoms using pytorch, and write into the view.
            eatoms = torch.as_tensor(data.eatoms, device=return_device)
            eatoms.copy_(atom_energy)
            if npairs > 0:
                if return_device == "cpu":
                    data.update_pair_forces_cpu(fij)
                else:
                    data.update_pair_forces_gpu(fij)

        self.mliap_data = None  # unhook data, see hooking above.

    def __getstate__(self):
        self.species_set = self.species_set.to(torch.device("cpu"))
        self.graph.to(torch.device("cpu"))
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)
        try:
            torch.ones(0).to(self.model_device)
        except RuntimeError:
            fallback = device_fallback()
            warnings.warn(f"Model device ({self.model_device}) not found, falling back to f{fallback}")
            self.model_device = fallback

        if not hasattr(self, "energy_unit"):
            self.energy_unit = None
        if not hasattr(self, "distance_unit"):
            self.distance_unit = None

        self.species_set = self.species_set.to(self.model_device)
        self.graph.to(self.model_device)
        self.clear_runtime_variables()


