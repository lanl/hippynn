import warnings

from ase import Atoms

from ..graphs import Predictor
from .algorithms import BFGSv3
from .utils import debatch, debatch_coords
from ..graphs.nodes.physics import GradientNode
from ..graphs import find_unique_relative
from ..graphs.nodes.base.node_functions import NodeNotFound
from ..tools import device_fallback


class Optimizer:
    def __init__(
            self,
            model,
            algorithm=BFGSv3,
            dump_traj=False,
            device=None,
            relocate_optimizer=False,
            force_key=None,
            force_sign=None,
    ):
        """
        :param model: graphmodule of model.
        :param algorithm: base optimizer algorithm to use. see algorithms.py
        :param dump_traj: whether to safe the optimization paths.
        :param device: where to do the computation.
        :param relocate_optimizer: where to put the optimizer itself.
        :param force_key: str or Node which specifies the forces on the coordinates.
            if None, try to auto-detect
        :param force_sign: whether the above about it force (sign=-1) or gradient (sign=+1).
         If none, try ot auto-detect from force key.
        """

        if isinstance(algorithm, type):
            algorithm = algorithm()

        if device is None:
            device = device_fallback
        self.device = device

        # TODO: possibly put this kind of logic into a separate function for re-use
        if force_key is None:
            try:
                force_node = find_unique_relative(model.nodes_to_compute, GradientNode)
            except NodeNotFound:
                try:
                    force_node = find_unique_relative(model.nodes_to_compute, lambda node: 'force' in node.name,)
                except Exception as ee:
                    raise ValueError("No automatic force node could be found for optimizer.") from ee
                # TODO even find an energy node if we can and add forces to it?
        else:
            force_node = find_unique_relative(model.nodes_to_compute, lambda node: node.db_name == force_key)

        additional_outputs = None
        if force_node is not None and force_node not in model.nodes_to_compute:
            additional_outputs = [force_node]

        self.predictor = Predictor.from_graph(
            model,
            additional_outputs=additional_outputs,
            return_device=self.device,
            model_device=self.device,
        )

        self.algorithm = algorithm
        if relocate_optimizer:
            self.algorithm.device = self.device
        self.dump_traj = dump_traj
        self.masks = None

        self.force_key = force_key

        if isinstance(force_key, str) and force_sign is None:
            force_node = self.predictor.graph.node_from_name(force_key)
            force_sign = force_node.torch_module.sign

        self.force_sign = force_sign

        if dump_traj:
            warnings.warn(
                "Dump batch trajectories every step requires copying \n \
                  and moving tensors to cpu, which is slow!"
            )

    def __call__(self, Z, R, padding_number=0, prefix_list=None):

        self.masks = None

        self.algorithm.reset(coords=R, numbers=Z)

        while not self.algorithm.stop_signal:
            # dump a step before optimization to write the initial coordinates
            # this also enable wrapping the io part into a single function
            ###TODO: consider separate dump_step and dump_traj into two operations
            # dump_step write a single step to $batch_size xyz files
            if self.dump_traj:
                self.dump_a_step(Z, self.algorithm.coords, padding_number, prefix_list)
            ret = self.predictor(Z=Z, R=self.algorithm.coords)
            # forces = -ret['Grad']
            # set this tunable, although I suggest to simply have 'F' in the model output
            forces = ret[self.force_key] * self.force_sign
            self.algorithm.coords = self.algorithm.coords.detach()
            self.algorithm(forces)

        # also return the model output from the last step
        # because the energies in it might be useful
        return self.algorithm.coords, ret

    def dump_a_step(self, numbers, coords, padding_number=0, prefix_list=None):
        N = coords.shape[0]
        if self.masks is None:
            self.debatched_numbers, debatched_coords, self.masks = debatch(numbers, coords, padding_number, return_mask=True)
        else:
            debatched_coords = debatch_coords(coords, self.masks)

        prefix_list = prefix_list or list(range(N))
        for n, c, prefix in zip(self.debatched_numbers, debatched_coords, prefix_list):
            atoms = Atoms(numbers=n.cpu(), positions=c.cpu())
            atoms.write("%s_traj.xyz" % (prefix), append=True)
