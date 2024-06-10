from ..graphs import Predictor
from .BatchOptimizer import *

class HippyNNBatchOptimizer():
    def __init__(self, model, optimizer=BatchFIRE(), dump_traj=False):
        
        # assume model is a hippynn model, and there will be 'Grad' in its output
        self.predictor = Predictor.from_graph(model)
        self.optimizer = optimizer
        self.dump_traj = dump_traj

    def __call__(self, Z, R):
        coords = R.clone()
        self.optimizer.reset(coords)
        while not self.optimizer.stop_signal:
            ret = self.predictor(Z=Z, R=coords)
            forces = -ret['Grad']
            self.optimizer(forces)
            coords = self.optimizer.coords
        return coords
    
    def dump_step(self, Z, R):
        # need to figure out how to dump a batch of structures
        pass