"""
Layers for getting L2 parameters
"""
import torch


class LPReg(torch.nn.Module):
    def __init__(self, network, p=2):
        super().__init__()
        self.network = network
        self.p = p

    def forward(self, *args):
        # args is ignored; the output is not dependent on the output of the network.
        params = list(self.network.regularization_params())
        if len(params) == 0:
            raise ValueError("L^p regularization has no parameters.")
        return sum(torch.sum(torch.abs(torch.pow(p, self.p))) for p in params)
