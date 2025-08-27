import torch

from .polynomial_invariants import compute_invariant_polynomial_collection
from .tensors import calc_invariants, cmaps

from ...custom_kernels.poly_triton import EvaluatePolynomials, get_next_polynomial_id
from ... import settings

polynomials_cache = {}

class HopInvariantLayerCombined(torch.nn.Module):
    def __init__(self, n_max, l_max, cmaps_=cmaps):
        super().__init__()
        self.l_max = l_max
        self.n_max = n_max
        self.cmaps = cmaps_
        self.id = get_next_polynomial_id()

        # register buffer to allow us to check the device that the layer is set to
        # (allowing use of .to(), .cpu(), etc.)
        self.register_buffer("device_check", torch.empty(0))

        try:
            import triton
            self.triton_available = True
        except:
            self.triton_available = False

    def forward(self, tensor_features):
        device = self.device_check.device

        cmaps_device = self.cmaps[0].device
        if cmaps_device != device:
            for c in self.cmaps.keys():
                self.cmaps[c] = self.cmaps[c].to(device)


        if device == 'cpu' or not settings.USE_POLYNOMIAL_INVARIANTS or not self.triton_available:
            return calc_invariants(self.l_max, self.n_max, tensor_features, self.cmaps)
        else:
            if (self.l_max, self.n_max, device) not in polynomials_cache:
                coefs, terms, polynomial_sizes = compute_invariant_polynomial_collection(self.n_max, self.l_max, self.cmaps)
                coefs = coefs.to(device)
                terms = terms.to(device)
                polynomial_sizes = polynomial_sizes.to(device)
                polynomials_cache[(self.l_max, self.n_max, device)] = (coefs, terms, polynomial_sizes)
            else:
                coefs, terms, polynomial_sizes = polynomials_cache[(self.l_max, self.n_max, device)]

            return EvaluatePolynomials.apply(tensor_features, coefs, terms, polynomial_sizes, self.id)