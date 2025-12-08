import pytest
import torch

from hippynn.layers.hiplayers.tensors import HopInvariantLayerTorch
from hippynn.layers.hiplayers.invariants import HopInvariantLayer, compute_invariant_polynomial_collection

def test_polynomial_invariants():

    n_point = 3
    torch.manual_seed(0)

    try:
        import triton
        triton_available = True
    except:
        triton_available = False

    if triton_available and torch.cuda.is_available():

        from hippynn.custom_kernels.poly_triton import EvaluatePolynomials

        for l_max in range(4):
            for n_max in range(1,5):

                n_tensor_comp = (l_max+1)**2
                tensor_features = torch.randn((n_point, n_tensor_comp), requires_grad=True, device='cuda')

                polyCollection = compute_invariant_polynomial_collection(n_max, l_max)
                polyCollection.set_device('cuda')
                invars_poly = EvaluatePolynomials.apply(tensor_features, polyCollection)

                torchInvariantLayer = HopInvariantLayerTorch(n_max, l_max)
                torchInvariantLayer = torchInvariantLayer.to('cuda')
                invars_torch = torchInvariantLayer(tensor_features)

                # during this check we need tensor features to be float32 because the 
                # old HopInvariantLayerTorch only supports float32

                assert torch.allclose(invars_poly, invars_torch, rtol=1e-4, atol=1e-4)

                tensor_features = tensor_features.to(torch.float64)

                assert torch.autograd.gradcheck(EvaluatePolynomials.apply, (tensor_features, polyCollection))
                assert torch.autograd.gradgradcheck(EvaluatePolynomials.apply, (tensor_features, polyCollection))

def test_invariants_wrapper():

    n_point = 3
    torch.manual_seed(0)

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    try:
        import triton
        triton_available = True
    except:
        triton_available = False

    for l_max in range(4):
        for n_max in range(1,5):

            n_tensor_comp = (l_max+1)**2
            tensor_features = torch.randn((n_point, n_tensor_comp), requires_grad=True, device=device)

            invariantLayer = HopInvariantLayer(n_max, l_max)
            invariantLayer = invariantLayer.to(device)
            invars_poly = invariantLayer(tensor_features)

            torchInvariantLayer = HopInvariantLayerTorch(n_max, l_max)
            torchInvariantLayer = torchInvariantLayer.to(device)
            invars_torch = torchInvariantLayer(tensor_features)

            assert torch.allclose(invars_poly, invars_torch, rtol=1e-4, atol=1e-4)

            # during this check we need tensor features to be float32 because the 
            # old HopInvariantLayerTorch only supports float32
            # Thus, we can only gradcheck if triton is available

            if triton_available and torch.cuda.is_available():
                tensor_features = tensor_features.to(torch.float64)

                assert torch.autograd.gradcheck(invariantLayer, (tensor_features,))
                assert torch.autograd.gradgradcheck(invariantLayer, (tensor_features,))