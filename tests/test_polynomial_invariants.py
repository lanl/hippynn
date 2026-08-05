import pytest
import torch
import time

from hippynn.layers.hiplayers.tensors import HopInvariantLayerTorch, TensorExtractor
from hippynn.layers.hiplayers.invariants import (
    HopInvariantLayer,
    compute_invariant_polynomial_collection,
    default_invariants_list,
    split_invariant,
    triton_available_with_gather,
)
from hippynn.layers.hiplayers.interactions import _invariant_counts


def test_default_invariant_definitions_parse():
    for invariant in default_invariants_list:
        indices, tensors = split_invariant(invariant)
        assert len(indices) == len(tensors)

        for index, tensor in zip(indices, tensors):
            assert len(index) == {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4}[tensor]


def test_polynomial_invariant_counts_match_interaction_lmax3():
    for n_max in range(1, 5):
        if triton_available_with_gather:
            unit_tensor_bases = {order: torch.ones(*([1] * order), 1) for order in range(4)}
            polyCollection = compute_invariant_polynomial_collection(
                n_max,
                3,
                tensor_bases=unit_tensor_bases,
                input_tensor_ordering=["zero", "one", "two", "three"],
            )
            _, _, polynomial_sizes, _ = polyCollection.get_polynomials()
            n_invariants = len(polynomial_sizes)
        else:
            tensor_features = torch.randn((3, 16))
            n_invariants = HopInvariantLayerTorch(n_max, 3)(tensor_features).shape[1]

        assert n_invariants == _invariant_counts[n_max, 3]


def evaluate_polynomial_collection_torch(x, polyCollection):
    coefs, terms, polynomial_sizes, _ = polyCollection.get_polynomials()

    outputs = []
    monomial_start = 0
    for polynomial_size in polynomial_sizes.tolist():
        values = x.new_zeros(x.shape[0])
        for monomial_idx in range(monomial_start, monomial_start + polynomial_size):
            term_indices = terms[monomial_idx]
            term_indices = term_indices[term_indices >= 0]
            values = values + coefs[monomial_idx].to(x) * x[:, term_indices].prod(dim=1)

        outputs.append(values)
        monomial_start += polynomial_size

    return torch.stack(outputs, dim=1)


def test_polynomial_invariants_are_rotation_invariant_lmax3():
    n_point = 11
    torch.manual_seed(0)

    rhats = torch.randn(n_point, 3)
    rhats = rhats / rhats.norm(dim=1, keepdim=True)

    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1

    tensor_extractor = TensorExtractor(l_max=3)
    tensor_features = torch.cat(tensor_extractor(rhats)[:4], dim=1)
    rotated_tensor_features = torch.cat(tensor_extractor(rhats @ rotation)[:4], dim=1)

    if triton_available_with_gather:
        start = time.perf_counter()
        polyCollection = compute_invariant_polynomial_collection(
            n_max=4,
            l_max=3,
            input_tensor_ordering=["zero", "one", "two", "three"],
        )

        start = time.perf_counter()
        invariants = evaluate_polynomial_collection_torch(tensor_features, polyCollection)
        rotated_invariants = evaluate_polynomial_collection_torch(rotated_tensor_features, polyCollection)
    else:
        start = time.perf_counter()
        invariant_layer = HopInvariantLayerTorch(n_max=4, l_max=3)
        invariants = invariant_layer(tensor_features)
        rotated_invariants = invariant_layer(rotated_tensor_features)

    assert torch.allclose(invariants, rotated_invariants, rtol=1e-4, atol=1e-4)

def test_polynomial_invariants():

    n_point = 3
    torch.manual_seed(0)

    if triton_available_with_gather and torch.cuda.is_available():

        from hippynn.custom_kernels.poly_triton import EvaluatePolynomials

        for l_max in range(4):
            for n_max in range(1, 5):

                n_tensor_comp = (l_max+1)**2
                tensor_features = torch.randn((n_point, n_tensor_comp), requires_grad=True, device='cuda')

                polyCollection = compute_invariant_polynomial_collection(
                    n_max,
                    l_max,
                    input_tensor_ordering=["zero", "one", "two", "three"][: l_max + 1],
                )
                polyCollection.set_device('cuda')
                invars_poly = EvaluatePolynomials.apply(tensor_features, polyCollection)

                invars_torch = evaluate_polynomial_collection_torch(tensor_features, polyCollection)

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

    for l_max in range(4):
        for n_max in range(1, 5):

            n_tensor_comp = (l_max+1)**2
            tensor_features = torch.randn((n_point, n_tensor_comp), requires_grad=True, device=device)

            invariantLayer = HopInvariantLayer(n_max, l_max)
            invariantLayer = invariantLayer.to(device)
            invars_poly = invariantLayer(tensor_features)

            if triton_available_with_gather and torch.cuda.is_available():
                polyCollection = compute_invariant_polynomial_collection(
                    n_max,
                    l_max,
                    input_tensor_ordering=["zero", "one", "two", "three"][: l_max + 1],
                )
                polyCollection.set_device(device)
                invars_torch = evaluate_polynomial_collection_torch(tensor_features, polyCollection)
            else:
                torchInvariantLayer = HopInvariantLayerTorch(n_max, l_max)
                torchInvariantLayer = torchInvariantLayer.to(device)
                invars_torch = torchInvariantLayer(tensor_features)

            assert torch.allclose(invars_poly, invars_torch, rtol=1e-4, atol=1e-4)

            # during this check we need tensor features to be float32 because the 
            # old HopInvariantLayerTorch only supports float32
            # Thus, we can only gradcheck if triton is available

            if triton_available_with_gather and torch.cuda.is_available():
                tensor_features = tensor_features.to(torch.float64)

                assert torch.autograd.gradcheck(invariantLayer, (tensor_features,))
                assert torch.autograd.gradgradcheck(invariantLayer, (tensor_features,))
