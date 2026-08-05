import torch
import torch.nn.functional as F

from .tensors import calc_invariants, cmaps
from packaging import version
from ... import settings

import string

"""
This module concerns itself with computing invariants by taking products and contractions of irreducible tensors. It defines a HopInvariantsLayer,
which is a custom torch layer that can be used to compute these invariants. By default, the invariants will be computed using the triton
kernel for evaluating polynomials. As such, most of the logic in this module concerns itself with evaluating the polynomials using triton.
However, if triton or cuda are not available, or using triton for invariants is disabled in the settings, then HopInvariantsLayer will fall
back on a pytorch implementation that is defined in layers/hiplayers/tensors.py.

We assume that all relevant irreducible tensors of order l are represented as coefficients that correspond to a
basis Bl for the space of irreducible tensors of order l. Furthermore, we assume that the basis coefficients
of all of moment tensors are concatenated into a single vector X = (x1, x2, ..., xn), where the basis coefficients
of each tensor corresponds to a contiguous slice of the entries of X. (for example, the first entry of X could correspond
to a tensor of order 0, then the next three correspond to a tensor of order 1, etc.).
"""


try:
    from ...custom_kernels.poly_triton import EvaluatePolynomials, PolynomialCollection

    triton_available_with_gather = True
    import triton
    if version.parse(triton.__version__) < version.parse("3.3.0"):
        triton_available_with_gather = False
except:
    triton_available_with_gather = False

# tensor ordering and list of invariants for the default invariants.
default_tensor_ordering = ["zero", "one", "two", "three"]

default_invariants_list = [
    "->,zero",
    "i,i->,one,one",
    "ij,ij->,two,two",
    "ij,ik,jk->,two,two,two",
    "ijk,ijk->,three,three",
    "ijk,ijl,abk,abl->,three,three,three,three",
    "i,ij,j->,one,two,one",
    "i,ij,jk,k->,one,two,two,one",
    "i,j,k,ijk->,one,one,one,three",
    "i,ijk,jkl,l->,one,three,three,one",
    "ij,jkl,ikl->,two,three,three",
    "ij,jk,ilm,klm->,two,two,three,three",
    "ij,ijk,kab,ab->,two,three,three,two",
]

def split_invariant(invariants):
    """
    splits an invariant string into a list of indices and tensors.
    """

    invars_split = invariants.replace("->","").split(",")

    assert len(invars_split) % 2 == 0, f"the number of indices and tensors in an invariant string is not equal ({invariants})."
    num_terms = len(invars_split) // 2
    indices = []
    tensors = []

    for term_idx in range(num_terms):
        indices.append( invars_split[term_idx].strip().replace("\n", "") )
        tensors.append( invars_split[term_idx + num_terms].strip().replace("\n", "") )

    return indices, tensors

def computeInvariantPolynomial(tensor_bases, invariant_input_offsets, invariant_code):
    """
    For a given invariant defined by contractions of irreducible moment tensors, represents that invariant
    as a multivariate polynomial.

    The polynomial that is returned is a function of the entries of X (e.g. 2*x1*x2 + 3*x3*x3).

    :param tensor_bases: The set of bases for tensors of each order used in the invariants. This should be a dictionary where tensor_bases[l] stores the basis for irreducible tensors
                         of order l. tensor_bases[l] should be an l+1 dimensional tensor, where the final axis of tensor_bases[l] should index the the basis elements.
                         For example tensor_bases[2][0,0,3] will represent, in the basis of irreducible tensors of order 2, the (0,0) entry of the third basis element.

    :param invariant_code: The invariant that is computed. The invariants should be specified as a list of string.
                           The string should take the same form as an einsum representation, but with two differences. First, instead of listing variable names at the end of the einsum string,
                           one should instead use a string that can be used to uniquely identify the tensor that you want to contract. The name can be anything, except that it cannot
                           include commas. The name will correspond with a contiguous slice of the basis coefficients stored in X. For example, if Tensor1 corresponds to a rank
                           1 tensor, then the name Tensor1 might correspond to entries 1, 2, and 3 of X, such that x1, x2, and x3 store the basis coefficients for Tensor1.
                           The way that one associates these names with slices of X is specified in the parameter invariant_input_offsets. Second, the entire einsum representation should be
                           a string. Because the reduction must lead to a scalar, there should be a comma right after the arrow. Then, there should be the names
                           of the tensors. For example, it could look like this: 'ijk,ijk->,Tensor1,Tensor1'. To specify a tensor of rank zero, the string would
                           look like this: '->,Tensor0'. Any strings with the arrow omitted (e.g. 'ijk,ijk,Tensor1,Tensor1) are also acceptable.

    :param invariant_input_offsets: A dictionary that is used to keep track of which parts of the vector X correspond to the various tensors.
                                    The keys the names for each tensor specified in invariant_code. If T is the name of a tensor, then invariant_input_offsets[T]
                                    stores the first location in the vector X that corresponds to the basis coefficients for tensor T.

    :return: The polynomial that can be used to compute the invariant from the vector X. The polynomial is specified by its monomials using two arrays. 
             The first array stores the coefficients of all monomials. For instance, if the polynomial is 2*x1*x3 + 4*x1*x1, the list of coefficients would store [2,4].
             The second array stores the indices of the terms in the monomial. It is a 2D array where each row corresponds to a different monomial. For instance, for the
             polynomial 2*x1*x3 + 4*x1*x1, the first row will store [1,3], and the second row will store [1,1].
    """

    invariant_code = invariant_code.replace("->","")
    invar_indices, invar_tensors = split_invariant(invariant_code)

    # For the zero order invariant, there is no einsum, it is just one monomial.
    if len(invar_indices) == 1:
        tensor_name = invar_tensors[0]
        offset = invariant_input_offsets[tensor_name]
        return torch.FloatTensor((1,)), torch.IntTensor(((offset,),))

    num_terms = len(invar_indices)

    # Each possible monomial in the polynomial will be the equivalent of choosing one basis coefficient from each tensor that we are contracting.
    # The coefficient corresponding to that monomial will be equal to performing our contraction on the corresponding basis elements.
    # We can compute all such coefficients by performing an einsum from adding one extra index to the end of each term, and then performing the einsum on the basis tensors (the elements of tensor_bases).
    # For example, the contraction ij,ij->,two,two becomes ijk,ijl->kl,tensor_bases[2],tensor_bases[2]. The resulting tensor will contain all of the coefficients for polynomials.

    # Here we set up the contraction that computes the polynomial coefficients.

    # get all of the letters used to specify the einsum.
    letters_used = set()

    for term_idx in range(num_terms):
        for letter in invar_indices[term_idx]:
            letters_used.add(letter)

    alphabet = {letter for letter in string.ascii_letters}

    # compute the einsum string by appending unused letters to the end of each term. Then, evaluate the einsum.

    unused_letters = alphabet - letters_used

    assert len(unused_letters) >= num_terms, "The invariant specified uses too many index letters; einsum ran out of letters to use."

    unused_letters = list(unused_letters)
    einsum_front = ""
    einsum_back = ""
    tensors_to_contract = []

    for term_idx in range(num_terms):
        code = invar_indices[term_idx]
        einsum_front += "," + code + unused_letters[term_idx]
        einsum_back += unused_letters[term_idx]
        tensors_to_contract.append( tensor_bases[len(code)] )

    einsum_string = einsum_front[1:] + "->" + einsum_back
    coef_tensor = torch.einsum(einsum_string, *tensors_to_contract)

    # Using a high n_max results in a massive amount of terms being added together.
    # Most of these coeffecients are zero, so we want to just disregard them
    # For l_max=3, n_max=12, this reduces 282,475,249 terms to 5,152,520 non zero terms (98% reduction)
    nonzero_basis_choices = torch.nonzero(coef_tensor, as_tuple=False)
    nonzero_coefs = coef_tensor[tuple(nonzero_basis_choices.T)]

    # This determines the starting index in the flattened feature vector for each tensor factor.
    # E.g. rank 0 -> 0, rank 1 -> 1, rank 2 -> 4, rank 3 -> 9
    # so "i,ij,j->,one,two,one" -> feature_offsets = [1, 4, 1]
    feature_offsets = torch.tensor(
        [invariant_input_offsets[invar_tensors[dim]] for dim in range(num_terms)],
        dtype=nonzero_basis_choices.dtype,
        device=nonzero_basis_choices.device,
    )
    monomial_terms = nonzero_basis_choices + feature_offsets

    # A lot of these remaining terms are actually identical, we want to combine them to save on compute
    # E.g.  2*x3*x1 + 4*x1*x3 is reduced into 6*x1*x3
    # For l_max=3, n_max=12, this reduces 5,152,520 terms to 2,512 (99% reduction)
    monomial_terms = torch.sort(monomial_terms, dim=1).values
    unique_terms, duplicate_map = torch.unique(monomial_terms, dim=0, return_inverse=True)
    unique_coefs = nonzero_coefs.new_zeros(unique_terms.shape[0])
    unique_coefs.scatter_add_(0, duplicate_map, nonzero_coefs)

    # When we combined those duplicate monomials, sometimes the new coeffecients equal 0, so just throw those out too
    # For l_max=3, n_max=12, this reduces 2,512 terms to 2,406 terms (4% reduction)
    nonzero_terms = unique_coefs != 0
    coefs_tensor = unique_coefs[nonzero_terms].to(dtype=torch.float32, device="cpu")
    terms_tensor = unique_terms[nonzero_terms].to(dtype=torch.int32, device="cpu")

    return coefs_tensor, terms_tensor

def compute_invariant_polynomial_collection(n_max, l_max, tensor_bases=cmaps, invariants=default_invariants_list, input_tensor_ordering=default_tensor_ordering):
    """
    For a given collection of invariants, specified as einsum strings, compute the collection of polynomials that corresponds to this collection
    of invariants.

    :param n_max: Maximum number of tensors that should be used to compute a single invariant. Any provided invariants
                  that use more than n_max tensors will be filtered out.

    :param l_max: Maximum tensor degree that should be used in an invariant. Any provided invariants that use tensors
                  of degree greater than l_max will be filtered out.

    :param tensor_bases: The set of bases for tensors of each order used in the invariants. This should be a dictionary where tensor_bases[l] stores the basis for irreducible tensors
                         of order l. tensor_bases[l] should be an l+1 dimensional tensor, where the final axis of tensor_bases[l] should index the the basis elements.
                         For example tensor_bases[2][0,0,3] will represent, in the basis of irreducible tensors of order 2, the (0,0) entry of the third basis element.
    
    :param invariant_code: The invariants that are computed. The invariants should be specified as a list of strings, where each string corresponds to a different invariant.
                           Each string should take the same form as an einsum representation, but with two differences. First, instead of listing variable names at the end of the einsum string,
                           one should instead use a string that can be used to uniquely identify the tensor that you want to contract. The name can be anything, except that it cannot
                           include commas. These names will correspond with a contiguous slice of the basis coefficients stored in X. For example, if Tensor1 corresponds to a rank
                           1 tensor, then the name Tensor1 might correspond to entries 1, 2, and 3 of X, such that x1, x2, and x3 store the basis coefficients for Tensor1.
                           The way that one associates these names with slices of X is specified in the parameter input_tensor_ordering. Second, the entire einsum representation should be
                           a string. Because the reduction must lead to a scalar, there should be a comma right after the arrow. Then, there should be the names
                           of the tensors. For example, it could look like this: 'ijk,ijk->,Tensor1,Tensor1'. To specify a tensor of rank zero, the string would
                           look like this: '->,Tensor0'. Any strings with the arrow omitted (e.g. 'ijk,ijk,Tensor1,Tensor1) are also acceptable.

    :param input_tensor_ordering: Specifies how the basis coefficient for each tensor will be inputted into each polynomial. Recall that
                                  the basis coefficients will be specified as some vector X = (x1,x2,...,xm), where each tensor's basis
                                  coefficients are stored in a contiguous slice of X. input_tensor_orderings is a list that should store
                                  all of the names of the tensors that were specified in the einsum strings of the invariants. The ordering
                                  of the names specifies the order in which the basis coefficients for each tensor appear in X. For example,
                                  if input_tensor_ordering is ['one', 'two', 'three'], this would mean that the basis coefficients for tensor
                                  'one' appears in X first, followed by the coefficients for 'two', and then the coefficients for 'three'.
    """

    # compute the order of each tensor (and make sure that the orders are listed consistently)
    tensor_orders = {}

    for i, invar in enumerate(invariants):

        invar_indices, invar_terms = split_invariant(invar)
        num_terms = len(invar_indices)

        for term_idx in range(num_terms):
            tensor_name = invar_terms[term_idx]
            tensor_order = len(invar_indices[term_idx])

            if tensor_name in tensor_orders:
                assert tensor_order == tensor_orders[tensor_name], f"Tensor {tensor_name} is used to represent two different orders: {tensor_orders[tensor_name]} and {tensor_order}."
            else:
                tensor_orders[tensor_name] = tensor_order

    # compute the dimension of the entire input.
    input_dimension = 0
    for k in tensor_orders:
        if tensor_orders[k] <= l_max:
            input_dimension += 2*tensor_orders[k]+1

    # Based on each order, compute the index in the vector X that corresponds to each tensor.
    # stored in the input_offsets.
    input_offsets = {}
    next_offset = 0

    for t in input_tensor_ordering:
        assert t in tensor_orders, f"Tensor {t} is never used in an invariant."

        input_offsets[t] = next_offset
        next_offset += 2*tensor_orders[t] + 1

    # Cut invariants out of the list based on n_max and l_max.
    invariants_kept = []
    for invar in invariants:
        _, invar_tensors = split_invariant(invar)
        n = len(invar_tensors)
        l = max( [tensor_orders[t] for t in invar_tensors] )

        if l <= l_max and n <= n_max:
            invariants_kept.append(invar)

    for l in tensor_bases:
        tensor_bases[l].requires_grad_(False)

    # compute the polynomials associated with every invariant that we are evaluating.
    coefs_set = []
    terms_set = []
    polynomial_sizes_set = []
    for i,invar in enumerate(invariants_kept):
        coefs, terms = computeInvariantPolynomial(tensor_bases, input_offsets, invar)
        coefs_set.append(coefs)
        terms_set.append(terms)
        polynomial_sizes_set.append(terms.shape[0])

    max_degrees = [ max( 
                            [ len(terms_set[i][j]) for j in range(len(terms_set[i])) ] 
                        ) for i in range(len(terms_set)) ]

    max_degree = max(max_degrees)

    terms_set_padded = []
    for i in range(len(coefs_set)):
        _, degree = terms_set[i].shape
        terms_set_padded.append(F.pad(terms_set[i], (0,max_degree-degree), value=-1))

    coefs = torch.hstack(coefs_set)
    terms = torch.vstack(terms_set_padded)
    polynomial_sizes = torch.IntTensor(polynomial_sizes_set)

    return PolynomialCollection( coefs, terms, polynomial_sizes, input_dimension )

class HopInvariantLayer(torch.nn.Module):
    """
    Pytorch layer that computes the invariants for HIP-HOP-NN. By default it will represent the invariants as a collection of polynomials
    and evaluate them using the custom triton kernel for evaluating polynomials. However, if triton or cuda are not available, or computing
    the polynomials with triton is disabled in the settings, then this layer will fall back on the pure PyTorch implementation specified in
    layers/hiplayers/tensors.py.

    :param n_max: The maximum number of tensors that should be used to compute any given invariant. Any
                  invariant that requires more than n_max tensors will be omitted.

    :param l_max: The maximum order of a tensor that should be used to compute an invariant. Any invariant
                  involving tensors of order greater than l_max will be omitted.

    :param cmaps_: The set of bases for tensors of each order used in the invariants. This should be a dictionary where tensor_bases[l] stores the basis for irreducible tensors
                   of order l. tensor_bases[l] should be an l+1 dimensional tensor, where the final axis of tensor_bases[l] should index the the basis elements.
                   For example tensor_bases[2][0,0,3] will represent, in the basis of irreducible tensors of order 2, the (0,0) entry of the third basis element.
    """

    def __init__(self, n_max, l_max, cmaps_=cmaps):
        super().__init__()
        self.l_max = l_max
        self.n_max = n_max
        self.cmaps = cmaps_

        # register buffer to allow us to check the device that the layer is set to
        # (allowing use of .to(), .cpu(), etc.)
        self.register_buffer("device_check", torch.empty(0))

        # will be used if polynomial invariants are active
        self.polynomials = None 

    def forward(self, tensor_features):
        device = self.device_check.device

        cmaps_device = self.cmaps[0].device
        if cmaps_device != device:
            for c in self.cmaps.keys():
                self.cmaps[c] = self.cmaps[c].to(device)

        if device == 'cpu' or not settings.USE_POLYNOMIAL_INVARIANTS or not triton_available_with_gather:
            return calc_invariants(self.l_max, self.n_max, tensor_features, self.cmaps)
        else:
            if self.polynomials is None:
                self.polynomials = compute_invariant_polynomial_collection(self.n_max, self.l_max, self.cmaps)
            self.polynomials.set_device(device)
            return EvaluatePolynomials.apply(tensor_features, self.polynomials)
        
    def __getstate__(self):
        state = self.__dict__.copy()
        # Make sure that self.polynomials is not saved, because this model may be reloaded onto a machine
        # where triton or cuda is not available. In that case, the polynomials object would be unnecessary.
        state["polynomials"] = None
        return state
