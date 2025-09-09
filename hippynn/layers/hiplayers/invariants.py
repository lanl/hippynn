import torch
import torch.nn.functional as F

from .tensors import calc_invariants, cmaps

from ... import settings

import itertools
import string

try:
    from ...custom_kernels.poly_triton import EvaluatePolynomials, PolynomialCollection
    triton_available = True
except:
    triton_available = False

# tensor ordering and list of invariants for the default invariants.
default_invariants_ordering = ["zero", "one", "two", "three"]

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

    :param invariant_input_offsets: A dictionary that is used to keep track of which parts of the vector X correspond to the various tensors.
                                    The keys are a unique name for each tensor - it can be anything that the user wants. The invariant_input_offsets[T]
                                    stores the location in the vector X that corresponds to the basis coefficients for tensor T.

    :param invariant_code: The invariant that is computed, represented using einsum notation. Here the names of the tensors specified in the einsum string should be keys
                           of invariant_input_offsets.

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

    # Will store the monomials. The keys will be a list of all of the indices (sorted), and the values
    # will be the corresponding coefficients. We sort the indices so that "like terms" are combined e.g.
    # x1*x2 is treated as the same as x2*x1 (both are represented by the list [1,2]).
    coefs = {}

    # Create an iterator that can be used to iterate through every index of the coef_tensor
    iter_range = []
    for dim in coef_tensor.shape:
        iter_range.append(range(dim))

    tensor_idxs = itertools.product(*iter_range)

    # Iterate through coef_tensor to fill out the coefs dictionary.
    for coordinates in tensor_idxs:

        coef = coef_tensor[*coordinates]
        if coef != 0:

            # Based on the coordinates in coef_tensor, find the terms in the monomial that correspond
            # using the input offsets.
            key_list = []
            for dim,coord in enumerate(coordinates):
                key_list.append( invariant_input_offsets[ invar_tensors[dim] ] + coord )

            key_list.sort()
            key = tuple(key_list)
            if key in coefs:
                coefs[key] += coef
            else:
                coefs[key] = coef

    # remove all monomials whose coefficient is zero.
    delete = []
    for c in coefs:
        if coefs[c] == 0:
            delete.append(c)
    
    for c in delete:
        del coefs[c]

    # create the tensor of coefficients
    coefs_tensor = torch.zeros(len(coefs),dtype=torch.float32)
    terms_tensor = torch.zeros((len(coefs),num_terms),dtype=torch.int32)
    
    for row, coef in enumerate(coefs):
        coefs_tensor[row] = coefs[coef]
        for col,entry in enumerate(coef):
            terms_tensor[row,col] = entry

    return coefs_tensor, terms_tensor

def compute_invariant_polynomial_collection(n_max, l_max, tensor_bases=cmaps, invariants=default_invariants_list, input_tensor_ordering=default_invariants_ordering):
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

        if device == 'cpu' or not settings.USE_POLYNOMIAL_INVARIANTS or not triton_available:
            return calc_invariants(self.l_max, self.n_max, tensor_features, self.cmaps)
        else:
            if self.polynomials is None:
                self.polynomials = compute_invariant_polynomial_collection(self.n_max, self.l_max, self.cmaps)
            self.polynomials.set_device(device)
            return EvaluatePolynomials.apply(tensor_features, self.polynomials)