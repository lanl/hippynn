import torch
import itertools

import torch.nn.functional as F

from ...custom_kernels.poly_triton import EvaluatePolynomials
from .tensors import cmaps

def computeInvariantPolynomial(C, invariant_input_offsets, invariant_code):
    """
    For a given invariant defined by contractions of irreducible moment tensors, represents that invariant
    as a multivariate polynomial.

    We assume that all relevant irreducible tensors of order l are represented as coefficients that correspond to a
    basis Bl for the space of irreducible tensors of order l. Furthermore, we assume that the basis coefficients
    of all of moment tensors are concatenated into a single vector X = (x1, x2, ..., xn), where the basis coefficients
    of each tensor corresponds to a contiguous slice of the entries of X. (for example, the first entry of X could correspond
    to a tensor of order 0, then the next three correspond to a tensor of order 1, etc.).

    The polynomial that is returned is a function of the entries of X (e.g. 2*x1*x2 + 3*x3*x3).

    :param C: The set of bases for tensors of order l. This should be a dictionary where C[l] stores the basis for irreducible tensors
              of order l. C[l] should be an l+1 dimensional tensor, where C[l][i] stores the i-th basis element.

    :param invariant_input_offsets: A dictionary that is used to keep track of which parts of the vector X correspond to the various tensors.
                                    The keys are a unique name for each tensor - it can be anything that the user wants. The invariant_input_offsets[T]
                                    stores the location in the vector X that corresponds to the basis coefficients for tensor T.

    :param invariant_code: A code that specifies that reduction that is being computed. It should take the same form
                           as an envsum representation, but with two differences. First, instead of listing variable names at the end of the envsum string,
                           one should instead use the tensor names specified in invariant_input_offsets. Second, the entire envsum representation should be
                           a string. Because the reduction must lead to a scalar, there should be a comma right after the arrow. Then, there should be the names
                           of the tensors. For example, it could look like this: 'ijk,ijk->,Tensor1,Tensor2'. To specify a tensor of rank zero, the string would
                           look like this: '->,Tensor0'.

    :return: The polynomial that can be used to compute the invariant from the vector X. The polynomial is specified by its monomials using two arrays. 
             The first array stores the coefficients of all monomials. For instance, if the polynomial is 2*x1*x3 + 4*x1*x1, the list of coefficients would store [2,4].
             The second array stores the indices of the terms in the monomial. It is a 2D array where each row corresponds to a different monomial. For instance, for the
             polynomial 2*x1*x3 + 4*x1*x1, the first row will store [1,3], and the second row will store [1,1].
    """

    # For the zero order invariant, there is no einsum, it is just the monomial x0.
    if invariant_code[0:2] == "->":
        tensor_name = invariant_code[3:]
        offset = invariant_input_offsets[tensor_name]
        return torch.FloatTensor((1,)), torch.IntTensor(((offset,),))

    # otherwise, build an einsum string corresponding to our contraction of the C tensors:
    alpha = "abcdefghijklmnopqrstuvwxyz"
    back_idx = 25
    front_idx = 0


    einsum_front_terms = []
    einsum_back = ""
    tensors_to_contract = []

    # first, construct one term for each code.
    # keep track of which tensors correspond to which terms.
    for idx,reduction in enumerate(invariant_code):
        einsum_front_terms.append(alpha[back_idx])
        einsum_back += alpha[back_idx]
        back_idx -= 1
        assert back_idx > front_idx,"Ran out of leters for einsum."
        
        tensors_to_contract.append( C[ len(reduction)-1 ] )

    # now add all of the einsum terms corresponding to the reductions
    for idx,reduction in enumerate(invariant_code):
        for other_idx in reduction[1:]:
            if other_idx > idx:
                next_letter = alpha[front_idx]
                front_idx += 1
                assert back_idx > front_idx,"Ran out of letters for einsum."

                einsum_front_terms[idx] += next_letter
                einsum_front_terms[other_idx] += next_letter

    einsum_front = einsum_front_terms[0]
    for einsum_term in einsum_front_terms[1:]:
        einsum_front += "," + einsum_term

    einsum_string = einsum_front + "->" + einsum_back
    coef_tensor = torch.einsum(einsum_string, *tensors_to_contract)

    iter_range = []
    for dim in coef_tensor.shape:
        iter_range.append(range(dim))

    coefs = {}

    tensor_idxs = itertools.product(*iter_range)
    for coordinates in tensor_idxs:

        value = coef_tensor[*coordinates]
        if value != 0:

            key_list = []
            for dim,coord in enumerate(coordinates):
                key_list.append( invariant_input_offsets[ invariant_code[dim][0] ] + coord )

            key_list.sort()
            key = tuple(key_list)
            if key in coefs:
                coefs[key] += value
            else:
                coefs[key] = value
    
    # delete all zero coefficients
    delete = []
    for c in coefs:
        if coefs[c] == 0:
            delete.append(c)
    
    for c in delete:
        del coefs[c]

    # create the tensor of coefficients
    coefs_tensor = torch.zeros(len(coefs),dtype=torch.float32)
    terms_tensor = torch.zeros((len(coefs),len(invariant_code)),dtype=torch.int32)
    
    for row, coef in enumerate(coefs):
        coefs_tensor[row] = coefs[coef]
        for col,entry in enumerate(coef):
            terms_tensor[row,col] = entry

    return coefs_tensor, terms_tensor

# invariants are specified as a tuple. The first is a tensor name. Then, there is a list of which other tensors we contract with.
# the number of other tensors that we contract with should match the degree
default_invariants_list = [
    ( ("zero",), ),
    ( ("one",1), ("one",0) ),
    ( ("two",1,1), ("two",0,0) ),
    ( ("two",1,2), ("two",0,2), ("two",0,1) ),
    ( ("three",1,1,1), ("three",0,0,0) ),
    ( ("three",1,1,2), ("three",0,0,3), ("three", 3,3,0), ("three", 2,2,1) ),
    ( ("one",1), ("two",0,2), ("one",1) ),
    ( ("one",1), ("two",0,2), ("two",1,3), ("one",2) ),
    ( ("three",1,2,3), ("one",0), ("one",0), ("one",0) ),
    ( ("one",1), ("three",0,2,2), ("three",3,1,1), ("one",2) ),
    ( ("two",1,2), ("three",0,2,2), ("three",0,1,1) ),
    ( ("two",1,2),("two",0,3),("three",0,3,3),("three",1,2,2) ),
    ( ("two",1,1),("three",0,0,2),("three",1,3,3),("two",2,2) ),
]

default_input_offsets = {
    "zero" : 0,
    "one" : 1,
    "two" : 4,
    "three": 9
}

class PolynomialInvariants(torch.nn.Module):
    def __init__(self, n_max, l_max, _cmaps=cmaps, possible_invars=default_invariants_list, input_offsets=default_input_offsets):
        super().__init__()
        self.l_max = l_max
        self.n_max = n_max

        cmaps = [
            _cmaps[0],
            _cmaps[1].permute(1,0),
            _cmaps[2].permute(2,0,1).reshape(5,3,3),
            _cmaps[3].permute(3,0,1,2).reshape(7,3,3,3),
        ]

        invars = []
        for possible_invar in possible_invars:
            l = max( [len(tup) for tup in possible_invar] ) - 1
            n = len(possible_invar)

            if l <= l_max and n <= n_max:
                invars.append(possible_invar)

        for c in cmaps:
            c.requires_grad_(False)

        coefs_set = []
        terms_set = []
        polynomial_sizes_set = []
        for i,invar in enumerate(invars):
            coefs, terms = computeInvariantPolynomial(invar, cmaps, input_offsets)
            coefs_set.append(coefs)
            terms_set.append(terms)
            polynomial_sizes_set.append(terms.shape[0])

        max_degrees = [ max( 
                                [ len(terms_set[i][j]) for j in range(len(terms_set[i])) ] 
                            ) for i in range(len(terms_set))]

        max_degree = max(max_degrees)

        terms_set_padded = []
        for i in range(len(coefs_set)):
            _, degree = terms_set[i].shape
            terms_set_padded.append(F.pad(terms_set[i], (0,max_degree-degree), value=-1))

        self.register_buffer( "coefs", torch.hstack(coefs_set).contiguous() )
        self.register_buffer( "terms", torch.vstack(terms_set_padded).contiguous().to(torch.int16) )
        self.register_buffer( "polynomial_sizes", torch.IntTensor(polynomial_sizes_set) )
        self.id_number = torch.IntTensor(0)

    def forward(self, tensor_features):
        return EvaluatePolynomials.apply(tensor_features, self.coefs, self.terms, self.polynomial_sizes, self.id_number)