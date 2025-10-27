import torch
import torch.nn.functional as F

import triton
import triton.language as tl

"""
This module pertains to simultaneously evaluating many multivariate polynomials defined on common inputs.
Each input point X is specified as a vector e.g. (x1,x2,x3). Each multivariate polynomial is defined on the entries
of each point. For example, we could have p_1 = 3*x1*x1 + x2*x3.

If one inputs a batch of points and a set of polynomials, then each polynomial
will be evaluated on each point. For example, if one inputs points X1 and X2, and polynomials p1 and p2,
then p1(X1), p1(X2), p2(X1), and p2(X2) will be computed.

The points are specified as a 2D tensor where each row corresponds to a different point X, and each column corresponds to a different entry of X.

The polynomials are specified in terms of the monomials. For example if p1 = x1*x2 + 4*x2*x2 and p2 = 3*x1*x1, then
our monomials would be x1*x2, 4*x2*x2, and 3*x1*x1. The polynomials are then stored in three different tensors:

The first tensor stores all of the coefficients of the monomials. In the example above, the coefficient list would store [1,4,3]. In various functions, this is typically
referred to as "coefs".

The second tensor stores the indices of the terms in each monomial. It is a 2D tensor, where each row corresponds to a different monomial,
and each column corresponds to a separate term. For example, the row associated with the monomial x1*x2 would store [1,2].
If the width of a row is greater than the number of terms, then the remaining entries in the row should be filled with -1. So if each row had four
entries, then the monomial x1*x2 could be represented as [1,2,-1,-1]. In various functions, this is typically referred to as "terms".

The final tensor stores how many monomials are in each polynomial. For example, if we have two monomials: p1 and p2, where p1 has two monomials, and p2 has one monomial, then
this list would store [2,1]. In various functions, this is typically referred to as "polynomial_sizes".

When the functions in this module are used, the polynomials should be wrapped in the PolynomialCollection class,
which wraps the coefficients, terms, and polynoimal sizes. It also caches derivatives when they are computed.
"""

class PolynomialCollection():

    """
    Wraps a collection of polynomials defined by a list of coefficients and terms.

    :param coefs: The coefficients of the monomials for this set of polynomials.

    :param terms: The terms of the monomials for this set of polynomials.

    :param polynomial_sizes: The number of monomials in each polynomial.

    :param input_dimension: The length of the vector X that will be used as the input for each polynomial.
    """

    def __init__(self, coefs, terms, polynomial_sizes, input_dimension):
        self.polynomials = {0 : (coefs, terms, polynomial_sizes, input_dimension)}
        self.max_derivative_level = 0

    def get_device(self):
        """
        Returns the device that each tensor member variable is currently stored on.
        """
        return self.polynomials[0][0].device
    
    def set_device(self,device):
        """
        Changes the device that each tensor member variable is stored on.
        """
        if self.get_device() != device:
            for i in range(self.max_derivative_level+1):
                coefs, terms, polynomial_sizes, input_dimension = self.polynomials[i]
                self.polynomials[i] = (coefs.to(device), terms.to(device), polynomial_sizes.to(device), input_dimension)
    
    def get_polynomials(self,derivative_level=0):
        """
        Returns the polynomials associated with this collection, or one of their analytic derivatives.

        :param derivative_level: How many successive derivatives should be computed. If this is set to 0, the
                                 original polynomial collection will be returned. Otherwise, for example, if derivative_level
                                 is set equal to 2, then the second derivative will be returned.

        :return: A tuple containing the coefficients, terms, polynomial sizes, and input dimension of the returned polynomial.
        """

        if derivative_level <= self.max_derivative_level:
            return self.polynomials[derivative_level]
        else:
            previous_derivative_level = self.get_polynomials(derivative_level-1)
            previous_coefs, previous_terms, previous_polynomial_sizes, previous_input_dimension = previous_derivative_level
            derivative = compute_derivative(previous_coefs, previous_terms, previous_polynomial_sizes, previous_input_dimension)
            self.polynomials[derivative_level] = derivative
            self.max_derivative_level = derivative_level
            return derivative

def compute_derivative(coefs_,terms_,polynomial_sizes_,input_dimension):
    """
    Suppose that, for a point X = ( x1, x2, ..., xn ), we evaluate many multivate polynomials p1, p2, ..., pm on X. Then, we derive some loss value L
    from all of the outputs of the polynomials. Then the partial derivative of L with respect to each entry of X is itself a multivariate polynomial.

    This function computes all of these partial derivatives, and represents them as a set of multivatiate polynomials. The input X' of each new multivatiate
    polynomial will be the values of the point X, along with all of the partial derivatives dL/dp1, dL/dp2, etc. When evaluating, one should concatenate 
    the partial derivatives dL/dpi onto the point X to form X' = (x1, x2, ..., xn, dL/dp1, dL/dp2, ..., dL/dpm ).

    :param coefs_: The coefficients of the monomials of the set of polynomials that we are taking the derivative of.

    :param terms_: The terms of the monomials of the set of polynomials that we are taking the derivative of.

    :param polynomial_sizes_: The polynomial sizes of the set of polynomials that we are taking the derivative of.

    :input_dimension: The number of entries in each point X. If X = ( x1, x2, ..., xn ), then this would be equal to n.

    :return: The functions for computing the partial derivatives. The polynomials are specified as a tuple containing coefficients, terms, and polynomial sizes.
    """

    # key is a tuple of 2 entries. First entry is the partial derivative that it belongs to (e.g. 1 if its for dL/dx1)
    # The second entry is a tuple of all of the terms of the monomial.
    derivatives_monomials = {}
    
    # Make copies of these tensors to prevent issues with autograd.
    coefs = coefs_.detach().clone()
    terms = terms_.detach().clone()
    polynomial_sizes = polynomial_sizes_.detach().clone()

    num_monomials, degree = terms.shape
    num_polynomials = len(polynomial_sizes)
    polynomial_sizes = list(polynomial_sizes)
    
    # Scan through each monomial, taking the derivative w.r.t. each term in each monomial. The results are accumulated in derivatives_monomials.

    monomial_idx = 0 # this keeps track of whichever monomial that we are currently processing.
    for poly in range(num_polynomials):
        for repeat in range(polynomial_sizes[poly]):
            monomial = terms[monomial_idx]
            for entry1 in range(len(monomial)): # This is the term that we are deriving with respect to
                if monomial[entry1] >= 0:
                    derivative_terms = []

                    for entry2 in range(len(monomial)): # This is other terms that remain in the derivative
                        if entry2 != entry1 and monomial[entry2] >= 0:
                            derivative_terms.append(monomial[entry2].item())

                    derivative_terms.append(poly + input_dimension)

                    # we sort the list of terms in order to combine monomials whose terms are in different orders.
                    # e.g. x1*x2 is recognized as being the same as x2*x1
                    derivative_terms.sort()

                    term_key = (monomial[entry1].detach().item(), tuple(derivative_terms))
                    if term_key in derivatives_monomials:
                        derivatives_monomials[term_key] = derivatives_monomials[term_key] + coefs[monomial_idx]
                    else:
                        derivatives_monomials[term_key] = coefs[monomial_idx]
            monomial_idx += 1

    # Create the list of coefficients and terms for each partial derivative.
    # This is derived from derivatives_monomials.
    derivatives_coefs = []
    derivatives_terms = []

    for i in range(input_dimension):
        derivatives_coefs.append([])
        derivatives_terms.append([])

    for monomial in derivatives_monomials:
        if derivatives_monomials[monomial] != 0.0:
            derivatives_coefs[monomial[0]].append(derivatives_monomials[monomial])
            derivatives_terms[monomial[0]].append(monomial[1])

    derivatives_degrees = []
    for i in range(input_dimension):
        if len(derivatives_terms[i]) == 0:
            derivatives_degrees.append(0)
        else:
            derivatives_degrees.append(max( [len(t) for t in derivatives_terms[i]] ))

    # compute the max degree. Compute the sizes of each polynomial, pad the monomials to all have the same degree, and convert everything else to tensors.
    max_derivative_degree = triton.next_power_of_2(max(derivatives_degrees))

    derivatives_polynomial_sizes = []
    derivatives_terms_padded = []

    for i in range(input_dimension):
        polynomial_size = len(derivatives_coefs[i])
        derivatives_polynomial_sizes.append(polynomial_size)
        derivatives_coefs[i] = torch.FloatTensor(derivatives_coefs[i])

        if polynomial_size > 0:

            padded_terms_tuples = []
            for term in derivatives_terms[i]:
                term = term + (-1,) * (max_derivative_degree - len(term))
                padded_terms_tuples.append(term)

            derivatives_terms_padded.append(torch.IntTensor(padded_terms_tuples))

    # Prepare the output tensors.
    derivatives_coefs_out = torch.hstack(derivatives_coefs).to(coefs.device)
    derivatives_terms_out = torch.vstack(derivatives_terms_padded).contiguous().to(coefs.device)
    derivatives_polynomial_sizes_out = torch.IntTensor(derivatives_polynomial_sizes).to(coefs.device)
    derivative_input_dimension = num_polynomials + input_dimension

    return derivatives_coefs_out, derivatives_terms_out, derivatives_polynomial_sizes_out, derivative_input_dimension

class EvaluatePolynomials(torch.autograd.Function):

    """
    Differentiable function whose purpose is to evaluate a set of multivatiate polynomials on a set of input points.
    The derivatives are cached in the dictionary derivative_cache. To distinguish between different levels of derivatives
    of the same set of polynomials (e.g. the first derivative and second derivative should be cached separately), which level
    of differentiation must also be specified.

    Since the derivative can be specified as a polynomial, the backward call of this function call's its own
    forward call in order to evaluate the derivative, insuring that it is infinitely differentiable.

    :param x: The set of points being inputted.

    :param polynomials: A PolynomialCollection storing the polynomials that should be evaluated.

    :param derivative_level: Which level of derivative this current set of polynomials represents. For example, if the set
                             of polynomials represents the first derivative, it should take the value of 1. If it is not a 
                             derivative, then it should take the value of 0.

    :return: The set of polynomials evaluated on the set of points x.
    """

    @staticmethod
    def forward(ctx, x, polynomials, derivative_level=0 ):

        # save terms for later
        ctx.save_for_backward(x)
        ctx.polynomials = polynomials
        ctx.derivative_level = derivative_level

        coefs, terms, polynomial_sizes, _ = polynomials.get_polynomials(derivative_level)

        num_polynomials = len(polynomial_sizes)

        # compute all relevant dimensions for differentiable terms
        num_points, input_dimension = x.shape
        input_dimension_rounded_up = triton.next_power_of_2(input_dimension)

        num_monomials, degree = terms.shape
        degree_rounded_up = triton.next_power_of_2(degree)

        # Pad the 2D tensors (X and terms) so that the number of columns is a power of 2
        if input_dimension != input_dimension_rounded_up:
            x = F.pad( x, (0, (input_dimension_rounded_up - input_dimension)) ).contiguous()

        if degree != degree_rounded_up:
            terms = F.pad( terms, (0, (degree_rounded_up - degree)), value=-1 ).contiguous()

        # run the kernel
        output = torch.zeros( (num_points,num_polynomials), dtype=x.dtype, device=x.device, requires_grad=True )

        grid = lambda meta: (
            triton.cdiv(num_points, meta['NUM_POINTS_TO_LOAD']),
        )

        if x.dtype == torch.float64:
            kernel_dtype = tl.float64
        else:
            kernel_dtype = tl.float32

        evaluate_polynomials_kernel[grid]( x, coefs, terms, polynomial_sizes, output, num_polynomials, num_monomials, input_dimension_rounded_up, num_points, degree_rounded_up, dtype=kernel_dtype )

        return output

    @staticmethod
    def backward(ctx, grad_output):

        x = ctx.saved_tensors[0]
        polynomials = ctx.polynomials
        derivative_level = ctx.derivative_level

        d_x = torch.hstack((x,grad_output)).contiguous()

        derivative_calc_output = EvaluatePolynomials.apply( d_x, polynomials, derivative_level+1 )

        return derivative_calc_output, None, None

def get_configs_polynomials():
    """
    Generates a list of combinations of hyperparameters to use when autotuning evaluate_polynomials_kernel.
    TODO: reduce the number of configurations to a more manageable size to decrease compile time.
    """
    configs = []
    for num_points_to_load in [128,256,512]:
        for num_monomials_to_load in [8,16,32]:
            for num_warps in [2,4,8,16,32]:
                for num_stages in [2,3,4,5,6]:
                    configs.append(triton.Config(kwargs={"NUM_POINTS_TO_LOAD" : num_points_to_load, "NUM_MONOMIALS_TO_LOAD" : num_monomials_to_load}, num_warps=num_warps, num_stages=num_stages))
                    return configs
    return configs

@triton.jit
def prod(x,y):
    """
    Helper function for evaluate_polynomials_kernel. Takes the product of the two inputs x and y.
    This function is defined for use alongside tl.reduce in evaluate_polynomials_kernel.
    """
    return x*y

@triton.autotune(configs=get_configs_polynomials(), key=["num_monomials"])
@triton.jit
def evaluate_polynomials_kernel(
    input_ptr,
    coefs_ptr,
    terms_ptr,
    polynomial_sizes_ptr,
    output_ptr,
    num_polynomials : tl.constexpr,
    num_monomials : tl.constexpr,
    input_dimension_rounded_up : tl.constexpr,
    num_points,
    degree_rounded_up : tl.constexpr,
    NUM_POINTS_TO_LOAD : tl.constexpr,
    NUM_MONOMIALS_TO_LOAD : tl.constexpr,
    dtype:tl.constexpr = tl.float32,
):
    """
    Triton kernel that is used to simultaneously evaluate many multivariate polynomials on a batch of points.

    :param input_ptr: Pointer to the array of input points.

    :param coefs_ptr: Pointer to the array of coefficients of the monomials.

    :param terms_ptr: Pointer to the array of terms. The number of columns MUST be a power of 2. Pad using -1 if necessary.

    :param polynomial_sizes_ptr: Pointer to the array of polynomial sizes.

    :param output_ptr: Pointer to the array that will store the output values. Each row corresponds to a different input point X_i. Each column corresponds to a different polynomial p_j.
                       Thus, the ij entry of the output_ptr will be p_j(X_i).

    :param num_monomials: The total number of monomials across all polynomials. This should equal len(coefs_ptr).

    :param input_dimension_rounded_up: Let n represent the number of entries of each X_i (so for (1,2,3), n=3). Then input_dimension_rounded_up is the smallest power of 2 that is larger than n.

    :param num_points: The number of points X_i, equal to the number of rows in the tensor pointed to by input_ptr.

    :param degree_rounded_up: Let n represent the maximum degree of all of the polynomials. Then degree_rounded_up is the smallest power of 2 that is larger than n. This is also the number of 
                        columns in the tensor pointed to by terms_ptr.

    :param NUM_POINTS_TO_LOAD: hyperparameter. This specifies the number of points to evaluate monomials on at the same time.

    :param NUM_MONOMIALS_TO_LOAD: hyperparameter. This specifies the number of monomials to evaluate at the same time.

    :param dtype: Specifies whether the output will be float32 or float64.
    """

    num_monomials_indexer = tl.arange(0,NUM_MONOMIALS_TO_LOAD)[None,:,None]

    # each PID is associated with a certain subset of the points X.
    x_pid = tl.program_id(0)

    # Load the slice of points X.
    x_start = x_pid * NUM_POINTS_TO_LOAD * input_dimension_rounded_up
    x_offsets = x_start + tl.arange(0,NUM_POINTS_TO_LOAD * input_dimension_rounded_up)
    x_mask = x_offsets < num_points * input_dimension_rounded_up

    x = tl.load(input_ptr + x_offsets, mask=x_mask)
    x = x.reshape((NUM_POINTS_TO_LOAD,input_dimension_rounded_up))

    # Compute output offsets. The polynomials are computed one at a time. Thus, these offsets
    # are used to store a single polynomial evaluated on many different points.
    point_offsets = x_pid * NUM_POINTS_TO_LOAD + tl.arange(0,NUM_POINTS_TO_LOAD)
    point_mask = point_offsets < num_points

    output_offsets = point_offsets * num_polynomials

    which_batch = 0 # index of the first monomial is loaded in the current batch of monomials.
    for poly in range(num_polynomials):

        polynomial_size = tl.load(polynomial_sizes_ptr + poly)        
        output = tl.zeros( (NUM_POINTS_TO_LOAD,), dtype=dtype )

        num_monomial_loops = tl.cdiv( polynomial_size, NUM_MONOMIALS_TO_LOAD )

        for monomial_idx in range(num_monomial_loops):
            # load the coefficients associated with the current monomials
            coef_arange = tl.arange(0,NUM_MONOMIALS_TO_LOAD) + which_batch
            coef_mask = coef_arange < num_monomials
            coef = tl.load( coefs_ptr + coef_arange, mask=coef_mask )[None,:]
            coef = coef.broadcast_to( (NUM_POINTS_TO_LOAD,NUM_MONOMIALS_TO_LOAD) )

            # load the terms (e.g. x_1, x_3, etc.) associated with each monomial.
            terms_arange = tl.arange(0,NUM_MONOMIALS_TO_LOAD * degree_rounded_up) + which_batch * degree_rounded_up
            terms_mask = terms_arange < num_monomials * degree_rounded_up
            terms_index = tl.load( terms_ptr + terms_arange, mask=terms_mask )

            terms_index = terms_index.reshape((1,NUM_MONOMIALS_TO_LOAD * degree_rounded_up))
            terms_index = terms_index.broadcast_to( (NUM_POINTS_TO_LOAD, NUM_MONOMIALS_TO_LOAD * degree_rounded_up) )

            # First, evaluate each term (e.g. x_1) on each point X_i. Then, take the product of all of these terms with each other,
            # and with the coefficients, in order to compute a value for each monomial. Every time that an entry of the terms is
            # equal to -1, rather than taking some value from the X_i, its value is replaced with the number 1, which will have no effect
            # on the product of the other terms.

            terms = tl.gather( x, terms_index, 1 )
            terms = tl.where( terms_index >= 0, terms, 1 )

            terms = terms.reshape((NUM_POINTS_TO_LOAD, NUM_MONOMIALS_TO_LOAD, degree_rounded_up))
            terms = tl.where(monomial_idx * NUM_MONOMIALS_TO_LOAD + num_monomials_indexer < polynomial_size, terms, 0 )
            terms = tl.reduce( terms, 2, prod )

            terms *= coef
            terms = tl.sum( terms, axis=1 )
            
            output += terms

            # It is possible that we loaded more monomials than are left in the current polynomial, and some of the monomials that
            # we loaded are associated with the next polynomial. If that happens, we will ignore those monomials during the current
            # calculation (using masking). However, in such an event, the first monomial to load during the next batch should be
            # the first monomial of the next polynomial.

            which_batch += min( polynomial_size - monomial_idx * NUM_MONOMIALS_TO_LOAD, NUM_MONOMIALS_TO_LOAD )

        tl.store(output_ptr + output_offsets + poly, output, mask=point_mask)