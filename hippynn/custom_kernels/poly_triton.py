import torch
import torch.nn.functional as F

import triton
import triton.language as tl

@triton.jit
def prod(x,y):
    return x*y

def get_configs_polynomials():
    configs = []
    for num_points_to_load in [128,256,512]:
        for num_monomials_to_load in [8,16,32]:
            for num_warps in [2,4,8,16,32]:
                for num_stages in [2,3,4,5,6]:
                    configs.append(triton.Config(kwargs={"NUM_POINTS_TO_LOAD" : num_points_to_load, "NUM_MONOMIALS_TO_LOAD" : num_monomials_to_load}, num_warps=num_warps, num_stages=num_stages))

    return configs

@triton.autotune(configs=get_configs_polynomials(), key=["num_monomials_rounded_up"])
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

    num_monomials_rounded_up : tl.constexpr = triton.next_power_of_2(num_monomials)

    x_pid = tl.program_id(0)

    monomials_coefs_offsets = tl.arange(0,num_monomials_rounded_up)
    monomials_coefs_mask = monomials_coefs_offsets < num_monomials
    monomials_coefs = tl.load( coefs_ptr + monomials_coefs_offsets, mask=monomials_coefs_mask )

    num_monomials_indexer = tl.arange(0,NUM_MONOMIALS_TO_LOAD)[None,:,None]

    x_start = x_pid * NUM_POINTS_TO_LOAD * input_dimension_rounded_up
    x_offsets = x_start + tl.arange(0,NUM_POINTS_TO_LOAD * input_dimension_rounded_up)
    x_mask = x_offsets < num_points * input_dimension_rounded_up

    x = tl.load(input_ptr + x_offsets, mask=x_mask)
    x = x.reshape((NUM_POINTS_TO_LOAD,input_dimension_rounded_up))

    point_offsets = x_pid * NUM_POINTS_TO_LOAD + tl.arange(0,NUM_POINTS_TO_LOAD)
    point_mask = point_offsets < num_points

    output_offsets = point_offsets * num_polynomials

    which_batch = 0
    for poly in range(num_polynomials):
        polynomial_size = tl.load(polynomial_sizes_ptr + poly)
        
        output = tl.zeros( (NUM_POINTS_TO_LOAD,), dtype=dtype )

        num_monomial_loops = polynomial_size // NUM_MONOMIALS_TO_LOAD
        if polynomial_size % NUM_MONOMIALS_TO_LOAD != 0:
            num_monomial_loops += 1

        for monomial_idx in range(num_monomial_loops):
            coef_arange = tl.arange(0,NUM_MONOMIALS_TO_LOAD) + which_batch
            coef_mask = coef_arange < num_monomials
            coef = tl.load( coefs_ptr + coef_arange, mask=coef_mask )[None,:]
            coef = coef.broadcast_to( (NUM_POINTS_TO_LOAD,NUM_MONOMIALS_TO_LOAD) )

            terms_arange = tl.arange(0,NUM_MONOMIALS_TO_LOAD * degree_rounded_up) + which_batch * degree_rounded_up
            terms_mask = terms_arange < num_monomials * degree_rounded_up
            terms_index = tl.load( terms_ptr + terms_arange, mask=terms_mask )

            terms_index = terms_index.reshape((1,NUM_MONOMIALS_TO_LOAD * degree_rounded_up))
            terms_index = terms_index.broadcast_to( (NUM_POINTS_TO_LOAD, NUM_MONOMIALS_TO_LOAD * degree_rounded_up) )

            terms = tl.gather( x, terms_index, 1 )
            terms = tl.where( terms_index >= 0, terms, 1 )

            terms = terms.reshape((NUM_POINTS_TO_LOAD, NUM_MONOMIALS_TO_LOAD, degree_rounded_up))
            terms = tl.where(monomial_idx * NUM_MONOMIALS_TO_LOAD + num_monomials_indexer < polynomial_size, terms, 0 )
            terms = tl.reduce( terms, 2, prod )

            terms *= coef
            terms = tl.sum( terms, axis=1 )
            
            output += terms
            which_batch += min( polynomial_size - monomial_idx * NUM_MONOMIALS_TO_LOAD, NUM_MONOMIALS_TO_LOAD )

        tl.store(output_ptr + output_offsets + poly, output, mask=point_mask)

derivative_cache = {}

def compute_derivative(coefs_,terms_,polynomial_sizes_,input_dimension):
    derivatives_monomials = {} # key is a tuple of 3 entries. First entry is the polynomial that it will ultimately belong to. The second entry is a tuple of all of the nondifferentiable terms. The third entry is a tuple of all of the differentiable terms.
    
    # Make copies of these tensors to prevent issues with autograd.
    coefs = coefs_.detach().clone()
    terms = terms_.detach().clone()
    polynomial_sizes = polynomial_sizes_.detach().clone()

    num_monomials, degree = terms.shape
    num_polynomials = len(polynomial_sizes)
    polynomial_sizes = list(polynomial_sizes)
    
    # Scan through each monomial, taking the derivative w.r.t. each term in each monomial. Accumulate the results.
    monomial_idx = 0
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

                    derivative_terms.sort()

                    term_key = (monomial[entry1].detach().item(), tuple(derivative_terms))
                    if term_key in derivatives_monomials:
                        derivatives_monomials[term_key] = derivatives_monomials[term_key] + coefs[monomial_idx]
                    else:
                        derivatives_monomials[term_key] = coefs[monomial_idx]
            monomial_idx += 1

    # Create lists of coefficients, terms, and nondifferentiable terms for each partial derivative from the above results.
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

    return derivatives_coefs_out, derivatives_terms_out, derivatives_polynomial_sizes_out


class EvaluatePolynomials(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, coefs, terms, polynomial_sizes, poly_idx, derivative_level=0 ):
        # save terms for later
        ctx.save_for_backward(x, coefs, terms, polynomial_sizes )
        ctx.poly_idx = poly_idx
        ctx.derivative_level = derivative_level

        num_polynomials = len(polynomial_sizes)

        # compute all relevant dimensions for differentiable terms
        num_points, input_dimension = x.shape
        input_dimension_rounded_up = triton.next_power_of_2(input_dimension)

        num_monomials, degree = terms.shape
        degree_rounded_up = triton.next_power_of_2(degree)
        num_monomials_rounded_up = triton.next_power_of_2(num_monomials)

        if input_dimension != input_dimension_rounded_up:
            x = F.pad( x, (0, (input_dimension_rounded_up - input_dimension)) ).contiguous() # :(

        if degree != degree_rounded_up:
            terms = F.pad( terms, (0, (degree_rounded_up - degree)), value=-1 ).contiguous() # :((

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

        x, coefs, terms, polynomial_sizes = ctx.saved_tensors
        poly_idx = ctx.poly_idx
        derivative_level = ctx.derivative_level

        if (poly_idx, derivative_level) in derivative_cache:
            derivative = derivative_cache[(poly_idx,derivative_level)]
        else:
            input_dimension = x.shape[1]

            derivative = compute_derivative(coefs, terms, polynomial_sizes, input_dimension)
            derivative_cache[(poly_idx,derivative_level)] = derivative

        d_coefs, d_terms, d_polynomial_sizes = derivative

        d_x = torch.hstack((x,grad_output)).contiguous()

        derivative_calc_output = EvaluatePolynomials.apply( d_x, d_coefs, d_terms, d_polynomial_sizes, poly_idx, derivative_level+1 )

        return derivative_calc_output, None, None, None, None, None