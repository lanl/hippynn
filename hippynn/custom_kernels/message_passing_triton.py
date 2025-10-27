import triton
import triton.language as tl
import torch

from .utils import resort_pairs_cached
from . import envsum

"""
This module concerns itself with implementing the message passing step for HIP-HOP-NN and HIP-NN-TS
using trition.
"""

def config_pruner(configs, nargs, **kwargs):
    """
    Trims the unnecessary config options based on T, s, and z sizes.
    """
    p2_t_block_size = triton.next_power_of_2(nargs["T_size"])
    p2_s_block_size = triton.next_power_of_2(nargs["s_size"])
    p2_z_block_size = triton.next_power_of_2(nargs["z_size"])

    used = set()
    for config in configs:

        t_block_size = min(p2_t_block_size, config.kwargs["T_BLOCK_SIZE"])
        s_block_size = min(p2_s_block_size, config.kwargs["S_BLOCK_SIZE"])
        z_block_size = min(p2_z_block_size, config.kwargs["Z_BLOCK_SIZE"])

        if (t_block_size, s_block_size, z_block_size, config.num_stages, config.num_warps) in used:
            continue

        used.add((t_block_size, s_block_size, z_block_size, config.num_stages, config.num_warps))

        yield triton.Config(
            {
                "T_BLOCK_SIZE": config.kwargs["T_BLOCK_SIZE"],
                "S_BLOCK_SIZE": config.kwargs["S_BLOCK_SIZE"],
                "Z_BLOCK_SIZE": config.kwargs["Z_BLOCK_SIZE"],
            },
            num_stages=config.num_stages,
            num_warps=config.num_warps,
        )

def get_autotune_config():
    """
    Gets possible configuation options for the tensor_products kernel.
    """

    output = []
    for z_block_size in [64,128,256]:
        for s_block_size in [4,8]:
            for t_block_size in [16]:
                for num_warps in [2,4,8]:
                    for num_stages in [2,3,4,5,6]:
                        output.append(triton.Config({"T_BLOCK_SIZE" : t_block_size, "S_BLOCK_SIZE" : s_block_size, "Z_BLOCK_SIZE" : z_block_size}, num_warps=num_warps, num_stages=num_stages))

    return output

@triton.autotune(configs=get_autotune_config(), key=["compute_Tsz", "compute_Esz", "compute_ETz", "compute_ETs"], prune_configs_by={"early_config_prune" : config_pruner}, reset_to_zero=["Tsz_out", "Esz_out", "ETz_out", "ETs_ij_out"])
@triton.jit
def tensor_products_kernel(
    Tsz_out,
    Esz_out,
    ETz_out,
    ETs_ij_out,
    E_ptr,
    T_ptr,
    s_ptr,
    z_ptr,
    atom1_ids_ptr,
    atom1_starts_ptr,
    pair_second_ptr,
    E_size,
    T_size : tl.constexpr,
    s_size : tl.constexpr,
    z_size : tl.constexpr,
    compute_Tsz : tl.constexpr,
    compute_Esz : tl.constexpr,
    compute_ETz : tl.constexpr,
    compute_ETs : tl.constexpr,
    T_BLOCK_SIZE : tl.constexpr,
    S_BLOCK_SIZE : tl.constexpr,
    Z_BLOCK_SIZE : tl.constexpr,
    dtype : tl.constexpr = tl.float32,
):
    """
    Let T, s, and z be as they are in the paper. Index atoms by i, neighbors by j,
    components of T by t, components of s by nu, and components of z by b. 
    Let E have the same shape as the product Tsz (summed over j).
    
    This kernel simultaneously computes up to four tensor products: Tsz (summed over j); Esz (summed over nu and b); TEz (summed over t and b), and TEs (summed over t and nu).
    The user can specify which of the four products they would like to compute.

    When computing a product that involves E, notice that the first axis of E indexes different atoms. Then when taking the outer product of E with e.g. T, the slice of E
    corresponding to atom i will be multiplied by each slice of T corresponding to a pair ij (where atom j is a neighbor of atom i).

    The user should never pass in None for any of the parameters that are pointers, even if that parameter will never be used. For example, if Tsz will not be computed,
    the user should still pass in a valid PyTorch tensor for Tsz_out.

    Certain hyperparameters allow the user to compute these products in blocks. In such cases, when one must sum over an axis (e.g. sum over b), that is split into multiple blocks,
    an atomic add operation must be used.

    :param Tsz_out: Pointer to the tensor where Tsz will be stored.

    :param Esz_out: Pointer to the tensor where Esz will be stored.

    :param ETz_out: Pointer to the tensor where ETz will be stored.

    :parm ETs_ij_out: Pointer to the tensor where ETs will be stored.

    :param E_ptr: Pointer to E.

    :param T_ptr: Pointer to T.

    :Param s_ptr: Pointer to s.

    :param z_ptr: Pointer to z.

    :param atom1_ids_ptr: Pointer to a tensor that stores the indices of every atom that has neighbors.

    :param atom1_starts_ptr: Pointer to a tensor. The list of pairs (specified in later parameters) is sorted such that, for each atom i,
                             a contiguous slice of the list of pairs contains every pair where the first atom in the pair is atom i. Index k of
                             atom_starts_ptr contains the location in the list of pairs corresponding to the beginning of that contiguous slice for 
                             atom atom1_ids_ptr[k].

    :param pair_second_ptr: Pointer to a tensor. This stores the second entry in the list of pairs for every pair. The first entry is not needed for computation,
                            so we only pass in the second entry.
    
    :param E_size: Scalar corresponding to E.shape[0].

    :param T_size: Scalar corresponding to T.shape[1].

    :param s_size: Scalar corresponding to s.shape[1].

    :param z_size: Scalar corresponding to z.shape[1].

    :param compute_Tsz: bool specifying whether Tsz should be computed.

    :param compute_Esz: bool specifying whether Esz should be computed.

    :param compute ETz: bool specifying whether ETz should be computed.

    :param compute ETs: bool specifying whether ETs should be computed.

    :param T_BLOCK_SIZE: hyperparameter stating the size of blocks that should be used to split up T.shape[1].

    :param S_BLOCK_SIZE: hyperparameter stating the size of blocks that should be used to split up s.shape[1].

    :param Z_BLOCK_SIZE: hyperparameter stating the size of blocks that should be used to split up z.shape[1].

    :param dtype: The data type used to store the various products (should be tl.float32 or tl.float64)

    """

    # atom_id indexes which atom we are loading. It only indexes atoms that have neighbors (whose ids are stored in atom1_ids_ptr)
    atom_id = tl.program_id(0)
    valid_atom_id = atom_id < E_size

    # these index the blocks for T, s, and z. Because the triton grid can only support three axes of program ids, we combine
    # the indices for T and s into a single axis, and subsequently compute the corresponding ids.
    Ts_id = tl.program_id(1)
    z_id = tl.program_id(2)

    num_T_blocks : tl.constexpr = tl.cdiv( T_size, T_BLOCK_SIZE )
    num_s_blocks : tl.constexpr = tl.cdiv( s_size, S_BLOCK_SIZE )

    T_id = Ts_id % num_T_blocks
    s_id = (Ts_id // num_T_blocks) % num_s_blocks

    # compute the atom id and range of pairs corresponding to the current atom.
    start = tl.load(atom1_starts_ptr + atom_id, mask=valid_atom_id, other=0)
    end = tl.load(atom1_starts_ptr + atom_id + 1, mask=valid_atom_id, other=0)
    target_id = tl.load(atom1_ids_ptr + atom_id, mask=valid_atom_id, other=0)

    # compute aranges and masks that will be used for loading.
    T_arange = T_id * T_BLOCK_SIZE + tl.arange(0,T_BLOCK_SIZE)
    s_arange = s_id * S_BLOCK_SIZE + tl.arange(0,S_BLOCK_SIZE)
    z_arange = z_id * Z_BLOCK_SIZE + tl.arange(0,Z_BLOCK_SIZE)

    T_mask = T_arange < T_size
    s_mask = s_arange < s_size
    z_mask = z_arange < z_size

    # Load in the chunk of E that corresponds to the current atom.
    E_offsets = (target_id * T_size * s_size * z_size) + (T_arange[:,None,None] * s_size * z_size) + (s_arange[None,:,None] * z_size) + z_arange[None,None,:]
    E_mask = T_mask[:,None,None] & s_mask[None,:,None] & z_mask[None,None,:]

    if (compute_Esz or compute_ETz) or compute_ETs:
        E = tl.load(E_ptr + E_offsets, mask=E_mask)
    else:
        E = tl.zeros((T_BLOCK_SIZE, S_BLOCK_SIZE, Z_BLOCK_SIZE), dtype=dtype)

    Tsz_accumulator = tl.zeros((T_BLOCK_SIZE, S_BLOCK_SIZE, Z_BLOCK_SIZE), dtype=dtype)

    # Iterative over the neighbors of the current atom. Compute Esz, ETz, and ETs for each neighbor, and also
    # accumulate Tsz over all of the neighbors of the current atom.
    # The logic below performs loads, products, sums, and stores depending on which products are to be computed.
    # Any redundant computations will be removed by the compiler.
    for neighbor in range(start,end):

        atom2 = tl.load(pair_second_ptr + neighbor)

        if (compute_Tsz or compute_Esz) or compute_ETz:
            z = tl.load( z_ptr + atom2 * z_size + z_arange, mask=z_mask )
        else:
            z = tl.zeros( (Z_BLOCK_SIZE,), dtype=dtype )

        if compute_ETz or compute_Esz:
            Ez = tl.sum(E * z[None,None,:], axis=2)
        else:
            Ez = tl.zeros( (T_BLOCK_SIZE, S_BLOCK_SIZE), dtype=dtype )

        if (compute_Tsz or compute_ETz) or compute_ETs:
            T = tl.load( T_ptr + neighbor * T_size + T_arange, mask=T_mask )
        else:
            T = tl.zeros( (T_BLOCK_SIZE,), dtype=dtype )

        if compute_ETz:
            ETz = tl.sum(Ez * T[:,None], axis=0)
            if Z_BLOCK_SIZE >= z_size and T_BLOCK_SIZE >= T_size:
                tl.store( ETz_out + neighbor * s_size + s_arange, ETz, mask=s_mask )
            else:
                tl.atomic_add( ETz_out + neighbor * s_size + s_arange, ETz, mask=s_mask )
        
        if (compute_Tsz or compute_Esz) or compute_ETs:
            s = tl.load( s_ptr + neighbor * s_size + s_arange, mask=s_mask )
        else:
            s = tl.zeros( (S_BLOCK_SIZE,), dtype=dtype )

        if compute_Esz:
            Esz = tl.sum(Ez * s[None,:], axis=1)

            if Z_BLOCK_SIZE >= z_size and S_BLOCK_SIZE >= s_size:
                tl.store( Esz_out + neighbor * T_size + T_arange, Esz, mask=T_mask )
            else:
                tl.atomic_add( Esz_out + neighbor * T_size + T_arange, Esz, mask=T_mask )

        if compute_Tsz or compute_ETs:
            Ts = T[:,None,None] * s[None,:,None]
        else:
            Ts = tl.zeros( (T_BLOCK_SIZE,S_BLOCK_SIZE), dtype=dtype )

        if compute_Tsz:
            Tsz = Ts * z[None,None,:]
            Tsz_accumulator += Tsz

        if compute_ETs:
            ETs = E * Ts
            ETs = ETs.reshape((T_BLOCK_SIZE * S_BLOCK_SIZE, Z_BLOCK_SIZE))
            ETs = tl.sum(ETs, axis=0)

            if S_BLOCK_SIZE >= s_size and T_BLOCK_SIZE >= T_size:
                tl.store(ETs_ij_out + neighbor * z_size + z_arange, ETs, mask=z_mask)
            else:
                tl.atomic_add(ETs_ij_out + neighbor * z_size + z_arange, ETs, mask=z_mask)

    if compute_Tsz:
        tl.store( Tsz_out + E_offsets, Tsz_accumulator, mask=E_mask )

def tensorMessagePassingHop(T,s,z,pair_first,pair_second):
    """
    Performs the message passing step using the fused kernel. This performs essentially the same operation as
    envsum, except that the outer product of T and s is not computed beforehand.

    This function essentially serves as a wrapper for TensorProductWrapper. It calls TensorProductWrapper with certain parameters
    to compute the message passing layer (and nothing more).

    :param T: Components of the irreducible moment tensors corresponding to each pair of neighbors.

    :param s: Sensitivities corresponding to each pair of neighbors

    :param z: Features corresponding to each pair of neighbors.

    :pair_first: In the list of pairs of neighboring atoms, stores the first entry of each pair.

    :pair_second: In the list of pairs of neighboring atoms, stores the second entry of each pair.

    :return: A 3D tensor storing the result of the message passing layer. The first axis indexes the different atoms. The second
             axis indexes the tensor component and sensitivity. The third axis indexes the features. The result is the same as
             what is returned by envsum.
    """
    argsort, atom1_ids, atom1_starts, pair_first, (T,s,pair_second) = resort_pairs_cached(pair_first, [T,s,pair_second])
    env = TensorProductWrapper.apply(None,T,s,z,True,False,False,False,pair_first,pair_second, atom1_ids, atom1_starts)[0]
    i,t,nu,b = env.shape
    return env.reshape((i,t*nu,b))

class TensorProductWrapper(torch.autograd.Function):
    """
    Let T, s, and z be as they are in the paper. Index atoms by i, neighbors by j,
    components of T by t, components of s by nu, and components of z by b. 
    Let E have the same shape as the product Tsz (summed over j).
    
    This function simultaneously computes up to four tensor products: Tsz (summed over j); Esz (summed over nu and b); TEz (summed over t and b), and TEs (summed over i, t and nu).
    The user can specify which of the four products they would like to compute.

    When computing a product that involves E, notice that the first axis of E indexes different atoms. Then when taking the outer product of E with e.g. T, the slice of E
    corresponding to atom i will be multiplied by each slice of T corresponding to a pair ij (where atom j is a neighbor of atom i).

    :param E:
    
    :param T:

    :param s:

    :param z:

    :param compute_Tsz: Bool storing whether the product Tsz should be computed.

    :param compute_Esz: Bool storing whether the product Esz should be computed.

    :param compute_ETz: Bool storing whether the product ETz should be computed.

    :param compute_ETs: Bool storing whether the product ETs should be computed.

    :param pair_first: In the list of pairs of neighboring atoms, stores the first entry of each pair.

    :param pair_second: In the list of pairs of neighboring atoms, stores the second entry of each pair.

    :param atom1_ids: Stores the list ids of all atoms that have neighbors.

    :param atom1_starts: The list of pairs (specified in later parameters) is sorted such that, for each atom i,
                             a contiguous slice of the list of pairs contains every pair where the first atom in the pair is atom i. Index k of
                             atom_starts contains the location in the list of pairs corresponding to the beginning of that contiguous slice for 
                             atom atom1_ids[k].

    :return: A tuple containing the products Tsz, Esz, TEz, and TEz. For each product that the user does not select to compute, a tensor containing a single zero will be returned.
    """

    @staticmethod
    def forward(ctx, E, T, s, z, compute_Tsz, compute_Esz, compute_ETz, compute_ETs, pair_first, pair_second, atom1_ids, atom1_starts):

        # compute relevant axis dimensions based on which tensors are avaiable.
        if compute_Tsz:
            ij, t = T.shape
            _, nu = s.shape
            i, b = z.shape

            device = T.device
            dtype = T.dtype
        else:
            i, t, nu, b = E.shape
            device = E.device
            dtype = E.dtype

            if s is not None:
                ij = s.shape[0]
            else:
                ij = T.shape[0]
        (n_atom_with_pairs,) = atom1_ids.shape

        # Create tensors that will be used to call the kernel tensor_products_kernel.
        # Note that the kernel does not directly compute TEs (summed over i, t, and nu). 
        # Instead, it computes TEs (summed over t and nu). Then, we compute the sum over i later.
        if dtype == torch.float32:
            tl_dtype = tl.float32
        else:
            tl_dtype = tl.float64

        if compute_Tsz:
            Tsz = torch.zeros((i,t,nu,b), device=device, dtype=dtype, requires_grad=True)
        else:
            Tsz = torch.zeros(1, device=device, dtype=dtype, requires_grad=False)
        
        if compute_Esz:
            Esz = torch.zeros((ij,t), device=device, dtype=dtype, requires_grad=True)
        else:
            Esz = torch.zeros(1, device=device, dtype=dtype, requires_grad=False)
        
        if compute_ETz:
            ETz = torch.zeros((ij,nu), device=device, dtype=dtype, requires_grad=True)
        else:
            ETz = torch.zeros(1, device=device, dtype=dtype, requires_grad=False)
        
        if compute_ETs:
            ETs_ij = torch.zeros((ij,b), device=device, dtype=dtype)
        else:
            ETs_ij = torch.zeros(1, device=device, dtype=dtype, requires_grad=False)

        # wrap the bools in a tensor so that we are able to save it for backwards (only tensors can be saved).

        ctx.save_for_backward(E, T, s, z, pair_first, pair_second, atom1_ids, atom1_starts)
        ctx.compute_Tsz = compute_Tsz
        ctx.compute_Esz = compute_Esz
        ctx.compute_ETz = compute_ETz
        ctx.compute_ETs = compute_ETs

        # run the kernel. Recall that the T and S block indices are squeezed into one axis (the second one) because Triton only allows up to 3 axes.
        grid = lambda META : (n_atom_with_pairs, triton.cdiv(t, META["T_BLOCK_SIZE"]) * triton.cdiv(nu, META["S_BLOCK_SIZE"]), triton.cdiv(b, META["Z_BLOCK_SIZE"]))

        tensor_products_kernel[grid](
            Tsz, Esz, ETz, ETs_ij,
            E,T,s,z,
            atom1_ids, atom1_starts, pair_second,
            i, t, nu, b,
            compute_Tsz, compute_Esz, compute_ETz, compute_ETs,
            dtype=tl_dtype,
        )

        # Sum ETs over i using index_add.
        if compute_ETs:
            ETs = torch.zeros((i,b), device=device,dtype=dtype)
            ETs.index_add_(0, pair_second, ETs_ij)
        else:
            ETs = None

        return Tsz, Esz, ETz, ETs

    @staticmethod
    def backward(ctx, grad_output_Tsz, grad_output_Esz, grad_output_ETz, grad_output_ETs):
        E, T, s, z, pair_first, pair_second, atom1_ids, atom1_starts = ctx.saved_tensors

        E_grad = T_grad = s_grad = z_grad = None

        # For each of the four products, if it was computed, we compute its gradient by calling TensorProductWrapper.
        # Then, we sum up all partial derivatives that correspond to the same term (e.g. after computing dL/dz from Tsz, Esz, and ETz, we sum up
        # all of the computed values for dL/dz )

        if ctx.compute_Tsz:
            Tsz_grad = TensorProductWrapper.apply( grad_output_Tsz, T, s, z, False, True, True, True, pair_first, pair_second, atom1_ids, atom1_starts )

            T_grad = Tsz_grad[1]
            s_grad = Tsz_grad[2]
            z_grad = Tsz_grad[3]

        if ctx.compute_Esz:
            Esz_grad = TensorProductWrapper.apply( E, grad_output_Esz, s, z, True, False, True, True, pair_first, pair_second, atom1_ids, atom1_starts )


            E_grad = Esz_grad[0]
            if ctx.compute_Tsz:
                s_grad = s_grad + Esz_grad[2]
                z_grad = z_grad + Esz_grad[3]
            else:
                s_grad = Esz_grad[2]
                z_grad = Esz_grad[3]
        
        if ctx.compute_ETz:
            ETz_grad = TensorProductWrapper.apply( E, T, grad_output_ETz, z, True, True, False, True, pair_first, pair_second, atom1_ids, atom1_starts )

            if E_grad is None:
                E_grad = ETz_grad[0]
            else:
                E_grad = E_grad + ETz_grad[0]
            
            if T_grad is None:
                T_grad = ETz_grad[1]
            else:
                T_grad = T_grad + ETz_grad[1]
            
            if z_grad is None:
                z_grad = ETz_grad[3]
            else:
                z_grad = z_grad + ETz_grad[3]

        if ctx.compute_ETs:
            ETs_grad = TensorProductWrapper.apply( E, T, s, grad_output_ETs, True, True, True, False, pair_first, pair_second, atom1_ids, atom1_starts )

            if E_grad is None:
                E_grad = ETs_grad[0]
            else:
                E_grad = E_grad + ETs_grad[0]
            
            if T_grad is None:
                T_grad = ETs_grad[1]
            else:
                T_grad = T_grad + ETs_grad[1]

            if s_grad is None:
                s_grad = ETs_grad[2]
            else:
                s_grad = s_grad + ETs_grad[2]

        return E_grad, T_grad, s_grad, z_grad, None, None, None, None, None, None, None, None, None

def tensorMessagePassingVec(in_features, sense_vals, pair_first, pair_second, dist_pairs, coord_pairs):
    """
    Triton implementation of the message passing step for HIP-NN-TS if l_max is equal to 1.

    :param in_features: Matrix. Each row corresponds to an atom and stores the features associated with that atom.

    :param sense_vals: Matrix. Each row corresponds to a pair of neighbors. Each row stores the outputs of the sensitivity
                       function applied to the corresponding pair.

    :param pair_first: Vector of integers. For the list of pairs of neighbors in message passing, this is a list storing the first atom in each pair.

    :param pair_second: Vector of integers. For the list of pairs of neighbors in message passing, this is a list storing the second atom in each pair.

    :param dist_pairs: Vector. Each entry stores the distance between a pair of neighbors.

    :param coord_pairs: Matrix. Each row corresponds to a pair of neighbors. Each row stores the position offset (as a vector) from the first atom
                        to the second.

    :return: The output of message passing for HIP-NN-TS.
    """    
    device, dtype = in_features.device, in_features.dtype
    ones_ = torch.ones((sense_vals.shape[0],1), device=device, dtype=dtype)
    rhats = (coord_pairs / dist_pairs.unsqueeze(1))
    T = torch.hstack((ones_, rhats))

    _, atom1_ids, atom1_starts, pair_first, (T,sense_vals,pair_second) = resort_pairs_cached(pair_first, [T,sense_vals,pair_second])

    env = TensorProductWrapper.apply(None,T,sense_vals,in_features,True,False,False,False,pair_first,pair_second,atom1_ids,atom1_starts)[0]
    i,t,nu,b = env.shape
    return env.reshape((i,t*nu,b))

def tensorMessagePassingQuad(in_features, sense_vals, pair_first, pair_second, dist_pairs, coord_pairs):
    """
    Triton implementation of the message passing step for HIP-NN-TS if l_max is equal to 2.

    :param in_features: Matrix. Each row corresponds to an atom and stores the features associated with that atom.

    :param sense_vals: Matrix. Each row corresponds to a pair of neighbors. Each row stores the outputs of the sensitivity
                       function applied to the corresponding pair.

    :param pair_first: Vector of integers. For the list of pairs of neighbors in message passing, this is a list storing the first atom in each pair.

    :param pair_second: Vector of integers. For the list of pairs of neighbors in message passing, this is a list storing the second atom in each pair.

    :param dist_pairs: Vector. Each entry stores the distance between a pair of neighbors.

    :param coord_pairs: Matrix. Each row corresponds to a pair of neighbors. Each row stores the position offset (as a vector) from the first atom
                        to the second.

    :return: The output of message passing for HIP-NN-TS.
    """    
    upper_ind = torch.as_tensor([0, 1, 2, 4, 5], dtype=torch.int64)

    device, dtype = in_features.device, in_features.dtype
    ones_ = torch.ones((sense_vals.shape[0],1), device=device, dtype=dtype)
    rhats = (coord_pairs / dist_pairs.unsqueeze(1))

    rhatsquad = rhats.unsqueeze(1) * rhats.unsqueeze(2)
    rhatsquad = (rhatsquad + rhatsquad.transpose(1, 2)) / 2
    tr = torch.diagonal(rhatsquad, dim1=1, dim2=2).sum(dim=1) / 3.0  # Add divide by 3 early to save flops
    tr = tr.unsqueeze(1).unsqueeze(2) * torch.eye(3, dtype=tr.dtype, device=tr.device).unsqueeze(0)
    rhatsquad = rhatsquad - tr
    rhatsqflat = rhatsquad.reshape(-1, 9)[:, upper_ind]  # Upper-diagonal part

    T = torch.hstack((ones_, rhats, rhatsqflat))

    _, atom1_ids, atom1_starts, pair_first, (T,sense_vals,pair_second) = resort_pairs_cached(pair_first, [T,sense_vals,pair_second])

    env = TensorProductWrapper.apply(None,T,sense_vals,in_features,True,False,False,False,pair_first,pair_second,atom1_ids,atom1_starts)[0]
    i,t,nu,b = env.shape
    return env.reshape((i,t*nu,b))

################# USED FOR ABLATION TESTING ONLY ##############################

def _tensorMessagePassingBackwardOnly(T,s,z,pair_first,pair_second):
    """
    Wrapper for EnvsumFusedGradient. This is used only for ablation testing and should not be used in a production environment.
    """

    argsort, atom1_ids, atom1_starts, pair_first, (T,s,pair_second) = resort_pairs_cached(pair_first, [T,s,pair_second])
    return _EnvsumFusedGradient.apply(T,s,z,pair_first,pair_second,atom1_ids,atom1_starts)

class _EnvsumFusedGradient(torch.autograd.Function):
    """
    Computes the message passing layer using envsum for the forward pass, but TensorProductWrapper for gradients.
    This is used only for ablation testing and should not be used in a production environment.
    """
    @staticmethod
    def forward(ctx, T, s, z, pair_first, pair_second, atom1_ids, atom1_starts):

        ctx.save_for_backward(T, s, z, pair_first, pair_second, atom1_ids, atom1_starts)

        sense = (T[:,:,None] * s[:,None,:]).flatten(1)
        return envsum(sense, z, pair_first, pair_second)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        T,s,z,pair_first,pair_second,atom1_ids,atom1_starts = ctx.saved_tensors

        grad_output = grad_output.reshape((grad_output.shape[0], T.shape[1], s.shape[1], z.shape[1]))

        _, partialT, partials, partialz = TensorProductWrapper.apply( grad_output, T, s, z, False, True, True, True, pair_first, pair_second, atom1_ids, atom1_starts )

        return partialT, partials, partialz, None, None, None, None