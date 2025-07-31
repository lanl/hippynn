import triton
import triton.language as tl
import torch

from .utils import resort_pairs_cached
from . import envsum, sensesum, featsum

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
def tensor_products(
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
    atom_id = tl.program_id(0)
    Ts_id = tl.program_id(1)
    z_id = tl.program_id(2)

    num_T_blocks : tl.constexpr = tl.where( T_size % T_BLOCK_SIZE == 0, T_size // T_BLOCK_SIZE, T_size // T_BLOCK_SIZE + 1 )
    num_s_blocks : tl.constexpr = tl.where( s_size % S_BLOCK_SIZE == 0, s_size // S_BLOCK_SIZE, s_size // S_BLOCK_SIZE + 1 )

    T_id = Ts_id % num_T_blocks
    s_id = (Ts_id // num_T_blocks) % num_s_blocks

    valid_atom_id = atom_id < E_size

    start = tl.load(atom1_starts_ptr + atom_id, mask=valid_atom_id, other=0)
    end = tl.load(atom1_starts_ptr + atom_id + 1, mask=valid_atom_id, other=0)
    target_id = tl.load(atom1_ids_ptr + atom_id, mask=valid_atom_id, other=0)

    T_arange = T_id * T_BLOCK_SIZE + tl.arange(0,T_BLOCK_SIZE)
    s_arange = s_id * S_BLOCK_SIZE + tl.arange(0,S_BLOCK_SIZE)
    z_arange = z_id * Z_BLOCK_SIZE + tl.arange(0,Z_BLOCK_SIZE)

    T_mask = T_arange < T_size
    s_mask = s_arange < s_size
    z_mask = z_arange < z_size

    E_offsets = (target_id * T_size * s_size * z_size) + (T_arange[:,None,None] * s_size * z_size) + (s_arange[None,:,None] * z_size) + z_arange[None,None,:]
    E_mask = T_mask[:,None,None] & s_mask[None,:,None] & z_mask[None,None,:]

    if (compute_Esz or compute_ETz) or compute_ETs:
        E = tl.load(E_ptr + E_offsets, mask=E_mask)
    else:
        E = tl.zeros((T_BLOCK_SIZE, S_BLOCK_SIZE, Z_BLOCK_SIZE), dtype=dtype)

    Tsz_accumulator = tl.zeros((T_BLOCK_SIZE, S_BLOCK_SIZE, Z_BLOCK_SIZE), dtype=dtype)

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

def envsum_fused(T,s,z,pair_first,pair_second):
    argsort, atom1_ids, atom1_starts, pair_first, (T,s,pair_second) = resort_pairs_cached(pair_first, [T,s,pair_second])
    env = TensorProductWrapper.apply(None,T,s,z,True,False,False,False,pair_first,pair_second, atom1_ids, atom1_starts)[0]
    i,t,nu,b = env.shape
    return env.reshape((i,t*nu,b))

class TensorProductWrapper(torch.autograd.Function):

    @staticmethod
    def forward(ctx, E, T, s, z, compute_Tsz, compute_Esz, compute_ETz, compute_ETs, pair_first, pair_second, atom1_ids, atom1_starts):

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

        bool_wrapper = torch.BoolTensor([compute_Tsz, compute_Esz, compute_ETz, compute_ETs])
        ctx.save_for_backward(E, T, s, z, bool_wrapper, pair_first, pair_second, atom1_ids, atom1_starts)

        grid = lambda META : (i, triton.cdiv(t, META["T_BLOCK_SIZE"]) * triton.cdiv(nu, META["S_BLOCK_SIZE"]), triton.cdiv(b, META["Z_BLOCK_SIZE"]))

        tensor_products[grid](
            Tsz, Esz, ETz, ETs_ij,
            E,T,s,z,
            atom1_ids, atom1_starts, pair_second,
            i, t, nu, b,
            compute_Tsz, compute_Esz, compute_ETz, compute_ETs,
            dtype=tl_dtype,
        )

        if compute_ETs:
            ETs = torch.zeros((i,b), device=device,dtype=dtype)
            ETs.index_add_(0, pair_second, ETs_ij)
        else:
            ETs = None

        return Tsz, Esz, ETz, ETs

    @staticmethod
    def backward(ctx, grad_output_Tsz, grad_output_Esz, grad_output_ETz, grad_output_ETs):
        E, T, s, z, bool_wrapper, pair_first, pair_second, atom1_ids, atom1_starts = ctx.saved_tensors

        compute_Tsz, compute_Esz, compute_ETz, compute_ETs = bool_wrapper

        E_grad = T_grad = s_grad = z_grad = None

        if compute_Tsz:
            Tsz_grad = TensorProductWrapper.apply( grad_output_Tsz, T, s, z, False, True, True, True, pair_first, pair_second, atom1_ids, atom1_starts )

            T_grad = Tsz_grad[1]
            s_grad = Tsz_grad[2]
            z_grad = Tsz_grad[3]

        if compute_Esz:
            Esz_grad = TensorProductWrapper.apply( E, grad_output_Esz, s, z, True, False, True, True, pair_first, pair_second, atom1_ids, atom1_starts )


            E_grad = Esz_grad[0]
            if compute_Tsz:
                s_grad = s_grad + Esz_grad[2]
                z_grad = z_grad + Esz_grad[3]
            else:
                s_grad = Esz_grad[2]
                z_grad = Esz_grad[3]
        
        if compute_ETz:
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

        if compute_ETs:
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


################# USED FOR ABLATION TESTING ONLY ##############################

def envsum_fused_default_gradient(T,s,z,pair_first,pair_second):
    argsort, atom1_ids, atom1_starts, pair_first, (T,s,pair_second) = resort_pairs_cached(pair_first, [T,s,pair_second])
    return EnvsumFusedDefaultGradient.apply(T,s,z,pair_first,pair_second, atom1_ids, atom1_starts)

class EnvsumFusedDefaultGradient(torch.autograd.Function):

    @staticmethod
    def forward(ctx, T, s, z, pair_first, pair_second, atom1_ids, atom1_starts):

        ij, t = T.shape
        _, nu = s.shape
        i, b = z.shape

        device = T.device
        dtype = T.dtype

        if dtype == torch.float32:
            tl_dtype = tl.float32
        else:
            tl_dtype = tl.float64

        Tsz = torch.zeros((i,t*nu,b), device=device, dtype=dtype, requires_grad=True)

        ctx.save_for_backward(T, s, z, pair_first, pair_second)

        grid = lambda META : (i, triton.cdiv(t, META["T_BLOCK_SIZE"]) * triton.cdiv(nu, META["S_BLOCK_SIZE"]), triton.cdiv(b, META["Z_BLOCK_SIZE"]))

        tensor_products[grid](
            Tsz, Tsz, Tsz, Tsz, # pass in the same pointer four times because we can't pass in None for the other pointers. All references to the other pointers will be compiled out.
            None,T,s,z,
            atom1_ids, atom1_starts, pair_second,
            i, t, nu, b,
            True, False, False, False,
            dtype=tl_dtype,
        )

        return Tsz

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()

        T, s, z, pair_first, pair_second = ctx.saved_tensors

        sense = s.unsqueeze(1) * T.unsqueeze(2)
        sense = sense.flatten(1)

        sense_grad = sensesum(grad_output, z, pair_first, pair_second)
        z_grad = featsum(grad_output, sense, pair_first, pair_second)

        sense_grad = sense_grad.reshape((s.shape[0],s.shape[1], T.shape[1]))
        T_grad = torch.sum(sense_grad * s[:,:,None], dim=1)
        s_grad = torch.sum(sense_grad * T[:,None,:], dim=2)

        return T_grad, s_grad, z_grad, None, None, None, None




def envsum_fused_gradient(T,s,z,pair_first,pair_second):
    argsort, atom1_ids, atom1_starts, pair_first, (T,s,pair_second) = resort_pairs_cached(pair_first, [T,s,pair_second])
    return EnvsumFusedGradient.apply(T,s,z,pair_first,pair_second,atom1_ids,atom1_starts)

class EnvsumFusedGradient(torch.autograd.Function):

    @staticmethod
    def forward(ctx, T, s, z, pair_first, pair_second, atom1_ids, atom1_starts):

        ctx.save_for_backward(T, s, z, pair_first, pair_second, atom1_ids, atom1_starts)

        sense = (s[:,:,None] * T[:,None,:]).flatten(1)
        return envsum(sense, z, pair_first, pair_second)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        T,s,z,pair_first,pair_second,atom1_ids,atom1_starts = ctx.saved_tensors

        grad_output = grad_output.reshape((grad_output.shape[0], T.shape[1], s.shape[1], z.shape[1]))

        _, partialT, partials, partialz = TensorProductWrapper.apply( grad_output, T, s, z, False, True, True, True, pair_first, pair_second, atom1_ids, atom1_starts )

        return partialT, partials, partialz, None, None, None, None