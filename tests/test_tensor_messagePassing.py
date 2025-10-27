import pytest
import torch

from hippynn.custom_kernels import envsum, hopMessagePassing, vecMessagePassing, quadMessagePassing

def default_message_passing_arguments(graph):

    if graph == "complete":
        n_atoms = 3
        n_pairs = 3
    elif graph == "sparse":
        n_atoms = 3
        n_pairs = 1
    
    t = 2 # unused for hip_nn_ts.
    nu = 2
    b = 2
    
    return n_atoms, n_pairs, t, nu, b
    

def generate_hop_message_passing_tensors(n_atoms,n_pairs,t,nu,b,device):
    torch.manual_seed(0)
    pair_list = []

    counter = 0
    while len(pair_list) < n_pairs:
        counter += 1
        p1 = torch.randint(0,n_atoms,(1,)).item()
        p2 = torch.randint(0,n_atoms,(1,)).item()
        if p1 < p2:
            pair = (p1,p2)
        elif p2 < p1:
            pair = (p2, p1)
        else:
            continue
        if pair not in pair_list:
            pair_list.append(pair)

    l = len(pair_list)
    for i in range(l):
        pair_list.append((pair_list[i][1],pair_list[i][0]))

    pair_first = torch.IntTensor([p[0] for p in pair_list]).to(device)
    pair_second = torch.IntTensor([p[1] for p in pair_list]).to(device)

    T = torch.randn((n_pairs*2,t),device=device, dtype=torch.float64, requires_grad=True)
    s = torch.randn((n_pairs*2,nu),device=device, dtype=torch.float64, requires_grad=True)
    z = torch.randn((n_atoms,b),device=device, dtype=torch.float64, requires_grad=True)

    return T,s,z,pair_first,pair_second

def generate_TS_message_passing_tensors(n_atoms,n_pairs,nu,b,device):
    torch.manual_seed(0)
    pair_list = []

    counter = 0
    while len(pair_list) < n_pairs:
        counter += 1
        p1 = torch.randint(0,n_atoms,(1,)).item()
        p2 = torch.randint(0,n_atoms,(1,)).item()
        if p1 < p2:
            pair = (p1,p2)
        elif p2 < p1:
            pair = (p2, p1)
        else:
            continue
        if pair not in pair_list:
            pair_list.append(pair)

    l = len(pair_list)
    for i in range(l):
        pair_list.append((pair_list[i][1],pair_list[i][0]))

    pair_first = torch.IntTensor([p[0] for p in pair_list]).to(device)
    pair_second = torch.IntTensor([p[1] for p in pair_list]).to(device)

    in_features = torch.randn((n_atoms,b),device=device, dtype=torch.float64, requires_grad=True)
    dist_pairs = torch.randn((n_pairs*2,),device=device, dtype=torch.float64, requires_grad=True)
    coord_pairs = torch.randn((n_pairs*2,3),device=device, dtype=torch.float64, requires_grad=True)
    sense_vals = torch.randn((n_pairs*2,nu),device=device, dtype=torch.float64, requires_grad=True)

    return in_features, sense_vals, pair_first, pair_second, dist_pairs, coord_pairs

def test_tensor_message_passing_kernel():

    try:
        import triton
        triton_available = True
    except:
        triton_available = False

    if triton_available and torch.cuda.is_available():
        for graph in ["complete", "sparse"]:
            n_atoms, n_pairs, t, nu, b = default_message_passing_arguments(graph)
            T,s,z,pair_first,pair_second = generate_hop_message_passing_tensors( n_atoms, n_pairs, t, nu, b, 'cuda' )

            sensitivity = s.unsqueeze(1) * T.unsqueeze(2)
            sense_flat = sensitivity.reshape(2*n_pairs, t * nu)
            envsum_message_passing_result = envsum(sense_flat, z, pair_first, pair_second)

            from hippynn.custom_kernels.message_passing_triton import tensorMessagePassingHop
            tensor_message_passing_result = tensorMessagePassingHop(T,s,z,pair_first,pair_second)

            assert torch.allclose(tensor_message_passing_result, envsum_message_passing_result, rtol=1e-6, atol=1e-6)
            assert torch.autograd.gradcheck(tensorMessagePassingHop, (T,s,z,pair_first,pair_second))
            assert torch.autograd.gradgradcheck(tensorMessagePassingHop, (T,s,z,pair_first,pair_second))

def test_hop_message_passing_wrappers():

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    n_atoms, n_pairs, t, nu, b = default_message_passing_arguments("complete")

    for layer in ["hop", "vec", "quad"]:

        if layer == "hop":
            T,s,z,pair_first,pair_second = generate_hop_message_passing_tensors( n_atoms, n_pairs, t, nu, b, device )            
            sensitivity = s.unsqueeze(1) * T.unsqueeze(2)
            sense_flat = sensitivity.reshape(2*n_pairs, t * nu)
            envsum_message_passing_result = envsum(sense_flat, z, pair_first, pair_second)
            
            tensor_message_passing_result = hopMessagePassing(T, s, z, pair_first, pair_second)

            assert torch.autograd.gradcheck(hopMessagePassing, (T,s,z,pair_first,pair_second))
            assert torch.autograd.gradgradcheck(hopMessagePassing, (T,s,z,pair_first,pair_second))

        elif layer == "vec":
            in_features, sense_vals, pair_first, pair_second, dist_pairs, coord_pairs = generate_TS_message_passing_tensors( n_atoms, n_pairs, nu, b, device )

            sense_vec = sense_vals.unsqueeze(1) * (coord_pairs / dist_pairs.unsqueeze(1)).unsqueeze(2)
            sense_vec = sense_vec.reshape(-1, nu * 3)
            sense_stacked = torch.concatenate([sense_vals, sense_vec], dim=1)

            envsum_message_passing_result = envsum(sense_stacked, in_features, pair_first, pair_second)

            tensor_message_passing_result = vecMessagePassing(in_features, sense_vals, pair_first, pair_second, dist_pairs, coord_pairs)

            assert torch.autograd.gradcheck(vecMessagePassing, (in_features, sense_vals, pair_first, pair_second, dist_pairs, coord_pairs))
            assert torch.autograd.gradgradcheck(vecMessagePassing, (in_features, sense_vals, pair_first, pair_second, dist_pairs, coord_pairs))
        elif layer == "quad":
            in_features, sense_vals, pair_first, pair_second, dist_pairs, coord_pairs = generate_TS_message_passing_tensors( n_atoms, n_pairs, nu, b, device )

            upper_ind = torch.as_tensor([0, 1, 2, 4, 5], dtype=torch.int64)

            rhats = coord_pairs / dist_pairs.unsqueeze(1)
            sense_vec = sense_vals.unsqueeze(1) * rhats.unsqueeze(2)
            sense_vec = sense_vec.reshape(-1, nu * 3)
            rhatsquad = rhats.unsqueeze(1) * rhats.unsqueeze(2)
            rhatsquad = (rhatsquad + rhatsquad.transpose(1, 2)) / 2
            tr = torch.diagonal(rhatsquad, dim1=1, dim2=2).sum(dim=1) / 3.0
            tr = tr.unsqueeze(1).unsqueeze(2) * torch.eye(3, dtype=tr.dtype, device=tr.device).unsqueeze(0)
            rhatsquad = rhatsquad - tr
            rhatsqflat = rhatsquad.reshape(-1, 9)[:, upper_ind]  # Upper-diagonal part
            sense_quad = sense_vals.unsqueeze(1) * rhatsqflat.unsqueeze(2)
            sense_quad = sense_quad.reshape(-1, nu * 5)
            sense_stacked = torch.concatenate([sense_vals, sense_vec, sense_quad], dim=1)

            envsum_message_passing_result = envsum(sense_stacked, in_features, pair_first, pair_second)

            tensor_message_passing_result = quadMessagePassing(in_features, sense_vals, pair_first, pair_second, dist_pairs, coord_pairs, upper_ind)

            assert torch.autograd.gradcheck(quadMessagePassing, (in_features, sense_vals, pair_first, pair_second, dist_pairs, coord_pairs, upper_ind))
            assert torch.autograd.gradgradcheck(quadMessagePassing, (in_features, sense_vals, pair_first, pair_second, dist_pairs, coord_pairs, upper_ind))

        assert torch.allclose(tensor_message_passing_result, envsum_message_passing_result, rtol=1e-6, atol=1e-6)