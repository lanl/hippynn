import torch

def torchMessagePassingHop(T,s,z,pair_first,pair_second, envsum):
    ij,t = T.shape
    _,nu = s.shape
    sensitivity = s.unsqueeze(1) * T.unsqueeze(2)
    sense_flat = sensitivity.reshape(ij, t * nu)
    return envsum(sense_flat, z, pair_first, pair_second)

def torchMessagePassingVec(in_features, sense_vals, pair_first, pair_second, dist_pairs, coord_pairs, envsum):
    _, nu = sense_vals.shape
    sense_vec = sense_vals.unsqueeze(1) * (coord_pairs / dist_pairs.unsqueeze(1)).unsqueeze(2)
    sense_vec = sense_vec.reshape(-1, nu * 3)
    sense_stacked = torch.concatenate([sense_vals, sense_vec], dim=1)
    return envsum(sense_stacked, in_features, pair_first, pair_second)

def torchMessagePassingQuad(in_features, sense_vals, pair_first, pair_second, dist_pairs, coord_pairs, upper_ind, envsum):
    _, nu = sense_vals.shape

    rhats = coord_pairs / dist_pairs.unsqueeze(1)
    sense_vec = sense_vals.unsqueeze(1) * rhats.unsqueeze(2)
    sense_vec = sense_vec.reshape(-1, nu * 3)
    rhatsquad = rhats.unsqueeze(1) * rhats.unsqueeze(2)
    rhatsquad = (rhatsquad + rhatsquad.transpose(1, 2)) / 2
    tr = torch.diagonal(rhatsquad, dim1=1, dim2=2).sum(dim=1) / 3.0  # Add divide by 3 early to save flops
    tr = tr.unsqueeze(1).unsqueeze(2) * torch.eye(3, dtype=tr.dtype, device=tr.device).unsqueeze(0)
    rhatsquad = rhatsquad - tr
    rhatsqflat = rhatsquad.reshape(-1, 9)[:, upper_ind]  # Upper-diagonal part
    sense_quad = sense_vals.unsqueeze(1) * rhatsqflat.unsqueeze(2)
    sense_quad = sense_quad.reshape(-1, nu * 5)
    sense_stacked = torch.concatenate([sense_vals, sense_vec, sense_quad], dim=1)

    # Message passing, stack sensitivities to coalesce custom kernel call.
    # shape (n_atoms, n_nu + 3*n_nu + 5*n_nu, n_feat)
    return envsum(sense_stacked, in_features, pair_first, pair_second)