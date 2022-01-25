import torch


def get_id_and_starts(key):
    key_ids = torch.unique_consecutive(key)
    n_keys = key_ids.shape[0]
    n_items = key.shape[0]
    key_diff = key[:-1] - key[1:]
    key_start = torch.empty(n_keys + 1, dtype=torch.long, device=key.device)
    key_start[1:-1] = torch.nonzero(key_diff, as_tuple=True)[0] + 1
    key_start[0] = 0
    key_start[-1] = n_items
    return key_ids, key_start


def resort_pairs(key, others):
    keysort, argsort = torch.sort(key)
    others = [o[argsort] for o in others]
    key_ids, key_starts = get_id_and_starts(keysort)
    return key_ids, key_starts, keysort, others

