"""
Internal utilities for batch optimizers
"""
from typing import Union, List, Tuple

import torch


def debatch_numbers(
    numbers: torch.Tensor, padding_number: int = 0, return_mask: bool = False
) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], torch.Tensor]]:
    # debatch a tensor of numbers to a list of numbers tensors in different length
    # enable returning the mask to reuse, useful when debatching a series of coordinates
    # with the same number tensor, like trajectory generated in an optimization task
    n = []
    masks = numbers != padding_number
    for z, m in zip(numbers, masks):
        n.append(z[m])

    return (n, masks) if return_mask else n


def debatch_coords(coords: torch.Tensor, masks: torch.Tensor) -> List[torch.Tensor]:
    # There is no way to distinguish which row is a padding in a given coordinates tensor
    # so this function must take masks as input, debatch to a list of tensors in different shape
    assert len(coords) == len(masks)
    c = []
    for r, m in zip(coords, masks):
        c.append(r[m])
    return c


def debatch(
    numbers: torch.Tensor, coords: torch.Tensor, numbers_padding: int = 0, return_mask: bool = False
) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], torch.Tensor]]:
    n, masks = debatch_numbers(numbers, numbers_padding, return_mask=True)
    c = debatch_coords(coords, masks)
    return (n, c, masks) if return_mask else (n, c)
