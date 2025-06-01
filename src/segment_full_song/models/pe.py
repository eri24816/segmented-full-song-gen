import math

import torch
import torch.nn as nn
from segment_full_song.utils.torch_utils.shape_guard import shape_guard
from torch import Tensor

def sinusoidal_positional_encoding(length, dim, base=10000.0):
    """
    Returns (length, dim)
    """
    pe = torch.zeros(length, dim)
    n_effective_dim = dim - dim % 2
    position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, n_effective_dim, 2).float() * (-math.log(base) / n_effective_dim))
    pe[:, 0:n_effective_dim:2] = torch.sin(position * div_term)
    pe[:, 1:n_effective_dim:2] = torch.cos(position * div_term)
    return pe


def binary_positional_encoding(length: int, dim: int):
    """
    Returns (length, dim)
    """
    res = []
    for i in range(length):
        res.append([int(x) for x in f"{i:0{dim}b}"][-dim:])
        # pad
        res[-1] += [0] * (dim - len(res[-1]))

    return torch.tensor(res, dtype=torch.float32).flip(dims=[1])


def one_hot_positional_encoding(length: int, dim: int):
    """
    Returns (length, dim)
    """
    return torch.eye(dim, dim).repeat(max((length) // dim + 1, 1), 1)[:length]

class StartEndPosEmb(nn.Module):
    """
    Returns a positional encoding relative to the start and end of the segment.
    """

    def __init__(self, max_duration: int, dim: int):
        super().__init__()
        assert dim % 2 == 0, "dim must be even"
        self.pe: torch.Tensor
        self.register_buffer("pe", sinusoidal_positional_encoding(max_duration, dim // 2))

    @shape_guard(
        shift_from_start="b",
        duration="b",
        pos="b n",
        _output="b n d",
    )
    def forward(self, shift_from_start: Tensor, duration: Tensor, pos: Tensor) -> Tensor:
        pos_from_start = pos + shift_from_start.unsqueeze(1)
        pos_from_start_pe = self.pe[pos_from_start]
        pos_from_end = duration.unsqueeze(1) - 1 - pos_from_start
        pos_from_end_pe = self.pe[pos_from_end]
        return torch.cat([pos_from_start_pe, pos_from_end_pe], dim=2)  # (b, f, d)
