from typing import Literal
import torch
from torch import Tensor


def pad_to_length(x: Tensor, dim: int, target_length: int, pad_value: float = 0):
    padding_shape = list(x.shape)
    padding_shape[dim] = target_length - x.shape[dim]
    return torch.cat([x, torch.full(padding_shape, pad_value, dtype=x.dtype, device=x.device)], dim=dim)

def pad(x: Tensor, dim: int, direction: Literal["left", "right"], size: int, pad_value: float):
    pad_shape = list(x.shape)
    pad_shape[dim] = size
    if direction == "left":
        return torch.cat([torch.full(pad_shape, pad_value, dtype=x.dtype, device=x.device), x], dim=dim)
    else:
        return torch.cat([x, torch.full(pad_shape, pad_value, dtype=x.dtype, device=x.device)], dim=dim)


def pad_and_stack(
    batch: list[Tensor], pad_dim: int, pad_value: float = 0, stack_dim: int = 0, target_length: int | None = None
) -> Tensor:
    if target_length is None:
        target_length = max(x.shape[pad_dim] for x in batch)
    return torch.stack(
        [pad_to_length(x, dim=pad_dim, target_length=target_length, pad_value=pad_value) for x in batch], stack_dim
    )

def pad_and_cat(
    batch: list[Tensor], pad_dim: int, pad_value: float = 0, cat_dim: int = 0, target_length: int | None = None
) -> Tensor:
    if target_length is None:
        target_length = max(x.shape[pad_dim] for x in batch)
    return torch.cat(
        [pad_to_length(x, dim=pad_dim, target_length=target_length, pad_value=pad_value) for x in batch], cat_dim
    )


def cat_to_right(
    x: torch.Tensor, value: torch.Tensor | int | float | list[int | float | Tensor] | torch.nn.Parameter, dim: int = -1
):
    """
    x: (1, ..., 1, length, d1:n)
    value: (..., d1:n)
    dim: int
    """
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value, dtype=x.dtype, device=x.device)

    if dim < 0:
        dim = x.ndim + dim

    n = x.ndim - dim
    # assert x.shape[x.ndim-n+1:] == value.shape[value.ndim-n+1:], f'shape mismatch, x: {x.shape}, value: {value.shape}'
    n_unsqueeze = x.ndim - value.ndim
    for _ in range(n_unsqueeze):
        value = value.unsqueeze(0)

    expand_shape = list(x.shape)
    expand_shape[dim] = -1
    value = value.expand(expand_shape)

    return torch.cat([x, value], dim=dim)


def cat_to_left(
    x: torch.Tensor,
    value: torch.Tensor | int | float | list[int | float | Tensor] | torch.nn.Parameter,
    dim: int = -1,
):
    """
    x: (1, ..., 1, length, d1:n)
    value: (..., d1:n)
    dim: int
    """
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value, dtype=x.dtype, device=x.device)

    if dim < 0:
        dim = x.ndim + dim

    n = x.ndim - dim
    assert x.shape[x.ndim - n + 1 :] == value.shape[value.ndim - n + 1 :], (
        f"shape mismatch, x: {x.shape}, value: {value.shape}"
    )
    n_unsqueeze = x.ndim - value.ndim
    for _ in range(n_unsqueeze):
        value = value.unsqueeze(0)

    expand_shape = list(x.shape)
    expand_shape[dim] = -1
    value = value.expand(expand_shape)

    return torch.cat([value, x], dim=dim)
