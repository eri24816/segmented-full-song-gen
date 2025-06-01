from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from safetensors.torch import load_file

from segment_full_song.models.gen.lm import BaseLM


def unwrap_lightning_state_dict(state_dict: dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        assert key.startswith('model.'), f"Key {key} does not start with 'model.'"
        new_key = key[len('model.'):]
        new_state_dict[new_key] = value
    return new_state_dict

def load_ckpt_state_dict(ckpt_path: Path, unwrap_lightning: bool = False):
    if ckpt_path.suffix == ".safetensors":
        state_dict = load_file(ckpt_path)
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]

    if unwrap_lightning:
        state_dict = unwrap_lightning_state_dict(state_dict)
    return state_dict


def lm_token_to_id(token: str | Iterable[str]):
    if isinstance(token, str):
        out = BaseLM.spec_token_to_id(token)
    else:
        out = [BaseLM.spec_token_to_id(t) for t in token]
    return torch.LongTensor(out)


def nucleus_sample(logits: torch.Tensor, p: float):
    """
    logits: (vocab_size)
    p: float close to 1
    return: scalar
    """
    probs = torch.softmax(logits, dim=0)
    sorted_probs, sorted_indices = torch.sort(probs, dim=0, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=0)
    selected_indices = []
    selected_probs = []
    for i in range(len(sorted_probs)):
        selected_indices.append(sorted_indices[i])
        selected_probs.append(sorted_probs[i])
        if cumulative_probs[i] > p:
            break
    # sample from selected_indices
    # normalize selected_probs
    selected_probs = torch.tensor(selected_probs)
    selected_probs = selected_probs / torch.sum(selected_probs)
    selected = torch.multinomial(selected_probs, 1)
    return selected_indices[selected]


def get_patch_size(in_size, stride_vertical, stride_horizontal):
    frame_height, frame_width = in_size

    patch_height = np.prod(a=stride_vertical).item()
    patch_width = np.prod(a=stride_horizontal).item()

    assert frame_height % patch_height == 0
    assert frame_width % patch_width == 0

    return patch_height, patch_width


def get_patch_info(in_size, stride_vertical, stride_horizontal):
    patch_size = get_patch_size(
        in_size,
        stride_vertical,
        stride_horizontal,
    )
    return (
        patch_size[0],
        patch_size[1],
        in_size[0] // patch_size[0],
        in_size[1] // patch_size[1],
    )
