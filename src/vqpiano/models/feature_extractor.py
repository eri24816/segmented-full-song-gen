from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
import x_transformers.x_transformers

from .network import MLP


from .token_sequence import TokenSequence
from .pe import one_hot_positional_encoding, StartEndPosEmb
from vqpiano.utils.torch_utils.tensor_op import cat_to_right, pad_to_length

import x_transformers


class FeatureExtractor(nn.Module):
    """
    - dim (int):
        - The dimension of the network and the output feature
    - num_layers (int):
        - The number of transformer layers
    - pitch_range (list[int]):
        - The range of the pitch
    - max_len (int):
        - The maximum length of the input. Positional encodings are initialized with this length.
    - reduce (bool):
        - If True, the output is reduced to a single vector (batch_size, dim).
        - If False, the output is a sequence of vectors (batch_size, num_tokens, dim).
    """

    @dataclass
    class Params:
        """
        - dim (int):
            - The dimension of the network and the output feature
        - num_layers (int):
            - The number of transformer layers
        - pitch_range (list[int]):
            - The range of the pitch
        - max_len (int):
            - The maximum length of the input. Positional encodings are initialized with this length.
        - reduce (bool):
            - If True, the output is reduced to a single vector (batch_size, dim).
            - If False, the output is a sequence of vectors (batch_size, num_tokens, dim).
        """

        dim: int
        num_layers: int
        pitch_range: list[int]
        num_pos: int
        reduce: bool
        condition_dim: int = 0

    def __init__(
        self,
        dim: int,
        num_layers: int,
        pitch_range: list[int],
        num_pos: int,
        max_note_duration: int,
        reduce: bool = True,
        condition_dim: int = 0,
        is_causal: bool = True,
        cross_attend: bool = False,
        use_start_end_pos: bool = False,
    ):
        super().__init__()
        self.pitch_range = pitch_range
        self.reduce = reduce
        self.num_pitch = pitch_range[1] - pitch_range[0]
        self.dim = dim
        self.use_cross_attn = cross_attend
        self.frame_emb = nn.Embedding(1, dim)
        self.pitch_emb = nn.Embedding(self.num_pitch, dim)
        self.velocity_emb = nn.Embedding(128, dim)
        self.duration_emb = nn.Embedding(max_note_duration, dim)
        self.use_start_end_pos = use_start_end_pos

        if condition_dim > 0:
            self.in_attn_transform = MLP(condition_dim, dim, dim, 1, residual=True)
        else:
            self.in_attn_transform = None

        self.transformer = x_transformers.x_transformers.AttentionLayers(
            causal=is_causal,
            dim=dim,
            depth=num_layers,
            heads=8,
            rotary_pos_emb=True,
            cross_attend=cross_attend,
        )

        beat_pe = one_hot_positional_encoding(num_pos, 32)  # (max_len, dim)
        self.beat_pe: torch.Tensor
        self.register_buffer("beat_pe", beat_pe)

        if self.use_start_end_pos:
            self.start_end_pe = StartEndPosEmb(num_pos, 32)

        if reduce:
            self.out_token_emb = torch.nn.Parameter(torch.randn(dim))
        else:
            self.out_token_emb = None

    def to(self, device: torch.device):  # type: ignore
        super().to(device)
        self.device = device

    def forward(
        self,
        input: TokenSequence,
        shift_from_segment_start: Tensor | None = None,
        segment_duration: Tensor | None = None,
        shift_from_song_start: Tensor | None = None,
        song_duration: Tensor | None = None,
        condition: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        context_pos: torch.Tensor | None = None,
    ):
        """
        - input: SymbolicRepresentation (batch_size, num_tokens)
        - condition: (batch_size, length, dim). Passed to the attention layers with in-attention
        - context: list of SymbolicRepresentation (batch_size, num_tokens). For cross-attention.
        - shift_from_song_start: int. Shift the positional encoding by this amount. The shifted position should not exceed num_pos.

        - returns: features extracted from the input. Used for downstream classification.
        return shape: (batch_size, num_tokens, dim)
        """
        # TODO: use rope

        if self.use_cross_attn:
            assert context is not None, "context must be provided for cross_attn=True"
        else:
            assert context is None, "context is not supported for cross_attn=False"

        if self.use_start_end_pos:
            assert (
                shift_from_segment_start is not None
                and segment_duration is not None
                and shift_from_song_start is not None
                and song_duration is not None
            ), (
                "The constructor is set to use_start_end_pos=True, so shift_from_segment_start, segment_duration, shift_from_song_start, song_duration must be provided"
            )
        else:
            assert (
                shift_from_segment_start is None
                and segment_duration is None
                and shift_from_song_start is None
                and song_duration is None
            ), (
                "The constructor is set to use_start_end_pos=False, so shift_from_segment_start, segment_duration, shift_from_song_start, song_duration, and context must be None"
            )

        # handle zero length input
        if input.length == 0 and not self.reduce:
            return torch.zeros(input.batch_size, 0, self.dim, device=input.device)

        if self.use_start_end_pos:
            pe = torch.cat(
                [
                    self.start_end_pe(shift_from_song_start, song_duration, input.pos),
                    self.start_end_pe(
                        shift_from_segment_start, segment_duration, input.pos
                    ),
                    self.beat_pe[input.pos],
                ],
                dim=2,
            )  # (batch_size, num_tokens, dim_pe)
        else:
            pe = self.beat_pe[input.pos]

        pe = pad_to_length(pe, dim=2, target_length=self.dim, pad_value=0)  # (batch_size, num_tokens, dim)

        x = input.is_note.unsqueeze(-1) * (
            self.pitch_emb(input.pitch) + self.velocity_emb(input.velocity) + self.duration_emb(input.note_duration)
        )  # (batch_size, num_tokens, dim)
        x = x + input.is_frame.unsqueeze(-1) * self.frame_emb(
            torch.zeros_like(input.pitch)
        )  # (batch_size, num_tokens, dim)
        x = x + pe  # (batch_size, num_tokens, dim)

        if self.reduce:
            # add out token to last position
            assert self.out_token_emb is not None
            x = cat_to_right(x, self.out_token_emb, dim=1)

        mask = ~input.is_pad.to(x.device)
        if self.reduce:
            mask = cat_to_right(mask, True, dim=1)

        if condition is not None:
            in_attn_cond = self.in_attn_transform(condition)
        else:
            in_attn_cond = None

        if self.use_cross_attn:
            if context.numel() == 0:
                # x_transformers does not support empty context
                # create a dummy context with one length, assign zero mask so it has no effect
                # please work, im going to sleep
                context = torch.zeros(input.batch_size, 1, self.dim, device=input.device)
                context_mask = torch.zeros(input.batch_size, 1, device=input.device, dtype=torch.bool)
                context_pos = torch.zeros(input.batch_size, 1, device=input.device, dtype=torch.long)

            x = self.transformer(
                x,
                mask=mask,
                in_attn_cond=in_attn_cond,
                pos=input.pos,  #! to check
                context=context,
                context_mask=context_mask,
                context_pos=context_pos,
            )  # (batch_size, num_tokens, dim)
        else:
            x = self.transformer(
                x,
                mask=mask,
                in_attn_cond=in_attn_cond,
            )  # (batch_size, num_tokens, dim)

        if self.reduce:
            # get the last token
            x = x[:, -1, :]  # (batch_size, dim)

        if self.reduce:
            assert x.shape == (input.batch_size, self.dim)
        else:
            assert x.shape == (input.batch_size, input.length, self.dim)

        return x
