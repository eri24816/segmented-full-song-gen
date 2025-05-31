from dataclasses import dataclass

import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor
from tqdm import tqdm

from .network import MLP

from .feature_extractor import FeatureExtractor
from .pe import (
    binary_positional_encoding,
    one_hot_positional_encoding,
    sinusoidal_positional_encoding,
)
from .token_sequence import TokenSequence


def nucleus_sample(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    logits: (classes)
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


class TokenGenerator(nn.Module):
    """
    A transformer decoder that autoregressively generate a sequence of tokens.
    """

    @dataclass
    class Params:
        dim: int
        num_layers: int
        pitch_range: list[int]
        num_pos: int
        condition_dim: int = 0

    @dataclass
    class Loss:
        token_type_loss: Tensor
        pitch_loss: Tensor
        velocity_loss: Tensor
        duration_loss: Tensor
        total_loss: Tensor
        pitch_acc: float
        velocity_acc: float
        duration_acc: float
        token_type_acc: float

    def __init__(
        self,
        dim: int,
        num_layers: int,
        pitch_range: list[int],
        num_pos: int,
        max_note_duration: int,
        condition_dim: int = 0,
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.pitch_range = pitch_range
        self.num_pos = num_pos
        self.num_pitch = pitch_range[1] - pitch_range[0]
        self.feature_extractor = FeatureExtractor(
            dim=dim,
            num_layers=num_layers,
            pitch_range=pitch_range,
            num_pos=num_pos,
            max_note_duration=max_note_duration,
            reduce=False,
            condition_dim=condition_dim,
            is_causal=True,
        )
        self.token_type_classifier = nn.Linear(dim, 2)
        self.pitch_classifier = nn.Linear(dim, self.num_pitch)
        self.out_pitch_emb = nn.Embedding(self.num_pitch, dim)
        self.velocity_classifier = MLP(dim, 256, 128, 0)
        self.out_velocity_emb = nn.Embedding(128, dim)
        self.duration_classifier = MLP(dim, 256, max_note_duration, 0)

        sinusoidal_pe = sinusoidal_positional_encoding(num_pos, dim - 5 - 32)
        binary_pe = binary_positional_encoding(num_pos, 5)
        one_hot_pe = one_hot_positional_encoding(num_pos, 32)

        pe = torch.cat([binary_pe, one_hot_pe, sinusoidal_pe], dim=1)  # (max_len, dim)
        self.pe: torch.Tensor
        self.register_buffer("pe", pe)

    def forward(
        self,
        x: TokenSequence,
        prompt: TokenSequence | None = None,
        condition: torch.Tensor | None = None,
    ) -> Loss:
        if prompt is not None:
            feature_extractor_input = prompt + x[:, :-1]
        else:
            feature_extractor_input = x[:, :-1]

        feature = self.feature_extractor(
            feature_extractor_input, condition=condition
        )  # (batch_size, num_tokens-1, dim)

        if prompt is not None:
            feature = feature[:, prompt.length :]

        target = x[:, 1:]  # (batch_size, num_tokens-1)

        token_type_logits = self.token_type_classifier(
            feature
        )  # (batch_size, num_tokens-1, 2)
        pitch_logits = self.pitch_classifier(
            feature
        )  # (batch_size, num_tokens-1, vocab_size)

        # velocity classifier can see ground truth pitch
        out_pitch_emb = self.out_pitch_emb(target.pitch)
        velocity_logits = self.velocity_classifier(
            feature + out_pitch_emb
        )  # (batch_size, num_tokens-1, 128)

        # duration classifier can see ground truth pitch and velocity
        out_velocity_emb = self.out_velocity_emb(target.velocity)
        duration_logits = self.duration_classifier(
            feature + out_pitch_emb + out_velocity_emb
        )  # (batch_size, num_tokens-1, max_note_duration)

        assert token_type_logits.shape[:-1] == target.token_type.shape
        assert pitch_logits.shape[:-1] == target.pitch.shape
        assert velocity_logits.shape[:-1] == target.velocity.shape
        assert duration_logits.shape[:-1] == target.note_duration.shape

        token_type_loss = (
            torch.nn.functional.cross_entropy(
                token_type_logits.transpose(1, 2),
                target.token_type,
                ignore_index=-1,
                reduction="none",
            )  # (batch_size, num_tokens-1)
            .sum(dim=1)  # sum over tokens to get -log(p(x))
            .mean()  # average over batch
        )
        pitch_loss = (
            (
                torch.nn.functional.cross_entropy(
                    pitch_logits.transpose(1, 2),
                    target.pitch,
                    ignore_index=-1,
                    reduction="none",
                )
                * target.is_note
            )
            .sum(dim=1)
            .mean()
        )
        velocity_loss = (
            (
                torch.nn.functional.cross_entropy(
                    velocity_logits.transpose(1, 2),
                    target.velocity,
                    ignore_index=-1,
                    reduction="none",
                )
                * target.is_note
            )
            .sum(dim=1)
            .mean()
        )
        duration_loss = (
            (
                torch.nn.functional.cross_entropy(
                    duration_logits.transpose(1, 2),
                    target.note_duration,
                    ignore_index=-1,
                    reduction="none",
                )
                * target.is_note
            )
            .sum(dim=1)
            .mean()
        )
        total_loss = token_type_loss + pitch_loss + velocity_loss + duration_loss

        token_type_acc = (
            (token_type_logits.detach().argmax(dim=2) == target.token_type)
            .float()
            .mean()
            .item()
        )
        pitch_acc = (
            ((pitch_logits.detach().argmax(dim=2) == target.pitch) * target.is_note)
            .float()
            .mean()
            .item()
        )
        velocity_acc = (
            (
                (velocity_logits.detach().argmax(dim=2) == target.velocity)
                * target.is_note
            )
            .float()
            .mean()
            .item()
        )
        duration_acc = (
            (
                (duration_logits.detach().argmax(dim=2) == target.note_duration)
                * target.is_note
            )
            .float()
            .mean()
            .item()
        )

        return TokenGenerator.Loss(
            total_loss=total_loss,
            token_type_loss=token_type_loss,
            pitch_loss=pitch_loss,
            velocity_loss=velocity_loss,
            duration_loss=duration_loss,
            token_type_acc=token_type_acc,
            pitch_acc=pitch_acc,
            velocity_acc=velocity_acc,
            duration_acc=duration_acc,
        )

    @torch.no_grad()
    def sample(
        self,
        duration: int,
        max_length: int | None = None,
        prompt: TokenSequence | None = None,
        condition: torch.Tensor | None = None,
        progress_bar: bool = False,
    ):
        """
        autoregressive sampling
        max_pos: number of frames of the output
        condition: (batch_size, dim)
        """

        assert duration <= self.num_pos, (
            f"duration must be less than or equal to {self.num_pos}"
        )

        if max_length is None:
            max_length = int(duration * 4.5)

        if prompt is not None:
            output = prompt.clone()
        else:
            output = TokenSequence(device=next(iter(self.parameters())).device)

        current_pos = output.duration

        output.add_frame()  # the "start token"

        if progress_bar:
            pbar = tqdm(total=duration - output.duration)

        last_pitch_in_frame = None
        for i in range(output.length, max_length):
            # check if max_length will be reached even if all the remaining tokens are frame tokens
            if duration - output.duration >= max_length - output.length:
                logger.warning("Generated sequence is too long. Pruning.")
                for _ in range(duration - output.duration):
                    output.add_frame()
                    if progress_bar:
                        pbar.update(1)
                break

            feature = self.feature_extractor(
                output, condition=condition
            )  # (b=1, length, dim)
            token_type_logits = self.token_type_classifier(feature[:, -1, :])[
                0
            ]  # (class)
            token_type_pred = nucleus_sample(token_type_logits, 0.95)  # scalar

            if token_type_pred == TokenSequence.FRAME:  # frame
                current_pos += 1

                if current_pos >= duration:
                    break  # last frame token is ommitted

                output.add_frame()
                last_pitch_in_frame = None

                if progress_bar:
                    pbar.update(1)

            elif token_type_pred == TokenSequence.NOTE:  # note
                # sample pitch
                pitch_logits = self.pitch_classifier(feature[:, -1, :])[0]  # (class)
                # predicted pitch must ascend in the same frame
                if last_pitch_in_frame is not None:
                    if last_pitch_in_frame == self.num_pitch - 1:
                        # This would not happen if the model is well trained. This is a protection for validating under-trained model.
                        current_pos += 1

                        if current_pos >= duration:
                            break  # last frame token is ommitted
                        output.add_frame()
                        if progress_bar:
                            pbar.update(1)
                        last_pitch_in_frame = None
                        continue
                    pitch_logits[: last_pitch_in_frame + 1] = -float("inf")
                pitch_pred = nucleus_sample(pitch_logits, 0.95)  # scalar

                # sample velocity
                out_pitch_emb = self.out_pitch_emb(pitch_pred.unsqueeze(0))
                velocity_logits = self.velocity_classifier(
                    feature[:, -1, :] + out_pitch_emb
                )[0]  # (class)
                velocity_pred = nucleus_sample(velocity_logits, 0.95)  # scalar

                # sample duration
                out_velocity_emb = self.out_velocity_emb(velocity_pred.unsqueeze(0))
                duration_logits = self.duration_classifier(
                    feature[:, -1, :] + out_pitch_emb + out_velocity_emb
                )[0]  # (class)
                duration_pred = nucleus_sample(duration_logits, 0.95)  # scalar

                output.add_note(pitch_pred, velocity_pred, duration_pred)
                last_pitch_in_frame = pitch_pred

            else:
                raise ValueError(f"What is this token type: {token_type_pred}")

        assert output.length <= max_length
        assert output.duration == duration
        return output
