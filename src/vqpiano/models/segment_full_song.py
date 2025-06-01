from dataclasses import dataclass
from typing import Sequence, TypedDict, cast
from music_data_analysis import Pianoroll
import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor
from tqdm import tqdm

from vqpiano.data.segment import get_context_for_target_segment
from vqpiano.models.network import MLP

from vqpiano.models.encoder_decoder import EncoderDecoder, VAEBottleneck
from vqpiano.models.feature_extractor import FeatureExtractor
from vqpiano.models.pe import (
    StartEndPosEmb,
)
from vqpiano.models.token_sequence import TokenSequence
import x_transformers

from vqpiano.utils.torch_utils.shape_guard import shape_guard

DEBUG = True


def nucleus_sample(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    logits: (classes,)
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


class GlobalVision(nn.Module):
    def __init__(self, input_dim: int, dim: int, num_layers: int, max_duration: int):
        super().__init__()
        self.dim = dim
        self.in_layer = nn.Linear(input_dim, dim)
        self.encoder = x_transformers.Encoder(
            dim=dim, depth=num_layers, rotary_pos_emb=True
        )
        self.pe = StartEndPosEmb(max_duration=max_duration, dim=dim)

    def forward(self, x: torch.Tensor, duration_in_bars: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, num_bars, dim)
        duration_in_bars: (batch_size)
        mask: (batch_size, num_bars)
        """
        x = self.in_layer(x)
        x += self.pe(
            shift_from_start=torch.zeros_like(duration_in_bars, device=x.device),
            duration=duration_in_bars,
            pos=torch.arange(x.shape[1], device=x.device)
            .unsqueeze(0)
            .repeat(x.shape[0], 1),
        )
        mask = torch.arange(x.shape[1], device=x.device) < duration_in_bars.unsqueeze(1)
        return self.encoder(x, mask=mask)


class SegmentFullSongModel(nn.Module):
    """
    Autoregressive decoder

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
        token_type_acc: float
        duration_acc: float

    class SegmentItem(TypedDict):
        tokens: TokenSequence  # (batch, length)
        song_duration: torch.Tensor  # (batch,)
        shift_from_song_start: torch.Tensor  # (batch,)
        segment_duration: torch.Tensor  # (batch,)
        shift_from_segment_start: torch.Tensor  # (batch,)

    class EncodedContext(TypedDict):
        context_feature: torch.Tensor  # (batch, sum(lengths), dim)
        context_mask: torch.Tensor  # (batch, sum(lengths))
        context_pos: torch.Tensor  # (batch, sum(lengths))

    def __init__(
        self,
        bar_embedder: EncoderDecoder,
        dim: int,
        encoder_num_layers: int,
        decoder_num_layers: int,
        pitch_range: list[int],
        max_note_duration: int,
        max_song_duration: int,
        max_forward_duration: int,
        max_context_duration: dict[str, int],
        max_tokens_rate: float,
        latent_dim: int,
        frames_per_bar: int,
        condition_dim: int = 0,
    ):
        super().__init__()
        # disable gradient for bar_embedder
        for param in bar_embedder.parameters():
            param.requires_grad = False
        self.bar_embedder = bar_embedder
        self.dim = dim
        self.pitch_range = pitch_range
        self.num_pos = max_forward_duration
        self.num_pitch = pitch_range[1] - pitch_range[0]
        self.max_note_duration = max_note_duration
        self.frames_per_bar = frames_per_bar
        self.max_forward_duration = max_forward_duration
        self.max_context_duration = max_context_duration
        self.max_tokens_rate = max_tokens_rate
        self.latent_dim = latent_dim
        self.encoder = FeatureExtractor(
            dim=dim,
            num_layers=encoder_num_layers,
            pitch_range=pitch_range,
            max_note_duration=max_note_duration,
            num_pos=max_song_duration,
            reduce=False,
            is_causal=False,
            cross_attend=False,
        )
        self.decoder = FeatureExtractor(
            dim=dim,
            num_layers=decoder_num_layers,
            pitch_range=pitch_range,
            max_note_duration=max_note_duration,
            num_pos=max_song_duration,
            reduce=False,
            condition_dim=condition_dim + dim,  # add dim for global feature
            is_causal=True,
            cross_attend=True,
        )

        self.global_vision = GlobalVision(
            input_dim=latent_dim,
            dim=dim,
            num_layers=3,
            max_duration=max_song_duration,
        )

        self.token_type_classifier = nn.Linear(dim, 2)
        self.pitch_classifier = nn.Linear(dim, self.num_pitch)
        self.out_pitch_emb = nn.Embedding(self.num_pitch, dim)
        self.velocity_classifier = MLP(dim, 256, 128, 0)
        self.out_velocity_emb = nn.Embedding(128, dim)
        self.duration_classifier = MLP(dim, 256, max_note_duration, 0)

        self.masked_bar_embedding = nn.Parameter(torch.randn(1, 1, latent_dim))

        self.context_type_embedding = nn.Embedding(4, dim)

    def encode_context(
        self, context: Sequence[SegmentItem], x_shift_from_song_start: torch.Tensor
    ) -> EncodedContext:
        device = context[0]["tokens"].device

        context_features = []
        for i, context_item in enumerate(context):
            context_features.append(
                self.encoder(
                    input=context_item["tokens"],
                    shift_from_song_start=context_item["shift_from_song_start"],
                    shift_from_segment_start=context_item["shift_from_segment_start"],
                    segment_duration=context_item["segment_duration"],
                    song_duration=context_item["song_duration"],
                )
                + self.context_type_embedding(torch.tensor([i], device=device))
            )  # (batch_size, length, dim)

        context_feature = torch.cat(
            context_features, dim=1
        )  # (batch_size, sum(lengths), dim)

        context_masks = []
        for context_item in context:
            context_masks.append(~context_item["tokens"].is_pad.to(device))
        context_mask = torch.cat(context_masks, dim=1)  # (batch_size, sum(lengths))

        # relative position to the target
        context_positions = []
        for context_item in context:
            context_positions.append(
                context_item["tokens"].pos
                + context_item["shift_from_song_start"].unsqueeze(1)
                - x_shift_from_song_start.unsqueeze(1)
            )
        context_pos = torch.cat(context_positions, dim=1)  # (batch_size, sum(lengths))

        return SegmentFullSongModel.EncodedContext(
            context_feature=context_feature,
            context_mask=context_mask,
            context_pos=context_pos,
        )

    @shape_guard(
        bar_embeddings="b bar self.latent_dim",
        bar_embeddings_mask="b bar",
        shift_from_song_start="b",
        song_duration="b",
        pos="b n",
        _output="b n self.dim",
    )
    def encode_bar_embeddings(
        self,
        bar_embeddings: torch.Tensor,
        bar_embeddings_mask: torch.Tensor,
        shift_from_song_start: torch.Tensor,
        song_duration: torch.Tensor,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        bar_embeddings: (batch_size, num_bars, dim)
        bar_embeddings_mask: (batch_size, num_bars)
        """
        duration_in_bars = (song_duration // self.frames_per_bar).long()

        bar_embeddings_mask = bar_embeddings_mask.unsqueeze(
            2
        )  # (batch_size, num_bars, 1)
        bar_embeddings = (
            bar_embeddings * bar_embeddings_mask
            + self.masked_bar_embedding * (~bar_embeddings_mask)
        )
        global_feature = self.global_vision(
            bar_embeddings, duration_in_bars
        )  # (batch_size, num_bars, dim)

        global_feature_batch = []
        for i in range(global_feature.shape[0]):
            global_feature_batch.append(
                global_feature[
                    i, (pos[i] + shift_from_song_start[i]) // self.frames_per_bar
                ]
            )
        global_feature = torch.stack(
            global_feature_batch, dim=0
        )  # (batch_size, num_tokens, dim)

        return global_feature

    @shape_guard(_output="1 self.latent_dim")
    def calculate_bar_embedding(self, tokens: TokenSequence) -> torch.Tensor:
        """
        input: one bar of tokens. No padding or frame tokens, only notes. Batch size is 1.
        output: bar embedding (batch_size=1, dim)
        """
        assert tokens.duration == self.frames_per_bar
        assert tokens.batch_size == 1
        assert (~tokens.is_pad).all(), f"tokens: {tokens}"
        return self.bar_embedder.encode(tokens)

    @shape_guard(
        x="b n",
        bar_embeddings="b bar self.latent_dim",
        bar_embeddings_mask="b bar",
    )
    def forward(
        self,
        x: SegmentItem,
        context: list[SegmentItem],
        bar_embeddings: torch.Tensor,
        bar_embeddings_mask: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> Loss:
        """
        x: current segment
        context: some other segments
        bar_embeddings: bar embeddings (batch_size, num_bars, dim)
        bar_embeddings_mask: (batch_size, num_bars)
        """
        assert x["tokens"].duration <= self.max_forward_duration

        # process context through encoder

        encoded_context = self.encode_context(context, x["shift_from_song_start"])

        decoder_input = x["tokens"][:, :-1]

        global_feature = self.encode_bar_embeddings(
            bar_embeddings,
            bar_embeddings_mask,
            x["shift_from_song_start"],
            x["song_duration"],
            decoder_input.pos,
        )  # (batch_size, num_tokens, dim)

        if condition is not None:
            condition = torch.cat([condition, global_feature], dim=1)
        else:
            condition = global_feature

        feature = self.decoder(
            input=decoder_input,
            condition=condition,
            context=encoded_context["context_feature"],
            context_mask=encoded_context["context_mask"],
            context_pos=encoded_context["context_pos"],
            shift_from_song_start=x["shift_from_song_start"],
            song_duration=x["song_duration"],
            shift_from_segment_start=x["shift_from_segment_start"],
            segment_duration=x["segment_duration"],
        )  # (batch_size, num_tokens, dim)

        target = x["tokens"][:, 1:]  # (batch_size, num_tokens-1)

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

        @shape_guard(logits="b n c", target="b n", mask="b n")
        def calculate_categorical_loss(logits, target, mask, ignore_index=-1):
            return (
                (
                    torch.nn.functional.cross_entropy(
                        logits.transpose(1, 2),
                        target,
                        ignore_index=ignore_index,
                        reduction="none",
                    )
                    * mask
                )
                .sum(dim=1)
                .mean()
            )

        token_type_loss = calculate_categorical_loss(
            token_type_logits, target.token_type, target.is_frame + target.is_note
        )

        pitch_loss = calculate_categorical_loss(
            pitch_logits, target.pitch, target.is_note
        )

        velocity_loss = calculate_categorical_loss(
            velocity_logits, target.velocity, target.is_note
        )

        duration_loss = calculate_categorical_loss(
            duration_logits, target.note_duration, target.is_note
        )

        total_loss = token_type_loss + pitch_loss + velocity_loss + duration_loss

        @shape_guard(logits="b n c", target="b n", mask="b n")
        def calculate_categorical_acc(logits, target, mask):
            return (
                ((logits.detach().argmax(dim=2) == target) * mask).float().sum()
                / mask.sum()
            ).item()

        token_type_acc = calculate_categorical_acc(
            token_type_logits, target.token_type, target.is_frame + target.is_note
        )

        pitch_acc = calculate_categorical_acc(
            pitch_logits, target.pitch, target.is_note
        )

        velocity_acc = calculate_categorical_acc(
            velocity_logits, target.velocity, target.is_note
        )

        duration_acc = calculate_categorical_acc(
            duration_logits, target.note_duration, target.is_note
        )

        return SegmentFullSongModel.Loss(
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
        context: Sequence[SegmentItem] | EncodedContext,
        shift_from_segment_start: int,
        segment_duration: int,
        shift_from_song_start: int,
        song_duration: int,
        bar_embeddings: torch.Tensor,
        bar_embeddings_mask: torch.Tensor,
        max_length: int | None = None,
        prompt: TokenSequence | None = None,
        condition: torch.Tensor | None = None,
    ):
        """
        autoregressive sampling
        max_pos: number of frames of the output
        condition: (batch_size, length, dim)
        """
        device = next(iter(self.parameters())).device
        assert duration <= self.num_pos, (
            f"duration must be less than or equal to {self.num_pos}"
        )

        if max_length is None:
            max_length = int(duration * 4.5)

        if isinstance(context, Sequence):
            encoded_context = self.encode_context(
                context, torch.tensor([shift_from_song_start], device=device)
            )
        else:
            encoded_context = context

        if prompt is not None:
            output = prompt.clone()
        else:
            output = TokenSequence(device=device)

        shift_from_song_start_tensor = torch.tensor(
            [shift_from_song_start], device=device
        )
        song_duration_tensor = torch.tensor([song_duration], device=device)
        shift_from_segment_start_tensor = torch.tensor(
            [shift_from_segment_start], device=device
        )
        segment_duration_tensor = torch.tensor([segment_duration], device=device)

        current_pos = output.duration

        output.add_frame()  # the "start token"

        last_pitch_in_frame = None

        for i in range(output.length, max_length):
            # check if max_length will be reached even if all the remaining tokens are frame tokens
            if duration - output.duration >= max_length - output.length:
                logger.warning("Generated sequence is too long. Pruning.")
                for _ in range(duration - output.duration):
                    output.add_frame()
                break

            global_feature = self.encode_bar_embeddings(
                bar_embeddings,
                bar_embeddings_mask,
                shift_from_song_start_tensor,
                song_duration_tensor,
                output.pos,
            )  # (batch_size, num_tokens, dim)

            if condition is not None:
                actual_condition = torch.cat([condition, global_feature], dim=2)
            else:
                actual_condition = global_feature

            feature = self.decoder(
                input=output,
                condition=actual_condition,
                context=encoded_context["context_feature"],
                context_mask=encoded_context["context_mask"],
                context_pos=encoded_context["context_pos"],
                shift_from_song_start=shift_from_song_start_tensor,
                song_duration=song_duration_tensor,
                shift_from_segment_start=shift_from_segment_start_tensor,
                segment_duration=segment_duration_tensor,
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

    def sample_segment(
        self,
        context: Sequence[SegmentItem],
        shift_from_song_start: int,
        segment_duration: int,
        song_duration: int,
        bar_embeddings: torch.Tensor,
        bar_embeddings_mask: torch.Tensor,
        max_length: int | None = None,
        prompt: TokenSequence | None = None,
        condition: torch.Tensor | None = None,
    ):
        if DEBUG:
            print(
                f"sample_segment: {shift_from_song_start // 32}, {((shift_from_song_start + segment_duration) // 32)}"
            )
        device = next(iter(self.parameters())).device
        encoded_context = self.encode_context(
            context, torch.tensor([shift_from_song_start], device=device)
        )

        if segment_duration <= self.max_forward_duration:
            if DEBUG:
                print(bar_embeddings_mask.int())
            return self.sample(
                duration=segment_duration,
                context=encoded_context,
                shift_from_segment_start=0,
                segment_duration=segment_duration,
                shift_from_song_start=shift_from_song_start,
                song_duration=song_duration,
                bar_embeddings=bar_embeddings,
                bar_embeddings_mask=bar_embeddings_mask,
                max_length=max_length,
                prompt=prompt,
                condition=condition,
            )

        else:
            # Segment length is longer than length of self-attention.
            # So we need to sample segment in a loop.
            result = TokenSequence(device=device)
            for target_duration in range(
                self.frames_per_bar,
                segment_duration + self.frames_per_bar,
                self.frames_per_bar,
            ):  # 32, 64, ...
                sample_output_duration = min(target_duration, self.max_forward_duration)
                sample_input_duration = sample_output_duration - self.frames_per_bar

                # truncate if result is too long
                truncate = max(0, result.duration - sample_input_duration)

                if truncate > 0:
                    # if truncate is non-zero, embed last truncated bar.
                    # following indices are bar indices relative to segment start.
                    first_non_truncated_bar = truncate // self.frames_per_bar
                    last_truncated_bar = first_non_truncated_bar - 1
                    bar_idx_to_embed = last_truncated_bar

                    tokens_to_embed = result.slice_pos(
                        bar_idx_to_embed * self.frames_per_bar,
                        (bar_idx_to_embed + 1) * self.frames_per_bar,
                    )

                    # bar_embeddings and bar_embeddings_mask are indexed relative to song start, so we need to convert bar_idx_to_embed to bar_idx_to_embed_from_song_start.
                    bar_idx_to_embed_from_song_start = bar_idx_to_embed + (
                        shift_from_song_start // self.frames_per_bar
                    )
                    bar_embeddings[0, bar_idx_to_embed_from_song_start] = (
                        self.calculate_bar_embedding(tokens_to_embed)
                    )
                    bar_embeddings_mask[0, bar_idx_to_embed_from_song_start] = True

                    if DEBUG:
                        print(
                            "encoded",
                            last_truncated_bar,
                            "duration",
                            result.duration // self.frames_per_bar,
                            "slice",
                            (bar_idx_to_embed, bar_idx_to_embed + 1),
                            "sliced_duration",
                            tokens_to_embed.duration // self.frames_per_bar,
                        )

                if DEBUG:
                    print(bar_embeddings_mask.int())
                    print(result.duration, sample_input_duration, truncate)
                    print(
                        "sample:",
                        (shift_from_song_start + truncate) // self.frames_per_bar,
                        "to",
                        (shift_from_song_start + truncate + sample_output_duration)
                        // self.frames_per_bar,
                    )
                prompt = result.slice_pos(truncate, None)

                assert prompt.duration <= self.max_forward_duration

                sample_output = self.sample(
                    prompt=prompt,
                    duration=prompt.duration + self.frames_per_bar,
                    context=encoded_context,
                    shift_from_segment_start=truncate,
                    segment_duration=segment_duration,
                    shift_from_song_start=shift_from_song_start + truncate,
                    song_duration=song_duration,
                    condition=condition,
                    bar_embeddings=bar_embeddings,
                    bar_embeddings_mask=bar_embeddings_mask,
                )

                result += sample_output.slice_pos(
                    sample_input_duration, sample_output_duration
                )

                assert result.duration == target_duration

            assert result.duration == segment_duration
            return result

    def sample_song(
        self,
        labels: str,
        lengths_in_bars: list[int],
        compose_order: list[int],
        given_segments: list[Pianoroll],
    ):
        """
        labels: list of labels
        lengths_in_bars: list of segment lengths in bars
        compose_order: list of segment indices
        given_segments: list of given segments. Must be first n of compose_order.
        """
        device = next(iter(self.parameters())).device
        assert len(labels) == len(lengths_in_bars) == len(compose_order)

        lengths = [32 * i for i in lengths_in_bars]
        segment_info_list = []

        for i, (label, length) in enumerate(zip(labels, lengths)):
            segment_info_list.append(
                {
                    "start": sum(lengths[:i]),
                    "end": sum(lengths[:i]) + length,
                    "label": label,
                }
            )
        # rearrange by compose_order
        segment_info_list = [segment_info_list[i] for i in compose_order]

        if DEBUG:
            for segment in segment_info_list:
                print(
                    f"segment: {segment['start'] // 32}, {segment['end'] // 32}, {segment['label']}"
                )

        full_duration = sum(lengths)

        composed_segments = []

        for given_segment_idx, given_segment in enumerate(given_segments):
            assert (
                given_segment.duration
                == segment_info_list[given_segment_idx]["end"]
                - segment_info_list[given_segment_idx]["start"]
            ), (
                f"{given_segment_idx}th given segment duration mismatch: expected {segment_info_list[given_segment_idx]['end'] - segment_info_list[given_segment_idx]['start']}, got {given_segment.duration}"
            )
            composed_segments.append(
                {
                    "pianoroll": given_segment,
                    "start": segment_info_list[given_segment_idx]["start"],
                    "end": segment_info_list[given_segment_idx]["end"],
                    "label": segment_info_list[given_segment_idx]["label"],
                }
            )

        # embed bars in given segments
        bar_embeddings = torch.zeros(
            1,
            full_duration // self.frames_per_bar,
            cast(VAEBottleneck, self.bar_embedder.bottleneck).output_dim,
            device=device,
        )
        bar_embeddings_mask = torch.zeros(
            bar_embeddings.shape[0:2], device=device, dtype=torch.bool
        )

        def calculate_bar_embeddings_for_segment(segment: dict):
            for pos_in_song in range(
                segment["start"], segment["end"], self.frames_per_bar
            ):
                pos_in_segment = pos_in_song - segment["start"]
                if DEBUG:
                    print(
                        "aaa",
                        segment["pianoroll"]
                        .slice(pos_in_segment, pos_in_segment + self.frames_per_bar)
                        .duration,
                        segment["pianoroll"].duration,
                        pos_in_segment,
                        pos_in_segment + self.frames_per_bar,
                    )
                bar_embeddings[0, pos_in_song // self.frames_per_bar] = (
                    self.calculate_bar_embedding(
                        TokenSequence.from_pianorolls(
                            [
                                segment["pianoroll"].slice(
                                    pos_in_segment, pos_in_segment + self.frames_per_bar
                                )
                            ],
                            max_note_duration=self.max_note_duration,
                            device=device,
                            max_tokens_rate=self.max_tokens_rate,
                        )
                    )
                )

                bar_embeddings_mask[0, pos_in_song // self.frames_per_bar] = True

        for segment in composed_segments:
            calculate_bar_embeddings_for_segment(segment)

        def find_segment_by_start(segments: list[dict], start: int):
            for segment in segments:
                if segment["start"] == start:
                    return segment
            raise ValueError(f"Segment not found: {start}")

        num_given_segments = len(composed_segments)

        # this loop generates each segment
        pbar = tqdm(segment_info_list[num_given_segments:])
        for target_segment_idx, target_segment in enumerate(pbar):
            target_segment_idx += num_given_segments

            """ prepare context for generation """

            context_info_dict = get_context_for_target_segment(
                segment_info_list, target_segment
            )
            context: list[SegmentFullSongModel.SegmentItem] = []
            for context_name in ["left", "right", "seed", "reference"]:
                context_info = context_info_dict[context_name]
                if context_info is None:
                    # just a dummy context with zero duration, which will take no effect in the model
                    context.append(
                        {
                            "tokens": TokenSequence(device=device),
                            "shift_from_song_start": torch.tensor([0], device=device),
                            "segment_duration": torch.tensor([0], device=device),
                            "song_duration": torch.tensor(
                                [full_duration], device=device
                            ),
                            "shift_from_segment_start": torch.tensor(
                                [0], device=device
                            ),
                        }
                    )
                    continue
                context_segment = find_segment_by_start(
                    composed_segments, context_info["start"]
                )
                max_context_duration = self.max_context_duration[context_name]
                context_duration = context_segment["end"] - context_segment["start"]
                if context_duration > max_context_duration:
                    if context_name == "left":
                        context_pr = context_segment["pianoroll"][
                            context_duration - max_context_duration :
                        ]
                        shift_from_song_start = (
                            context_segment["end"] - max_context_duration
                        )
                        shift_from_segment_start = (
                            shift_from_song_start - context_segment["start"]
                        )
                    else:
                        context_pr = context_segment["pianoroll"][:max_context_duration]
                        shift_from_song_start = context_segment["start"]
                        shift_from_segment_start = 0
                else:
                    context_pr = context_segment["pianoroll"]
                    shift_from_song_start = context_segment["start"]
                    shift_from_segment_start = 0

                if DEBUG:
                    print(
                        context_name,
                        context_pr.duration,
                        context_pr,
                        {
                            "song_duration": full_duration // self.frames_per_bar,
                            "shift_from_song_start": shift_from_song_start
                            // self.frames_per_bar,
                            "segment_duration": context_duration // self.frames_per_bar,
                            "shift_from_segment_start": shift_from_segment_start
                            // self.frames_per_bar,
                        },
                    )

                context.append(
                    {
                        "tokens": TokenSequence.from_pianorolls(
                            [context_pr],
                            need_frame_tokens=False,
                            max_note_duration=self.max_note_duration,
                            device=device,
                            max_tokens_rate=self.max_tokens_rate,
                        ),
                        "song_duration": torch.tensor([full_duration], device=device),
                        "shift_from_song_start": torch.tensor(
                            [shift_from_song_start], device=device
                        ),
                        "segment_duration": torch.tensor(
                            [context_segment["end"] - context_segment["start"]],
                            device=device,
                        ),
                        "shift_from_segment_start": torch.tensor(
                            [shift_from_segment_start], device=device
                        ),
                    }
                )

            """ generate segment """

            pbar.set_postfix(
                {
                    "segment_duration": target_segment["end"] - target_segment["start"],
                    "label": target_segment["label"],
                }
            )
            sample = self.sample_segment(
                context=context,
                shift_from_song_start=target_segment["start"],
                segment_duration=target_segment["end"] - target_segment["start"],
                song_duration=full_duration,
                bar_embeddings=bar_embeddings,
                bar_embeddings_mask=bar_embeddings_mask,
            )

            min_pitch = self.pitch_range[0]

            composed_segment = {
                "pianoroll": sample.to_pianoroll(min_pitch),
                "start": target_segment["start"],
                "end": target_segment["end"],
                "label": target_segment["label"],
            }

            # debug purpose
            cast(Pianoroll, composed_segment["pianoroll"]).metadata.name = (
                target_segment["label"] + str(target_segment_idx)
            )

            composed_segments.append(composed_segment)

            calculate_bar_embeddings_for_segment(composed_segment)

        composed_segments_sorted = sorted(composed_segments, key=lambda x: x["start"])
        assert composed_segments_sorted[0]["start"] == 0
        assert composed_segments_sorted[-1]["end"] == full_duration

        for i in range(len(composed_segments_sorted) - 1):
            # assert composed_segments_sorted[i]["end"] == composed_segments_sorted[i + 1]["start"]
            if (
                composed_segments_sorted[i]["end"]
                != composed_segments_sorted[i + 1]["start"]
            ):
                print(
                    f"Segment {i} and {i + 1} are not contiguous: {composed_segments_sorted[i]['end']} != {composed_segments_sorted[i + 1]['start']}"
                )
                print("compose_segment_sorted", composed_segments_sorted)

        full_song = Pianoroll.cat(
            [segment["pianoroll"] for segment in composed_segments_sorted]
        )
        # log generated segment

        annotations: list[tuple[int, str]] = []
        for i, segment in enumerate(composed_segments):
            annotation = segment["label"] + str(i)
            annotations.append((segment["start"], annotation))

        return full_song, annotations


# def test():
#     device = "cpu"
#     from vqpiano.models.factory import create_model

#     encoder_decoder = cast(
#         EncoderDecoder,
#         create_model(OmegaConf.load("config/simple_ar/model_vae.yaml").model),
#     )
#     from safetensors.torch import load_file

#     encoder_decoder.load_state_dict(
#         load_file(
#             "wandb/run-20250404_013005-i41ffa2m/files/checkpoints/epoch=4-step=1000000.safetensors",
#             device=device,
#         )
#     )

#     model = SegmentFullSongModel(
#         bar_embedder=encoder_decoder,
#         dim=128,
#         encoder_num_layers=6,
#         decoder_num_layers=6,
#         pitch_range=[21, 108],
#         max_forward_duration=32 * 8,
#         max_song_duration=32 * 256,
#         max_context_duration={"left": 32, "right": 32, "seed": 32, "reference": 32},
#         max_tokens_rate=4.5,
#         latent_dim=128,
#         frames_per_bar=32,
#         condition_dim=0,
#     ).to(device)

#     context: Sequence[SegmentFullSongModel.SegmentItem] = [
#         {
#             "tokens": TokenSequence.from_pianorolls(
#                 [
#                     Pianoroll([Note(0, 60, 100), Note(3, 62, 100)]),
#                 ]
#             ).to(device),
#             "shift_from_song_start": torch.tensor([32]).to(device),
#             "segment_duration": torch.tensor([32 * 8]).to(device),
#             "shift_from_segment_start": torch.tensor([0]).to(device),
#             "song_duration": torch.tensor([1024]).to(device),
#         },
#         {
#             "tokens": TokenSequence.from_pianorolls(
#                 [
#                     Pianoroll(
#                         [Note(20, 60, 100), Note(23, 62, 100), Note(28, 62, 100)]
#                     ),
#                 ]
#             ).to(device),
#             "shift_from_song_start": torch.tensor([-32]).to(device),
#             "segment_duration": torch.tensor([32 * 8]).to(device),
#             "shift_from_segment_start": torch.tensor([0]).to(device),
#             "song_duration": torch.tensor([1024]).to(device),
#         },
#     ]

#     model.eval()
#     output, _ = model.sample_song(
#         labels="ABAB",
#         lengths_in_bars=[1, 2, 10, 12],
#         compose_order=[0, 1, 3, 2],
#         given_segments=[
#             Pianoroll([Note(0, 60, 100), Note(3, 28, 100)]),
#             Pianoroll([Note(20, 60, 100), Note(23, 62, 100), Note(61, 62, 100)]),
#         ],
#     )
#     output.to_midi("output.mid")

#     context: Sequence[SegmentFullSongModel.SegmentItem] = [
#         {
#             "tokens": TokenSequence.from_pianorolls(
#                 [
#                     Pianoroll([Note(0, 60, 100), Note(3, 62, 100)]),
#                     Pianoroll([Note(0, 60, 100), Note(61, 62, 100)]),
#                 ]
#             ).to(device),
#             "shift_from_song_start": torch.tensor([32, 128]).to(device),
#             "segment_duration": torch.tensor([32 * 8, 32 * 8]).to(device),
#             "shift_from_segment_start": torch.tensor([0, 0]).to(device),
#             "song_duration": torch.tensor([1024, 1024]).to(device),
#         },
#         {
#             "tokens": TokenSequence.from_pianorolls(
#                 [
#                     Pianoroll(
#                         [Note(20, 60, 100), Note(23, 62, 100), Note(28, 62, 100)]
#                     ),
#                     Pianoroll([Note(20, 60, 100), Note(23, 62, 100)]),
#                 ]
#             ).to(device),
#             "shift_from_song_start": torch.tensor([128, 64]).to(device),
#             "segment_duration": torch.tensor([32 * 8, 32 * 8]).to(device),
#             "shift_from_segment_start": torch.tensor([0, 0]).to(device),
#             "song_duration": torch.tensor([1024, 1024]).to(device),
#         },
#     ]

#     x = TokenSequence.from_pianorolls(
#         [
#             Pianoroll([Note(0, 60, 100), Note(3, 62, 100), Note(3, 89, 100)]),
#             Pianoroll([Note(0, 60, 100), Note(61, 62, 100)]),
#         ]
#     ).to(device)

#     x = SegmentFullSongModel.SegmentItem(
#         tokens=x,
#         shift_from_song_start=torch.tensor([0, 0]).to(device),
#         segment_duration=torch.tensor([32 * 8, 32 * 8]).to(device),
#         shift_from_segment_start=torch.tensor([0, 0]).to(device),
#         song_duration=torch.tensor([1024, 1024]).to(device),
#     )

#     bar_embeddings = torch.zeros(
#         2,
#         16,
#         cast(VAEBottleneck, encoder_decoder.bottleneck).output_dim,
#         device=device,
#     )
#     bar_embeddings_mask = torch.ones(2, 16, device=device, dtype=torch.bool)
#     print(
#         model(
#             x=x,
#             context=context,
#             bar_embeddings=bar_embeddings,
#             bar_embeddings_mask=bar_embeddings_mask,
#         )
#     )


# if __name__ == "__main__":
#     test()
