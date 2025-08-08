from dataclasses import dataclass
from typing import Sequence, TypeVar, TypedDict, cast
import miditoolkit
from music_data_analysis import Pianoroll
import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from segment_full_song.data.segment import get_context_for_target_segment
from segment_full_song.models.network import MLP

from segment_full_song.models.encoder_decoder import EncoderDecoder, VAEBottleneck
from segment_full_song.models.feature_extractor import FeatureExtractor
from segment_full_song.models.pe import (
    StartEndPosEmb,
)
from segment_full_song.models.token_sequence import TokenSequence
import x_transformers

from segment_full_song.utils.torch_utils.shape_guard import shape_guard
from loguru import logger
from typing import Callable

DEBUG = True

T = TypeVar("T", bound=miditoolkit.MidiFile | Pianoroll | None)


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
        max_tokens: int,
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
        self.max_tokens = max_tokens
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
            use_start_end_pos=True,
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
            use_start_end_pos=True,
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
        self.duration_classifier = MLP(dim, 256, max_note_duration + 1, 0)

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
        _output="b n self.dim",
    )
    def encode_bar_embeddings(
        self,
        bar_embeddings: torch.Tensor,
        bar_embeddings_mask: torch.Tensor,
        shift_from_song_start: torch.Tensor,
        song_duration: torch.Tensor,
        pos: torch.Tensor | None = None,
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

        if pos is None:
            return global_feature

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
        generate_note_callback: Callable[[tuple[int, int, int, int]], None]
        | None = None,
        top_p: float = 0.95,
    ):
        """
        autoregressive sampling
        max_pos: number of frames of the output
        condition: (batch_size, length, dim)
        """
        assert condition is None, "condition is not supported currently"
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


        global_feature = self.encode_bar_embeddings(
            bar_embeddings,
            bar_embeddings_mask,
            shift_from_song_start_tensor,
            song_duration_tensor,
        )  # (batch_size, num_tokens, dim)

        current_pos = output.duration

        output.add_frame()  # the "start token"

        last_pitch_in_frame = None
        cache = None

        for i in range(output.length, max_length):
            # check if max_length will be reached even if all the remaining tokens are frame tokens
            if duration - output.duration >= max_length - output.length:
                logger.warning("Generated sequence is too long. Pruning.")
                for _ in range(duration - output.duration):
                    output.add_frame()
                break



            feature, cache = self.decoder(
                input=output,
                condition=global_feature[:, (output.pos[0]+shift_from_song_start_tensor[0])//self.frames_per_bar],
                context=encoded_context["context_feature"],
                context_mask=encoded_context["context_mask"],
                context_pos=encoded_context["context_pos"],
                shift_from_song_start=shift_from_song_start_tensor,
                song_duration=song_duration_tensor,
                shift_from_segment_start=shift_from_segment_start_tensor,
                segment_duration=segment_duration_tensor,
                return_hiddens=True,
                # cache=cache,
            )  # (b=1, 1, dim)


            feature = feature[:, -1, :]

            token_type_logits = self.token_type_classifier(feature)[0]  # (class)

            token_type_pred = nucleus_sample(token_type_logits, top_p)  # scalar

            if token_type_pred == TokenSequence.FRAME:  # frame
                current_pos += 1

                if current_pos >= duration:
                    break  # last frame token is ommitted

                output.add_frame()
                last_pitch_in_frame = None

            elif token_type_pred == TokenSequence.NOTE:  # note
                # sample pitch
                pitch_logits = self.pitch_classifier(feature)[0]  # (class)
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
                pitch_pred = nucleus_sample(pitch_logits, top_p)  # scalar

                # sample velocity
                out_pitch_emb = self.out_pitch_emb(pitch_pred.unsqueeze(0))
                velocity_logits = self.velocity_classifier(feature + out_pitch_emb)[
                    0
                ]  # (class)
                velocity_pred = nucleus_sample(velocity_logits, top_p)  # scalar

                # sample duration
                out_velocity_emb = self.out_velocity_emb(velocity_pred.unsqueeze(0))
                duration_logits = self.duration_classifier(
                    feature + out_pitch_emb + out_velocity_emb
                )[0]  # (class)
                duration_pred = nucleus_sample(duration_logits, top_p)  # scalar

                output.add_note(pitch_pred, velocity_pred, duration_pred)
                last_pitch_in_frame = pitch_pred

                if generate_note_callback is not None:
                    generate_note_callback(
                        (
                            shift_from_song_start + current_pos,
                            int(pitch_pred.item() + self.pitch_range[0]),
                            int(velocity_pred.item()),
                            int(duration_pred.item()),
                        )
                    )

            else:
                raise ValueError(f"What is this token type: {token_type_pred}")

        assert output.length <= max_length
        assert output.duration == duration
        return output

    @torch.no_grad()
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
        duration: int | None = None,
        generate_note_callback: Callable[[tuple[int, int, int, int]], None]
        | None = None,
        top_p: float = 0.95,
    ):
        if duration is None:
            duration = segment_duration

        logger.info(
            f"sample_segment: [{shift_from_song_start // 32}, {((shift_from_song_start + duration) // 32)}]"
        )
        device = next(iter(self.parameters())).device
        encoded_context = self.encode_context(
            context, torch.tensor([shift_from_song_start], device=device)
        )

        if segment_duration <= self.max_forward_duration:
            logger.info(f"bar_embeddings_mask: {bar_embeddings_mask.int().tolist()}")
            return self.sample(
                duration=duration,
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
                generate_note_callback=generate_note_callback,
                top_p=top_p,
            )

        else:
            # Segment length is longer than length of self-attention.
            # So we need to sample segment in a loop.
            prompt_duration = prompt.duration if prompt is not None else 0


            if prompt is not None:
                result = prompt.clone()
                prompt_bars = prompt_duration // self.frames_per_bar
                for bar_idx in range(prompt_bars - 8):
                    bar_idx_from_song_start = bar_idx + (
                        shift_from_song_start // self.frames_per_bar
                    )
                    bar_idx_from_song_start = bar_idx + (
                        shift_from_song_start // self.frames_per_bar
                    )
                    bar_embeddings[0, bar_idx_from_song_start] = (
                        self.calculate_bar_embedding(
                            result.slice_pos(
                                bar_idx * self.frames_per_bar,
                                (bar_idx + 1) * self.frames_per_bar,
                            )
                        )
                    )
                    bar_embeddings_mask[0, bar_idx_from_song_start] = True


            else:
                result = TokenSequence(device=device)
            for target_duration in range(
                prompt_duration + self.frames_per_bar,
                duration + self.frames_per_bar,
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

                    logger.info(f"encoded {bar_idx_to_embed_from_song_start}")

                logger.info(
                    f"bar_embeddings_mask: {bar_embeddings_mask.int().tolist()}"
                )
                logger.info(
                    f"sampling [{((shift_from_song_start + truncate) // self.frames_per_bar)}, {((shift_from_song_start + truncate + sample_output_duration) // self.frames_per_bar)})"
                )
                prompt_for_sample = result.slice_pos(truncate, None)

                assert prompt_for_sample.duration <= self.max_forward_duration

                sample_output = self.sample(
                    prompt=prompt_for_sample,
                    duration=prompt_for_sample.duration + self.frames_per_bar,
                    context=encoded_context,
                    shift_from_segment_start=truncate,
                    segment_duration=segment_duration,
                    shift_from_song_start=shift_from_song_start + truncate,
                    song_duration=song_duration,
                    condition=condition,
                    bar_embeddings=bar_embeddings,
                    bar_embeddings_mask=bar_embeddings_mask,
                    generate_note_callback=generate_note_callback,
                    top_p=top_p,
                )

                result += sample_output.slice_pos(
                    sample_input_duration, sample_output_duration
                )

                assert result.duration == target_duration

            assert result.duration == duration
            return result

    @torch.no_grad()
    def sample_song(
        self,
        labels: str | list[str],
        lengths_in_bars: list[int],
        compose_order: list[int],
        given_segments: list[Pianoroll],
        top_p: float = 0.95,
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
                            max_tokens=self.max_tokens,
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

            existing_segments = segment_info_list[:target_segment_idx]
            context_info_dict = get_context_for_target_segment(
                existing_segments, target_segment
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
                            max_tokens=self.max_tokens,
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
                top_p=top_p,
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
            logger.info(
                f"encoded [{composed_segment['start'] // self.frames_per_bar}, {composed_segment['end'] // self.frames_per_bar})"
            )

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
            if i == 0:
                annotation = f'{annotation} (seed)'
            annotations.append((segment["start"], annotation))

        return full_song, annotations

    # class Segment(TypedDict):
    #     """
    #     This class is used to make the SegmentFullSong.generate() api clear and easy to use.
    #     It is different from the SegmentFullSongModel.SegmentItem class.
    #     """

    #     start_bar: int
    #     end_bar: int
    #     label: str
    Segment = TypedDict("Segment", {"start_bar": int, "end_bar": int, "label": str})

    @torch.no_grad()
    def generate(
        self,
        segments: Sequence[Segment],
        target_start_bar: int | None = None,
        target_end_bar: int | None = None,
        existing_pianoroll: Pianoroll | None = None,
        seed_start_bar: int | None = None,
        generate_note_callback: Callable[[tuple[int, int, int, int]], None]
        | None = None,
    ) -> Pianoroll:
        """
        segments: list of segments to specify the song's structure
        existing_pianoroll: the existing music content where the generation will add to
        target_start_bar: the start bar of the target segment
        target_end_bar: the end bar of the target segment (exclusive)
        """

        if target_start_bar is None:
            target_start_bar = 0
        if target_end_bar is None:
            target_end_bar = segments[-1]["end_bar"]
        if existing_pianoroll is None:
            existing_pianoroll = Pianoroll(
                notes=[],
                duration=target_end_bar * self.frames_per_bar,
            )

        logger.info(f"Generating segment from {target_start_bar} to {target_end_bar}")

        device = next(self.parameters()).device
        song_duration = existing_pianoroll.duration
        target_start_frame = target_start_bar * self.frames_per_bar
        target_end_frame = target_end_bar * self.frames_per_bar

        existing_pianoroll.metadata.name = "input"

        def is_complete_segment(segment: SegmentFullSongModel.Segment) -> bool:
            if (
                len(
                    existing_pianoroll.slice(
                        segment["start_bar"] * self.frames_per_bar,
                        segment["end_bar"] * self.frames_per_bar,
                    ).notes
                )
                == 0
            ):
                return False
            # if overlap with target, return False
            if segment["start_bar"] <= target_start_bar < segment["end_bar"]:
                return False
            if segment["start_bar"] < target_end_bar <= segment["end_bar"]:
                return False
            return True

        existing_segments = [
            {
                "start": segment["start_bar"] * self.frames_per_bar,
                "end": segment["end_bar"] * self.frames_per_bar,
                "label": segment["label"],
                "pianoroll": existing_pianoroll.slice(
                    segment["start_bar"] * self.frames_per_bar,
                    segment["end_bar"] * self.frames_per_bar,
                ),
            }
            for segment in segments
            if is_complete_segment(segment)
        ]

        # identify segments that overlap with target segment
        pre_overlap_segment = None
        post_overlap_segment = None
        for segment in segments:
            if segment["start_bar"] < target_start_bar < segment["end_bar"]:
                pr = existing_pianoroll.slice(
                    segment["start_bar"] * self.frames_per_bar,
                    target_start_bar * self.frames_per_bar,
                )
                pre_overlap_segment = {
                    "start": segment["start_bar"] * self.frames_per_bar,
                    "end": segment["end_bar"] * self.frames_per_bar,
                    "label": segment["label"],
                    "pianoroll": pr,
                }

            if segment["start_bar"] < target_end_bar < segment["end_bar"]:
                pr = existing_pianoroll.slice(
                    target_end_bar * self.frames_per_bar,
                    segment["end_bar"] * self.frames_per_bar,
                )
                if len(pr.notes) > 0:
                    post_overlap_segment = {
                        "start": segment["start_bar"] * self.frames_per_bar,
                        "end": segment["end_bar"] * self.frames_per_bar,
                        "label": segment["label"],
                        "pianoroll": pr,
                    }

        segments_to_generate = []
        cursor = target_start_frame
        for segment in segments:
            if (
                cursor >= segment["start_bar"] * self.frames_per_bar
                and cursor < segment["end_bar"] * self.frames_per_bar
            ):
                segments_to_generate.append(
                    {
                        "start": segment["start_bar"] * self.frames_per_bar,
                        "end": segment["end_bar"] * self.frames_per_bar,
                        "label": segment["label"],
                    }
                )
                cursor = segment["end_bar"] * self.frames_per_bar
                if cursor >= target_end_frame:
                    break


        for segment in existing_segments:
            if segment["start"] < target_start_frame:
                logger.info(
                    f'Existing segment "{segment["label"]}": [{segment["start"] // self.frames_per_bar},{segment["end"] // self.frames_per_bar})'
                )


        if pre_overlap_segment is not None:
            logger.info(
                f'Partial existing segment "{pre_overlap_segment["label"]}": [{pre_overlap_segment["start"] // self.frames_per_bar},{target_start_bar})'
            )

        for segment in segments_to_generate:
            logger.info(
                f'Segment to generate "{segment["label"]}": [{max(segment["start"] // self.frames_per_bar, target_start_bar)},{min(segment["end"] // self.frames_per_bar, target_end_bar)})'
            )

        if post_overlap_segment is not None:
            logger.info(
                f'Partial existing segment "{post_overlap_segment["label"]}": [{target_end_bar},{post_overlap_segment["end"] // self.frames_per_bar})'
            )

        for segment in existing_segments:
            if segment["end"] > target_end_frame:
                logger.info(
                    f'Existing segment "{segment["label"]}": [{segment["start"] // self.frames_per_bar},{segment["end"] // self.frames_per_bar})'
                )

        bar_embeddings = torch.zeros(
            1,
            song_duration // self.frames_per_bar,
            cast(VAEBottleneck, self.bar_embedder.bottleneck).output_dim,
            device=device,
        )
        bar_embeddings_mask = torch.zeros(
            (1, song_duration // self.frames_per_bar), device=device, dtype=torch.bool
        )

        def calculate_bar_embeddings_between(
            pianoroll: Pianoroll, start_bar: int, end_bar: int
        ):
            assert pianoroll.duration == (end_bar - start_bar) * self.frames_per_bar
            for local_bar_idx, bar_idx in enumerate(range(start_bar, end_bar)):
                bar_embeddings_mask[0, bar_idx] = True
                bar_embeddings[0, bar_idx] = self.calculate_bar_embedding(
                    TokenSequence.from_pianorolls(
                        [pianoroll.slice(local_bar_idx * self.frames_per_bar, (local_bar_idx + 1) * self.frames_per_bar)],
                        max_note_duration=self.max_note_duration,
                        device=device,
                        max_tokens=self.max_tokens,
                    )
                )

        for segment in existing_segments:
            calculate_bar_embeddings_between(
                segment["pianoroll"],
                segment["start"] // self.frames_per_bar,
                segment["end"] // self.frames_per_bar,
            )

        # if pre_overlap_segment is not None:
        #     # calculate bar embeddings between pre_overlap_segment["start"] and target_start_bar
        #     calculate_bar_embeddings_between(
        #         pre_overlap_segment["pianoroll"],
        #         pre_overlap_segment["start"] // self.frames_per_bar,
        #         target_start_bar,
        #     )

        if (
            post_overlap_segment is not None
            and len(
                existing_pianoroll.slice(
                    target_end_frame, post_overlap_segment["end"]
                ).notes
            )
            > 0
        ):
            # calculate bar embeddings between target_end_bar and post_overlap_segment["end"]
            calculate_bar_embeddings_between(
                post_overlap_segment["pianoroll"],
                target_end_bar,
                post_overlap_segment["end"] // self.frames_per_bar,
            )

        for target_segment in segments_to_generate:
            context_info_dict = get_context_for_target_segment(
                existing_segments, target_segment
            )

            # the get_context_for_target_segment function take the first segment as seed. Override it if seed_start_bar is not None
            if seed_start_bar is not None:
                for segment in existing_segments:
                    if segment["start"] // self.frames_per_bar == seed_start_bar:
                        context_info_dict["seed"] = segment
                        break

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
                                [song_duration], device=device
                            ),
                            "shift_from_segment_start": torch.tensor(
                                [0], device=device
                            ),
                        }
                    )
                    continue

                def find_segment_by_start(segments: list[dict], start: int):
                    for segment in segments:
                        if segment["start"] == start:
                            return segment
                    raise ValueError(f"Segment not found: {start}")

                context_segment = find_segment_by_start(
                    existing_segments, context_info["start"]
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

                context.append(
                    {
                        "tokens": TokenSequence.from_pianorolls(
                            [context_pr],
                            need_frame_tokens=False,
                            max_note_duration=self.max_note_duration,
                            device=device,
                            max_tokens=self.max_tokens,
                        ),
                        "song_duration": torch.tensor([song_duration], device=device),
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
                logger.info(f"context {context_name}: {context_pr}")


            prompt = None
            if (
                pre_overlap_segment is not None
                and target_segment == segments_to_generate[0]
            ):
                prompt = TokenSequence.from_pianorolls(
                    [
                        existing_pianoroll.slice(
                            pre_overlap_segment["start"], target_start_frame
                        )
                    ],
                    max_note_duration=self.max_note_duration,
                    device=device,
                    max_tokens=self.max_tokens,
                )

            if (
                post_overlap_segment is not None
                and target_segment == segments_to_generate[-1]
                and len(
                    existing_pianoroll.slice(
                        target_end_frame, post_overlap_segment["end"]
                    ).notes
                )
                > 0
            ):
                # do a trick that override the left context with the content in the post_overlap_segment
                pr = existing_pianoroll.slice(
                    target_end_frame,
                    min(
                        post_overlap_segment["end"],
                        target_end_frame + self.max_context_duration["right"],
                    ),
                )
                new_right_context = {
                    "tokens": TokenSequence.from_pianorolls(
                        [
                            pr
                        ],
                        max_note_duration=self.max_note_duration,
                        device=device,
                        max_tokens=self.max_tokens,
                    ),
                    "song_duration": torch.tensor([song_duration], device=device),
                    "shift_from_song_start": torch.tensor(
                        [target_end_frame], device=device
                    ),
                    "segment_duration": torch.tensor(
                        [pr.duration],
                        device=device,
                    ),
                    "shift_from_segment_start": torch.tensor([0], device=device),
                }
                context[1] = new_right_context  # type: ignore
                logger.info(f"overriding context right: {pr}")


            # generate target segment

            generated_target = self.sample_segment(
                context=context,
                shift_from_song_start=target_segment["start"],
                segment_duration=target_segment["end"] - target_segment["start"],
                song_duration=song_duration,
                bar_embeddings=bar_embeddings,
                bar_embeddings_mask=bar_embeddings_mask,
                prompt=prompt,
                duration=min(target_segment["end"], target_end_frame) - target_segment["start"],
                generate_note_callback=generate_note_callback,
            ).to_pianoroll()

            # if the last generated segment is partial, extend it to the end of the segment
            if target_segment == segments_to_generate[-1]:
                if post_overlap_segment is not None:
                    generated_target = Pianoroll.cat([generated_target, post_overlap_segment["pianoroll"]])
                else:
                    generated_target.duration = target_segment["end"] - target_segment["start"]
                assert generated_target.duration == target_segment["end"] - target_segment["start"]

            generated_target.metadata.name = f"generated_{target_segment['label']}"

            # merge generated target into existing segments
            existing_segments.append(
                {
                    "start": target_segment["start"],
                    "end": target_segment["end"],
                    "label": target_segment["label"],
                    "pianoroll": generated_target,
                }
            )

            if target_segment != segments_to_generate[-1]:
                # calculate bar embeddings between target_segment["start"] and target_segment["end"]
                calculate_bar_embeddings_between(
                    generated_target,
                    target_segment["start"] // self.frames_per_bar,
                    target_segment["end"] // self.frames_per_bar,
                )

        # return the whole pianoroll
        segments_sorted_by_start = sorted(existing_segments, key=lambda x: x["start"])
        # return Pianoroll.cat(
        #     [segment["pianoroll"] for segment in segments_sorted_by_start]
        # ) >> segments_sorted_by_start[0]["start"]
        result = Pianoroll([], duration=song_duration)
        for segment in segments_sorted_by_start:
            result += segment["pianoroll"] >> segment["start"]
        return result
