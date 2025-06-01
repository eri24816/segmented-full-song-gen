from functools import partial
from pathlib import Path
import random
from typing import TypedDict
import numpy as np
import music_data_analysis
import torch

from segment_full_song.data.dataset import FullSongPianorollDataset
from segment_full_song.models.token_sequence import TokenSequence


def get_compose_order(segments: list[dict]):
    # simulate that human make this music.

    # First, the composer comes up with the seed segment (possibly chorus).
    # To identify the seed segment, the model looks for the segment label with most bars in total.
    # Within segments with this label, it selects the segment that is clost to the middle of the song.
    import random

    duration = max(segment["end"] for segment in segments)
    segment_compose_order = []

    n_bars_per_label = [0] * (max(segment["label"] for segment in segments) + 1)
    for i in range(len(segments)):
        n_bars_per_label[segments[i]["label"]] += (
            segments[i]["end"] - segments[i]["start"]
        ) // 32

    # print("n_bars_per_label", n_bars_per_label)

    label = np.argmax(n_bars_per_label)
    # print("label", label)

    selected_segment = None
    for segment in segments:
        if segment["label"] == label:
            if selected_segment is None:
                selected_segment = segment
            elif abs(segment["start"] - duration // 2) < abs(
                selected_segment["start"] - duration // 2
            ):
                selected_segment = segment

    segment_compose_order.append(selected_segment)

    # Next, the composer writes the second-most bars segment.

    if len(n_bars_per_label) > 2:
        label = np.argsort(n_bars_per_label)[-2]
        # print("label", label)

        selected_segment = None
        for segment in segments:
            if segment["label"] == label:
                if selected_segment is None:
                    selected_segment = segment
                elif abs(segment["start"] - duration // 2) < abs(
                    selected_segment["start"] - duration // 2
                ):
                    selected_segment = segment

        segment_compose_order.append(selected_segment)

    # print("segment_compose_order", segment_compose_order)

    # randomly permute the remaining segments
    remaining_segments = [
        segment for segment in segments if segment not in segment_compose_order
    ]
    random.shuffle(remaining_segments)

    segment_compose_order.extend(remaining_segments)

    # print("segment_compose_order", segment_compose_order)

    return segment_compose_order


def create_testing_dataset(
    path: Path,
    max_duration: int = 100000000000,
    min_duration: int = 0,
):
    def transform(song: music_data_analysis.Song):
        segments_info = song.read_json("segmentation")
        segment_compose_order = get_compose_order(segments_info)
        pr = song.read_pianoroll("pianoroll")
        pr.duration = int(
            song.read_json("duration") / 64 * pr.frames_per_beat
        )  # due to analysis code bug, pr.duration sometimes is not the same as song_duration
        result = {
            "segments": segment_compose_order,
            "pr": pr,
        }
        return result

    ds = FullSongPianorollDataset(
        path,
        transform=transform,
        max_duration=max_duration,
        split="test",
        min_duration=min_duration,
    )
    return ds


class SampleTrainingSegmentsResultItem(TypedDict):
    start: int
    end: int
    shift_from_segment_start: int
    segment_duration: int
    label: int


def get_context_for_target_segment(
    segments: list[dict],
    target_segment: dict,
) -> dict[str, SampleTrainingSegmentsResultItem]:
    target_index = segments.index(target_segment)
    already_composed_segments = segments[:target_index]

    nearest_left_segment = None
    nearest_left_segment_distance = float("inf")
    for segment in reversed(already_composed_segments):
        if segment["end"] > target_segment["start"]:
            continue
        left_segment_distance = target_segment["start"] - segment["end"]
        if left_segment_distance < nearest_left_segment_distance:
            nearest_left_segment_distance = left_segment_distance
            nearest_left_segment = segment

    nearest_right_segment = None
    nearest_right_segment_distance = float("inf")
    for segment in already_composed_segments:
        if segment["start"] < target_segment["end"]:
            continue
        right_segment_distance = segment["start"] - target_segment["end"]
        if right_segment_distance < nearest_right_segment_distance:
            nearest_right_segment_distance = right_segment_distance
            nearest_right_segment = segment

    reference_segment = None
    for segment in already_composed_segments:
        if segment["label"] == target_segment["label"]:
            reference_segment = segment
            break

    if target_index == 0:
        seed_segment = None
    else:
        seed_segment = segments[0]

    # print("target_index", target_index)
    # print("left_segment", nearest_left_segment)
    # print("right_segment", nearest_right_segment)
    # print("seed_segment", seed_segment)
    # print("reference_segment", reference_segment)

    selected_segments = {
        "target": target_segment,
        "left": nearest_left_segment,
        "right": nearest_right_segment,
        "seed": seed_segment,
        "reference": reference_segment,
    }
    return selected_segments


def sample_training_segments(
    segments: list[dict],
    max_context_duration: dict[str, int],
) -> tuple[dict[str, SampleTrainingSegmentsResultItem], list[dict]]:
    # for training, sample a segment from the segment_compose_order
    # target_index = random.randint(0, len(segment_compose_order) - 1)

    segment_compose_order = get_compose_order(segments)

    target_index = random.randint(0, len(segment_compose_order) - 1)
    target_segment = segment_compose_order[target_index]

    selected_segments = get_context_for_target_segment(
        segment_compose_order, target_segment
    )

    result: dict[str, SampleTrainingSegmentsResultItem] = {}
    for k, full_seg in selected_segments.items():
        if full_seg is None:
            result[k] = {
                "start": 0,
                "end": 0,
                "shift_from_segment_start": 0,
                "segment_duration": 0,
                "label": -1,
            }
            continue
        elif full_seg["end"] - full_seg["start"] > max_context_duration[k]:
            if k == "target":
                shift = random.randint(
                    0, full_seg["end"] - full_seg["start"] - max_context_duration[k] - 1
                )
                shift = shift - (shift % 32)  # quantize to bar
                start = full_seg["start"] + shift
                end = start + max_context_duration[k]
            elif k == "left":
                # right most
                start = full_seg["end"] - max_context_duration[k]
                end = full_seg["end"]
            elif k in ["seed", "reference", "right"]:
                # left most
                start = full_seg["start"]
                end = start + max_context_duration[k]
            else:
                raise ValueError(f"Unknown segment type: {k}")
        else:
            start = full_seg["start"]
            end = full_seg["end"]
        assert start < end
        result[k] = {
            "start": start,
            "end": end,
            "shift_from_segment_start": start - full_seg["start"],
            "segment_duration": full_seg["end"] - full_seg["start"],
            "label": full_seg["label"],
        }

    return result, segment_compose_order


def transform(
    song: music_data_analysis.Song,
    bar_embedding_prop: str,
    max_note_duration: int,
    max_context_duration: dict[str, int],
    max_tokens: int | None = None,
    max_tokens_rate: float | None = None,
) -> dict:
    segments_info = song.read_json("segmentation")
    sampled_song_segments, segment_compose_order = sample_training_segments(
        segments_info, max_context_duration
    )
    result = {}
    song_duration = int(song.read_json("duration") / 64 * 8)
    for k, segment in sampled_song_segments.items():
        pr = song.read_pianoroll("pianoroll")
        pr.duration = song_duration  # due to analysis code bug, pr.duration sometimes is not the same as song_duration
        pr = pr.slice(segment["start"], segment["end"])
        result[k] = {
            "pianoroll": pr,
            "tokens": TokenSequence.from_pianorolls(
                [pr],
                need_end_token=k == "target",
                need_frame_tokens=k == "target",
                max_tokens=max_tokens,
                max_tokens_rate=max_tokens_rate,
                max_note_duration=max_note_duration,
            ),
            "shift_from_song_start": segment["start"],  # absolute position in song
            "song_duration": song_duration,
            "shift_from_segment_start": segment["shift_from_segment_start"],
            "segment_duration": segment["segment_duration"],
            "label": segment["label"],
        }
    bar_embeddings = song.read_pt(bar_embedding_prop)  # [n_bars, 128]
    bar_embeddings_mask = torch.zeros(
        bar_embeddings.shape[0], dtype=torch.bool
    )  # [n_bars]

    # only provide bar embeddings for segments prior to the target segment
    target_idx_in_segment_compose_order = None
    for i, segment in enumerate(segment_compose_order):
        if (
            segment["start"]
            <= sampled_song_segments["target"]["start"]
            < segment["end"]
        ):
            target_idx_in_segment_compose_order = i
            target_sample_start = sampled_song_segments["target"]["start"]
            target_segment_start = segment["start"]
            break

    assert target_idx_in_segment_compose_order is not None
    for segment in segment_compose_order[:target_idx_in_segment_compose_order]:
        bar_embeddings_mask[segment["start"] // 32 : segment["end"] // 32] = True

    # also, in the target segment and before the target sample, provide bar embeddings.
    bar_embeddings_mask[target_segment_start // 32 : target_sample_start // 32] = True

    # the multiplication here is not necessary. It just makes the code clear that masked embeddings are not used in training.
    result["bar_embeddings"] = bar_embeddings * bar_embeddings_mask.unsqueeze(1).float()
    result["bar_embeddings_mask"] = bar_embeddings_mask
    return result


def create_training_dataset(
    path: Path,
    max_context_duration: dict[str, int],
    bar_embedding_prop: str,
    max_note_duration: int,
    max_duration: int = 100000000000,
    min_duration: int = 0,
    max_tokens: int | None = None,
    max_tokens_rate: float | None = None,
    train_set_ratio: float = 0.9,
):
    _transform = partial(
        transform,
        bar_embedding_prop=bar_embedding_prop,
        max_tokens=max_tokens,
        max_tokens_rate=max_tokens_rate,
        max_context_duration=max_context_duration,
        max_note_duration=max_note_duration,
    )
    ds = FullSongPianorollDataset(
        path,
        transform=_transform,
        max_duration=max_duration,
        min_duration=min_duration,
        split="train",
        train_set_ratio=train_set_ratio,
    )
    return ds


if __name__ == "__main__":
    # from segment_full_song.data.factory import pr_dataset_collate_fn

    # ds = create_training_dataset(
    #     Path("link/dataset/pop80k_k"),
    #     max_context_duration={"target": 256, "left": 256, "right": 256, "seed": 256, "reference": 256},
    # )
    # pprint(pr_dataset_collate_fn([ds[5], ds[456]]))

    # plot histogram of tokens length
    import numpy as np

    # Load the dataset
    ds = create_training_dataset(
        Path("link/dataset/pop80k_k"),
        max_context_duration={
            "target": 256,
            "left": 256,
            "right": 256,
            "seed": 256,
            "reference": 256,
        },
        bar_embedding_prop="latent_i41ffa2m_1m",
    )
    for i in range(1):
        i = 20502  # random.randint(0, len(ds))
        print(i)
        print(ds[i])
