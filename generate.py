import argparse
from pathlib import Path

import torch
import safetensors.torch
from segment_full_song import create_model
from segment_full_song.models.segment_full_song import SegmentFullSongModel


def parse_segments_str(segments_str: str) -> "list[dict]":
    """
    Example:
    A4B16C8D8B8C8E8
    ->
    [
        {"length_in_bars": 4, "label": "A"},
        {"length_in_bars": 16, "label": "B"},
        {"length_in_bars": 8, "label": "C"},
        {"length_in_bars": 8, "label": "D"},
        {"length_in_bars": 8, "label": "B"},
        {"length_in_bars": 8, "label": "C"},
        {"length_in_bars": 8, "label": "E"},
        ...
    ]
    """
    import re

    segments = []

    # Regular expression to match a label followed by a number
    pattern = re.compile(r"([A-Za-z]+)(\d+)")
    matches = pattern.findall(segments_str)

    for label, duration in matches:
        duration = int(duration)
        segment = {
            "length_in_bars": duration,
            "label": label,
        }
        segments.append(segment)

    return segments


def main(args):
    model = create_model(args.model_config)
    assert isinstance(model, SegmentFullSongModel)
    model.load_state_dict(safetensors.torch.load_file(args.ckpt))
    if torch.cuda.is_available():
        model.to("cuda")
    model.eval()

    segments = parse_segments_str(args.segments)
    i = 0
    for _ in range(args.n_samples):
        generated_song, _ = model.sample_song(
            labels=[segment["label"] for segment in segments],
            lengths_in_bars=[segment["length_in_bars"] for segment in segments],
            compose_order=list(map(int, args.compose_order)),
            given_segments=[],
            top_p=1,
        )
        args.output_path.mkdir(parents=True, exist_ok=True)
        output_path = args.output_path / f"{i}.mid"
        while output_path.exists():
            i += 1
            output_path = args.output_path / f"{i}.mid"
        generated_song.to_midi(output_path)
        i += 1


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--model_config", type=str, default="config/model/segment_full_song.yaml"
    )
    arg_parser.add_argument("--output_path", type=Path, default="generated")
    arg_parser.add_argument(
        "--ckpt", type=str, default="pretrained_ckpt/epoch=274-step=1650000.safetensors"
    )
    arg_parser.add_argument(
        "--segments",
        type=str,
        required=True,
        help="Example: A4B8C8D8B8C8E8",
    )
    arg_parser.add_argument(
        "--compose_order",
        nargs="+",
        required=True,
        help="Example: 2 0 1 3 4 5 6",
    )
    arg_parser.add_argument(
        "--n_samples",
        "-n",
        type=int,
        default=1,
    )
    args = arg_parser.parse_args()

    main(args)
