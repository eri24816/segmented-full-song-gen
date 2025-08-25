import argparse
from pathlib import Path
import re

from music_data_analysis import Pianoroll
import torch
import safetensors.torch
from segment_full_song import create_model
from segment_full_song.data.segment import get_compose_order
from segment_full_song.models.segment_full_song import SegmentFullSongModel
from segment_full_song.data.dataset import FullSongPianorollDataset



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


def generate(model:SegmentFullSongModel, segments_str: str, compose_order: list[int], seed_segment_pr: Pianoroll, top_p: float = 1):
    segments = parse_segments_str(segments_str)
    generated_song, annotations = model.sample_song(
        labels=[segment["label"] for segment in segments],
        lengths_in_bars=[segment["length_in_bars"] for segment in segments],
        compose_order=list(map(int, compose_order)),
        given_segments=[seed_segment_pr],
        top_p=top_p,
    )
    return generated_song, annotations

def main(args, song_list, structures: list[dict]|None=None):
    model = create_model(args.model_config)
    assert isinstance(model, SegmentFullSongModel)
    model.load_state_dict(safetensors.torch.load_file(args.ckpt))
    if torch.cuda.is_available():
        model.to("cuda")
    model.eval()

    ds = FullSongPianorollDataset(args.dataset_path, split='test')

    for i, song_name in enumerate(song_list):

        song = ds.ds.get_song(song_name)
        segments = song.read_json('segmentation')
        compose_order = get_compose_order(segments)
        seed_segment = compose_order[0]
        seed_segment_pr = Pianoroll.from_midi(song.read_midi('synced_midi')).slice(seed_segment['start'], seed_segment['end'], allow_out_of_range=True)
        assert seed_segment_pr.duration == seed_segment['end'] - seed_segment['start']

        if structures is None:
            structures = [
                {
                    'segments_str': ''.join([f'{"ABCDEF"[segment["label"]]}{segment["end"]-segment["start"]}' for segment in segments]),
                    'compose_order': [segments.index(s) for s in compose_order],
                }
            ]

        for structure in structures:
            print(f"Generating {song_name} with structure {structure['segments_str']}")
            generated_song, annotations = generate(model, structure['segments_str'], structure['compose_order'], seed_segment_pr, top_p=1)
            song_id = song_name.split('/')[1]

            if len(structures) > 1:
                output_name = f"{i}_{song_id}_{re.sub(r'\d+', '', structure['segments_str'])}.mid"
            else:
                output_name = f"{i}_{song_id}.mid"

            generated_output_path = args.output_path / "ours_p1" / output_name
            generated_output_path.parent.mkdir(parents=True, exist_ok=True)

            generated_song.to_midi(generated_output_path, markers=annotations)

            seed_output_path = args.output_path / "seed" / output_name
            seed_output_path.parent.mkdir(parents=True, exist_ok=True)
            seed_segment_pr.to_midi(seed_output_path)

            original_output_path = args.output_path / "original" / output_name
            original_output_path.parent.mkdir(parents=True, exist_ok=True)
            original_pr = Pianoroll.from_midi(song.read_midi('synced_midi'))
            # add markers
            annotations: list[tuple[int, str]] = []
            for segment in segments:
                annotations.append((segment["start"], 'ABCDEF'[segment["label"]]))
            # add '(seed)' to the marker of the seed segment
            for j, annotation in enumerate(annotations):
                if annotation[0] == seed_segment['start']:
                    annotations[j] = (annotation[0], f'{annotation[1]} (seed)')
            original_pr.to_midi(original_output_path, markers=annotations)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--model_config", type=str, default="config/model/segment_full_song.yaml"
    )
    arg_parser.add_argument("--output_path", type=Path, default=Path("eval_results/midi/generated_from_seed"))
    arg_parser.add_argument(
        "--ckpt", type=str, default="pretrained_ckpt/epoch=384-step=2000000.safetensors"
    )
    arg_parser.add_argument(
        "--dataset_path", type=Path
    )
    arg_parser.add_argument(
        "--song_list", type=str, default="eval_results/song_list/8_bars_seed.txt"
    )
    args = arg_parser.parse_args()


    song_list = Path(args.song_list).read_text().splitlines()

    # structures = [
    #     {
    #         'segments_str': 'A8B16C8D8E8',
    #         'compose_order': [2,0,1,3,4],
    #     },
    #     {
    #         'segments_str': 'A4B8C8D8B8C16E8',
    #         'compose_order': [2,5,1,4,3,0,6],
    #     },
    #     {
    #         'segments_str': 'A8B8C4B8D8B8E4',
    #         'compose_order': [1,0,2,3,4,5,6],
    #     },
    #     {
    #         'segments_str': 'A4B8C8A8C8A8D8',
    #         'compose_order': [3, 0, 1, 2, 4, 5, 6],
    #     },
    # ]



    # main(args, song_list, structures)


    main(args, song_list)
