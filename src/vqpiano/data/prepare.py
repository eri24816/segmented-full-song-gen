import argparse
import json
import shutil
from pathlib import Path

import joblib
from loguru import logger
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--dataset_path", type=Path, required=True, help="Path to the dataset")
    parser.add_argument("--config_path", type=Path, default="config/dataset.json", help="Path to the dataset config")
    parser.add_argument("--save_path", type=Path, required=True, help="Path to save the prepared dataset")
    return parser.parse_args()


def prepare_maestro(dataset_path, save_path, config):
    meta_file = dataset_path / "maestro-v3.0.0.json"
    metadata = json.load(meta_file.open("r"))

    song_ids = list(metadata["midi_filename"].keys())

    manifest = {"split": {"train": [], "validation": [], "test": []}, "path": {}}
    for sid in song_ids:
        manifest["split"][metadata["split"][sid]].append(sid)

    def _process_song(sid):
        midi_path = dataset_path / metadata["midi_filename"][sid]
        song_path = f"files/{sid}"
        output_path = save_path / song_path

        manifest["path"][sid] = song_path
        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True)

        output_midi_path = output_path / "original.mid"
        shutil.copy(midi_path, output_midi_path)
        # process_midi(output_midi_path, fs=config["piano_roll_resolution"])

    (save_path / "manifest.json").write_text(json.dumps(manifest, indent=2))
    joblib.Parallel(n_jobs=-1)(joblib.delayed(_process_song)(sid) for sid in song_ids)


def prepare_pop1k7(dataset_path, save_path, config):
    logger.info("Preparing pop1k7 dataset")

    midi_files = []
    for src_dir in (dataset_path / "midi_analyzed").iterdir():
        for midi_file in src_dir.iterdir():
            midi_files.append(midi_file)
            # print(midi_file)
    midi_files = sorted(midi_files, key=lambda x: int(x.stem))

    def _process_song(midi_path):
        song_path = f"files/{midi_path.stem}"
        output_path = save_path / song_path

        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True)

        output_midi_path = output_path / "original.mid"
        shutil.copy(midi_path, output_midi_path)
        # process_midi(output_midi_path, fs=config["piano_roll_resolution"])

    # (save_path / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for _ in tqdm(
        joblib.Parallel(n_jobs=-1, return_as="generator")((joblib.delayed(_process_song)(sid) for sid in midi_files)),
        total=len(midi_files),
    ):
        pass


if __name__ == "__main__":
    args = parse_args()
    assert args.dataset_path.exists(), f"Dataset path {args.dataset_path} does not exist"
    assert args.save_path.exists(), f"Save path {args.save_path} does not exist"
    assert args.config_path.exists(), f"Config file {args.config_path} does not exist"

    config = json.load(args.config_path.open("r"))

    if args.name == "maestro":
        prepare_maestro(args.dataset_path, args.save_path, config)
    if args.name == "pop1k7":
        prepare_pop1k7(args.dataset_path, args.save_path, config)
    else:
        raise ValueError(f"Unknown dataset name: {args.name}")
