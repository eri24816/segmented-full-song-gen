import json
import math
from functools import cache
from pathlib import Path
from typing import Literal

import music_data_analysis
import numpy as np
import pretty_midi
import torch
from loguru import logger
from torch.nn import functional as F
from torch.utils.data import Dataset, IterableDataset

from vqpiano.data.utils import get_piano_roll_onset
from vqpiano.utils.tokenizer import Tokenizer
from vqpiano.utils.vocab import Word


class MaestroDataset(Dataset):
    def __init__(self, path):
        self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]

        return data


class Pop1k7Dataset(Dataset):
    def __init__(self, path, config, partition="train", cache=True):
        logger.info(f"Loading dataset from {path}")
        self.path = Path(path)
        self.resolution = config["piano_roll_resolution"]
        self.midi_files: list[Path] = []
        self.cache = cache

        for src_dir in (self.path / "midi_analyzed").iterdir():
            for midi_file in src_dir.iterdir():
                self.midi_files.append(midi_file)
        self.midi_files = sorted(self.midi_files, key=lambda x: int(x.stem))

    def __len__(self):
        return len(self.midi_files)

    def __getitem__(self, idx: int):
        logger.trace(f"Loading {self.midi_files[idx]}")
        midi_file = self.midi_files[idx]

        return {
            "midi_file": midi_file,
            "midi_data": self.load_midi_data(midi_file),
        }

    def load_midi_data(self, midi_path):
        if self.cache:
            fn = self._load_midi_data_cache
        else:
            fn = self._load_midi_data_cache.__wrapped__

        return fn(midi_path)

    @cache
    def _load_midi_data_cache(self, midi_path):
        midi_data = pretty_midi.PrettyMIDI(str(midi_path))
        assert len(midi_data.instruments) == 1, f"Expected 1 instrument, got {len(midi_data.instruments)}, {midi_path}"
        return midi_data


class Pop80kDataset(Dataset):
    def __init__(
        self,
        path: Path,
        resolution,
        min_duration,
        max_duration,
        min_note_count,
        sampling_frequency,
        pitch_range,
        frame_width,
        tokenizer: Tokenizer,
        use_cache=False,
    ):
        logger.info(f"Loading Pop80kDataset from {path}")
        self.path = path
        self.midi_files: list[Path] = []
        self.use_cache = use_cache

        self.resolution = resolution
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_note_count = min_note_count

        self.fs = sampling_frequency
        self.pitch_range = pitch_range
        self.frame_width = frame_width
        self.tokenizer = tokenizer

        # this manifest is large. It should be dereferenced in this function so it won't be copied to loader workers
        manifest = json.load((path / "manifest.json").open("r"))
        for file_name, metadata in manifest["samples"].items():
            if min_duration is not None:
                if metadata["duration"] < min_duration:
                    continue
            if max_duration is not None:
                if metadata["duration"] > max_duration:
                    continue
            if min_note_count is not None:
                if metadata["note_count"] < min_note_count:
                    continue

            # only store the file name, not the full path
            self.midi_files.append(file_name)

        logger.info(f"Number of valid files: {len(self.midi_files)}")

    def __len__(self):
        return len(self.midi_files)

    def __getitem__(self, idx: int):
        logger.trace(f"Loading {self.midi_files[idx]}")
        midi_file = self.path / "midi" / self.midi_files[idx]
        midi_data: pretty_midi.PrettyMIDI = self.load_midi_data(midi_file)
        logger.trace(f"Loaded {midi_file}, duration: {round(midi_data.get_end_time(), 2)}")

        pr_onset = torch.from_numpy(get_piano_roll_onset(midi_data, self.fs)).float()  # P=128, T
        logger.trace(f"pr_onset shape: {pr_onset.shape}")

        # sample a segment
        max_len = pr_onset.shape[1]
        if max_len > self.frame_width:
            start_idx = np.random.choice(
                range(0, max_len - self.frame_width + 1, self.fs)
            )  # +1 to include the last one
        else:
            start_idx = 0
        sample_onset = pr_onset[:, start_idx : start_idx + self.frame_width]
        sample_tokens, sample_ids = self.tokenizer.tokenize(sample_onset)

        logger.trace(f"sample_onset shape: {sample_onset.shape}")
        logger.trace(f"number of sample_tokens: {len(sample_tokens)}")

        sample_labels = sample_ids.clone()
        sample_labels[sample_labels == self.tokenizer.get_idx("pad")] = -100

        sample_onset = sample_onset[self.pitch_range[0] : self.pitch_range[1], :]  # clamp
        sample_onset = sample_onset.unsqueeze(0)  # 1 channel

        logger.trace(f"padded sample_onset shape: {sample_onset.shape}")
        logger.trace(f"number of padded sample_tokens: {len(sample_tokens)}")

        return {
            "midi_file": midi_file,
            "sample_onset": sample_onset,
            "sample_tokens": sample_tokens,
            "sample_ids": sample_ids,
            "sample_labels": sample_labels,
        }

    def load_midi_data(self, midi_path):
        if self.use_cache:
            fn = self._load_midi_data_with_cache
        else:
            fn = self._load_midi_data

        return fn(midi_path)

    def _load_midi_data(self, midi_path):
        midi_data = pretty_midi.PrettyMIDI(str(midi_path))
        assert len(midi_data.instruments) == 1, f"Expected 1 instrument, got {len(midi_data.instruments)}, {midi_path}"
        return midi_data

    _load_midi_data_with_cache = cache(_load_midi_data)

class LatentDiffFullSongPianorollDataset(Dataset):
    def __init__(
        self,
        dataset_path: Path,
        frames_per_beat: int = 8,
        beats_per_bar: int = 4,
        min_duration: int = 0,
        max_duration: int = 100000000000,
        input_file_format: Literal["pianoroll", "midi"] = "pianoroll",
        props: list[str] = [],
    ):
        self.ds = music_data_analysis.Dataset(dataset_path)
        self.frames_per_beat = frames_per_beat
        self.beats_per_bar = beats_per_bar
        self.input_file_format = input_file_format
        self.props = props
        self.songs: list[music_data_analysis.Song] = []
        for song in self.ds.songs():
            duration: int = song.read_json("duration") * self.frames_per_beat // 64  # the duration is in 1/64 beat
            if duration >= min_duration and duration <= max_duration:
                self.songs.append(song)

        print("FullSongPianorollDataset initialized with", len(self), "segments from", len(self.songs), "songs")

    def __len__(self):
        return len(self.songs)

    def load_latent(self, song: music_data_analysis.Song, prop_name: str):
        latent = song.read_pt(prop_name)
        return latent

    def load_pianoroll(self, song: music_data_analysis.Song):
        if self.input_file_format == "pianoroll":
            pr = song.read_pianoroll("pianoroll", frames_per_beat=self.frames_per_beat)
        elif self.input_file_format == "midi":
            midi = song.read_midi("synced_midi")
            pr = music_data_analysis.Pianoroll.from_midi(midi, frames_per_beat=self.frames_per_beat)
        else:
            raise ValueError(f"Unknown input file format: {self.input_file_format}")
        return pr

    def __getitem__(self, idx):
        song: music_data_analysis.Song = self.songs[idx]
        # song: music_data_analysis.Song = self.ds.songs()[6]  #! for debugging
        result = {}
        for prop_name in self.props:
            if prop_name == "pianoroll":
                prop = self.load_pianoroll(song)
            elif prop_name.startswith("latent"):
                prop = self.load_latent(song, prop_name)
                prop_name = "latent"
            elif prop_name == "duration":
                prop = torch.tensor(
                    song.read_json("duration") // 64 * self.frames_per_beat,
                    dtype=torch.long,
                )
            elif prop_name == "n_bars":
                prop = torch.tensor(
                    song.read_json("duration") // 64 // self.beats_per_bar,
                    dtype=torch.long,
                )
            else:
                prop = song.read_json(prop_name)
            result[prop_name] = prop

        return result


class BasicArithmeticOperationsDataset(IterableDataset):
    def __init__(self, max_num):
        self.max_num = max_num

    def __iter__(self):
        return self

    def __next__(self):
        a = np.random.randint(0, self.max_num)
        b = np.random.randint(0, self.max_num)
        sum = f"{a} + {b} = {a + b}"
        diff = f"{a} - {b} = {a - b}"
        prod = f"{a} * {b} = {a * b}"
        quot = f"{a} / {b} = {round(a / b, 2)}" if b != 0 else "{a} / {b} = N"
        return {
            "a": str(a),
            "b": str(b),
            "sum": sum,
            "diff": diff,
            "prod": prod,
            "quot": quot,
        }


def collate_midi_data(
    batch: list[dict],
    frame_width,
    tokenizer: Tokenizer,
    max_seq_len: int,
):
    output_onset = []
    output_tokens = []
    output_ids = []
    output_labels = []

    for data in batch:
        sample_onset = data["sample_onset"]
        sample_tokens = data["sample_tokens"]
        sample_ids = data["sample_ids"]
        sample_labels = data["sample_labels"]

        # pad if needed
        if sample_onset.shape[-1] < frame_width:
            torch.nn.functional.pad(
                sample_onset,
                (0, frame_width - sample_onset.shape[-1]),
                "constant",
                0,
            )

        if len(sample_tokens) > max_seq_len:
            sample_tokens = sample_tokens[:max_seq_len]
            sample_ids = sample_ids[:max_seq_len]
        else:
            sample_tokens += [Word("pad")] * (max_seq_len - len(sample_tokens))
            sample_ids = F.pad(
                sample_ids,
                (0, max_seq_len - len(sample_ids)),
                "constant",
                tokenizer.get_idx("pad"),
            )

        sample_labels = sample_ids.clone()
        sample_labels[sample_labels == tokenizer.get_idx("pad")] = -100

        logger.trace(f"padded sample_onset shape: {sample_onset.shape}")
        logger.trace(f"number of padded sample_tokens: {len(sample_tokens)}")

        output_onset.append(sample_onset)
        output_tokens.append(sample_tokens)
        output_ids.append(sample_ids)
        output_labels.append(sample_labels)

    output_onset = torch.stack(output_onset)
    output_ids = torch.stack(output_ids)
    output_labels = torch.stack(output_labels)

    logger.trace(f"output shape: {output_onset.shape}")

    return {
        "onset": output_onset,
        "tokens": output_tokens,
        "ids": output_ids,
        "labels": output_labels,
    }


def collate_midi_data_lm(batch: list[dict], max_seq_len, fs, frame_width, vq_dim, pitch_range=[0, 128]):
    """
    seq_len: length of the frame sequence
    fs: samples per second
    seg_len: width of a frame
    vq_dim: dimension of quantized frame
    """
    output_onset = []
    output_sos = []
    output_eos = []
    output_padded = []

    patch_per_frame = vq_dim[0] * vq_dim[1]
    num_frames = max_seq_len // patch_per_frame

    for data in batch:
        midi_file: Path = data["midi_file"]
        midi_data: pretty_midi.PrettyMIDI = data["midi_data"]
        logger.trace(f"Collating {midi_file}, duration: {round(midi_data.get_end_time(), 2)}")

        pr_onset = torch.from_numpy(get_piano_roll_onset(midi_data, fs)).float()  # P=128, T
        pr_onset = pr_onset[pitch_range[0] : pitch_range[1], :]
        logger.trace(f"pr_onset shape: {pr_onset.shape}")

        # sample a segment
        data_len = pr_onset.shape[1]
        sample_width = frame_width * num_frames
        with_sos = False  # song start
        with_eos = False  # song end
        if data_len > sample_width:
            start_idx = np.random.choice(range(0, data_len - sample_width + 1, fs))
            if start_idx == 0:
                with_sos = True
            if start_idx + sample_width == data_len:
                with_eos = True
        else:
            start_idx = 0
            with_sos = True
            with_eos = True
        sample_onset = pr_onset[:, start_idx : start_idx + sample_width]
        logger.trace(f"Sample frames: {math.ceil(sample_onset.shape[-1] / frame_width)}")
        num_sample_frames = math.ceil(sample_onset.shape[-1] / frame_width)

        # pad if needed
        padded = torch.zeros((vq_dim[0], vq_dim[1])).bool()
        padded = padded.unsqueeze(-1).expand(-1, -1, num_sample_frames)
        if sample_onset.shape[-1] < sample_width:
            padded = torch.cat(
                [
                    padded,
                    torch.ones(vq_dim[0], vq_dim[1], num_frames - num_sample_frames).bool(),
                ],
                dim=-1,
            )
            sample_onset = torch.nn.functional.pad(
                sample_onset,
                (0, sample_width - sample_onset.shape[-1]),
                "constant",
                0,
            )

        sample_onset = sample_onset.unsqueeze(0)  # 1 channel
        padded = padded.reshape(vq_dim[0] * vq_dim[1] * num_frames)

        logger.trace(f"sample_onset shape: {sample_onset.shape}")

        output_onset.append(sample_onset)
        output_sos.append(with_sos)
        output_eos.append(with_eos)
        output_padded.append(padded)

    output_onset = torch.stack(output_onset)
    output_sos = torch.tensor(output_sos, dtype=torch.bool)
    output_eos = torch.tensor(output_eos, dtype=torch.bool)
    output_padded = torch.stack(output_padded)
    logger.trace(f"output_onset shape: {output_onset.shape}")
    logger.trace(f"output_padded shape: {output_padded.shape}")

    return {
        "onset": output_onset,  # lengthes are not equal
        "sos": output_sos,
        "eos": output_eos,
        "padded": output_padded,
        "vq_dim": vq_dim,
        "num_frames": num_frames,
    }


def collate_arithmatic_data(batch: list[dict], max_seq_len):
    output = {
        "equation": [],
        "label_mask": [],
    }
    for data in batch:
        seq = np.random.choice([data["sum"], data["diff"], data["prod"], data["quot"]])
        # remove while space
        seq = seq.replace(" ", "")
        # split into characters
        seq = list(seq)
        # get position of "="
        eq_pos = seq.index("=")
        # create label mask (1 for tokens which are not calculated for loss)
        label_mask = [0] * len(seq)
        for i in range(0, eq_pos + 1):
            label_mask[i] = 1

        seq = ["<bos>"] + seq + ["<eos>"]
        label_mask = [1] + label_mask + [0]

        # pad or truncate
        if len(seq) < max_seq_len:
            seq += ["<pad>"] * (max_seq_len - len(seq))
            label_mask += [1] * (max_seq_len - len(label_mask))
        else:
            seq = seq[:max_seq_len]
            label_mask = label_mask[:max_seq_len]

        output["equation"].append(seq)
        output["label_mask"].append(torch.tensor(label_mask, dtype=torch.bool))

    output["label_mask"] = torch.stack(output["label_mask"])  # type: ignore

    return output
