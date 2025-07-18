import hashlib
from pathlib import Path
from typing import Any, Callable, Literal
import torch
from torch.utils.data import Dataset
import music_data_analysis


class SegmentIndexer:
    """
    Example:
    num_segments_list = [1, 2, 3]
    indexer = SegmentIndexer(num_segments_list)
    print(indexer[0]) # (0, 0)
    print(indexer[1]) # (1, 0)
    print(indexer[2]) # (1, 1)
    print(indexer[3]) # (2, 0)
    print(indexer[4]) # (2, 1)
    print(indexer[5]) # (2, 2)
    """

    def __init__(self, num_segments_list: list[int]):
        self.num_segments_list = torch.tensor(num_segments_list)
        self.length = int(self.num_segments_list.sum().item())
        self.cum_num_segments = torch.cumsum(self.num_segments_list, dim=0)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        song_idx = torch.searchsorted(self.cum_num_segments, idx + 1)
        segment_idx = (
            idx - self.cum_num_segments[song_idx] + self.num_segments_list[song_idx]
        )
        return int(song_idx.item()), int(segment_idx.item())


# use md5 to split the dataset into train and test


def is_train_sample(song_name: str, train_set_ratio: float):
    if song_name.startswith("debug_train"):
        return True
    if song_name.startswith("debug_test"):
        return False
    # Hash the string to a hex digest
    hash_digest = hashlib.md5(song_name.encode("utf-8")).hexdigest()
    # Convert the hex digest to an integer
    hash_int = int(hash_digest, 16)
    # Normalize it to a float in [0, 1)
    hash_float = hash_int / 16**32
    # Return True if it's in the train split
    return hash_float < train_set_ratio

channels_with_pop_music = """@0AdRiaNleE0
@aldy32
@AnCoongPiano
@Animenzzz
@AnimeProAnimeonPiano
@bellaandlucas
@BrokenFKey
@bunnypiano1246
@cateen_hayatosumino
@catrionesmusic5311
@CharlesSzczepanek
@CIPMusic
@DaranPianoTutorial
@DooPiano
@easypianoarrangements
@flowmusicpiano
@FonziMGM
@FrancescoParrino
@Fukane
@GabrielPiano1
@GerardChua
@GrimCatPiano
@HalcyonMusic
@hanppyeom
@JacobsPiano
@jichanpark
@JovaMusique
@JoyceLeong
@JRTranscription
@KatherineCordova
@Keyboard_Man
@Keyboard_Yoon
@KeyNomad503
@Lamipiano
@Lazypianist
@marasy8
@marvdamspiano
@Montechait
@mortengildbergmusic
@Nicepianosheets
@OORpiano%E9%8B%BC%E7%90%B4%E3%83%94%E3%82%A2%E3%83%8E
@PairPiano
@panpianoatelier
@PatrikPietschmann
@pianicast
@PianoDeuss
@PianoinU
@pianonline-kdramaostkpop5811
@pianotutorial7630
@Piano-X
@PineappleChord
@RiyandiKusuma
@RuRusPiano
@SangeoMusic
@SheetMusicBoss
@ShinGiwonPiano
@shingiwonpiano2
@SLSMusic
@solkeyspiano
@st_vanie
@suupiano
@TehIshter
@TheTheorist
@thisispiano
@TorbyBrand
@YourPianoCover
@zzzAnimeonPiano""".split("\n")


def get_channel_of_song(song_name):
    return song_name.split("/")[0]

class PianorollDataset(Dataset):
    def __init__(
        self,
        dataset_path: Path,
        frames_per_beat: int = 8,
        hop_length: int = 32,
        length: int = 32 * 8,
        min_start_overlap: int = 32,
        min_end_overlap: int = 32,
        input_file_format: Literal["pianoroll", "midi"] = "pianoroll",
        transform: Callable | None = None,
        transform_kwargs: dict | None = None,
        split: Literal["train", "test"] = "train",
        train_set_ratio: float = 1,
    ):
        self.ds = music_data_analysis.Dataset(dataset_path)
        self.frames_per_beat = frames_per_beat
        self.hop_length = hop_length
        self.length = length
        self.input_file_format = input_file_format
        self.start_pre_pad = length - min_start_overlap
        self.end_pre_pad = length - min_end_overlap + self.hop_length
        self.transform = transform
        self.transform_kwargs = transform_kwargs or {}
        self.songs = self.ds.songs()
        self.song_n_segments = []

        if split == "train":
            self.songs = [
                song
                for song in self.songs
                if is_train_sample(song.song_name, train_set_ratio)
                and get_channel_of_song(song.song_name) in channels_with_pop_music
            ]
        elif split == "test":
            self.songs = [
                song
                for song in self.songs
                if not is_train_sample(song.song_name, train_set_ratio)
                and get_channel_of_song(song.song_name) in channels_with_pop_music
            ]
        else:
            raise ValueError(f"Invalid split: {split}")

        for song in self.songs:
            duration: int = (
                song.read_json("duration") * self.frames_per_beat // 64
            )  # the duration is in 1/64 beat
            self.song_n_segments.append(
                (duration - self.length + self.start_pre_pad + self.end_pre_pad)
                // self.hop_length
            )

        self.indexer = SegmentIndexer(self.song_n_segments)

        print(
            "PianorollDataset initialized with",
            len(self),
            "segments from",
            len(self.songs),
            "songs, filtered from",
            len(self.ds),
            "songs, Split:",
            split,
        )

    def __len__(self):
        return len(self.indexer)

    def __getitem__(self, idx):
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} is out of bounds for the dataset of length {len(self)}"
            )
        song_idx, segment_idx = self.indexer[idx]
        song = self.songs[song_idx]
        segment_start = segment_idx * self.hop_length - self.start_pre_pad
        segment_end = segment_start + self.length
        segment = song.get_segment(
            segment_start, segment_end, frames_per_beat=self.frames_per_beat
        )
        if self.transform is not None:
            return self.transform(segment, **self.transform_kwargs)
        else:
            return segment


class FullSongPianorollDataset(Dataset):
    def __init__(
        self,
        dataset_path: Path,
        frames_per_beat: int = 8,
        min_duration: int = 0,
        max_duration: int = 100000000000,
        input_file_format: Literal["pianoroll", "midi"] = "pianoroll",
        transform: Callable[[music_data_analysis.Song], Any] | None = None,
        split: Literal["train", "test", "all"] = "all",
        train_set_ratio: float = 0.9,
    ):
        self.ds = music_data_analysis.Dataset(
            dataset_path, song_search_index="synced_midi"
        )
        self.frames_per_beat = frames_per_beat
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.input_file_format = input_file_format
        self.transform = transform
        self.songs = self.ds.songs()

        original_len = len(self.songs)

        self.songs = [
            song
            for song in self.songs
            if song.read_json("duration") * self.frames_per_beat // 64
            < self.max_duration
            and song.read_json("duration") * self.frames_per_beat // 64
            >= self.min_duration
        ]

        if split == "train":
            self.songs = [
                song
                for song in self.songs
                if is_train_sample(song.song_name, train_set_ratio)
                and get_channel_of_song(song.song_name) in channels_with_pop_music
            ]
        elif split == "test":
            self.songs = [
                song
                for song in self.songs
                if not is_train_sample(song.song_name, train_set_ratio)
                and get_channel_of_song(song.song_name) in channels_with_pop_music
            ]
        elif split == "all":
            pass
        else:
            raise ValueError(f"Invalid split: {split}")

        print(
            f"FullSongPianorollDataset initialized with {len(self.songs)} songs filtered from {original_len} songs. Split: {split}"
        )

    def __len__(self):
        return len(self.songs)

    def __getitem__(self, idx):
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} is out of bounds for the dataset of length {len(self)}"
            )
        song: music_data_analysis.Song = self.songs[idx]
        if self.transform is not None:
            return self.transform(song)
        else:
            return song
