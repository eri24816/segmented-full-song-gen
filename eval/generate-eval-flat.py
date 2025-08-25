import argparse
from pathlib import Path
import re
import random

from music_data_analysis import Pianoroll
import torch
import safetensors.torch
from segment_full_song import create_model
from segment_full_song.data.segment import get_compose_order
from segment_full_song.models.segment_full_song import SegmentFullSongModel
from segment_full_song.data.dataset import FullSongPianorollDataset
from segment_full_song.models.token_generator import TokenGenerator
from segment_full_song.models.token_sequence import TokenSequence



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


def generate(model:SegmentFullSongModel, labels: list[str], lengths_in_bars: list[int], compose_order: list[int], seed_segment_pr: Pianoroll, top_p: float = 1):
    generated_song, annotations = model.sample_song(
        labels=labels,
        lengths_in_bars=lengths_in_bars,
        compose_order=list(map(int, compose_order)),
        given_segments=[seed_segment_pr],
        top_p=top_p,
    )
    return generated_song, annotations



song_list = [
    {
        "idx": 0,
        "song_name": "@Montechait/xE0PxTufXAo/0_176",
        "segmentation": [
            {
                "start": 0,
                "end": 6,
                "label": 0
            },
            {
                "start": 6,
                "end": 20,
                "label": 1
            },
            {
                "start": 20,
                "end": 26,
                "label": 2
            },
            {
                "start": 26,
                "end": 30,
                "label": 3
            },
            {
                "start": 30,
                "end": 38,
                "label": 1
            },
            {
                "start": 38,
                "end": 42,
                "label": 4
            }
        ],
        "seed_segment": {
            "start": 30,
            "end": 38,
            "label": 1
        }
    },
    {
        "idx": 1,
        "song_name": "@AnimeProAnimeonPiano/HU_yW-SpwMs/0_163",
        "segmentation": [
            {
                "start": 0,
                "end": 12,
                "label": 0
            },
            {
                "start": 12,
                "end": 16,
                "label": 1
            },
            {
                "start": 16,
                "end": 24,
                "label": 2
            },
            {
                "start": 24,
                "end": 31,
                "label": 0
            },
            {
                "start": 31,
                "end": 36,
                "label": 1
            },
            {
                "start": 36,
                "end": 44,
                "label": 2
            },
            {
                "start": 44,
                "end": 48,
                "label": 0
            }
        ],
        "seed_segment": {
            "start": 16,
            "end": 24,
            "label": 2
        }
    },
    {
        "idx": 3,
        "song_name": "@DooPiano/KFJ3gNMq6do/2696_2886",
        "segmentation": [
            {
                "start": 0,
                "end": 5,
                "label": 0
            },
            {
                "start": 5,
                "end": 13,
                "label": 1
            },
            {
                "start": 13,
                "end": 17,
                "label": 2
            },
            {
                "start": 17,
                "end": 24,
                "label": 3
            },
            {
                "start": 24,
                "end": 29,
                "label": 1
            },
            {
                "start": 29,
                "end": 33,
                "label": 2
            },
            {
                "start": 33,
                "end": 36,
                "label": 0
            },
            {
                "start": 36,
                "end": 40,
                "label": 4
            },
            {
                "start": 40,
                "end": 41,
                "label": 1
            },
            {
                "start": 41,
                "end": 45,
                "label": 2
            },
            {
                "start": 45,
                "end": 50,
                "label": 3
            },
            {
                "start": 50,
                "end": 54,
                "label": 0
            }
        ],
        "seed_segment": {
            "start": 29,
            "end": 33,
            "label": 2
        }
    },
    {
        "idx": 6,
        "song_name": "@easypianoarrangements/Z3u2UB6ZDWE/0_180",
        "segmentation": [
            {
                "start": 0,
                "end": 5,
                "label": 0
            },
            {
                "start": 5,
                "end": 14,
                "label": 1
            },
            {
                "start": 14,
                "end": 18,
                "label": 2
            },
            {
                "start": 18,
                "end": 24,
                "label": 3
            },
            {
                "start": 24,
                "end": 32,
                "label": 1
            },
            {
                "start": 32,
                "end": 36,
                "label": 2
            },
            {
                "start": 36,
                "end": 41,
                "label": 4
            },
            {
                "start": 41,
                "end": 49,
                "label": 1
            },
            {
                "start": 49,
                "end": 54,
                "label": 2
            }
        ],
        "seed_segment": {
            "start": 24,
            "end": 32,
            "label": 1
        }
    },
    {
        "idx": 10,
        "song_name": "@AnCoongPiano/Th33EEqnFTM/0_180",
        "segmentation": [
            {
                "start": 0,
                "end": 4,
                "label": 0
            },
            {
                "start": 4,
                "end": 13,
                "label": 1
            },
            {
                "start": 13,
                "end": 21,
                "label": 2
            },
            {
                "start": 21,
                "end": 24,
                "label": 0
            },
            {
                "start": 24,
                "end": 31,
                "label": 3
            },
            {
                "start": 31,
                "end": 38,
                "label": 4
            },
            {
                "start": 38,
                "end": 44,
                "label": 0
            }
        ],
        "seed_segment": {
            "start": 21,
            "end": 24,
            "label": 0
        }
    },
    {
        "idx": 13,
        "song_name": "@ShinGiwonPiano/_6-qXHCXMCQ/3003_3188",
        "segmentation": [
            {
                "start": 0,
                "end": 5,
                "label": 0
            },
            {
                "start": 5,
                "end": 8,
                "label": 1
            },
            {
                "start": 8,
                "end": 17,
                "label": 2
            },
            {
                "start": 17,
                "end": 25,
                "label": 1
            },
            {
                "start": 25,
                "end": 33,
                "label": 2
            },
            {
                "start": 33,
                "end": 39,
                "label": 3
            },
            {
                "start": 39,
                "end": 46,
                "label": 4
            }
        ],
        "seed_segment": {
            "start": 25,
            "end": 33,
            "label": 2
        }
    },
    {
        "idx": 16,
        "song_name": "@suupiano/WZv0Hh1R-tg/0_106",
        "segmentation": [
            {
                "start": 0,
                "end": 8,
                "label": 0
            },
            {
                "start": 8,
                "end": 17,
                "label": 1
            },
            {
                "start": 17,
                "end": 25,
                "label": 0
            },
            {
                "start": 25,
                "end": 32,
                "label": 1
            },
            {
                "start": 32,
                "end": 41,
                "label": 2
            },
            {
                "start": 41,
                "end": 47,
                "label": 0
            },
            {
                "start": 47,
                "end": 50,
                "label": 3
            }
        ],
        "seed_segment": {
            "start": 17,
            "end": 25,
            "label": 0
        }
    },
    {
        "idx": 17,
        "song_name": "@zzzAnimeonPiano/To6ne2f_IC8/0_140",
        "segmentation": [
            {
                "start": 0,
                "end": 3,
                "label": 0
            },
            {
                "start": 3,
                "end": 8,
                "label": 1
            },
            {
                "start": 8,
                "end": 24,
                "label": 2
            },
            {
                "start": 24,
                "end": 40,
                "label": 3
            },
            {
                "start": 40,
                "end": 48,
                "label": 2
            },
            {
                "start": 48,
                "end": 55,
                "label": 3
            },
            {
                "start": 55,
                "end": 60,
                "label": 2
            }
        ],
        "seed_segment": {
            "start": 40,
            "end": 48,
            "label": 2
        }
    },
    {
        "idx": 18,
        "song_name": "@ShinGiwonPiano/IF1AWxnFQsA/0_204",
        "segmentation": [
            {
                "start": 0,
                "end": 4,
                "label": 0
            },
            {
                "start": 4,
                "end": 11,
                "label": 1
            },
            {
                "start": 11,
                "end": 16,
                "label": 2
            },
            {
                "start": 16,
                "end": 25,
                "label": 3
            },
            {
                "start": 25,
                "end": 33,
                "label": 1
            },
            {
                "start": 33,
                "end": 37,
                "label": 2
            },
            {
                "start": 37,
                "end": 42,
                "label": 4
            }
        ],
        "seed_segment": {
            "start": 25,
            "end": 33,
            "label": 1
        }
    },
    {
        "idx": 19,
        "song_name": "@PairPiano/Th62O5KR5KQ/0_205",
        "segmentation": [
            {
                "start": 0,
                "end": 7,
                "label": 0
            },
            {
                "start": 7,
                "end": 14,
                "label": 1
            },
            {
                "start": 14,
                "end": 20,
                "label": 2
            },
            {
                "start": 20,
                "end": 27,
                "label": 0
            },
            {
                "start": 27,
                "end": 33,
                "label": 1
            },
            {
                "start": 33,
                "end": 41,
                "label": 2
            },
            {
                "start": 41,
                "end": 48,
                "label": 3
            },
            {
                "start": 48,
                "end": 55,
                "label": 4
            },
            {
                "start": 55,
                "end": 59,
                "label": 5
            }
        ],
        "seed_segment": {
            "start": 33,
            "end": 41,
            "label": 2
        }
    },
    {
        "idx": 21,
        "song_name": "@BrokenFKey/AbR1y2o67GE/0_111",
        "segmentation": [
            {
                "start": 0,
                "end": 15,
                "label": 0
            },
            {
                "start": 15,
                "end": 31,
                "label": 1
            },
            {
                "start": 31,
                "end": 37,
                "label": 2
            },
            {
                "start": 37,
                "end": 42,
                "label": 3
            },
            {
                "start": 42,
                "end": 48,
                "label": 4
            },
            {
                "start": 48,
                "end": 52,
                "label": 0
            },
            {
                "start": 52,
                "end": 56,
                "label": 5
            }
        ],
        "seed_segment": {
            "start": 0,
            "end": 15,
            "label": 0
        }
    },
    {
        "idx": 23,
        "song_name": "@DooPiano/nTY_2MjQ2CI/1806_1991",
        "segmentation": [
            {
                "start": 0,
                "end": 5,
                "label": 0
            },
            {
                "start": 5,
                "end": 13,
                "label": 1
            },
            {
                "start": 13,
                "end": 22,
                "label": 2
            },
            {
                "start": 22,
                "end": 30,
                "label": 3
            },
            {
                "start": 30,
                "end": 38,
                "label": 4
            },
            {
                "start": 38,
                "end": 46,
                "label": 5
            },
            {
                "start": 46,
                "end": 54,
                "label": 4
            }
        ],
        "seed_segment": {
            "start": 30,
            "end": 38,
            "label": 4
        }
    },
    {
        "idx": 24,
        "song_name": "@hanppyeom/5r6gKSGtUU0/0_194",
        "segmentation": [
            {
                "start": 0,
                "end": 4,
                "label": 0
            },
            {
                "start": 4,
                "end": 8,
                "label": 1
            },
            {
                "start": 8,
                "end": 15,
                "label": 2
            },
            {
                "start": 15,
                "end": 24,
                "label": 3
            },
            {
                "start": 24,
                "end": 32,
                "label": 1
            },
            {
                "start": 32,
                "end": 35,
                "label": 2
            },
            {
                "start": 35,
                "end": 40,
                "label": 3
            },
            {
                "start": 40,
                "end": 48,
                "label": 1
            },
            {
                "start": 48,
                "end": 51,
                "label": 4
            }
        ],
        "seed_segment": {
            "start": 24,
            "end": 32,
            "label": 1
        }
    },
    {
        "idx": 29,
        "song_name": "@easypianoarrangements/DuYNHAx6eRY/0_222",
        "segmentation": [
            {
                "start": 0,
                "end": 8,
                "label": 0
            },
            {
                "start": 8,
                "end": 14,
                "label": 1
            },
            {
                "start": 14,
                "end": 21,
                "label": 2
            },
            {
                "start": 21,
                "end": 25,
                "label": 3
            },
            {
                "start": 25,
                "end": 33,
                "label": 0
            },
            {
                "start": 33,
                "end": 39,
                "label": 1
            },
            {
                "start": 39,
                "end": 46,
                "label": 2
            },
            {
                "start": 46,
                "end": 50,
                "label": 3
            },
            {
                "start": 50,
                "end": 54,
                "label": 0
            }
        ],
        "seed_segment": {
            "start": 25,
            "end": 33,
            "label": 0
        }
    },
    {
        "idx": 30,
        "song_name": "@TheTheorist/Y2gU41nTyeY/0_235",
        "segmentation": [
            {
                "start": 0,
                "end": 3,
                "label": 0
            },
            {
                "start": 3,
                "end": 11,
                "label": 1
            },
            {
                "start": 11,
                "end": 16,
                "label": 2
            },
            {
                "start": 16,
                "end": 24,
                "label": 3
            },
            {
                "start": 24,
                "end": 32,
                "label": 1
            },
            {
                "start": 32,
                "end": 37,
                "label": 2
            },
            {
                "start": 37,
                "end": 45,
                "label": 3
            },
            {
                "start": 45,
                "end": 50,
                "label": 4
            },
            {
                "start": 50,
                "end": 58,
                "label": 3
            },
            {
                "start": 58,
                "end": 60,
                "label": 5
            }
        ],
        "seed_segment": {
            "start": 37,
            "end": 45,
            "label": 3
        }
    },
    {
        "idx": 33,
        "song_name": "@TheTheorist/FRwP9UwanVo/0_193",
        "segmentation": [
            {
                "start": 0,
                "end": 4,
                "label": 0
            },
            {
                "start": 4,
                "end": 12,
                "label": 1
            },
            {
                "start": 12,
                "end": 20,
                "label": 2
            },
            {
                "start": 20,
                "end": 24,
                "label": 3
            },
            {
                "start": 24,
                "end": 32,
                "label": 1
            },
            {
                "start": 32,
                "end": 40,
                "label": 2
            },
            {
                "start": 40,
                "end": 46,
                "label": 3
            },
            {
                "start": 46,
                "end": 49,
                "label": 4
            }
        ],
        "seed_segment": {
            "start": 32,
            "end": 40,
            "label": 2
        }
    },
    {
        "idx": 34,
        "song_name": "@ShinGiwonPiano/MA58itJ3Zic/12721_12950",
        "segmentation": [
            {
                "start": 0,
                "end": 5,
                "label": 0
            },
            {
                "start": 5,
                "end": 12,
                "label": 1
            },
            {
                "start": 12,
                "end": 20,
                "label": 2
            },
            {
                "start": 20,
                "end": 26,
                "label": 3
            },
            {
                "start": 26,
                "end": 33,
                "label": 1
            },
            {
                "start": 33,
                "end": 41,
                "label": 2
            },
            {
                "start": 41,
                "end": 45,
                "label": 4
            },
            {
                "start": 45,
                "end": 54,
                "label": 2
            },
            {
                "start": 54,
                "end": 58,
                "label": 5
            }
        ],
        "seed_segment": {
            "start": 33,
            "end": 41,
            "label": 2
        }
    },
    {
        "idx": 35,
        "song_name": "@FonziMGM/FrDbz00Wa-4/0_90",
        "segmentation": [
            {
                "start": 0,
                "end": 12,
                "label": 0
            },
            {
                "start": 12,
                "end": 22,
                "label": 1
            },
            {
                "start": 22,
                "end": 37,
                "label": 2
            },
            {
                "start": 37,
                "end": 41,
                "label": 0
            }
        ],
        "seed_segment": {
            "start": 37,
            "end": 41,
            "label": 0
        }
    },
    {
        "idx": 36,
        "song_name": "@aldy32/NbM1Z_6rRjQ/0_215",
        "segmentation": [
            {
                "start": 0,
                "end": 7,
                "label": 0
            },
            {
                "start": 7,
                "end": 14,
                "label": 1
            },
            {
                "start": 14,
                "end": 23,
                "label": 0
            },
            {
                "start": 23,
                "end": 30,
                "label": 1
            },
            {
                "start": 30,
                "end": 39,
                "label": 0
            },
            {
                "start": 39,
                "end": 45,
                "label": 2
            },
            {
                "start": 45,
                "end": 55,
                "label": 0
            },
            {
                "start": 55,
                "end": 60,
                "label": 3
            }
        ],
        "seed_segment": {
            "start": 30,
            "end": 39,
            "label": 0
        }
    },
    {
        "idx": 37,
        "song_name": "@thisispiano/FjO3THMI59w/0_160",
        "segmentation": [
            {
                "start": 0,
                "end": 6,
                "label": 0
            },
            {
                "start": 6,
                "end": 13,
                "label": 1
            },
            {
                "start": 13,
                "end": 20,
                "label": 2
            },
            {
                "start": 20,
                "end": 28,
                "label": 1
            },
            {
                "start": 28,
                "end": 38,
                "label": 3
            },
            {
                "start": 38,
                "end": 47,
                "label": 1
            }
        ],
        "seed_segment": {
            "start": 20,
            "end": 28,
            "label": 1
        }
    },
    {
        "idx": 38,
        "song_name": "@AnimeProAnimeonPiano/luhQWdB6ikE/0_179",
        "segmentation": [
            {
                "start": 0,
                "end": 5,
                "label": 0
            },
            {
                "start": 5,
                "end": 12,
                "label": 1
            },
            {
                "start": 12,
                "end": 19,
                "label": 2
            },
            {
                "start": 19,
                "end": 28,
                "label": 3
            },
            {
                "start": 28,
                "end": 34,
                "label": 1
            },
            {
                "start": 34,
                "end": 41,
                "label": 2
            },
            {
                "start": 41,
                "end": 49,
                "label": 3
            },
            {
                "start": 49,
                "end": 54,
                "label": 4
            }
        ],
        "seed_segment": {
            "start": 19,
            "end": 28,
            "label": 3
        }
    },
    {
        "idx": 39,
        "song_name": "@FonziMGM/NxGVE9mGwlo/0_185",
        "segmentation": [
            {
                "start": 0,
                "end": 28,
                "label": 0
            },
            {
                "start": 28,
                "end": 42,
                "label": 1
            },
            {
                "start": 42,
                "end": 53,
                "label": 2
            },
            {
                "start": 53,
                "end": 60,
                "label": 0
            }
        ],
        "seed_segment": {
            "start": 53,
            "end": 60,
            "label": 0
        }
    },
    {
        "idx": 40,
        "song_name": "@JovaMusique/DU57LNpvVeA/0_123",
        "segmentation": [
            {
                "start": 0,
                "end": 2,
                "label": 0
            },
            {
                "start": 2,
                "end": 10,
                "label": 1
            },
            {
                "start": 10,
                "end": 17,
                "label": 2
            },
            {
                "start": 17,
                "end": 25,
                "label": 1
            },
            {
                "start": 25,
                "end": 30,
                "label": 3
            },
            {
                "start": 30,
                "end": 37,
                "label": 2
            },
            {
                "start": 37,
                "end": 46,
                "label": 1
            },
            {
                "start": 46,
                "end": 50,
                "label": 4
            }
        ],
        "seed_segment": {
            "start": 17,
            "end": 25,
            "label": 1
        }
    },
    {
        "idx": 41,
        "song_name": "@easypianoarrangements/8So7bur_a18/0_213",
        "segmentation": [
            {
                "start": 0,
                "end": 3,
                "label": 0
            },
            {
                "start": 3,
                "end": 7,
                "label": 1
            },
            {
                "start": 7,
                "end": 14,
                "label": 2
            },
            {
                "start": 14,
                "end": 23,
                "label": 3
            },
            {
                "start": 23,
                "end": 27,
                "label": 1
            },
            {
                "start": 27,
                "end": 34,
                "label": 2
            },
            {
                "start": 34,
                "end": 43,
                "label": 3
            },
            {
                "start": 43,
                "end": 50,
                "label": 4
            },
            {
                "start": 50,
                "end": 55,
                "label": 3
            },
            {
                "start": 55,
                "end": 59,
                "label": 5
            }
        ],
        "seed_segment": {
            "start": 34,
            "end": 43,
            "label": 3
        }
    },
    {
        "idx": 42,
        "song_name": "@Nicepianosheets/4Zd4hmHZUWU/8_189",
        "segmentation": [
            {
                "start": 0,
                "end": 5,
                "label": 0
            },
            {
                "start": 5,
                "end": 15,
                "label": 1
            },
            {
                "start": 15,
                "end": 24,
                "label": 2
            },
            {
                "start": 24,
                "end": 32,
                "label": 0
            },
            {
                "start": 32,
                "end": 40,
                "label": 1
            },
            {
                "start": 40,
                "end": 52,
                "label": 2
            }
        ],
        "seed_segment": {
            "start": 15,
            "end": 24,
            "label": 2
        }
    },
    {
        "idx": 43,
        "song_name": "@Lamipiano/Xb5n9GC9jAA/0_123",
        "segmentation": [
            {
                "start": 0,
                "end": 6,
                "label": 0
            },
            {
                "start": 6,
                "end": 10,
                "label": 1
            },
            {
                "start": 10,
                "end": 17,
                "label": 2
            },
            {
                "start": 17,
                "end": 22,
                "label": 0
            },
            {
                "start": 22,
                "end": 30,
                "label": 1
            },
            {
                "start": 30,
                "end": 39,
                "label": 3
            },
            {
                "start": 39,
                "end": 41,
                "label": 1
            },
            {
                "start": 41,
                "end": 47,
                "label": 0
            },
            {
                "start": 47,
                "end": 56,
                "label": 1
            }
        ],
        "seed_segment": {
            "start": 22,
            "end": 30,
            "label": 1
        }
    },
    {
        "idx": 44,
        "song_name": "@JovaMusique/XeGXhC6OHXg/0_205",
        "segmentation": [
            {
                "start": 0,
                "end": 7,
                "label": 0
            },
            {
                "start": 7,
                "end": 16,
                "label": 1
            },
            {
                "start": 16,
                "end": 27,
                "label": 2
            },
            {
                "start": 27,
                "end": 34,
                "label": 0
            },
            {
                "start": 34,
                "end": 44,
                "label": 3
            },
            {
                "start": 44,
                "end": 50,
                "label": 0
            },
            {
                "start": 50,
                "end": 60,
                "label": 4
            }
        ],
        "seed_segment": {
            "start": 27,
            "end": 34,
            "label": 0
        }
    },
    {
        "idx": 45,
        "song_name": "@SheetMusicBoss/s_DjlbgWsI4/3607_3791",
        "segmentation": [
            {
                "start": 0,
                "end": 27,
                "label": 0
            },
            {
                "start": 27,
                "end": 37,
                "label": 1
            },
            {
                "start": 37,
                "end": 43,
                "label": 0
            }
        ],
        "seed_segment": {
            "start": 37,
            "end": 43,
            "label": 0
        }
    },
    {
        "idx": 46,
        "song_name": "@GrimCatPiano/rXSDlFHiWOA/0_239",
        "segmentation": [
            {
                "start": 0,
                "end": 8,
                "label": 0
            },
            {
                "start": 8,
                "end": 12,
                "label": 1
            },
            {
                "start": 12,
                "end": 16,
                "label": 2
            },
            {
                "start": 16,
                "end": 24,
                "label": 3
            },
            {
                "start": 24,
                "end": 28,
                "label": 1
            },
            {
                "start": 28,
                "end": 36,
                "label": 2
            },
            {
                "start": 36,
                "end": 45,
                "label": 4
            },
            {
                "start": 45,
                "end": 54,
                "label": 2
            }
        ],
        "seed_segment": {
            "start": 28,
            "end": 36,
            "label": 2
        }
    },
    {
        "idx": 47,
        "song_name": "@Montechait/ZB_zbWXUkW8/0_265",
        "segmentation": [
            {
                "start": 0,
                "end": 9,
                "label": 0
            },
            {
                "start": 9,
                "end": 16,
                "label": 1
            },
            {
                "start": 16,
                "end": 18,
                "label": 2
            },
            {
                "start": 18,
                "end": 21,
                "label": 0
            },
            {
                "start": 21,
                "end": 27,
                "label": 3
            },
            {
                "start": 27,
                "end": 33,
                "label": 1
            },
            {
                "start": 33,
                "end": 41,
                "label": 4
            },
            {
                "start": 41,
                "end": 49,
                "label": 1
            },
            {
                "start": 49,
                "end": 54,
                "label": 2
            }
        ],
        "seed_segment": {
            "start": 27,
            "end": 33,
            "label": 1
        }
    },
    {
        "idx": 48,
        "song_name": "@RuRusPiano/NFGjSywNW9s/0_247",
        "segmentation": [
            {
                "start": 0,
                "end": 15,
                "label": 0
            },
            {
                "start": 15,
                "end": 22,
                "label": 1
            },
            {
                "start": 22,
                "end": 30,
                "label": 0
            },
            {
                "start": 30,
                "end": 46,
                "label": 1
            },
            {
                "start": 46,
                "end": 51,
                "label": 0
            }
        ],
        "seed_segment": {
            "start": 22,
            "end": 30,
            "label": 0
        }
    },
    {
        "idx": 49,
        "song_name": "@aldy32/5HdO1BVqgN8/0_212",
        "segmentation": [
            {
                "start": 0,
                "end": 5,
                "label": 0
            },
            {
                "start": 5,
                "end": 13,
                "label": 1
            },
            {
                "start": 13,
                "end": 20,
                "label": 2
            },
            {
                "start": 20,
                "end": 29,
                "label": 1
            },
            {
                "start": 29,
                "end": 36,
                "label": 2
            },
            {
                "start": 36,
                "end": 41,
                "label": 1
            },
            {
                "start": 41,
                "end": 49,
                "label": 2
            },
            {
                "start": 49,
                "end": 56,
                "label": 3
            },
            {
                "start": 56,
                "end": 59,
                "label": 4
            }
        ],
        "seed_segment": {
            "start": 29,
            "end": 36,
            "label": 2
        }
    },
    {
        "idx": 50,
        "song_name": "@pianotutorial7630/yunPkYlJA28/0_174",
        "segmentation": [
            {
                "start": 0,
                "end": 4,
                "label": 0
            },
            {
                "start": 4,
                "end": 12,
                "label": 1
            },
            {
                "start": 12,
                "end": 18,
                "label": 0
            },
            {
                "start": 18,
                "end": 26,
                "label": 1
            },
            {
                "start": 26,
                "end": 30,
                "label": 0
            },
            {
                "start": 30,
                "end": 33,
                "label": 1
            },
            {
                "start": 33,
                "end": 44,
                "label": 0
            }
        ],
        "seed_segment": {
            "start": 26,
            "end": 30,
            "label": 0
        }
    },
    {
        "idx": 51,
        "song_name": "@JovaMusique/_rjm0eJrhTw/0_198",
        "segmentation": [
            {
                "start": 0,
                "end": 9,
                "label": 0
            },
            {
                "start": 9,
                "end": 17,
                "label": 1
            },
            {
                "start": 17,
                "end": 24,
                "label": 2
            },
            {
                "start": 24,
                "end": 32,
                "label": 1
            },
            {
                "start": 32,
                "end": 37,
                "label": 3
            },
            {
                "start": 37,
                "end": 45,
                "label": 4
            },
            {
                "start": 45,
                "end": 52,
                "label": 1
            },
            {
                "start": 52,
                "end": 58,
                "label": 3
            }
        ],
        "seed_segment": {
            "start": 24,
            "end": 32,
            "label": 1
        }
    },
    {
        "idx": 52,
        "song_name": "@panpianoatelier/NTMakG1uQ7E/0_101",
        "segmentation": [
            {
                "start": 0,
                "end": 5,
                "label": 0
            },
            {
                "start": 5,
                "end": 13,
                "label": 1
            },
            {
                "start": 13,
                "end": 21,
                "label": 2
            },
            {
                "start": 21,
                "end": 25,
                "label": 0
            },
            {
                "start": 25,
                "end": 33,
                "label": 3
            },
            {
                "start": 33,
                "end": 42,
                "label": 4
            },
            {
                "start": 42,
                "end": 49,
                "label": 5
            }
        ],
        "seed_segment": {
            "start": 33,
            "end": 42,
            "label": 4
        }
    },
    {
        "idx": 53,
        "song_name": "@solkeyspiano/RVlRoJ64GNQ/4_212",
        "segmentation": [
            {
                "start": 0,
                "end": 5,
                "label": 0
            },
            {
                "start": 5,
                "end": 12,
                "label": 1
            },
            {
                "start": 12,
                "end": 20,
                "label": 2
            },
            {
                "start": 20,
                "end": 29,
                "label": 1
            },
            {
                "start": 29,
                "end": 32,
                "label": 2
            },
            {
                "start": 32,
                "end": 41,
                "label": 3
            },
            {
                "start": 41,
                "end": 50,
                "label": 2
            }
        ],
        "seed_segment": {
            "start": 29,
            "end": 32,
            "label": 2
        }
    },
    {
        "idx": 54,
        "song_name": "@JovaMusique/KE63YngNJLU/0_193",
        "segmentation": [
            {
                "start": 0,
                "end": 8,
                "label": 0
            },
            {
                "start": 8,
                "end": 15,
                "label": 1
            },
            {
                "start": 15,
                "end": 20,
                "label": 2
            },
            {
                "start": 20,
                "end": 24,
                "label": 3
            },
            {
                "start": 24,
                "end": 32,
                "label": 4
            },
            {
                "start": 32,
                "end": 36,
                "label": 2
            },
            {
                "start": 36,
                "end": 40,
                "label": 3
            },
            {
                "start": 40,
                "end": 48,
                "label": 4
            },
            {
                "start": 48,
                "end": 52,
                "label": 5
            }
        ],
        "seed_segment": {
            "start": 24,
            "end": 32,
            "label": 4
        }
    },
    {
        "idx": 55,
        "song_name": "@AnimeProAnimeonPiano/LDaBLamwKJE/0_176",
        "segmentation": [
            {
                "start": 0,
                "end": 6,
                "label": 0
            },
            {
                "start": 6,
                "end": 12,
                "label": 1
            },
            {
                "start": 12,
                "end": 22,
                "label": 2
            },
            {
                "start": 22,
                "end": 30,
                "label": 3
            },
            {
                "start": 30,
                "end": 36,
                "label": 1
            },
            {
                "start": 36,
                "end": 45,
                "label": 4
            },
            {
                "start": 45,
                "end": 51,
                "label": 5
            }
        ],
        "seed_segment": {
            "start": 30,
            "end": 36,
            "label": 1
        }
    },
    {
        "idx": 56,
        "song_name": "@Montechait/135-z4by0eI/0_159",
        "segmentation": [
            {
                "start": 0,
                "end": 6,
                "label": 0
            },
            {
                "start": 6,
                "end": 12,
                "label": 1
            },
            {
                "start": 12,
                "end": 20,
                "label": 2
            },
            {
                "start": 20,
                "end": 27,
                "label": 3
            },
            {
                "start": 27,
                "end": 32,
                "label": 1
            },
            {
                "start": 32,
                "end": 40,
                "label": 2
            },
            {
                "start": 40,
                "end": 49,
                "label": 4
            },
            {
                "start": 49,
                "end": 57,
                "label": 2
            }
        ],
        "seed_segment": {
            "start": 32,
            "end": 40,
            "label": 2
        }
    },
    {
        "idx": 57,
        "song_name": "@easypianoarrangements/tOTpaVlt3os/0_188",
        "segmentation": [
            {
                "start": 0,
                "end": 1,
                "label": 0
            },
            {
                "start": 1,
                "end": 5,
                "label": 1
            },
            {
                "start": 5,
                "end": 13,
                "label": 2
            },
            {
                "start": 13,
                "end": 16,
                "label": 1
            },
            {
                "start": 16,
                "end": 22,
                "label": 2
            },
            {
                "start": 22,
                "end": 25,
                "label": 3
            },
            {
                "start": 25,
                "end": 27,
                "label": 1
            },
            {
                "start": 27,
                "end": 36,
                "label": 2
            },
            {
                "start": 36,
                "end": 39,
                "label": 1
            },
            {
                "start": 39,
                "end": 45,
                "label": 2
            }
        ],
        "seed_segment": {
            "start": 27,
            "end": 36,
            "label": 2
        }
    },
    {
        "idx": 58,
        "song_name": "@jichanpark/M0iD1gVvxlI/0_126",
        "segmentation": [
            {
                "start": 0,
                "end": 10,
                "label": 0
            },
            {
                "start": 10,
                "end": 14,
                "label": 1
            },
            {
                "start": 14,
                "end": 19,
                "label": 2
            },
            {
                "start": 19,
                "end": 23,
                "label": 3
            },
            {
                "start": 23,
                "end": 32,
                "label": 0
            },
            {
                "start": 32,
                "end": 36,
                "label": 1
            },
            {
                "start": 36,
                "end": 41,
                "label": 2
            },
            {
                "start": 41,
                "end": 45,
                "label": 3
            }
        ],
        "seed_segment": {
            "start": 23,
            "end": 32,
            "label": 0
        }
    },
    {
        "idx": 59,
        "song_name": "@JacobsPiano/6r_rbXzh18E/0_211",
        "segmentation": [
            {
                "start": 0,
                "end": 10,
                "label": 0
            },
            {
                "start": 10,
                "end": 17,
                "label": 1
            },
            {
                "start": 17,
                "end": 29,
                "label": 2
            },
            {
                "start": 29,
                "end": 36,
                "label": 0
            },
            {
                "start": 36,
                "end": 44,
                "label": 1
            },
            {
                "start": 44,
                "end": 52,
                "label": 0
            }
        ],
        "seed_segment": {
            "start": 29,
            "end": 36,
            "label": 0
        }
    },
    {
        "idx": 60,
        "song_name": "@KeyNomad503/gYnwb8N0Isw/0_184",
        "segmentation": [
            {
                "start": 0,
                "end": 5,
                "label": 0
            },
            {
                "start": 5,
                "end": 11,
                "label": 1
            },
            {
                "start": 11,
                "end": 14,
                "label": 2
            },
            {
                "start": 14,
                "end": 19,
                "label": 3
            },
            {
                "start": 19,
                "end": 24,
                "label": 0
            },
            {
                "start": 24,
                "end": 29,
                "label": 1
            },
            {
                "start": 29,
                "end": 35,
                "label": 4
            },
            {
                "start": 35,
                "end": 42,
                "label": 2
            },
            {
                "start": 42,
                "end": 52,
                "label": 3
            },
            {
                "start": 52,
                "end": 56,
                "label": 5
            }
        ],
        "seed_segment": {
            "start": 14,
            "end": 19,
            "label": 3
        }
    },
    {
        "idx": 61,
        "song_name": "@JovaMusique/e2fBcZOX6VI/0_213",
        "segmentation": [
            {
                "start": 0,
                "end": 5,
                "label": 0
            },
            {
                "start": 5,
                "end": 12,
                "label": 1
            },
            {
                "start": 12,
                "end": 21,
                "label": 2
            },
            {
                "start": 21,
                "end": 29,
                "label": 3
            },
            {
                "start": 29,
                "end": 36,
                "label": 1
            },
            {
                "start": 36,
                "end": 41,
                "label": 2
            },
            {
                "start": 41,
                "end": 49,
                "label": 4
            },
            {
                "start": 49,
                "end": 53,
                "label": 5
            }
        ],
        "seed_segment": {
            "start": 29,
            "end": 36,
            "label": 1
        }
    },
    {
        "idx": 62,
        "song_name": "@PianoinU/Fn8LwvXQOkE/0_163",
        "segmentation": [
            {
                "start": 0,
                "end": 5,
                "label": 0
            },
            {
                "start": 5,
                "end": 13,
                "label": 1
            },
            {
                "start": 13,
                "end": 16,
                "label": 2
            },
            {
                "start": 16,
                "end": 25,
                "label": 3
            },
            {
                "start": 25,
                "end": 31,
                "label": 4
            },
            {
                "start": 31,
                "end": 37,
                "label": 2
            },
            {
                "start": 37,
                "end": 46,
                "label": 3
            },
            {
                "start": 46,
                "end": 54,
                "label": 5
            },
            {
                "start": 54,
                "end": 56,
                "label": 3
            }
        ],
        "seed_segment": {
            "start": 37,
            "end": 46,
            "label": 3
        }
    },
    {
        "idx": 63,
        "song_name": "@Montechait/Ix2J64gmyT0/0_186",
        "segmentation": [
            {
                "start": 0,
                "end": 18,
                "label": 0
            },
            {
                "start": 18,
                "end": 26,
                "label": 1
            },
            {
                "start": 26,
                "end": 34,
                "label": 2
            },
            {
                "start": 34,
                "end": 42,
                "label": 0
            },
            {
                "start": 42,
                "end": 51,
                "label": 1
            },
            {
                "start": 51,
                "end": 56,
                "label": 3
            }
        ],
        "seed_segment": {
            "start": 34,
            "end": 42,
            "label": 0
        }
    }
]

def main(args):
    random.seed(325)

    model = create_model(args.model_config)
    assert isinstance(model, TokenGenerator)
    model.load_state_dict(safetensors.torch.load_file(args.ckpt))
    if torch.cuda.is_available():
        model.to("cuda")
    model.eval()

    ds = FullSongPianorollDataset(args.dataset_path, split='test')
    from tqdm import tqdm
    for i, song_info in tqdm(enumerate(song_list)):

        # convert to frames
        song_info['segmentation'] = [
            {
                "start": segment['start'] * 32,
                "end": segment['end'] * 32,
                "label": "ABCDEF"[segment['label']]
            } for segment in song_info['segmentation']
        ]

        song_info['seed_segment'] = {
            "start": song_info['seed_segment']['start'] * 32,
            "end": song_info['seed_segment']['end'] * 32,
            "label": "ABCDEF"[song_info['seed_segment']['label']]
        }

        song = ds.ds.get_song(song_info['song_name'])
        segments = song_info['segmentation']
        seed_segment = song_info['seed_segment']
        seed_segment_pr = Pianoroll.from_midi(song.read_midi('synced_midi')).slice(seed_segment['start'], seed_segment['end'], allow_out_of_range=True)
        assert seed_segment_pr.duration == seed_segment['end'] - seed_segment['start']

        labels = [segment['label'] for segment in segments]
        lengths_in_bars = [(segment['end'] - segment['start'])//32 for segment in segments]


        seed_segment_idx = None
        for j, segment in enumerate(segments):
            if segment['start'] == seed_segment['start'] and segment['end'] == seed_segment['end']:
                seed_segment_idx = j
        assert seed_segment_idx is not None
        compose_order = [seed_segment_idx]
        remaining_segment_idxs = [j for j in range(len(segments)) if j != seed_segment_idx]
        random.shuffle(remaining_segment_idxs)
        compose_order.extend(remaining_segment_idxs)


        print(labels, lengths_in_bars, compose_order)

        print(f"Generating {song_info['song_name']}")

        generated_song = model.sample(duration=256, max_length=950)

        duration = max([segment['end'] for segment in segments])
        while generated_song.duration < duration:
            current_duration = generated_song.duration
            generated = model.sample(duration=256, prompt=generated_song.slice_pos(current_duration-32*7, current_duration), max_length=950)
            generated_song = generated_song + generated.slice_pos(32*7, 32*8)

        generated_song = generated_song.to_pianoroll(min_pitch=21)

        song_id = song_info['song_name'].split('/')[1]

        # add markers
        annotations: list[tuple[int, str]] = []
        for segment in segments:
            annotations.append((segment["start"], segment["label"]))
        # add '(seed)' to the marker of the seed segment
        for j, annotation in enumerate(annotations):
            if annotation[0] == seed_segment['start']:
                annotations[j] = (annotation[0], f'{annotation[1]} (seed)')

        output_name = f"{i}_{song_id}.mid"

        generated_output_path = args.output_path / "flat" / output_name
        generated_output_path.parent.mkdir(parents=True, exist_ok=True)

        generated_song.to_midi(generated_output_path, markers=annotations)

        # original_pr.to_midi(original_output_path, markers=annotations)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--model_config", type=str, default="config/model/simple_ar.yaml"
    )
    arg_parser.add_argument("--output_path", type=Path, default=Path("eval_results/midi/eval/flat/midi"))
    arg_parser.add_argument(
        "--ckpt", type=str, default="wandb/run-20250721_013952-ahz9fo2m/files/checkpoints/epoch=7-step=3000000.safetensors"
    )
    arg_parser.add_argument(
        "--dataset_path", type=Path
    )

    args = arg_parser.parse_args()

    main(args)
