from collections import defaultdict
from typing import Self, Sequence, cast
import torch
from music_data_analysis import Note, Pianoroll
from torch import Tensor

from vqpiano.utils.torch_utils.tensor_op import cat_to_right, pad_and_cat, pad_and_stack

Tensorable = Tensor | int | float | list[int | float]


def tokenize(
    pr: Pianoroll,
    max_note_duration: int,
    max_length: int | None = None,
    pitch_range: list[int] = [21, 109],
    pad: bool = False,
    need_end_token: bool = False,
    need_frame_tokens: bool = True,
):
    """
    token type:
        -1: pad
        0: frame
        1: note
    """
    current_frame = 0
    tokens = []
    token_types = []
    pos = []

    if need_frame_tokens:
        tokens.append([0, 0, 0])
        token_types.append(0)
        pos.append(current_frame)

    for bar in pr.iter_over_bars():
        last_note_of_pitch = {}
        notes_with_successor = []
        for note in bar:
            if (
                note.pitch in last_note_of_pitch
                and last_note_of_pitch[note.pitch].offset == note.onset
            ):
                notes_with_successor.append(last_note_of_pitch[note.pitch])
            last_note_of_pitch[note.pitch] = note

        for note in bar:
            # add frame tokens until the note
            if note.onset > current_frame:
                for _ in range(note.onset - current_frame):
                    current_frame += 1

                    if need_frame_tokens:
                        tokens.append([0, 0, 0])
                        token_types.append(0)
                        pos.append(current_frame)

                current_frame = note.onset

            # calculate duration
            has_successor = note in notes_with_successor
            offset_at_bar_end = cast(int, note.offset) % pr.frames_per_bar == 0
            if has_successor or offset_at_bar_end:
                # 0 is a special class for "to be determined with onset of the next note with same pitch"
                duration = 0
            else:
                duration = cast(int, note.offset) - note.onset
                assert duration > 0
                if duration >= max_note_duration:
                    duration = max_note_duration - 1

            assert note.pitch >= pitch_range[0] and note.pitch <= pitch_range[1]
            tokens.append([note.pitch - pitch_range[0], note.velocity, duration])
            token_types.append(1)
            pos.append(note.onset)

    for current_frame in range(current_frame + 1, pr.duration):
        if need_frame_tokens:
            tokens.append([0, 0, 0])
            token_types.append(0)
            pos.append(current_frame)

    # end token is a frame token with pos 0.
    if need_end_token:
        assert need_frame_tokens, (
            "if need_end_token is True, need_frame_tokens must be True"
        )
        tokens.append([0, 0, 0])
        token_types.append(0)
        pos.append(0)

    if pad and max_length is not None and max_length > len(tokens):
        for _ in range(max_length - len(tokens)):
            tokens.append([0, 0, 0])
            token_types.append(-1)
            pos.append(current_frame)

    if max_length is not None and len(tokens) > max_length:
        print(f"Truncating the input from {len(tokens)} to {max_length}")
        tokens = tokens[:max_length]
        token_types = token_types[:max_length]
        pos = pos[:max_length]

    tokens = torch.tensor(tokens)
    token_types = torch.tensor(token_types)
    pos = torch.tensor(pos)

    # handle empty pianoroll
    if tokens.shape[0] == 0:
        tokens = torch.zeros(0, 3, dtype=torch.long)
        token_types = torch.zeros(0, dtype=torch.long)
        pos = torch.zeros(0, dtype=torch.long)

    return tokens, token_types, pos


class TokenSequence:
    """

    Represent a sequence of notes with a sequence of tokens. Batched.

    Example:
    ```python
        pr1 = Pianoroll(
            [
                Note(onset=0, pitch=60, velocity=100),
                Note(onset=3, pitch=60, velocity=100),
            ],
            duration=4,
        )
        pr2 = Pianoroll(
            [
                Note(onset=0, pitch=60, velocity=100),
                Note(onset=1, pitch=60, velocity=100),
                Note(onset=1, pitch=67, velocity=100),
                Note(onset=3, pitch=60, velocity=100),
            ],
            duration=4,
        )
        print(SymbolicRepresentation.from_pianorolls([pr1, pr2]))
        # SymbolicRepresentation(
        #   [0].pos:      0       0       1       2        3        3       0    0
        #   [0].token:  Frame  N39/100  Frame   Frame    Frame   N39/100   Pad   Pad
        #   [1].pos:      0       0       1       1        1        2       3    3
        #   [1].token:  Frame  N39/100  Frame  N39/100  N46/100   Frame   Frame  N39/100
        # )

        print(
            SymbolicRepresentation.from_pianorolls_sliced(
                [pr1, pr2], [slice(0, 2), slice(2, 4)]
            )
        )
        # [SymbolicRepresentation(
        #   [0].pos:      0       0       1       0     0
        #   [0].token:  Frame  N39/100  Frame    Pad    Pad
        #   [1].pos:      0       0       1       1     1
        #   [1].token:  Frame  N39/100  Frame  N39/100  N46/100
        # ), SymbolicRepresentation(
        #   [0].pos:      0      1    1
        #   [0].token:  Frame  Frame  N39/100
        #   [1].pos:      0      1    1
        #   [1].token:  Frame  Frame  N39/100
        # )]
    ```
    """

    PAD = -1
    FRAME = 0
    NOTE = 1

    @classmethod
    def from_pianorolls(
        cls,
        batch: list[Pianoroll],
        max_note_duration: int,
        max_tokens: int | None = None,
        max_tokens_rate: float | None = None,
        device: torch.device | str = torch.device("cpu"),
        need_end_token: bool = False,
        need_frame_tokens: bool = True,
    ) -> Self:
        """
        Convert a list of music_data_analysis.Pianoroll objects to a SymbolicRepresentation object, stacked in the batch dimension.
        Can be used as a collate_fn for a DataLoader.
        """

        if max_tokens_rate is not None:
            assert max_tokens is None, (
                "max_tokens and max_tokens_rate cannot be both provided"
            )
            max_tokens = int(max(pr.duration for pr in batch) * max_tokens_rate)

        tokens_batch = []
        token_types_batch = []
        pos_batch = []
        for pr in batch:
            tokens, token_types, pos = tokenize(
                pr,
                max_note_duration=max_note_duration,
                max_length=max_tokens,
                need_end_token=need_end_token,
                need_frame_tokens=need_frame_tokens,
            )
            tokens_batch.append(tokens)
            token_types_batch.append(token_types)
            pos_batch.append(pos)
        tokens_batch = pad_and_stack(tokens_batch, 0)
        token_types_batch = pad_and_stack(token_types_batch, 0, pad_value=-1)
        pos_batch = pad_and_stack(pos_batch, 0)

        tokens_batch = tokens_batch.to(device)
        token_types_batch = token_types_batch.to(device)
        pos_batch = pos_batch.to(device)

        return cls(tokens_batch, token_types_batch, pos_batch, device)

    @classmethod
    def from_pianorolls_sliced(
        cls,
        batch: list[Pianoroll],
        slices: Sequence[slice],
        max_note_duration: int,
        max_tokens: Sequence[int | None] | None = None,
        need_end_token: list[bool] | None = None,
    ) -> list[Self]:
        """
        Recieve a list of music_data_analysis.Pianoroll objects, stack them in the batch dimension, slice them in the time dimension,
        and build each slice into a SymbolicRepresentation object. Each slice is padded individually so that each batch item in one slice
        is guaranteed to have the same duration and length.
        Can be used as a collate_fn for a DataLoader.
        """
        if max_tokens is None:
            max_tokens = [None] * len(slices)
        if need_end_token is None:
            need_end_token = [False] * len(slices)

        tokens_all: list[list[Tensor]] = [[] for _ in range(len(slices))]
        token_types_all: list[list[Tensor]] = [[] for _ in range(len(slices))]
        pos_all: list[list[Tensor]] = [[] for _ in range(len(slices))]
        for pr in batch:
            for i, (max_token, slice, need_end_token_item) in enumerate(
                zip(max_tokens, slices, need_end_token)
            ):
                tokens, token_types, pos = tokenize(
                    pr.slice(slice.start, slice.stop),
                    max_length=max_token,
                    need_end_token=need_end_token_item,
                    max_note_duration=max_note_duration,
                )
                tokens_all[i].append(tokens)
                token_types_all[i].append(token_types)
                pos_all[i].append(pos)

        result: list[Self] = []
        for i in range(len(slices)):
            tokens_batch = pad_and_stack(tokens_all[i], 0)
            token_types_batch = pad_and_stack(token_types_all[i], 0, pad_value=-1)
            pos_batch = pad_and_stack(pos_all[i], 0)

            result.append(cls(tokens_batch, token_types_batch, pos_batch))

        return result

    @classmethod
    def cat_batch(cls, batch: list["TokenSequence"]):
        """
        Concatenate a list of SymbolicRepresentation objects in batch dim.
        """
        result = cls(
            device=batch[0].device,
            token=pad_and_cat([item.token for item in batch], pad_dim=1, cat_dim=0),
            token_type=pad_and_cat(
                [item.token_type for item in batch], pad_dim=1, pad_value=-1
            ),
            pos=pad_and_cat([item.pos for item in batch], pad_dim=1),
        )
        return result

    @classmethod
    def cat_time(cls, batch: list["TokenSequence"]):
        """
        Concatenate a list of SymbolicRepresentation objects in time dim.
        """
        result = cls(device=batch[0].device)
        for i in range(len(batch)):
            result += batch[i]
        return result

    def __init__(
        self,
        token: Tensorable | None = None,
        token_type: Tensorable | None = None,
        pos: Tensorable | None = None,
        device: torch.device | str = torch.device("cpu"),
    ):
        """
        token: (batch_size, length, data_dim)
        token_type: (batch_size, length)
        pos: (batch_size, length)
        """
        self.device = device

        if token is None or token_type is None or pos is None:
            assert token is None and token_type is None and pos is None, (
                "if one argument is provided, all arguments must be provided"
            )
            token = torch.zeros(1, 0, 3, dtype=torch.long, device=device)
            token_type = torch.zeros(1, 0, dtype=torch.long, device=device)
            pos = torch.zeros(1, 0, dtype=torch.long, device=device)

        if not isinstance(token, torch.Tensor):
            token = torch.tensor(token)
        if not isinstance(token_type, torch.Tensor):
            token_type = torch.tensor(token_type)
        if not isinstance(pos, torch.Tensor):
            pos = torch.tensor(pos)

        assert token.shape[0:1] == token_type.shape[0:1] == pos.shape[0:1], (
            "batch size or length mismatch"
        )

        self.token = token
        self.token_type = token_type
        self.pos = pos

    @property
    def is_frame(self):
        return self.token_type == self.FRAME

    @property
    def is_note(self):
        return self.token_type == self.NOTE

    @property
    def is_pad(self):
        return self.token_type == self.PAD

    @property
    def pitch(self):
        return self.token[:, :, 0]

    @property
    def velocity(self):
        return self.token[:, :, 1]

    @property
    def note_duration(self):
        return self.token[:, :, 2]

    @property
    def length(self):
        return self.token.shape[1]

    @property
    def max_pos(self):
        return int(self.pos.max().item())

    @property
    def duration(self):
        if self.length == 0:
            return 0
        return int(self.pos.max().item() + 1)

    @property
    def batch_size(self):
        return self.token.shape[0]

    @property
    def shape(self):
        return (self.batch_size, self.length)

    def __add__(self, other: "TokenSequence"):
        return TokenSequence(
            cat_to_right(self.token, other.token, dim=1),
            cat_to_right(self.token_type, other.token_type, dim=1),
            cat_to_right(self.pos, other.pos + self.duration * ~other.is_pad, dim=1),
            self.device,
        )

    def __len__(self):
        return self.length

    def __getitem__(self, index: slice | tuple[slice, ...]):
        new_token = self.token[index]
        new_token_type = self.token_type[index]
        new_pos = self.pos[index]

        assert new_token.ndim == self.token.ndim
        assert new_token_type.ndim == self.token_type.ndim
        assert new_pos.ndim == self.pos.ndim

        # normalize pos
        if new_pos.numel() > 0:
            new_pos = new_pos - new_pos[new_token_type != self.PAD].min()

        return TokenSequence(new_token, new_token_type, new_pos, self.device)

    def slice_pos(self, start: int | None, end: int | None):
        assert self.batch_size == 1, "slice is only supported for batch size 1"
        start_idx = (
            torch.searchsorted(self.pos[0], start) if start is not None else None
        )
        end_idx = torch.searchsorted(self.pos[0], end) if end is not None else None
        return self[:, start_idx:end_idx]

    def to(self, device: torch.device | str):
        self.device = device
        self.token = self.token.to(device)
        self.token_type = self.token_type.to(device)
        self.pos = self.pos.to(device)
        return self

    def add_frame(self):
        self.pos = cat_to_right(self.pos, self.duration, dim=1)
        self.token = cat_to_right(self.token, [0, 0, 0], dim=1)
        self.token_type = cat_to_right(self.token_type, 0, dim=1)
        return self

    def add_note(
        self, pitch: int | Tensor, velocity: int | Tensor, duration: int | Tensor
    ):
        self.pos = cat_to_right(self.pos, self.duration - 1, dim=1)
        self.token = cat_to_right(self.token, [pitch, velocity, duration], dim=1)
        self.token_type = cat_to_right(self.token_type, self.NOTE, dim=1)
        return self

    def add_pad(self):
        self.pos = cat_to_right(self.pos, 0, dim=1)
        self.token = cat_to_right(self.token, [0, 0, 0], dim=1)
        self.token_type = cat_to_right(self.token_type, self.PAD, dim=1)
        return self

    def pop_frame(self):
        assert self.token_type[:, -1] == self.FRAME
        self.token = self.token[:, :-1, :]
        self.token_type = self.token_type[:, :-1]
        self.pos = self.pos[:, :-1]
        return self

    def clone(self):
        return TokenSequence(
            self.token.clone(), self.token_type.clone(), self.pos.clone(), self.device
        )

    def shift_pos(self, shift: int):
        self.pos = self.pos + shift * ~self.is_pad
        return self

    def to_pianoroll(
        self,
        min_pitch: int = 21,
        batch_item: int = 0,
        frames_per_beat: int = 8,
        beats_per_bar: int = 4,
    ) -> Pianoroll:
        notes: list[Note] = []
        durations: list[int] = []
        for i in range(self.token_type.shape[1]):
            if self.token_type[batch_item, i] == self.NOTE:
                time = self.pos[batch_item, i].item()
                pitch = self.pitch[batch_item, i].item()
                velocity = self.velocity[batch_item, i].item()
                duration = self.note_duration[batch_item, i].item()
                assert isinstance(time, int)
                assert isinstance(pitch, int)
                assert isinstance(velocity, int)
                assert isinstance(duration, int)
                notes.append(
                    Note(onset=time, pitch=pitch + min_pitch, velocity=velocity)
                )
                durations.append(duration)

        # apply note duration

        notes_per_pitch = defaultdict(list)
        for note in notes:
            notes_per_pitch[note.pitch].append(note)

        successors = {}
        for pitch, notes_of_pitch in notes_per_pitch.items():
            for i in range(len(notes_of_pitch)):
                if i == len(notes_of_pitch) - 1:
                    successors[notes_of_pitch[i]] = None
                else:
                    successors[notes_of_pitch[i]] = notes_of_pitch[i + 1]

        frames_per_bar = frames_per_beat * beats_per_bar
        for i in range(len(notes)):
            if durations[i] == 0:
                if successors[notes[i]] is not None:
                    # hold until the next note
                    notes[i].offset = successors[notes[i]].onset
                else:
                    # hold until the end of the bar
                    notes[i].offset = (
                        (notes[i].onset + frames_per_bar) // frames_per_bar
                    ) * frames_per_bar
            else:
                notes[i].offset = notes[i].onset + durations[i]

        return Pianoroll(
            notes,
            duration=self.duration,
            beats_per_bar=beats_per_bar,
            frames_per_beat=frames_per_beat,
        )

    def to_midi(self, min_pitch: int, path=None, apply_pedal=False, bpm=105):
        pianoroll = self.to_pianoroll(min_pitch)
        return pianoroll.to_midi(path, apply_pedal, bpm)

    def __iter__(self):
        for i in range(self.batch_size):
            yield self[i : i + 1]

    def __repr__(self):
        def token_to_str(token_type: int, token: Tensor):
            if token_type == self.PAD:
                return "Pad"
            elif token_type == self.FRAME:
                return "Frame"
            elif token_type == self.NOTE:
                return f"N{token[0].item()}/{token[1].item()}/{token[2].item()}"
            else:
                raise ValueError(f"Invalid token type: {token_type}")

        max_length = 32
        max_batch_size = 5
        display_length = min(self.length, max_length)
        display_batch_size = min(self.batch_size, max_batch_size)

        table: list[list[str]] = []

        for i in range(display_batch_size):
            row = []
            row.append(f"[{i}].pos:".ljust(10))
            for j in range(display_length):
                row.append(str(self.pos[i, j].item()))
            table.append(row)

            row = []
            row.append(f"[{i}].token:".ljust(10))
            for j in range(display_length):
                row.append(token_to_str(int(self.token_type[i, j]), self.token[i, j]))
            table.append(row)

        if self.length > display_length:
            for row in table:
                row.append("...")

        # equal length of each column
        for col in range(display_length):
            max_length = max(len(row[col]) for row in table)
            for row in table:
                row[col] = row[col].center(max_length)

        result = (
            "SymbolicRepresentation(\n  "
            + "\n  ".join(["  ".join(row) for row in table])
            + "\n)"
        )
        if self.batch_size > max_batch_size:
            result += "\n..."
        return result


if __name__ == "__main__":
    pr1 = Pianoroll(
        [
            Note(onset=0, pitch=60, velocity=100, offset=3),
            Note(onset=3, pitch=60, velocity=100, offset=6),
            Note(onset=3, pitch=62, velocity=100, offset=5),
            Note(onset=6, pitch=60, velocity=100, offset=7),
            Note(onset=6, pitch=62, velocity=100, offset=8),
            Note(onset=31, pitch=60, velocity=100, offset=32),
        ],
        duration=32,
    )
    print(TokenSequence.from_pianorolls([pr1], need_frame_tokens=False))
    print(
        TokenSequence.from_pianorolls([pr1], need_frame_tokens=False)
        .to_pianoroll()
        .notes
    )
    assert (
        TokenSequence.from_pianorolls([pr1], need_frame_tokens=False)
        .to_pianoroll()
        .notes
        == pr1.notes
    )
    assert (
        TokenSequence.from_pianorolls([pr1], need_frame_tokens=True)
        .to_pianoroll()
        .notes
        == pr1.notes
    )
    assert (
        TokenSequence.from_pianorolls(
            [pr1], need_frame_tokens=True, need_end_token=True
        )
        .to_pianoroll()
        .notes
        == pr1.notes
    )
