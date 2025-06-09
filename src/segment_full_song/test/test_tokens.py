from pathlib import Path
from music_data_analysis import Pianoroll
from segment_full_song.models.token_sequence import TokenSequence


# pr1 = Pianoroll(
#     [
#         Note(onset=0, pitch=60, velocity=100, offset=3),
#         Note(onset=3, pitch=60, velocity=100, offset=6),
#         Note(onset=3, pitch=62, velocity=100, offset=5),
#         Note(onset=6, pitch=60, velocity=100, offset=7),
#         Note(onset=6, pitch=62, velocity=100, offset=8),
#         Note(onset=31, pitch=60, velocity=100, offset=64),
#         Note(onset=31, pitch=50, velocity=100, offset=33),
#         Note(onset=33, pitch=50, velocity=100, offset=64),
#     ],
#     duration=64,
# )
pr1 = Pianoroll.from_midi(
    Path(
        "dataset/pop80k_k/synced_midi/@SangeoMusic/0Zo2uPKfzvg/3438_3730.mid"
    )
)
pr1.to_midi("test.midi")

TokenSequence.from_pianorolls(
    [pr1], max_note_duration=64, need_frame_tokens=True
).to_pianoroll().to_midi("test2.midi")

for note in pr1.notes:
    if note.offset - note.onset > 63:
        note.offset = note.onset + 63


def check_note_equal(pr1, pr2):
    for note1, note2 in zip(pr1, pr2):
        try:
            assert note1.onset == note2.onset
            assert note1.pitch == note2.pitch
            assert note1.velocity == note2.velocity
            assert note1.offset == note2.offset
        except AssertionError:
            print(note1, note2)
            raise AssertionError


print(
    TokenSequence.from_pianorolls([pr1], max_note_duration=63, need_frame_tokens=False)
)
print(
    TokenSequence.from_pianorolls([pr1], max_note_duration=63, need_frame_tokens=False)
    .to_pianoroll()
    .notes
)
check_note_equal(
    TokenSequence.from_pianorolls([pr1], max_note_duration=63, need_frame_tokens=False)
    .to_pianoroll()
    .notes,
    pr1.notes,
)
check_note_equal(
    TokenSequence.from_pianorolls([pr1], max_note_duration=63, need_frame_tokens=True)
    .to_pianoroll()
    .notes,
    pr1.notes,
)
check_note_equal(
    TokenSequence.from_pianorolls(
        [pr1], max_note_duration=63, need_frame_tokens=True, need_end_token=True
    )
    .to_pianoroll()
    .notes,
    pr1.notes,
)
