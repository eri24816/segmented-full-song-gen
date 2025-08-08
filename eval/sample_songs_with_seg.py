from pathlib import Path
from music_data_analysis import Dataset, Pianoroll
import random
import json
from tqdm import tqdm

from segment_full_song.data.dataset import FullSongPianorollDataset
from segment_full_song.data.segment import get_compose_order

random.seed(325)

def check_seed_segment_length(segmentation: list[dict], length: int) -> bool:
    compose_order = get_compose_order(segmentation)
    seed_segment = compose_order[0]
    if seed_segment['end'] - seed_segment['start'] != length:
        return False
    return True


output_name = 'eval_results/song_list/user_study.json'

ds = FullSongPianorollDataset(Path('../pop80k_k'), split='test')

n = 64
min_duration = 32*40
max_duration = 32*60

result = []
song_idxs = list(range(len(ds)))
random.shuffle(song_idxs)

idx = 0
for song_idx in tqdm(song_idxs):
    song = ds[song_idx]
    segmentation = song.read_json('segmentation')

    # if not check_seed_segment_length(segmentation, 32*8):
    #     continue

    pr = Pianoroll.from_midi(song.read_midi('synced_midi'))

    if pr.duration > max_duration:
        continue
    if pr.duration < min_duration:
        continue


    compose_order = get_compose_order(segmentation)
    seed_segment = compose_order[0]

    if seed_segment['end'] - seed_segment['start'] >= 32*10:
        continue


    annotations: list[tuple[int, str]] = []
    for segment in segmentation:
        annotations.append((segment["start"], 'ABCDEF'[segment["label"]]))

    Path(f'eval_results/song_list/user_study').mkdir(parents=True, exist_ok=True)
    pr.to_midi(f'eval_results/song_list/user_study/{idx}_{song.song_name.split("/")[1]}.mid', markers=annotations)

    for segment in segmentation:
        segment['start'] = segment['start'] // 32
        segment['end'] = segment['end'] // 32


    result.append({
        'idx': idx,
        'song_name': song.song_name,
        'segmentation': segmentation,
        'seed_segment': seed_segment,
    })


    idx += 1
    if len(result) >= n:
        break

# save result
Path(output_name).parent.mkdir(parents=True, exist_ok=True)
with open(output_name, 'w') as f:
    json.dump(result, f, indent=4)