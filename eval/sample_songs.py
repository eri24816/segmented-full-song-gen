from pathlib import Path
from music_data_analysis import Dataset
import random
import json
from tqdm import tqdm

from segment_full_song.data.dataset import FullSongPianorollDataset
from segment_full_song.data.segment import get_compose_order

def check_seed_segment_length(segmentation: list[dict], length: int) -> bool:
    compose_order = get_compose_order(segmentation)
    seed_segment = compose_order[0]
    if seed_segment['end'] - seed_segment['start'] != length:
        return False
    return True


output_name = 'eval/song_list/8_bars_seed.txt'

ds = FullSongPianorollDataset(Path('../pop80k_k'), split='test')

n = 1000000
min_duration = 32*40
max_duration = 32*60

result = []
song_idxs = list(range(len(ds)))
random.shuffle(song_idxs)

for song_idx in tqdm(song_idxs):
    song = ds[song_idx]
    segmentation = song.read_json('segmentation')

    if not check_seed_segment_length(segmentation, 32*8):
        continue

    result.append(song.song_name)

    if len(result) >= n:
        break

# save result
Path(output_name).parent.mkdir(parents=True, exist_ok=True)
with open(output_name, 'w') as f:
    f.write('\n'.join(result))