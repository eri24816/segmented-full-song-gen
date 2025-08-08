from pathlib import Path
from typing import Iterable
from music_data_analysis import Pianoroll, Dataset, Song
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import numpy as np
from music_data_analysis.processors.segmentation import get_overlap_sim, get_skyline
import json
import argparse
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

    cummulative_bars = 0
    for label, duration in matches:
        duration = int(duration)
        segment = {
            "start": cummulative_bars*32,
            "end": (cummulative_bars + duration)*32,
            "label": label,
        }
        segments.append(segment)
        cummulative_bars += duration
    return segments

if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument('--dataset_path', type=Path, default=Path('../pop80k_k'))
    args.add_argument('--input_midi_dir', type=Path, default=None)
    args.add_argument('--output_name', type=str, default='dataset')
    args = args.parse_args()

    def get_label(segments: list[dict[str, int]], start: int) -> int:
        for segment in segments:
            if segment['start'] <= start < segment['end']:
                return segment['label']
        return segments[-1]['label']


    if args.input_midi_dir is not None:
        # load all midi in the folder
        midi_files = args.input_midi_dir.glob('*.mid')
        iterable = list(midi_files)
        get_pr = lambda x: Pianoroll.from_midi(x, track_idx=-1)
        structures = {
            'ABCDE':{
                'segments_str': 'A8B16C8D8E8',
                'compose_order': [2,0,1,3,4],
            },
            'ABCDBCE':{
                'segments_str': 'A4B8C8D8B8C16E8',
                'compose_order': [2,5,1,4,3,0,6],
            },
            'ABCBDBE':{
                'segments_str': 'A8B8C4B8D8B8E4',
                'compose_order': [1,0,2,3,4,5,6],
            },
            'ABCACAD':{
                'segments_str': 'A4B8C8A8C8A8D8',
                'compose_order': [3, 0, 1, 2, 4, 5, 6],
            },
        }
        def get_segments(file_name: str) -> list[dict[str, int]]:
            structure_name = Path(file_name).stem.split('_')[-1]
            structure = structures[structure_name]
            return parse_segments_str(structure['segments_str'])

    else:
        ds = FullSongPianorollDataset(args.dataset_path, min_duration=128, max_duration=6400, split='test') # type: ignore
        get_pr = lambda x: Pianoroll.from_midi(x.read_midi('synced_midi'))
        get_segments = lambda x: x.read_json('segmentation')

        iterable = ds

    bar_overall_sims = []
    skyline_overall_sims = []
    same_label_bar_sims = []
    same_label_skyline_sims = []
    different_label_bar_sims = []
    different_label_skyline_sims = []

    n = min(200, len(iterable))

    for i in tqdm(random.sample(range(len(iterable)), n)):
        song = iterable[i]
        pr = get_pr(song)
        segments = get_segments(song) # list of {'start': int, 'end': int, 'label': int}
        for i, bar_a in enumerate(pr.iter_over_bars_pr()):
            label_a = get_label(segments, i*32)
            for j, bar_b in enumerate(pr.iter_over_bars_pr()):
                label_b = get_label(segments, j*32)
                if i < j:
                    if len(bar_a.notes) == 0 or len(bar_b.notes) == 0:
                        continue
                    bar_sim = get_overlap_sim(bar_a, bar_b)
                    bar_overall_sims.append(bar_sim)
                    skyline_a = get_skyline(bar_a)
                    skyline_b = get_skyline(bar_b)
                    skyline_sim = get_overlap_sim(skyline_a, skyline_b)
                    skyline_overall_sims.append(skyline_sim)

                    if label_a == label_b:
                        same_label_bar_sims.append(bar_sim)
                        same_label_skyline_sims.append(skyline_sim)
                    else:
                        different_label_bar_sims.append(bar_sim)
                        different_label_skyline_sims.append(skyline_sim)



    # get percentiles of the overall sims
    # bar_overall_percentiles = [np.percentile(bar_overall_sims, i+10) for i in range(0, 100, 10)]
    # skyline_overall_percentiles = [np.percentile(skyline_overall_sims, i+10) for i in range(0, 100, 10)]

    # print([i+50 for i in range(0, 100, 50)])
    # print(bar_overall_percentiles)
    # print(skyline_overall_percentiles)

    # def create_histogram(data: list[float], separation_points: list) -> list[float]:
    #     histogram = [0.0] * len(separation_points)
    #     for i in range(len(data)):
    #         for j in range(len(separation_points)):
    #             if data[i] < separation_points[j]:
    #                 histogram[j] += 1
    #                 break

    #     return [x / len(data) for x in histogram]

    # bar_overall_hist = create_histogram(bar_overall_sims, bar_overall_percentiles)
    # bar_same_label_hist = create_histogram(same_label_bar_sims, bar_overall_percentiles)
    # bar_different_label_hist = create_histogram(different_label_bar_sims, bar_overall_percentiles)
    # skyline_overall_hist = create_histogram(skyline_overall_sims, skyline_overall_percentiles)
    # skyline_same_label_hist = create_histogram(same_label_skyline_sims, skyline_overall_percentiles)
    # skyline_different_label_hist = create_histogram(different_label_skyline_sims, skyline_overall_percentiles)


    # # First subplot for bar similarity
    # x_values = [i for i in range(len(bar_overall_hist))]
    # ax1.plot(x_values, bar_overall_hist, linewidth=2, marker='o', markersize=4, alpha=0.7, label='Overall')
    # ax1.plot(x_values, bar_same_label_hist, linewidth=2, marker='s', markersize=4, alpha=0.7, label='Same Label')
    # ax1.plot(x_values, bar_different_label_hist, linewidth=2, marker='^', markersize=4, alpha=0.7, label='Different Label')

    # ax1.set_title('Bar Similarity')
    # ax1.set_xlabel('Similarity')
    # ax1.set_ylabel('Proportion')
    # ax1.legend()

    # # Second subplot for skyline similarity
    # x_values = [i for i in range(len(skyline_overall_hist))]
    # ax2.plot(x_values, skyline_overall_hist, linewidth=2, marker='o', markersize=4, alpha=0.7, label='Overall')
    # ax2.plot(x_values, skyline_same_label_hist, linewidth=2, marker='s', markersize=4, alpha=0.7, label='Same Label')
    # ax2.plot(x_values, skyline_different_label_hist, linewidth=2, marker='^', markersize=4, alpha=0.7, label='Different Label')
    # ax2.set_title('Skyline Similarity')
    # ax2.set_xlabel('Similarity')
    # ax2.set_ylabel('Proportion')
    # ax2.legend()

    # plt.tight_layout()
    # plt.show()

    # First subplot for bar similarity

    results = {}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    num_bins = 32
    bin_size = 1 / num_bins
    x_values = [i * bin_size for i in range(num_bins)]
    print(x_values)
    n, bins = np.histogram(bar_overall_sims, bins=num_bins)
    n = n / len(bar_overall_sims)
    results['bar_overall'] = n.tolist()
    ax1.plot(bins[:-1], n, label='Overall')
    n, bins = np.histogram(same_label_bar_sims, bins=num_bins)
    n = n / len(same_label_bar_sims)
    results['bar_same_label'] = n.tolist()
    ax1.plot(bins[:-1], n, label='Same Label')
    n, bins = np.histogram(different_label_bar_sims, bins=num_bins)
    n = n / len(different_label_bar_sims)
    results['bar_different_label'] = n.tolist()
    ax1.plot(bins[:-1], n, label='Different Label')

    ax1.set_title('Bar Similarity')
    ax1.set_xlabel('Similarity')
    ax1.set_ylabel('Proportion')
    ax1.legend()

    # Second subplot for skyline similarity
    x_values = [i * bin_size for i in range(num_bins)]
    n, bins = np.histogram(skyline_overall_sims, bins=num_bins)
    n = n / len(skyline_overall_sims)
    results['skyline_overall'] = n.tolist()
    ax2.plot(bins[:-1], n, label='Overall')
    n, bins = np.histogram(same_label_skyline_sims, bins=num_bins)
    n = n / len(same_label_skyline_sims)
    results['skyline_same_label'] = n.tolist()
    ax2.plot(bins[:-1], n, label='Same Label')
    n, bins = np.histogram(different_label_skyline_sims, bins=num_bins)
    n = n / len(different_label_skyline_sims)
    results['skyline_different_label'] = n.tolist()
    ax2.plot(bins[:-1], n, label='Different Label')
    ax2.set_title('Skyline Similarity')
    ax2.set_xlabel('Similarity')
    ax2.set_ylabel('Proportion')
    ax2.legend()

    # save means
    results['bar_overall_mean'] = float(np.mean(bar_overall_sims))
    results['bar_same_label_mean'] = float(np.mean(same_label_bar_sims))
    results['bar_different_label_mean'] = float(np.mean(different_label_bar_sims))
    results['skyline_overall_mean'] = float(np.mean(skyline_overall_sims))
    results['skyline_same_label_mean'] = float(np.mean(same_label_skyline_sims))
    results['skyline_different_label_mean'] = float(np.mean(different_label_skyline_sims))

    Path('eval_results/metrics/similarity').mkdir(parents=True, exist_ok=True)
    with open(f'eval_results/metrics/similarity/{args.output_name}.json', 'w') as f:
        json.dump(results, f)

    plt.tight_layout()
    plt.show()

    plt.savefig('eval_results/metrics/similarity/dataset.png')