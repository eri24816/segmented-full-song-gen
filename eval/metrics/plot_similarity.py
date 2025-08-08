names = ['dataset', 'ours', 'whole_song', 'ours_p1']
result_dir = 'eval_results/metrics/similarity'

import json
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))

# Plot bar_overall
plt.subplot(1, 3, 1)
num_bins = 32
bin_size = 1 / num_bins
x = [i * bin_size for i in range(num_bins)]
for name in names:
    with open(f'{result_dir}/{name}.json', 'r') as f:
        data = json.load(f)
    plt.plot(x, data['bar_overall'], label=name)
plt.title('Similarity between 2 bars')
plt.legend()

# Plot bar_same_label
plt.subplot(1, 3, 2)
for name in names:
    with open(f'{result_dir}/{name}.json', 'r') as f:
        data = json.load(f)
    plt.plot(x, data['bar_same_label'], label=name)
plt.title('Similarity between 2 bars with same label')
plt.legend()

# Plot bar_different_label
plt.subplot(1, 3, 3)
for name in names:
    with open(f'{result_dir}/{name}.json', 'r') as f:
        data = json.load(f)
    plt.plot(x, data['bar_different_label'], label=name)
plt.title('Similarity between 2 bars with different label')
plt.legend()

plt.tight_layout()
plt.show()