from copy import deepcopy

import numpy as np
import pretty_midi
import torch
from matplotlib import pyplot as plt


def save_img(img, path):
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    plt.imsave(path, np.flip(img, axis=0))


def adjust_notes_with_pedal(midi_data):
    new_midi_data = deepcopy(midi_data)

    for instrument in new_midi_data.instruments:
        # Get sustain pedal events
        sustain_events = [cc for cc in instrument.control_changes if cc.number == 64]
        sustain_events.sort(key=lambda x: x.time)  # Ensure events are sorted by time

        if not sustain_events:
            continue  # Skip if no pedal data

        # Track the last pedal release time
        last_pedal_release = None
        is_pedal_down = False

        # Iterate through notes
        for note in instrument.notes:
            # Find the next pedal release after the note start
            for cc in sustain_events:
                if cc.time >= note.start:
                    if cc.value > 64:
                        is_pedal_down = True  # Pedal is pressed
                    else:
                        if is_pedal_down:  # Pedal is released
                            last_pedal_release = cc.time
                            is_pedal_down = False
                        break  # Stop looking after the first release

            # Extend note end time if pedal was down
            if last_pedal_release and note.end < last_pedal_release:
                note.end = last_pedal_release

    return new_midi_data


def get_piano_roll_onset(midi_data: pretty_midi.PrettyMIDI, fs: int):
    # without the +1, indexerror may happen
    times = np.arange(0, midi_data.get_end_time() + 1, 1.0 / fs)
    pr_onset = np.zeros((128, times.shape[0]), dtype=float)

    for notes in midi_data.instruments:
        for note in notes.notes:
            # x_values = np.arange(0, pr_onset.shape[1])
            # sigma = 1
            # # Standard deviation
            # probs = stats.norm.pdf(x_values, note.start * fs, sigma)
            # pr_onset[note.pitch, :] += probs

            start = round(note.start * fs)
            pr_onset[note.pitch, start] = 1

            # pr_onset[note.pitch, round(note.start * fs) : max(round(note.end * fs), round(note.start * fs) + 1)] = 1

    return pr_onset


def get_quantized_midi(midi_data, fs):
    new_midi_data = deepcopy(midi_data)
    st = 1 / fs
    for note in new_midi_data.instruments[0].notes:
        note.start = round(note.start / st) * st
        note.end = max(round(note.end / st) * st, note.start + st)
    return new_midi_data
