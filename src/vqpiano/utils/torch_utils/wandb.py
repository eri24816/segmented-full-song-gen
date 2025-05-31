import os
from pathlib import Path

from music_data_analysis import Pianoroll
import torch
from miditoolkit import MidiFile

import wandb

from .media import save_img
from .os import run_command

import matplotlib.pyplot as plt
import numpy as np

def pianorolls_to_img(pianorolls: list[Pianoroll], path: Path, size_factor: int = 1, annotations: list[tuple[int, str]] = []):
    """
    Convert the pianoroll to a image
    """

    def create_fig_with_size(w:int, h:int, dpi:int = 100):
        fig = plt.figure(figsize=(w/dpi, h/dpi), dpi=dpi)
        ax = plt.Axes(fig, [0, 0, 1, 1])  # use full figure area
        ax.set_axis_off()
        fig.add_axes(ax)
        return fig, ax

    img = np.zeros((88 * len(pianorolls), max(pr.duration for pr in pianorolls)))
    for i, pr in enumerate(pianorolls):
        for time, pitch, vel, offset in pr.iter_over_notes_unpack():
            if time >= pr.duration:
                print("Warning: time >= duration") #TODO: fix this
                continue
            img[pitch - 21 + 88 * i, time] = vel
    # enlarge the image
    img = np.repeat(img, size_factor, axis=0)
    img = np.repeat(img, size_factor, axis=1)

    # add bar lines
    for t in range(img.shape[1]):
        if t % pianorolls[0].frames_per_bar == 0:
            img[:, t * size_factor] += 20

    # inverse y
    img = np.flip(img, axis=0)

    fig, ax = create_fig_with_size(img.shape[1], img.shape[0])
    ax.imshow(img, vmin=0, vmax=127)
    for t, text in annotations:
        ax.text(t * size_factor+3, 3, text, ha='left', va='top', fontsize=12)
    fig.savefig(path)
    plt.close()

def log_image(img: torch.Tensor, name: str, step: int, save_dir: Path = Path("")):
    """
    image shape: (c, h, w) or (h, w)
    """
    save_dir = Path(wandb.run.dir) / save_dir  # type: ignore
    save_dir.mkdir(exist_ok=True)
    img_path = save_dir / f"{name}_{step:08}.jpg"
    if img.numel() == 0:
        img = torch.zeros(16, 16)
    save_img(img, img_path)
    image = wandb.Image(str(img_path))
    wandb.log({name: image}, step=step)
    return image

def log_pianoroll(
    pianoroll: Pianoroll | list[Pianoroll],
    name: str,
    step: int,
    save_dir: Path = Path(""),
    annotations: list[tuple[int, str]] = [],
    format: str = "jpg",
):
    save_dir = Path(wandb.run.dir) / save_dir  # type: ignore
    save_dir.mkdir(exist_ok=True)
    img_path = save_dir / f"{name}_{step:08}.{format}"
    if isinstance(pianoroll, Pianoroll):
        pianoroll = [pianoroll]
    pianorolls_to_img(pianoroll, img_path, annotations=annotations)
    image = wandb.Image(str(img_path))
    wandb.log({name: image}, step=step)
    return image

def log_midi_as_audio(
    midi: MidiFile, name: str, step: int, save_dir: Path = Path(""), soundfont_path: Path | None = None
):
    if soundfont_path is None:
        soundfont_path = Path(os.getenv("SOUNDFONT_PATH", "./ignore/FluidR3_GM.sf2"))
    assert soundfont_path.exists(), (
        f"Soundfont path {soundfont_path} does not exist. Please set the SOUNDFONT_PATH environment variable to a valid path."
    )
    save_dir = Path(wandb.run.dir) / save_dir  # type: ignore
    save_dir.mkdir(exist_ok=True)
    midi_path = save_dir / "midi" / f"{name}/{step:08}.mid"
    os.makedirs(midi_path.parent, exist_ok=True)
    midi.dump(str(midi_path))

    wav_path = save_dir / "wav" / f"{name}/{step:08}.wav"
    mp3_path = save_dir / "mp3" / f"{name}/{step:08}.mp3"
    os.makedirs(wav_path.parent, exist_ok=True)
    os.makedirs(mp3_path.parent, exist_ok=True)
    run_command(f'fluidsynth -F "{wav_path}" "{soundfont_path}" "{midi_path}"', output_level="none")
    run_command(f'ffmpeg -y -i "{wav_path}" -b:a 192k "{mp3_path}"', output_level="none")
    # delete the wav file
    wav_path.unlink()
    audio = wandb.Audio(str(mp3_path), caption=f"{name} {step}")
    wandb.log({name: audio}, step=step)
    return audio
