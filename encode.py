from pathlib import Path
from omegaconf import OmegaConf
import safetensors.torch
import torch
from segment_full_song.data.dataset import FullSongPianorollDataset
from segment_full_song.models.factory import create_model

from segment_full_song.models.encoder_decoder import EncoderDecoder
from segment_full_song.models.token_sequence import TokenSequence

import argparse
import loguru
from tqdm import tqdm

# python encode.py --model_config config/model_token.yaml --dataset_config config/dataset_pop80k_k_pr.yaml --ckpt_path wandb/run-20250404_013005-i41ffa2m/files/checkpoints/epoch=1-step=600000.ckpt --output_name latent_i41ffa2m_600k


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_config", type=Path)
    parser.add_argument("dataset_config", type=Path)
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--output_name", type=str, default="latent")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model_config = OmegaConf.load(args.model_config)
    dataset_config = OmegaConf.load(args.dataset_config)

    model: EncoderDecoder = create_model(model_config.model)  # type: ignore

    model.load_state_dict(safetensors.torch.load_file(args.ckpt_path))

    ds = FullSongPianorollDataset(Path(dataset_config.path))

    model.eval().cuda()

    output_dir = Path(dataset_config.path) / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    loguru.logger.info(f"Encoding {len(ds)} songs")

    max_batch_size = 300
    for song in tqdm(ds, desc="Encoding"):
        pr = song.read_pianoroll("pianoroll")
        duration = song.read_json("duration")
        if pr.duration == int(duration / 64 * pr.frames_per_beat):
            # print("skip")
            continue
        pr.duration = int(duration / 64 * pr.frames_per_beat)
        bars = list(pr.iter_over_bars_pr())

        result_batches = []
        for i in range(0, len(bars), max_batch_size):
            batch = bars[i : i + max_batch_size]
            repr = TokenSequence.from_pianorolls(
                batch,
                device=torch.device("cuda"),
                max_note_duration=model_config.model.max_note_duration,
            )
            with torch.no_grad():
                latent = model.encode(repr)
            result_batches.append(latent)

        latent = torch.cat(result_batches, dim=0)

        assert latent.shape[0] == len(bars)

        output_path = output_dir / f"{pr.metadata.name}.pt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        latent = latent.to(torch.float16)
        torch.save(latent, output_path)
