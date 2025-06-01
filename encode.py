from pathlib import Path
from omegaconf import OmegaConf
import torch
from segment_full_song.data.dataset import FullSongPianorollDataset
from segment_full_song.models.factory import create_model

from segment_full_song.models.encoder_decoder import EncoderDecoder
from segment_full_song.models.token_sequence import TokenSequence
from segment_full_song.models.utils import load_ckpt_state_dict

import argparse
import loguru
from tqdm import tqdm

# python encode.py --model_config config/model_token.yaml --dataset_config config/dataset_pop80k_k_pr.yaml --ckpt_path wandb/run-20250404_013005-i41ffa2m/files/checkpoints/epoch=1-step=600000.ckpt --output_name latent_i41ffa2m_600k


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=Path, required=True)
    parser.add_argument("--dataset_config", type=Path, required=True)
    parser.add_argument("--ckpt_path", type=Path, required=True)
    parser.add_argument("--output_name", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model_config = OmegaConf.load(args.model_config)
    dataset_config = OmegaConf.load(args.dataset_config)

    model: EncoderDecoder = create_model(model_config.model)  # type: ignore

    model.load_state_dict(
        load_ckpt_state_dict(
            args.ckpt_path,
            unwrap_lightning=True,
        )
    )
    ds = FullSongPianorollDataset(Path(dataset_config.path), props=["pianoroll"])
    model.eval().cuda()

    output_dir = Path(dataset_config.path) / args.output_name

    loguru.logger.info(f"Encoding {len(ds)} songs")

    max_batch_size = 5000
    for song in tqdm(ds, desc="Encoding"):
        pr = song["pianoroll"]
        bars = list(pr.iter_over_bars_pr())

        result_batches = []
        for i in range(0, len(bars), max_batch_size):
            batch = bars[i : i + max_batch_size]
            repr = TokenSequence.from_pianorolls(batch, device=torch.device("cuda"))
            with torch.no_grad():
                latent = model.encode(repr)
            result_batches.append(latent)

        latent = torch.cat(result_batches, dim=0)

        assert latent.shape[0] == len(bars)

        output_path = output_dir / f"{pr.metadata.name}.pt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        latent = latent.to(torch.float16)
        torch.save(latent, output_path)
