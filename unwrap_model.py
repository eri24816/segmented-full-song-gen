import argparse
import os
from pathlib import Path

import torch

from segment_full_song.models.factory import create_model
from segment_full_song.training.factory import create_training_wrapper


def export_model(ckpt_path: Path, save_dir: Path, prefix: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_config = ckpt["model_config"]

    model = create_model(model_config.model)
    training_wrapper = create_training_wrapper(model_config, model)
    training_wrapper.load_state_dict(ckpt["state_dict"])
    training_wrapper.export_model(save_dir, prefix)


if __name__ == "__main__":
    # TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    parser = argparse.ArgumentParser(
        description="Export Lightning model to Torch model"
    )
    parser.add_argument(
        "ckpt_path",
        type=Path,
        help="Path to the checkpoint file of the Lightning model",
    )
    args = parser.parse_args()

    export_model(
        args.ckpt_path, save_dir=args.ckpt_path.parent, prefix=args.ckpt_path.stem
    )
