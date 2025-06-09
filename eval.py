from typing import Any
from segment_full_song.config_types import ModelAndTrainingParams
from segment_full_song.data.factory import (
    SimpleARDatasetParams,
    create_dataloader_simple_ar_reconstruct,
)
from segment_full_song.evaluate import evaluate_simple_ar
from segment_full_song.models.encoder_decoder import EncoderDecoder
from segment_full_song.models.utils import load_ckpt_state_dict
from segment_full_song.utils.env import init_env
import argparse
from pathlib import Path


def main_simple_ar(args):
    params: ModelAndTrainingParams[EncoderDecoder.Params, Any]
    dataset_params: SimpleARDatasetParams
    params, dataset_params = init_env(args)
    model = EncoderDecoder(params.model_params)

    if args.pretrained_ckpt_path:
        state_dict = load_ckpt_state_dict(args.pretrained_ckpt_path)
        new_state_dict = {}
        for key, value in state_dict.items():
            assert key.startswith("model."), f"Key {key} does not start with 'model.'"
            new_key = key[len("model.") :]
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict)

    dl = create_dataloader_simple_ar_reconstruct(dataset_params, 32 * 16)
    evaluate_simple_ar(model, dl, args.num_samples, args.out_dir, device=args.device)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_config", type=Path, required=True)
    parser.add_argument("dataset_config", type=Path, required=True)
    parser.add_argument("--pretrained_ckpt_path", type=Path, required=True)
    parser.add_argument("--task", type=str)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--matmul_precision", type=str, default="high")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--out_dir", type=Path, default=Path("out"))
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.task == "simple_ar":
        main_simple_ar(args)
    else:
        raise ValueError(f"Invalid task: {args.task}")
