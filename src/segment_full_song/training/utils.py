import random
import sys
from datetime import datetime
from pathlib import Path

import dotenv
import lightning as LT
import numpy as np
import torch
from lightning.pytorch.loggers.wandb import WandbLogger
from loguru import logger
from omegaconf import OmegaConf
import matplotlib
from segment_full_song.env import TEST_BATCH_SIZE, TEST_NUM_WORKERS


class ExceptionCallback(LT.Callback):
    def on_exception(self, trainer, pl_module, exception):
        print(f"{type(exception).__name__}: {exception}")


class ModelConfigEmbedderCallback(LT.Callback):
    def __init__(self, model_config):
        self.model_config = model_config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["model_config"] = self.model_config


def init_env(args):
    matplotlib.use("agg")
    dotenv.load_dotenv()

    model_config = OmegaConf.load(args.model_config)
    dataset_config = OmegaConf.load(args.dataset_config)

    dataloader_config = OmegaConf.create()
    dataloader_config.batch_size = (
        model_config.training.batch_size if not args.test else TEST_BATCH_SIZE
    )
    dataloader_config.num_workers = (
        args.num_workers if not args.test else TEST_NUM_WORKERS
    )
    dataloader_config.pin_memory = args.pin_memory
    dataloader_config.shuffle = model_config.training.shuffle
    dataloader_config.drop_last = model_config.training.drop_last
    if args.persistent_workers is not None:
        dataloader_config.persistent_workers = args.persistent_workers
    else:
        dataloader_config.persistent_workers = args.num_workers > 0

    if args.test:
        logger.remove()
        logger.add(sys.stderr, level="TRACE")

    if hasattr(args, "out_dir"):
        args.out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("args: {}", args)
    logger.info("model_config: {}", model_config)
    logger.info("dataset_config: {}", dataset_config)

    # Set seed
    seed = model_config.training.seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(0)

    torch.set_float32_matmul_precision(args.matmul_precision)

    return model_config, dataset_config, dataloader_config


def setup_wandb(args, training_wrapper, model_config, dataset_config):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"test_{current_time}" if args.test else current_time
    wandb_logger = WandbLogger(
        project=args.name,
        name=experiment_name,
        group=args.wandb_group if args.wandb_group else model_config.model.type,
        save_dir=args.save_dir,
    )
    wandb_logger.watch(training_wrapper)
    # find directory with wandb_logger.experiment.id in "wandb" directory
    experiment_dir = next(
        (Path(args.save_dir) / "wandb").glob(f"*{wandb_logger.experiment.id}*")
    )
    save_dir = experiment_dir / "files"
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Save dir: {}", experiment_dir)

    # Combine args and config dicts
    if wandb_logger:
        args_dict = vars(args)
        args_dict.update({"model_config": model_config})
        args_dict.update({"dataset_config": dataset_config})
        wandb_logger.experiment.config.update(args_dict)

    return wandb_logger, save_dir


def get_training_strategy(args):
    # Set multi-GPU strategy if specified
    if args.strategy:
        if args.strategy == "deepspeed":
            from lightning.pytorch.strategies import DeepSpeedStrategy

            strategy = DeepSpeedStrategy(
                stage=2,
                contiguous_gradients=True,
                overlap_comm=True,
                reduce_scatter=True,
                reduce_bucket_size=5 * (10**8),
                allgather_bucket_size=5 * (10**8),
                load_full_weights=True,
            )
        else:
            raise ValueError(f"Unknown strategy: {args.strategy}")
    else:
        from lightning.pytorch.strategies import DDPStrategy

        strategy = DDPStrategy() if args.num_gpus > 1 else "auto"

    return strategy


# def deprecated_load_model(model_config, ckpt_path=None, ckpt_target=None):
#     model = model_factory(model_config)
#     if ckpt_path:
#         copy_state_dict(model, load_ckpt_state_dict(ckpt_path), ckpt_target)
#     return model


def copy_state_dict(model, state_dict, target=None, strict=True):
    """
    Load state_dict to model, but only for keys that match exactly.
    If target is specified, Prepends target to keys in state_dict.

    Args:
        model (nn.Module): model to load state_dict.
        state_dict (OrderedDict): state_dict to load.
    """
    model_state_dict = model.state_dict()
    non_matched_keys = set(state_dict.keys())

    for key in state_dict:
        if target:
            tgt_key = f"{target}.{key}"
        else:
            tgt_key = key

        if (
            tgt_key in model_state_dict
            and state_dict[key].shape == model_state_dict[tgt_key].shape
        ):
            if isinstance(state_dict[key], torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                state_dict[key] = state_dict[key].data
            model_state_dict[tgt_key] = state_dict[key]
            non_matched_keys.remove(key)  # remove matched key

    if non_matched_keys:
        logger.warning(f"Keys not matched: {non_matched_keys}")
        # hold and ask user to continue
        response = input(
            "Some keys in the state_dict did not match the model's keys. Do you want to continue? (y/[n]): "
        )
        if response.lower() != "y":
            raise ValueError("Loading state_dict aborted by user.")
    model.load_state_dict(model_state_dict, strict=strict)
