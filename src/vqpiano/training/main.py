import argparse
import multiprocessing
from pathlib import Path

import lightning as LT
from lightning.pytorch.callbacks import ModelCheckpoint
from loguru import logger

import wandb
from vqpiano.data.factory import create_dataloader
from vqpiano.env import PROJECT_NAME
from vqpiano.models.factory import create_model
from vqpiano.training.factory import (
    create_demo_callback,
    create_training_wrapper,
)
from vqpiano.training.utils import (
    ExceptionCallback,
    ModelConfigEmbedderCallback,
    get_training_strategy,
    init_env,
    setup_wandb,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=PROJECT_NAME)
    parser.add_argument("--wandb_group", "-g", type=str, default=None)
    parser.add_argument("--model_config", type=Path, required=True)
    parser.add_argument("--dataset_config", type=Path, required=True)
    parser.add_argument("--save_dir", type=Path, required=True)

    parser.add_argument(
        "--num_workers", type=int, default=multiprocessing.cpu_count() // 2
    )
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--matmul_precision", type=str, default="high")
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--persistent_workers", type=bool, default=False)

    parser.add_argument("--pretrained_ckpt_path", type=Path, default=None)
    parser.add_argument("--ckpt_path", type=Path, default=None)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--profile", "-p", action="store_true")

    return parser.parse_args()

def main(args):
    model_config, dataset_config, dataloader_config = init_env(args)

    model = create_model(model_config.model)

    train_dl, test_ds = create_dataloader(
        dataset_config,
        dataloader_config,
        model_config,
    )

    training_wrapper = create_training_wrapper(model_config, model)

    wandb_logger, save_dir = setup_wandb(
        args, training_wrapper, model_config, dataset_config
    )

    exc_callback = ExceptionCallback()

    ckpt_callback = ModelCheckpoint(
        every_n_train_steps=model_config.training.checkpoint_steps,
        dirpath=save_dir / "checkpoints",
        save_top_k=-1,
    )

    save_model_config_callback = ModelConfigEmbedderCallback(model_config)

    demo_callback = create_demo_callback(
        model_config,
        test_ds=test_ds,
    )

    if model_config.model.type == "segment_full_song":
        # accumulate_grad_batches is done in the training wrapper because manual optimization is used
        accumulate_grad_batches = 1
    else:
        accumulate_grad_batches = model_config.training.accum_batches

    trainer = LT.Trainer(
        devices=args.num_gpus,
        accelerator="gpu",
        strategy=get_training_strategy(args),
        precision=model_config.training.precision,
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=[
            ckpt_callback,
            demo_callback,
            exc_callback,
            save_model_config_callback,
        ],
        logger=wandb_logger,
        log_every_n_steps=1,
        max_steps=model_config.training.max_steps,
        default_root_dir=args.save_dir,
        gradient_clip_val=model_config.training.gradient_clip_val,
        reload_dataloaders_every_n_epochs=0,
        fast_dev_run=args.test,
        num_sanity_val_steps=1,
    )

    logger.info("Starting training")
    trainer.fit(training_wrapper, train_dl, ckpt_path=args.ckpt_path)

    if wandb_logger:
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
