import argparse
import multiprocessing
from pathlib import Path

import lightning as LT
from lightning.pytorch.callbacks import ModelCheckpoint
from loguru import logger

import wandb
from vqpiano.data.factory import dataloader_factory
from vqpiano.env import PROJECT_NAME
from vqpiano.models.factory import model_factory
from vqpiano.models.utils import load_ckpt_state_dict
from vqpiano.training.factory import (
    demo_callback_factory,
    training_wrapper_factory,
)
from vqpiano.training.utils import (
    ExceptionCallback,
    ModelConfigEmbedderCallback,
    copy_state_dict,
    get_training_strategy,
    init_env,
    setup_wandb,
)


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=PROJECT_NAME)
    parser.add_argument("--wandb_group", "-g", type=str, default=None)
    parser.add_argument("--task", type=str)
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


def main_simple_ar(args):
    model_config, dataset_config, dataloader_config = init_env(args)

    logger.info("Creating model")
    model = model_factory(model_config.model)

    logger.info("Creating dataloader")
    train_dl = dataloader_factory(
        dataset_config,
        dataloader_config,
        model_config,
        model,
    )

    logger.info("Creating training wrapper")
    training_wrapper = training_wrapper_factory(model_config, model)

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
    demo_callback = demo_callback_factory(
        model_config,
        dataset_config,
        save_dir=save_dir / "demo",
    )

    trainer = LT.Trainer(
        devices=args.num_gpus,
        accelerator="gpu",
        strategy=get_training_strategy(args),
        precision=model_config.training.precision,
        accumulate_grad_batches=model_config.training.accum_batches,
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


def main_segment_full_song(args):
    model_config, dataset_config, dataloader_config = init_env(args)

    logger.info("Creating model")
    model = model_factory(model_config.model)

    logger.info("Creating dataloader")
    train_dl, test_dl = dataloader_factory(
        dataset_config,
        dataloader_config,
        model_config,
        model,
    )

    logger.info("Creating training wrapper")
    training_wrapper = training_wrapper_factory(model_config, model)

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
    demo_callback = demo_callback_factory(
        model_config,
        dataset_config,
        save_dir=save_dir / "demo",
        test_dl=test_dl,
    )

    trainer = LT.Trainer(
        devices=args.num_gpus,
        accelerator="gpu",
        strategy=get_training_strategy(args),
        precision=model_config.training.precision,
        # accumulate_grad_batches=model_config.training.accum_batches, # this is done in the training wrapper because manual optimization is used
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


def main_vae(args):
    model_config, dataset_config, dataloader_config = init_env(args)

    logger.info("Creating model")
    model = model_factory(model_config.model)

    logger.info("Creating dataloader")
    train_dl = dataloader_factory(
        dataset_config,
        dataloader_config,
        model_config,
        model,
    )

    if args.pretrained_ckpt_path:
        copy_state_dict(model, load_ckpt_state_dict(args.pretrained_ckpt_path))

    logger.info("Creating training wrapper")
    training_wrapper = training_wrapper_factory(model_config, model)

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
    demo_callback = demo_callback_factory(
        model_config,
        dataset_config,
        save_dir=save_dir / "demo",
    )

    trainer = LT.Trainer(
        devices=args.num_gpus,
        accelerator="gpu",
        strategy=get_training_strategy(args),
        precision=model_config.training.precision,
        accumulate_grad_batches=model_config.training.accum_batches,
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
    args = parser_args()
    if args.task == "vae":
        main_vae(args)
    elif args.task == "simple_ar":
        main_simple_ar(args)
    elif args.task == "segment_full_song":
        main_segment_full_song(args)
    else:
        raise ValueError(f"Unknown task: {args.task}")
