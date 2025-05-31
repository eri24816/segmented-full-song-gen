from torch.utils.data import DataLoader, Dataset

from vqpiano.training.segment_full_song import (
    SegmentFullSongDemoCallback,
    SegmentFullSongTrainingWrapper,
)
from vqpiano.training.simple_ar import (
    SimpleARDemoCallback,
    SimpleARTrainingWrapper,
)
from vqpiano.training.vae import (
    VAEDemoCallback,
    VAETrainingWrapper,
)


def training_wrapper_factory(model_config, model):
    if model_config.type == "simple_ar":
        wrapper = SimpleARTrainingWrapper(
            model,
            pitch_range=model_config.model.pitch_range,
            lr=model_config.training.lr,
            betas=model_config.training.betas,
            eps=model_config.training.eps,
            weight_decay=model_config.training.weight_decay,
        )

    elif model_config.type == "segment_full_song":
        wrapper = SegmentFullSongTrainingWrapper(
            model,
            pitch_range=model_config.model.pitch_range,
            lr=model_config.training.lr,
            accum_batches=model_config.training.accum_batches,
        )
    elif model_config.type == "vae":
        wrapper = VAETrainingWrapper(
            model,
            max_tokens_prompt=model_config.model.max_tokens_prompt,
            max_tokens_target=model_config.model.max_tokens_target,
            pitch_range=model_config.model.pitch_range,
            lr=model_config.training.lr,
        )

    else:
        raise ValueError(f"Unknown model type: {model_config['type']}")
    return wrapper


def demo_callback_factory(
    model_config, dataset_config, save_dir, test_dl: DataLoader | Dataset | None = None
):
    if model_config.type == "simple_ar":
        callback = SimpleARDemoCallback(
            demo_every=model_config.training.demo_steps,
            duration=model_config.model.duration,
        )
    elif model_config.type == "segment_full_song":
        callback = SegmentFullSongDemoCallback(
            demo_every=model_config.training.demo_steps,
            test_dl=test_dl,
            max_context_duration=model_config.model.max_context_duration,
            # max_tokens_rate=model_config.model.max_tokens_rate,
            max_tokens=model_config.model.max_tokens,
        )
    elif model_config.type == "vae":
        callback = VAEDemoCallback(
            demo_every=model_config.training.demo_steps,
            dataset_path=dataset_config.path,
        )
    else:
        raise ValueError(f"Unknown model type: {model_config['type']}")
    return callback
