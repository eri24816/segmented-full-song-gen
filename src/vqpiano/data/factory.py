from pathlib import Path

import music_data_analysis
import torch
import torch.utils.data

from vqpiano.data.dataset import PianorollDataset
from vqpiano.data.segment import (
    create_testing_dataset as create_segment_testing_dataset,
)
from vqpiano.data.segment import (
    create_training_dataset as create_segment_training_dataset,
)
from vqpiano.models.token_sequence import TokenSequence
from vqpiano.utils.torch_utils.tensor_op import pad_and_stack


def collate_fn(batch):
    prop_names = batch[0].keys()
    result = {}
    for prop_name in prop_names:
        prop_list = [item[prop_name] for item in batch]
        if isinstance(prop_list[0], dict):
            result[prop_name] = collate_fn(prop_list)
        elif isinstance(prop_list[0], music_data_analysis.Pianoroll):
            result[prop_name] = prop_list
        elif isinstance(prop_list[0], TokenSequence):
            result[prop_name] = TokenSequence.cat_batch(prop_list)
        elif isinstance(prop_list[0], int | float | list | torch.Tensor):
            if not isinstance(prop_list[0], torch.Tensor):
                prop_list = [torch.tensor(prop) for prop in prop_list]
            if prop_list[0].ndim >= 1:
                result[prop_name] = pad_and_stack(prop_list, pad_dim=0)
            else:
                result[prop_name] = torch.stack(prop_list)
        else:
            raise ValueError(
                f"Unknown prop type: {type(prop_list[0])}, prop_name: {prop_name}"
            )
    return result


def tokens_transform(
    segment: music_data_analysis.SongSegment, max_tokens: int, max_note_duration: int
):
    pr = segment.read_pianoroll("pianoroll")
    tokens = TokenSequence.from_pianorolls(
        [pr],
        need_end_token=True,
        max_tokens=max_tokens,
        max_note_duration=max_note_duration,
    )
    return {"tokens": tokens, "pianoroll": pr}


def create_dataloader(dataset_config, dataloader_config, model_config):
    if dataset_config.name == "tokens":
        train_ds = PianorollDataset(
            Path(dataset_config.path),
            frames_per_beat=8,
            length=model_config.model.duration,
            transform=tokens_transform,
            transform_kwargs={
                "max_tokens": model_config.model.max_tokens,
                "max_note_duration": model_config.model.max_note_duration,
            },
            min_end_overlap=model_config.model.duration,
            split="train",
            train_set_ratio=0.9,
        )

        train_dl = torch.utils.data.DataLoader(
            train_ds,
            collate_fn=collate_fn,
            **dataloader_config,
        )

        if hasattr(model_config.model, "test_duration"):
            test_ds = PianorollDataset(
                Path(dataset_config.path),
                frames_per_beat=8,
                length=model_config.model.test_duration,
                transform=tokens_transform,
                transform_kwargs={
                    "max_tokens": 100000000,
                    "max_note_duration": model_config.model.max_note_duration,
                },
                split="test",
                train_set_ratio=0.9,
            )
        else:
            test_ds = None

        return train_dl, test_ds

    elif dataset_config["name"] == "segment_full_song":
        train_ds = create_segment_training_dataset(
            Path(dataset_config.path),
            max_context_duration=model_config.model.max_context_duration,
            min_duration=dataset_config.min_duration,
            max_duration=dataset_config.max_duration,
            max_tokens=model_config.model.max_tokens,
            max_note_duration=model_config.model.max_note_duration,
            bar_embedding_prop=dataset_config.bar_embedding_prop,
        )

        train_loader = torch.utils.data.DataLoader(
            train_ds,
            collate_fn=collate_fn,
            **dataloader_config,
        )

        test_dataset = create_segment_testing_dataset(
            Path(dataset_config.path),
            min_duration=dataset_config.min_duration,
            max_duration=dataset_config.max_duration,
        )

        return train_loader, test_dataset
    else:
        raise ValueError(f"Unknown dataset name: {dataset_config['name']}")
