from functools import partial
from pathlib import Path
from typing import cast

import music_data_analysis
import torch
import torch.utils.data

from vqpiano.data.dataset import (
    LatentDiffFullSongPianorollDataset,
    Pop80kDataset,
    collate_midi_data,
)
from vqpiano.data.pianoroll_dataset import PianorollDataset
from vqpiano.data.segment import (
    create_testing_dataset as create_segment_testing_dataset,
)
from vqpiano.data.segment import (
    create_training_dataset as create_segment_training_dataset,
)
from vqpiano.models.representation import SymbolicRepresentation
from vqpiano.utils.chord import chord_to_chroma
from vqpiano.utils.torch_utils.tensor_op import pad_and_stack


def pr_dataset_collate_fn(
    batch,
    need_tokens=False,
    need_end_token=False,
    max_tokens: int | None = None,
    max_tokens_rate: float | None = None,
):
    prop_names = batch[0].keys()
    result = {}
    for prop_name in prop_names:
        prop_list = [item[prop_name] for item in batch]
        if isinstance(prop_list[0], dict):
            result[prop_name] = pr_dataset_collate_fn(
                prop_list,
                need_tokens=need_tokens,
                need_end_token=need_end_token,
                max_tokens=max_tokens,
                max_tokens_rate=max_tokens_rate,
            )
        elif isinstance(prop_list[0], music_data_analysis.Pianoroll):
            result[prop_name] = prop_list
            if need_tokens:
                result["tokens"] = SymbolicRepresentation.from_pianorolls(
                    prop_list,
                    need_end_token=need_end_token,
                    max_tokens=max_tokens,
                    max_tokens_rate=max_tokens_rate,
                )
        elif isinstance(prop_list[0], SymbolicRepresentation):
            result[prop_name] = SymbolicRepresentation.cat_batch(prop_list)
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


# def dataloader_factory(dataset_config, batch_size, num_workers, fs, seg_len, pitch_range):
def dataloader_factory(dataset_config, dataloader_config, model_config, model=None):
    if dataset_config["name"] == "pop1k7":
        # dataset = Pop1k7Dataset(dataset_config["path"], dataset_config, partition="train")
        # return torch.utils.data.DataLoader(
        #     dataset,
        #     batch_size=kwargs["batch_size"],
        #     num_workers=kwargs["num_workers"],
        #     pin_memory=False,
        #     shuffle=True,
        #     drop_last=True,
        #     persistent_workers=kwargs["num_workers"] > 0,
        #     collate_fn=partial(
        #         collate_midi_data,
        #         fs=kwargs["fs"],
        #         frame_width=kwargs["frame_width"],
        #         pitch_range=kwargs["pitch_range"],
        #         tokenizer=kwargs["tokenizer"],
        #     ),
        # )
        raise NotImplementedError
    elif dataset_config["name"] == "pop80k":
        dataset = Pop80kDataset(
            Path(dataset_config.path),
            resolution=dataset_config.pianoroll_resolution,
            min_duration=dataset_config.min_duration,
            max_duration=dataset_config.max_duration,
            min_note_count=dataset_config.min_note_count,
            sampling_frequency=model_config.model.sampling_frequency,
            pitch_range=model_config.model.pitch_range,
            frame_width=model_config.model.encoder.in_size[1],
            tokenizer=model.tokenizer,
        )
        collate_fn = partial(
            collate_midi_data,
            frame_width=model_config.model.encoder.in_size[1],
            tokenizer=model.tokenizer,
            max_seq_len=model_config.model.decoder.midi.max_seq_len,
        )

        return torch.utils.data.DataLoader(
            dataset,
            collate_fn=collate_fn,
            **dataloader_config,
        )

    elif dataset_config["name"] == "pop80k_lm":
        # dataset = Pop80kDataset(Path(dataset_config["path"]), dataset_config)
        # return torch.utils.data.DataLoader(
        #     dataset,
        #     batch_size=kwargs["batch_size"],
        #     num_workers=kwargs["num_workers"],
        #     pin_memory=False,
        #     shuffle=True,
        #     drop_last=True,
        #     persistent_workers=kwargs["num_workers"] > 0,
        #     collate_fn=partial(
        #         collate_midi_data_lm,
        #         max_seq_len=kwargs["max_seq_len"],
        #         fs=kwargs["fs"],
        #         frame_width=kwargs["frame_width"],
        #         vq_dim=kwargs["vq_dim"],
        #         pitch_range=kwargs["pitch_range"],
        #     ),
        # )
        raise NotImplementedError
    elif dataset_config.name == "pr":
        collate_fn = partial(
            pr_dataset_collate_fn,
            need_tokens=True,
            need_end_token=True,
            max_tokens=model_config.model.max_seq_length,
        )

        def transform(segment: music_data_analysis.SongSegment):
            return {"pianoroll": segment.read_pianoroll("pianoroll")}

        dataset = PianorollDataset(
            Path(dataset_config.path),
            frames_per_beat=8,
            length=model_config.model.duration,
            transform=transform,
            min_end_overlap=model_config.model.duration,
        )

        return torch.utils.data.DataLoader(
            dataset,
            collate_fn=collate_fn,
            **dataloader_config,
        )

    elif dataset_config.name == "pr_with_features":
        collate_fn = partial(
            pr_dataset_collate_fn,
            need_tokens=True,
            need_end_token=True,
        )

        def transform(segment: music_data_analysis.SongSegment):
            def normalize(x: list[float], mean: float, std: float):
                return [(x - mean) / std for x in x]

            result = {}
            result["pianoroll"] = segment.read_pianoroll("pianoroll")
            features = []
            features.append(normalize(segment.read_json("velocity"), 64, 24))  # n_bars
            features.append(normalize(segment.read_json("density"), 15, 8))  # n_bars
            features.append(normalize(segment.read_json("polyphony"), 3, 1.5))  # n_bars
            pitch = cast(dict, segment.read_json("pitch"))
            features.append(normalize(pitch["low"], 64, 24))  # n_bars
            features.append(normalize(pitch["high"], 64, 24))  # n_bars
            chords = cast(dict, segment.read_json("chords"))

            features = torch.tensor(features)  # d, n_bars
            features = features.T  # n_bars, d

            chord_tensor = torch.tensor(
                [
                    chord_to_chroma(quality, root)
                    for quality, root in zip(chords["quality"], chords["root"])
                ]
            )  # n_bars*4, 12
            chord_tensor = chord_tensor.view(-1, 4 * 12)  # n_bars, 48
            result["features"] = torch.cat([features, chord_tensor], dim=1)  # n_bars, d
            return result

        dataset = PianorollDataset(
            Path(dataset_config.path),
            frames_per_beat=8,
            length=model_config.model.prompt_duration
            + model_config.model.target_duration,
            transform=transform,
        )

        return torch.utils.data.DataLoader(
            dataset,
            collate_fn=collate_fn,
            **dataloader_config,
        )

    elif dataset_config.name == "latent_diff":
        dataset = LatentDiffFullSongPianorollDataset(
            Path(dataset_config.path),
            frames_per_beat=8,
            props=["n_bars", dataset_config.latent_prop],
            max_duration=dataset_config.max_duration,
        )

        return torch.utils.data.DataLoader(
            dataset,
            collate_fn=pr_dataset_collate_fn,
            **dataloader_config,
        )

    elif dataset_config["name"] == "segment_full_song":
        dataset = create_segment_training_dataset(
            Path(dataset_config.path),
            max_context_duration=model_config.model.max_context_duration,
            max_duration=dataset_config.max_duration,
            # max_tokens_rate=model_config.model.max_tokens_rate,
            max_tokens=model_config.model.max_tokens,
            bar_embedding_prop=dataset_config.bar_embedding_prop,
        )

        train_loader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=pr_dataset_collate_fn,
            **dataloader_config,
        )

        test_dataset = create_segment_testing_dataset(
            Path(dataset_config.path),
            max_duration=dataset_config.max_duration,
        )

        return train_loader, test_dataset

    elif dataset_config["name"] == "arithmetic":
        # dataset = BasicArithmeticOperationsDataset(max_num=dataset_config.max_number)
        # return torch.utils.data.DataLoader(
        #     dataset,
        #     batch_size=model_config.training.batch_size if not args.test else 1,
        #     num_workers=args.num_workers if not args.test else 0,
        #     pin_memory=args.pin_memory,
        #     shuffle=model_config.training.shuffle,
        #     drop_last=model_config.training.drop_last,
        #     persistent_workers=args.persistent_workers,
        #     collate_fn=partial(
        #         collate_arithmatic_data,
        #         max_seq_len=model_config.max_seq_len,
        #     ),
        # )
        raise NotImplementedError
    else:
        raise ValueError(f"Unknown dataset name: {dataset_config['name']}")
