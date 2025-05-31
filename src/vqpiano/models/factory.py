from typing import Callable, Tuple, cast

from omegaconf import OmegaConf
from torch import Tensor

from vqpiano.models.encoder_decoder import (
    BottleneckLoss,
    EncoderDecoder,
    VAEBottleneck,
    identity_bottleneck,
)
from vqpiano.models.feature_extractor import FeatureExtractor
from vqpiano.models.token_generator import TokenGenerator
from vqpiano.models.segment_full_song import SegmentFullSongModel


def create_model(config):
    if config.type == "simple_ar":
        model = TokenGenerator(
            dim=config.dim,
            num_layers=config.num_layers,
            pitch_range=config.pitch_range,
            num_pos=config.duration,
            max_note_duration=config.max_note_duration,
        )
        return model

    elif config.type == "segment_full_song":
        encoder_decoder = cast(
            EncoderDecoder,
            create_model(OmegaConf.load("config/simple_ar/model_vae.yaml").model),
        )
        from safetensors.torch import load_file

        encoder_decoder.load_state_dict(
            load_file(
                "wandb/run-20250404_013005-i41ffa2m/files/checkpoints/epoch=4-step=1000000.safetensors",
                device="cuda",
            )
        )

        model = SegmentFullSongModel(
            bar_embedder=encoder_decoder,
            dim=config.dim,
            pitch_range=config.pitch_range,
            encoder_num_layers=config.encoder_num_layers,
            decoder_num_layers=config.decoder_num_layers,
            max_forward_duration=config.max_forward_duration,
            max_song_duration=config.max_song_duration,
            max_context_duration=config.max_context_duration,
            max_tokens_rate=config.max_tokens_rate,
            latent_dim=config.latent_dim,
            frames_per_bar=config.frames_per_bar,
        )
        return model
    elif config.type == "vae":
        encoder = FeatureExtractor(
            dim=config.encoder.dim,
            num_layers=config.encoder.num_layers,
            pitch_range=config.pitch_range,
            num_pos=config.duration,
            is_causal=False,
            max_note_duration=config.max_note_duration,
        )
        decoder = TokenGenerator(
            dim=config.decoder.dim,
            num_layers=config.decoder.num_layers,
            pitch_range=config.pitch_range,
            num_pos=config.duration,
            condition_dim=config.bottleneck.vae_params.latent_dim,
            max_note_duration=config.max_note_duration,
        )

        if config.bottleneck.type == "vae":
            bottleneck: Callable[[Tensor], Tuple[Tensor, BottleneckLoss]]
            assert config.bottleneck.vae_params is not None, (
                "vae_params must be provided if bottleneck type is 'vae'"
            )
            bottleneck = VAEBottleneck(
                input_dim=config.encoder.dim,
                output_dim=config.bottleneck.vae_params.latent_dim,
                beta=config.bottleneck.vae_params.beta,
                beta_cycle_steps=config.bottleneck.vae_params.beta_cycle_steps,
                beta_start_step=config.bottleneck.vae_params.beta_start_step,
            )
        else:
            bottleneck = identity_bottleneck

        return EncoderDecoder(
            encoder=encoder,
            decoder=decoder,
            bottleneck=bottleneck,
            duration=config.duration,
            max_tokens=config.max_tokens,
            pitch_range=config.pitch_range,
        )
    else:
        raise ValueError(f"Unknown model type: {config['type']}")
