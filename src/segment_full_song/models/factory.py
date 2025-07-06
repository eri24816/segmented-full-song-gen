from typing import Callable, Tuple, cast

from omegaconf import OmegaConf
from torch import Tensor

from segment_full_song.models.encoder_decoder import (
    BottleneckLoss,
    EncoderDecoder,
    VAEBottleneck,
    identity_bottleneck,
)
from segment_full_song.models.feature_extractor import FeatureExtractor
from segment_full_song.models.token_generator import TokenGenerator
from segment_full_song.models.segment_full_song import SegmentFullSongModel


def create_model(config):
    if isinstance(config, str):
        config = OmegaConf.load(config).model

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
        bar_embedder = cast(
            EncoderDecoder,
            create_model(config.bar_embedder),
        )

        model = SegmentFullSongModel(
            bar_embedder=bar_embedder,
            dim=config.dim,
            pitch_range=config.pitch_range,
            max_note_duration=config.max_note_duration,
            encoder_num_layers=config.encoder_num_layers,
            decoder_num_layers=config.decoder_num_layers,
            max_forward_duration=config.max_forward_duration,
            max_song_duration=config.max_song_duration,
            max_context_duration=config.max_context_duration,
            max_tokens=config.max_tokens,
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
