from dataclasses import dataclass
from typing import Callable, Tuple

from music_data_analysis import Pianoroll
import torch
import torch.nn as nn
from torch import Tensor

from .feature_extractor import FeatureExtractor
from .token_generator import TokenGenerator
from .token_sequence import TokenSequence


@dataclass
class BottleneckLoss:
    total_loss: Tensor


class VAEBottleneck(nn.Module):
    @dataclass
    class Loss:
        total_loss: Tensor
        kl_loss: Tensor
        beta: float
        rms_mu: Tensor
        rms_std: Tensor

    def cyclic_beta(self, step: int, beta_cycle_steps: int, beta_start_step: int):
        if step < beta_start_step:
            return 0
        return min(
            1, 2 * ((step - beta_start_step) % beta_cycle_steps / beta_cycle_steps)
        )

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        beta: float,
        beta_cycle_steps: int,
        beta_start_step: int,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.beta = beta
        self.beta_cycle_steps = beta_cycle_steps
        self.beta_start_step = beta_start_step
        self.mu_proj = nn.Linear(input_dim, output_dim)
        self.logvar_proj = nn.Linear(input_dim, output_dim)
        self.step = 0

    def forward(self, x: Tensor, sample_latent: bool = True):
        """
        x: b, d
        """
        mu: Tensor = self.mu_proj(x)  # b, d
        logvar: Tensor = self.logvar_proj(x)  # b, d

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        kl_loss = (0.5 * (mu**2 + logvar.exp() - logvar - 1).sum(dim=-1)).mean()
        beta = (
            self.cyclic_beta(self.step, self.beta_cycle_steps, self.beta_start_step)
            * self.beta
        )
        total_loss = kl_loss * beta
        rms_mu = mu.norm(2, dim=-2).pow(2).mean().sqrt()
        rms_std = std.pow(2).mean().sqrt()

        if sample_latent:
            x = mu + eps * std
        else:
            x = mu

        return x, self.Loss(
            total_loss=total_loss,
            kl_loss=kl_loss,
            beta=beta,
            rms_mu=rms_mu,
            rms_std=rms_std,
        )


def identity_bottleneck(x: Tensor) -> Tuple[Tensor, BottleneckLoss]:
    return x, BottleneckLoss(total_loss=torch.tensor(0, device=x.device))


class EncoderDecoder(nn.Module):
    """
    input tokens: b, l
    output: b, d
    """

    @dataclass
    class Loss:
        total_loss: Tensor
        reconstruction: TokenGenerator.Loss
        bottleneck: BottleneckLoss

    def __init__(
        self,
        encoder,
        decoder,
        bottleneck,
        duration: int,
        max_tokens: int,
        pitch_range: list[int],
    ):
        super().__init__()

        self.duration = duration
        self.max_tokens = max_tokens
        self.pitch_range = pitch_range

        self.encoder: FeatureExtractor = encoder
        self.bottleneck: Callable[[Tensor], Tuple[Tensor, BottleneckLoss]] = bottleneck
        self.decoder: TokenGenerator = decoder

    def set_step(self, step: int):
        if isinstance(self.bottleneck, VAEBottleneck):
            self.bottleneck.step = step

    def forward(
        self,
        x: TokenSequence,
        handcrafted_latent: torch.Tensor | None = None,
    ):
        target_embed = self.encoder(x)  # b, d

        latent, bottleneck_loss = self.bottleneck(target_embed)  # b, d

        if handcrafted_latent is not None:
            latent = torch.cat([latent, handcrafted_latent], dim=-1)  # b, d + d_hard

        # then, predict target
        reconstruction_loss = self.decoder.forward(x=x, condition=latent)
        return EncoderDecoder.Loss(
            total_loss=reconstruction_loss.total_loss + bottleneck_loss.total_loss,
            reconstruction=reconstruction_loss,
            bottleneck=bottleneck_loss,
        )

    def encode(self, x: TokenSequence, sample_latent: bool = False):
        if isinstance(self.bottleneck, VAEBottleneck):
            latent, _ = self.bottleneck(self.encoder(x), sample_latent)
        else:
            latent, _ = self.bottleneck(self.encoder(x))

        return latent

    @torch.no_grad()
    def decode(self, latent: Tensor):
        return self.decoder.sample(duration=self.duration, condition=latent)

    @torch.no_grad()
    def reconstruct(
        self,
        x: TokenSequence,
        sample_latent: bool = True,
    ):
        latent = self.encode(x, sample_latent)
        return self.decode(latent)

    """
    Applications
    """

    @torch.no_grad()
    def encode_pianoroll(
        self,
        pianoroll: Pianoroll,
        max_batch_size: int = 100,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Input: pianoroll
        Output: n_bars, d
        """
        result_batches = []
        bars = list(pianoroll.iter_over_bars_pr())
        for i in range(0, len(bars), max_batch_size):
            batch = bars[i : i + max_batch_size]
            repr = TokenSequence.from_pianorolls(
                batch, device=device, max_note_duration=self.encoder.max_note_duration
            )
            with torch.no_grad():
                latent = self.encode(repr)
            result_batches.append(latent)

        latent = torch.cat(result_batches, dim=0)
        return latent