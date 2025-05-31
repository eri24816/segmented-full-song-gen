from dataclasses import dataclass
from typing import Callable, Tuple

from music_data_analysis import Pianoroll
import torch
import torch.nn as nn
from torch import Tensor

from .feature_extractor import FeatureExtractor
from .token_generator import TokenGenerator
from .representation import SymbolicRepresentation
from tqdm import tqdm


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
        target_duration: int,
        prompt_duration: int,
        max_tokens_target: int,
        max_tokens_prompt: int,
        pitch_range: list[int],
    ):
        super().__init__()

        self.target_duration = target_duration
        self.prompt_duration = prompt_duration
        self.max_tokens_target = max_tokens_target
        self.max_tokens_prompt = max_tokens_prompt
        self.pitch_range = pitch_range

        self.encoder: FeatureExtractor = encoder
        self.decoder: TokenGenerator = decoder
        self.bottleneck: Callable[[Tensor], Tuple[Tensor, BottleneckLoss]] = bottleneck

    def set_step(self, step: int):
        if isinstance(self.bottleneck, VAEBottleneck):
            self.bottleneck.step = step

    def forward(
        self,
        target: SymbolicRepresentation,
        prompt: SymbolicRepresentation,
        handcrafted_latent: torch.Tensor | None = None,
    ):
        target_embed = self.encoder(target)  # b, d

        latent, bottleneck_loss = self.bottleneck(target_embed)  # b, d

        if handcrafted_latent is not None:
            latent = torch.cat([latent, handcrafted_latent], dim=-1)  # b, d + d_hard

        # then, predict target
        reconstruction_loss = self.decoder.forward(
            x=target, prompt=prompt, condition=latent
        )
        return EncoderDecoder.Loss(
            total_loss=reconstruction_loss.total_loss + bottleneck_loss.total_loss,
            reconstruction=reconstruction_loss,
            bottleneck=bottleneck_loss,
        )

    @torch.no_grad()
    def reconstruct(
        self,
        target: SymbolicRepresentation,
        prompt: SymbolicRepresentation,
        sample_latent: bool = True,
    ):
        latent = self.encode(target, sample_latent)

        return self.decoder.sample(
            duration=prompt.duration + target.duration,
            prompt=prompt,
            condition=latent,
        )

    def encode(self, x: SymbolicRepresentation, sample_latent: bool = False):
        if isinstance(self.bottleneck, VAEBottleneck):
            latent, _ = self.bottleneck(self.encoder(x), sample_latent)
        else:
            latent, _ = self.bottleneck(self.encoder(x))

        return latent

    @torch.no_grad()
    def sample(self, prompt: SymbolicRepresentation, duration: int, latent: Tensor):
        return self.decoder.sample(
            duration=prompt.duration + duration, prompt=prompt, condition=latent
        )

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
            repr = SymbolicRepresentation.from_pianorolls(batch, device=device)
            with torch.no_grad():
                latent = self.encode(repr)
            result_batches.append(latent)

        latent = torch.cat(result_batches, dim=0)
        return latent

    @torch.no_grad()
    def decode_autoregressive(
        self,
        latents: torch.Tensor,
        n_prompt_bars: int,
        given_prompt_bars: list[SymbolicRepresentation] | None = None,
    ):
        """
        latents: bar, d
        if given_prompt_bars is None, the first iterations the model will receive empty bars as prompts. It will feel
        generating the beginning of the piece.

        To make the model generate bars from the middle of the piece, pass the previous bars as given_prompt_bars.
        """
        bars = []

        if given_prompt_bars is None:
            for i in range(n_prompt_bars):
                bar = SymbolicRepresentation(device=latents.device)
                for _ in range(32):
                    bar.add_frame()
                bars.append(bar)
        else:
            assert len(given_prompt_bars) == n_prompt_bars, (
                f"{len(given_prompt_bars)} != {n_prompt_bars}"
            )
            bars = given_prompt_bars.copy()

        for i, latent in enumerate(tqdm(latents, desc="Decoding...")):
            prompt = SymbolicRepresentation.cat_time(bars[i : i + n_prompt_bars])

            prediction = self.decoder.sample(
                duration=prompt.duration + self.target_duration,
                prompt=prompt,
                condition=latent.unsqueeze(0),
            )
            assert prediction.duration == prompt.duration + self.target_duration
            bars.append(prediction[:, prompt.length :])

        if given_prompt_bars is None:
            # remove the padding bars
            return SymbolicRepresentation.cat_time(bars[n_prompt_bars:])
        else:
            return SymbolicRepresentation.cat_time(bars)
