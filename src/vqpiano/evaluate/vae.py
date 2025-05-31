import random
from pathlib import Path
from typing import Sequence, TypedDict

import torch
from tqdm import tqdm

from vqpiano.models.encoder_decoder import EncoderDecoder
from vqpiano.models.token_sequence import TokenSequence


def reconstruct_independent(
    model: EncoderDecoder, gt_bars: TokenSequence, latents: torch.Tensor
):
    result = TokenSequence(device=latents.device)

    for i in tqdm(range(len(latents)), desc="Generating..."):
        prompt_bars = gt_bars[max(0, i - 4) : i]
        prompt = TokenSequence(device=latents.device)
        pad = max(0, 4 - prompt_bars.batch_size) * 32
        for _ in range(pad):
            prompt.add_frame()
        for bar in prompt_bars:
            prompt += bar
        prediction = model.decoder.sample(
            duration=prompt.duration + model.duration,
            prompt=prompt,
            condition=latents[i].unsqueeze(0),
        )
        assert prediction.duration == prompt.duration + model.duration
        result += prediction[:, prompt.length :]
    return result


def reconstruct_autoregressive(
    model: EncoderDecoder,
    latents: torch.Tensor,
    n_prompt_bars: int,
    given_prompt_bars: list[TokenSequence] | None = None,
):
    """
    if given_prompt_bars is None, the first iterations the model will receive empty bars as prompts. It will feel
    generating the beginning of the piece.

    To make the model generate bars from the middle of the piece, pass the previous bars as given_prompt_bars.
    """
    bars = []

    if given_prompt_bars is None:
        for i in range(n_prompt_bars):
            bar = TokenSequence(device=latents.device)
            for _ in range(32):
                bar.add_frame()
            bars.append(bar)
    else:
        assert len(given_prompt_bars) == n_prompt_bars, (
            f"{len(given_prompt_bars)} != {n_prompt_bars}"
        )
        bars = given_prompt_bars.copy()

    for i in tqdm(range(len(latents)), desc="Generating..."):
        prompt = TokenSequence.cat_time(bars[i : i + n_prompt_bars])

        prediction = model.decoder.sample(
            duration=prompt.duration + model.duration,
            prompt=prompt,
            condition=latents[i].unsqueeze(0),
        )
        assert prediction.duration == prompt.duration + model.duration
        bars.append(prediction[:, prompt.length :])

    if given_prompt_bars is None:
        # remove the padding bars
        return TokenSequence.cat_time(bars[n_prompt_bars:])
    else:
        return TokenSequence.cat_time(bars)


class EvaluateVAEResult(TypedDict):
    reconst_ind: TokenSequence
    reconst: TokenSequence
    reconst_sampled_latent: TokenSequence
    gt: TokenSequence


def evaluate_vae(
    model: EncoderDecoder,
    eval_ds: Sequence[TokenSequence],
    num_samples: int,
    out_dir: Path | None = None,
    max_bars: int = 100,
    device: torch.device = torch.device("cuda"),
):
    model.eval()

    result: list[EvaluateVAEResult] = []

    for sample_idx in range(num_samples):
        batch: TokenSequence = eval_ds[random.randint(0, len(eval_ds) - 1)].to(device)

        if len(batch) > max_bars:
            batch = batch[:max_bars]

        # reconstuction

        latents = model.encode(batch, sample_latent=False)
        reconst_ind = reconstruct_independent(model, batch, latents)

        n_prompt_bars = model.prompt_duration // 32
        prompt_bars = [batch[i : i + 1] for i in range(0, n_prompt_bars)]

        reconst = reconstruct_autoregressive(
            model, latents[n_prompt_bars:], n_prompt_bars, given_prompt_bars=prompt_bars
        )

        latents = model.encode(batch, sample_latent=True)
        reconst_sampled_latent = reconstruct_autoregressive(
            model, latents[n_prompt_bars:], n_prompt_bars, given_prompt_bars=prompt_bars
        )

        if out_dir is not None:
            reconst.to_midi(
                model.pitch_range[0], out_dir / f"reconst/reconst_{sample_idx}.mid"
            )
            reconst_ind.to_midi(
                model.pitch_range[0], out_dir / f"reconst/reconst_ind_{sample_idx}.mid"
            )
            reconst_sampled_latent.to_midi(
                model.pitch_range[0],
                out_dir / f"reconst/reconst_sampled_latent_{sample_idx}.mid",
            )

        gt = TokenSequence(device=device)
        for bar in batch:
            gt += bar
        if out_dir is not None:
            gt.to_midi(model.pitch_range[0], out_dir / f"reconst/gt_{sample_idx}.mid")

        result.append(
            EvaluateVAEResult(
                reconst_ind=reconst_ind,
                reconst=reconst,
                reconst_sampled_latent=reconst_sampled_latent,
                gt=gt,
            )
        )

    return result
