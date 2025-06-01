from pathlib import Path
from typing import Sized

import lightning as LT
from music_data_analysis import Pianoroll
import torch
from lightning.fabric.utilities.rank_zero import rank_zero_only
from safetensors.torch import save_file

from torch.utils.data import Dataset
from segment_full_song.models.encoder_decoder import EncoderDecoder
from segment_full_song.models.token_sequence import TokenSequence
from segment_full_song.utils.data import iter_dataclass
from segment_full_song.utils.torch_utils.wandb import log_image, log_midi_as_audio


class VAETrainingWrapper(LT.LightningModule):
    def __init__(
        self,
        model: EncoderDecoder,
        max_tokens: int,
        pitch_range: list[int],
        lr,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
    ):
        super().__init__()
        self.example_input_array = {
            "x": TokenSequence(
                token=torch.zeros(2, max_tokens, 3, dtype=torch.long),
                token_type=torch.zeros(2, max_tokens, dtype=torch.long),
                pos=torch.zeros(2, max_tokens, dtype=torch.long),
            ),
        }
        self.model = model
        self.pitch_range = pitch_range
        self.optim_config = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.model.parameters(),
            **self.optim_config,
        )
        return opt

    def training_step(
        self,
        batch: dict[str, TokenSequence],
        batch_idx: int,
    ):
        self.model.train()
        tokens = batch["tokens"]
        self.model.set_step(self.global_step)

        loss: EncoderDecoder.Loss = self.model(x=tokens)

        metrics = {}

        metrics["loss"] = loss.total_loss.item()

        for k, v in iter_dataclass(loss.bottleneck):
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().item()
            metrics[f"bottleneck/{k}"] = v

        for k, v in iter_dataclass(loss.reconstruction):
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().item()
            metrics[f"reconst/{k}"] = v

        self.log_dict(metrics)

        return loss.total_loss

    def export_model(self, save_dir: Path, prefix: str, use_safetensors=True):
        if use_safetensors:
            save_file(self.model.state_dict(), save_dir / f"{prefix}.safetensors")
        else:
            torch.save(
                {"state_dict": self.model.state_dict()}, save_dir / f"{prefix}.pt"
            )


class VAEDemoCallback(LT.Callback):
    def __init__(self, demo_every: int, test_ds: Dataset):
        super().__init__()

        self.demo_every = demo_every
        self.test_ds = test_ds
    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(  # type: ignore
        self,
        trainer,
        pl_module: VAETrainingWrapper,
        outputs,
        batch: tuple[TokenSequence, TokenSequence],
        batch_idx,
    ):
        if (
            torch.cuda.memory_reserved()
            >= torch.cuda.get_device_properties(0).total_memory * 0.95
        ):
            torch.cuda.empty_cache()
        min_pitch = pl_module.model.pitch_range[0]
        pl_module.model.eval()
        if pl_module.global_step == 1 or pl_module.global_step % self.demo_every == 0:
            pl_module.model.eval()

            assert isinstance(self.test_ds, Sized)
            pr: Pianoroll = self.test_ds[
                int(torch.randint(0, len(self.test_ds), (1,)))
            ]["pianoroll"]

            n_iter = pr.duration // pl_module.model.duration
            reconst_result = []
            for i in range(n_iter):
                bar = pr.slice(
                    i * pl_module.model.duration, (i + 1) * pl_module.model.duration
                )
                input = TokenSequence.from_pianorolls(
                    [bar],
                    max_tokens=pl_module.model.max_tokens,
                    max_note_duration=pl_module.model.encoder.max_note_duration,
                ).to(pl_module.device)
                reconst_result.append(
                    pl_module.model.reconstruct(input, sample_latent=False)
                )

            reconst_result = TokenSequence.cat_time(reconst_result)

            sampled_reconst_result = []
            for i in range(n_iter):
                bar = pr.slice(
                    i * pl_module.model.duration, (i + 1) * pl_module.model.duration
                )
                input = TokenSequence.from_pianorolls(
                    [bar],
                    max_tokens=pl_module.model.max_tokens,
                    max_note_duration=pl_module.model.encoder.max_note_duration,
                ).to(pl_module.device)
                sampled_reconst_result.append(
                    pl_module.model.reconstruct(input, sample_latent=True)
                )

            sampled_reconst_result = TokenSequence.cat_time(sampled_reconst_result)

            log_midi_as_audio(
                pr.to_midi(),
                "gt",
                pl_module.global_step,
            )
            log_image(
                pr.to_img_tensor(),
                "gt_pr",
                pl_module.global_step,
            )

            log_midi_as_audio(
                reconst_result.to_midi(min_pitch),
                "reconst",
                pl_module.global_step,
            )
            log_image(
                reconst_result.to_pianoroll(min_pitch).to_img_tensor(),
                "reconst_pr",
                pl_module.global_step,
            )

            log_midi_as_audio(
                sampled_reconst_result.to_midi(min_pitch),
                "sampled_reconst",
                pl_module.global_step,
            )
            log_image(
                sampled_reconst_result.to_pianoroll(min_pitch).to_img_tensor(),
                "sampled_reconst_pr",
                pl_module.global_step,
            )
