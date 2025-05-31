from pathlib import Path

import lightning as LT
import torch
from lightning.fabric.utilities.rank_zero import rank_zero_only
from safetensors.torch import save_file

from vqpiano.models.token_generator import TokenGenerator
from vqpiano.models.token_sequence import TokenSequence
from vqpiano.utils.data import iter_dataclass
from vqpiano.utils.torch_utils.wandb import log_midi_as_audio, log_pianoroll


class SimpleARTrainingWrapper(LT.LightningModule):
    def __init__(
        self,
        model: TokenGenerator,
        pitch_range: list[int],
        lr,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
    ):
        super().__init__()
        self.example_input_array = {
            "x": TokenSequence(
                token=torch.zeros(3, 16, 3, dtype=torch.long),
                token_type=torch.zeros(3, 16, dtype=torch.long),
                pos=torch.zeros(3, 16, dtype=torch.long),
            )
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

    def training_step(self, batch, batch_idx: int):
        self.model.train()
        loss = self(x=batch["tokens"])
        metrics = {}
        for k, v in iter_dataclass(loss):
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().item()
            metrics[f"generation/{k}"] = v
        self.log_dict(metrics)

        return loss.total_loss

    def export_model(self, save_dir: Path, prefix: str, use_safetensors=True):
        if use_safetensors:
            save_file(self.model.state_dict(), save_dir / f"{prefix}.safetensors")
        else:
            torch.save(
                {"state_dict": self.model.state_dict()}, save_dir / f"{prefix}.pt"
            )


class SimpleARDemoCallback(LT.Callback):
    def __init__(self, demo_every: int, duration: int = 32 * 8):
        super().__init__()

        self.demo_every = demo_every
        self.duration = duration

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(  # type: ignore
        self,
        trainer,
        pl_module: SimpleARTrainingWrapper,
        outputs,
        batch: TokenSequence,
        batch_idx,
    ):
        if (
            torch.cuda.memory_reserved()
            >= torch.cuda.get_device_properties(0).total_memory * 0.95
        ):
            torch.cuda.empty_cache()
        if pl_module.global_step == 1 or pl_module.global_step % self.demo_every == 0:
            pl_module.model.eval()

            min_pitch = pl_module.model.pitch_range[0]
            sample = pl_module.model.sample(duration=self.duration, progress_bar=True)
            log_midi_as_audio(
                sample.to_midi(min_pitch), "generation", pl_module.global_step
            )
            log_pianoroll(
                [sample.to_pianoroll(min_pitch)],
                "generation_pr",
                pl_module.global_step,
                format="png",
            )

