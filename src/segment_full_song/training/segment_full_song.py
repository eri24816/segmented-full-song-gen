from pathlib import Path
import random
import traceback

import lightning as LT
import loguru
from music_data_analysis import Pianoroll
import torch
from lightning.fabric.utilities.rank_zero import rank_zero_only
from safetensors.torch import save_file

from segment_full_song.models.segment_full_song import SegmentFullSongModel
from segment_full_song.utils.data import iter_dataclass
from segment_full_song.utils.torch_utils.wandb import log_midi_as_audio, log_pianoroll
from typing import cast


class SegmentFullSongTrainingWrapper(LT.LightningModule):
    def __init__(
        self,
        model: SegmentFullSongModel,
        pitch_range: list[int],
        lr,
        max_tokens: int,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        lr_scheduler_gamma: float = 1,
        accum_batches: int = 1,
        continue_on_oom: bool = True,
    ):
        super().__init__()
        self.model = model
        self.pitch_range = pitch_range
        self.optim_config = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        self.lr_scheduler_config = {"gamma": lr_scheduler_gamma}
        self.automatic_optimization = False
        self.actual_step = -1  # lightning fails to calculate right global_step for automatic_optimization = False
        self.accum_batches = accum_batches
        self.max_tokens = max_tokens
        self.continue_on_oom = continue_on_oom

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.model.parameters(),
            **self.optim_config,
        )
        if self.lr_scheduler_config["gamma"] != 1:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=opt,
                gamma=self.lr_scheduler_config["gamma"],
            )

            return [opt], [scheduler]
        else:
            return opt

    def training_step(self, batch, batch_idx: int):
        self.model.train()
        # print(
        #     " ",
        #     batch["target"]["tokens"].length,
        #     batch["left"]["tokens"].length,
        #     batch["right"]["tokens"].length,
        #     batch["seed"]["tokens"].length,
        #     batch["reference"]["tokens"].length,
        # )

        num_tokens = max(
            batch["target"]["tokens"].length,
            batch["left"]["tokens"].length,
            batch["right"]["tokens"].length,
            batch["seed"]["tokens"].length,
            batch["reference"]["tokens"].length,
        )
        if num_tokens > self.max_tokens:
            loguru.logger.info(
                f"Skipping this batch because it's too long. {num_tokens} > {self.max_tokens}"
            )
            return None

        self.actual_step += 1
        try:
            loss = self.model(
                x=batch["target"],
                context=[
                    batch["left"],
                    batch["right"],
                    batch["seed"],
                    batch["reference"],
                ],
                bar_embeddings=batch["bar_embeddings"],
                bar_embeddings_mask=batch["bar_embeddings_mask"],
            )
        except Exception as e:
            if isinstance(e, torch.cuda.OutOfMemoryError) and self.continue_on_oom:
                torch.cuda.empty_cache()
                loguru.logger.warning("Out of memory. Skipping this batch.")
                return None
            else:
                raise e
        opt = cast(torch.optim.Optimizer, self.optimizers())

        try:
            self.manual_backward(loss.total_loss)
        except torch.cuda.OutOfMemoryError:
            loguru.logger.warning(
                "Out of memory while backwarding. Skipping optimizer step."
            )
            torch.cuda.empty_cache()
            opt.zero_grad()
            return
        if self.actual_step % self.accum_batches == 0:
            opt.step()
            opt.zero_grad()

        scheduler = cast(torch.optim.lr_scheduler.ExponentialLR, self.lr_schedulers())

        if self.actual_step % 5 == 0:
            metrics = {}
            for k, v in iter_dataclass(loss):
                if isinstance(v, torch.Tensor):
                    v = v.detach().cpu().item()
                metrics[k] = v
            metrics["lr"] = scheduler.get_last_lr()[0]
            metrics["actual_step"] = self.actual_step
            self.log_dict(metrics)

        scheduler.step()

    def export_model(self, save_dir: Path, prefix: str, use_safetensors=True):
        if use_safetensors:
            save_file(self.model.state_dict(), save_dir / f"{prefix}.safetensors")
        else:
            torch.save(
                {"state_dict": self.model.state_dict()}, save_dir / f"{prefix}.pt"
            )


class SegmentFullSongDemoCallback(LT.Callback):
    def __init__(
        self,
        demo_every: int,
        test_ds: torch.utils.data.Dataset,
        max_context_duration: dict[str, int],
        max_tokens: int,
    ):
        super().__init__()

        self.demo_every = demo_every
        self.test_ds = test_ds
        self.max_context_duration = max_context_duration
        self.max_tokens = max_tokens

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(  # type: ignore
        self,
        trainer,
        pl_module: SegmentFullSongTrainingWrapper,
        outputs,
        batch: dict,
        batch_idx,
    ):
        step = pl_module.actual_step
        if (
            torch.cuda.memory_reserved()
            >= torch.cuda.get_device_properties(0).total_memory * 0.95
        ):
            torch.cuda.empty_cache()
        if step % self.demo_every == 0 and step > 0:
            pl_module.model.eval()

            gt = self.test_ds[random.randint(0, len(self.test_ds) - 1)]

            segment_info_list = gt["segments"]

            segment_info_list = gt["segments"]

            ordered_segment_info_list = sorted(
                segment_info_list, key=lambda x: x["start"]
            )
            labels = "".join(
                [
                    "ABCDEFGH"[segment_info["label"]]
                    for segment_info in ordered_segment_info_list
                ]
            )
            lengths_in_bars = [
                (segment_info["end"] - segment_info["start"])
                // pl_module.model.frames_per_bar
                for segment_info in ordered_segment_info_list
            ]

            compose_order = [
                ordered_segment_info_list.index(segment_info)
                for segment_info in segment_info_list
            ]

            use_given_segments = (step // self.demo_every) % 2 == 0
            if use_given_segments:
                # first segment is given

                pr: Pianoroll = gt["pr"]
                given_segments = [
                    pr[segment_info_list[0]["start"] : segment_info_list[0]["end"]]
                ]
            else:
                # generate from scratch
                given_segments = []

            try:
                generated_song, annotations = pl_module.model.sample_song(
                    labels=labels,
                    lengths_in_bars=lengths_in_bars,
                    compose_order=compose_order,
                    given_segments=given_segments,
                )

                # log generated segment
                log_midi_as_audio(
                    generated_song.to_midi(markers=annotations),
                    "generation",
                    step,
                )
                if use_given_segments:
                    log_midi_as_audio(pr.to_midi(markers=annotations), "gt", step)

                    log_pianoroll(
                        [pr, generated_song],
                        "pr",
                        step,
                        annotations=annotations,
                        format="png",
                    )
                else:
                    log_pianoroll(
                        generated_song,
                        "pr",
                        step,
                        annotations=annotations,
                        format="png",
                    )

            except Exception:
                loguru.logger.error(traceback.format_exc())
                return
