from pathlib import Path
from typing import Callable, Sequence

import lightning as LT
import torch
from lightning.fabric.utilities.rank_zero import rank_zero_only
from music_data_analysis import Pianoroll
from safetensors.torch import save_file

from vqpiano.data.pianoroll_dataset import PianorollDataset
from vqpiano.evaluate.vae import evaluate_vae
from vqpiano.models.encoder_decoder import EncoderDecoder
from vqpiano.models.token_sequence import TokenSequence
from vqpiano.utils.data import iter_dataclass
from vqpiano.utils.torch_utils.wandb import log_image, log_midi_as_audio


class VAETrainingWrapper(LT.LightningModule):
    def __init__(
        self,
        model: EncoderDecoder,
        max_tokens_prompt: int,
        max_tokens_target: int,
        pitch_range: list[int],
        lr,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
    ):
        super().__init__()
        self.example_input_array = {
            "prompt": TokenSequence(
                token=torch.zeros(2, max_tokens_prompt, 2, dtype=torch.long),
                token_type=torch.zeros(2, max_tokens_prompt, dtype=torch.long),
                pos=torch.zeros(2, max_tokens_prompt, dtype=torch.long),
            ),
            "target": TokenSequence(
                token=torch.zeros(2, max_tokens_target, 2, dtype=torch.long),
                token_type=torch.zeros(2, max_tokens_target, dtype=torch.long),
                pos=torch.zeros(2, max_tokens_target, dtype=torch.long),
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
        batch: tuple[TokenSequence, TokenSequence],
        batch_idx: int,
    ):
        self.model.train()
        prompt, target = batch
        self.model.set_step(self.global_step)

        prompt = prompt.to(self.device)
        target = target.to(self.device)

        loss: EncoderDecoder.Loss = self.model(target=target, prompt=prompt)

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
    def __init__(self, demo_every: int, dataset_path, length: int = 32 * 16):
        super().__init__()

        self.demo_every = demo_every
        dataset = PianorollDataset(
            Path(dataset_path),
            frames_per_beat=8,
            length=length,
            min_start_overlap=length,
            min_end_overlap=length,
        )

        def collate_fn(pr_list: list[Pianoroll]):
            pr = pr_list[0]
            bars = []
            for bar in pr.iter_over_bars_pr(bar_length=32):
                bars.append(bar)

            result = TokenSequence.from_pianorolls(bars)

            return result

        class MyDataLoader(Sequence[TokenSequence]):
            def __init__(
                self,
                dataset: PianorollDataset,
                collate_fn: Callable[[list[Pianoroll]], TokenSequence],
                batch_size: int,
            ):
                self.dataset = dataset
                self.collate_fn = collate_fn
                self.batch_size = batch_size

            def __getitem__(self, index: int) -> TokenSequence:  # type: ignore
                return self.collate_fn([self.dataset[index]])

            def __len__(self):
                return len(self.dataset)

        self.eval_ds = MyDataLoader(
            dataset,
            collate_fn=collate_fn,
            batch_size=1,
        )

        # self.eval_ds = create_dataloader_simple_ar_reconstruct(dataset_params, 32 * 16)

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
        if pl_module.global_step % 4 == 0:
            torch.cuda.empty_cache()
        min_pitch = pl_module.model.pitch_range[0]
        pl_module.model.eval()
        if batch_idx % self.demo_every == 0:
            pl_module.model.eval()

            eval_result = evaluate_vae(
                model=pl_module.model,
                eval_ds=self.eval_ds,
                num_samples=1,
                device=pl_module.device,
            )[0]

            for name, sample in eval_result.items():
                assert isinstance(sample, TokenSequence)
                log_midi_as_audio(
                    sample.to_midi(min_pitch),
                    name,
                    pl_module.global_step,
                )

            try:
                log_image(
                    torch.cat(
                        [
                            eval_result["gt"].to_pianoroll(min_pitch).to_img_tensor(),
                            eval_result["reconst_ind"]
                            .to_pianoroll(min_pitch)
                            .to_img_tensor(),
                            eval_result["reconst"]
                            .to_pianoroll(min_pitch)
                            .to_img_tensor(),
                            eval_result["reconst_sampled_latent"]
                            .to_pianoroll(min_pitch)
                            .to_img_tensor(),
                        ],
                        dim=0,
                    ),
                    "pr",
                    pl_module.global_step,
                )
            except Exception as e:  # TODO: fix this
                print("Error logging image", e)
