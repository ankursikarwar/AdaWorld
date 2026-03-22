import threading
from os import makedirs, path
from typing import Callable, Dict, Iterable, Tuple

import numpy as np
import piq
import torch
from PIL import Image
from einops import rearrange
from kmeans_pytorch import kmeans
from lightning import LightningModule
from torch import Tensor
from torch.optim import AdamW, Optimizer

OptimizerCallable = Callable[[Iterable], Optimizer]

from lam.modules import LatentActionModel


class LAM(LightningModule):
    def __init__(
            self,
            image_channels: int = 3,
            # Latent action autoencoder
            lam_model_dim: int = 512,
            lam_latent_dim: int = 32,
            lam_patch_size: int = 16,
            lam_enc_blocks: int = 8,
            lam_dec_blocks: int = 8,
            lam_num_heads: int = 8,
            lam_dropout: float = 0.0,
            beta: float = 0.01,
            log_interval: int = 1000,
            log_path: str = "log_imgs",
            optimizer: OptimizerCallable = AdamW
    ) -> None:
        super(LAM, self).__init__()
        self.lam = LatentActionModel(
            in_dim=image_channels,
            model_dim=lam_model_dim,
            latent_dim=lam_latent_dim,
            patch_size=lam_patch_size,
            enc_blocks=lam_enc_blocks,
            dec_blocks=lam_dec_blocks,
            num_heads=lam_num_heads,
            dropout=lam_dropout
        )
        self.beta = beta
        self.log_interval = log_interval
        self.log_path = log_path
        self.optimizer = optimizer

        self.save_hyperparameters()

    def shared_step(self, batch: Dict) -> Tuple:
        outputs = self.lam(batch)
        gt_future_frames = batch["videos"][:, 1:]

        # Compute loss
        mse_loss = ((gt_future_frames - outputs["recon"]) ** 2).mean()
        kl_loss = -0.5 * torch.sum(1 + outputs["z_var"] - outputs["z_mu"] ** 2 - outputs["z_var"].exp(), dim=1).mean()
        loss = mse_loss + self.beta * kl_loss

        # Compute monitoring measurements
        gt = gt_future_frames.clamp(0, 1).reshape(-1, *gt_future_frames.shape[2:]).permute(0, 3, 1, 2)
        recon = outputs["recon"].clamp(0, 1).reshape(-1, *outputs["recon"].shape[2:]).permute(0, 3, 1, 2)
        psnr = piq.psnr(gt, recon).mean()
        ssim = piq.ssim(gt, recon).mean()
        return outputs, loss, (
            ("mse_loss", mse_loss),
            ("kl_loss", kl_loss),
            ("psnr", psnr),
            ("ssim", ssim)
        )

    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:
        # Compute the training loss
        outputs, loss, aux_losses = self.shared_step(batch)

        # Log the training loss
        self.log_dict(
            {**{"train_loss": loss}, **{f"train/{k}": v for k, v in aux_losses}},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True
        )

        self.log(
            "global_step",
            self.global_step,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=False
        )

        if batch_idx % self.log_interval == 0 and self.global_rank == 0:
            # Save to disk immediately (fast, no network)
            self.log_images(batch, outputs, "train")
            # Cache CPU tensors for async WandB upload in on_train_batch_end
            self._wandb_cache = {
                "input": batch["videos"][0, 0].clamp(0, 1).detach().cpu(),
                "gt": batch["videos"][0, 1].clamp(0, 1).detach().cpu(),
                "pred": outputs["recon"][0, 0].clamp(0, 1).detach().cpu(),
                "split": "train",
                "step": self.global_step,
            }
        return loss

    def on_train_batch_end(self, out, batch, batch_idx) -> None:
        # Only rank 0, only when there's something to upload
        if not hasattr(self, "_wandb_cache"):
            return
        cache = self._wandb_cache
        del self._wandb_cache

        # Find WandB logger once
        wandb_logger = None
        for logger in self.loggers:
            if type(logger).__name__ == "WandbLogger":
                wandb_logger = logger
                break
        if wandb_logger is None:
            return

        # Fire-and-forget background thread — no CUDA, pure CPU/network, won't block DDP
        def _upload(cache, wandb_logger):
            try:
                import wandb
                inp = (cache["input"].numpy() * 255).astype(np.uint8)
                gt = (cache["gt"].numpy() * 255).astype(np.uint8)
                pred = (cache["pred"].numpy() * 255).astype(np.uint8)
                # Grid: row 1 = input -> GT next frame, row 2 = input -> predicted next frame
                row1 = np.concatenate([inp, gt], axis=1)
                row2 = np.concatenate([inp, pred], axis=1)
                grid = np.concatenate([row1, row2], axis=0)
                wandb_logger.experiment.log({
                    f"{cache['split']}/input_frame": wandb.Image(inp, caption="Input (frame t)"),
                    f"{cache['split']}/gt_next_frame": wandb.Image(gt, caption="GT (frame t+1)"),
                    f"{cache['split']}/predicted_frame": wandb.Image(pred, caption="Predicted (frame t+1)"),
                    f"{cache['split']}/frame_comparison": wandb.Image(
                        grid, caption="Top: input→GT | Bottom: input→predicted"),
                    "trainer/global_step": cache["step"],
                })
            except Exception as e:
                print(f"[WandB] Image upload failed: {e}")

        thread = threading.Thread(target=_upload, args=(cache, wandb_logger), daemon=True)
        thread.start()

    # @torch.no_grad()
    # def validation_step(self, batch: Dict, batch_idx: int) -> Tensor:
    #     # Compute the validation loss
    #     outputs, loss, aux_losses = self.shared_step(batch)
    #
    #     # Log the validation loss
    #     self.log_dict(
    #         {**{"val_loss": loss}, **{f"val/{k}": v for k, v in aux_losses}},
    #         prog_bar=True,
    #         logger=True,
    #         on_step=True,
    #         on_epoch=True,
    #         sync_dist=True
    #     )
    #
    #     if batch_idx % self.log_interval == 0:  # Start of the epoch
    #         self.log_images(batch, outputs, "val")
    #     return loss

    @torch.no_grad()
    def test_step(self, batch: Dict, batch_idx: int) -> Tensor:
        # Compute the test loss
        outputs, loss, aux_losses = self.shared_step(batch)

        # Log the test loss
        self.log_dict(
            {**{"test_loss": loss}, **{f"test/{k}": v for k, v in aux_losses}},
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True
        )

        self.log_images(batch, outputs, "test")
        return loss

    def log_images(self, batch: Dict, outputs: Dict, split: str) -> None:
        # Saves a side-by-side comparison image to disk only — no network calls here
        gt_seq = batch["videos"][0].clamp(0, 1).cpu()
        recon_seq = outputs["recon"][0].clamp(0, 1).detach().cpu()
        recon_seq = torch.cat([gt_seq[:1], recon_seq], dim=0)
        compare_seq = torch.cat([gt_seq, recon_seq], dim=1)
        compare_seq = rearrange(compare_seq * 255, "t h w c -> h (t w) c")
        compare_seq = compare_seq.detach().numpy().astype(np.uint8)
        img_path = path.join(self.log_path, f"{split}_step{self.global_step:06}.png")
        makedirs(path.dirname(img_path), exist_ok=True)
        img = Image.fromarray(compare_seq)
        try:
            img.save(img_path)
        except Exception:
            pass

    # def on_test_epoch_end(self) -> None:
    #     # For init specialized world models
    #     torch.save(self.lam.mu_record, f"latent_action_stats.pt")
    #
    #     # For action creation as generative interactive environments
    #     cluster_ids, cluster_centers = kmeans(
    #         X=self.lam.mu_record,
    #         num_clusters=8,
    #         distance="euclidean"
    #     )
    #     torch.save(cluster_centers, f"latent_action_centers.pt")

    def configure_optimizers(self) -> Optimizer:
        optim = self.optimizer(self.parameters())
        return optim
