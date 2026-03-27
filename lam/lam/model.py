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
            lam_causal_temporal: bool = False,
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
            dropout=lam_dropout,
            causal_temporal=lam_causal_temporal
        )
        self.beta = beta
        self.log_interval = log_interval
        self.log_path = log_path
        self.optimizer = optimizer

        self.save_hyperparameters()

    def shared_step(self, batch: Dict) -> Tuple:
        outputs = self.lam(batch)
        gt_future_frames = batch["videos"][:, 1:]

        # Guard against fp16 overflow producing NaN/Inf in recon
        outputs["recon"] = torch.nan_to_num(outputs["recon"], nan=0.0, posinf=1.0, neginf=0.0)
        recon_safe = outputs["recon"]

        # Compute loss
        mse_loss = ((gt_future_frames - recon_safe) ** 2).mean()
        # Guard z_var.exp() against fp16 overflow (exp overflows fp16 above ~88)
        z_var_safe = torch.nan_to_num(outputs["z_var"], nan=0.0, posinf=10.0, neginf=-10.0)
        kl_loss = -0.5 * torch.sum(1 + z_var_safe - outputs["z_mu"] ** 2 - z_var_safe.exp(), dim=1).mean()
        loss = mse_loss + self.beta * kl_loss

        # Compute monitoring measurements
        gt = gt_future_frames.clamp(0, 1).reshape(-1, *gt_future_frames.shape[2:]).permute(0, 3, 1, 2)
        recon = recon_safe.clamp(0, 1).reshape(-1, *recon_safe.shape[2:]).permute(0, 3, 1, 2)
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

        # Option B: per-timestep MSE — shows whether early vs late pairs are harder
        T = batch["videos"].shape[1]
        if T > 2:
            gt_future = batch["videos"][:, 1:]
            recon = outputs["recon"]
            self.log_dict(
                {f"train/mse_t{t}": ((gt_future[:, t] - recon[:, t]) ** 2).mean()
                 for t in range(T - 1)},
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                logger=True
            )

        if batch_idx % self.log_interval == 0 and self.global_rank == 0:
            # Save to disk immediately (fast, no network)
            self.log_images(batch, outputs, "train")
            # Cache CPU tensors for async WandB upload in on_train_batch_end
            n_vis = min(4, batch["videos"].shape[0])
            self._wandb_cache = {
                "gt_frames": batch["videos"][:n_vis].clamp(0, 1).detach().cpu(),
                "recon_frames": outputs["recon"][:n_vis].clamp(0, 1).detach().cpu(),
                "split": "train",
                "step": self.global_step,
            }
        return loss

    def on_train_epoch_end(self) -> None:
        # Option A: epoch-averaged smooth curves — Lightning auto-aggregates
        # metrics logged with on_epoch=True. We pull them from the callback_metrics
        # dict and re-log under clean "smooth/" names so WandB shows one clean line
        # per metric (not the noisy per-step line) for easy visual comparison.
        smooth_keys = [
            ("train/mse_loss_epoch", "smooth/mse_loss"),
            ("train/kl_loss_epoch",  "smooth/kl_loss"),
            ("train/psnr_epoch",     "smooth/psnr"),
            ("train/ssim_epoch",     "smooth/ssim"),
        ]
        logged = {}
        for src_key, dst_key in smooth_keys:
            val = self.trainer.callback_metrics.get(src_key)
            if val is not None:
                logged[dst_key] = val.item() if hasattr(val, "item") else float(val)

        # Also add per-timestep epoch averages if they exist (from Option B)
        for key, val in self.trainer.callback_metrics.items():
            if key.startswith("train/mse_t") and key.endswith("_epoch"):
                t_idx = key[len("train/mse_t"):-len("_epoch")]
                logged[f"smooth/mse_t{t_idx}"] = val.item() if hasattr(val, "item") else float(val)

        if logged and self.global_rank == 0:
            wandb_logger = None
            for logger in self.loggers:
                if type(logger).__name__ == "WandbLogger":
                    wandb_logger = logger
                    break
            if wandb_logger is not None:
                import wandb
                wandb_logger.experiment.log({
                    **logged,
                    "trainer/global_step": self.global_step,
                    "epoch": self.current_epoch,
                })

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
                gt = (cache["gt_frames"].numpy() * 255).astype(np.uint8)   # (n, T, H, W, C)
                rc = (cache["recon_frames"].numpy() * 255).astype(np.uint8) # (n, T-1, H, W, C)
                n, T = gt.shape[:2]
                # For each sample: two rows side by side across time
                #   Row 1 (GT):    [f0 | f1 | f2 | ... | f_{T-1}]
                #   Row 2 (Recon): [f0 | recon_1 | ... | recon_{T-1}]
                rows = []
                for i in range(n):
                    gt_row   = np.concatenate([gt[i, t] for t in range(T)], axis=1)
                    recon_row = np.concatenate([gt[i, 0]] + [rc[i, t] for t in range(T - 1)], axis=1)
                    rows.append(np.concatenate([gt_row, recon_row], axis=0))
                grid = np.concatenate(rows, axis=0)
                wandb_logger.experiment.log({
                    f"{cache['split']}/segment_grid": wandb.Image(
                        grid, caption=f"Top: GT frames | Bottom: f0 + reconstructions (T={T}, {n} samples)"),
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
