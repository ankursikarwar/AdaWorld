"""
LAM + CLIP Alignment Model.

Trains the LAM encoder + MLP projection so that aggregated latent actions
from a full video clip align with CLIP text embeddings of the action template.

Architecture:
    Video (T frames) → sample K consecutive pairs evenly across clip
        → LAM.forward on each pair → K z_mu vectors (32-dim) + K reconstructions
        → mean pool z_mu → canonical action (32-dim)
        → MLP → u (768-dim) → L2 normalize

    Action template → precomputed CLIP text embedding (frozen) → v (768-dim)

    L_total = L_recon + L_align
    where:
        L_recon = MSE + beta * KL  (on ALL sampled pairs, not just one)
        L_align = regression (1 - cos_sim) OR contrastive (InfoNCE)

Design decisions:
    - Pair-wise LAM processing: LAM was pretrained with T=2, so we process
      consecutive pairs (f_t, f_{t+1}) independently, not the full video at once
    - Evenly spaced sampling: pick K pairs evenly across the video duration
      so canonical action captures the full action arc (begin to end)
    - CLIP embeddings precomputed: only 174 unique SSv2 templates, encoded once
      in __init__ and stored as a registered buffer
    - Reconstruction is a proper loss (not regularization), computed on all pairs
"""
import json
import threading
from os import makedirs, path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch import Tensor
from torch.optim import AdamW, Optimizer
from lam.model import LAM
from lam.modules import LatentActionModel

OptimizerCallable = Callable[[Iterable], Optimizer]


class LAMClipAlign(LightningModule):
    def __init__(
            self,
            # --- LAM pretrained checkpoint ---
            lam_checkpoint: str = "",
            # --- LAM architecture (must match checkpoint) ---
            image_channels: int = 3,
            lam_model_dim: int = 1024,
            lam_latent_dim: int = 32,
            lam_patch_size: int = 16,
            lam_enc_blocks: int = 16,
            lam_dec_blocks: int = 16,
            lam_num_heads: int = 16,
            # --- CLIP ---
            clip_model_name: str = "openai/clip-vit-large-patch14",
            clip_dim: int = 768,
            labels_json: str = "/home/mila/a/ankur.sikarwar/scratch/WORLD_MODEL_PROJECT/data/ssv2/raw/labels/labels.json",
            # --- MLP projection ---
            projection_hidden_dim: int = 256,
            # --- Loss ---
            loss_type: str = "regression",  # "regression" or "contrastive"
            temperature_init: float = 0.07,
            beta: float = 0.0002,           # KL weight in reconstruction loss
            # --- Pair sampling ---
            num_pairs_per_video: int = 15,  # how many consecutive pairs to sample per video
            max_pairs_per_chunk: int = 32,  # process LAM pairs in chunks for memory
            # --- Training ---
            log_interval: int = 500,
            lr: float = 1e-4,
            lr_lam: float = 2.5e-5,
            lr_projection: float = 1e-3,
            weight_decay: float = 1e-2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.loss_type = loss_type
        self.beta = beta
        self.num_pairs_per_video = num_pairs_per_video
        self.max_pairs_per_chunk = max_pairs_per_chunk
        self.log_interval = log_interval
        self.lam_latent_dim = lam_latent_dim
        self.clip_dim = clip_dim
        self.lr = lr
        self.lr_lam = lr_lam
        self.lr_projection = lr_projection
        self.weight_decay = weight_decay

        # =====================================================================
        # 1. LAM (pretrained, will be finetuned)
        # =====================================================================
        # Load the full LAM Lightning module, then extract the LatentActionModel.
        # This preserves all pretrained weights (encoder + decoder + VAE head).
        if lam_checkpoint and path.exists(lam_checkpoint):
            print(f"Loading pretrained LAM from: {lam_checkpoint}")
            lam_module = LAM.load_from_checkpoint(
                lam_checkpoint, map_location="cpu",
                image_channels=image_channels,
                lam_model_dim=lam_model_dim,
                lam_latent_dim=lam_latent_dim,
                lam_patch_size=lam_patch_size,
                lam_enc_blocks=lam_enc_blocks,
                lam_dec_blocks=lam_dec_blocks,
                lam_num_heads=lam_num_heads,
            )
            self.lam = lam_module.lam
            del lam_module
        else:
            print("No LAM checkpoint provided, initializing from scratch")
            self.lam = LatentActionModel(
                in_dim=image_channels,
                model_dim=lam_model_dim,
                latent_dim=lam_latent_dim,
                patch_size=lam_patch_size,
                enc_blocks=lam_enc_blocks,
                dec_blocks=lam_dec_blocks,
                num_heads=lam_num_heads,
            )

        # =====================================================================
        # 2. MLP projection head: latent_dim (32) → clip_dim (768)
        # =====================================================================
        # Two-layer MLP with GELU and LayerNorm.
        # Maps the 32-dim canonical action to 768-dim CLIP embedding space.
        self.projection = nn.Sequential(
            nn.Linear(lam_latent_dim, projection_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(projection_hidden_dim),
            nn.Linear(projection_hidden_dim, clip_dim),
        )

        # =====================================================================
        # 3. CLIP text encoder (frozen) + precomputed template embeddings
        # =====================================================================
        # Load CLIP text encoder and tokenizer.
        # Only used once during __init__ to precompute embeddings for all 174
        # SSv2 action templates. Then the encoder is deleted to save memory.
        # Lazy import: transformers + huggingface-hub can interfere with wandb
        # service initialization if imported at module level before wandb.init()
        from transformers import CLIPTokenizer, CLIPTextModel

        print(f"Loading CLIP text encoder: {clip_model_name}")
        tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        text_encoder = CLIPTextModel.from_pretrained(clip_model_name)
        text_encoder.eval()

        # Load all SSv2 action templates from labels.json
        # labels.json maps template_string → class_index
        # e.g. {"Approaching something with your camera": "0", ...}
        with open(labels_json) as f:
            template2idx = json.load(f)

        # Sort by index for deterministic ordering
        self.all_templates = sorted(template2idx.keys(), key=lambda t: int(template2idx[t]))
        # Build lookup that handles both formats:
        # labels.json uses "Approaching something with your camera"
        # train.json uses "Approaching [something] with your camera"
        # We normalize by stripping brackets so both formats map to the same index
        self.template_to_idx = {}
        for i, t in enumerate(self.all_templates):
            self.template_to_idx[t] = i                                        # original (no brackets)
            self.template_to_idx[self._add_brackets(t)] = i                    # with brackets
        print(f"  Loaded {len(self.all_templates)} SSv2 action templates")

        # Precompute CLIP embeddings for ALL templates (174 total — trivial cost)
        # These never change, so we store them as a frozen buffer.
        with torch.no_grad():
            tokens = tokenizer(
                self.all_templates, padding=True, truncation=True,
                max_length=77, return_tensors="pt"
            )
            outputs = text_encoder(**tokens)
            template_embeddings = F.normalize(outputs.pooler_output, p=2, dim=-1)

        # Register as buffer: automatically moves to the right device (GPU)
        # Shape: (174, 768)
        self.register_buffer("template_embeddings", template_embeddings)
        print(f"  Precomputed CLIP embeddings: {self.template_embeddings.shape}")

        # Free CLIP model — we only needed it for precomputation
        del text_encoder, tokenizer

        # =====================================================================
        # 4. Learnable temperature for contrastive loss
        # =====================================================================
        # Log-scale so temperature = exp(log_temperature) is always positive.
        # Initialized to 1/0.07 ≈ 14.3 (same as CLIP).
        self.log_temperature = nn.Parameter(
            torch.tensor(np.log(1.0 / temperature_init))
        )

    @staticmethod
    def _add_brackets(template: str) -> str:
        """Convert 'Approaching something with your camera'
        to 'Approaching [something] with your camera'."""
        import re
        result = template
        for word in ["something", "somewhere", "number of", "part"]:
            result = result.replace(word, f"[{word}]")
        return result

    # =========================================================================
    # Core methods
    # =========================================================================

    def sample_pairs(
            self, videos: Tensor, num_frames: Tensor
    ) -> Tuple[Tensor, List[List[int]], Tensor]:
        """Sample consecutive frame pairs evenly across each video.

        For a video with T frames, we sample num_pairs_per_video values of t
        evenly from [0, T-2], then take pairs (f_t, f_{t+1}).

        This ensures:
        - Pairs are truly consecutive (1 frame apart) — in-distribution for LAM
        - Coverage spans the full video (beginning to end)
        - Fixed number of pairs per video for efficient batching

        Args:
            videos: (B, T_max, H, W, C) padded video batch
            num_frames: (B,) actual frame count per video

        Returns:
            all_pairs: (total_pairs, 2, H, W, C) — all pairs stacked
            pair_indices: list of B lists, each containing pair start indices
            pairs_per_video: (B,) — number of pairs per video
        """
        B = videos.shape[0]
        K = self.num_pairs_per_video
        all_pairs = []
        pair_indices = []
        pairs_per_video = []

        for i in range(B):
            T_i = num_frames[i].item()
            num_possible = T_i - 1  # total consecutive pairs in this video

            if num_possible <= K:
                # Video is short — use ALL consecutive pairs
                indices = list(range(num_possible))
            else:
                # Sample K pairs evenly spaced across [0, num_possible-1]
                indices = torch.linspace(0, num_possible - 1, K).long().tolist()

            for t in indices:
                all_pairs.append(videos[i, t:t + 2])  # (2, H, W, C)
            pair_indices.append(indices)
            pairs_per_video.append(len(indices))

        all_pairs = torch.stack(all_pairs)  # (total_pairs, 2, H, W, C)
        pairs_per_video = torch.tensor(pairs_per_video, device=videos.device)

        return all_pairs, pair_indices, pairs_per_video

    def process_pairs(
            self, all_pairs: Tensor, pairs_per_video: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Run LAM.forward() on all pairs in chunks, compute reconstruction loss
        and collect z_mu for canonical action computation.

        Each chunk goes through the full LAM (encode + decode), giving us:
        - z_mu: for aggregation into canonical action
        - recon: for reconstruction loss (MSE + KL)

        Reconstruction loss is accumulated across chunks and averaged at the end,
        so all pairs contribute equally regardless of chunk boundaries.

        Args:
            all_pairs: (total_pairs, 2, H, W, C)
            pairs_per_video: (B,) — number of pairs per video

        Returns:
            canonical_actions: (B, latent_dim) — mean-pooled z_mu per video
            recon_loss: scalar — MSE + beta * KL averaged over all pairs
        """
        total_pairs = all_pairs.shape[0]
        chunk_size = self.max_pairs_per_chunk

        all_z_mu = []
        mse_sum = torch.tensor(0.0, device=all_pairs.device)
        kl_sum = torch.tensor(0.0, device=all_pairs.device)

        for start in range(0, total_pairs, chunk_size):
            chunk = all_pairs[start:start + chunk_size]  # (C, 2, H, W, C)
            C = chunk.shape[0]

            # Full LAM forward: encode + decode
            outputs = self.lam({"videos": chunk})

            # Collect z_mu for canonical action
            all_z_mu.append(outputs["z_mu"])  # (C, latent_dim)

            # Reconstruction loss on this chunk
            gt = chunk[:, 1:]  # (C, 1, H, W, C_img)
            mse = ((gt - outputs["recon"]) ** 2).mean()
            kl = (-0.5 * torch.sum(
                1 + outputs["z_var"] - outputs["z_mu"] ** 2
                - outputs["z_var"].exp(), dim=1
            )).mean()

            mse_sum = mse_sum + mse
            kl_sum = kl_sum + kl

        # Average across chunks
        num_chunks = (total_pairs + chunk_size - 1) // chunk_size
        mse_avg = mse_sum / num_chunks
        kl_avg = kl_sum / num_chunks
        recon_loss = mse_avg + self.beta * kl_avg

        # Concatenate all z_mu and mean pool per video
        all_z_mu = torch.cat(all_z_mu, dim=0)  # (total_pairs, latent_dim)

        # Split by video and mean pool
        B = len(pairs_per_video)
        canonical_actions = torch.zeros(B, self.lam_latent_dim, device=all_pairs.device)
        offset = 0
        for i in range(B):
            n = pairs_per_video[i].item()
            canonical_actions[i] = all_z_mu[offset:offset + n].mean(dim=0)
            offset += n

        return canonical_actions, recon_loss, mse_avg, kl_avg

    def lookup_clip_embeddings(self, templates: List[str]) -> Tensor:
        """Look up precomputed CLIP embeddings for the given templates.

        Since we precomputed embeddings for all 174 templates in __init__,
        this is just an index lookup — no CLIP forward pass needed.

        Args:
            templates: list of B template strings

        Returns:
            embeddings: (B, clip_dim) — L2-normalized CLIP text embeddings
        """
        indices = []
        for t in templates:
            if t in self.template_to_idx:
                indices.append(self.template_to_idx[t])
            else:
                # Fallback: find closest template (shouldn't happen with clean data)
                print(f"[WARNING] Unknown template: {t}, using index 0")
                indices.append(0)
        indices = torch.tensor(indices, device=self.device)
        return self.template_embeddings[indices]  # (B, clip_dim), already normalized

    def compute_alignment_loss(
            self, u: Tensor, v: Tensor
    ) -> Tuple[Tensor, Dict]:
        """Compute alignment loss between projected actions (u) and CLIP embeddings (v).

        Two modes:
        - Regression: L = 1 - mean(cos_sim(u_i, v_i))
          Simple pairwise matching. Each video's projected action should match
          its own template embedding.

        - Contrastive (InfoNCE): symmetric cross-entropy over the similarity matrix.
          u_i should be most similar to v_i and dissimilar to all v_j (j≠i).
          This provides stronger learning signal by using negatives from the batch.

        Args:
            u: (B, clip_dim) — L2-normalized projected canonical actions
            v: (B, clip_dim) — L2-normalized CLIP text embeddings

        Returns:
            loss: scalar
            info: dict with metrics
        """
        if self.loss_type == "regression":
            cos_sim = (u * v).sum(dim=-1)  # (B,)
            loss = (1 - cos_sim).mean()
            info = {
                "align_loss": loss,
                "mean_cos_sim": cos_sim.mean(),
            }

        elif self.loss_type == "contrastive":
            temperature = self.log_temperature.exp()
            logits = (u @ v.T) * temperature  # (B, B)
            labels = torch.arange(len(u), device=u.device)

            # Symmetric InfoNCE: average of u→v and v→u cross-entropy
            loss_u2v = F.cross_entropy(logits, labels)
            loss_v2u = F.cross_entropy(logits.T, labels)
            loss = (loss_u2v + loss_v2u) / 2

            with torch.no_grad():
                cos_sim = (u * v).sum(dim=-1)
                acc_u2v = (logits.argmax(dim=1) == labels).float().mean()
                acc_v2u = (logits.T.argmax(dim=1) == labels).float().mean()

            info = {
                "align_loss": loss,
                "mean_cos_sim": cos_sim.mean(),
                "temperature": temperature,
                "acc_u2v": acc_u2v,
                "acc_v2u": acc_v2u,
            }
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        return loss, info

    # =========================================================================
    # Lightning training / validation
    # =========================================================================

    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:
        videos = batch["videos"]          # (B, T_max, H, W, C)
        num_frames = batch["num_frames"]  # (B,)
        templates = batch["templates"]    # list of B strings

        # Step 1: Sample consecutive pairs evenly across each video
        all_pairs, pair_indices, pairs_per_video = self.sample_pairs(videos, num_frames)

        # Step 2: Process all pairs through LAM (encode + decode)
        #   → canonical_actions: (B, 32) mean-pooled z_mu
        #   → recon_loss: MSE + KL on ALL pairs
        canonical_actions, recon_loss, mse_loss, kl_loss = self.process_pairs(
            all_pairs, pairs_per_video
        )

        # Step 3: Project canonical action to CLIP space via MLP
        u = self.projection(canonical_actions)  # (B, 768)
        u = F.normalize(u, p=2, dim=-1)

        # Step 4: Look up precomputed CLIP embeddings (no forward pass)
        v = self.lookup_clip_embeddings(templates)  # (B, 768)

        # Step 5: Compute alignment loss
        align_loss, align_info = self.compute_alignment_loss(u, v)

        # Step 6: Total loss = reconstruction + alignment (equal weight)
        total_loss = recon_loss + align_loss

        # --- Logging ---
        log_dict = {
            "train/total_loss": total_loss,
            "train/align_loss": align_info["align_loss"],
            "train/recon_loss": recon_loss,
            "train/mse_loss": mse_loss,
            "train/kl_loss": kl_loss,
            "train/mean_cos_sim": align_info["mean_cos_sim"],
        }
        if self.loss_type == "contrastive":
            log_dict["train/temperature"] = align_info["temperature"]
            log_dict["train/acc_u2v"] = align_info["acc_u2v"]
            log_dict["train/acc_v2u"] = align_info["acc_v2u"]

        self.log_dict(log_dict, prog_bar=True, logger=True,
                      on_step=True, on_epoch=True, sync_dist=True)
        self.log("global_step", self.global_step, prog_bar=True,
                 logger=True, on_step=True, on_epoch=False)

        # --- WandB image logging (rank 0 only, every log_interval steps) ---
        if batch_idx % self.log_interval == 0 and self.global_rank == 0:
            # Use first pair of first video for visualization
            with torch.no_grad():
                viz_pair = all_pairs[:1]  # (1, 2, H, W, C)
                viz_out = self.lam({"videos": viz_pair})
            self._wandb_cache = {
                "input": viz_pair[0, 0].clamp(0, 1).detach().cpu(),
                "gt": viz_pair[0, 1].clamp(0, 1).detach().cpu(),
                "pred": viz_out["recon"][0, 0].clamp(0, 1).detach().cpu(),
                "template": templates[0],
                "cos_sim": align_info["mean_cos_sim"].item(),
                "step": self.global_step,
            }

        return total_loss

    def on_train_batch_end(self, out, batch, batch_idx) -> None:
        if not hasattr(self, "_wandb_cache"):
            return
        cache = self._wandb_cache
        del self._wandb_cache

        wandb_logger = None
        for logger in self.loggers:
            if type(logger).__name__ == "WandbLogger":
                wandb_logger = logger
                break
        if wandb_logger is None:
            return

        def _upload(cache, wandb_logger):
            try:
                import wandb
                inp = (cache["input"].numpy() * 255).astype(np.uint8)
                gt = (cache["gt"].numpy() * 255).astype(np.uint8)
                pred = (cache["pred"].numpy() * 255).astype(np.uint8)
                row1 = np.concatenate([inp, gt], axis=1)
                row2 = np.concatenate([inp, pred], axis=1)
                grid = np.concatenate([row1, row2], axis=0)
                wandb_logger.experiment.log({
                    "train/frame_comparison": wandb.Image(
                        grid,
                        caption=f"Top: input→GT | Bottom: input→pred\n"
                                f"template: {cache['template']}\n"
                                f"cos_sim: {cache['cos_sim']:.3f}"),
                    "trainer/global_step": cache["step"],
                })
            except Exception as e:
                print(f"[WandB] Image upload failed: {e}")

        thread = threading.Thread(target=_upload, args=(cache, wandb_logger), daemon=True)
        thread.start()

    @torch.no_grad()
    def validation_step(self, batch: Dict, batch_idx: int) -> Tensor:
        videos = batch["videos"]
        num_frames = batch["num_frames"]
        templates = batch["templates"]

        all_pairs, pair_indices, pairs_per_video = self.sample_pairs(videos, num_frames)
        canonical_actions, recon_loss, mse_loss, kl_loss = self.process_pairs(
            all_pairs, pairs_per_video
        )
        u = self.projection(canonical_actions)
        u = F.normalize(u, p=2, dim=-1)
        v = self.lookup_clip_embeddings(templates)
        align_loss, align_info = self.compute_alignment_loss(u, v)
        total_loss = recon_loss + align_loss

        log_dict = {
            "val/total_loss": total_loss,
            "val/align_loss": align_info["align_loss"],
            "val/recon_loss": recon_loss,
            "val/mean_cos_sim": align_info["mean_cos_sim"],
        }
        if self.loss_type == "contrastive":
            log_dict["val/acc_u2v"] = align_info["acc_u2v"]
            log_dict["val/acc_v2u"] = align_info["acc_v2u"]

        self.log_dict(log_dict, prog_bar=True, logger=True,
                      on_step=True, on_epoch=True, sync_dist=True)
        return total_loss

    def configure_optimizers(self) -> Optimizer:
        # Optimize: LAM (finetuned) + MLP projection + temperature
        # CLIP text encoder is NOT included — it was deleted after precomputation
        # Separate LRs: LAM gets lower LR (large pretrained model),
        # MLP + temperature get higher LR (small, randomly initialized)
        params = [
            {"params": self.lam.parameters(), "lr": self.lr_lam},
            {"params": self.projection.parameters(), "lr": self.lr_projection},
            {"params": [self.log_temperature], "lr": self.lr_projection},
        ]
        return AdamW(params, weight_decay=self.weight_decay)
