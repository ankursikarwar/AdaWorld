"""
SSv2 dataset for CLIP-aligned LAM training.

Loads full video clips (all frames) paired with action template strings.
Templates use [something] placeholders to strip object specifics,
e.g. "Pushing [something] from left to right".

Key differences from VideoDataset:
- Loads ALL frames (up to max_frames), not just 2
- Returns action template string alongside video
- Custom collate handles variable-length videos via padding + num_frames mask
- Filters out videos without SSv2 labels
"""
import json
from os import listdir, path
from random import randint
from typing import Any, Callable, Dict, List

import cv2 as cv
import torch
import torch.nn.functional as F
from einops import rearrange
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class SSv2ClipDataset(Dataset):
    """Loads full SSv2 video clips with their action template labels."""

    def __init__(
            self,
            split_path: str,
            labels_dir: str,
            max_frames: int = 80,
            resolution: int = 256,
            output_format: str = "t h w c",
    ) -> None:
        super().__init__()
        self.max_frames = max_frames
        self.resolution = resolution
        self.output_format = output_format

        # --- Load SSv2 labels: video_id -> template ---
        # Templates use [something] placeholders, stripping object specifics
        # e.g. "Pushing [something] from left to right" instead of "Pushing cup..."
        self.id2template = {}
        for fname in ["train.json", "validation.json"]:
            label_path = path.join(labels_dir, fname)
            if path.exists(label_path):
                with open(label_path) as f:
                    for entry in json.load(f):
                        self.id2template[entry["id"]] = entry["template"]

        # --- Collect video files, filter to only those with labels ---
        cache_path = split_path.rstrip("/") + ".filelist"
        if path.exists(cache_path):
            with open(cache_path) as f:
                all_files = [line.strip() for line in f if line.strip()]
        else:
            all_files = []
            for file_name in listdir(split_path):
                if file_name.endswith(".mp4") or file_name.endswith(".webm"):
                    all_files.append(path.join(split_path, file_name))

        # Only keep videos that have a template label
        self.file_names = []
        self.templates = []
        for fpath in all_files:
            vid = path.basename(fpath).replace(".webm", "").replace(".mp4", "")
            if vid in self.id2template:
                self.file_names.append(fpath)
                self.templates.append(self.id2template[vid])

        print(f"SSv2ClipDataset: {len(self.file_names)}/{len(all_files)} videos have labels")

        # Collect unique templates for reference
        self.unique_templates = sorted(set(self.templates))
        print(f"  Unique action templates: {len(self.unique_templates)}")

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, idx: int) -> Dict:
        video_path = self.file_names[idx]
        template = self.templates[idx]
        while True:
            try:
                video, num_frames = self.load_full_video(video_path)
                return {
                    "videos": video,           # (T, H, W, C) float32, T <= max_frames
                    "num_frames": num_frames,   # actual number of frames (int)
                    "template": template,       # action template string
                }
            except Exception:
                # On failure, pick a random different video
                idx = randint(0, len(self) - 1)
                video_path = self.file_names[idx]
                template = self.templates[idx]

    def load_full_video(self, video_path: str) -> tuple:
        """Load all frames from video (up to max_frames).

        Returns:
            video: (T, H, W, C) tensor, T = min(total_frames, max_frames)
            num_frames: actual number of frames loaded
        """
        cap = cv.VideoCapture(video_path)
        frames = []
        for _ in range(self.max_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame)
            frames.append(frame)
        cap.release()

        if len(frames) < 2:
            raise ValueError(f"Video too short ({len(frames)} frames): {video_path}")

        num_frames = len(frames)
        video = torch.stack(frames) / 255.0  # (T, H, W, C)

        # Center crop to square
        if video.shape[1] != video.shape[2]:
            square_len = min(video.shape[1], video.shape[2])
            h_crop = (video.shape[1] - square_len) // 2
            w_crop = (video.shape[2] - square_len) // 2
            video = video[:, h_crop:h_crop + square_len, w_crop:w_crop + square_len]

        # Resize to target resolution
        if video.shape[1] != self.resolution or video.shape[2] != self.resolution:
            video = rearrange(video, "t h w c -> c t h w")
            video = F.interpolate(video, self.resolution, mode="bicubic")
            video = rearrange(video, f"c t h w -> {self.output_format}")
        else:
            video = rearrange(video, f"t h w c -> {self.output_format}")

        return video, num_frames


def collate_ssv2_clip(batch: List[Dict]) -> Dict:
    """Custom collate that pads variable-length videos to the longest in the batch.

    Returns:
        videos: (B, T_max, H, W, C) - padded with zeros
        num_frames: (B,) - actual frame count per video
        templates: list of B template strings
    """
    max_t = max(item["num_frames"] for item in batch)
    H, W, C = batch[0]["videos"].shape[1], batch[0]["videos"].shape[2], batch[0]["videos"].shape[3]

    padded_videos = torch.zeros(len(batch), max_t, H, W, C)
    num_frames = torch.zeros(len(batch), dtype=torch.long)
    templates = []

    for i, item in enumerate(batch):
        t = item["num_frames"]
        padded_videos[i, :t] = item["videos"][:t]
        num_frames[i] = t
        templates.append(item["template"])

    return {
        "videos": padded_videos,
        "num_frames": num_frames,
        "templates": templates,
    }


class LightningSSv2ClipDataset(LightningDataModule):
    """Lightning DataModule for SSv2 CLIP-aligned training."""

    def __init__(
            self,
            data_root: str = "../data",
            labels_dir: str = "/home/mila/a/ankur.sikarwar/scratch/WORLD_MODEL_PROJECT/data/ssv2/raw/labels",
            max_frames: int = 80,
            resolution: int = 256,
            output_format: str = "t h w c",
            batch_size: int = 8,
            num_workers: int = 8,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.labels_dir = labels_dir
        self.max_frames = max_frames
        self.resolution = resolution
        self.output_format = output_format
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.save_hyperparameters()

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = SSv2ClipDataset(
                split_path=path.join(self.data_root, "ssv2_our", "train"),
                labels_dir=self.labels_dir,
                max_frames=self.max_frames,
                resolution=self.resolution,
                output_format=self.output_format,
            )
            self.val_dataset = SSv2ClipDataset(
                split_path=path.join(self.data_root, "ssv2_our", "test"),
                labels_dir=self.labels_dir,
                max_frames=self.max_frames,
                resolution=self.resolution,
                output_format=self.output_format,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_ssv2_clip,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_ssv2_clip,
            drop_last=False,
        )
