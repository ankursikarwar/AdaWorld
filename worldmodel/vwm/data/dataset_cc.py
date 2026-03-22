"""
Cross-cluster (cc) dataloaders that read from HuggingFace / local parquet files
instead of raw video directories. Drop-in replacement for dataset.py classes.

The parquet files store raw video bytes. At load time, bytes are written to a
temp file and decoded with OpenCV — identical preprocessing to the original.
"""

import ast
import math
import os
import tempfile
from os import path
from random import choices, randint
from typing import Dict, List, Tuple

import cv2 as cv
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from einops import rearrange
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def decode_video_from_bytes(video_bytes: bytes, video_format: str = "mp4") -> cv.VideoCapture:
    """Write video bytes to a temp file and return an OpenCV VideoCapture."""
    suffix = f".{video_format}"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(video_bytes)
    tmp.flush()
    tmp.close()
    cap = cv.VideoCapture(tmp.name)
    return cap, tmp.name


class ParquetVideoDataset(Dataset):
    """
    Reads video bytes from parquet shards. Replaces VideoDataset.
    Each parquet row has: video_bytes, video_format, dataset, environment, split, filename.
    """

    def __init__(
            self,
            parquet_paths: List[str],
            dataset_name: str = "",
            padding: str = "repeat",
            randomize: bool = False,
            resolution: int = 256,
            n_context_frames: int = 5,
            output_format: str = "t c h w",
            color_aug: bool = True
    ):
        super().__init__()
        self.padding = padding
        self.randomize = randomize
        self.resolution = resolution
        self.n_context_frames = n_context_frames
        self.output_format = output_format
        self.color_aug = color_aug
        self.dataset_name = dataset_name

        # Load all parquet files into a single table and keep in memory as columns
        tables = []
        for p in sorted(parquet_paths):
            tables.append(pq.read_table(p))
        self.table = tables[0] if len(tables) == 1 else __import__("pyarrow").concat_tables(tables)
        self.length = len(self.table)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict:
        while True:
            try:
                row = self.table.slice(idx, 1)
                video_bytes = row.column("video_bytes")[0].as_py()
                video_format = row.column("video_format")[0].as_py()
                ds_name = row.column("dataset")[0].as_py()
                image_seq = self.load_video_slice_from_bytes(
                    video_bytes, video_format, ds_name,
                    self.n_context_frames + 1,
                    None if self.randomize else 0
                )
                return self.build_data_dict(image_seq)
            except Exception:
                idx = randint(0, self.length - 1)

    def load_video_slice_from_bytes(
            self,
            video_bytes: bytes,
            video_format: str,
            dataset_name: str,
            num_frames: int,
            start_frame: int = None,
            frame_skip: int = 1
    ) -> List:
        cap, tmp_path = decode_video_from_bytes(video_bytes, video_format)
        try:
            total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

            # Frame skip logic matching original dataset.py
            if "retro" in dataset_name:
                frame_skip = 4
            elif "procgen" not in dataset_name and "ssv2" not in dataset_name and "mira" not in dataset_name:
                frame_skip = 2
            num_frames = num_frames * frame_skip

            start_frame = randint(0, max(0, total_frames - num_frames)) if start_frame is None else start_frame
            cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
            frames = []
            for _ in range(num_frames):
                ret, frame = cap.read()
                if ret:
                    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                    frame = torch.from_numpy(frame)
                    frames.append(frame)
                else:
                    if self.padding == "none":
                        pass
                    elif self.padding == "repeat":
                        frames.extend([frames[-1]] * (num_frames - len(frames)))
                    elif self.padding == "zero":
                        frames.extend([torch.zeros_like(frames[-1])] * (num_frames - len(frames)))
                    elif self.padding == "random":
                        frames.extend([torch.rand_like(frames[-1])] * (num_frames - len(frames)))
                    else:
                        raise ValueError(f"Invalid padding type: {self.padding}")
                    break
        finally:
            cap.release()
            os.unlink(tmp_path)

        video = torch.stack(frames[::frame_skip]) / 255.0

        # Crop to square
        if video.shape[1] != video.shape[2]:
            square_len = min(video.shape[1], video.shape[2])
            h_crop = (video.shape[1] - square_len) // 2
            w_crop = (video.shape[2] - square_len) // 2
            video = video[:, h_crop:h_crop + square_len, w_crop:w_crop + square_len]

        if video.shape[-2] != self.resolution or video.shape[-3] != self.resolution:
            video = rearrange(video, "t h w c -> t c h w")
            video = F.interpolate(video, self.resolution, mode="bicubic")
            video = rearrange(video, f"t c h w -> {self.output_format}")
        else:
            video = rearrange(video, f"t h w c -> {self.output_format}")

        if self.color_aug:
            video = (video + torch.rand(1) * 0.2 - 0.1).clamp(0, 1)
        return [frame for frame in video]

    def build_data_dict(self, image_seq: List) -> Dict:
        context_len = choices(range(1, self.n_context_frames + 1))[0]
        filled_seq = [torch.zeros_like(image_seq[0])] * (self.n_context_frames - context_len) + image_seq
        next_frames = torch.Tensor(filled_seq[self.n_context_frames])
        prev_frames = torch.stack(filled_seq[:self.n_context_frames])
        lam_inputs = torch.stack(filled_seq[self.n_context_frames - 1:self.n_context_frames + 1])
        context_len = torch.Tensor([context_len])
        next_frames = next_frames * 2.0 - 1.0
        prev_frames = prev_frames * 2.0 - 1.0
        context_aug = torch.Tensor(choices(range(8))) / 10
        img_seq = torch.cat([prev_frames, next_frames[None]])
        data_dict = {
            "img_seq": img_seq,
            "cond_frames_without_noise": prev_frames[-1],
            "cond_frames": prev_frames[-1] + 0.02 * torch.randn_like(prev_frames[-1]),
            "lam_inputs": lam_inputs,
            "context_len": context_len,
            "context_aug": context_aug
        }
        return data_dict


class ParquetVideoDatasetDiscreteActionSpace(Dataset):
    """
    Reads video bytes from parquet for discrete-action adaptation.
    Expects parquet rows with an additional 'action' column (int).
    """

    def __init__(
            self,
            parquet_paths: List[str],
            randomize: bool = False,
            resolution: int = 256,
            n_context_frames: int = 5,
            output_format: str = "t c h w",
            color_aug: bool = True
    ):
        super().__init__()
        self.randomize = randomize
        self.resolution = resolution
        self.n_context_frames = n_context_frames
        self.output_format = output_format
        self.color_aug = color_aug

        tables = []
        for p in sorted(parquet_paths):
            tables.append(pq.read_table(p))
        self.table = tables[0] if len(tables) == 1 else __import__("pyarrow").concat_tables(tables)
        self.length = len(self.table)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict:
        while True:
            try:
                row = self.table.slice(idx, 1)
                video_bytes = row.column("video_bytes")[0].as_py()
                video_format = row.column("video_format")[0].as_py()
                action = row.column("action")[0].as_py()
                image_seq = self.load_video_slice_from_bytes(
                    video_bytes, video_format, self.n_context_frames + 1
                )
                raw_action = torch.Tensor([int(action)])
                return self.build_data_dict(image_seq, raw_action)
            except Exception:
                idx = randint(0, self.length - 1)

    def load_video_slice_from_bytes(
            self,
            video_bytes: bytes,
            video_format: str,
            num_frames: int
    ) -> List:
        cap, tmp_path = decode_video_from_bytes(video_bytes, video_format)
        try:
            total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
            assert num_frames == total_frames

            frames = []
            for _ in range(num_frames):
                ret, frame = cap.read()
                if ret:
                    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                    frame = torch.from_numpy(frame)
                    frames.append(frame)
                else:
                    raise NotImplementedError
        finally:
            cap.release()
            os.unlink(tmp_path)

        video = torch.stack(frames) / 255.0

        if video.shape[1] != video.shape[2]:
            square_len = min(video.shape[1], video.shape[2])
            h_crop = (video.shape[1] - square_len) // 2
            w_crop = (video.shape[2] - square_len) // 2
            video = video[:, h_crop:h_crop + square_len, w_crop:w_crop + square_len]

        if video.shape[-2] != self.resolution or video.shape[-3] != self.resolution:
            video = rearrange(video, "t h w c -> t c h w")
            video = F.interpolate(video, self.resolution, mode="bicubic")
            video = rearrange(video, f"t c h w -> {self.output_format}")
        else:
            video = rearrange(video, f"t h w c -> {self.output_format}")

        if self.color_aug:
            video = (video + torch.rand(1) * 0.2 - 0.1).clamp(0, 1)
        return [frame for frame in video]

    def build_data_dict(self, image_seq: List, raw_action: torch.Tensor) -> Dict:
        context_len = choices(range(1, self.n_context_frames + 1))[0]
        image_seq = image_seq[self.n_context_frames - context_len:]
        filled_seq = [torch.zeros_like(image_seq[0])] * (self.n_context_frames - context_len) + image_seq
        next_frames = torch.Tensor(filled_seq[self.n_context_frames])
        prev_frames = torch.stack(filled_seq[:self.n_context_frames])
        context_len = torch.Tensor([context_len])
        next_frames = next_frames * 2.0 - 1.0
        prev_frames = prev_frames * 2.0 - 1.0
        context_aug = torch.Tensor(choices(range(8))) / 10
        img_seq = torch.cat([prev_frames, next_frames[None]])
        data_dict = {
            "img_seq": img_seq,
            "cond_frames_without_noise": prev_frames[-1],
            "cond_frames": prev_frames[-1] + 0.02 * torch.randn_like(prev_frames[-1]),
            "context_len": context_len,
            "context_aug": context_aug,
            "raw_action": raw_action
        }
        return data_dict


class ParquetVideoDatasetContinuousActionSpace(Dataset):
    """
    Reads video bytes from parquet for continuous-action adaptation.
    Expects parquet rows with an additional 'action' column (string repr of list).
    """

    def __init__(
            self,
            parquet_paths: List[str],
            randomize: bool = False,
            resolution: int = 256,
            n_context_frames: int = 5,
            output_format: str = "t c h w",
            color_aug: bool = True
    ):
        super().__init__()
        self.randomize = randomize
        self.resolution = resolution
        self.n_context_frames = n_context_frames
        self.output_format = output_format
        self.color_aug = color_aug

        tables = []
        for p in sorted(parquet_paths):
            tables.append(pq.read_table(p))
        self.table = tables[0] if len(tables) == 1 else __import__("pyarrow").concat_tables(tables)
        self.length = len(self.table)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict:
        while True:
            try:
                row = self.table.slice(idx, 1)
                video_bytes = row.column("video_bytes")[0].as_py()
                video_format = row.column("video_format")[0].as_py()
                action_str = row.column("action")[0].as_py()
                raw_action = ast.literal_eval(action_str)
                image_seq = self.load_video_slice_from_bytes(
                    video_bytes, video_format, self.n_context_frames + 1
                )
                return self.build_data_dict(image_seq, raw_action)
            except Exception:
                idx = randint(0, self.length - 1)

    def load_video_slice_from_bytes(
            self,
            video_bytes: bytes,
            video_format: str,
            num_frames: int
    ) -> List:
        cap, tmp_path = decode_video_from_bytes(video_bytes, video_format)
        try:
            total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
            assert num_frames == total_frames

            frames = []
            for _ in range(num_frames):
                ret, frame = cap.read()
                if ret:
                    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                    frame = torch.from_numpy(frame)
                    frames.append(frame)
                else:
                    raise NotImplementedError
        finally:
            cap.release()
            os.unlink(tmp_path)

        video = torch.stack(frames) / 255.0

        if video.shape[1] != video.shape[2]:
            square_len = min(video.shape[1], video.shape[2])
            h_crop = (video.shape[1] - square_len) // 2
            w_crop = (video.shape[2] - square_len) // 2
            video = video[:, h_crop:h_crop + square_len, w_crop:w_crop + square_len]

        if video.shape[-2] != self.resolution or video.shape[-3] != self.resolution:
            video = rearrange(video, "t h w c -> t c h w")
            video = F.interpolate(video, self.resolution, mode="bicubic")
            video = rearrange(video, f"t c h w -> {self.output_format}")
        else:
            video = rearrange(video, f"t h w c -> {self.output_format}")

        if self.color_aug:
            video = (video + torch.rand(1) * 0.2 - 0.1).clamp(0, 1)
        return [frame for frame in video]

    def build_data_dict(self, image_seq: List, raw_action: List) -> Dict:
        context_len = choices(range(1, self.n_context_frames + 1))[0]
        image_seq = image_seq[self.n_context_frames - context_len:]
        filled_seq = [torch.zeros_like(image_seq[0])] * (self.n_context_frames - context_len) + image_seq
        next_frames = torch.Tensor(filled_seq[self.n_context_frames])
        prev_frames = torch.stack(filled_seq[:self.n_context_frames])
        context_len = torch.Tensor([context_len])
        next_frames = next_frames * 2.0 - 1.0
        prev_frames = prev_frames * 2.0 - 1.0
        context_aug = torch.Tensor(choices(range(8))) / 10
        img_seq = torch.cat([prev_frames, next_frames[None]])
        data_dict = {
            "img_seq": img_seq,
            "cond_frames_without_noise": prev_frames[-1],
            "cond_frames": prev_frames[-1] + 0.02 * torch.randn_like(prev_frames[-1]),
            "context_len": context_len,
            "context_aug": context_aug,
            "raw_action": torch.Tensor(raw_action)
        }
        return data_dict


class MultiSourceParquetSamplerDataset(Dataset):
    """
    Replaces MultiSourceSamplerDataset. Loads parquet shards grouped by
    dataset/environment and applies the same weighted sampling.
    """

    def __init__(
            self,
            data_root: str,
            env_source: str = "game",
            split: str = "train",
            samples_per_epoch: int = 60000,
            sampling_strategy: str = "pi",
            **kwargs
    ):
        self.samples_per_epoch = samples_per_epoch

        # data_root should point to the parquet output directory
        # Structure: data_root/{dataset}/{split}/shard_*.parquet
        shard_groups = []  # list of (group_name, [parquet_paths])

        if env_source in ("procgen", "retro", "game"):
            sources = []
            if env_source in ("procgen", "game"):
                sources.append("procgen")
            if env_source in ("retro", "game"):
                sources.append("retro")

            for src in sources:
                split_dir = os.path.join(data_root, src, split)
                if not os.path.isdir(split_dir):
                    continue
                # All shards for this source+split form one group per environment
                # Since parquet shards mix environments, we load all shards as one group
                # and the sampling is across individual rows
                shards = sorted([
                    os.path.join(split_dir, f)
                    for f in os.listdir(split_dir) if f.endswith(".parquet")
                ])
                if shards:
                    # Load table and split by environment for weighted sampling
                    table = pq.read_table(shards[0] if len(shards) == 1 else split_dir,
                                          use_pandas_metadata=False)
                    if len(shards) > 1:
                        import pyarrow as pa
                        tables = [pq.read_table(s) for s in shards]
                        table = pa.concat_tables(tables)

                    # Group by environment
                    env_col = table.column("environment").to_pylist()
                    unique_envs = sorted(set(env_col))
                    for env in unique_envs:
                        mask = [e == env for e in env_col]
                        env_table = table.filter(mask)
                        shard_groups.append((f"{src}/{env}", env_table))

        elif env_source in ("mira", "ssv2_our", "ssv2_our_mira"):
            srcs = [env_source] if env_source != "ssv2_our_mira" else ["ssv2_our", "mira"]
            for src in srcs:
                split_dir = os.path.join(data_root, src, split)
                if not os.path.isdir(split_dir):
                    continue
                shards = sorted([
                    os.path.join(split_dir, f)
                    for f in os.listdir(split_dir) if f.endswith(".parquet")
                ])
                if shards:
                    import pyarrow as pa
                    tables = [pq.read_table(s) for s in shards]
                    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
                    shard_groups.append((src, table))
        else:
            # Try as a direct dataset name
            split_dir = os.path.join(data_root, env_source, split)
            if os.path.isdir(split_dir):
                shards = sorted([
                    os.path.join(split_dir, f)
                    for f in os.listdir(split_dir) if f.endswith(".parquet")
                ])
                if shards:
                    import pyarrow as pa
                    tables = [pq.read_table(s) for s in shards]
                    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
                    shard_groups.append((env_source, table))

        if not shard_groups:
            raise ValueError(f"No parquet data found for env_source={env_source}, split={split} in {data_root}")

        # Create sub-datasets from each group
        self.subsets = []
        for group_name, table in tqdm(shard_groups, desc="Loading parquet subsets..."):
            print("Subset:", group_name, f"({len(table)} videos)")
            ds = _TableVideoDataset(table, group_name, **kwargs)
            self.subsets.append(ds)
        print("Number of subsets:", len(self.subsets))

        # Compute sampling probabilities
        if sampling_strategy == "sample":
            probs = [len(d) for d in self.subsets]
        elif sampling_strategy == "dataset":
            probs = [1 for _ in self.subsets]
        elif sampling_strategy == "log":
            probs = [math.log(len(d)) if len(d) else 0 for d in self.subsets]
        elif sampling_strategy == "pi":
            probs = [len(d) ** 0.43 for d in self.subsets]
        else:
            raise ValueError(f"Unavailable sampling strategy: {sampling_strategy}")
        total_prob = sum(probs)
        assert total_prob > 0, "No sample is available"
        self.sample_probs = [x / total_prob for x in probs]

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> Dict:
        subset = choices(self.subsets, self.sample_probs)[0]
        sample_idx = randint(0, len(subset) - 1)
        return subset[sample_idx]


class _TableVideoDataset(Dataset):
    """Internal dataset wrapping a PyArrow table with video bytes."""

    def __init__(
            self,
            table,
            group_name: str = "",
            padding: str = "repeat",
            randomize: bool = False,
            resolution: int = 256,
            n_context_frames: int = 5,
            output_format: str = "t c h w",
            color_aug: bool = True
    ):
        self.table = table
        self.group_name = group_name
        self.padding = padding
        self.randomize = randomize
        self.resolution = resolution
        self.n_context_frames = n_context_frames
        self.output_format = output_format
        self.color_aug = color_aug

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx: int) -> Dict:
        while True:
            try:
                row = self.table.slice(idx, 1)
                video_bytes = row.column("video_bytes")[0].as_py()
                video_format = row.column("video_format")[0].as_py()
                ds_name = row.column("dataset")[0].as_py()

                cap, tmp_path = decode_video_from_bytes(video_bytes, video_format)
                try:
                    image_seq = self._load_frames(cap, ds_name)
                finally:
                    cap.release()
                    os.unlink(tmp_path)

                return self._build_data_dict(image_seq)
            except Exception:
                idx = randint(0, len(self) - 1)

    def _load_frames(self, cap, dataset_name: str) -> List:
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        num_frames = self.n_context_frames + 1
        frame_skip = 1

        if "retro" in dataset_name:
            frame_skip = 4
        elif "procgen" not in dataset_name and "ssv2" not in dataset_name and "mira" not in dataset_name:
            frame_skip = 2
        num_frames = num_frames * frame_skip

        start_frame = randint(0, max(0, total_frames - num_frames)) if self.randomize else 0
        cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        for _ in range(num_frames):
            ret, frame = cap.read()
            if ret:
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame)
                frames.append(frame)
            else:
                if self.padding == "none":
                    pass
                elif self.padding == "repeat":
                    frames.extend([frames[-1]] * (num_frames - len(frames)))
                elif self.padding == "zero":
                    frames.extend([torch.zeros_like(frames[-1])] * (num_frames - len(frames)))
                elif self.padding == "random":
                    frames.extend([torch.rand_like(frames[-1])] * (num_frames - len(frames)))
                break

        video = torch.stack(frames[::frame_skip]) / 255.0

        if video.shape[1] != video.shape[2]:
            square_len = min(video.shape[1], video.shape[2])
            h_crop = (video.shape[1] - square_len) // 2
            w_crop = (video.shape[2] - square_len) // 2
            video = video[:, h_crop:h_crop + square_len, w_crop:w_crop + square_len]

        if video.shape[-2] != self.resolution or video.shape[-3] != self.resolution:
            video = rearrange(video, "t h w c -> t c h w")
            video = F.interpolate(video, self.resolution, mode="bicubic")
            video = rearrange(video, f"t c h w -> {self.output_format}")
        else:
            video = rearrange(video, f"t h w c -> {self.output_format}")

        if self.color_aug:
            video = (video + torch.rand(1) * 0.2 - 0.1).clamp(0, 1)
        return [frame for frame in video]

    def _build_data_dict(self, image_seq: List) -> Dict:
        context_len = choices(range(1, self.n_context_frames + 1))[0]
        filled_seq = [torch.zeros_like(image_seq[0])] * (self.n_context_frames - context_len) + image_seq
        next_frames = torch.Tensor(filled_seq[self.n_context_frames])
        prev_frames = torch.stack(filled_seq[:self.n_context_frames])
        lam_inputs = torch.stack(filled_seq[self.n_context_frames - 1:self.n_context_frames + 1])
        context_len = torch.Tensor([context_len])
        next_frames = next_frames * 2.0 - 1.0
        prev_frames = prev_frames * 2.0 - 1.0
        context_aug = torch.Tensor(choices(range(8))) / 10
        img_seq = torch.cat([prev_frames, next_frames[None]])
        data_dict = {
            "img_seq": img_seq,
            "cond_frames_without_noise": prev_frames[-1],
            "cond_frames": prev_frames[-1] + 0.02 * torch.randn_like(prev_frames[-1]),
            "lam_inputs": lam_inputs,
            "context_len": context_len,
            "context_aug": context_aug
        }
        return data_dict


class VideoDataSampler_cc(LightningDataModule):
    """
    Drop-in replacement for VideoDataSampler that reads from parquet data.
    Use in configs with: target: vwm.data.dataset_cc.VideoDataSampler_cc
    """

    def __init__(
            self,
            data_root: str,
            env_source: str = "game",
            batch_size: int = 1,
            num_workers: int = 8,
            resolution: int = 256,
            n_context_frames: int = 5,
            prefetch_factor: int = 4,
            shuffle: bool = True,
            samples_per_epoch: int = 60000
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor if num_workers > 0 else None
        self.shuffle = shuffle
        self.train_dataset = MultiSourceParquetSamplerDataset(
            data_root=data_root, env_source=env_source, split="train", randomize=True,
            resolution=resolution, n_context_frames=n_context_frames,
            samples_per_epoch=samples_per_epoch
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor
        )

    def test_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor
        )

    def val_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor
        )
