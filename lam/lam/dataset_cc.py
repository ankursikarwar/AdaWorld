"""
Cross-cluster (cc) dataloaders for LAM training.
Reads from parquet files instead of raw video directories.
"""

import math
import os
import tempfile
from os import path
from random import choices, randint
from typing import Any, Callable, Dict

import cv2 as cv
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from einops import rearrange
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data import get_worker_info
from tqdm import tqdm


def exists(var) -> bool:
    return var is not None


def default(var, val) -> Any:
    return var if exists(var) else val


def default_worker_init_fn(worker_id: int) -> None:
    torch.manual_seed(torch.initial_seed() + worker_id)
    worker_info = get_worker_info()

    if exists(worker_info):
        dataset = worker_info.dataset
        glob_start = dataset._start
        glob_end = dataset._end

        per_worker = int((glob_end - glob_start) / worker_info.num_workers)
        worker_id = worker_info.id

        dataset._start = glob_start + worker_id * per_worker
        dataset._end = min(dataset._start + per_worker, glob_end)


def decode_video_from_bytes(video_bytes: bytes, video_format: str = "mp4"):
    """Write video bytes to a temp file and return an OpenCV VideoCapture."""
    suffix = f".{video_format}"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(video_bytes)
    tmp.flush()
    tmp.close()
    cap = cv.VideoCapture(tmp.name)
    return cap, tmp.name


class LightningDataset_cc(LightningDataModule):
    """Same as LightningDataset but for cc dataloaders."""

    def __init__(
            self,
            *args,
            batch_size: int = 8,
            num_workers: int = 16,
            train_shuffle: bool = True,
            val_shuffle: bool = False,
            val_batch_size: int = None,
            worker_init_fn: Callable = None,
            collate_fn: Callable = None,
            train_sampler: Callable = None,
            test_sampler: Callable = None,
            val_sampler: Callable = None
    ) -> None:
        super().__init__()
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

        val_batch_size = default(val_batch_size, batch_size)

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.train_shuffle = train_shuffle
        self.val_shuffle = val_shuffle
        self.train_sampler = train_sampler
        self.test_sampler = test_sampler
        self.val_sampler = val_sampler
        self.collate_fn = collate_fn
        self.worker_init_fn = worker_init_fn

    def train_dataloader(self) -> DataLoader:
        if isinstance(self.train_dataset, IterableDataset):
            worker_init_fn = default(self.worker_init_fn, default_worker_init_fn)
        else:
            worker_init_fn = self.worker_init_fn
        return DataLoader(
            self.train_dataset,
            sampler=self.train_sampler,
            batch_size=self.batch_size,
            shuffle=self.train_shuffle,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn
        )

    def val_dataloader(self) -> DataLoader:
        if isinstance(self.train_dataset, IterableDataset):
            worker_init_fn = default(self.worker_init_fn, default_worker_init_fn)
        else:
            worker_init_fn = self.worker_init_fn
        return DataLoader(
            self.val_dataset,
            sampler=self.val_sampler,
            batch_size=self.val_batch_size,
            shuffle=self.val_shuffle,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn
        )

    def test_dataloader(self) -> DataLoader:
        if isinstance(self.train_dataset, IterableDataset):
            worker_init_fn = default(self.worker_init_fn, default_worker_init_fn)
        else:
            worker_init_fn = self.worker_init_fn
        return DataLoader(
            self.test_dataset,
            sampler=self.test_sampler,
            batch_size=self.val_batch_size,
            shuffle=self.val_shuffle,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn
        )


class VideoDataset_cc(Dataset):
    """
    Reads video bytes from a PyArrow table. Replaces VideoDataset for LAM.
    """

    def __init__(
            self,
            table: pa.Table,
            group_name: str = "",
            padding: str = "repeat",
            randomize: bool = False,
            resolution: int = 256,
            num_frames: int = 16,
            output_format: str = "t h w c",
            color_aug: bool = True
    ) -> None:
        super().__init__()
        self.table = table
        self.group_name = group_name
        self.padding = padding
        self.randomize = randomize
        self.resolution = resolution
        self.num_frames = num_frames
        self.output_format = output_format
        self.color_aug = color_aug

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, idx: int) -> Dict:
        while True:
            try:
                row = self.table.slice(idx, 1)
                video_bytes = row.column("video_bytes")[0].as_py()
                video_format = row.column("video_format")[0].as_py()
                ds_name = row.column("dataset")[0].as_py()

                video = self.load_video_from_bytes(
                    video_bytes, video_format, ds_name,
                    self.num_frames,
                    None if self.randomize else 0
                )
                data = self.build_data_dict(video)
                data["video_path"] = f"parquet://{self.group_name}/{idx}"
                return data
            except Exception:
                idx = randint(0, len(self) - 1)

    def load_video_from_bytes(
            self,
            video_bytes: bytes,
            video_format: str,
            dataset_name: str,
            num_frames: int,
            start_frame: int = None,
            frame_skip: int = 1
    ) -> Tensor:
        cap, tmp_path = decode_video_from_bytes(video_bytes, video_format)
        try:
            total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
            if "retro" in dataset_name:
                frame_skip = 4
            elif "procgen" not in dataset_name and "ssv2" not in dataset_name and "mira" not in dataset_name:
                frame_skip = 2
            num_frames = num_frames * frame_skip

            start_frame = start_frame if exists(start_frame) else randint(0, max(0, total_frames - num_frames))
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

        if video.shape[1] != video.shape[2]:
            square_len = min(video.shape[1], video.shape[2])
            h_crop = (video.shape[1] - square_len) // 2
            w_crop = (video.shape[2] - square_len) // 2
            video = video[:, h_crop:h_crop + square_len, w_crop:w_crop + square_len]

        if video.shape[-2] != self.resolution or video.shape[-3] != self.resolution:
            video = rearrange(video, "t h w c -> c t h w")
            video = F.interpolate(video, self.resolution, mode="bicubic")
            video = rearrange(video, f"c t h w -> {self.output_format}")
        else:
            video = rearrange(video, f"t h w c -> {self.output_format}")
        return video

    def build_data_dict(self, video: Tensor) -> Dict:
        if self.color_aug:
            video = (video + torch.rand(1) * 0.2 - 0.1).clamp(0, 1)
        return {"videos": video}


class MultiSourceSamplerDataset_cc(Dataset):
    """
    Replaces MultiSourceSamplerDataset for LAM. Reads from parquet shards.
    """

    def __init__(
            self,
            data_root: str,
            env_source: str = "game",
            split: str = "train",
            samples_per_epoch: int = 1000000,
            sampling_strategy: str = "sample",
            color_aug: bool = True,
            **kwargs
    ) -> None:
        self.samples_per_epoch = samples_per_epoch

        shard_groups = []

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
                shards = sorted([
                    os.path.join(split_dir, f)
                    for f in os.listdir(split_dir) if f.endswith(".parquet")
                ])
                if shards:
                    tables = [pq.read_table(s) for s in shards]
                    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]

                    env_col = table.column("environment").to_pylist()
                    unique_envs = sorted(set(env_col))
                    for env in unique_envs:
                        mask = [e == env for e in env_col]
                        env_table = table.filter(mask)
                        shard_groups.append((f"{src}/{env}", env_table))

        elif env_source in ("ssv2_our_mira",):
            for src in ["ssv2_our", "mira"]:
                split_dir = os.path.join(data_root, src, split)
                if not os.path.isdir(split_dir):
                    print(f"[DATA] Skipping {src}/{split}: directory not found")
                    continue
                shards = sorted([
                    os.path.join(split_dir, f)
                    for f in os.listdir(split_dir) if f.endswith(".parquet")
                ])
                print(f"[DATA] {src}/{split}: found {len(shards)} parquet shards")
                if shards:
                    tables = []
                    for i, s in enumerate(shards):
                        print(f"[DATA]   Loading shard {i+1}/{len(shards)}: {os.path.basename(s)} ({os.path.getsize(s) / 1e9:.1f} GB)")
                        tables.append(pq.read_table(s))
                    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
                    print(f"[DATA] {src}/{split}: {len(table)} videos loaded, {table.nbytes / 1e9:.1f} GB in memory")
                    shard_groups.append((src, table))
        else:
            split_dir = os.path.join(data_root, env_source, split)
            if os.path.isdir(split_dir):
                shards = sorted([
                    os.path.join(split_dir, f)
                    for f in os.listdir(split_dir) if f.endswith(".parquet")
                ])
                if shards:
                    tables = [pq.read_table(s) for s in shards]
                    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
                    shard_groups.append((env_source, table))

        if not shard_groups:
            raise ValueError(f"No parquet data found for env_source={env_source}, split={split} in {data_root}")

        self.subsets = []
        for group_name, table in tqdm(shard_groups, desc="Loading parquet subsets..."):
            print("Subset:", group_name, f"({len(table)} videos)")
            self.subsets.append(VideoDataset_cc(
                table=table, group_name=group_name, color_aug=color_aug, **kwargs
            ))
        print("Number of subsets:", len(self.subsets))

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


class LightningVideoDataset_cc(LightningDataset_cc):
    """
    Drop-in replacement for LightningVideoDataset that reads from parquet.
    Use in LAM configs with: class_path: lam.dataset_cc.LightningVideoDataset_cc
    """

    def __init__(
            self,
            data_root: str,
            env_source: str = "game",
            padding: str = "repeat",
            randomize: bool = False,
            resolution: int = 256,
            num_frames: int = 16,
            output_format: str = "t h w c",
            samples_per_epoch: int = 1000000,
            sampling_strategy: str = "sample",
            **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.data_root = data_root
        self.env_source = env_source
        self.padding = padding
        self.randomize = randomize
        self.resolution = resolution
        self.num_frames = num_frames
        self.output_format = output_format
        self.samples_per_epoch = samples_per_epoch
        self.sampling_strategy = sampling_strategy

        self.save_hyperparameters()

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = MultiSourceSamplerDataset_cc(
                data_root=self.data_root,
                env_source=self.env_source,
                split="train",
                padding=self.padding,
                randomize=self.randomize,
                resolution=self.resolution,
                num_frames=self.num_frames,
                output_format=self.output_format,
                samples_per_epoch=self.samples_per_epoch,
                sampling_strategy=self.sampling_strategy
            )
            try:
                self.val_dataset = MultiSourceSamplerDataset_cc(
                    data_root=self.data_root,
                    env_source=self.env_source,
                    split="test",
                    padding=self.padding,
                    randomize=self.randomize,
                    resolution=self.resolution,
                    num_frames=self.num_frames,
                    output_format=self.output_format,
                    samples_per_epoch=self.samples_per_epoch // 1000,
                    sampling_strategy=self.sampling_strategy,
                    color_aug=False
                )
            except ValueError:
                print(f"No test split found for env_source={self.env_source}, using train for val")
                self.val_dataset = MultiSourceSamplerDataset_cc(
                    data_root=self.data_root,
                    env_source=self.env_source,
                    split="train",
                    padding=self.padding,
                    randomize=self.randomize,
                    resolution=self.resolution,
                    num_frames=self.num_frames,
                    output_format=self.output_format,
                    samples_per_epoch=self.samples_per_epoch // 1000,
                    sampling_strategy=self.sampling_strategy,
                    color_aug=False
                )
        elif stage == "test":
            self.test_dataset = MultiSourceSamplerDataset_cc(
                data_root=self.data_root,
                env_source=self.env_source,
                split="test",
                padding=self.padding,
                randomize=self.randomize,
                resolution=self.resolution,
                num_frames=self.num_frames,
                output_format=self.output_format,
                samples_per_epoch=self.samples_per_epoch // 1000,
                sampling_strategy=self.sampling_strategy,
                color_aug=False
            )
        else:
            raise ValueError(f"Invalid stage: {stage}")
