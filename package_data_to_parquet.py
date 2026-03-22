"""
Package video data into Parquet files for portable HuggingFace upload.

Stores raw video bytes in parquet shards (~2GB each) so the entire dataset
can be pulled from HuggingFace Hub and used directly by the _cc dataloaders.

Usage:
    python package_data_to_parquet.py \
        --data_root /home/mila/a/ankur.sikarwar/scratch/WORLD_MODEL_PROJECT/data \
        --output_dir /home/mila/a/ankur.sikarwar/scratch/WORLD_MODEL_PROJECT/parquet_data \
        --datasets procgen retro mira ssv2_our \
        --shard_size_gb 2.0
"""

import argparse
import os
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm


def get_video_files(directory):
    """Collect all .mp4 and .webm files in a directory."""
    files = []
    for f in sorted(os.listdir(directory)):
        if f.endswith(".mp4") or f.endswith(".webm"):
            files.append(os.path.join(directory, f))
    return files


def build_file_manifest(data_root, datasets):
    """
    Build a list of (video_path, dataset, environment, split, filename, format) tuples.
    """
    manifest = []

    for ds in datasets:
        ds_path = os.path.join(data_root, ds)
        if not os.path.exists(ds_path):
            print(f"WARNING: {ds_path} does not exist, skipping.")
            continue

        if ds in ("procgen", "retro"):
            # Structure: dataset/environment/split/*.mp4
            for env in sorted(os.listdir(ds_path)):
                env_path = os.path.join(ds_path, env)
                if not os.path.isdir(env_path):
                    continue
                for split in ("train", "test"):
                    split_path = os.path.join(env_path, split)
                    if not os.path.isdir(split_path):
                        continue
                    for vf in get_video_files(split_path):
                        fmt = "mp4" if vf.endswith(".mp4") else "webm"
                        manifest.append((vf, ds, env, split, os.path.basename(vf), fmt))

        elif ds in ("mira", "ssv2_our"):
            # Structure: dataset/split/*.mp4 or *.webm
            for split in ("train", "test"):
                split_path = os.path.join(ds_path, split)
                if not os.path.isdir(split_path):
                    continue
                for vf in get_video_files(split_path):
                    fmt = "mp4" if vf.endswith(".mp4") else "webm"
                    manifest.append((vf, ds, "", split, os.path.basename(vf), fmt))

    return manifest


def write_shards(manifest, output_dir, dataset_name, split, shard_size_bytes):
    """Write parquet shards for a given dataset+split combination."""
    # Filter manifest for this dataset+split
    entries = [(p, ds, env, sp, fn, fmt) for p, ds, env, sp, fn, fmt in manifest
               if ds == dataset_name and sp == split]

    if not entries:
        return 0

    shard_dir = os.path.join(output_dir, dataset_name, split)
    os.makedirs(shard_dir, exist_ok=True)

    schema = pa.schema([
        ("video_bytes", pa.binary()),
        ("video_format", pa.string()),
        ("dataset", pa.string()),
        ("environment", pa.string()),
        ("split", pa.string()),
        ("filename", pa.string()),
    ])

    shard_idx = 0
    current_size = 0
    batch_data = {col: [] for col in schema.names}
    total_written = 0

    desc = f"{dataset_name}/{split}"
    for video_path, ds, env, sp, fn, fmt in tqdm(entries, desc=desc):
        try:
            with open(video_path, "rb") as f:
                video_bytes = f.read()
        except Exception as e:
            print(f"WARNING: Could not read {video_path}: {e}")
            continue

        batch_data["video_bytes"].append(video_bytes)
        batch_data["video_format"].append(fmt)
        batch_data["dataset"].append(ds)
        batch_data["environment"].append(env)
        batch_data["split"].append(sp)
        batch_data["filename"].append(fn)
        current_size += len(video_bytes)

        if current_size >= shard_size_bytes:
            table = pa.table(batch_data, schema=schema)
            shard_path = os.path.join(shard_dir, f"shard_{shard_idx:05d}.parquet")
            pq.write_table(table, shard_path)
            total_written += len(batch_data["filename"])
            print(f"  Wrote {shard_path} ({len(batch_data['filename'])} videos, {current_size / 1e9:.2f} GB)")

            shard_idx += 1
            current_size = 0
            batch_data = {col: [] for col in schema.names}

    # Write remaining
    if batch_data["filename"]:
        table = pa.table(batch_data, schema=schema)
        shard_path = os.path.join(shard_dir, f"shard_{shard_idx:05d}.parquet")
        pq.write_table(table, shard_path)
        total_written += len(batch_data["filename"])
        print(f"  Wrote {shard_path} ({len(batch_data['filename'])} videos, {current_size / 1e9:.2f} GB)")

    return total_written


def main():
    parser = argparse.ArgumentParser(description="Package video data into Parquet shards")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory containing procgen/, retro/, mira/, ssv2_our/")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for parquet files")
    parser.add_argument("--datasets", nargs="+",
                        default=["procgen", "retro", "mira", "ssv2_our"],
                        help="Which datasets to package")
    parser.add_argument("--shard_size_gb", type=float, default=2.0,
                        help="Target shard size in GB")
    args = parser.parse_args()

    shard_size_bytes = int(args.shard_size_gb * 1e9)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Building file manifest...")
    manifest = build_file_manifest(args.data_root, args.datasets)
    print(f"Total files found: {len(manifest)}")

    # Group by dataset and split
    ds_splits = set()
    for _, ds, _, sp, _, _ in manifest:
        ds_splits.add((ds, sp))

    total = 0
    for ds, sp in sorted(ds_splits):
        n = write_shards(manifest, args.output_dir, ds, sp, shard_size_bytes)
        total += n
        print(f"  {ds}/{sp}: {n} videos written")

    print(f"\nDone! Total videos packaged: {total}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
