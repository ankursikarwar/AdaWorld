"""
Upload parquet data to HuggingFace Hub.

Usage:
    # Login first:
    #   huggingface-cli login

    python upload_to_hf.py \
        --parquet_dir /home/mila/a/ankur.sikarwar/scratch/WORLD_MODEL_PROJECT/parquet_data \
        --repo_id YOUR_USERNAME/adaworld-data \
        --datasets procgen retro mira ssv2_our
"""

import argparse
import os
from huggingface_hub import HfApi, create_repo


def main():
    parser = argparse.ArgumentParser(description="Upload parquet data to HuggingFace Hub")
    parser.add_argument("--parquet_dir", type=str, required=True,
                        help="Directory containing parquet shards")
    parser.add_argument("--repo_id", type=str, required=True,
                        help="HuggingFace repo ID (e.g., username/adaworld-data)")
    parser.add_argument("--datasets", nargs="+",
                        default=["procgen", "retro", "mira", "ssv2_our"],
                        help="Which datasets to upload")
    parser.add_argument("--repo_type", type=str, default="dataset",
                        help="Repository type")
    parser.add_argument("--private", action="store_true",
                        help="Make the repository private")
    args = parser.parse_args()

    api = HfApi()

    # Create repository if it doesn't exist
    try:
        create_repo(args.repo_id, repo_type=args.repo_type, private=args.private, exist_ok=True)
        print(f"Repository {args.repo_id} ready.")
    except Exception as e:
        print(f"Note: {e}")

    # Upload each dataset
    for ds in args.datasets:
        ds_dir = os.path.join(args.parquet_dir, ds)
        if not os.path.isdir(ds_dir):
            print(f"WARNING: {ds_dir} does not exist, skipping {ds}")
            continue

        for split in ("train", "test"):
            split_dir = os.path.join(ds_dir, split)
            if not os.path.isdir(split_dir):
                continue

            parquet_files = sorted([
                f for f in os.listdir(split_dir) if f.endswith(".parquet")
            ])

            if not parquet_files:
                continue

            print(f"\nUploading {ds}/{split} ({len(parquet_files)} shards)...")

            for pf in parquet_files:
                local_path = os.path.join(split_dir, pf)
                remote_path = f"data/{ds}/{split}/{pf}"
                file_size_gb = os.path.getsize(local_path) / 1e9

                print(f"  Uploading {remote_path} ({file_size_gb:.2f} GB)...")
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=remote_path,
                    repo_id=args.repo_id,
                    repo_type=args.repo_type,
                )

    print(f"\nDone! Data available at: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
