"""Quick sanity check that video_path returned by dataset matches the actual video loaded."""
import json
import os
import sys

sys.path.insert(0, ".")
from lam.dataset import VideoDataset

LABELS_DIR = "/home/mila/a/ankur.sikarwar/scratch/WORLD_MODEL_PROJECT/data/ssv2/raw/labels"

# Load labels
id2template = {}
for fname in ["train.json", "validation.json"]:
    with open(os.path.join(LABELS_DIR, fname)) as f:
        for entry in json.load(f):
            id2template[entry["id"]] = entry["template"]

dataset = VideoDataset(
    split_path="../data/ssv2_our/test",
    padding="repeat",
    randomize=True,
    resolution=256,
    num_frames=2,
    output_format="t h w c",
    color_aug=False,
)

print(f"Dataset has {len(dataset)} videos\n")

# Check first 10 samples
print("idx | video_path returned by __getitem__ | video_id | label")
print("-" * 90)
for i in range(10):
    sample = dataset[i]
    vpath = sample["video_path"]
    vid = os.path.basename(vpath).replace(".webm", "").replace(".mp4", "")
    expected_path = dataset.file_names[i]
    label = id2template.get(vid, "UNKNOWN")
    match = "OK" if vpath == expected_path else f"RETRY (expected {os.path.basename(expected_path)})"
    print(f"{i:3d} | {os.path.basename(vpath):30s} | {vid:8s} | {label[:50]:50s} | {match}")

# Batch check via DataLoader
import torch
loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
batch = next(iter(loader))
print(f"\nDataLoader batch check (batch_size=4, shuffle=False):")
for i, p in enumerate(batch["video_path"]):
    vid = os.path.basename(p).replace(".webm", "").replace(".mp4", "")
    label = id2template.get(vid, "UNKNOWN")
    print(f"  batch[{i}]: {os.path.basename(p)} -> {vid} -> {label[:60]}")

# Coverage stats
total = len(dataset)
matched = sum(1 for f in dataset.file_names
              if os.path.basename(f).replace(".webm", "").replace(".mp4", "") in id2template)
print(f"\nLabel coverage: {matched}/{total} ({100*matched/total:.1f}%)")
