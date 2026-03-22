"""Extract latent actions from trained LAM and visualize with UMAP."""
import json
import os
import sys
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, ".")
from lam.model import LAM
from lam.dataset import VideoDataset

CKPT = "/network/scratch/a/ankur.sikarwar/WORLD_MODEL_PROJECT/lam/exp_ckpts_ssv2_our_v1/last.ckpt"
NUM_SAMPLES = 10000
BATCH_SIZE = 64
SAVE_DIR = "/network/scratch/a/ankur.sikarwar/WORLD_MODEL_PROJECT/lam"
LABELS_DIR = "/home/mila/a/ankur.sikarwar/scratch/WORLD_MODEL_PROJECT/data/ssv2/raw/labels"

SPLITS = {
    "train": "../data/ssv2_our/train",
    "test": "../data/ssv2_our/test",
}

# Load SSv2 action labels: video_id -> template
print("Loading SSv2 action labels...")
id2template = {}
for fname in ["train.json", "validation.json"]:
    with open(os.path.join(LABELS_DIR, fname)) as f:
        for entry in json.load(f):
            id2template[entry["id"]] = entry["template"]
print(f"  Loaded labels for {len(id2template)} videos")

# Load label-to-index mapping
with open(os.path.join(LABELS_DIR, "labels.json")) as f:
    template2idx = json.load(f)

# Load model
print("Loading checkpoint...")
model = LAM.load_from_checkpoint(CKPT, map_location="cpu")
model.eval()
model.cuda()


def extract_latents(split_path, num_samples):
    dataset = VideoDataset(
        split_path=split_path,
        padding="repeat",
        randomize=True,
        resolution=256,
        num_frames=2,
        output_format="t h w c",
        color_aug=False,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=False,
    )
    all_z_mu = []
    all_ids = []
    count = 0
    with torch.no_grad():
        for batch in tqdm(loader):
            videos = batch["videos"].cuda()
            paths = batch["video_path"]  # list of actual video paths used
            outputs = model.lam.encode(videos)
            z_mu = outputs["z_mu"].cpu().numpy()
            all_z_mu.append(z_mu)
            # Extract video IDs from the actual paths returned by the dataset
            all_ids.extend([
                os.path.basename(p).replace(".webm", "").replace(".mp4", "")
                for p in paths
            ])
            count += z_mu.shape[0]
            if count >= num_samples:
                break
    z_mu = np.concatenate(all_z_mu, axis=0)[:num_samples]
    ids = all_ids[:num_samples]
    return z_mu, ids


# Extract latent actions for both splits
split_data = {}
split_ids = {}
for name, path in SPLITS.items():
    print(f"Extracting latent actions from {name} ({NUM_SAMPLES} samples)...")
    z_mu, ids = extract_latents(path, NUM_SAMPLES)
    split_data[name] = z_mu
    split_ids[name] = ids
    print(f"  Collected {z_mu.shape[0]} vectors of dim {z_mu.shape[1]}")

# Build label arrays
def get_templates(ids):
    return [id2template.get(vid, None) for vid in ids]

all_z = np.concatenate([split_data["train"], split_data["test"]], axis=0)
all_split_labels = np.array(["train"] * len(split_data["train"]) + ["test"] * len(split_data["test"]))
all_templates = get_templates(split_ids["train"]) + get_templates(split_ids["test"])

# Fit UMAP with tighter params for more structure
print("Running UMAP on combined train+test (n_neighbors=30, min_dist=0.05)...")
import umap

reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.05, random_state=42)
embedding = reducer.fit_transform(all_z)

emb_train = embedding[all_split_labels == "train"]
emb_test = embedding[all_split_labels == "test"]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Figure 1: train vs test colored by split ---
print("Plotting train vs test...")
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.scatter(emb_train[:, 0], emb_train[:, 1], c="tab:blue", s=3, alpha=0.3, label="train")
ax.scatter(emb_test[:, 0], emb_test[:, 1], c="tab:orange", s=3, alpha=0.3, label="test")
ax.legend(markerscale=5)
ax.set_title("UMAP of LAM Latent Actions — Train vs Test (ssv2_our, 10k samples)")
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
fig.savefig(f"{SAVE_DIR}/umap_train_vs_test.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {SAVE_DIR}/umap_train_vs_test.png")

# --- Figure 2: separate subplots colored by L2 norm ---
print("Plotting L2 norm...")
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
for ax, name, emb, z_mu in [
    (axes[0], "train", emb_train, split_data["train"]),
    (axes[1], "test", emb_test, split_data["test"]),
]:
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=np.linalg.norm(z_mu, axis=1),
                    cmap="viridis", s=3, alpha=0.5)
    plt.colorbar(sc, ax=ax, label="L2 norm of latent action")
    ax.set_title(f"{name} (n={len(z_mu)})")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
fig.suptitle("UMAP of LAM Latent Actions — colored by L2 norm (ssv2_our)", fontsize=14)
fig.savefig(f"{SAVE_DIR}/umap_l2norm_train_test.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {SAVE_DIR}/umap_l2norm_train_test.png")

# --- Figure 3: colored by action category ---
print("Plotting by action category...")
# Find top-N most frequent templates for cleaner visualization
from collections import Counter
valid_templates = [t for t in all_templates if t is not None]
template_counts = Counter(valid_templates)
top_k = 20
top_templates = [t for t, _ in template_counts.most_common(top_k)]

# Assign colors
template_to_color = {t: i for i, t in enumerate(top_templates)}
colors = []
for t in all_templates:
    if t in template_to_color:
        colors.append(template_to_color[t])
    else:
        colors.append(-1)  # "other"
colors = np.array(colors)

# Shorten template names for legend
def shorten(t, max_len=40):
    t = t.replace("[something]", "[sth]").replace("[somewhere]", "[sw]")
    return t[:max_len] + "..." if len(t) > max_len else t

fig, ax = plt.subplots(1, 1, figsize=(14, 10))

# Plot "other" category first (grey, behind)
mask_other = colors == -1
if mask_other.any():
    ax.scatter(embedding[mask_other, 0], embedding[mask_other, 1],
               c="lightgrey", s=2, alpha=0.15, label=f"other ({mask_other.sum()})")

# Plot top-k categories
cmap = plt.cm.get_cmap("tab20", top_k)
for i, template in enumerate(top_templates):
    mask = colors == i
    ax.scatter(embedding[mask, 0], embedding[mask, 1],
               c=[cmap(i)], s=8, alpha=0.6,
               label=f"{shorten(template)} ({mask.sum()})")

ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1), markerscale=3,
          title="Action category (top 20)", title_fontsize=8)
ax.set_title("UMAP of LAM Latent Actions — by Action Category (ssv2_our)")
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
fig.savefig(f"{SAVE_DIR}/umap_action_categories.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {SAVE_DIR}/umap_action_categories.png")

# Print top-20 category counts
print(f"\nTop {top_k} action categories:")
for t, c in template_counts.most_common(top_k):
    print(f"  {c:5d}  {t}")

# Save raw data
np.savez(f"{SAVE_DIR}/latent_actions.npz",
         z_mu_train=split_data["train"], z_mu_test=split_data["test"],
         umap_train=emb_train, umap_test=emb_test,
         ids_train=split_ids["train"], ids_test=split_ids["test"],
         templates=np.array(all_templates, dtype=object))
print(f"Saved raw data to {SAVE_DIR}/latent_actions.npz")
