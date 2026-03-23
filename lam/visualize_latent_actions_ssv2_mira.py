"""Extract latent actions from trained LAM (ssv2_our+mira) and visualize with UMAP.
Includes both SSv2 and MiraData samples to show how both data sources distribute."""
import json
import os
import sys
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, ".")
from lam.model import LAM
from lam.dataset import VideoDataset

CKPT = "/network/scratch/a/ankur.sikarwar/WORLD_MODEL_PROJECT/lam/exp_ckpts_ssv2_our_mira_v1/last.ckpt"
NUM_SAMPLES_SSV2 = 10000
NUM_SAMPLES_MIRA = 5000
BATCH_SIZE = 64
SAVE_DIR = "/network/scratch/a/ankur.sikarwar/WORLD_MODEL_PROJECT/lam/umap_ssv2_our_mira"
LABELS_DIR = "/home/mila/a/ankur.sikarwar/scratch/WORLD_MODEL_PROJECT/data/ssv2/raw/labels"

SPLITS = {
    "ssv2_train": "../data/ssv2_our/train",
    "ssv2_test": "../data/ssv2_our/test",
    "mira_train": "../data/mira/train",
}

os.makedirs(SAVE_DIR, exist_ok=True)

# Load SSv2 action labels: video_id -> template
print("Loading SSv2 action labels...")
id2template = {}
for fname in ["train.json", "validation.json"]:
    with open(os.path.join(LABELS_DIR, fname)) as f:
        for entry in json.load(f):
            id2template[entry["id"]] = entry["template"]
print(f"  Loaded labels for {len(id2template)} videos")

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
            paths = batch["video_path"]
            outputs = model.lam.encode(videos)
            z_mu = outputs["z_mu"].cpu().numpy()
            all_z_mu.append(z_mu)
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


# Extract latent actions for all splits
split_data = {}
split_ids = {}
for name, path in SPLITS.items():
    num = NUM_SAMPLES_MIRA if "mira" in name else NUM_SAMPLES_SSV2
    print(f"Extracting latent actions from {name} ({num} samples)...")
    z_mu, ids = extract_latents(path, num)
    split_data[name] = z_mu
    split_ids[name] = ids
    print(f"  Collected {z_mu.shape[0]} vectors of dim {z_mu.shape[1]}")

# Build combined arrays
all_z = np.concatenate([split_data[k] for k in SPLITS.keys()], axis=0)
# Source label: "ssv2" or "mira"
source_labels = []
for name in SPLITS.keys():
    source = "mira" if "mira" in name else "ssv2"
    source_labels.extend([source] * len(split_data[name]))
source_labels = np.array(source_labels)

# SSv2 action templates (mira gets None)
def get_templates(ids, source):
    if source == "mira":
        return [None] * len(ids)
    return [id2template.get(vid, None) for vid in ids]

all_templates = []
for name in SPLITS.keys():
    source = "mira" if "mira" in name else "ssv2"
    all_templates.extend(get_templates(split_ids[name], source))

# Fit UMAP on all data together
print(f"Running UMAP on combined data ({len(all_z)} samples, n_neighbors=30, min_dist=0.05)...")
import umap

reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.05, random_state=42)
embedding = reducer.fit_transform(all_z)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Figure 1: SSv2 vs Mira colored by data source ---
print("Plotting SSv2 vs Mira...")
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
mask_ssv2 = source_labels == "ssv2"
mask_mira = source_labels == "mira"
ax.scatter(embedding[mask_ssv2, 0], embedding[mask_ssv2, 1],
           c="tab:blue", s=3, alpha=0.3, label=f"SSv2 ({mask_ssv2.sum()})")
ax.scatter(embedding[mask_mira, 0], embedding[mask_mira, 1],
           c="tab:red", s=3, alpha=0.3, label=f"MiraData ({mask_mira.sum()})")
ax.legend(markerscale=5)
ax.set_title("UMAP of LAM Latent Actions — SSv2 vs MiraData (ssv2_our+mira model)")
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
fig.savefig(f"{SAVE_DIR}/umap_ssv2_vs_mira.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {SAVE_DIR}/umap_ssv2_vs_mira.png")

# --- Figure 2: SSv2 train vs test vs Mira ---
print("Plotting train/test/mira splits...")
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
offset = 0
colors_splits = {"ssv2_train": "tab:blue", "ssv2_test": "tab:orange", "mira_train": "tab:red"}
for name in SPLITS.keys():
    n = len(split_data[name])
    emb = embedding[offset:offset + n]
    ax.scatter(emb[:, 0], emb[:, 1], c=colors_splits[name], s=3, alpha=0.3,
               label=f"{name} ({n})")
    offset += n
ax.legend(markerscale=5)
ax.set_title("UMAP of LAM Latent Actions — by Split (ssv2_our+mira model)")
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
fig.savefig(f"{SAVE_DIR}/umap_splits.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {SAVE_DIR}/umap_splits.png")

# --- Figure 3: colored by L2 norm, separate panels for SSv2 and Mira ---
print("Plotting L2 norm...")
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
for ax, mask, title in [
    (axes[0], mask_ssv2, f"SSv2 (n={mask_ssv2.sum()})"),
    (axes[1], mask_mira, f"MiraData (n={mask_mira.sum()})"),
]:
    z = all_z[mask]
    emb = embedding[mask]
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=np.linalg.norm(z, axis=1),
                    cmap="viridis", s=3, alpha=0.5)
    plt.colorbar(sc, ax=ax, label="L2 norm of latent action")
    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
fig.suptitle("UMAP of LAM Latent Actions — colored by L2 norm (ssv2_our+mira model)", fontsize=14)
fig.savefig(f"{SAVE_DIR}/umap_l2norm.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {SAVE_DIR}/umap_l2norm.png")

# --- Figure 4: SSv2 action categories (mira shown as grey background) ---
print("Plotting by action category...")
from collections import Counter
ssv2_templates = [t for t in all_templates if t is not None]
template_counts = Counter(ssv2_templates)
top_k = 20
top_templates = [t for t, _ in template_counts.most_common(top_k)]

template_to_color = {t: i for i, t in enumerate(top_templates)}
colors = []
for t, src in zip(all_templates, source_labels):
    if src == "mira":
        colors.append(-2)  # mira
    elif t in template_to_color:
        colors.append(template_to_color[t])
    else:
        colors.append(-1)  # ssv2 other
colors = np.array(colors)

def shorten(t, max_len=40):
    t = t.replace("[something]", "[sth]").replace("[somewhere]", "[sw]")
    return t[:max_len] + "..." if len(t) > max_len else t

fig, ax = plt.subplots(1, 1, figsize=(14, 10))

# Plot mira first (light red, behind everything)
mask_mira_c = colors == -2
if mask_mira_c.any():
    ax.scatter(embedding[mask_mira_c, 0], embedding[mask_mira_c, 1],
               c="lightsalmon", s=2, alpha=0.15, label=f"MiraData ({mask_mira_c.sum()})")

# Plot SSv2 "other" category (grey)
mask_other = colors == -1
if mask_other.any():
    ax.scatter(embedding[mask_other, 0], embedding[mask_other, 1],
               c="lightgrey", s=2, alpha=0.15, label=f"SSv2 other ({mask_other.sum()})")

# Plot top-k SSv2 categories
cmap = plt.cm.get_cmap("tab20", top_k)
for i, template in enumerate(top_templates):
    mask = colors == i
    ax.scatter(embedding[mask, 0], embedding[mask, 1],
               c=[cmap(i)], s=8, alpha=0.6,
               label=f"{shorten(template)} ({mask.sum()})")

ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1), markerscale=3,
          title="Action category (top 20 SSv2 + Mira)", title_fontsize=8)
ax.set_title("UMAP of LAM Latent Actions — by Action Category (ssv2_our+mira model)")
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
fig.savefig(f"{SAVE_DIR}/umap_action_categories.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {SAVE_DIR}/umap_action_categories.png")

# Print top-20 category counts
print(f"\nTop {top_k} SSv2 action categories:")
for t, c in template_counts.most_common(top_k):
    print(f"  {c:5d}  {t}")

# Save raw data
np.savez(f"{SAVE_DIR}/latent_actions.npz",
         z_mu_ssv2_train=split_data["ssv2_train"], z_mu_ssv2_test=split_data["ssv2_test"],
         z_mu_mira=split_data["mira_train"],
         embedding=embedding, source_labels=source_labels,
         templates=np.array(all_templates, dtype=object))
print(f"Saved raw data to {SAVE_DIR}/latent_actions.npz")
