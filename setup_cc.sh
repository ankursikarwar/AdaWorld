#!/bin/bash
# =============================================================================
# Cross-cluster setup script
#
# Run this on a new cluster to download data from HuggingFace and prepare
# for training. After running this, training is as simple as:
#
#   cd worldmodel
#   torchrun --nproc_per_node=8 train.py \
#       --base configs/training/adaworld_cc.yaml \
#       --num_nodes 1 --n_devices 8
#
# Usage:
#   bash setup_cc.sh <hf_repo_id> [data_dir]
#
# Example:
#   bash setup_cc.sh sikarwarank/adaworld_data
#   bash setup_cc.sh sikarwarank/adaworld_data /scratch/$USER/custom_path
# =============================================================================

set -e

HF_REPO_ID="${1:?Usage: bash setup_cc.sh <hf_repo_id> [data_dir]}"

# Default to scratch directory for fast I/O on compute clusters
# Tries common scratch env vars, falls back to ~/scratch
if [ -n "${SCRATCH:-}" ]; then
    DEFAULT_DATA_DIR="$SCRATCH/WORLD_MODEL_PROJECT/parquet_data"
elif [ -n "${SLURM_TMPDIR:-}" ]; then
    # SLURM_TMPDIR is per-job local SSD — fast but ephemeral, not ideal for persistent data
    DEFAULT_DATA_DIR="$SCRATCH/WORLD_MODEL_PROJECT/parquet_data"
elif [ -d "$HOME/scratch" ]; then
    DEFAULT_DATA_DIR="$HOME/scratch/WORLD_MODEL_PROJECT/parquet_data"
elif [ -d "/scratch/$USER" ]; then
    DEFAULT_DATA_DIR="/scratch/$USER/WORLD_MODEL_PROJECT/parquet_data"
else
    DEFAULT_DATA_DIR="$HOME/scratch/WORLD_MODEL_PROJECT/parquet_data"
fi

DATA_DIR="${2:-$DEFAULT_DATA_DIR}"

echo "============================================"
echo " AdaWorld Cross-Cluster Setup"
echo "============================================"
echo " HF Repo:  $HF_REPO_ID"
echo " Data Dir: $DATA_DIR"
echo "============================================"

# 1. Install dependencies if needed
echo ""
echo "[1/3] Checking dependencies..."
pip install pyarrow huggingface_hub --quiet 2>/dev/null || true

# 2. Download data from HuggingFace
echo ""
echo "[2/3] Downloading data from HuggingFace..."
echo "  This may take a while for large datasets."
echo ""

# Use huggingface-cli to download the entire dataset repo
huggingface-cli download "$HF_REPO_ID" \
    --repo-type dataset \
    --local-dir "$DATA_DIR" \
    --local-dir-use-symlinks False

# The HF download places files under data/ subdirectory.
# Restructure so parquet_data/{dataset}/{split}/shard_*.parquet
if [ -d "$DATA_DIR/data" ]; then
    echo ""
    echo "  Restructuring downloaded data..."
    # Move contents of data/ up one level
    for ds_dir in "$DATA_DIR/data"/*/; do
        ds_name=$(basename "$ds_dir")
        mv "$ds_dir" "$DATA_DIR/$ds_name" 2>/dev/null || true
    done
    rmdir "$DATA_DIR/data" 2>/dev/null || true
fi

# 3. Verify
echo ""
echo "[3/3] Verifying data..."
echo ""
for ds in procgen retro mira ssv2_our; do
    if [ -d "$DATA_DIR/$ds" ]; then
        count=$(find "$DATA_DIR/$ds" -name "*.parquet" | wc -l)
        echo "  $ds: $count parquet shards"
    fi
done

echo ""
echo "============================================"
echo " Setup complete!"
echo ""
echo " To train the world model:"
echo "   cd worldmodel"
echo "   torchrun --nproc_per_node=8 train.py \\"
echo "       --base configs/training/adaworld_cc.yaml \\"
echo "       --num_nodes 1 --n_devices 8"
echo ""
echo " To train LAM:"
echo "   cd lam"
echo "   python main.py fit --config config/lam_game_cc.yaml"
echo ""
echo " Make sure data_root in the _cc configs points to:"
echo "   $DATA_DIR"
echo "============================================"
