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

# 1. Load required modules (adjust versions for your cluster)
echo ""
echo "[1/4] Loading modules and setting up Python environment..."
module load StdEnv/2023 intel/2023.2.1 cuda/11.8 arrow/17 python/3.10 opencv 2>/dev/null || {
    echo "  WARNING: 'module load' not available or some modules missing."
    echo "  Make sure CUDA 11.8, Python 3.10, pyarrow, and opencv are available."
}
VENV_DIR="${DATA_DIR%/*}/venv_adaworld"
if [ ! -d "$VENV_DIR" ]; then
    echo "  Creating virtual environment at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
echo "  Installing requirements..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pip install -r "$SCRIPT_DIR/requirements_cc.txt" --quiet 2>&1 | tail -3

# 2. Download data from HuggingFace
echo ""
echo "[2/4] Downloading data from HuggingFace..."
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
echo "[3/4] Verifying data..."
echo ""
for ds in procgen retro mira ssv2_our; do
    if [ -d "$DATA_DIR/$ds" ]; then
        count=$(find "$DATA_DIR/$ds" -name "*.parquet" | wc -l)
        echo "  $ds: $count parquet shards"
    fi
done

# 4. Activation reminder
echo ""
echo "[4/4] Environment ready."
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
echo ""
echo " Activate the env before training:"
echo "   source $VENV_DIR/bin/activate"
echo "============================================"
