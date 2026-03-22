#!/bin/bash
# Submit with: sbatch slurm_annotate.sh
#
# Annotates MiraData videos at clip / segment / frame-pair levels.
# Shards work across 8 tasks. Each task handles ~435 of the 3478 videos.
#
# To switch model, change --backend / --model / --api_base below.

#SBATCH --job-name=annotate_%a
#SBATCH --output=/network/scratch/a/ankur.sikarwar/WORLD_MODEL_PROJECT/logs/annotate_%A_%a.out
#SBATCH --error=/network/scratch/a/ankur.sikarwar/WORLD_MODEL_PROJECT/logs/annotate_%A_%a.err
#SBATCH --array=0-7
#SBATCH --partition=long-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00

mkdir -p /network/scratch/a/ankur.sikarwar/WORLD_MODEL_PROJECT/logs

module load anaconda/3
conda activate adaworld

cd /home/mila/a/ankur.sikarwar/Work/WORLD_MODEL_PROJECT/AdaWorld

# ── Option A: Gemini API (default) ───────────────────────────────────────────
python annotate_videos.py \
    --backend gemini \
    --model gemini-3-pro-preview \
    --api_key $GEMINI_API_KEY \
    --gemini_native \
    --n_segments 4 \
    --clip_frames 8 \
    --segment_frames 16 \
    --frame_sample_fps 16.0 \
    --wandb_project mira-annotations \
    --wandb_run_name "gemini-3-pro-preview-task${SLURM_ARRAY_TASK_ID}" \
    --wandb_log_every 50 \
    --task_id $SLURM_ARRAY_TASK_ID \
    --n_tasks 8

# ── Option B: Local vLLM ──────────────────────────────────────────────────────
# (start vLLM server separately on a GPU node first)
# python annotate_videos.py \
#     --backend openai \
#     --model Qwen/Qwen2.5-VL-7B-Instruct \
#     --api_base http://localhost:8000/v1 \
#     --n_segments 4 \
#     --clip_frames 8 \
#     --segment_frames 6 \
#     --frame_sample_fps 2.0 \
#     --wandb_project adaworld-annotations \
#     --wandb_run_name "qwen-vl-task${SLURM_ARRAY_TASK_ID}" \
#     --wandb_log_every 50 \
#     --task_id $SLURM_ARRAY_TASK_ID \
#     --n_tasks 8
