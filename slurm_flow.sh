#!/bin/bash
# Submit with: sbatch slurm_flow.sh
# Computes RAFT-small optical flow descriptors for MiraData videos.
# 8 GPU tasks, each handling ~435 of 3478 videos.

#SBATCH --job-name=flow_%a
#SBATCH --output=/network/scratch/a/ankur.sikarwar/WORLD_MODEL_PROJECT/logs/flow_%A_%a.out
#SBATCH --error=/network/scratch/a/ankur.sikarwar/WORLD_MODEL_PROJECT/logs/flow_%A_%a.err
#SBATCH --array=0-7
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

mkdir -p /network/scratch/a/ankur.sikarwar/WORLD_MODEL_PROJECT/logs

module load anaconda/3
module load cuda/11.7
conda activate adaworld

cd /home/mila/a/ankur.sikarwar/Work/WORLD_MODEL_PROJECT/AdaWorld

python compute_optical_flow.py \
    --device cuda \
    --model_variant sea_raft_m \
    --frame_sample_fps 4.0 \
    --max_frame_pairs 15 \
    --wandb_project mira-flow \
    --wandb_run_name "sea-raft-m-task${SLURM_ARRAY_TASK_ID}" \
    --task_id $SLURM_ARRAY_TASK_ID \
    --n_tasks 8
