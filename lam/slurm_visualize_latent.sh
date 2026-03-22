#!/bin/bash
#SBATCH --job-name=lam_umap
#SBATCH --output=/network/scratch/a/ankur.sikarwar/WORLD_MODEL_PROJECT/logs/lam_umap_%j.out
#SBATCH --error=/network/scratch/a/ankur.sikarwar/WORLD_MODEL_PROJECT/logs/lam_umap_%j.err
#SBATCH --partition=short-unkillable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100l:4
#SBATCH --mem=40G
#SBATCH --time=01:00:00

mkdir -p /network/scratch/a/ankur.sikarwar/WORLD_MODEL_PROJECT/logs

module load anaconda/3
conda activate adaworld

cd /home/mila/a/ankur.sikarwar/Work/WORLD_MODEL_PROJECT/AdaWorld/lam

python visualize_latent_actions.py
