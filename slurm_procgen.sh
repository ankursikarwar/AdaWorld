#!/bin/bash
# Submit with: sbatch slurm_procgen.sh
#

#SBATCH --job-name=procgen_%a
#SBATCH --output=/network/scratch/a/ankur.sikarwar/WORLD_MODEL_PROJECT/logs/procgen_%A_%a.out
#SBATCH --error=/network/scratch/a/ankur.sikarwar/WORLD_MODEL_PROJECT/logs/procgen_%A_%a.err
#SBATCH --array=0-15
#SBATCH --partition=long-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2              # procgen is single-threaded; 2 for overhead
#SBATCH --mem=4G                       # peak usage ~120MB; 4G is generous
#SBATCH --time=24:00:00               # each game ~9hrs on login node; 24hrs = 2.5x safety margin

mkdir -p /network/scratch/a/ankur.sikarwar/WORLD_MODEL_PROJECT/logs

module load anaconda/3
conda activate adaworld

cd /home/mila/a/ankur.sikarwar/Work/WORLD_MODEL_PROJECT/AdaWorld

python sample_procgen_single.py \
    --task_id $SLURM_ARRAY_TASK_ID \
    --num_logs 10000 \
    --timeout 1000 \
    --root data
