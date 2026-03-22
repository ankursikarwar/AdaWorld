#!/bin/bash
# Submit with: sbatch slurm_retro.sh
# Runs 932 jobs in parallel (one per game), max 200 concurrent at a time.
# Each game: 100 train + 10 test episodes x 1000 steps @ ~0.6s/episode.
# Estimated time per game: ~10 mins (with video encoding). 2hr = 12x safety margin.

#SBATCH --job-name=retro_%a
#SBATCH --output=/network/scratch/a/ankur.sikarwar/WORLD_MODEL_PROJECT/logs/retro_%A_%a.out
#SBATCH --error=/network/scratch/a/ankur.sikarwar/WORLD_MODEL_PROJECT/logs/retro_%A_%a.err
#SBATCH --array=0-931%50           # 932 games, max 50 running simultaneously
#SBATCH --partition=long-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2          # retro is single-threaded; 2 for overhead
#SBATCH --mem=4G                   # retro is lightweight; 4G is generous
#SBATCH --time=02:00:00            # ~10 mins per game; 2hr = 12x safety margin

mkdir -p /network/scratch/a/ankur.sikarwar/WORLD_MODEL_PROJECT/logs

# Need virtual display for stable-retro (headless server)
Xvfb :${SLURM_ARRAY_TASK_ID} -screen 0 1024x768x24 &
XVFB_PID=$!
export DISPLAY=:${SLURM_ARRAY_TASK_ID}
sleep 1

module load anaconda/3
conda activate retro

cd /home/mila/a/ankur.sikarwar/Work/WORLD_MODEL_PROJECT/AdaWorld

python sample_retro_single.py \
    --task_id $SLURM_ARRAY_TASK_ID \
    --num_logs 100 \
    --timeout 1000 \
    --root data

# Clean up virtual display
kill $XVFB_PID
