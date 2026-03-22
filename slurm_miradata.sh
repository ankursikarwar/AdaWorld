#!/bin/bash
# Submit with: sbatch slurm_miradata.sh
# Downloads 8K MiraData video clips (3D rendered scenes) from YouTube via yt-dlp,
# clips them with ffmpeg, then flattens into data/mira/train/.

#SBATCH --job-name=miradata
#SBATCH --output=/network/scratch/a/ankur.sikarwar/WORLD_MODEL_PROJECT/logs/miradata_%j.out
#SBATCH --error=/network/scratch/a/ankur.sikarwar/WORLD_MODEL_PROJECT/logs/miradata_%j.err
#SBATCH --partition=long-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4          # yt-dlp + ffmpeg can use multiple cores
#SBATCH --mem=16G
#SBATCH --time=24:00:00            # 8K YouTube downloads + ffmpeg clips

mkdir -p /network/scratch/a/ankur.sikarwar/WORLD_MODEL_PROJECT/logs

module load anaconda/3
conda activate adaworld

RAW_DIR=/network/scratch/a/ankur.sikarwar/WORLD_MODEL_PROJECT/data/mira/raw_videos
STAGING_DIR=/network/scratch/a/ankur.sikarwar/WORLD_MODEL_PROJECT/data/mira/staging
TRAIN_DIR=/network/scratch/a/ankur.sikarwar/WORLD_MODEL_PROJECT/data/mira/train

mkdir -p $RAW_DIR $STAGING_DIR $TRAIN_DIR

cd /home/mila/a/ankur.sikarwar/Work/WORLD_MODEL_PROJECT/AdaWorld

# Step 1: Download and clip videos into staging dir
python download_miradata_360p.py \
    --meta_csv data/data_list/miradata_8k.csv \
    --raw_video_save_dir $RAW_DIR \
    --clip_video_save_dir $STAGING_DIR

# Step 2: Flatten all clips into data/mira/train/
echo "Flattening clips into train dir..."
find $STAGING_DIR -name "*.mp4" | while read f; do
    fname=$(basename "$f")
    ln -sf "$f" "$TRAIN_DIR/$fname"
done

echo "Done. Total clips:"
ls $TRAIN_DIR | wc -l
