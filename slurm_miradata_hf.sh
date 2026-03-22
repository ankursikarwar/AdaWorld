#!/bin/bash
# Submit with: sbatch slurm_miradata_hf.sh
# Downloads pre-processed MiraData clips from Little-Podi/AdaWorld on HuggingFace (~28.7 GB, 9 zips)

#SBATCH --job-name=miradata_hf
#SBATCH --output=/network/scratch/a/ankur.sikarwar/WORLD_MODEL_PROJECT/logs/miradata_hf_%j.out
#SBATCH --error=/network/scratch/a/ankur.sikarwar/WORLD_MODEL_PROJECT/logs/miradata_hf_%j.err
#SBATCH --partition=long-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00

mkdir -p /network/scratch/a/ankur.sikarwar/WORLD_MODEL_PROJECT/logs

module load anaconda/3
conda activate adaworld

TRAIN_DIR=/network/scratch/a/ankur.sikarwar/WORLD_MODEL_PROJECT/data/mira/train
TMP_DIR=/network/scratch/a/ankur.sikarwar/WORLD_MODEL_PROJECT/data/mira/hf_zips

mkdir -p $TRAIN_DIR $TMP_DIR

cd /home/mila/a/ankur.sikarwar/Work/WORLD_MODEL_PROJECT/AdaWorld

python3 -c "
from huggingface_hub import hf_hub_download
import os, zipfile

train_dir = '$TRAIN_DIR'
tmp_dir = '$TMP_DIR'

zips = ['00','02','03','04','05','06','07','08','09']

for z in zips:
    fname = f'miradata/{z}.zip'
    local_path = os.path.join(tmp_dir, f'{z}.zip')
    if os.path.exists(local_path):
        print(f'Already downloaded: {z}.zip')
    else:
        print(f'Downloading {z}.zip ...')
        downloaded = hf_hub_download(
            repo_id='Little-Podi/AdaWorld',
            repo_type='dataset',
            filename=fname,
            local_dir=tmp_dir,
            local_dir_use_symlinks=False
        )
        os.rename(downloaded, local_path) if downloaded != local_path else None
        print(f'Done: {z}.zip')

    print(f'Extracting {z}.zip into train dir...')
    with zipfile.ZipFile(local_path, 'r') as zf:
        for member in zf.namelist():
            # Extract flat into train_dir (strip any directory prefix)
            filename = os.path.basename(member)
            if not filename or not filename.endswith('.mp4'):
                continue
            target = os.path.join(train_dir, filename)
            if os.path.exists(target):
                continue
            with zf.open(member) as src, open(target, 'wb') as dst:
                dst.write(src.read())
    print(f'Extracted {z}.zip')

print('Done. Total clips in train:')
print(len([f for f in os.listdir(train_dir) if f.endswith('.mp4')]))
"
