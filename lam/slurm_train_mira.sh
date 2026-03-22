#!/bin/bash
# Submit with: sbatch slurm_train_mira.sh
# Training: LAM on MiraData only | Partition: long | Time: 24h

#SBATCH --job-name=lam_mira
#SBATCH --output=/network/scratch/a/ankur.sikarwar/WORLD_MODEL_PROJECT/logs/lam_mira_%j.out
#SBATCH --error=/network/scratch/a/ankur.sikarwar/WORLD_MODEL_PROJECT/logs/lam_mira_%j.err
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100l:4
#SBATCH --mem=80G
#SBATCH --time=24:00:00

SCRATCH=/network/scratch/a/ankur.sikarwar/WORLD_MODEL_PROJECT/lam
mkdir -p /network/scratch/a/ankur.sikarwar/WORLD_MODEL_PROJECT/logs
mkdir -p $SCRATCH/exp_ckpts_mira_v1
mkdir -p $SCRATCH/exp_logs
mkdir -p $SCRATCH/exp_imgs_mira
mkdir -p $SCRATCH/wandb_cache
mkdir -p $SCRATCH/wandb_data

module load anaconda/3
conda activate adaworld

cd /home/mila/a/ankur.sikarwar/Work/WORLD_MODEL_PROJECT/AdaWorld/lam

# Pre-build file list cache for mira/train
/home/mila/a/ankur.sikarwar/.conda/envs/adaworld/bin/python -c "
from os import listdir, path
data_root = '../data'
for split in ['train']:
    split_path = path.join(data_root, 'mira', split)
    cache_path = split_path + '.filelist'
    if path.exists(cache_path):
        print(f'Cache already exists: {cache_path}')
    else:
        print(f'Scanning {split_path} ...')
        files = [path.join(split_path, f) for f in listdir(split_path)
                 if f.endswith('.mp4') or f.endswith('.webm')]
        with open(cache_path, 'w') as f:
            f.write('\n'.join(files))
        print(f'Cached {len(files)} files -> {cache_path}')
"

# NCCL settings
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export MASTER_PORT=29601

# Route all WandB I/O to scratch
export WANDB_CACHE_DIR=$SCRATCH/wandb_cache
export WANDB_DATA_DIR=$SCRATCH/wandb_data
export WANDB_DIR=$SCRATCH/exp_logs

# Unset SLURM task env vars so Lightning uses TorchElasticEnvironment (torchrun)
unset SLURM_NTASKS SLURM_PROCID SLURM_LOCALID SLURM_NODEID SLURM_STEP_NODELIST

CKPT_DIR="$SCRATCH/exp_ckpts_mira_v1"
LAST_CKPT="$CKPT_DIR/last.ckpt"

if [ -f "$LAST_CKPT" ]; then
    echo "Resuming from checkpoint: $LAST_CKPT"
    RESUME_ARG="--ckpt_path $LAST_CKPT"
else
    echo "No checkpoint found, starting from scratch."
    RESUME_ARG=""
fi

torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --master_port=$MASTER_PORT \
    main.py fit \
    --config config/lam_mira.yaml \
    $RESUME_ARG \
    2>&1 | tee -a $SCRATCH/output_train_mira.log
