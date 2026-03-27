#!/bin/bash
# Submit with: sbatch slurm_train_ssv2_our_mira_cc_seg_bs12_lr5e6.sh
# Training: LAM segment-level (T=8, bs12, lr=5e-6) on SSv2_our + MiraData | CC cluster | 4x H100

#SBATCH --job-name=lam_bs12_lr5e6
#SBATCH --output=/scratch/a/ankur99/WORLD_MODEL_PROJECT/logs/lam_ssv2_our_mira_cc_seg_bs12_lr5e6_%j.out
#SBATCH --error=/scratch/a/ankur99/WORLD_MODEL_PROJECT/logs/lam_ssv2_our_mira_cc_seg_bs12_lr5e6_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ankursikarwar01041999@gmail.com

SCRATCH=/scratch/a/ankur99/WORLD_MODEL_PROJECT
CODE_DIR=/home/a/ankur99/Work/WORLD_MODEL_PROJECT/AdaWorld/lam

mkdir -p $SCRATCH/logs
mkdir -p $SCRATCH/lam/lam_logs

module load StdEnv/2023 intel/2023.2.1
module load cuda/11.8 arrow/17 python/3.10 httpproxy

# Activate venv
source $SCRATCH/venv_adaworld/bin/activate

cd $CODE_DIR

# NCCL settings
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export MASTER_PORT=29610

# Route all WandB I/O to scratch
export WANDB_CACHE_DIR=$SCRATCH/lam/wandb_cache
export WANDB_DATA_DIR=$SCRATCH/lam/wandb_data
export WANDB_DIR=$SCRATCH/lam/lam_logs
export WANDB_INIT_TIMEOUT=300

# Unset SLURM task env vars so Lightning uses TorchElasticEnvironment (torchrun)
unset SLURM_NTASKS SLURM_PROCID SLURM_LOCALID SLURM_NODEID SLURM_STEP_NODELIST

CKPT_DIR="$SCRATCH/lam/lam_logs/exp_ckpts_ssv2_our_mira_seg_bs12_lr5e6"
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
    main_cc.py fit \
    --config config/lam_ssv2_our_mira_cc_seg_bs12_lr5e6.yaml \
    $RESUME_ARG \
    2>&1 | tee -a $SCRATCH/lam/lam_logs/output_train_ssv2_our_mira_cc_seg_bs12_lr5e6.log
