#!/bin/bash

#SBATCH --job-name=2d_fno
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/homes/%u/RBC-FNO-Surrogate/logs/%x_%A_%a.out

#SBATCH --exclude=worker-15

cd ~/RBC-FNO-Surrogate

# Load non-python libraries that might not be available on compute cluster
source ~/miniforge3/etc/profile.d/conda.sh
conda activate uv

srun uv run python scripts/train_2d_rbc.py seed=$SLURM_ARRAY_TASK_ID \
    model=2d_fno \
    options=gpucluster \
    data.ra=10000000
