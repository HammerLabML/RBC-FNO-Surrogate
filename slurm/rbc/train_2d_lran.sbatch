#!/bin/bash

#SBATCH --job-name=2d_lran
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-4%5
#SBATCH --output=/homes/%u/RBC-FNO-Surrogate/logs/slurm/%x_%A_%a.out
#SBATCH --mail-type=FAIL

source ~/miniforge3/etc/profile.d/conda.sh
conda activate rbcpinn

cd ~/RBC-FNO-Surrogate
srun uv run python scripts/train_2d_rbc.py seed=$SLURM_ARRAY_TASK_ID model=2d_lran options=gpucluster