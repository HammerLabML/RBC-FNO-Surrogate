#!/bin/bash
#SBATCH --job-name=upload
#SBATCH --partition=cpu
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/homes/%u/RBC-FNO-Surrogate/logs/%x_%j.out

cd ~/RBC-FNO-Surrogate
srun uv run python data/upload_data.py
