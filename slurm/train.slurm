#!/bin/bash
#SBATCH --job-name=dit-fmnist
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out

# Example SLURM script: train on a single GPU.
# Pass your cluster's GPU partition with -p, plus any train.py flags:
#   sbatch -p <gpu-partition> slurm/train.slurm
#   sbatch -p <gpu-partition> slurm/train.slurm --wandb
#   sbatch -p <gpu-partition> slurm/train.slurm --max-steps 300   # smoke test

cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
uv run python train.py "$@"
