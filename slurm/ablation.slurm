#!/bin/bash
#SBATCH --job-name=dit-ablation
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out

# Example SLURM script: run the FID ablation on a single GPU.
#   sbatch -p <gpu-partition> slurm/ablation.slurm
#   sbatch -p <gpu-partition> slurm/ablation.slurm --n 5000

cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1
uv run python ablation.py "$@"
