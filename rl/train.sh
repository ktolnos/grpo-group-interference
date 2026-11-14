#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --job-name=ggi-rl
# One training run on Environment 1. Submit from /scratch/eop:
#   sbatch --time=3:00:00 --output=<log> rl/train.sh --name <run> --baseline group ...
# 96G: the fp32 master weights and AdamW's moments for Qwen3-4B live in host memory.
source ~/.bashrc
REPO=/project/6101830/eop/grpo-group-interference
VENV=/project/6101830/eop/unlearning-reward-hacking/.venv
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd $REPO/rl
exec $VENV/bin/python -u train.py "$@"
