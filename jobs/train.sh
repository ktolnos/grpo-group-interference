#!/bin/bash
#SBATCH --job-name=ggi-rl
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --output=logs/%x-%j.out
# One GRPO run (or, with --eval_only True, rollouts without updates). All arguments go to
# rl/train.py, e.g.
#   sbatch jobs/train.sh --name env1_grpo_shared_s0 --kill_pid_hint True --seed 0
#   sbatch --time=3:00:00 --mem=48G jobs/train.sh --name eval_base_shared --eval_only True \
#       --max_steps 5 --data_seed 5000000
# A run resumes from its last checkpoint when resubmitted under the same --name.
# 128G: fp32 master weights and AdamW moments for Qwen3-4B live in host memory (96G hit
# its limit). Node kn018 trained 3x slower than other L40S nodes (2026-11-16); exclude it
# with --exclude=kn018 if it is still slow.
source "$SLURM_SUBMIT_DIR/jobs/common.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$REPO/rl" && exec "$PY" -u train.py "$@"
