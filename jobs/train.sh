#!/bin/bash
#SBATCH --job-name=ggi-rl
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --output=logs/%x-%j.out
# One GRPO run (or, with --eval_only True, rollouts without updates). All arguments go to
# rl/train.py, e.g.
#   sbatch jobs/train.sh --name deadlock_recovery_shared_grpo_s0 --seed 0
#   sbatch --time=1:00:00 --mem=32G --cpus-per-task=12 jobs/train.sh \
#       --name deadlock_recovery_shared_eval_base --eval_only True --groups_per_step 40 \
#       --max_steps 1 --data_seed 5000000
# A run resumes from its last checkpoint when resubmitted under the same --name.
# 128G: fp32 master weights and AdamW moments for Qwen3-4B live in host memory (96G hit
# its limit). For an L40S instead: sbatch --gres=gpu:l40s:1 jobs/train.sh ...
# Node kn018 trained 3x slower than other L40S nodes (2026-11-16); exclude it with
# --exclude=kn018 if it is still slow.
source "$SLURM_SUBMIT_DIR/jobs/common.sh"
# TRL picks the colocated vLLM's rendezvous port by testing 29500 first and binding it
# later, so runs that start together on one node collide (three did, 2026-11-19).
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$REPO/rl" && exec "$PY" -u train.py "$@"
