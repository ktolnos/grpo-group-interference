#!/bin/bash
#SBATCH --job-name=ggi-test
#SBATCH --time=0:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/%x-%j.out
# Model-free check of the environment and the rollout path (CPU only, ~3 min).
source "$SLURM_SUBMIT_DIR/jobs/common.sh"
cd "$REPO/rl" && "$PY" -u test_rollout.py
