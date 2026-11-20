#!/bin/bash
#SBATCH --job-name=ggi-eval
#SBATCH --time=1:30:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --output=logs/%x-%j.out
# Three evaluations of one model on the evaluation seeds, 160 rollouts each, no trainer:
#   shared     four rollouts of the model per machine
#   isolated   each rollout on a machine of its own, with the same seeds and deadlocks
#   crossplay  one rollout of the model per machine among three scripted agents that
#              touch only their own jobs; the reward they lose is measured exactly
# shared minus isolated is what interference costs the model among copies of itself;
# crossplay is what one rollout takes from fixed victims.
#   sbatch jobs/eval.sh <tag> <model or checkpoint path> [environment flags, e.g. --start_name_arg False]
# Outputs: $GGI_OUT/runs/deadlock_recovery_{shared,isolated,crossplay}_eval_<tag>
source "$SLURM_SUBMIT_DIR/jobs/common.sh"
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
T=$1; M=$2; shift 2
cd "$REPO/rl"
run() { "$PY" -u train.py --eval_only True --model "$M" --data_seed 5000000 "$@" || echo "FAILED: $*"; }
run --name "deadlock_recovery_shared_eval_$T" --groups_per_step 40 --max_steps 1 "$@"
run --name "deadlock_recovery_isolated_eval_$T" --groups_per_step 40 --max_steps 1 --shared_machine False "$@"
run --name "deadlock_recovery_crossplay_eval_$T" --groups_per_step 80 --max_steps 2 --eval_peers careful "$@"
