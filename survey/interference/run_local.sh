#!/bin/bash
#SBATCH --job-name=local-interference
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out
# verl prime_code, SkyRL-Gym SQL and Python tools, NeMo-RL code environment. Needs numpy and pandas (the
# project's environment has both). From the repository root, after fetch.sh:
# sbatch survey/interference/run_local.sh
HERE=$SLURM_SUBMIT_DIR/survey/interference
export PATH=${GGI_VENV:-$SCRATCH/venvs/ggi}/bin:$PATH TMPDIR=/tmp
echo "## verl prime_code"; python $HERE/verl_prime_code_test.py $HERE/pkgs
mkdir -p /tmp/dbs
echo "## SkyRL-Gym"; python $HERE/skyrl_test.py $HERE/pkgs /tmp/dbs
echo "## NeMo-RL code environment"; python $HERE/nemo_rl_code_env_test.py ${EXT:-$SCRATCH/ext}/NeMo-RL/nemo_rl/environments/code_environment.py
