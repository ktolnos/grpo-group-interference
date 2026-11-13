#!/bin/bash
# Environment 1 on a CPU node. Submit from /scratch/eop:
#   cd /scratch/eop && sbatch /project/6101830/eop/grpo-group-interference/env1/run.sh <args to run.py>
#SBATCH --account=aip-gigor
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --job-name=ggi-env1
#SBATCH --output=/scratch/eop/outputs/ggi/env1/logs/%j.out
cd /project/6101830/eop/grpo-group-interference/env1
python3 -u run.py "$@"
