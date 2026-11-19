#!/bin/bash
#SBATCH --job-name=ggi-sim
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
# The tabular simulations of paper.md (CPU only). Figures: python sim/analyze.py <out>.
source "$SLURM_SUBMIT_DIR/jobs/common.sh"
O=$GGI_OUT_ROOT/sim; mkdir -p "$O"
for exp in ${EXPS:-measure train spite defense}; do
  "$PY" "$REPO/sim/sim.py" "$exp" --out "$O/$exp.jsonl" --procs 16
done
