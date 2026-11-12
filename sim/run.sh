#!/bin/bash
#SBATCH --account=aip-gigor
#SBATCH --job-name=ggh-spite
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=/scratch/eop/outputs/ggh/spite/run-%j.out
source ~/.bashrc
PY=/scratch/eop/venv-urh/bin/python
SIM=/project/6101830/eop/grpo-group-interference/sim/sim.py
OUT=/scratch/eop/outputs/ggh/spite
for exp in ${EXPS:-measure train spite defense}; do
  $PY $SIM "$exp" --out "$OUT/$exp.jsonl" --procs 16
done
