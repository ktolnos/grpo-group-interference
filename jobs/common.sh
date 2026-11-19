# Sourced by every job script. Submit jobs from the repository root (so that
# $SLURM_SUBMIT_DIR is the repository), with the account set once per shell:
#   export SBATCH_ACCOUNT=<your account>      # e.g. aip-gigor
#   sbatch jobs/<script>.sh ...
# GGI_VENV  the virtual environment made by `uv sync` (default: $SCRATCH/venvs/ggi)
# GGI_OUT   where runs go (default: $SCRATCH/ggi; training under rl/, API runs under env1/)
set -euo pipefail
REPO=${GGI_REPO:-$SLURM_SUBMIT_DIR}
SCRATCH=${SCRATCH:-/scratch/$USER}
export GGI_OUT_ROOT=${GGI_OUT:-$SCRATCH/ggi}
export GGI_OUT=$GGI_OUT_ROOT/rl
VENV=${GGI_VENV:-$SCRATCH/venvs/ggi}
PY=$VENV/bin/python
# Caches off /home; vLLM's Triton kernels compile with nvcc.
export HF_HOME=${HF_HOME:-$SCRATCH/cache/hf} UV_CACHE_DIR=${UV_CACHE_DIR:-$SCRATCH/cache/uv}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$SCRATCH/cache/triton}
module load gcc cuda/13.2 >/dev/null 2>&1 || true
[ -x "$PY" ] || { echo "no python at $PY: run 'UV_PROJECT_ENVIRONMENT=$VENV uv sync' in $REPO"; exit 1; }
mkdir -p "$GGI_OUT" "$GGI_OUT_ROOT/env1"
