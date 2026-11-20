#!/bin/bash
#SBATCH --job-name=fetch-interference
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out
# Clone the surveyed code at the commits tested, copy the modules under test into pkgs/
# (unmodified except import paths), and pull the SandboxFusion server image (~9 GB,
# ~30 min; best as a job from the repository root, compute nodes have internet:
# sbatch survey/interference/fetch.sh).
set -euo pipefail
EXT=${EXT:-$SCRATCH/ext}
HERE=${SLURM_SUBMIT_DIR:+$SLURM_SUBMIT_DIR/survey/interference}   # sbatch runs a copy of this script
HERE=${HERE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
mkdir -p "$EXT" && cd "$EXT"
get() { [ -d "$1" ] || git clone -q "$2" "$1"; git -C "$1" checkout -q "$3"; }
get tinker-cookbook https://github.com/thinking-machines-lab/tinker-cookbook.git 28de41e
get verl https://github.com/volcengine/verl.git a4e0dd2
get SkyRL https://github.com/NovaSky-AI/SkyRL.git 387bce8
get NeMo-RL https://github.com/NVIDIA-NeMo/RL.git 1fcd48d
rm -rf "$HERE/pkgs" && mkdir -p "$HERE/pkgs/skyrl_tools"
cp -r verl/verl/utils/reward_score/prime_code "$HERE/pkgs/prime_code"
cp SkyRL/skyrl-gym/skyrl_gym/tools/{core.py,sql.py,python.py} "$HERE/pkgs/skyrl_tools/"
cp SkyRL/skyrl-gym/skyrl_gym/envs/sql/utils.py "$HERE/pkgs/skyrl_tools/sql_reward.py"
sed -i 's/from skyrl_gym.tools.core import/from skyrl_tools.core import/' "$HERE"/pkgs/skyrl_tools/{sql,python}.py
touch "$HERE/pkgs/skyrl_tools/__init__.py"
if [ ! -f "$EXT/sandbox-fusion.sif" ]; then
  set +u; module load apptainer; set -u
  APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-$SCRATCH/cache/apptainer} \
    apptainer pull "$EXT/sandbox-fusion.sif" docker://volcengine/sandbox-fusion:server-20250609
fi
