#!/bin/bash
# Clone the surveyed code at the commits tested, copy the modules under test into pkgs/
# (unmodified except import paths), and pull the SandboxFusion server image (~9 GB,
# ~30 min; run this part on a compute node, which has internet: sbatch --time=1:00:00
# --cpus-per-task=4 --mem=16G survey/interference/fetch.sh).
set -euo pipefail
EXT=${EXT:-$SCRATCH/ext}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$EXT" && cd "$EXT"
get() { [ -d "$1" ] || git clone -q "$2" "$1"; git -C "$1" checkout -q "$3"; }
get tinker-cookbook https://github.com/thinking-machines-lab/tinker-cookbook.git 28de41e
get verl https://github.com/volcengine/verl.git a4e0dd2
get SkyRL https://github.com/NovaSky-AI/SkyRL.git 387bce8
rm -rf "$HERE/pkgs" && mkdir -p "$HERE/pkgs/skyrl_tools"
cp -r verl/verl/utils/reward_score/prime_code "$HERE/pkgs/prime_code"
cp SkyRL/skyrl-gym/skyrl_gym/tools/{core.py,sql.py,python.py} "$HERE/pkgs/skyrl_tools/"
cp SkyRL/skyrl-gym/skyrl_gym/envs/sql/utils.py "$HERE/pkgs/skyrl_tools/sql_reward.py"
sed -i 's/from skyrl_gym.tools.core import/from skyrl_tools.core import/' "$HERE"/pkgs/skyrl_tools/{sql,python}.py
touch "$HERE/pkgs/skyrl_tools/__init__.py"
if [ ! -f "$EXT/sandbox-fusion.sif" ]; then
  module load apptainer
  APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-$SCRATCH/cache/apptainer} \
    apptainer pull "$EXT/sandbox-fusion.sif" docker://volcengine/sandbox-fusion:server-20250609
fi
