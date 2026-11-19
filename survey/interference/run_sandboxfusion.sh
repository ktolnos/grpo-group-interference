#!/bin/bash
#SBATCH --job-name=sf-interference
#SBATCH --time=0:40:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out
# SandboxFusion server image as tinker-cookbook's code_rl README runs it (its local.yaml
# mounted over the image's), then as the image ships. Docker is not available on the
# cluster, so apptainer runs the same image; code then runs as the job's user instead of
# root, which does not change what one request can do to another.
# From the repository root, after fetch.sh: sbatch survey/interference/run_sandboxfusion.sh
module load apptainer
EXT=${EXT:-$SCRATCH/ext}
HERE=$SLURM_SUBMIT_DIR/survey/interference
CFG=$EXT/tinker-cookbook/tinker_cookbook/recipes/code_rl/sandbox_config/local.yaml
LCB=$EXT/tinker-cookbook/tinker_cookbook/recipes/code_rl/lcb_utils.py
export TMPDIR=/tmp

serve() {   # port, extra apptainer args
  local port=$1; shift
  apptainer exec --writable-tmpfs --no-home --pwd /root/sandbox "$@" \
    --env _BYTEFAAS_RUNTIME_PORT=$port $EXT/sandbox-fusion.sif bash -c 'cd /root/sandbox && bash scripts/run.sh' \
    > /tmp/sf_server_$port.log 2>&1 &
  for i in $(seq 150); do curl -s -o /dev/null http://127.0.0.1:$port/ && break; sleep 2; done
}

P=$((20000 + SLURM_JOB_ID % 20000))
echo "##### tinker code_rl config"
serve $P --bind $CFG:/root/sandbox/sandbox/configs/local.yaml
python3 $HERE/sf_client.py $LCB http://127.0.0.1:$P/run_code
kill %1; sleep 3
echo "##### image default config"
serve $((P + 1))
python3 $HERE/sf_client.py $LCB http://127.0.0.1:$((P + 1))/run_code
