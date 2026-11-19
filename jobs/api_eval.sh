#!/bin/bash
#SBATCH --job-name=ggi-api
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/%x-%j.out
# One hosted model (OpenAI-compatible API; default endpoint Vector Inference, key in
# $VECTOR_INFERENCE_API_KEY) on Environment 1, four conditions: shared machine (15
# episodes x 4 agents) and isolated (40 episodes x 1 agent), each without and with the
# kill tool's "Prefer the pid" sentence. Seeds start at 5000000 in every condition.
#   sbatch jobs/api_eval.sh gpt-oss-120b gptoss
# A model that is asleep on the proxy fails every call for 5-10 minutes: wake it first.
# run.py appends to its output, so a rerun under the same tag needs the old files moved.
source "$SLURM_SUBMIT_DIR/jobs/common.sh"
M=$1; T=$2; O=$GGI_OUT_ROOT/env1/api; mkdir -p "$O"
cd "$REPO/env1"
for hint in "" "--kill-pid-hint"; do
  h=$([ -n "$hint" ] && echo pidhint || echo plain)
  "$PY" -u run.py --policy "llm:$M" --G 4 --episodes 15 --seed 5000000 --parallel 5 $hint --out "$O/${T}_shared_$h.jsonl" &
  "$PY" -u run.py --policy "llm:$M" --G 1 --episodes 40 --seed 5000000 --parallel 10 $hint --out "$O/${T}_isolated_$h.jsonl" &
done
wait
