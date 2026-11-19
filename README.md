# GRPO group interference

Does a group-relative baseline (GRPO, leave-one-out) reward a rollout for lowering the
rewards of the other rollouts in its group? The theory and tabular simulations are in
`paper.md`; `env1/` is a test environment where several language-model agents share a
machine and can kill each other's jobs, and `rl/` trains on it with GRPO. Results so far:
[RESULTS.md](RESULTS.md).

| path | contents |
|---|---|
| `RESULTS.md` | what the environment and training runs have shown, and what is open |
| `env1/` | the environment (`harness.py`, `simulate.py`, `group_server.py`), hosted-model runner (`run.py`), training-awareness probe |
| `rl/` | GRPO training (`train.py`, recipe in `grpo_recipe.py`), batched rollouts (`rollout.py`), model-free test, `summarize.py` |
| `jobs/` | Slurm scripts: `test.sh`, `train.sh`, `api_eval.sh`, `sim.sh` |
| `sim/` | tabular simulations and figures of the draft |
| `paper.md`, `drafts/`, `KNOWN_RELATED_WORK.md`, `notes/` | draft, earlier versions, related work, search notes |
| `PENDING_CHANGES.md` | environment changes held until the next forced rerun |

## Setup (Killarney or any Slurm cluster with L40S/H100 GPUs)

```bash
git clone git@github.com:ktolnos/grpo-group-interference.git
cd grpo-group-interference            # keep it on /project; sbatch is refused from /home
UV_PROJECT_ENVIRONMENT=$SCRATCH/venvs/ggi uv sync --locked   # ~10 GB, keep it off /project
export SBATCH_ACCOUNT=<your account>   # e.g. aip-gigor; every job needs one
sbatch jobs/test.sh                    # CPU, ~3 min: environment and rollout path, no model
```

Then, always from the repository root:

```bash
# GRPO, shared machine, training configuration (~20 min/step on an L40S)
sbatch jobs/train.sh --name env1_grpo_shared_s0 --kill_pid_hint True --seed 0
# the untrained model, no updates, 160 rollouts on the evaluation seeds
sbatch --time=3:00:00 --mem=48G jobs/train.sh --name eval_base_shared --eval_only True \
    --max_steps 5 --data_seed 5000000
# a hosted model, four conditions (needs $VECTOR_INFERENCE_API_KEY)
sbatch jobs/api_eval.sh gpt-oss-120b gptoss
python rl/summarize.py --steps $SCRATCH/ggi/rl/runs/env1_grpo_shared_s0
```

Environment variables (all optional): `GGI_VENV` (default `$SCRATCH/venvs/ggi`),
`GGI_OUT` (outputs, default `$SCRATCH/ggi`), `HF_HOME` (default `$SCRATCH/cache/hf`).
Logs go to `logs/`. A training run resumes from its last checkpoint when resubmitted
under the same `--name`.

Notes:
- The environment runs each episode in a private process namespace (`unshare --user
  --pid`), which compute nodes allow; it does not need root.
- `train.py --help` lists every option; the experimental arms are `--baseline
  {group,loo,other_groups}`, `--shared_machine`, `--kill_pid_hint` and
  `--kill_by_name_tool`.
- Experiments of tens of GPU-hours should have their design agreed first (arms, seeds,
  GPU-hours per arm).
