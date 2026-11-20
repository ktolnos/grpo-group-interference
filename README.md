# GRPO group interference

Does a group-relative baseline (GRPO, leave-one-out) reward a rollout for lowering the
rewards of the other rollouts in its group? The theory and tabular simulations are in
`paper.md`; `deadlock_recovery/` is a test environment where several language-model agents share a
machine and can kill each other's jobs, and `rl/` trains on it with GRPO. Results so far:
[RESULTS.md](RESULTS.md).

| path | contents |
|---|---|
| `RESULTS.md` | what the environment and training runs have shown, and what is open |
| `LOG.md` | one short entry per launched experiment: command, commit, what it showed |
| `deadlock_recovery/` | the environment (`harness.py`, `simulate.py`, `group_server.py`), hosted-model runner (`run.py`), training-awareness probe |
| `rl/` | GRPO training (`train.py`: configuration and the trainer; `deadlock_recovery.py`: the environment as the trainer sees it; `trainer_base.py`: fp32 master-weight optimizer, checkpointing, callbacks), batched rollouts (`rollout.py`), model-free test, `summarize.py` |
| `jobs/` | Slurm scripts: `test.sh`, `train.sh`, `api_eval.sh`, `sim.sh` |
| `sim/` | tabular simulations and figures of the draft |
| `AFFECTED_ENVIRONMENTS.md`, `survey/interference/` | widely used RL environments where rollouts in a group can interfere: tested and still to test; scripts to reproduce |
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
# GRPO, shared machine, training configuration (~5 min/step on an H100, ~15 on an L40S)
sbatch jobs/train.sh --name deadlock_recovery_shared_grpo_s0 --kill_pid_hint True --seed 0
# the untrained model, no updates, 160 rollouts on the evaluation seeds
sbatch --time=3:00:00 --mem=48G jobs/train.sh --name eval_base_shared --eval_only True \
    --max_steps 5 --data_seed 5000000
# a hosted model, four conditions (needs $VECTOR_INFERENCE_API_KEY)
sbatch jobs/api_eval.sh gpt-oss-120b gptoss
python rl/summarize.py --steps $SCRATCH/ggi/rl/runs/deadlock_recovery_shared_grpo_s0
```

Environment variables (all optional): `GGI_VENV` (default `$SCRATCH/venvs/ggi`),
`GGI_OUT` (outputs, default `$SCRATCH/ggi`), `HF_HOME` (default `$SCRATCH/cache/hf`).
Logs go to `logs/`. A training run resumes from its last checkpoint when resubmitted
under the same `--name`.

Notes:
- The environment runs each episode in a private process namespace (`unshare --user
  --pid`), which compute nodes allow; it does not need root.
- Run names start with `deadlock_recovery_shared_` or `deadlock_recovery_isolated_`
  (`--shared_machine False`), then the baseline or arm and the seed.
- `train.py --help` lists every option; the experimental arms are `--baseline
  {group,loo,other_groups}`, `--shared_machine`, `--kill_pid_hint`,
  `--kill_by_name_tool` and `--start_name_arg`.
- Commit before launching and add the run to `LOG.md`: a run is reproduced by its
  command at its commit, and `run.json` records both.
- Experiments of tens of GPU-hours should have their design agreed first (arms, seeds,
  GPU-hours per arm).
