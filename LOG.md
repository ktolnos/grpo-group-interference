# Experiment log

One short entry per launched experiment: what was run and what it showed. Newest first.
Fuller tables are in [RESULTS.md](RESULTS.md).

Before launching, commit; a run is reproduced by its command at its commit (both are
also written to the run's `run.json`). The code is not kept able to rerun old entries:
check out the entry's commit for that.

Commands are arguments to `sbatch jobs/train.sh`. Outputs are in
`$SCRATCH/ggi/rl/runs/<name>/`.

## 2026-11-19

Everything on this date was launched from uncommitted changes on top of `7ece755`, so
none of it is reproducible from a commit. The environment was still called `env1`.

**Stability pair: lr 2e-6 without and with KL** (1 H100 each, seed 0, shared machine,
group baseline)
- `--name env1_grpo_shared_pidhint_lr2e6_nokl_s0 --kill_pid_hint True --seed 0 --beta 0`
- `--name env1_grpo_shared_pidhint_lr2e6_kl_s0 --kill_pid_hint True --seed 0 --beta 0.04`
- Without KL (60 steps, 5 h): reward 0.12 → 0.85, finished 4% → 97%, no collapse. The
  lower learning rate alone was enough; at 5e-6 the same setup collapsed after step 16.
- With KL (cancelled at step 47): reward 0.13 → 0.16. A coefficient of 0.04 holds the
  policy too close to the untrained model to learn the task. Smaller values untested.
- Interference went down, not up, without KL: rollouts that killed a groupmate's running
  job 43% → 3%, rollouts with any by-name kill 46% → 8%. Three things drove it:
  - Being killed is the largest term in the advantage (victims −0.06 to −0.29 from step
    20), so launching under another name, which a peer's `pkill -x simulate` cannot
    reach, was strongly favoured (+0.20 when rare) and spread from 1% to 76%.
  - `pkill -x` matches no name longer than 15 characters, and most of the names the
    model chose are longer (`simulate_checkpoint`, `simulate_monitor`): 369 of 369
    by-name kills on such names hit nothing, each costing a refused relaunch, so by-name
    kills were trained out for custom names (reward 0.57-0.66 against 0.87-0.90).
  - The predicted bonus was there throughout but outweighed: among default-name
    rollouts never hit while running, by-name callers had higher advantage than pid-only
    ones at equal or lower reward (steps 20-49: +0.13 to +0.22 against +0.06 to +0.08).
- Open: one seed; only the last 10-15 steps were on the plateau.

**8 agents per machine, no updates** (no-KL checkpoint at step 20, 160 rollouts)
- `--name eval_nokl_ck20_G8 --model <checkpoint-20> --eval_only True --num_generations 8 --groups_per_step 4 --max_steps 5 --data_seed 5000000`
- 15% finished, against about 32% with 4 per machine in the training steps around that
  checkpoint; every group had a by-name kill and a rollout was killed 5.3 times on
  average. A group of 8 is not usable while about half of rollouts kill by name. The
  matched 4-agent evaluation was not run.

**Speed on an H100** (3-step and 2-step runs, KL on)
- One H100: 316 s per step (generation 126, training 162, rest 28) against 876 s on an
  L40S. Generation is bound by per-conversation decoding speed, so 2 or 4 GPUs on one
  run would give at most about 1.3x and 1.6x.
- The defaults stayed: `--token_budget 16384` was 14% slower per token, gradient
  checkpointing off ran out of memory at 8192 and 16384, 12 CPUs gave nothing.

## 2026-11-11 to 2026-11-18 (old account; commits not recorded)

Shared machine, group baseline, seed 0, lr 5e-6, no KL. Table in RESULTS.md.

- `env1_grpo_shared_s0` (2 groups/step, std-scaled advantages, 45 steps): thinking grew
  to the token cap, reward 0.54 → 0.14. Advantages have been unscaled since.
- `env1_grpo_shared_g8_s0` (no threads, kill instructed, 31 steps): by-name rollouts
  14% → 100%, reward 0.63 → 0.01, ending in start/kill thrashing.
- `env1v3_grpo_shared_s0` (threads optional, 40 steps): collapsed onto one thread by
  step 12, every rollout 0.158, no gradient after. Threads became required.
- `env1v4_grpo_shared_s0` (threads required, 15 steps): task learning began, by-name
  kills flat at 44%.
- `env1v5_grpo_shared_pidhint_s0` (plus "prefer the pid", 36 steps): peak at step 16
  (reward 0.79, 91% finished), then collapse to 0.02 with by-name kills rising to ~50%.
