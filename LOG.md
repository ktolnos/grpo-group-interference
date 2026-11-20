# Experiment log

One short entry per launched experiment: what was run and what it showed. Newest first.
Fuller tables are in [RESULTS.md](RESULTS.md).

Before launching, commit; a run is reproduced by its command at its commit (both are
also written to the run's `run.json`). The code is not kept able to rerun old entries:
check out the entry's commit for that.

Commands are arguments to `sbatch jobs/train.sh`. Outputs are in
`$SCRATCH/ggi/rl/runs/<name>/`.

## 2026-11-20: how much reward the trained models take from others

Commit `COMMIT`. `sbatch jobs/eval.sh <tag> <model> [flags]` runs three evaluations of 160
rollouts on seeds from 5000000: shared, isolated (same seeds and deadlocks), and crossplay
(one rollout among three scripted agents that touch only their own jobs; their reward
loss is exact). Outputs `deadlock_recovery_{shared,isolated,crossplay}_eval_<tag>`.

- `base_noname Qwen/Qwen3-4B --start_name_arg False`
- `base_name Qwen/Qwen3-4B`
- `noname_s0_ck60 <deadlock_recovery_shared_grpo_noname_s0/checkpoint-60> --start_name_arg False`
- `name_s0_ck60 <deadlock_recovery_shared_grpo_name_s0/checkpoint-60>`
- Running.

## 2026-11-20: one pid per kill

Commit `8b186c1`. As the entry below, but `kill` takes a single pid again (a list of names
is still allowed), and "prefer the pid" is on by default. 1 H100 each, jobs 7812090 and 7812091.

**Training with and without the job-name argument** (60 steps, seed 0, group baseline)
- `--name deadlock_recovery_shared_grpo_name_s0 --seed 0`
- `--name deadlock_recovery_shared_grpo_noname_s0 --seed 0 --start_name_arg False`
- Question: with the escapes closed (renaming, in the second run; by-name kills that
  cannot match long names, in both), does interference with groupmates rise under the
  group baseline?
- Both ran 60 steps (7.6 h and 6.9 h). Neither escaped interference, and in neither did
  it fall.
- With the job name: interference rose to saturation and the task was lost. Rollouts that
  killed a groupmate's running job 49% → 96% (steps 0-9 → 50-59), by-name kills hitting a
  groupmate 57% → 92%, reward 0.10 → 0.18 (steps 10-19) → 0.07, 1% finished at the end,
  21 model calls per rollout (kill/restart thrashing). Renaming did not spread this time
  (3-13%).
- Without the job name: the task was half learned and interference stayed. Reward 0.18 →
  0.52, finished 11% → 56%, flat from step 30; interfering rollouts 32% → 43-53%; by-pid
  kills hitting a groupmate 7% → 14-21%; about 70% of rollouts were killed by a groupmate
  throughout.
- In every block of both runs, interfering rollouts had positive mean advantage and the
  rest negative (no-name run: +0.014 to +0.055 against −0.011 to −0.045; with the name:
  +0.002 to +0.018 against −0.017 to −0.044).
- With the name, the policy itself moved to killing by name: the first kill of a rollout
  nobody had yet hit was by name 54% → 89% of the time (flat at 19-27% without the name).
  By-name kills made when the caller's own job was already dead, which do nothing for the
  caller, went from 0.38 to 4.1 per rollout, each stopping 1.3 groupmates' running jobs.
- The `name` argument alone raises by-name killing in the untrained model: 60% of
  rollouts in steps 0-4 with it, 33% without.
- The reasoning never mentions other users or agents. Agents take every `simulate` in the
  listing for their own ("kill 189823 (the deadlocked one) and let 189824 continue"), and
  by-pid hits come from killing the one `simulate` listed after their own was killed.
- Open: no control yet (other-groups baseline, isolated machines), so the rise is not
  attributed to the group baseline; one seed each. Why the by-name rate stayed flat
  without the name, with the same sign of advantage, is not explained.

## 2026-11-19 (evening): lists of pids, process list in wait results

Commit `1d2822d`. `kill` takes lists of pids or names and matches names on their first 15
bytes; a wait's result ends with the status output; lr 2e-6, no KL. 1 H100 each.

**Untrained model** (160 rollouts, no updates)
- `--name deadlock_recovery_shared_eval_base --eval_only True --max_steps 5 --data_seed 5000000 --kill_pid_hint True`
- Reward 0.11, 4% finished. 36% of rollouts killed a groupmate's running job; 36% had a
  by-name kill that hit a groupmate and 17% a by-pid kill that did (3-5% in the first
  steps of earlier runs). 43 of the 46 by-pid calls that hit a groupmate were lists: the
  agent takes every `simulate` in the listing for its own and kills them all, in one
  case after reasoning that pids are safer than names.

**Training with and without the job-name argument** (cancelled at steps 14 and 18)
- `--name deadlock_recovery_shared_grpo_name_s0 --kill_pid_hint True --seed 0`
- `--name deadlock_recovery_shared_grpo_noname_s0 --kill_pid_hint True --seed 0 --start_name_arg False`
- Kept as `..._name_pidlist_s0` and `..._noname_pidlist_s0`.
- Without the job name, list kills grew with task learning: 0.39 → 1.25 per rollout,
  rollouts whose by-pid kill hit a groupmate 16% → 41% (steps 0-4 → 15-16), reward 0.15 →
  0.29. With the job name, no trend in 13 steps.
- Rollouts whose kills hit a groupmate earned less and had positive advantage (no-name
  run: reward 0.17-0.18, advantage +0.012 by pid and +0.033 by name; rollouts that hit
  nobody 0.22, −0.023).
- Stopped because a list kill cannot be told apart from clearing one's own deadlock
  without working out which pid is one's own.

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
