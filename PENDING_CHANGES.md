# Pending changes to Environment 1

Changes we have decided are worth making, but not on their own: each is applied only
when something else forces a rerun, so that results already collected stay comparable.
Remove an entry once it is applied, noting the commit.

## Wording

- **`wait` after a full wait says "no change", which reads as "no progress".**
  The text is `simulate: no change after 10 min` followed by `running, step 9/24`. "No
  change" means the job is still running, but the untrained model reads it as stuck:
  in `eval_threads` 14 of 55 kill-by-name calls were on a healthy job, with reasoning
  like "the process is stuck... kill it and restart". Proposed: `simulate: still running
  after 10 min`. (harness.py, the line that builds the wait result.)

## Open: the threads environment collapsed under training

`env1v3_grpo_shared_s0` (job 7627384, 40 steps, 2026-11-15) converged on one thread and
no kills by step 12, after which every group had identical rewards (0.158 = progress
19/24 x 0.2) and no gradient. One thread is a zero-variance floor; the untrained model
rarely handles a deadlock (it kills its own deadlocked job in about a quarter of
rollouts), so trying more threads scored below the floor on average and was trained
away. Kill by name fell from 19% to 0% along with it, so the run says nothing about
the group baseline. Candidates, not yet chosen:

- `threads` required, or a default above 1, so every episode meets deadlocks and the
  model must learn to handle them; the trade-off is then explored from above.
- Say again in the request that a hung job must be killed before relaunching, if the
  model cannot discover it from the "locked by pid N" error.
- A limit on tool calls per message. Late in the run replies held up to ~100 queued
  `status`/`wait` calls, and these long lists are most of the calls that hit the token
  cap (56% of rollouts at step 39).

## Optional

- `kill` with neither pid nor name answers `kill: usage: kill PID`, which does not
  mention the name form.
