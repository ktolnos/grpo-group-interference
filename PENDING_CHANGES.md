# Pending changes to Environment 1

Changes we have decided are worth making, but not on their own: each is applied only
when something else forces a rerun, so that results already collected stay comparable.
Remove an entry once it is applied, noting the commit.

## Applied

- 2026-11-15: `wait` after a full wait says `still running after N min` (was `no change`,
  which the model read as "no progress"); `threads` is required and the request asks for
  more than one thread, after `env1v3_grpo_shared_s0` collapsed onto one thread (a
  zero-variance reward floor of 0.158) by step 12.

## Optional

- `kill` with neither pid nor name answers `kill: usage: kill PID`, which does not
  mention the name form.
