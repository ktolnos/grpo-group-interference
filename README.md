# GRPO group interference

Does a group-relative baseline (GRPO, leave-one-out) reward a rollout for lowering the
rewards of the other rollouts in its group? Started 2026-11-11 as a spin-off of
`../grpo-gradient-hacking` (see `spite/NOTE.md` there for the first note).

| path | contents |
|---|---|
| `paper.md` | working draft: theory, classes of interference, mitigations, simulations, related work, planned experiments |
| `sim/sim.py` | tabular simulation (sabotage, defense, five advantage estimators) |
| `sim/analyze.py` | figures and summary; reads `/scratch/eop/outputs/ggh/spite/` by default |
| `sim/run.sh` | Slurm script, CPU only; submit from `/scratch/eop` |
| `sim/figures/` | figures 1–4 of the draft |
| `KNOWN_RELATED_WORK.md` | 166 related works with links, abstracts (arXiv ones exact, from the arXiv API), relation to our claim, and what was read |
| `notes/related/` | the five search reports behind it, and the scripts that merged them |
| `drafts/` | earlier versions of `paper.md` |

Simulation outputs are in `/scratch/eop/outputs/ggh/spite/` (job 7355443).

Status: theory and tabular simulations done. The theory is a special case of Gao et al.,
arXiv:2611.02404 (2 Nov 2026); what is open is a demonstration that training produces
interference through shared machines or joint judges. `env1/` is a test environment for that
(several agents each run a job on one machine and can kill or rename job processes) and `rl/`
trains Qwen3-4B on it with GRPO. One exploratory training run on an earlier version of the
environment exists (one seed, no controls); the environment has changed since, so no result
is final. Every larger run needs an approved design first (see `paper.md`, Section 7).
