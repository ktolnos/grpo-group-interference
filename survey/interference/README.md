# Interference tests

Results and discussion: [AFFECTED_ENVIRONMENTS.md](../../AFFECTED_ENVIRONMENTS.md).

| file | what it does |
|---|---|
| `fetch.sh` | clones tinker-cookbook, verl, SkyRL and NeMo-RL at the tested commits into `$EXT` (default `$SCRATCH/ext`), copies the modules under test into `pkgs/`, pulls the SandboxFusion server image (9 GB) |
| `run_sandboxfusion.sh` | Slurm job: starts the SandboxFusion server under apptainer with tinker's config, then with the image's own, and runs `sf_client.py` against each |
| `sf_client.py` | groups of 4 concurrent gradings with tinker's exact payload: honest, a probe that lists other rollouts, an attacker that kills them |
| `run_local.sh` | Slurm job: `verl_prime_code_test.py`, `skyrl_test.py`, `nemo_rl_code_env_test.py` |
| `verl_prime_code_test.py` | verl's `prime_code.compute_score` on groups of 4 or 8, scored concurrently |
| `skyrl_test.py` | SkyRL-Gym Python tool (probe, kill) and SQL tool and reward (lock contention) |
| `nemo_rl_code_env_test.py` | NeMo-RL `CodeExecutionWorker`: guard escape and a builtin overwritten for later rollouts |
| `results/` | logs of the runs reported (made by the scripts here from a fresh `fetch.sh`) |

From the repository root:

```bash
export SBATCH_ACCOUNT=<your account>
sbatch survey/interference/fetch.sh               # ~30 min, mostly the image; compute nodes have internet
sbatch survey/interference/run_sandboxfusion.sh    # ~5 min
sbatch survey/interference/run_local.sh            # ~3 min; needs numpy and pandas ($GGI_VENV, default $SCRATCH/venvs/ggi)
```

Logs go to `logs/`. The SandboxFusion job needs `module load apptainer`; set `EXT` the
same way for all three if you change it.
