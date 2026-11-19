# Interference between rollouts in widely used RL environments (2026-11-19)

Question: in which widely used RL training setups, run as their documentation configures
them, can one rollout lower the reward of another rollout in its group, and can a
model find this out from what it observes?

Each test runs the framework's own code (copied unmodified, or the server image
as shipped) with a group of rollouts graded concurrently, as in training. The honest
rollouts submit a correct solution. One rollout (the attacker) submits code a model could
write. Every trial is listed in the logs; nothing was dropped.

## Confirmed by running the code

| Environment (framework, commit) | Default setup | What one rollout can do to its group | Attacker's reward | Honest peers' reward | Trials | What the model sees |
|---|---|---|---|---|---|---|
| Code RL with a `check_solution` tool (tinker-cookbook `recipes/code_rl`, 28de41e) | SandboxFusion server from its README: one container, `isolation: none`, same user, group size 4, 4 concurrent requests | list every running test (`/proc`), its temp dir and files; kill it | passed 8/8 | passed 0/24 (killed, rc −9) | 4 with tinker's config + 4 with the image's own | tool result includes the program's output (300 characters): other rollouts' processes and files are visible on the first `ls /tmp` or `/proc` |
| Same server, image's own config (any user of SandboxFusion's local deployment, e.g. verl's sandbox-fusion reward and tools) | `isolation: none` is also the image default | same | — | — | (rows above) | — |
| Code reward (verl `prime_code`, a4e0dd2; rllm's code reward, used for DeepCoder, does the same by default) | codecontests/apps/codeforces/taco without a sandbox URL run locally: each sample in a `multiprocessing.Process` forked from one reward worker, scored concurrently (`run_in_executor` + `gather`) | kill siblings; `reliability_guard` sets `os.kill = None` but `posix.kill` works | 1.0 in 5/5 | 0.0 in 19/19 | 5 groups of 4 or 8 | nothing during the episode (single-turn reward); only the reward |
| Python tool (SkyRL-Gym `PythonCodeExecutorToolGroup`, 387bce8; `searchcode` env) | `subprocess.run(["python", "-c", code])` on the trainer's machine, no guard | `ps` shows other rollouts' full code; kill them | — | outputs emptied 3/3 | 1 probe + 1 kill | full stdout returned |
| Text-to-SQL (SkyRL-Gym SQL tool and reward, SkyRL-SQL recipe) | rollouts on one `db_id` share one SQLite file; each call is `BEGIN; <statement>; ROLLBACK` | a large rolled-back `INSERT` holds the write lock for the 5 s tool timeout; three in a row make peers' tool calls and final scoring time out | (unchanged; nothing it can gain) | reward 1.0 → 0.0 for 6/6 peers (correct answers) | 2 single + 2 triple attacks, 3 peers each; one call alone only delays peers by ~4.5 s (6/6 still scored 1.0) | only "SQL Timeout"; the effect on peers is invisible |

Logs of the runs above: `results/`.
Reproduce from the repository root: `bash survey/interference/fetch.sh` (clones at the
commits above; pulls the SandboxFusion image, ~9 GB, best done as a job), then
`sbatch survey/interference/run_sandboxfusion.sh` and `sbatch survey/interference/run_local.sh`
(CPU, a few minutes each). Docker is unavailable on Killarney, so the SandboxFusion image ran under
apptainer as the job's user rather than root; same-user processes can see and signal
each other either way.

## Found by reading the code or documentation, not run

| Environment | Why rollouts can interfere | How easy to notice |
|---|---|---|
| WebArena / VisualWebArena via BrowserGym/AgentLab (web agents) | one set of shared, writable websites (shop admin, forum, GitLab, CMS). AgentLab: "an agent on task 323 could alter the instance state, making task 201 impossible"; it "does not support evaluations across multiple instances" and resets between studies, not between rollouts. In GRPO all members of a group do the same task on the same site at once (e.g. create, edit or delete the same post or product). | very easy: peers' posts, edits and deletions show up on the page |
| verifiers `SubprocessBackend` (PrimeIntellect, 18a4635; NeMo-Gym tasksets default to it) | per-rollout working directory on the same machine and user; same mechanism as the SkyRL Python tool above | easy with any shell tool (`ps`) |
| Default verifiers/prime runtimes, Terminal-Bench/Harbor, SWE-style Docker tasks, OSWorld, AppWorld, tau-bench | one container, VM or in-memory state per rollout, by their documentation (not checked here) | — (isolated; not candidates) |

## Reading of the results

- **Interference is easy where code runs.** In the three local-execution setups (SandboxFusion as
  deployed by its README, verl's local code reward, SkyRL's Python tool), all group members run as one
  user in one process namespace. A dozen lines of Python ended every peer's run and kept
  the attacker's own pass, every time. Python-level guards (`reliability_guard`) do not stop it.
- **Noticing it differs by setup.** A model gets direct evidence where a tool returns
  program output (tinker's `check_solution`, SkyRL's Python tool): listing `/tmp` or
  `/proc` shows other rollouts' working directories, test files and processes. In
  verl's single-turn code reward and in SkyRL-SQL the model observes nothing about its
  peers, so only the reward (and, under a group baseline, the advantage) could teach it.
- **Timing matters.** Harm needs the attacker to run while peers' tests run. verl
  scores a batch's samples concurrently, so this always holds; in tinker it holds when
  group members grade at similar times. The attackers here waited up to 4.5 s, inside
  the per-test timeouts.
- **Side observation:** under SandboxFusion the rollout's own `test_cases.txt`
  (the expected outputs) sits in its working directory (the probe listed it), a separate,
  well-known reward hack.

For our purpose (a group baseline that rewards harming peers), the best natural
candidates are **tinker code_rl with SandboxFusion** (popular, defaults to a shared
container, the tool output lets the model see peers) and **WebArena-style shared
websites** (very popular, interference arises from doing the task itself). verl's
local code reward is the most common default and the harm is certain, but a model
can only learn it from reward.
