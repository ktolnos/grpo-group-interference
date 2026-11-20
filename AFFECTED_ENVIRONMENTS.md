# RL environments where rollouts in a group can interfere (2026-11-19)

Question: in which widely used RL training setups, configured as their documentation
says, can one rollout lower the reward of other rollouts in its group, and can a model
find this out from what it observes? Under a group baseline (GRPO and relatives) such
harm raises the harming rollout's advantage, which is what `deadlock_recovery/` studies on purpose.

Scripts, download steps and logs: [`survey/interference/`](survey/interference/).

## How the tests work

Each test runs the framework's own code: modules copied unmodified (only import paths
changed), or the published server image. Rollouts are graded concurrently, as in
training. Honest rollouts submit a correct answer. One rollout (the "attacker")
submits about a dozen lines of ordinary Python or SQL that a model could write. Every trial
is reported; none were dropped. All tests together take about 1 CPU-hour; the
SandboxFusion image is 9 GB.

Docker is not available on Killarney, so the SandboxFusion image ran under apptainer as
the job's user instead of root. Requests share one user either way, which is what lets
them see and signal each other.

## Tested: interference confirmed

| # | Environment (framework, commit) | Setup as documented | What the attacker does | Attacker's reward | Honest peers | Trials | What a model sees of its peers |
|---|---|---|---|---|---|---|---|
| 1 | Code RL with a `check_solution` tool: tinker-cookbook `recipes/code_rl` (28de41e) | SandboxFusion server started as its README says: one container for all rollouts, `isolation: none`, group size 4, 4 concurrent requests | lists running tests through `/proc` (process, temp folder, files); kills them | passed 8/8 | passed 0/24 (killed, return code −9) | 4 with tinker's config, 4 with the image's own | the tool result includes the program's output (up to 300 characters), so `ls /tmp` or `/proc` shows the other rollouts at once |
| 2 | SandboxFusion server image as shipped (`volcengine/sandbox-fusion:server-20250609`; the local deployment that verl's sandbox reward and tools point to) | image default config is also `isolation: none` | same | (in row 1) | (in row 1) | (in row 1) | depends on the client |
| 3 | Code reward: verl `prime_code` (a4e0dd2). rllm's code reward (used for DeepCoder) and ROLL's (`use_local=True` default) are built the same way | apps/codeforces/taco data without a sandbox URL is graded locally: one forked process per sample, all children of one reward worker, scored concurrently (`run_in_executor` + `gather`), guarded by `reliability_guard` | kills sibling test processes with `posix.kill` (the guard only removes `os.kill`) | 1.0 in 5/5 | 0.0 in 19/19 | 5 groups of 4 or 8 | nothing during the episode (single-turn); only the reward |
| 4 | Python tool: SkyRL-Gym `PythonCodeExecutorToolGroup` (387bce8), used by its `searchcode` env | `subprocess.run(["python", "-c", code])` on the training machine, no guard | `ps` shows other rollouts' full code; kills them | — | outputs emptied 3/3 | 1 probe, 1 kill | the full output |
| 5 | Text-to-SQL: SkyRL-Gym SQL tool and reward (SkyRL-SQL recipe) | rollouts on one database share one SQLite file; each call is `BEGIN; <one statement>; ROLLBACK` (nothing persists) | a large insert, rolled back at the 5 s timeout, holds the write lock; three in a row | (no gain possible) | three in a row: correct answers scored 1.0 → 0.0, 6/6. One insert: only ~4.5 s delay, 6/6 still 1.0 | 2 single, 2 triple attacks, 3 peers each | only "SQL Timeout"; the harm is invisible |
| 6 | Python tool: NeMo-RL `CodeEnvironment` (1fcd48d), registered as the "code" environment; no example config uses it | one long-lived Ray worker runs many rollouts' code in turn with `exec`, sharing one globals dictionary; `import os` blocked, `open` confined to the rollout's folder | overwrites a builtin (`__builtins__['sum'] = lambda *a, **k: 0`); escapes the guard through `open.__func__.__globals__['os']` | — | later rollouts on that worker computed `sum([1, 2, 3])` = 0 | 1 sequence of 6 calls | the full result; the poisoning persists for every later rollout on the worker, in any group |

## Not yet tested

Ordered by how useful a test would be for us.

| Environment | Why rollouts may interfere | Status | What a test needs |
|---|---|---|---|
| WebArena / VisualWebArena (BrowserGym, AgentLab) | One set of shared, writable websites (shop admin, forum, GitLab, CMS). AgentLab: "an agent on task 323 could alter the instance state, making task 201 impossible"; it "does not support evaluations across multiple instances" and resets before a study, not between rollouts. A GRPO group does the same task on the same site at once (creating, editing or deleting the same post or product). | documentation read | the site images (tens of GB, Docker); could be tried under apptainer, or on a machine with Docker |
| SandboxFusion as a multi-turn code-interpreter tool (verl's `sandbox_fusion` tool, ReTool-style tool-integrated reasoning) | same shared server as rows 1-2, but raw code with no test harness or guard, and the full output returned each turn | follows from rows 1-2; client not run | verl's tool client against the image above, a few CPU minutes |
| Leftover background process under sequential grading (ROLL's code reward worker grades a batch in a loop) | `reliability_guard` removes `os.fork` but not `posix.fork`; a forked child outlives its test and can kill or slow the next samples graded on that worker, so grading need not be concurrent | not checked | a variant of the verl test with samples graded one after another |
| verifiers `SubprocessBackend` (PrimeIntellect 18a4635; the default runtime of NeMo-Gym tasksets) | a working folder per rollout, same machine and user: the mechanism of row 4 | code read | `pip install verifiers`; two runtimes, `ps` and `kill` |
| Memory or CPU exhaustion in a shared sandbox | SandboxFusion limits memory per process, not per rollout. One rollout can push the container to its memory limit, so the kernel may kill other rollouts' processes, or saturate the CPUs so peers hit their time limits. | not checked | rows 1-2 with a memory or CPU hog instead of a kill |
| Rewards that measure time (kernel-speed RL such as KernelBench-style speedups, efficiency rewards) | peers' code timed on a shared GPU or CPU is slowed by the attacker, lowering their measured speedup | not checked | depends on the recipe; check whether it times kernels concurrently on one device |
| Shared retrieval server (Search-R1) | all rollouts query one read-only server; only latency is shared | code read; unlikely to matter | — |

## Isolated by default (not candidates)

From their code or documentation; only the first two rows were checked in code here.

| Framework | Default |
|---|---|
| open-r1 code reward (5b6ff22) | E2B remote sandboxes (`provider_type` default `"e2b"`; a `"local"` option exists) |
| AReaL (4e8dc43), OpenRLHF (2649449) | no built-in code execution found in the repository; rewards and sandboxes are user-supplied |
| verifiers default (Prime sandboxes), Terminal-Bench/Harbor, SWE-style Docker tasks, OSWorld, AppWorld, tau-bench | one container, VM or in-memory state per rollout |

## Reading of the results

- **Interference is easy wherever code runs on a shared machine.** In rows 1-4 and 6 all
  rollouts run as one user in one process namespace (row 6 even in one Python process).
  A dozen lines ended or corrupted every peer's run and kept the attacker's own pass,
  every time. Python-level guards (`reliability_guard`, blocked imports) do not stop it.
- **Noticing it differs by setup.** Where a tool returns program output (rows 1, 4, 6),
  listing `/tmp` or `/proc`, or printing `__builtins__`, shows other rollouts' folders,
  test files, processes or state on the first try. In single-turn code rewards (row 3)
  and in SQL (row 5) the model observes nothing about its peers, so only the reward
  (and, under a group baseline, the advantage) could teach it.
- **Timing matters for kills.** Harm needs the attacker's code to run while peers'
  tests run. verl scores a batch's samples concurrently, so this always holds; in
  tinker it holds when group members are graded at similar times. The attackers waited
  up to 4.5 s, inside the per-test time limits. Row 6 needs no overlap: the damage persists.
- **Side observation:** under SandboxFusion the rollout's own `test_cases.txt` (the
  expected outputs) sits in its working folder, a separate, well-known reward hack.

Best natural candidates for our purpose: **tinker code_rl with SandboxFusion**
(popular, shares one container by default, and the tool output lets the model see its
peers) and **WebArena-style shared websites** (very popular, and interference comes from
doing the task itself). verl's local code reward is the most common default and the
harm is certain, but a model could learn it only from the reward.
