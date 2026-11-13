"""Run episodes of Environment 1 and append one JSON line per episode.

Examples:
  python run.py --policy scripted:careful,careful,broad,hider --episodes 3 --out /scratch/eop/outputs/ggi/env1/smoke.jsonl
  python run.py --policy llm:gpt-oss-120b --episodes 5 --out /scratch/eop/outputs/ggi/env1/gpt-oss.jsonl
"""

import argparse
import json
import os
import time

from harness import Config, run_episode
from policies import llm_policy, scripted

p = argparse.ArgumentParser()
p.add_argument("--policy", required=True, help="scripted:<kind,kind,...> or llm:<model>")
p.add_argument("--episodes", type=int, default=1)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", required=True)
p.add_argument("--G", type=int, default=4)
p.add_argument("--steps", type=int, default=30)
p.add_argument("--deadline", type=int, default=60)
p.add_argument("--isolation", default="unshare", choices=["unshare", "none"])
p.add_argument("--no-kill-by-name", action="store_true")
p.add_argument("--hide-pid-in-status", action="store_true")
p.add_argument("--rpm", type=float, default=100)
p.add_argument("--max-tokens", type=int, default=4096)
p.add_argument("--temperature", type=float, default=1.0)
p.add_argument("--reasoning-effort", default=None)
a = p.parse_args()

cfg = Config(G=a.G, steps=a.steps, deadline=a.deadline, isolation=a.isolation,
             kill_by_name_tool=not a.no_kill_by_name, status_shows_pid=not a.hide_pid_in_status)
kind, _, spec = a.policy.partition(":")
client = None
if kind == "scripted":
    policy = scripted(spec.split(","))
else:
    from llm import Client
    params = {"max_tokens": a.max_tokens, "temperature": a.temperature}
    if a.reasoning_effort:
        params["reasoning_effort"] = a.reasoning_effort
    client = Client(spec, rpm=a.rpm, **params)
    policy = llm_policy(client)

os.makedirs(os.path.dirname(a.out), exist_ok=True)
work = os.environ.get("SLURM_TMPDIR") or os.path.dirname(a.out)
KEYS = ["finished", "final_step", "tool_calls", "killed_own_stuck", "killed_own_running",
        "killed_others_by_pid", "killed_others_by_name", "times_killed_by_others", "renamed"]
with open(a.out, "a") as f:
    for i in range(a.episodes):
        t0 = time.time()
        ep = run_episode(cfg, policy, a.seed + i, workdir=work)
        ep["policy"], ep["wall_seconds"] = a.policy, round(time.time() - t0, 1)
        if client:
            ep["usage"], ep["api_errors"] = dict(client.usage), dict(client.errors)
        f.write(json.dumps(ep) + "\n")
        f.flush()
        print(f"episode seed={ep['seed']} rounds={ep['rounds']} wall={ep['wall_seconds']}s "
              f"hang plans={ep['plans']}")
        for s in ep["agents"]:
            print("  agent", s["agent"], {k: s[k] for k in KEYS}, flush=True)
        if client:
            print("  usage", client.usage, "errors", client.errors, flush=True)
