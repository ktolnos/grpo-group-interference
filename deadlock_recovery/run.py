"""Run episodes of Deadlock Recovery and append one JSON line per episode.

Examples:
  python run.py --policy scripted:careful,careful,broad,hider --episodes 3 --out $SCRATCH/ggi/deadlock_recovery/smoke.jsonl
  python run.py --policy llm:gpt-oss-120b --episodes 5 --out $SCRATCH/ggi/deadlock_recovery/gpt-oss.jsonl
"""

import argparse
import json
import os
import time

T_RUN = time.perf_counter()

from harness import Config, run_episode
from policies import llm_policy, scripted, with_scripted_peers

p = argparse.ArgumentParser()
p.add_argument("--policy", required=True, help="scripted:<kind,kind,...> or llm:<model>")
p.add_argument("--episodes", type=int, default=1)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", required=True)
p.add_argument("--G", type=int, default=4)
p.add_argument("--no-kill-by-name", action="store_true")
p.add_argument("--kill-pid-hint", action="store_true",
               help="the kill tool's description says to prefer the pid (the training configuration)")
p.add_argument("--rpm", type=float, default=100)
p.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint; default: Vector Inference")
p.add_argument("--no-key", action="store_true", help="for a local server that needs no API key")
p.add_argument("--no-start-name", action="store_true",
               help="start_job takes no name: every job runs under the default name")
p.add_argument("--parallel", type=int, default=1, help="episodes to run at the same time")
p.add_argument("--scripted-peers", default=None,
               help="with an llm policy: agent 0 is the model and agents 1.. follow these "
                    "scripted kinds (comma-separated: careful, broad, hider), e.g. "
                    "'broad,careful,careful'. Read agent 0's row of the results")
p.add_argument("--max-tokens", type=int, default=8192,
               help="cap on reasoning plus reply; a cap, not a target")
p.add_argument("--temperature", type=float, default=1.0)
p.add_argument("--reasoning-effort", default=None, choices=["low", "medium", "high"],
               help="for models that take it (gpt-oss); the server default applies if unset")
p.add_argument("--no-thinking", action="store_true",
               help="for Qwen models served by vLLM: turn the reasoning block off")
a = p.parse_args()

cfg = Config(G=a.G, kill_by_name_tool=not a.no_kill_by_name, kill_pid_hint=a.kill_pid_hint,
             start_name_arg=not a.no_start_name)
kind, _, spec = a.policy.partition(":")
client = None
if kind == "scripted":
    policy = scripted(spec.split(","))
else:
    from llm import Client
    params = {"max_tokens": a.max_tokens, "temperature": a.temperature}
    if a.reasoning_effort:
        params["reasoning_effort"] = a.reasoning_effort
    if a.no_thinking:
        params["chat_template_kwargs"] = {"enable_thinking": False}
    client = Client(spec, base_url=a.base_url, rpm=a.rpm,
                    **({"key_var": None} if a.no_key else {}), **params)
    policy = llm_policy(client)
    if a.scripted_peers:
        policy = with_scripted_peers(policy, a.scripted_peers.split(","))

os.makedirs(os.path.dirname(a.out), exist_ok=True)
work = os.environ.get("SLURM_TMPDIR") or os.path.dirname(a.out)
KEYS = ["reward", "finish_round", "model_calls", "killed_own_stuck", "killed_own_running",
        "killed_others_running_by_name", "killed_others_stuck_by_name",
        "killed_others_running_by_pid", "times_killed_by_others", "renamed", "kill_calls",
        "kill_by_name_calls", "threads"]
timings = []


def one(i):
    t0 = time.time()
    try:
        ep = run_episode(cfg, policy, a.seed + i, workdir=work)
    except Exception:                     # one broken episode must not lose the others
        import traceback
        return {"seed": a.seed + i, "error": traceback.format_exc()}
    ep["policy"], ep["wall_seconds"] = a.policy, round(time.time() - t0, 1)
    return ep


from concurrent.futures import ThreadPoolExecutor
with open(a.out, "a") as f, ThreadPoolExecutor(a.parallel) as pool:
    for ep in pool.map(one, range(a.episodes)):
        if "error" in ep:
            print(f"episode seed={ep['seed']} FAILED\n{ep['error']}", flush=True)
            continue
        if client:
            ep["usage"], ep["api_errors"] = dict(client.usage), dict(client.errors)
        f.write(json.dumps(ep) + "\n")
        f.flush()
        print(f"episode seed={ep['seed']} rounds={ep['rounds']} wall={ep['wall_seconds']}s")
        for s in ep["agents"]:
            print("  agent", s["agent"], {k: s[k] for k in KEYS}, flush=True)
        print("  timing", ep["timing"], flush=True)
        if client:
            print("  usage so far", client.usage, "errors", client.errors, flush=True)
        timings.append(ep["timing"])
if timings:
    print("mean per episode:", {k: round(sum(t[k] for t in timings) / len(timings), 3)
                                for k in timings[0]})
    run_s, ep_s = time.perf_counter() - T_RUN, sum(t["total_s"] for t in timings)
    # End to end for the whole script. With one episode at a time, the rest is imports,
    # client setup and writing records; with several at once the episodes overlap.
    if a.parallel == 1:
        print(f"run total {run_s:.2f}s = episodes {ep_s:.2f}s + outside episodes {run_s - ep_s:.2f}s")
    else:
        print(f"run total {run_s:.2f}s for {len(timings)} episodes, {a.parallel} at a time "
              f"(episode times add up to {ep_s:.2f}s)")
