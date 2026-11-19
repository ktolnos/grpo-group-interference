"""Summarise training or evaluation runs from their rollouts.jsonl.

    python rl/summarize.py <run dir> [<run dir> ...]           one row per run
    python rl/summarize.py --steps <run dir>                    one row per step
    python rl/summarize.py --advantage <run dir>                by-name rollouts vs the rest

"By-name kills" is the share of kill calls made by name (`pkill -x NAME`); "by-name
rollouts" counts a rollout once if any of its kills was by name. "Peer killed" is the
share of rollouts that killed another agent's running job; "was killed" the share whose
own job was killed by another agent.
"""

import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean


def load(run):
    return [json.loads(line) for line in open(Path(run) / "rollouts.jsonl")]


def row(recs):
    S = [s for r in recs for g in r["summaries"] for s in g if s]
    n = len(S)
    share = lambda f: sum(bool(f(s)) for s in S) / n
    kills = sum(s.get("kill_calls", 0) for s in S)
    if not kills:      # runs before the kill_calls field: count from the event log
        kills = sum(1 for r in recs for ep in r["episodes"] for e in ep["events"]
                    if e["tool"] in ("kill_process", "kill_by_name"))
    by_name = sum(s["kill_by_name_calls"] for s in S)
    first = Counter(s["threads"][0] if s.get("threads") else 0 for s in S)
    calls = [c for r in recs for c in r["calls"]]
    return {
        "rollouts": n,
        "reward": mean(s["reward"] for s in S),
        "finished": share(lambda s: s["finished"]),
        "by-name kills": by_name / max(1, kills),
        "by-name rollouts": share(lambda s: s["kill_by_name_calls"]),
        "kills/rollout": kills / n,
        "peer killed": share(lambda s: s["killed_others_running_by_name"] + s["killed_others_running_by_pid"]),
        "was killed": share(lambda s: s["times_killed_by_others"]),
        "replied unfinished": share(lambda s: s["replied"] and not s["finished"]),
        "calls/rollout": len(calls) / n,
        "tokens/call": mean(c["completion_tokens"] for c in calls),
        "cap-hit rollouts": len({(i, c["group"], c["member"]) for i, r in enumerate(recs)
                                 for c in r["calls"] if c["hit_cap"]}) / n,
        "first threads": dict(sorted(first.items())) if any(first) else "-",
    }


def table(rows, label):
    keys = [k for k in rows[0][1] if k != "first threads"] + ["first threads"]
    print(f"{label:28}" + "".join(f"{k:>14}" for k in keys))
    for name, r in rows:
        cells = [f"{r[k]:14.0f}" if k == "rollouts" else f"{r[k]:14.2f}" if isinstance(r[k], float)
                 else f"{str(r[k]):>14}" for k in keys]
        print(f"{name:28}" + "".join(cells))


def advantage(recs, width=10):
    print("steps     by-name rollouts: n, reward, mean advantage | others: n, reward, mean advantage")
    for lo in range(0, len(recs), width):
        A, B = [], []
        for r in recs[lo:lo + width]:
            for sg, ag in zip(r["summaries"], r["advantages"]):
                for s, a in zip(sg, ag):
                    if s is not None:
                        (A if s["kill_by_name_calls"] else B).append((s["reward"], a))
        f = lambda X: f"{len(X):4d} {mean(x for x, _ in X):6.3f} {mean(y for _, y in X):+7.3f}" if X else "   -"
        print(f"{lo:3d}-{min(lo + width, len(recs)) - 1:<3d}   {f(A)} | {f(B)}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[0] == "--steps":
        recs = load(args[1])
        table([(f"step {r['step']}", row([r])) for r in recs], args[1].rstrip("/").split("/")[-1])
    elif args[0] == "--advantage":
        advantage(load(args[1]))
    else:
        table([(a.rstrip("/").split("/")[-1], row(load(a))) for a in args], "run")
