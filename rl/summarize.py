"""Summarise training or evaluation runs from their rollouts.jsonl.

    python rl/summarize.py <run dir> [<run dir> ...]           one row per run
    python rl/summarize.py --steps <run dir>                    one row per step
    python rl/summarize.py --advantage <run dir>                interfering rollouts vs the rest

Interference is counted by what a kill call hit, not by how it was addressed
(`harness.kill_outcomes`, recomputed here from each episode's event log):
  harm done       reward a rollout's kills took from its groupmates, as estimated from
                  the event log (`harness.harm`); undisturbed is the reward the rollouts
                  would have averaged without it. exact harm is the same quantity measured
                  against scripted peers (`train.py --eval_peers`), where there is one
  interfered      rollouts that killed a groupmate's running job, by any means
  by name, by pid rollouts with a by-name / by-pid kill that hit a groupmate's job,
                  running or deadlocked
  by-name         rollouts with any kill by name (`pkill -x`), whoever it hit
  shared name     rollouts whose job name matches a groupmate's in its first 15 bytes
  was killed      rollouts whose job was killed by a groupmate
"""

import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deadlock_recovery import harness  # noqa: E402


def load(run):
    """The run's steps, each rollout summary extended with its kill outcomes."""
    recs = [json.loads(line) for line in open(Path(run) / "rollouts.jsonl")]
    for r in recs:
        episodes = iter(r["episodes"])
        for group in r["summaries"]:
            # A shared group is one episode; an isolated rollout is an episode of its own.
            ep = next(episodes) if r["shared_machine"] and any(group) else None
            for s in group:
                if s is not None:
                    e = ep if r["shared_machine"] else next(episodes)
                    assert e["agents"][s["agent"]]["reward"] == s["reward"]
                    s.update(harness.kill_outcomes(s["agent"], e["events"]))
                    s.update(harness.harm(config(e), e["events"], e["agents"])[s["agent"]])
    return recs


def config(episode):
    known = harness.Config.__dataclass_fields__
    return harness.Config(**{k: v for k, v in episode["config"].items() if k in known})


def interfered(s):
    return s["killed_others_running_by_name"] + s["killed_others_running_by_pid"]


def row(recs):
    S = [s for r in recs for g in r["summaries"] for s in g if s]
    n = len(S)
    share = lambda f: sum(bool(f(s)) for s in S) / n
    first = Counter(s["threads"][0] if s["threads"] else 0 for s in S)
    calls = [c for r in recs for c in r["calls"]]
    return {
        "rollouts": n,
        "reward": mean(s["reward"] for s in S),
        "finished": share(lambda s: s["finished"]),
        "harm done": mean(s["harm_done"] for s in S),
        "undisturbed": mean(s["reward"] + s["harm_received"] for s in S),
        "exact harm": mean(s["harm_to_scripted"] for s in S) if "harm_to_scripted" in S[0] else "-",
        "interfered": share(interfered),
        "by name": share(lambda s: s["interfering_calls_by_name"]),
        "by pid": share(lambda s: s["interfering_calls_by_pid"]),
        "by-name": share(lambda s: s["kill_by_name_calls"]),
        "shared name": share(lambda s: s["name_shared"]),
        "was killed": share(lambda s: s["times_killed_by_others"]),
        "kills/rollout": sum(s["kill_calls"] for s in S) / n,
        "stopped early": share(lambda s: s["replied"] and not s["finished"]),
        "calls/rollout": len(calls) / n,
        "tokens/call": mean(c["completion_tokens"] for c in calls),
        "cap-hit": len({(i, c["group"], c["member"]) for i, r in enumerate(recs)
                        for c in r["calls"] if c["hit_cap"]}) / n,
        "first threads": dict(sorted(first.items())) if any(first) else "-",
    }


def table(rows, label):
    keys = [k for k in rows[0][1] if k != "first threads"] + ["first threads"]
    w = max(len(label), *(len(name) for name, _ in rows)) + 2
    print(f"{label:{w}}" + "".join(f"{k:>14}" for k in keys))
    for name, r in rows:
        cells = [f"{r[k]:14.0f}" if k == "rollouts" else f"{r[k]:14.2f}" if isinstance(r[k], float)
                 else f"{str(r[k]):>14}" for k in keys]
        print(f"{name:{w}}" + "".join(cells))


def advantage(recs, width=10):
    print("steps     interfering rollouts: n, reward, mean advantage | others: n, reward, mean advantage")
    for lo in range(0, len(recs), width):
        A, B = [], []
        for r in recs[lo:lo + width]:
            for sg, ag in zip(r["summaries"], r["advantages"]):
                for s, a in zip(sg, ag):
                    if s is not None:
                        (A if interfered(s) else B).append((s["reward"], a))
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
