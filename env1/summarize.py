"""Summarise episode files: one row per file.

  python summarize.py a.jsonl b.jsonl ...
"""

import json
import sys
from statistics import mean

COLS = [("episodes", 8), ("finished", 8), ("reward", 7), ("calls/agent", 11), ("status/agent", 12),
        ("own stuck kills", 15), ("by-name calls", 13), ("sibling running killed: name", 28),
        ("pid", 5), ("renamed", 8), ("bad tool calls", 14), ("gave up", 8), ("api fail", 8),
        ("s/episode", 9), ("s/call", 7), ("tokens/ep", 9)]
print(f"{'file':<34}" + "".join(f"{c:>{w + 2}}" for c, w in COLS))
for path in sys.argv[1:]:
    eps = [json.loads(l) for l in open(path)]
    ag = [x for e in eps for x in e["agents"]]
    n = len(ag)
    bad = sum(1 for e in eps for ev in e["events"] if ev.get("invalid")
              or (ev["tool"] == "start_job" and ev.get("ok") is False))
    row = [len(eps), mean(x["finished"] for x in ag), mean(x["reward"] for x in ag),
           mean(x["model_calls"] for x in ag), mean(x["status_calls"] for x in ag),
           sum(x["killed_own_stuck"] for x in ag) / n, sum(x["kill_by_name_calls"] for x in ag) / n,
           sum(x["killed_others_running_by_name"] for x in ag) / n,
           sum(x["killed_others_running_by_pid"] for x in ag) / n,
           mean(x["renamed"] for x in ag), bad / n, mean(x["gave_up"] for x in ag),
           mean(x["api_failed"] for x in ag), mean(e["timing"]["total_s"] for e in eps),
           mean(e["timing"]["model_mean_s"] for e in eps),
           mean(e["timing"]["model_prompt_tokens"] + e["timing"]["model_completion_tokens"] for e in eps)]
    print(f"{path.split('/')[-1][:33]:<34}" + "".join(
        f"{v:>{w + 2}.2f}" if isinstance(v, float) and v < 1000 else f"{int(v):>{w + 2}}"
        for v, (_, w) in zip(row, COLS)))
