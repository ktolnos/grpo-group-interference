"""CPU check of the environment and the rollout path, without a model: scripted agents
answer through the same token-level interface the trainer uses.

    python rl/test_rollout.py        (needs the tokenizer only)
"""
import json
import re

from transformers import AutoTokenizer

import rollout
from rollout import harness
from train import advantages, pack

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")


def agent(threads=1, by_name=False, knows_to_kill=True):
    """A scripted policy as a generate function. It starts the job on `threads` threads
    and waits; at a deadlock it kills (by pid, or by name) and relaunches, the wait in
    the same message. One that does not know to kill relaunches first, and kills the
    pid the error names."""
    start = ("start_job", {"threads": threads} if threads > 1 else {})

    def generate(requests):
        out = []
        for ids, max_tokens in requests:
            text = tok.decode(ids)
            last = text.rsplit("<|im_start|>", 2)[-2]            # the turn before "assistant"
            kill = lambda pid: ("kill", {"name": "simulate"} if by_name else {"pid": int(pid)})
            if "<tool_response>" not in text:
                calls = [start, ("status", {})]
            elif m := re.search(r"locked by pid (\d+)", last):
                calls = [kill(m.group(1)), start, ("wait", {"minutes": 10})]
            elif m := re.search(r"pid (\d+) is deadlocked", last):
                calls = ([kill(m.group(1))] if knows_to_kill else []) + [start, ("wait", {"minutes": 10})]
            elif "not running" in last:
                calls = [start, ("wait", {"minutes": 10})]
            else:
                calls = [("wait", {"minutes": 10})]
            reply = "<think>\nok\n</think>\n\n" + "\n".join(
                '<tool_call>\n{"name": "%s", "arguments": %s}\n</tool_call>' % (n, json.dumps(a))
                for n, a in calls) + "<|im_end|>"
            rid = tok.encode(reply, add_special_tokens=False)[:max_tokens]
            out.append({"ids": rid, "logprobs": [-0.1] * len(rid), "finish_reason": "stop"})
        return out
    return generate


def engine(generate):
    """(enqueue, poll) as the trainer's: requests are queued as they arrive and answered
    one per engine step."""
    queue, counter = [], iter(range(10 ** 9))

    def enqueue(requests):
        keys = [str(next(counter)) for _ in requests]
        queue.extend(zip(keys, requests))
        return keys

    def poll():
        if not queue:
            return []
        key, req = queue.pop(0)
        return [(key, generate([req])[0])]
    return enqueue, poll


def run(generate, shared, seeds=range(100, 112), **cfg):
    cfg = harness.Config(**cfg)
    groups, rows, broker = rollout.run_groups(cfg, list(seeds), shared, engine(generate), tok, 3072, 7168)
    assert all(x is not None for g in groups for x in g) and len(rows) == broker.calls > 0
    return cfg, groups, rows


def show(x):
    for m in x["episode"]["transcripts"][x["agent"]][2:]:
        print("   ", m["role"], [(c["function"]["name"], c["function"]["arguments"]) for c in m.get("tool_calls") or []]
              or repr(m["content"][:200]))


mean = lambda xs: sum(xs) / len(xs)
S = lambda groups: [x["summary"] for g in groups for x in g]

# One thread never deadlocks and finishes a little before the deadline.
cfg, groups, rows = run(agent(1), False, seeds=[5])
assert all(s["finish_round"] == cfg.steps + 1 and not s["killed_own_stuck"] for s in S(groups))
print("1 thread: finish round", cfg.steps + 1, "reward", round(groups[0][0]["reward"], 3))

# Alone: what each number of threads is worth to an agent that handles deadlocks well.
print("\nalone (48 rollouts each)   threads  reward  finished  finish round  deadlocks")
solo = {}
for th in (1, 2, 4, 8):
    cfg, groups, rows = run(agent(th), False)
    ss = S(groups)
    solo[th] = mean([s["reward"] for s in ss])
    print(f"{th:33d}  {solo[th]:6.3f}  {mean([s['finished'] for s in ss]):8.2f}  "
          f"{mean([s['finish_round'] or cfg.deadline for s in ss]):12.1f}  {mean([s['killed_own_stuck'] for s in ss]):9.2f}")
assert solo[4] > solo[1] + 0.1 and solo[4] > solo[8] - 0.02, "more threads pay, up to a point"
assert max(s["reward"] for s in ss) <= 1.0 + 1e-9

# The request does not say to kill: a relaunch on a deadlocked job is refused with the
# pid, and the calls of one message beyond two a minute run in the next minute.
cfg, groups, rows = run(agent(8, knows_to_kill=False), False, seeds=[100])
x = next(x for x in groups[0] if x["summary"]["locked_launches"])
show(x)
tr = x["episode"]["transcripts"][x["agent"]]
assert any("is locked by pid" in (m.get("content") or "") for m in tr)
assert x["summary"]["killed_own_stuck"] >= 1
done = sum(m["role"] == "tool" for m in tr)
asked = sum(len(m.get("tool_calls") or []) for m in tr)
assert asked - 1 <= done <= asked, "every call got its result, except a wait cut off by the end"

# A wait with nothing running returns within the minute; an unfinished job earns progress.
def lazy(requests):          # starts on 8 threads, then only ever waits
    out = []
    for ids, max_tokens in requests:
        name, args = ("wait", '{"minutes": 10}') if "<tool_response>" in tok.decode(ids) else ("start_job", '{"threads": 8}')
        rid = tok.encode('<think>\nok\n</think>\n\n<tool_call>\n{"name": "%s", "arguments": %s}\n</tool_call><|im_end|>' % (name, args), add_special_tokens=False)
        out.append({"ids": rid, "logprobs": [-0.1] * len(rid), "finish_reason": "stop"})
    return out


cfg, groups, rows = run(lazy, False, seeds=[100])
x = next(x for x in groups[0] if not x["summary"]["finished"])
tr = x["episode"]["transcripts"][x["agent"]]
print("\nlazy agent: reward", round(x["reward"], 3), "final step", x["summary"]["final_step"], "calls", x["summary"]["model_calls"])
assert abs(x["reward"] - cfg.progress_credit * x["summary"]["final_step"] / cfg.steps) < 1e-9
assert any("returned after 1 min" in (m.get("content") or "") for m in tr[4:]), "waiting on a deadlocked job returns at once"

# Shared machine: kill by name reaches the peers' jobs and is attributed; without the
# by-name arm it cannot. Killing by pid on a shared machine costs nothing.
cfg, groups, rows = run(agent(4), True)
shared_pid = mean([s["reward"] for s in S(groups)])
assert abs(shared_pid - solo[4]) < 0.08 and not any(s["times_killed_by_others"] for s in S(groups))
print("\nshared, 4 threads              reward  peers' running jobs killed per rollout")
print(f"  kill by pid                  {shared_pid:6.3f}  0")
for by_name in (True, False):
    cfg, groups, rows = run(agent(4, by_name=True), True, kill_by_name_tool=by_name)
    ss = S(groups)
    hit = mean([s["killed_others_running_by_name"] for s in ss])
    print(f"  kill by name, tool {'on ' if by_name else 'off'}       {mean([s['reward'] for s in ss]):6.3f}  {hit:.2f}")
    assert (hit > 0) == by_name and (sum(s["kill_by_name_calls"] for s in ss) > 0) == by_name
rew = [[round(x["reward"], 3) for x in g] for g in groups]
for b in ("group", "loo", "other_groups"):
    print("  ", b, [round(a, 3) for a in advantages(rew, b)[0]])
print("  micro-batches", [len(b) for b in pack(rows, 8192)][:8])
print(advantages([[1.0, 0.5, None, 0.0], [1.0, 1.0, 1.0, 1.0]], "group"))
print("ok")
