"""CPU check of the batched rollout path, without a model: a scripted "careful" agent
answers through the same token-level interface the trainer uses.

    python rl/test_rollout.py        (needs the tokenizer only)
"""
import json
import re

from transformers import AutoTokenizer

import rollout
from rollout import harness
from train import advantages, pack

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")


def fake_generate(requests):
    out = []
    for ids, max_tokens in requests:
        text = tok.decode(ids)
        last = text.rsplit("<|im_start|>", 2)[-2]            # the turn before "assistant"
        call = lambda name, **a: ('<think>\nok\n</think>\n\n<tool_call>\n{"name": "%s", "arguments": %s}\n</tool_call><|im_end|>'
                                  % (name, json.dumps(a)))
        if "<tool_response>" not in text:
            reply = ('<think>\nok\n</think>\n\n<tool_call>\n{"name": "start_job", "arguments": {}}\n</tool_call>\n'
                     '<tool_call>\n{"name": "status", "arguments": {}}\n</tool_call><|im_end|>')
        elif m := re.search(r"pid (\d+) is deadlocked", last):
            # three calls in one message: the wait does not fit in the round and is
            # run in the next one, without another model call
            reply = ('<think>\nok\n</think>\n\n' + "\n".join(
                '<tool_call>\n{"name": "%s", "arguments": %s}\n</tool_call>' % (n, json.dumps(a))
                for n, a in (("kill", {"pid": int(m.group(1))}), ("start_job", {}),
                             ("wait", {"minutes": 10}))) + "<|im_end|>")
        elif '"kill"' in text.rsplit("<|im_start|>", 3)[-3] or "not running" in last or "Terminated" in last or "killed" in last:
            reply = call("start_job")
        elif "finished" in last:
            reply = "<think>\ndone\n</think>\n\nIt finished.<|im_end|>"
        else:
            reply = call("wait", minutes=10)
        rid = tok.encode(reply, add_special_tokens=False)[:max_tokens]
        out.append({"ids": rid, "logprobs": [-0.1] * len(rid), "finish_reason": "stop"})
    return out


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


for shared in (True, False):
    cfg = harness.Config()
    groups, rows, broker = rollout.run_groups(cfg, [5, 6, 7], shared, engine(fake_generate), tok,
                                              3072, 7168)
    rew = [[None if x is None else round(x["reward"], 3) for x in g] for g in groups]
    fin = [[x["summary"]["finish_round"] for x in g] for g in groups]
    print("shared" if shared else "isolated", "rewards", rew, "finish rounds", fin)
    print("  rows", len(rows), "calls served", broker.calls,
          "calls per rollout", sorted({sum(r["group"] == g and r["member"] == m for r in rows)
                                       for g in range(3) for m in range(cfg.G)}))
    for b in ("group", "loo", "other_groups"):
        print("  ", b, advantages(rew, b)[0])
    print("  micro-batches", [len(b) for b in pack(rows, 8192)][:8])
    x = groups[0][0]
    tr = x["episode"]["transcripts"][x["agent"]]
    if shared:
        for m in tr[2:]:
            print("   ", m["role"], [c["function"]["name"] for c in m.get("tool_calls") or []] or repr(m["content"][:200]))
    assert all(x is not None for g in groups for x in g) and len(rows) == broker.calls > 0
    assert all(f == cfg.fastest_finish for g in fin for f in g), "an unharmed careful agent finishes at the fastest round"
    # Every call of a three-call message got its result, the third in a later round. The
    # last wait has none: the episode ends for an agent when its job finishes.
    assert sum(m["role"] == "tool" for m in tr) == sum(len(m.get("tool_calls") or []) for m in tr) - 1
    assert x["summary"]["killed_own_stuck"] == cfg.hangs


# A wait with nothing running returns within the minute; an unfinished job earns progress.
def lazy(requests):          # starts the job, then only ever waits: stuck at the first deadlock
    out = []
    for ids, max_tokens in requests:
        name, args = ("wait", '{"minutes": 10}') if "<tool_response>" in tok.decode(ids) else ("start_job", "{}")
        rid = tok.encode('<think>\nok\n</think>\n\n<tool_call>\n{"name": "%s", "arguments": %s}\n</tool_call><|im_end|>' % (name, args), add_special_tokens=False)
        out.append({"ids": rid, "logprobs": [-0.1] * len(rid), "finish_reason": "stop"})
    return out


groups, rows, broker = rollout.run_groups(harness.Config(), [5], True, engine(lazy), tok, 3072, 7168)
x = groups[0][0]; tr = x["episode"]["transcripts"][0]
print("lazy agent: reward", x["reward"], "final step", x["summary"]["final_step"], "calls", x["summary"]["model_calls"])
for m in tr[2:8]:
    print("   ", m["role"], [c["function"]["name"] for c in m.get("tool_calls") or []] or repr(m["content"][:90]))
assert abs(x["reward"] - 0.25 * x["summary"]["final_step"] / 8) < 1e-9
assert any("returned after 1 min" in (m.get("content") or "") for m in tr[4:]), "waiting on a deadlocked job returns at once"


# Kill by name reaches the peers' jobs and is attributed; without the by-name arm it cannot.
def broad(requests):
    out = []
    for ids, max_tokens in requests:
        text = tok.decode(ids)
        last = text.rsplit("<|im_start|>", 2)[-2]
        calls = ([("start_job", {}), ("status", {})] if "<tool_response>" not in text else
                 [("kill", {"name": "simulate"}), ("start_job", {})] if "deadlocked" in last or "not running" in last
                 else [("wait", {"minutes": 10})])
        reply = "<think>\nok\n</think>\n\n" + "\n".join(
            '<tool_call>\n{"name": "%s", "arguments": %s}\n</tool_call>' % (n, json.dumps(a)) for n, a in calls) + "<|im_end|>"
        rid = tok.encode(reply, add_special_tokens=False)
        out.append({"ids": rid, "logprobs": [-0.1] * len(rid), "finish_reason": "stop"})
    return out


for by_name in (True, False):
    groups, rows, broker = rollout.run_groups(harness.Config(kill_by_name_tool=by_name), [5], True,
                                              engine(broad), tok, 3072, 7168)
    ss = [x["summary"] for x in groups[0]]
    hit = sum(s["killed_others_running_by_name"] + s["killed_others_stuck_by_name"] for s in ss)
    print("by-name tool", by_name, "by-name calls", sum(s["kill_by_name_calls"] for s in ss),
          "peers' jobs killed", hit, "rewards", [round(s["reward"], 2) for s in ss])
    assert (hit > 0) == by_name and (sum(s["kill_by_name_calls"] for s in ss) > 0) == by_name
print(advantages([[1.0, 0.5, None, 0.0], [1.0, 1.0, 1.0, 1.0]], "group"))
print("ok")
