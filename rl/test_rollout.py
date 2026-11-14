"""CPU check of the batched rollout path, without a model: a scripted "careful" agent
answers through the same token-level interface the trainer uses.

    python rl/test_rollout.py        (needs the tokenizer only)
"""
import re
import sys

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
                                  % (name, __import__("json").dumps(a)))
        if "<tool_response>" not in text:
            reply = call("start_job")
        elif m := re.search(r"pid (\d+) is deadlocked", last):
            # three calls in one message: the wait does not fit in the round and is
            # run in the next one, without another model call
            reply = ('<think>\nok\n</think>\n\n' + "\n".join(
                '<tool_call>\n{"name": "%s", "arguments": %s}\n</tool_call>' % (n, __import__("json").dumps(a))
                for n, a in (("kill_process", {"pid": int(m.group(1))}), ("start_job", {}),
                             ("wait", {"minutes": 10}))) + "<|im_end|>")
        elif "kill_process" in text.rsplit("<|im_start|>", 3)[-3] or "not running" in last or "Terminated" in last or "killed" in last:
            reply = call("start_job")
        elif "finished" in last:
            reply = "<think>\ndone\n</think>\n\nIt finished.<|im_end|>"
        else:
            reply = call("wait", minutes=10)
        rid = tok.encode(reply, add_special_tokens=False)[:max_tokens]
        out.append({"ids": rid, "logprobs": [-0.1] * len(rid), "finish_reason": "stop"})
    return out


for shared in (True, False):
    cfg = harness.Config()
    groups, rows, broker = rollout.run_groups(cfg, [5, 6, 7], shared, fake_generate, tok,
                                              3072, 7168)
    rew = [[None if x is None else round(x["reward"], 3) for x in g] for g in groups]
    fin = [[x["summary"]["finish_round"] for x in g] for g in groups]
    print("shared" if shared else "isolated", "rewards", rew, "finish rounds", fin)
    print("  rows", len(rows), "generation batches", len(broker.batches),
          "largest", max(c for c, _ in broker.batches),
          "calls per rollout", sorted({sum(r["group"] == g and r["member"] == m for r in rows)
                                       for g in range(3) for m in range(cfg.G)}))
    for b in ("group", "loo", "other_groups"):
        print("  ", b, advantages(rew, b)[0])
    print("  micro-batches", [len(b) for b in pack(rows, 8192)][:8])
    tr = groups[0][0]["episode"]["transcripts"][groups[0][0]["agent"]]
    assert not any("not run" in (m.get("content") or "") for m in tr), "no call is refused"
    if shared:
        for m in tr[2:]:
            print("   ", m["role"], [c["function"]["name"] for c in m.get("tool_calls") or []] or repr(m["content"][:70]))
    assert all(x is not None for g in groups for x in g) and len(rows) > 0
    assert all(f == cfg.fastest_finish for g in fin for f in g), "an unharmed careful agent finishes at the fastest round"
print(advantages([[1.0, 0.5, None, 0.0], [1.0, 1.0, 1.0, 1.0]], "group"))
print("ok")
