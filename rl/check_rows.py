"""Rebuild the training rows of logged rollouts from their transcripts and check what is
in the prompt (no loss) and what is in the completion (loss).

    python rl/check_rows.py <run>/rollouts.jsonl [step ...]
"""
import json
import sys

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
recs = [json.loads(l) for l in open(sys.argv[1])]
steps = [int(s) for s in sys.argv[2:]] or [0]
shown = False
tot = dict(calls=0, prompt_match=0, think_in_prompt=0, results_in_completion=0,
           calls_with_tool_call=0, tool_call_in_completion=0, completion_match=0,
           capped=0, prompt_tokens=0, completion_tokens=0, tool_result_tokens=0)
for rec in recs:
    if rec["step"] not in steps:
        continue
    logged = {}
    for c in rec["calls"]:
        logged.setdefault((c["group"], c["member"]), []).append(c)
    for g, ep in enumerate(rec["episodes"]):
        for m, tr in enumerate(ep["transcripts"]):
            k = 0
            for i, msg in enumerate(tr):
                if msg["role"] != "assistant":
                    continue
                c = logged[(g, m)][k]
                k += 1
                ctx = [{a: b for a, b in x.items() if a != "_extra"} for x in tr[:i]]
                prompt = tok.apply_chat_template(ctx, tools=ep["tools"],
                                                 add_generation_prompt=True, tokenize=False)
                n_prompt = len(tok(prompt, add_special_tokens=False)["input_ids"])
                think = (msg.get("_extra") or {}).get("reasoning") or ""
                # the completion as the model wrote it: thinking, text, tool calls
                body = tok.apply_chat_template(
                    ctx + [{a: b for a, b in msg.items() if a != "_extra"}], tools=ep["tools"],
                    tokenize=False)[len(prompt) - len("<|im_start|>assistant\n"):]
                body = body[len("<|im_start|>assistant\n"):]
                n_comp = len(tok("<think>\n" + think + "\n</think>\n\n" + body,
                                 add_special_tokens=False)["input_ids"])
                tot["calls"] += 1
                tot["prompt_match"] += n_prompt == c["prompt_tokens"]
                tot["think_in_prompt"] += "<think>" in prompt
                tot["results_in_completion"] += "<tool_response>" in body
                has_call = bool(msg.get("tool_calls"))
                tot["calls_with_tool_call"] += has_call
                tot["tool_call_in_completion"] += has_call and "<tool_call>" in body
                tot["capped"] += c["hit_cap"]
                tot["completion_match"] += abs(n_comp - c["completion_tokens"]) <= 8
                tot["prompt_tokens"] += c["prompt_tokens"]
                tot["completion_tokens"] += c["completion_tokens"]
                tot["tool_result_tokens"] += sum(
                    len(tok(x["content"], add_special_tokens=False)["input_ids"])
                    for x in tr[:i] if x["role"] == "tool")
                if not shown and k == 3:
                    shown = True
                    print("=== PROMPT of a third call (not in the loss), from the user turn on ===")
                    print(prompt[prompt.index("<|im_start|>user"):])
                    print("=== COMPLETION (in the loss; thinking shortened) ===")
                    print("<think>\n" + think[:200] + " [...] " + think[-100:] + "\n</think>\n\n" + body)
                    print("=== logged prompt/completion tokens", c["prompt_tokens"],
                          c["completion_tokens"], "rebuilt", n_prompt, n_comp)
            assert k == len(logged[(g, m)]), "every logged call has an assistant message"
print(json.dumps(tot, indent=1))
