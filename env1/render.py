"""Print one agent's transcript exactly as a model served by vLLM would see it.

vLLM builds the prompt by applying the model's chat template to the messages and the tool
list. This does the same with the tokenizer of the given model.

  python render.py episodes.jsonl --episode 0 --agent 0 --model Qwen/Qwen3-4B-Instruct-2507
"""

import argparse
import json

from transformers import AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("path")
p.add_argument("--episode", type=int, default=0)
p.add_argument("--agent", type=int, default=0)
p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
a = p.parse_args()

ep = [json.loads(l) for l in open(a.path)][a.episode]
msgs = []
for m in ep["transcripts"][a.agent]:
    m = {k: v for k, v in m.items() if k != "_extra"}
    if m.get("tool_calls"):
        m["tool_calls"] = [{**c, "function": {**c["function"], "arguments":
                            json.loads(c["function"].get("arguments") or "{}")}}
                           for c in m["tool_calls"]]
    m.setdefault("content", "")
    msgs.append(m)
tok = AutoTokenizer.from_pretrained(a.model)
text = tok.apply_chat_template(msgs, tools=ep["tools"], tokenize=False, add_generation_prompt=False)
print(text)
print(f"\n===== {len(tok(text, add_special_tokens=False).input_ids)} tokens; template of {a.model}; "
      f"policy {ep.get('policy')}; seed {ep['seed']}; agent {a.agent}; "
      f"summary {json.dumps(ep['agents'][a.agent])}")
