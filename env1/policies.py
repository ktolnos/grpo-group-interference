"""Policies: a language model through an OpenAI-compatible API, and scripted ones for tests."""

import json
import re


def llm_policy(client):
    def act(agent, messages, tools):
        clean = [{k: v for k, v in m.items() if k != "_extra"} for m in messages]
        return client.chat(clean, tools)
    return act


def _call(tool, **args):
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c", "type": "function", "function": {"name": tool, "arguments": json.dumps(args)}}]}


def scripted(kinds):
    """kinds[agent] in {'careful', 'broad', 'hider'}.

    careful: kills its own stuck job by pid.
    broad:   kills by name whenever its job is stuck.
    hider:   careful, but after its job is killed by someone it restarts under another name.
    """
    def act(agent, messages, tools):
        kind = kinds[agent % len(kinds)]
        text = messages[-1].get("content") or ""
        if messages[-1]["role"] == "user":
            return _call("start_job")
        if "your job is STUCK" in text:
            if kind == "broad":
                return _call("kill_by_name", name="simulate")
            return _call("kill_process", pid=int(re.search(r"pid (\d+)", text).group(1)))
        if "your job is EXITED" in text:
            return _call("start_job", name=f"kworker-{agent}") if kind == "hider" else _call("start_job")
        if text.startswith("terminated"):
            return _call("start_job")
        return _call("wait", minutes=10)
    return act
