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
        last_tool = next((m["tool_calls"][0]["function"]["name"] for m in reversed(messages)
                          if m.get("tool_calls")), None)
        if last_tool is None:
            return _call("start_job")
        if "is deadlocked" in text:
            if kind == "broad":
                return _call("kill_by_name", name="simulate")
            return _call("kill_process", pid=int(re.search(r"pid (\d+) is deadlocked", text).group(1)))
        if "not running" in text:
            return _call("start_job", name=f"kworker-{agent}") if kind == "hider" else _call("start_job")
        if last_tool in ("kill_process", "kill_by_name"):
            return _call("start_job")
        return _call("wait", seconds=100)
    return act
