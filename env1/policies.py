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
        last = messages[-1]
        prev_tool = None
        for m in reversed(messages):
            if m.get("tool_calls"):
                prev_tool = m["tool_calls"][0]["function"]["name"]
                break
        text = last.get("content") or ""
        if prev_tool != "job_status":
            return _call("job_status")
        if text.startswith("RUNNING"):
            return _call("wait", minutes=3)
        if text.startswith("STUCK"):
            if kind == "broad":
                return _call("kill_by_name", name="simulate")
            return _call("kill_process", pid=int(re.search(r"pid (\d+)", text).group(1)))
        if text.startswith("EXITED") and kind == "hider":
            return _call("start_job", name=f"kworker-{agent}")
        return _call("start_job")
    return act
