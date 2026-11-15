"""Environment 1 rollouts for training: every agent's model call, in every group, goes
through one batched generation call, and each call becomes one training row.

A rollout is one agent's episode: several model calls, each with its own context. The
context of a call is the conversation so far *without* the thinking of earlier calls,
which is how the evaluation served the model (the OpenAI-style API does not send
reasoning back) and keeps a row near 2.7k tokens instead of one 10k-20k-token sequence.
So a rollout is not one token sequence, and the unit of the loss is the call: a row is
(prompt ids, completion ids, vLLM log-probs), and every row of a rollout carries that
rollout's advantage.

Episodes run in threads and block inside `Broker.submit`; the main thread generates once
every live episode is waiting, so one `generate` call serves all of them.
"""

import json
import os
import re
import sys
import threading
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "env1"))
import harness  # noqa: E402

TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def parse_reply(text):
    """Qwen3's reply format -> an OpenAI-style assistant message.

    `<think>...</think>` is the reasoning, `<tool_call>{json}</tool_call>` blocks are the
    tool calls (the hermes format vLLM's server parses), the rest is the text. A call
    whose JSON does not parse is left in the text, as the server does.
    """
    for end in ("<|im_end|>", "<|endoftext|>"):
        text = text.replace(end, "")
    reasoning = None
    if "</think>" in text:
        reasoning, text = text.split("</think>", 1)
        reasoning = reasoning.replace("<think>", "").strip()
    elif "<think>" in text:                       # the cap ran out inside the thinking
        reasoning, text = text.replace("<think>", "").strip(), ""
    calls = []
    for m in TOOL_CALL.finditer(text):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
            continue
        args = obj.get("arguments", {})
        calls.append({"id": f"call_{len(calls)}", "type": "function", "function": {
            "name": obj["name"],
            "arguments": args if isinstance(args, str) else json.dumps(args)}})
        text = text.replace(m.group(0), "")
    msg = {"role": "assistant", "content": text.strip() or None}
    if calls:
        msg["tool_calls"] = calls
    if reasoning:
        msg["reasoning"] = reasoning
    return msg


class Broker:
    """Collects the model calls of all running episodes and serves them in batches.

    An episode announces how many calls it is about to make (`expect`), its agent threads
    block in `submit`, and `serve` generates as soon as no episode is doing anything else.
    """

    def __init__(self, episodes, generate):
        self.cv = threading.Condition()
        self.alive = episodes           # episodes still running
        self.busy = episodes            # episodes neither waiting for replies nor finished
        self.expected = 0
        self.pending = []
        self.generate = generate        # list of (prompt ids, max tokens) -> list of results
        self.lock = threading.Lock()    # a fast tokenizer must not be used by two threads at once
        self.batches = []               # (calls, seconds) per generation call

    def expect(self, n):
        with self.cv:
            self.busy -= 1
            self.expected += n
            self.cv.notify_all()

    def cancel(self):                   # a call that was announced but will not be made
        with self.cv:
            self.expected -= 1
            self.cv.notify_all()

    def finished(self):
        with self.cv:
            self.alive -= 1
            self.busy -= 1
            self.cv.notify_all()

    def submit(self, episode, ids, max_tokens):
        slot = {"episode": episode, "ids": ids, "max_tokens": max_tokens}
        with self.cv:
            self.pending.append(slot)
            self.cv.notify_all()
            self.cv.wait_for(lambda: "out" in slot)
        return slot["out"]

    def serve(self):
        while True:
            with self.cv:
                self.cv.wait_for(lambda: self.busy == 0 and len(self.pending) == self.expected)
                if not self.pending:
                    return
                batch, self.pending, self.expected = self.pending, [], 0
                # Counted as busy before the replies are handed out, so the next check
                # cannot run between an episode receiving them and announcing its next batch.
                self.busy += len({s["episode"] for s in batch})
            t0 = time.perf_counter()
            try:
                outs = self.generate([(s["ids"], s["max_tokens"]) for s in batch])
            except BaseException:
                outs = [None] * len(batch)
                with self.cv:
                    for s, o in zip(batch, outs):
                        s["out"] = o
                    self.cv.notify_all()
                raise
            self.batches.append((len(batch), time.perf_counter() - t0))
            with self.cv:
                for s, o in zip(batch, outs):
                    s["out"] = o
                self.cv.notify_all()


    def serve_stream(self, submit, poll):
        """Feed the engine as calls arrive instead of in rounds.

        submit([(prompt ids, max tokens)]) queues requests and returns one key each;
        poll() advances the engine by one step and returns [(key, result)] for requests
        that finished. A call is answered as soon as it is done, so an episode waits for
        the longest of its own agents' calls, not for the longest call of every episode.
        """
        inflight, calls, t_busy = {}, 0, 0.0
        while True:
            with self.cv:
                if not self.pending and not inflight:
                    if self.alive == 0:
                        break
                    self.cv.wait(timeout=0.05)      # episodes are busy in the environment
                    continue
                batch, self.pending = self.pending, []
            t0 = time.perf_counter()
            try:
                if batch:
                    keys = submit([(s["ids"], s["max_tokens"]) for s in batch])
                    inflight.update(zip(keys, batch))
                    calls += len(batch)
                done = poll() if inflight else []
            except BaseException:
                with self.cv:
                    for s in list(inflight.values()) + batch:
                        s.setdefault("out", None)
                    self.cv.notify_all()
                raise
            t_busy += time.perf_counter() - t0
            if done:
                with self.cv:
                    for key, out in done:
                        inflight.pop(key)["out"] = out
                    self.cv.notify_all()
        self.batches.append((calls, t_busy))


class TokenPolicy:
    """The harness's policy for one episode: renders the context, asks the broker for a
    completion, parses it, and records the call as a training row."""

    def __init__(self, episode, broker, tokenizer, cap, window, rows):
        self.episode, self.broker, self.tok = episode, broker, tokenizer
        self.cap, self.window, self.rows = cap, window, rows

    def expect(self, n):
        self.broker.expect(n)

    def __call__(self, agent, messages, tools):
        try:
            clean = [{k: v for k, v in m.items() if k != "_extra"} for m in messages]
            with self.broker.lock:
                ids = list(self.tok.apply_chat_template(
                    clean, tools=tools, add_generation_prompt=True, tokenize=True,
                    return_dict=False))
            room = self.window - len(ids)
        except Exception:
            traceback.print_exc()
            room = 0
        if room < 16:                   # no room left in the window: the agent's turn ends
            self.broker.cancel()
            return None
        max_tokens = min(self.cap, room)
        out = self.broker.submit(self.episode, ids, max_tokens)
        if out is None:
            return None
        with self.broker.lock:
            text = self.tok.decode(out["ids"], skip_special_tokens=False)
        msg = parse_reply(text)
        capped = out["finish_reason"] == "length"
        msg["_meta"] = {"prompt_tokens": len(ids), "completion_tokens": len(out["ids"]),
                        "finish_reason": out["finish_reason"],
                        "hit_cap": capped and max_tokens == self.cap,
                        "hit_window": capped and max_tokens < self.cap}
        self.rows.append({"episode": self.episode, "agent": agent, "prompt_ids": ids,
                          "completion_ids": list(out["ids"]), "logprobs": list(out["logprobs"]),
                          **msg["_meta"]})
        return msg


def run_groups(cfg, seeds, shared, generate, tokenizer, cap, window, workdir=None, stream=None):
    """One group of cfg.G rollouts per seed. Returns (groups, rows, broker).

    shared=True: the G agents of a group work on one machine (one Environment 1 episode).
    shared=False: each agent gets a machine to itself, with the same seed and so the same
    request and deadlock plan as agent 0; nothing an agent does can reach another.

    groups[g] is a list of G dicts {"reward", "summary", "episode"} (None for a rollout
    whose episode failed); rows are the model calls, each tagged with (group, member).
    """
    import dataclasses
    specs = []                          # (group, members, episode config)
    for g, seed in enumerate(seeds):
        if shared:
            specs.append((g, list(range(cfg.G)), cfg, seed))
        else:
            solo = dataclasses.replace(cfg, G=1)
            specs.extend((g, [m], solo, seed) for m in range(cfg.G))
    broker = Broker(len(specs), generate)
    rows, results = [], [None] * len(specs)

    def work(i):
        _, _, ecfg, seed = specs[i]
        try:
            results[i] = harness.run_episode(
                ecfg, TokenPolicy(i, broker, tokenizer, cap, window, rows), seed, workdir=workdir)
        except Exception:
            traceback.print_exc()
        finally:
            broker.finished()

    threads = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(len(specs))]
    for t in threads:
        t.start()
    if stream:                      # (submit, poll): see Broker.serve_stream
        broker.serve_stream(*stream)
    else:
        broker.serve()
    for t in threads:
        t.join()

    groups = [[None] * cfg.G for _ in seeds]
    for i, (g, members, _, _) in enumerate(specs):
        ep = results[i]
        if ep is None:
            continue
        for a, m in enumerate(members):
            groups[g][m] = {"reward": ep["agents"][a]["reward"], "summary": ep["agents"][a],
                            "episode": ep, "agent": a}
    for r in rows:
        g, members, _, _ = specs[r["episode"]]
        r["group"], r["member"] = g, members[r["agent"]]
    rows = [r for r in rows if groups[r["group"]][r["member"]] is not None]
    return groups, rows, broker
