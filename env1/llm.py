"""Minimal OpenAI-compatible chat client with tool calling, pacing and retries.

Defaults point at Vector Inference. No caching: agent rollouts are sampled.
"""

import json
import os
import threading
import time
import urllib.error
import urllib.request


class RateLimiter:
    def __init__(self, rpm):
        self.interval, self.lock, self.next_at = 60.0 / rpm, threading.Lock(), 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            at = max(now, self.next_at)
            self.next_at = at + self.interval
        if at > now:
            time.sleep(at - now)


class Client:
    def __init__(self, model, base_url=None, key_var="VECTOR_INFERENCE_API_KEY", rpm=100.0,
                 max_retries=6, timeout=300, **params):
        self.model = model
        self.url = (base_url or os.environ.get("VECTOR_BASE_URL",
                                               "https://proxy.vectorinstitute.ai/v1")) + "/chat/completions"
        key = os.environ.get(key_var)
        if not key:
            raise SystemExit(f"{key_var} is not set")
        self.headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        self.limiter, self.max_retries, self.timeout = RateLimiter(rpm), max_retries, timeout
        self.params = params
        self.errors = {}
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
        self.lock = threading.Lock()

    def chat(self, messages, tools):
        """Returns the assistant message dict, or None if every retry failed."""
        body = {"model": self.model, "messages": messages, "tools": tools, **self.params}
        for attempt in range(self.max_retries):
            self.limiter.wait()
            try:
                req = urllib.request.Request(self.url, json.dumps(body).encode(), self.headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    out = json.loads(r.read())
            except urllib.error.HTTPError as e:
                err = f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}"
                with self.lock:
                    self.errors[err[:120]] = self.errors.get(err[:120], 0) + 1
                if e.code in (400, 401, 403, 404):
                    return None
                time.sleep(min(2 ** attempt, 60))
                continue
            except Exception as e:  # timeouts, connection resets
                with self.lock:
                    self.errors[type(e).__name__] = self.errors.get(type(e).__name__, 0) + 1
                time.sleep(min(2 ** attempt, 60))
                continue
            msg = out["choices"][0]["message"]
            u = out.get("usage") or {}
            with self.lock:
                self.usage["calls"] += 1
                for k in ("prompt_tokens", "completion_tokens"):
                    self.usage[k] += u.get(k) or 0
            if msg.get("tool_calls") or msg.get("content"):
                return msg
            # Reasoning models return nothing when max_tokens runs out inside the
            # reasoning block: raise the budget and try again.
            if out["choices"][0].get("finish_reason") == "length" and "max_tokens" in body:
                body["max_tokens"] = min(body["max_tokens"] * 2, 16384)
        return None
