"""Minimal OpenAI-compatible chat client with tool calling, pacing and retries.

Defaults point at Vector Inference. No caching: agent rollouts are sampled.
"""

import collections
import json
import os
import threading
import time
import urllib.error
import urllib.request


class RateLimiter:
    """At most `rpm` calls in any 60-second window, with no spacing between calls.

    Calls made at the same moment go out together; a call waits only when the window is
    full. (Spacing calls evenly, 60/rpm seconds apart, would delay every simultaneous call
    of a group after the first, and the proxy accepts bursts of at least 32.)
    """

    def __init__(self, rpm, window=60.0):
        self.rpm, self.window = int(rpm), window
        self.lock, self.sent = threading.Lock(), collections.deque()

    def wait(self):
        """Blocks until this call may go; returns the seconds it waited."""
        t0 = time.monotonic()
        while True:
            with self.lock:
                now = time.monotonic()
                while self.sent and now - self.sent[0] >= self.window:
                    self.sent.popleft()
                if len(self.sent) < self.rpm:
                    self.sent.append(now)
                    return now - t0
                delay = self.window - (now - self.sent[0])
            time.sleep(max(delay, 0.01))


class Client:
    def __init__(self, model, base_url=None, key_var="VECTOR_INFERENCE_API_KEY", rpm=100.0,
                 max_retries=3, timeout=120, **params):
        self.model = model
        self.url = (base_url or os.environ.get("VECTOR_BASE_URL",
                                               "https://proxy.vectorinstitute.ai/v1")) + "/chat/completions"
        key = os.environ.get(key_var) if key_var else "EMPTY"      # a local server needs no key
        if not key:
            raise SystemExit(f"{key_var} is not set")
        self.headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        self.limiter, self.max_retries, self.timeout = RateLimiter(rpm), max_retries, timeout
        self.params = params
        self.errors = {}
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
        self.lock = threading.Lock()

    def chat(self, messages, tools):
        """Returns the assistant message dict, or None if every retry failed.

        The message carries "_meta": attempts, seconds spent in our rate limiter, in HTTP
        requests and in back-off sleeps, token counts and the finish reason.
        """
        body = {"model": self.model, "messages": messages, "tools": tools, **self.params}
        meta = {"attempts": 0, "limiter_s": 0.0, "http_s": 0.0, "backoff_s": 0.0, "errors": []}
        self.last_meta = threading.local() if not hasattr(self, "last_meta") else self.last_meta
        self.last_meta.value = meta

        def backoff(attempt):
            d = min(2 ** attempt, 60)
            time.sleep(d)
            meta["backoff_s"] += d

        for attempt in range(self.max_retries):
            meta["limiter_s"] += self.limiter.wait()
            meta["attempts"] += 1
            t0 = time.perf_counter()
            try:
                req = urllib.request.Request(self.url, json.dumps(body).encode(), self.headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    out = json.loads(r.read())
            except urllib.error.HTTPError as e:
                meta["http_s"] += time.perf_counter() - t0
                meta["errors"].append(f"HTTP {e.code}")
                err = f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}"
                with self.lock:
                    self.errors[err[:120]] = self.errors.get(err[:120], 0) + 1
                if e.code in (400, 401, 403, 404):
                    return None
                backoff(attempt)
                continue
            except Exception as e:  # timeouts, connection resets
                meta["http_s"] += time.perf_counter() - t0
                meta["errors"].append(type(e).__name__)
                with self.lock:
                    self.errors[type(e).__name__] = self.errors.get(type(e).__name__, 0) + 1
                backoff(attempt)
                continue
            meta["http_s"] += time.perf_counter() - t0
            msg = out["choices"][0]["message"]
            u = out.get("usage") or {}
            meta.update(prompt_tokens=u.get("prompt_tokens"),
                        completion_tokens=u.get("completion_tokens"),
                        finish_reason=out["choices"][0].get("finish_reason"))
            with self.lock:
                self.usage["calls"] += 1
                for k in ("prompt_tokens", "completion_tokens"):
                    self.usage[k] += u.get(k) or 0
            if msg.get("tool_calls") or msg.get("content"):
                msg["_meta"] = {k: round(v, 3) if isinstance(v, float) else v
                                for k, v in meta.items()}
                return msg
            meta["errors"].append("empty reply")
            # An empty reply means max_tokens ran out inside the reasoning block. The cap is
            # set high from the start (it costs nothing unless used), so this is retried as
            # it is and shows up in the call's "errors".
        return None
