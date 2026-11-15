"""`simulate`: the job the agents must run to completion; a stand-in for a flaky solver.

Two modes:

  --run     the solver itself. A real process with a real pid and a settable process name.
            It does no work of its own: it advances one step each time the group server
            writes "tick" to its stdin, so that an episode's time is counted in rounds.
            - takes a lock on the run directory and holds it while alive, so a second
              launch in the same directory exits at once (exit code 3);
            - resumes from checkpoint.json, and spends its first round loading;
            - runs on --threads threads: more threads do more steps per round
              (steps_per_round) and deadlock at more steps (deadlocks);
            - when it deadlocks it stays alive but stops advancing, and must be killed and
              launched again. The plan file lists steps in the order they become deadlock
              steps as threads are added: a run on t threads deadlocks, once each, at the
              first deadlocks(t) of them. So the number of deadlocks is fixed by the
              threads, and only where they fall differs between jobs;
            - writes result.json and exits 0 at the last step;
            - keeps state.json (pid, state, step) up to date for --status.
  --status  prints the state of the run in the current directory, from state.json, the
            process table and exit.json (written by the launcher when a run dies).
"""

import argparse
import ctypes
import fcntl
import json
import os
import sys

SIGNAME = {15: "SIGTERM", 9: "SIGKILL", 1: "SIGHUP", 2: "SIGINT"}
MAX_THREADS = 8


def steps_per_round(threads):
    """1, 2, 3, 4 steps a minute on 1, 2, 4, 8 threads."""
    return threads.bit_length()


def deadlocks(threads):
    """Deadlocks in a whole job run on this many threads: 0, 1, 2, 6 on 1, 2, 4, 8."""
    return [0, 1, 2, 2, 3, 4, 5, 6][threads - 1]


def describe(st, steps):
    """One line, as `simulate --status` prints it. st: state, step, and pid / signal / code."""
    of = f"step {st['step']}/{steps}"
    s = st["state"]
    th = f", {st['threads']} thread{'s' if st.get('threads', 1) > 1 else ''}" if st.get("threads") else ""
    if s == "LOADING":                # the first minute after a start: nothing computed yet
        return (f"simulate: starting, pid {st['pid']}, loading checkpoint at {of}" if st["step"]
                else f"simulate: starting, pid {st['pid']}")
    if s == "RUNNING":
        return f"simulate: running, pid {st['pid']}, {of}{th}"
    if s == "STUCK":
        return f"simulate: pid {st['pid']} is deadlocked at {of} (alive, no progress)"
    if s == "EXITED":
        sig = st.get("signal")
        why = (f"killed by signal {sig} ({SIGNAME.get(sig, '?')})" if sig
               else f"exited with code {st.get('code')}")
        return f"simulate: not running; last run was {why}; checkpoint at {of}"
    if s == "FINISHED":
        return f"simulate: finished, {steps}/{steps} steps, result.json written"
    return "simulate: no run in this directory yet"


def load(path, default=None):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return default


def clock():
    """Episode time in minutes, kept by the group server."""
    try:
        return int(open(os.environ["SIM_CLOCK"]).read())
    except (KeyError, OSError, ValueError):
        return 0


def read_status():
    st = load("state.json")
    if st is None:
        return {"state": "NOT_STARTED", "step": 0}
    if st["state"] == "FINISHED":
        return st
    ex = load("exit.json")
    if ex and ex["pid"] == st["pid"] or not os.path.exists(f"/proc/{st['pid']}"):
        ck = load("checkpoint.json", {"step": 0})
        ex = ex if ex and ex["pid"] == st["pid"] else {}
        return {"state": "EXITED", "step": ck["step"], "signal": ex.get("signal"),
                "code": ex.get("code")}
    return st


def status(a):
    print(describe(read_status(), a.steps))
    first = load("first_launch.json")
    if first is not None:
        print(f"simulate: {clock() - first} min since first launch")


def run(a):
    # Process name as shown in /proc/<pid>/comm (what `ps` and `pkill` match), max 15 chars.
    ctypes.CDLL(None).prctl(15, a.name.encode()[:15], 0, 0, 0)

    def say(state, step, **kw):
        sys.stdout.write(json.dumps({"state": state, "step": step, **kw}) + "\n")
        sys.stdout.flush()

    lock = open("job.lock", "a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock.seek(0)
        say("LOCKED", -1, holder=lock.read().strip())
        sys.exit(3)
    lock.seek(0)
    lock.truncate()
    lock.write(str(os.getpid()))
    lock.flush()

    ck = load("checkpoint.json", {"step": 0, "hangs_done": []})
    plan = set(json.load(open(a.plan))[:deadlocks(a.threads)])
    step, hung, loading = ck["step"], False, True
    if not os.path.exists("first_launch.json"):
        json.dump(clock(), open("first_launch.json", "w"))

    def put(path, obj):
        json.dump(obj, open(path + ".tmp", "w"))
        os.replace(path + ".tmp", path)

    def report(state):
        put("state.json", {"pid": os.getpid(), "state": state, "step": step, "threads": a.threads})
        say(state, step)

    def save(s):
        ck["step"] = s
        put("checkpoint.json", ck)

    report("LOADING")
    for line in sys.stdin:
        if line.strip() != "tick":
            continue
        if loading:                       # the first round after any launch makes no progress
            loading = False
            report("RUNNING")
            continue
        if hung:
            report("STUCK")
            continue
        for _ in range(steps_per_round(a.threads)):
            step += 1
            if step in plan and step not in ck["hangs_done"]:
                ck["hangs_done"].append(step)
                save(ck["step"])          # remember the deadlock was hit; the step is not saved
                hung = True
                break
            if step >= a.steps:
                save(step)
                put("result.json", {"done": True, "steps": step})
                report("FINISHED")
                sys.exit(0)
            if step % a.ckpt_every == 0:
                save(step)
        report("STUCK" if hung else "RUNNING")


if __name__ == "__main__":
    p = argparse.ArgumentParser(prog="simulate")
    p.add_argument("--run", action="store_true")
    p.add_argument("--status", action="store_true")
    p.add_argument("--name", default="simulate", help="process name (default: simulate)")
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--ckpt-every", type=int, default=1)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--plan", help="json file: steps, in the order they become deadlock steps")
    a = p.parse_args()
    status(a) if a.status else run(a)
