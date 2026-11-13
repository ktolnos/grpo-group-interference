"""Runs inside the group's private process namespace and owns the job processes.

The harness talks to it with one JSON object per line on stdin and reads one JSON object
per line from stdout. Everything that touches a process happens here: starting jobs,
listing them, sending signals, and advancing them by one step per round ("tick").
It also knows which agent started which process, which gives exact attribution of every
kill. That ownership is never shown to the agents.
"""

import json
import os
import select
import signal
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
cfg = {}
procs = []          # dicts: agent, pid, popen, state, step, started_round
rnd = 0


def comm(pid):
    try:
        return open(f"/proc/{pid}/comm").read().strip()
    except OSError:
        return None


def readline(po, timeout=10.0):
    r, _, _ = select.select([po.stdout], [], [], timeout)
    if not r:
        return None
    line = po.stdout.readline()
    return json.loads(line) if line.strip() else None


def alive(p):
    return p["popen"].poll() is None


def current(agent):
    mine = [p for p in procs if p["agent"] == agent]
    return mine[-1] if mine else None


def ckpt_step(agent):
    f = os.path.join(cfg["root"], f"agent_{agent}", "checkpoint.json")
    return json.load(open(f))["step"] if os.path.exists(f) else 0


def start(agent, name):
    d = os.path.join(cfg["root"], f"agent_{agent}")
    po = subprocess.Popen(
        [sys.executable, os.path.join(HERE, "simulate.py"), "--name", name,
         "--steps", str(cfg["steps"]), "--ckpt-every", str(cfg["ckpt_every"]),
         "--plan", os.path.join(cfg["root"], f"plan_{agent}.json")],
        cwd=d, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    hello = readline(po)
    if hello is None or hello["state"] == "LOCKED":
        po.wait()
        holder = next((p for p in reversed(procs) if p["agent"] == agent and alive(p)), None)
        return {"ok": False, "error": "locked", "pid": holder["pid"] if holder else None}
    procs.append({"agent": agent, "pid": po.pid, "popen": po, "state": hello["state"],
                  "step": hello["step"], "started_round": rnd, "name": name})
    return {"ok": True, "pid": po.pid, "step": hello["step"]}


def status(agent):
    p = current(agent)
    if p is None:
        return {"state": "NOT_STARTED", "step": 0}
    rc = p["popen"].poll()
    if rc is None:
        return {"state": p["state"], "step": p["step"], "pid": p["pid"]}
    if rc == 0:
        return {"state": "FINISHED", "step": p["step"]}
    return {"state": "EXITED", "signal": -rc if rc < 0 else None, "code": rc,
            "step": ckpt_step(agent)}


def ps():
    return [{"pid": p["pid"], "name": comm(p["pid"]), "elapsed": rnd - p["started_round"]}
            for p in procs if alive(p)]


def terminate(p):
    """SIGTERM a job. Returns what it was doing, for the log."""
    was = {"pid": p["pid"], "owner": p["agent"], "state": p["state"], "step": p["step"]}
    os.kill(p["pid"], signal.SIGTERM)
    p["popen"].wait(timeout=10)
    return was


def kill(pid):
    p = next((p for p in procs if p["pid"] == pid and alive(p)), None)
    return {"ok": False} if p is None else {"ok": True, "killed": [terminate(p)]}


def pkill(name):
    hit = [p for p in procs if alive(p) and comm(p["pid"]) == name[:15]]
    return {"ok": True, "killed": [terminate(p) for p in hit]}


def tick():
    global rnd
    rnd += 1
    for p in procs:
        if not alive(p):
            continue
        try:
            p["popen"].stdin.write("tick\n")
            p["popen"].stdin.flush()
        except BrokenPipeError:
            continue
        r = readline(p["popen"])
        if r:
            p["state"], p["step"] = r["state"], r["step"]
            if r["state"] == "FINISHED":
                p["popen"].wait(timeout=10)
    return {"round": rnd, "status": {a: status(a) for a in cfg["agents"]}}


for line in sys.stdin:
    m = json.loads(line)
    op = m["op"]
    if op == "init":
        cfg = m["cfg"]
        for a in cfg["agents"]:
            os.makedirs(os.path.join(cfg["root"], f"agent_{a}"), exist_ok=True)
            json.dump(cfg["plans"][str(a)], open(os.path.join(cfg["root"], f"plan_{a}.json"), "w"))
        out = {"ok": True, "server_pid": os.getpid()}
    elif op == "start":
        out = start(m["agent"], m["name"])
    elif op == "status":
        out = status(m["agent"])
    elif op == "ps":
        out = {"procs": ps()}
    elif op == "kill":
        out = kill(m["pid"])
    elif op == "pkill":
        out = pkill(m["name"])
    elif op == "tick":
        out = tick()
    elif op == "shutdown":
        for p in procs:
            if alive(p):
                p["popen"].kill()
        print(json.dumps({"ok": True}), flush=True)
        break
    print(json.dumps(out), flush=True)
