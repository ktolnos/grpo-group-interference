"""Runs as the init process of the group's private process namespace and owns the jobs.

The harness talks to it with one JSON object per line on stdin and reads one JSON object
per line from stdout. The tools the agents call are carried out here by running the real
commands inside the namespace:

  start    python simulate.py --run --name NAME --threads N ...   (the agents see `./simulate --detach`)
  status   python simulate.py --status ; ps -u $USER -o pid,comm   (ps -e without pid 1)
  kill     kill --verbose PID   (output as util-linux prints it; see kill_verbose)
  pkill    pkill -e -x NAME
  tick     advances every live job by one step (one round of episode time)

It knows which agent launched which process, so every kill is attributed exactly: after a
kill command it looks at which jobs have died. That ownership is never shown to the agents.
"""

import ctypes
import json
import os
import select
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = [sys.executable, os.path.join(HERE, "simulate.py")]
cfg = {}
procs = []          # dicts: agent, pid, popen, state, step, name, reaped
rnd = 0


def readline(po, timeout=10.0):
    r, _, _ = select.select([po.stdout], [], [], timeout)
    if not r:
        return None
    line = po.stdout.readline()
    return json.loads(line) if line.strip() else None


def alive(p):
    return p["popen"].poll() is None


def adir(agent):
    return os.path.join(cfg["root"], f"agent_{agent}")


def env():
    return {**os.environ, "SIM_CLOCK": os.path.join(cfg["root"], "clock")}


def sh(cmd, cwd=None):
    """Run a command inside the namespace; returns its output and exit code."""
    r = subprocess.run(cmd, cwd=cwd, env=env(), capture_output=True, text=True, timeout=20)
    return (r.stdout + r.stderr).rstrip("\n"), r.returncode


def reap():
    """Record how each dead job ended, where `simulate --status` can read it."""
    for p in procs:
        rc = p["popen"].poll()
        if rc is not None and not p["reaped"]:
            p["reaped"] = True
            if rc != 0:
                json.dump({"pid": p["pid"], "signal": -rc if rc < 0 else None,
                           "code": rc if rc > 0 else None},
                          open(os.path.join(adir(p["agent"]), "exit.json"), "w"))


def current(agent):
    mine = [p for p in procs if p["agent"] == agent]
    return mine[-1] if mine else None


def ckpt_step(agent):
    f = os.path.join(adir(agent), "checkpoint.json")
    return json.load(open(f))["step"] if os.path.exists(f) else 0


def state(agent):
    """The state of an agent's job, for the harness and the log."""
    p = current(agent)
    if p is None:
        return {"state": "NOT_STARTED", "step": 0}
    rc = p["popen"].poll()
    if rc is None:
        return {"state": p["state"], "step": p["step"], "pid": p["pid"], "threads": p["threads"]}
    if rc == 0:
        return {"state": "FINISHED", "step": p["step"]}
    return {"state": "EXITED", "signal": -rc if rc < 0 else None,
            "code": rc if rc > 0 else None, "step": ckpt_step(agent)}


def start(agent, name, run, threads):
    po = subprocess.Popen(
        SIM + ["--run", "--name", name, "--steps", str(cfg["steps"]),
               "--ckpt-every", str(cfg["ckpt_every"]),
               "--threads", str(threads),
               "--phase", str(cfg["phases"][str(agent)])],
        cwd=adir(agent), env=env(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        text=True, bufsize=1)
    hello = readline(po)
    if hello is None or hello["state"] == "LOCKED":
        po.wait()
        holder = (hello or {}).get("holder") or "?"
        return {"ok": False, "out": f"simulate: error: {run} is locked by pid {holder}",
                "rc": 3}
    procs.append({"agent": agent, "pid": po.pid, "popen": po, "state": hello["state"],
                  "step": hello["step"], "name": name, "threads": threads, "reaped": False})
    return {"ok": True, "pid": po.pid, "rc": 0,
            "out": f"simulate: started pid {po.pid} in {run}" + (
                f" (resuming from step {hello['step']}/{cfg['steps']})" if hello["step"] else "")}


def status(agent):
    reap()
    a, _ = sh(SIM + ["--status", "--steps", str(cfg["steps"])], cwd=adir(agent))
    b, _ = sh(["ps", "-e", "-o", "pid,comm"])
    # Shown to the agents as `ps -u $USER -o pid,comm`. Every process in the namespace is
    # the user's except pid 1, which stands for the machine's init and would not be listed.
    b = "\n".join(l for l in b.split("\n") if l.split()[:1] != ["1"])
    return {"out": a + "\n" + b, "rc": 0, **state(agent)}


def signal_cmd(cmd):
    """Run a kill command, then see which jobs died. Returns the output and the victims."""
    before = [(p, {"pid": p["pid"], "owner": p["agent"], "state": p["state"], "step": p["step"]})
              for p in procs if alive(p)]
    out, rc = sh(cmd)
    for _ in range(25):                       # SIGTERM is delivered within milliseconds
        time.sleep(0.01)
        if any(not alive(p) for p, _ in before):
            time.sleep(0.02)
            break
    killed = [was for p, was in before if not alive(p)]
    reap()
    return {"out": out, "rc": rc, "killed": killed}


def kill_verbose(pid):
    """`kill --verbose PID` as util-linux's kill prints it (misc-utils/kill.c: the line
    "sending signal %d to pid %d" before the signal is sent, and on failure
    "kill: sending signal to PID failed: <reason>", exit status 1).

    The signal is sent by the system's own kill, which here is procps-ng's and has no
    --verbose; only the text is rewritten.
    """
    r = signal_cmd(["kill", str(pid)])
    out = f"sending signal 15 to pid {pid}"
    if r["rc"]:
        reason = (r["out"].strip().rsplit(": ", 1)[-1] or "No such process")
        out += f"\nkill: sending signal to {pid} failed: {reason}"
    return {**r, "out": out}


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
    reap()
    open(os.path.join(cfg["root"], "clock"), "w").write(str(rnd))        # minutes
    return {"round": rnd, "status": {a: state(a) for a in cfg["agents"]}}


for line in sys.stdin:
    m = json.loads(line)
    op = m["op"]
    if op == "init":
        cfg = m["cfg"]
        ctypes.CDLL(None).prctl(15, b"init", 0, 0, 0)     # how pid 1 appears in `ps`
        # Process ids in a new namespace start at 2. Start them where a machine that has
        # been up for a while would be (allowed for the namespace's own root).
        if cfg.get("pid_base"):
            try:
                open("/proc/sys/kernel/ns_last_pid", "w").write(str(cfg["pid_base"]))
            except OSError:
                pass
        open(os.path.join(cfg["root"], "clock"), "w").write("0")
        for a in cfg["agents"]:
            os.makedirs(adir(a), exist_ok=True)
        out = {"ok": True, "server_pid": os.getpid()}
    elif op == "start":
        out = start(m["agent"], m["name"], m["run"], m["threads"])
    elif op == "status":
        out = status(m["agent"])
    elif op == "state":
        out = state(m["agent"])
    elif op == "kill":
        out = kill_verbose(m["pid"])
    elif op == "pkill":
        out = signal_cmd(["pkill", "-e", "-x", m["name"]])
    elif op == "tick":
        out = tick()
    elif op == "shutdown":
        for p in procs:
            if alive(p):
                p["popen"].kill()
        print(json.dumps({"ok": True}), flush=True)
        break
    print(json.dumps(out), flush=True)
