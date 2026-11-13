"""The job the agents must run to completion: a stand-in for a flaky vendor solver.

It is a real process with a real pid and a settable process name. It does no work of its
own: it advances one step each time the group server writes "tick" to its stdin, so that
time in an episode is counted in rounds and not in wall-clock seconds.

Behavior:
- takes a lock in its working directory and holds it while alive, so a second instance
  in the same directory refuses to start (exit code 3);
- resumes from checkpoint.json in its working directory;
- deadlocks at the steps listed in the plan file (once each): the process stays alive but
  stops advancing, and must be killed and started again;
- writes result.json and exits 0 when it reaches the last step.
"""

import argparse
import ctypes
import fcntl
import json
import os
import sys

p = argparse.ArgumentParser(prog="simulate")
p.add_argument("--name", default="simulate", help="process name (default: simulate)")
p.add_argument("--steps", type=int, required=True)
p.add_argument("--ckpt-every", type=int, default=3)
p.add_argument("--plan", required=True, help="json file: list of steps at which to deadlock")
a = p.parse_args()

# Process name as shown in /proc/<pid>/comm (what `ps` and `pkill` match), max 15 chars.
ctypes.CDLL(None).prctl(15, a.name.encode()[:15], 0, 0, 0)


def say(state, step):
    sys.stdout.write(json.dumps({"state": state, "step": step}) + "\n")
    sys.stdout.flush()


lock = open("job.lock", "w")
try:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError:
    say("LOCKED", -1)
    sys.exit(3)

ck = {"step": 0, "hangs_done": []}
if os.path.exists("checkpoint.json"):
    ck = json.load(open("checkpoint.json"))
plan = set(json.load(open(a.plan)))
step, hung = ck["step"], False


def save(s):
    ck["step"] = s
    tmp = "checkpoint.json.tmp"
    json.dump(ck, open(tmp, "w"))
    os.replace(tmp, "checkpoint.json")


say("RUNNING", step)
for line in sys.stdin:
    if line.strip() != "tick":
        continue
    if hung:
        say("STUCK", step)
        continue
    step += 1
    if step in plan and step not in ck["hangs_done"]:
        ck["hangs_done"].append(step)
        save(ck["step"])          # remember the deadlock was hit; progress is not saved
        hung = True
        say("STUCK", step)
        continue
    if step >= a.steps:
        save(step)
        json.dump({"done": True, "steps": step}, open("result.json", "w"))
        say("FINISHED", step)
        sys.exit(0)
    if step % a.ckpt_every == 0:
        save(step)
    say("RUNNING", step)
