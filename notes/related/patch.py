import html, re, subprocess, time
from pathlib import Path
p = Path(__file__).parent / "_body.md"
t = p.read_text()
parts = re.split(r"(?m)^(?=### )", t)
need = {}
for i, e in enumerate(parts):
    if re.search(r"Abstract \(verbatim\):\*\* ARXIV", e):
        m = re.search(r"arXiv id:\*\* *([a-z\-]+/\d{7}|\d{4}\.\d{4,5})", e)
        if m: need[m.group(1)] = i
got = {}
ids = sorted(need)
for j in range(0, len(ids), 5):
    for attempt in range(6):
        x = subprocess.run(["curl", "-s", "-m", "60", "https://export.arxiv.org/api/query?max_results=20&id_list=" + ",".join(ids[j:j+5])], capture_output=True, text=True).stdout
        if "<entry>" in x: break
        time.sleep(15)
    for e in x.split("<entry>")[1:]:
        m = re.search(r"<id>http://arxiv.org/abs/(.*?)(v\d+)?</id>", e); s = re.search(r"<summary>(.*?)</summary>", e, re.S)
        if m and s: got[m.group(1)] = html.unescape(" ".join(s.group(1).split()))
    time.sleep(5)
for k, i in need.items():
    if k in got:
        parts[i] = re.sub(r"Abstract \(verbatim\):\*\* ARXIV[^\n]*", lambda _: "Abstract (verbatim, from the arXiv API):** " + got[k], parts[i], count=1)
p.write_text("".join(parts))
print(len(need), len(got), [k for k in need if k not in got])
