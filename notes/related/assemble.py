"""Assemble KNOWN_RELATED_WORK.md: fill arXiv abstracts from the arXiv API (exact text),
drop search logs, mark cross-file duplicates."""
import html, re, subprocess, sys, time
from pathlib import Path

D = Path(__file__).parent
files = sorted(D.glob("0*.md"))
text = {f: f.read_text() for f in files}
ids = sorted({m for t in text.values() for m in re.findall(r"arXiv id:\*\* *([a-z\-]+/\d{7}|\d{4}\.\d{4,5})", t)})
abst, title = {}, {}
for i in range(0, len(ids), 15):
    chunk = ids[i:i + 15]
    for attempt in range(5):
        x = subprocess.run(["curl", "-s", "-m", "60",
                            "https://export.arxiv.org/api/query?max_results=50&id_list=" + ",".join(chunk)],
                           capture_output=True, text=True).stdout
        if "<entry>" in x:
            break
        time.sleep(20)
    for e in x.split("<entry>")[1:]:
        m = re.search(r"<id>http://arxiv.org/abs/(.*?)(v\d+)?</id>", e)
        s = re.search(r"<summary>(.*?)</summary>", e, re.S)
        t = re.search(r"<title>(.*?)</title>", e, re.S)
        if m and s:
            abst[m.group(1)] = html.unescape(" ".join(s.group(1).split()))
            title[m.group(1)] = html.unescape(" ".join(t.group(1).split()))
    time.sleep(4)
missing = [i for i in ids if i not in abst]
print("ids", len(ids), "fetched", len(abst), "missing", missing, file=sys.stderr)

seen, out, stats = {}, [], dict(entries=0, dup=0, filled=0, notfound=0)
for f in files:
    t = text[f]
    t = re.split(r"\n## Search log", t)[0].rstrip() + "\n"
    parts = re.split(r"(?m)^(?=### )", t)
    res = [parts[0]]
    for p in parts[1:]:
        head = p.splitlines()[0][4:].strip()
        m = re.search(r"arXiv id:\*\* *([a-z\-]+/\d{7}|\d{4}\.\d{4,5})", p)
        key = m.group(1) if m else re.sub(r"\W+", "", head.lower())[:60]
        tail = ""
        k = re.search(r"(?m)^(#{1,2} .*)$", p)       # a section heading following this entry
        if k:
            p, tail = p[:k.start()], p[k.start():]
        if key in seen:
            stats["dup"] += 1
            rel = re.search(r"- \*\*Relation:\*\*.*", p)
            res.append(f"### {head}\n- Duplicate: full entry under \"{seen[key]}\"."
                       + (f"\n- **Additional note from this search:** {rel.group(0)[16:].strip()}" if rel else "")
                       + "\n\n" + tail)
            continue
        seen[key] = f.stem
        stats["entries"] += 1
        if m and m.group(1) in abst:
            a = abst[m.group(1)]
            p = re.sub(r"(Abstract \(verbatim\):\*\*) ARXIV[^\n]*",
                       lambda _: f"**Abstract (verbatim, arXiv API; title there: \"{title[m.group(1)]}\"):** " + a
                       if False else "Abstract (verbatim, from the arXiv API):** " + a, p, count=1)
            stats["filled"] += 1
        if "NOT FOUND" in p or re.search(r"Abstract \(verbatim\):\*\* ARXIV", p):
            stats["notfound"] += 1
        res.append(p + tail)
    out.append("".join(res))
print(stats, file=sys.stderr)
(D / "_body.md").write_text("\n\n---\n\n".join(out))
(D / "_titles.txt").write_text("\n".join(f"{k}\t{v}" for k, v in title.items()))
