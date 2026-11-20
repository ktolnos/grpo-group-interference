"""Compile paper.tex to paper.pdf. The ICML style files sit next to it in this folder.

    python paper/build_paper.py            compile once
    python paper/build_paper.py --watch    recompile whenever a source file changes

Intermediate files go to paper/build/. latexmk runs pdflatex and bibtex as often as
needed.
"""
import argparse
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent
BUILD = ROOT / "build"
SOURCES = ["paper.tex", "refs.bib", "figures", "../sim/figures"]   # watched if present


def build():
    env = dict(os.environ)
    # bibtex runs on build/paper.aux, so it is told where the .bst and .bib files are;
    # the trailing ':' keeps TeX's own search path
    for var in ("BSTINPUTS", "BIBINPUTS"):
        env[var] = f"{ROOT}:{env.get(var, '')}"
    r = subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "-file-line-error",
         f"-outdir={BUILD}", "paper.tex"],
        cwd=ROOT, env=env, capture_output=True, text=True, errors="replace")
    log = BUILD / "paper.log"
    text = log.read_text(errors="replace") if log.exists() else r.stdout
    if r.returncode != 0:
        errors = [l for l in text.splitlines() if re.match(r"(\S+:\d+: |! )", l)]
        print("FAILED\n" + "\n".join(errors[:10] or r.stdout.splitlines()[-15:]))
        print(f"full log: {log}")
        return False
    shutil.copy(BUILD / "paper.pdf", ROOT / "paper.pdf")
    pages = re.search(r"Output written on .*\((\d+) pages?", text)
    warnings = [l for l in text.splitlines()
                if re.search(r"undefined|multiply.defined|Overfull \\hbox", l)]
    print(f"paper.pdf: {pages.group(1) if pages else '?'} pages, {len(warnings)} warnings")
    for l in warnings[:10]:
        print("  " + l)
    return True


def stamp():
    files = []
    for s in SOURCES:
        p = ROOT / s
        files += [f for f in p.rglob("*") if f.is_file()] if p.is_dir() else [p] * p.exists()
    return max((f.stat().st_mtime for f in files), default=0)


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    a.add_argument("--watch", action="store_true", help="recompile when a source file changes")
    args = a.parse_args()
    ok = build()
    if not args.watch:
        sys.exit(0 if ok else 1)
    last = stamp()
    while True:
        time.sleep(1)
        if (now := stamp()) != last:
            last = now
            build()
