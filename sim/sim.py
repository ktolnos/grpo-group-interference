"""Does a group-relative baseline reward harming the other rollouts in the group?

A policy has three Bernoulli decisions per rollout, each a single logit:
  sabotage  (prob p): costs the actor c and harms k other rollouts of the same group;
  defend    (prob q): costs e and blocks all harm to this rollout (only when enabled);
  task      (prob t): success adds 1 to the reward (learned; supplies ordinary reward noise).

Reward of rollout j:  r_j = y_j - c s_j - e f_j - h * hits_j * (1 - f_j),
where hits_j counts saboteurs in the group that targeted j and h is the harm per hit
(h = d, or h = d / k with --split so that each act does total harm d).

Advantage estimators:
  grpo      r - group mean                         (Dr. GRPO: no std scaling)
  grpo_std  (r - group mean) / group std           (original GRPO scaling)
  rloo      r - mean of the other G - 1 rewards
  ext       r - mean of a different, independent group (baseline independent of the action)
  diff      as ext, minus the harm this rollout inflicted (a difference reward)

Expected action gap of the sabotage decision (no defense), with H = k h the harm per act:
  grpo  H / G - (1 - 1/G) c        rloo  H / (G - 1) - c
  ext   -c                         diff  -c - H
The true gradient of the mean reward credits sabotage with -c - H, like diff.

Modes:
  measure   lr = 0 at p = q = t = 1/2; estimate the action gap from the batch
  train     SGD or Adam on the logits; record the trajectory
"""

import argparse
import itertools
import json
import multiprocessing as mp

import numpy as np


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    return float(np.log(p / (1 - p)))


def predicted_gap(est, G, H, c):
    return {"grpo": H / G - (1 - 1 / G) * c,
            "rloo": H / (G - 1) - c,
            "ext": -c,
            "diff": -c - H}.get(est)


def run(cfg):
    rng = np.random.default_rng(cfg["seed"])
    G, B, k = cfg["G"], cfg["B"], cfg["k"]
    c, d, e = cfg["c"], cfg["d"], cfg.get("e", 0.0)
    h = d / k if cfg.get("split") else d
    defense, est = cfg.get("defense", False), cfg["est"]
    th = np.array([logit(cfg.get("p0", 0.5)),
                   logit(cfg.get("q0", 0.5)) if defense else -np.inf,
                   0.0])
    learn = np.array([True, defense, True])
    lr, T, rec = cfg.get("lr", 0.0), cfg["T"], cfg.get("rec", 25)
    m = np.zeros(3)
    v = np.zeros(3)
    eye = np.eye(G, dtype=bool)
    gap_s, gap_d, traj = [], [], []
    for step in range(T + 1):
        ps, pd, pt = sig(th)
        s = rng.random((B, G)) < ps
        f = rng.random((B, G)) < pd if defense else np.zeros((B, G), bool)
        y = rng.random((B, G)) < pt
        keys = rng.random((B, G, G))
        keys[:, eye] = 2.0                          # never target oneself
        tgt = keys.argsort(-1).argsort(-1) < k      # tgt[b, i, j]: i targets j
        hits = (s[:, :, None] & tgt).sum(1)
        inflicted = (s[:, :, None] & tgt & ~f[:, None, :]).sum(2) * h
        r = y - c * s - e * f - h * hits * (1 - f)
        mean = r.mean(1, keepdims=True)
        if est == "grpo":
            A = r - mean
        elif est == "grpo_std":
            A = (r - mean) / (r.std(1, keepdims=True) + 1e-4)
        elif est == "rloo":
            A = r - (r.sum(1, keepdims=True) - r) / (G - 1)
        elif est in ("ext", "diff"):
            A = r - np.roll(mean, 1, axis=0)
            if est == "diff":
                A = A - inflicted
        sc = np.stack([s - ps, f - pd, y - pt])     # scores of the three logits
        g = (A[None] * sc).reshape(3, -1).mean(1)
        if cfg["mode"] == "measure":
            gap_s.append(g[0] / (ps * (1 - ps)))
            if defense:
                gap_d.append(g[1] / (pd * (1 - pd)))
            continue
        if step % rec == 0:
            traj.append([step, float(ps), float(pd) if defense else 0.0, float(pt),
                         float(r.mean())])
        if step == T:
            break
        if cfg.get("opt", "sgd") == "adam":
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            upd = (m / (1 - 0.9 ** (step + 1))) / (np.sqrt(v / (1 - 0.999 ** (step + 1))) + 1e-8)
        else:
            upd = g
        th = np.where(learn, np.clip(th + lr * upd, -12, 12), th)
    out = dict(cfg)
    if cfg["mode"] == "measure":
        gs = np.array(gap_s)
        out["gap"] = float(gs.mean())
        out["gap_se"] = float(gs.std(ddof=1) / np.sqrt(len(gs)))
        out["gap_pred"] = predicted_gap(est, G, k * h, c)
        if defense:
            out["gap_def"] = float(np.mean(gap_d))
    else:
        out["traj"] = traj
    return out


def configs(exp):
    if exp == "measure":
        for G, kmode, split, est in itertools.product(
                [2, 4, 8, 16, 32], ["one", "half", "all"], [False, True],
                ["grpo", "grpo_std", "rloo", "ext", "diff"]):
            k = {"one": 1, "half": max(1, G // 2), "all": G - 1}[kmode]
            if G == 2 and kmode != "one":
                continue
            yield dict(exp=exp, mode="measure", G=G, k=k, kmode=kmode, split=split,
                       c=0.1, d=0.4, est=est, B=max(256, 8192 // G), T=200, seed=0)
    elif exp == "train":
        for k, ratio, est, seed in itertools.product(
                [1, 4, 7], [0.5, 1, 2, 4, 8, 16],
                ["grpo", "grpo_std", "rloo", "ext", "diff"], range(3)):
            yield dict(exp=exp, mode="train", G=8, k=k, c=0.05, d=0.05 * ratio,
                       ratio=ratio, est=est, B=64, T=3000, lr=1.0, p0=0.1, seed=seed)
    elif exp == "spite":
        for G, est, seed in itertools.product(
                [2, 3, 4, 6, 8, 12, 16, 32], ["grpo", "rloo"], range(3)):
            yield dict(exp=exp, mode="train", G=G, k=G - 1, split=True, c=0.05, d=0.4,
                       est=est, B=max(16, 512 // G), T=3000, lr=1.0, p0=0.1, seed=seed)
    elif exp == "defense":
        for (k, d), opt, seed in itertools.product(
                [(7, 0.3), (1, 0.6)], ["sgd", "adam"], range(3)):
            yield dict(exp=exp, mode="train", G=8, k=k, c=0.05, d=d, e=0.3 if k == 7 else 0.3,
                       defense=True, est="grpo", B=64, T=20000, rec=20,
                       lr=1.0 if opt == "sgd" else 0.01, opt=opt, p0=0.1, q0=0.1, seed=seed)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["measure", "train", "spite", "defense"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    cfgs = list(configs(a.exp))
    with mp.Pool(a.procs) as pool, open(a.out, "w") as fh:
        for res in pool.imap_unordered(run, cfgs):
            fh.write(json.dumps(res) + "\n")
            fh.flush()
    print(a.exp, len(cfgs), "configs done")
