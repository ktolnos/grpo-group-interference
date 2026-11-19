"""Figures and summary tables for the group-sabotage simulations (sim.py)."""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(sys.argv[1])        # the directory jobs/sim.sh wrote ($SCRATCH/ggi/sim)
FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)

# Reference categorical palette, fixed order; marker shape is the secondary encoding.
COL = {"grpo": "#2a78d6", "grpo_std": "#eb6834", "rloo": "#1baf7a", "ext": "#eda100",
       "diff": "#e87ba4"}
MRK = {"grpo": "o", "grpo_std": "s", "rloo": "^", "ext": "D", "diff": "v"}
NAME = {"grpo": "GRPO (group mean)", "grpo_std": "GRPO, std-scaled",
        "rloo": "leave-one-out (RLOO)", "ext": "independent baseline",
        "diff": "difference reward"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})


def load(name):
    p = OUT / f"{name}.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []


def final(r, idx):
    return r["traj"][-1][idx]


summary = {}

# --- Figure 1: measured vs predicted action gap -------------------------------------
M = load("measure")
if M:
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    lim = [-0.55, 0.25]
    ax.plot(lim, lim, color=INK2, lw=1, ls="--", zorder=1)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.axvline(0, color=INK2, lw=0.8)
    for est in ["grpo", "rloo", "ext", "diff"]:
        rs = [r for r in M if r["est"] == est and r["gap_pred"] >= lim[0]]
        ax.errorbar([r["gap_pred"] for r in rs], [r["gap"] for r in rs],
                    yerr=[1.96 * r["gap_se"] for r in rs], fmt=MRK[est], ms=5,
                    color=COL[est], mec="#fcfcfb", mew=0.8, elinewidth=0.8,
                    label=NAME[est], zorder=3)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("predicted action gap of sabotage")
    ax.set_ylabel("measured action gap of sabotage")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_gap.png", dpi=200)
    err = [abs(r["gap"] - r["gap_pred"]) / r["gap_se"] for r in M if r["gap_pred"] is not None]
    summary["measure"] = {"n": len(err), "max_abs_z": float(np.max(err)),
                          "frac_within_2se": float(np.mean(np.array(err) < 2))}
    rows = sorted(M, key=lambda r: (r["est"], r["split"], r["G"], r["k"]))
    summary["measure_rows"] = [
        {k: r[k] for k in ["est", "G", "k", "split", "gap", "gap_se", "gap_pred"]} for r in rows]

# --- Figure 2: final sabotage probability vs harm/cost ratio --------------------------
Tr = load("train")
if Tr:
    fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.2), sharey=True)
    tab = {}
    for ax, k in zip(axs, [1, 4, 7]):
        thr = 7 / k                      # (G - 1) / k with G = 8
        ax.axvspan(thr, 32, color="#f0efec", zorder=0)
        ax.axvline(thr, color=INK2, ls="--", lw=1)
        for est in ["grpo", "grpo_std", "rloo", "ext", "diff"]:
            by = defaultdict(list)
            for r in Tr:
                if r["k"] == k and r["est"] == est:
                    by[r["ratio"]].append(final(r, 1))
            xs = sorted(by)
            ys = [np.mean(by[x]) for x in xs]
            tab[(k, est)] = dict(zip(xs, ys))
            ax.plot(xs, ys, marker=MRK[est], ms=4.5, lw=2, color=COL[est],
                    mec="#fcfcfb", mew=0.8, label=NAME[est])
        ax.set_xscale("log", base=2)
        ax.set_xlim(0.4, 20)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"each act harms {k} of 7 peers" if k < 7 else "each act harms all 7 peers",
                     fontsize=9, color=INK)
        ax.set_xlabel("harm per target / cost to actor (d / c)")
    axs[0].set_ylabel("P(sabotage) after 3,000 steps")
    axs[0].legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_threshold.png", dpi=200)
    summary["train"] = {f"k{k}_{est}": v for (k, est), v in tab.items()}
    # mean reward at the end, relative to the no-sabotage optimum of 1
    rew = defaultdict(list)
    for r in Tr:
        rew[(r["k"], r["ratio"], r["est"])].append(final(r, 4))
    summary["train_reward"] = {f"k{k}_r{ra}_{e}": float(np.mean(v)) for (k, ra, e), v in rew.items()}

# --- Figure 3: spite with a fixed total harm, vs group size ---------------------------
S = load("spite")
if S:
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.axvspan(1.5, 9, color="#f0efec", zorder=0)
    ax.axvline(9, color=INK2, ls="--", lw=1)
    tab = {}
    for est in ["grpo", "rloo"]:
        by = defaultdict(list)
        for r in S:
            if r["est"] == est:
                by[r["G"]].append(final(r, 1))
        xs = sorted(by)
        ys = [np.mean(by[x]) for x in xs]
        tab[est] = dict(zip(xs, ys))
        ax.plot(xs, ys, marker=MRK[est], ms=5, lw=2, color=COL[est], mec="#fcfcfb",
                mew=0.8, label=NAME[est])
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8, 16, 32])
    ax.set_xticklabels(["2", "4", "8", "16", "32"])
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("group size G")
    ax.set_ylabel("P(sabotage) after 3,000 steps")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_spite.png", dpi=200)
    summary["spite"] = tab

# --- Figure 4: sabotage and defense ------------------------------------------------------
D = load("defense")
if D:
    fig, axs = plt.subplots(1, 3, figsize=(10.2, 3.3))
    for col, (k, d) in enumerate([(7, 0.3), (1, 0.6)]):
        ax = axs[col]
        G, c, e = 8, 0.05, 0.3
        pstar, qstar = e / (d * k), 1 - (G - 1) * c / (k * d)
        for opt, ls in [("sgd", "-"), ("adam", ":")]:
            for r in [x for x in D if x["k"] == k and x["opt"] == opt]:
                t = np.array(r["traj"])
                ax.plot(t[:, 1], t[:, 2], ls=ls, lw=1.2,
                        color=COL["grpo"] if opt == "sgd" else COL["grpo_std"], alpha=0.8)
        ax.plot([pstar], [qstar], marker="*", ms=11, color=INK, zorder=5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("P(sabotage)")
        ax.set_ylabel("P(defend)")
        ax.set_title(f"harm {d} to {'all 7 peers' if k == 7 else '1 peer'}; "
                     f"equilibrium ({pstar:.2f}, {qstar:.2f})", fontsize=8.5, color=INK)
    ax = axs[2]
    for opt in ["sgd", "adam"]:
        for i, r in enumerate([x for x in D if x["k"] == 7 and x["opt"] == opt]):
            t = np.array(r["traj"])
            ax.plot(t[:, 0], t[:, 4], lw=1.2, alpha=0.8,
                    color=COL["grpo"] if opt == "sgd" else COL["grpo_std"],
                    label=("SGD" if opt == "sgd" else "Adam") if i == 0 else None)
    ax.axhline(1.0, color=INK2, ls="--", lw=1)
    ax.set_xlabel("step")
    ax.set_ylabel("mean reward (harm to all 7 peers)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_defense.png", dpi=200)
    dsum = {}
    for r in D:
        t = np.array(r["traj"])
        late = t[t[:, 0] >= 10000]
        dsum.setdefault(f"k{r['k']}_{r['opt']}", []).append(
            dict(p_mean=float(late[:, 1].mean()), q_mean=float(late[:, 2].mean()),
                 p_range=[float(late[:, 1].min()), float(late[:, 1].max())],
                 reward=float(late[:, 4].mean())))
    summary["defense"] = dsum

json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
print(json.dumps({k: v for k, v in summary.items() if k != "measure_rows"}, indent=1)[:6000])
