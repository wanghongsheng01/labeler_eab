"""Figures of the paper (PDF, IEEE column widths). Reads results/*.csv, writes paper/figs/*.pdf."""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R, F = f"{HERE}/results", f"{HERE}/paper/figs"
os.makedirs(F, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.5, "axes.axisbelow": True,
                     "pdf.fonttype": 42})
COL = {"ds-flash": "#2a78d6", "ds-pro": "#eb6834", "glm-5.3-flash": "#1baf7a", "kimi-k2.6": "#6250d6"}
MK = {"ds-flash": "o", "ds-pro": "s", "glm-5.3-flash": "^", "kimi-k2.6": "D"}
NAME = {"ds-flash": "DS-Flash", "ds-pro": "DS-Pro", "glm-5.3-flash": "GLM-5.3-Flash", "kimi-k2.6": "Kimi-K2.6"}
TASKS = ["imdb", "agnews", "offensive", "yelp", "amazon", "emotion", "hate"]  # by increasing composition sensitivity
TN = {"imdb": "IMDB", "agnews": "AG News", "offensive": "Offensive", "yelp": "Yelp", "amazon": "Amazon",
      "emotion": "Emotion", "hate": "Hate"}
MODELS = list(COL)


def fig_composition():
    C = pd.read_csv(f"{R}/rev_composition.csv").query("B == 10")  # cluster-bootstrap intervals
    fig, ax = plt.subplots(figsize=(7.0, 2.0))
    off = np.linspace(-0.27, 0.27, len(MODELS))
    for k, m in enumerate(MODELS):
        x = C[C.model == m].set_index("task").reindex(TASKS)
        xs = np.arange(len(TASKS)) + off[k]
        ax.errorbar(xs, 100 * x.comp, yerr=100 * np.vstack([x.comp - x.comp_lo, x.comp_hi - x.comp]), fmt=MK[m],
                    ms=4.2, color=COL[m], elinewidth=0.8, capsize=0, label=f"{NAME[m]}: new neighbors")
        ax.errorbar(xs, 100 * x.rerun, yerr=100 * np.vstack([x.rerun - x.rerun_lo, x.rerun_hi - x.rerun]), fmt=MK[m],
                    ms=4.2, mfc="white", color=COL[m], elinewidth=0.8, capsize=0, label=f"{NAME[m]}: rerun")
        ax.plot(xs, 100 * x.pos, "_", ms=7, mew=1.4, color=COL[m])  # same calls, positions permuted (Block P)
    ax.plot([], [], "_", ms=7, mew=1.4, color="#55544f", label="same calls, permuted")
    ax.set_xticks(range(len(TASKS)), [TN[t] for t in TASKS])
    ax.set_ylabel("Labels changed (%)")
    ax.set_xlim(-0.5, len(TASKS) - 0.5)
    ax.legend(ncol=5, fontsize=6.2, loc="upper left", handletextpad=0.3, columnspacing=1.0)
    ax.set_ylim(0, 31)
    fig.tight_layout(pad=0.3)
    fig.savefig(f"{F}/composition.pdf")


def fig_order():
    O = pd.read_csv(f"{R}/rev_order.csv").query("B == 10")  # vs. mean of 4 random orders, cluster-robust, Holm
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 2.1), sharey=True)
    off = np.linspace(-0.27, 0.27, len(MODELS))
    for ax, o, title in zip(axs, ["sorted", "keyed"], ["Clustered on the attribute", "Clustered on text length"]):
        for k, m in enumerate(MODELS):
            x = O[(O.model == m) & (O.order == o)].set_index("task").reindex(TASKS)
            sig = x.p_holm < 0.05
            ys = np.arange(len(TASKS)) + off[k]
            ax.scatter(x.diff_pp[sig], ys[sig.values], marker=MK[m], s=14, color=COL[m], label=NAME[m])
            ax.scatter(x.diff_pp[~sig], ys[~sig.values], marker=MK[m], s=14, facecolor="white", edgecolor=COL[m], lw=0.8)
        ax.axvline(0, color="#8a8984", lw=0.6)
        ax.set_title(title, fontsize=7.5)
        ax.set_xlim(-8, 9)
        ax.set_xlabel("Accuracy change (pp)")
        ax.grid(axis="y", visible=False)
    axs[0].set_yticks(range(len(TASKS)), [TN[t] for t in TASKS])
    axs[0].invert_yaxis()
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, fontsize=6, loc="lower center", bbox_to_anchor=(0.55, 0.0), handletextpad=0.1, columnspacing=0.6)
    fig.tight_layout(pad=0.3, rect=(0, 0.08, 1, 1))
    fig.savefig(f"{F}/order.pdf")


def fig_batch():
    c = pd.read_csv(f"{R}/main_batchcurve.csv").query("model == 'ds-flash'")
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.95))
    TC = dict(zip(TASKS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]))
    for t in TASKS:
        x = c[(c.task == t) & (c.fmt == "batch")].sort_values("B")
        s = c[(c.task == t) & (c.fmt == "single")]
        a0 = s.acc.iloc[0]
        axs[0].plot(x.B, 100 * (x.acc - a0), "-", lw=1.2, color=TC[t], label=TN[t])
        axs[1].plot(x.B, x.usd_1k / s.usd_1k.iloc[0], "-", lw=1.2, color=TC[t])
    for ax in axs:
        ax.set_xscale("log"); ax.set_xticks([1, 10, 100], ["1", "10", "100"]); ax.set_xticks([5, 20, 50], minor=True)
        ax.set_xlabel("Tuples per call $B$")
    axs[0].axhline(0, color="#8a8984", lw=0.6)
    axs[0].set_ylabel("Accuracy vs. one-per-call (pp)")
    axs[1].set_ylabel("Cost vs. one-per-call")
    axs[1].axhline(1, color="#8a8984", lw=0.6)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=7, fontsize=5.6, loc="upper center", bbox_to_anchor=(0.5, 1.0), handletextpad=0.2,
               columnspacing=0.6, handlelength=1.0)
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.9))
    fig.savefig(f"{F}/batch.pdf")


def fig_refusal():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    rows = []
    for t, K in [("agnews", 4), ("hate", 2)]:
        d = pd.read_csv(f"{R}/main/B10__glm-5.3-flash__{t}.csv")
        for o in ["shuffled", "keyed", "sorted"]:
            x = d[d.order == o]; k = x[x.pred != -2]
            a, b = np.bincount(x.gold, minlength=K) / len(x), np.bincount(k.gold, minlength=K) / len(k)
            rows.append(dict(task=t, order=o, shift=100 * np.abs(a - b).max(), dropped=len(x) - len(k)))
    D = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(3.5, 1.55))
    ON = {"shuffled": "Random (canonical)", "keyed": "Clustered on text length", "sorted": "Clustered on the attribute"}
    OC = {"shuffled": "#9ec5f4", "keyed": "#5598e7", "sorted": "#184f95"}
    for i, t in enumerate(["agnews", "hate"]):
        for j, o in enumerate(["shuffled", "keyed", "sorted"]):
            r = D[(D.task == t) & (D.order == o)].iloc[0]
            y = i + (j - 1) * 0.26
            ax.barh(y, r["shift"], height=0.22, color=OC[o], label=ON[o] if i == 0 else None)
            ax.text(r["shift"] + 0.05, y, f"{r['shift']:.2f} pp ({int(r['dropped'])} tuples lost)", va="center", fontsize=6)
    ax.set_yticks([0, 1], ["AG News", "Hate"]); ax.invert_yaxis()
    ax.set_xlabel("Largest shift of a GROUP BY class share (pp)")
    ax.set_xlim(0, 4.6); ax.grid(axis="y", visible=False)
    ax.legend(fontsize=6, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=3, handlelength=1, columnspacing=0.8)
    fig.tight_layout(pad=0.3)
    fig.savefig(f"{F}/refusal.pdf")
    D.to_csv(f"{R}/main_refusal_bias.csv", index=False)


if __name__ == "__main__":
    fig_composition(); fig_order(); fig_batch(); fig_refusal()
    print("ok")
