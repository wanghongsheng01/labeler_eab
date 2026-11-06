"""Block B (composition, physical order) and R (rerun baseline) of the main runs: tests H1 and H2 per dataset.
Writes results/main_composition.csv and results/main_order.csv."""
import glob, os, re
import numpy as np, pandas as pd
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(HERE, "results")
pd.set_option("display.width", 220)


PRIMARY = ("ds-flash", "ds-pro", "glm-5.3-flash", "kimi-k2.6")  # labelers of the analysis plan; others: analyze_xvendor.py


def load(prefix):
    fs = [f for f in glob.glob(f"{R}/main/{prefix}__*.csv") if os.path.basename(f).split("__")[1] in PRIMARY]
    return pd.concat([pd.read_csv(f) for f in fs]) if fs else pd.DataFrame()


def boot_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return np.percentile(m, [2.5, 97.5])


B = pd.concat([load("B"), load("B10")])
Rr = pd.concat([load("R"), load("R10")])
comp, order = [], []
for (m, t, b), x in B.groupby(["model", "task", "B"]):
    piv = x.pivot_table(index="tid", columns="order", values="pred", aggfunc="first")
    gold = x.groupby("tid").gold.first().loc[piv.index]
    ok = piv["shuffled"] >= 0  # valid labels only: unparsed (-1), refused (-2) and failed (-3) answers are not labels
    ch = np.concatenate([((piv[o] != piv["shuffled"]) & ok & (piv[o] >= 0)).values[(ok & (piv[o] >= 0)).values]
                         for o in ["shuffled1", "shuffled2", "shuffled3"] if o in piv])
    row = dict(model=m, task=t, B=b, n=len(piv), comp=ch.mean(), comp_lo=boot_ci(ch.astype(float))[0],
               comp_hi=boot_ci(ch.astype(float))[1])
    rr = Rr[(Rr.model == m) & (Rr.task == t) & (Rr.B == b)] if len(Rr) else Rr
    if len(rr):
        r1 = rr.set_index("tid").pred.reindex(piv.index)
        ch2 = ((r1 != piv["shuffled"]) & ok & (r1 >= 0)).values[(ok & (r1 >= 0)).values]
        row.update(rerun=ch2.mean(), rerun_lo=boot_ci(ch2.astype(float))[0], rerun_hi=boot_ci(ch2.astype(float))[1])
    comp.append(row)
    for o in ["sorted", "keyed"]:
        if o not in piv: continue
        a, c = piv["shuffled"] == gold, piv[o] == gold
        n01, n10 = int((a & ~c).sum()), int((~a & c).sum())
        K = int(gold.max()) + 1
        l1 = lambda p: np.abs(np.bincount(p[p >= 0].astype(int), minlength=K)[:K] / len(p) - np.bincount(gold, minlength=K) / len(gold)).sum()
        order.append(dict(model=m, task=t, B=b, order=o, acc_canon=a.mean(), acc_order=c.mean(), diff_pp=100 * (c.mean() - a.mean()),
                          p=binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0,
                          L1_canon=l1(piv["shuffled"].values), L1_order=l1(piv[o].values)))
C, O = pd.DataFrame(comp), pd.DataFrame(order)
if len(O):  # Holm correction within block B (all order contrasts)
    k = len(O); idx = np.argsort(O.p.values); adj = np.empty(k)
    adj[idx] = np.minimum(1, np.maximum.accumulate((k - np.arange(k)) * O.p.values[idx]))
    O["p_holm"] = adj
C.to_csv(f"{R}/main_composition.csv", index=False); O.to_csv(f"{R}/main_order.csv", index=False)
print("Composition (share of labels changed by new neighbours) vs rerun\n", C.round(4).to_string())
if "rerun" in C:
    h = C.dropna(subset=["rerun"])
    print(f"\nH1: composition > rerun (CI-separated) in {(h.comp_lo > h.rerun_hi).sum()}/{len(h)} cells; "
          f"point estimate larger in {(h.comp > h.rerun).sum()}/{len(h)}")
print("\nPhysical order vs canonical\n", O.round(4).to_string())
if len(O):
    s = O[O.order == "sorted"]
    print(f"\nH2: sorted order changes accuracy (Holm p<0.05) in {(s.p_holm < 0.05).sum()}/{len(s)} cells; "
          f"keyed (attribute-independent) in {(O[O.order == 'keyed'].p_holm < 0.05).sum()}/{(O.order == 'keyed').sum()}")
