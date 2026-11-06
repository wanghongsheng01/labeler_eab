"""Query-level instability, contrast effect and position effect (Sections V-A and V-B of the paper).
Writes results/main_querylevel.csv and results/main_contagion.csv; prints the pooled tests."""
import glob, os, sys
import numpy as np, pandas as pd
from scipy.stats import binomtest, chi2_contingency, spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot
from tasks import load

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = f"{HERE}/results"
RND = ["shuffled", "shuffled1", "shuffled2", "shuffled3"]
PRIMARY = ("ds-flash", "ds-pro", "glm-5.3-flash", "kimi-k2.6")  # labelers of the analysis plan; others: analyze_xvendor.py
G = lambda *ps: [f for p in ps for f in glob.glob(f"{R}/main/{p}") if os.path.basename(f).split("__")[1] in PRIMARY]
B = pd.concat([pd.read_csv(f) for f in G("B__*.csv", "B10__*.csv")])
RR = pd.concat([pd.read_csv(f) for f in G("R__*.csv", "R10__*.csv")])


def maj(r):  # majority over the four random compositions; ties go to the canonical label
    v = [a for a in r if a >= 0]
    if not v: return -1
    c = pd.Series(v).value_counts(); top = c[c == c.max()].index
    return r[0] if r[0] in top else top[0]


def querylevel():
    rows = []
    for (m, t, b), x in B.groupby(["model", "task", "B"]):
        piv = x.pivot_table(index="tid", columns="order", values="pred", aggfunc="first")
        gold = x.groupby("tid").gold.first().loc[piv.index]
        acc = [(piv[o] == gold).mean() for o in RND]
        mv = np.array([maj(r) for r in piv[RND].values])
        K, jac, cnt = int(gold.max()) + 1, [], []
        for c in range(K):
            S0 = set(piv.index[piv["shuffled"] == c])
            for o in RND[1:]:
                S = set(piv.index[piv[o] == c])
                if S0 | S: jac.append(len(S0 & S) / len(S0 | S))
            cs = [(piv[o] == c).sum() for o in RND]
            cnt.append((max(cs) - min(cs)) / max(1, np.mean(cs)))
        # baseline: the same filter under the canonical grouping and its identical rerun
        rr = RR[(RR.model == m) & (RR.task == t) & (RR.B == b)].set_index("tid").pred.reindex(piv.index)
        jr = []
        for c in range(K):
            S0, S = set(piv.index[piv["shuffled"] == c]), set(piv.index[rr == c])
            if S0 | S: jr.append(len(S0 & S) / len(S0 | S))
        rows.append(dict(model=m, task=t, B=b, acc_mean=np.mean(acc), acc_min=min(acc), acc_max=max(acc),
                         acc_maj4=(mv == gold).mean(), jac_min=min(jac), jac_mean=np.mean(jac), count_range_max=max(cnt),
                         jac_rerun_min=min(jr) if jr else np.nan,
                         # refusals (-2), unparsed (-1) and errors (-3) are not labels
                         unstable_any=(piv[RND].where(piv[RND] >= 0).nunique(axis=1) > 1).mean()))
    D = pd.DataFrame(rows)
    D.to_csv(f"{R}/main_querylevel.csv", index=False)
    print(D.round(3).to_string())
    print("majority of 4 compositions minus mean accuracy (pp):", (100 * (D.acc_maj4 - D.acc_mean)).describe().round(2).to_dict())
    C = pd.read_csv(f"{R}/main_composition.csv").merge(D, on=["model", "task", "B"])
    print("Spearman(composition sensitivity, error rate):", spearmanr(C.comp, 1 - C.acc_mean))


def contrast():
    """For every label change between the canonical and another random composition (B=10): share of new neighbours
    whose gold label equals the new label minus the same share among the old neighbours. Positive = drift toward
    the neighbours; negative = contrast effect."""
    out = []
    for (m, t), x in B[B.B == 10].groupby(["model", "task"]):
        df = load(t); gold = df.gold.values
        n = x.groupby("order").tid.nunique().iloc[0]
        nb = {}
        for o in RND:
            tids = pilot.order_of(df, o, n)
            for i in range(0, n, 10):
                g = tids[i:i + 10]
                for u in g: nb[(o, u)] = [v for v in g if v != u]
        piv = x.pivot_table(index="tid", columns="order", values="pred", aggfunc="first")
        d, d_old, d_pred = [], [], []
        for o in RND[1:]:
            for u in piv.index[(piv[o] != piv["shuffled"]) & (piv[o] >= 0) & (piv["shuffled"] >= 0)]:
                new, old = piv.at[u, o], piv.at[u, "shuffled"]
                d.append(np.mean(gold[nb[(o, u)]] == new) - np.mean(gold[nb[("shuffled", u)]] == new))
                # robustness: the same measure for the old label, and with the neighbours' predicted labels
                d_old.append(np.mean(gold[nb[(o, u)]] == old) - np.mean(gold[nb[("shuffled", u)]] == old))
                d_pred.append(np.mean(piv[o].reindex(nb[(o, u)]).values == new) -
                              np.mean(piv["shuffled"].reindex(nb[("shuffled", u)]).values == new))
        d, d_old, d_pred = np.array(d), np.array(d_old), np.array(d_pred)
        pos, neg = int((d > 0).sum()), int((d < 0).sum())
        out.append(dict(model=m, task=t, changed=len(d), mean_shift=d.mean(), pos=pos, neg=neg,
                        p=binomtest(pos, pos + neg).pvalue if pos + neg else 1.0,
                        old_pos=int((d_old > 0).sum()), old_neg=int((d_old < 0).sum()),
                        pred_pos=int((d_pred > 0).sum()), pred_neg=int((d_pred < 0).sum())))
    D = pd.DataFrame(out)
    D.to_csv(f"{R}/main_contagion.csv", index=False)
    print(D.round(4).to_string())
    P, N = int(D.pos.sum()), int(D.neg.sum())
    print(f"pooled: {P + N} non-tie changes, toward neighbours {P}, away {N}, sign test p = {binomtest(P, P + N).pvalue:.2e}; "
          f"significant cells: negative {int(((D.p < .05) & (D.mean_shift < 0)).sum())}, positive {int(((D.p < .05) & (D.mean_shift > 0)).sum())}")
    for k, what in [("old", "old label, gold of neighbours (contrast predicts more)"), ("pred", "new label, predictions of neighbours")]:
        P, N = int(D[f"{k}_pos"].sum()), int(D[f"{k}_neg"].sum())
        print(f"robustness [{what}]: more {P}, less {N}, sign test p = {binomtest(P, P + N).pvalue:.2e}")


def position():
    d = pd.concat([pd.read_csv(f) for f in glob.glob(f"{R}/main/A__*.csv")])
    for b in (50, 100):
        x = d[(d.fmt == "batch") & (d.B == b)].copy()
        x["q"] = pd.cut(x.pos, [-1] + [b * k / 5 - 1 for k in range(1, 6)], labels=[1, 2, 3, 4, 5])
        x["ok"] = x.pred == x.gold
        print(f"B={b}: accuracy by position quintile", x.groupby("q").ok.mean().round(4).tolist(),
              "chi2 p =", round(chi2_contingency(pd.crosstab(x.q, x.ok))[1], 3))


if __name__ == "__main__":
    querylevel(); contrast(); position()
