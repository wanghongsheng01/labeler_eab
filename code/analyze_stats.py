"""Paired significance tests for the pilot: prompt-format effect, in-batch context effect, physical-order effect,
and a fit of the per-tuple input-token cost model in_tok(B) = c + a / B."""
import glob, os
import numpy as np, pandas as pd
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(HERE, "results")


def mcnemar(a, b, gold):
    """a, b: prediction Series indexed by tid. Returns (acc_a, acc_b, diff_pp, exact p)."""
    idx = a.index.intersection(b.index)
    ca, cb, g = a.loc[idx].values == gold.loc[idx].values, b.loc[idx].values == gold.loc[idx].values, gold.loc[idx]
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return ca.mean(), cb.mean(), 100 * (cb.mean() - ca.mean()), p


rows = []
E1 = pd.concat([pd.read_csv(f) for f in glob.glob(f"{R}/pilot/E1__*.csv") + glob.glob(f"{R}/pilot/E1b__*.csv")])
for (m, t), x in E1.groupby(["model", "task"]):
    gold = x.groupby("tid").gold.first()
    get = lambda B, fmt: x[(x.B == B) & (x.fmt == fmt)].set_index("tid").pred
    s1 = get(1, "single")
    if "batch" in set(x[x.B == 1].fmt):
        rows.append(("format: single -> batch-format B=1", m, t, *mcnemar(s1, get(1, "batch"), gold)))
        for B in [10, 50, 100]:
            if B in set(x.B): rows.append((f"context: batch-format B=1 -> B={B}", m, t, *mcnemar(get(1, "batch"), get(B, "batch"), gold)))
    for B in [10, 50]:
        if B in set(x.B): rows.append((f"total: single -> B={B}", m, t, *mcnemar(s1, get(B, "batch"), gold)))
E3 = glob.glob(f"{R}/pilot/E3__*.csv")
if E3:
    E3 = pd.concat([pd.read_csv(f) for f in E3])
    for (m, t, B), x in E3[E3.order.isin(["sorted", "shuffled"])].groupby(["model", "task", "B"]):
        gold = x.groupby("tid").gold.first()
        rows.append((f"order: shuffled -> label-sorted, B={B}", m, t,
                     *mcnemar(x[x.order == "shuffled"].set_index("tid").pred, x[x.order == "sorted"].set_index("tid").pred, gold)))
S = pd.DataFrame(rows, columns=["contrast", "model", "task", "acc_from", "acc_to", "diff_pp", "p"])
S.to_csv(f"{R}/pilot_stats.csv", index=False)
pd.set_option("display.width", 200)
print(S.round(4).to_string())

# cost model: per-tuple input tokens vs batch size (batch format only)
print("\nfit in_tok(B) = c + a/B (batch format):")
for (m, t), x in E1[E1.fmt == "batch"].groupby(["model", "task"]):
    g = x.groupby("B").in_tok.mean()
    A = np.c_[np.ones(len(g)), 1 / g.index.values]
    (c, a), res, *_ = np.linalg.lstsq(A, g.values, rcond=None)
    pred = A @ [c, a]
    print(f"{m:10s} {t:8s} tuple_tokens c={c:6.1f} overhead a={a:6.1f}  max_rel_err={np.max(np.abs(pred - g.values) / g.values):.3f}"
          f"  overhead share at B=1: {a / (a + c):.2f}")
