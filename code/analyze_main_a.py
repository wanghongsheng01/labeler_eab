"""Block A/A4 of the main runs: batch curves, format vs context decomposition (H4), JSON vs indexed format, cost model.
Writes results/main_batchcurve.csv and results/main_decomposition.csv."""
import glob, os
import numpy as np, pandas as pd
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(HERE, "results")
pd.set_option("display.width", 220)
d = pd.concat([pd.read_csv(f) for f in glob.glob(f"{R}/main/A__*.csv") + glob.glob(f"{R}/main/A4__*.csv")])

curve = d.groupby(["model", "task", "fmt", "B"]).apply(lambda x: pd.Series(dict(
    n=len(x), acc=(x.pred == x.gold).mean(), unparsed=(x.pred == -1).mean(), missing=x.missing.mean(),
    usd_1k=x.usd.sum() / len(x) * 1000, out_tok=x.out_tok.mean()))).reset_index()
curve.to_csv(f"{R}/main_batchcurve.csv", index=False)


def mcn(x, a, b):
    pa, pb = (x[(x.fmt == f) & (x.B == B)].set_index("tid") for f, B in (a, b))
    idx = pa.index.intersection(pb.index)
    ca, cb = (pa.pred.loc[idx] == pa.gold.loc[idx]).values, (pb.pred.loc[idx] == pb.gold.loc[idx]).values
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    return 100 * (cb.mean() - ca.mean()), binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0


rows = []
for (m, t), x in d.groupby(["model", "task"]):
    r = dict(model=m, task=t)
    r["format_pp"], r["format_p"] = mcn(x, ("single", 1), ("batch", 1))
    for B in (10, 50, 100):
        if ((x.fmt == "batch") & (x.B == B)).any():
            r[f"context{B}_pp"], r[f"context{B}_p"] = mcn(x, ("batch", 1), ("batch", B))
    if (x.fmt == "json").any():
        r["json_vs_batch_B10_pp"], r["json_vs_batch_B10_p"] = mcn(x, ("batch", 10), ("json", 10))
        r["json_cost_ratio_B10"] = curve.query("model==@m and task==@t and fmt=='json' and B==10").usd_1k.iloc[0] / \
            curve.query("model==@m and task==@t and fmt=='batch' and B==10").usd_1k.iloc[0]
    r["saving_single_to_B50"] = 1 - curve.query("model==@m and task==@t and fmt=='batch' and B==50").usd_1k.iloc[0] / \
        curve.query("model==@m and task==@t and fmt=='single' and B==1").usd_1k.iloc[0]
    rows.append(r)
D = pd.DataFrame(rows)
D.to_csv(f"{R}/main_decomposition.csv", index=False)
print(D.round(3).to_string())
print("\nsignificant (p<0.05) format effects:", int((D.format_p < 0.05).sum()), "/", len(D),
      " context (B=1->50):", int((D.context50_p < 0.05).sum()), "/", len(D))
print("\nds-flash accuracy by B (batch format):")
print(curve.query("model=='ds-flash' and fmt=='batch'").pivot(index="task", columns="B", values="acc").round(3).to_string())
print("\nds-flash unparsed/missing for JSON by B:")
print(curve.query("model=='ds-flash' and fmt=='json'").pivot(index="task", columns="B", values="unparsed").round(3).to_string())
