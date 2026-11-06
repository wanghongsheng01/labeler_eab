"""Block T (2026-11-06): entity matching (Abt-Buy) with our template and BatchER's, Chinese review stars, and the
Sema-style structured-output template on four tasks; DS-Flash, B = 10. Same quantities as the main analysis:
composition sensitivity (canonical vs. one further random order), rerun flip rate, and the accuracy change of the
sorted and keyed orders against the mean of the two random orders (z-test clustered by call).
Writes results/xt_summary.csv."""
import glob, os, sys
import numpy as np, pandas as pd
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = f"{HERE}/results/main"


def zdiff(ok_o, base, cl):
    d = ok_o.astype(float) - base; m = d.mean()
    s = pd.Series(d - m).groupby(cl).sum().values
    se = np.sqrt((s ** 2).sum()) / len(d)
    return 100 * m, 2 * norm.sf(abs(m / se)) if se > 0 else 1.0


def f1(pred, gold):
    tp = ((pred == 1) & (gold == 1)).sum(); fp = ((pred == 1) & (gold == 0)).sum(); fn = ((pred != 1) & (gold == 1)).sum()
    return 2 * tp / max(1, 2 * tp + fp + fn)


def main():
    rows = []
    for f in sorted(glob.glob(f"{M}/T[EZJ]__*.csv")):
        d = pd.read_csv(f)
        for (model, task, fmt), x in d.groupby(["model", "task", "fmt"]):
            df = pilot.load(task)
            r0 = x[x.rep == 0]
            piv = r0.pivot_table(index="tid", columns="order", values="pred", aggfunc="first")
            gold = r0.groupby("tid").gold.first().loc[piv.index]
            n = len(piv); canon = piv["shuffled"]
            ok = (canon >= 0) & (piv["shuffled1"] >= 0)
            rr = x[x.rep == 1].set_index("tid").pred.reindex(piv.index)
            base = np.mean([(piv[o] == gold).values for o in ["shuffled", "shuffled1"]], axis=0)
            r = dict(model=model, task=task, template=fmt, n=n, acc=float((canon == gold).mean()),
                     unparsed=float((r0.pred == -1).mean()),
                     comp=float((piv["shuffled1"] != canon)[ok].mean()),
                     rerun=float((rr != canon)[(canon >= 0) & (rr >= 0)].mean()) if rr.notna().any() else np.nan)
            if task == "abtbuy":
                r["f1"] = f1(canon.values, gold.values)
            for o in ["sorted", "keyed"]:
                if o in piv:
                    cl = pd.Series({t: i // 10 for i, t in enumerate(pilot.order_of(df, o, n, task))}).reindex(piv.index).values
                    r[f"{o}_pp"], r[f"{o}_p"] = zdiff((piv[o] == gold).values, base, cl)
                    if task == "abtbuy":
                        r[f"{o}_f1"] = f1(piv[o].values, gold.values)
            rows.append(r)
    D = pd.DataFrame(rows)
    D.to_csv(f"{HERE}/results/xt_summary.csv", index=False)
    print(D.round(4).to_string())


if __name__ == "__main__":
    main()
