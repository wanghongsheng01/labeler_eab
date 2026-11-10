"""Block F (selective filters) and Block H (DBpedia in its stored order): summaries for Section V-B.

F: per (labeler, task, share of positives, condition) the recall, the false-positive rate and the relative error of
   COUNT(*) of the filter; the context effect on the false-positive rate is the batch with B = 10 (mean of the two
   groupings) against the batch template with B = 1 on the same negatives, tested with the exact McNemar test on the
   first grouping.
H: accuracy, share of tuples without an aligned answer, and accuracy on the answered tuples, per order.
Output: results/f_summary.csv, results/h_summary.csv"""
import glob, os
import numpy as np, pandas as pd
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = f"{HERE}/results/main"


def mcnemar(a, b):
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    return binomtest(n01, n01 + n10).pvalue if n01 + n10 else 1.0


def main():
    rows = []
    for f in sorted(glob.glob(f"{M}/F__*.csv")):
        d = pd.read_csv(f)
        for (model, task, prev), x in d.groupby(["model", "task", "prev"]):
            piv = x.pivot_table(index="tid", columns="cond", values="pred", aggfunc="first")
            gold = x.groupby("tid").gold.first().reindex(piv.index).values
            pos = gold == 1
            r = dict(model=model, task=task, prev=prev, n=len(piv))
            for c in ["single", "batch_b1", "batch_g1", "batch_g2"]:
                sel = piv[c].values == 1
                r[f"recall_{c}"], r[f"fpr_{c}"] = sel[pos].mean(), sel[~pos].mean()
                r[f"count_err_{c}"] = (sel.sum() - pos.sum()) / pos.sum()
            r["fpr_b10"] = (r["fpr_batch_g1"] + r["fpr_batch_g2"]) / 2
            r["count_err_b10"] = (r["count_err_batch_g1"] + r["count_err_batch_g2"]) / 2
            r["p_fpr_context"] = mcnemar(piv["batch_b1"].values[~pos] == 1, piv["batch_g1"].values[~pos] == 1)
            rows.append(r)
    F = pd.DataFrame(rows); F.to_csv(f"{HERE}/results/f_summary.csv", index=False)
    print(F[["model", "task", "prev", "fpr_single", "fpr_batch_b1", "fpr_b10", "p_fpr_context", "count_err_single",
             "count_err_batch_b1", "count_err_b10"]].round(3).to_string())
    rows = []
    for f in sorted(glob.glob(f"{M}/H__*.csv")):
        d = pd.read_csv(f)
        for (model, order), x in d.groupby(["model", "order"]):
            v = x[x.pred >= 0]
            rows.append(dict(model=model, order=order, acc=(x.pred == x.gold).mean(), unaligned=1 - len(v) / len(x),
                             acc_answered=(v.pred == v.gold).mean()))
    H = pd.DataFrame(rows); H.to_csv(f"{HERE}/results/h_summary.csv", index=False); print(H.round(4).to_string())


if __name__ == "__main__":
    main()
