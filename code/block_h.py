"""Block H (2026-11-09): a table scanned in the order in which it is distributed.

The DBpedia-14 test split is stored sorted by class (as are 7 of 28 popular test splits; storage_order.py). We label
the same 1,000 tuples with B = 10 in two random orders and in the stored order, and compare accuracy (z-test clustered
by call against the mean of the two random orders) and composition sensitivity.
Usage: python block_h.py <model> [...]
Output: results/main/H__<model>__dbpedia.csv"""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot

M = f"{pilot.HERE}/results/main"
TASK, N, B = "dbpedia", 1000, 10


def main(models):
    df = pilot.load(TASK)
    for model in models:
        d = pd.concat([pilot.run_cell(model, TASK, B, "batch", o, 0, N) for o in ["shuffled", "shuffled1", "stored"]])
        d.to_csv(f"{M}/H__{model}__{TASK}.csv", index=False)
        piv = d.pivot_table(index="tid", columns="order", values="pred", aggfunc="first")
        gold = df.gold.values[piv.index.values]
        ok = (piv.shuffled >= 0) & (piv.shuffled1 >= 0)
        cs = float((piv.shuffled != piv.shuffled1)[ok].mean())
        base = np.mean([(piv[o].values == gold) for o in ["shuffled", "shuffled1"]], axis=0)
        diff = (piv["stored"].values == gold) - base
        cl = pd.Series({t: i // B for i, t in enumerate(pilot.order_of(df, "stored", N))}).reindex(piv.index).values
        s = pd.Series(diff - diff.mean()).groupby(cl).sum().values
        se = np.sqrt((s ** 2).sum()) / len(diff)
        print(model, TASK, "acc random", round(float(base.mean()), 4), "stored minus random (pp)",
              round(100 * diff.mean(), 2), "p", round(float(2 * norm.sf(abs(diff.mean() / se))), 5), "CS", round(cs, 4), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
