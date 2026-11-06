"""Expected GROUP BY bias from refusals under random layouts (Monte Carlo), against the sorted layout (F4).
A call is refused iff it contains a trigger (all six triggers were refused whenever sent alone). For each layout we
drop the refused calls and measure (a) the largest absolute change of a gold-class share and (b) the signed change of
the triggers' class share. Writes results/refusal_mc.csv."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = pd.read_csv(f"{HERE}/results/main/D__glm-5.3-flash.csv")


def shifts(order, gold, trig, K, cls, B=10):
    keep = np.ones(len(order), bool)
    for i in range(0, len(order), B):
        if any(t in trig for t in order[i:i + B]): keep[i:i + B] = False
    g_all = np.bincount(gold[order], minlength=K) / len(order)
    g_kept = np.bincount(gold[order][keep], minlength=K) / keep.sum()
    return 100 * np.abs(g_kept - g_all).max(), 100 * (g_all[cls] - g_kept[cls])


rows = []
for task in ["agnews", "hate"]:
    df = pilot.load(task); n = 1000
    sub = np.array(pilot.order_of(df, "shuffled", n))
    x = D[D.task == task]
    trig = set(x.tid[(x.alone_rep0 == "refused") & (x.alone_rep1 == "refused")])
    gold = df.gold.values; K = gold.max() + 1
    cls = int(pd.Series(gold[list(trig)]).mode()[0])
    rng = np.random.default_rng(0)
    mc = np.array([shifts(rng.permutation(sub), gold, trig, K, cls) for _ in range(20000)])
    srt = shifts(np.array(pilot.order_of(df, "sorted", n)), gold, trig, K, cls)
    rows.append(dict(task=task, triggers=len(trig), random_maxabs_mean=mc[:, 0].mean(), random_signed_mean=mc[:, 1].mean(),
                     sorted_maxabs=srt[0], sorted_signed=srt[1],
                     ratio_maxabs=srt[0] / mc[:, 0].mean(), ratio_signed=srt[1] / mc[:, 1].mean()))
R = pd.DataFrame(rows); R.to_csv(f"{HERE}/results/refusal_mc.csv", index=False); print(R.round(3).to_string())
