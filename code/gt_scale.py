"""How far group testing over existing runs scales (no API calls; 2026-11-09).

Block D decodes refusal triggers from the seven calls that already contain each tuple (six groupings of B = 10 and the
rerun of the canonical one). Here the same seven groupings of 1,000 AG News tuples receive k synthetic triggers, and a
call that contains a trigger is refused with probability q (q = 1, and q = 0.93, the rate at which the real triggers
were refused inside batches). The decoder is the one of block_d.py: a greedy minimum hitting set of the refused calls.
Triggers are placed (uniform) at random or (topical) all in one class, as the real ones were; the class is World. The
decoded candidates are then confirmed by one call each, as in Block D, so the reported quantities are the recall of
the decode and the confirmation calls it needs, against bisection, which needs about 2 log2(B) calls per refused call.
Output: results/gt_scale.csv"""
import os, sys, collections
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot
from tasks import load

HERE = pilot.HERE
N, B, TRIALS = 1000, 10, 300
ORDERS = ["shuffled", "shuffled1", "shuffled2", "shuffled3", "sorted", "keyed", "shuffled"]  # last one: the rerun


def decode(refused):
    rem, trig = list(refused), []
    while rem:
        t = collections.Counter(t for g in rem for t in g).most_common(1)[0][0]
        trig.append(t); rem = [g for g in rem if t not in g]
    return set(trig)


def main():
    df = load("agnews")
    calls = []
    for o in ORDERS:
        tids = pilot.order_of(df, o, N, "agnews")
        calls += [frozenset(int(t) for t in tids[i:i + B]) for i in range(0, N, B)]
    sub = np.asarray(pilot.order_of(df, "shuffled", N))
    world = sub[df.gold.values[sub] == 0]
    rng = np.random.default_rng(5)
    rows = []
    for place in ["uniform", "topical"]:
        for q in [1.0, 0.93]:
            for k in [1, 2, 5, 10, 20, 30, 50]:
                rec, exact, conf, nref = [], [], [], []
                for _ in range(TRIALS):
                    T = set(int(t) for t in rng.choice(sub if place == "uniform" else world, k, replace=False))
                    refused = [g for g in calls if g & T and rng.random() < q]
                    D = decode(refused)
                    rec.append(len(D & T) / k); exact.append(D == T); conf.append(len(D)); nref.append(len(refused))
                rows.append(dict(placement=place, q=q, k=k, recall=np.mean(rec), exact=np.mean(exact),
                                 confirm_calls=np.mean(conf), refused_calls=np.mean(nref),
                                 bisection_calls=np.mean(nref) * 2 * np.log2(B)))
    D = pd.DataFrame(rows)
    D.to_csv(f"{HERE}/results/gt_scale.csv", index=False)
    print(D.round(3).to_string())


if __name__ == "__main__":
    main()
