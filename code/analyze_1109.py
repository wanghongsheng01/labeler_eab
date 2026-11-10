"""No-call analyses added on 2026-11-09 (Sections V-D, V-F and VI).

two_groupings: in each of the 28 cells with B = 10, composition sensitivity from every pair of the four random
               groupings of Block B against the estimate from all four (support for L2).
glm_reasoning: GLM rerun flip rate of the canonical calls split by whether either copy produced reasoning tokens
               (needs results/main/calls_main.jsonl.gz from export_calls.py).
block_c:       Block C (controlled neighbors) for every labeler that ran it: accuracy under same-class, other-class
               and random neighbors, and exact McNemar tests of same and other against random on the same targets.
Output: results/l2_two_groupings.csv, results/glm_reasoning_rerun.csv, results/block_c_summary.csv"""
import glob, gzip, itertools, json, os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = f"{HERE}/results/main"
PLANNED = ["ds-flash", "ds-pro", "glm-5.3-flash", "kimi-k2.6"]
TASKS7 = ["imdb", "agnews", "yelp", "hate", "offensive", "emotion", "amazon"]


def two_groupings():
    rows = []
    for m in PLANNED:
        for f in glob.glob(f"{M}/B10__{m}__*.csv") + glob.glob(f"{M}/B__{m}__*.csv"):
            d = pd.read_csv(f); d = d[(d.B == 10) & (d.rep == 0)]
            piv = d.pivot_table(index="tid", columns="order", values="pred", aggfunc="first")
            R = ["shuffled", "shuffled1", "shuffled2", "shuffled3"]
            if not all(o in piv for o in R):
                continue

            def cs(a, b):
                x, y = piv[a].values, piv[b].values; ok = (x >= 0) & (y >= 0)
                return np.mean(x[ok] != y[ok])
            full = np.mean([cs("shuffled", o) for o in R[1:]])
            dev = max(abs(cs(a, b) - full) for a, b in itertools.combinations(R, 2))
            rows.append(dict(model=m, task=f.split("__")[-1][:-4], cs_four=full, max_abs_dev_pair=dev))
    D = pd.DataFrame(rows); D.to_csv(f"{HERE}/results/l2_two_groupings.csv", index=False)
    print(f"cells {len(D)}; largest deviation of a pair {D.max_abs_dev_pair.max():.4f}, median {D.max_abs_dev_pair.median():.4f}")


def glm_reasoning():
    calls = {}
    for l in gzip.open(f"{M}/calls_main.jsonl.gz", "rt"):
        d = json.loads(l)
        if d["model"] == "glm-5.3-flash" and d["B"] == 10 and d["order"] == "shuffled" and d["block"] in ("B10", "R10"):
            calls[(d["task"], d["block"], d["call"])] = d
    rows = []
    for t in TASKS7:
        a = pd.read_csv(f"{M}/B10__glm-5.3-flash__{t}.csv")
        a = a[(a.order == "shuffled") & (a.B == 10) & (a.rep == 0)].set_index("tid").pred
        b = pd.read_csv(f"{M}/R10__glm-5.3-flash__{t}.csv"); b = b[b.order == "shuffled"].set_index("tid").pred
        for (task, blk, c), d0 in calls.items():
            if task != t or blk != "B10" or (t, "R10", c) not in calls:
                continue
            d1 = calls[(t, "R10", c)]
            x, y = a.reindex(d0["tids"]).values, b.reindex(d0["tids"]).values
            ok = (x >= 0) & (y >= 0)
            r0, r1 = (d0.get("reasoning_tokens") or 0) > 0, (d1.get("reasoning_tokens") or 0) > 0
            rows.append(dict(task=t, reasoning="both" if r0 and r1 else "none" if not (r0 or r1) else "one",
                             labels=int(ok.sum()), flips=int((x != y)[ok].sum())))
    D = pd.DataFrame(rows).groupby("reasoning")[["labels", "flips"]].sum()
    D["rerun_flip_rate"] = D.flips / D.labels
    D.to_csv(f"{HERE}/results/glm_reasoning_rerun.csv"); print(D)


def block_c():
    from scipy.stats import binomtest
    rows = []
    for f in sorted(glob.glob(f"{M}/C__*.csv")):
        d = pd.read_csv(f)
        piv = d.pivot_table(index="tid", columns="cond", values="pred", aggfunc="first")
        gold = d.groupby("tid").gold.first().reindex(piv.index).values
        ok = {c: piv[c].values == gold for c in ["same", "other", "mixed"]}
        r = dict(model=d.model.iloc[0], task=d.task.iloc[0], n=len(piv))
        for c in ["same", "other", "mixed"]:
            r[f"acc_{c}"] = ok[c].mean()
        for c in ["same", "other"]:
            a, b = ok["mixed"], ok[c]
            n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
            r[f"{c}_minus_mixed_pp"] = 100 * (b.mean() - a.mean())
            r[f"p_{c}"] = binomtest(n01, n01 + n10).pvalue if n01 + n10 else 1.0
        rows.append(r)
    D = pd.DataFrame(rows); D.to_csv(f"{HERE}/results/block_c_summary.csv", index=False); print(D.round(4).to_string())


if __name__ == "__main__":
    two_groupings(); glm_reasoning(); block_c()
