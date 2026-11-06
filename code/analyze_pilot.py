"""Summaries of the pilot (E1 batch curve, E2 repeats, E3 composition/order). Writes results/pilot_*.csv."""
import glob, os, itertools
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(HERE, "results")
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)


def load(exp):
    fs = glob.glob(f"{R}/pilot/{exp}__*.csv")
    return pd.concat([pd.read_csv(f) for f in fs]) if fs else None


def e1():
    d = load("E1")
    if d is None: return
    g = d.groupby(["model", "task", "B", "fmt"])
    s = g.apply(lambda x: pd.Series(dict(
        n=len(x), acc=(x.pred == x.gold).mean(), refused=(x.status == "refused").mean(),
        missing=x.missing.mean(), unparsed=((x.pred == -1) & ~x.missing).mean(),
        usd_1k=x.usd.sum() / len(x) * 1000, in_tok=x.in_tok.mean(), out_tok=x.out_tok.mean(),
        tuples_per_s=len(x) / x.wall.iloc[0]))).reset_index()
    # accuracy on the tuples that every B of the same model answered (removes refusal/missing effects)
    s.to_csv(f"{R}/pilot_E1.csv", index=False)
    print("E1 batch curve\n", s.round(4).to_string())
    # position effect within large batches
    big = d[(d.B >= 20) & (d.status == "ok")].copy()
    big["pos_q"] = pd.cut(big.pos / big.B, [-.01, .25, .5, .75, 1.0], labels=["Q1", "Q2", "Q3", "Q4"])
    p = big.groupby(["model", "B", "pos_q"]).apply(lambda x: pd.Series(dict(
        acc=(x.pred == x.gold).mean(), missing=x.missing.mean()))).unstack("pos_q")
    p.to_csv(f"{R}/pilot_E1_position.csv")
    print("\nE1 position quartile (all tasks pooled)\n", p.round(3).to_string())


def e2():
    d = load("E2")
    if d is None: return
    rows = []
    for (m, t, B), x in d.groupby(["model", "task", "B"]):
        P = x.pivot_table(index="tid", columns="rep", values="pred", aggfunc="first")
        gold = x.groupby("tid").gold.first().loc[P.index].values
        A = P.values
        accs = (A == gold[:, None]).mean(0)
        flip = (np.nanmax(A, 1) != np.nanmin(A, 1)).mean()
        pair = np.mean([(A[:, i] != A[:, j]).mean() for i, j in itertools.combinations(range(A.shape[1]), 2)])
        maj = np.array([np.bincount(r[r >= 0].astype(int), minlength=10).argmax() if (r >= 0).any() else -1 for r in A])
        # prevalence of the positive / first-minority class as a downstream COUNT-style aggregate
        prev = [(A[:, i] == 1).mean() for i in range(A.shape[1])]
        usd = x.groupby("rep").usd.sum().mean() / len(P) * 1000
        rows.append(dict(model=m, task=t, B=B, reps=A.shape[1], n=len(P), acc_mean=accs.mean(), acc_sd=accs.std(ddof=1),
                         acc_min=accs.min(), acc_max=accs.max(), flip_rate=flip, pair_disagree=pair,
                         acc_majority=(maj == gold).mean(), prev1_sd=np.std(prev, ddof=1), usd_1k=usd))
    s = pd.DataFrame(rows); s.to_csv(f"{R}/pilot_E2.csv", index=False)
    print("\nE2 repeats\n", s.round(4).to_string())


def e3():
    d3, d2 = load("E3"), load("E2")
    if d3 is None: return
    rows = []
    # composition: same 500 tuples, B=10, three random orders, rep 0; compare with rep-to-rep change at fixed order
    for (m, t), x in d3[d3.order.str.startswith("shuffled") & (d3.B == 10)].groupby(["model", "task"]):
        base = d2[(d2.model == m) & (d2.task == t) & (d2.B == 10)] if d2 is not None else None
        if base is None or base.empty: continue
        b0 = base[base.rep == 0].set_index("tid").pred
        b1 = base[base.rep == 1].set_index("tid").pred
        o = [x[x.order == k].set_index("tid").pred for k in ["shuffled1", "shuffled2"]]
        o = [s for s in o if len(s)]
        if not o: continue
        comp = np.mean([(s != b0.loc[s.index]).mean() for s in o])
        rows.append(dict(model=m, task=t, change_new_neighbours=comp, change_same_prompt_rerun=(b1 != b0.loc[b1.index]).mean()))
    c = pd.DataFrame(rows); c.to_csv(f"{R}/pilot_E3_composition.csv", index=False)
    print("\nE3 composition\n", c.round(4).to_string())
    rows = []
    for (m, t, B), x in d3[d3.order.isin(["sorted", "shuffled"])].groupby(["model", "task", "B"]):
        r = {"model": m, "task": t, "B": B}
        for o in ["sorted", "shuffled"]:
            y = x[x.order == o]
            if y.empty: continue
            r[f"acc_{o}"] = (y.pred == y.gold).mean()
            K = int(max(y.gold.max(), y.pred.max())) + 1
            pp = np.bincount(y.pred[y.pred >= 0].astype(int), minlength=K) / len(y)
            pg = np.bincount(y.gold.astype(int), minlength=K) / len(y)
            r[f"prev_L1_{o}"] = np.abs(pp - pg).sum()  # error of the GROUP BY count distribution
            r[f"missing_{o}"] = y.missing.mean()
        rows.append(r)
    s = pd.DataFrame(rows); s.to_csv(f"{R}/pilot_E3_order.csv", index=False)
    print("\nE3 physical order\n", s.round(4).to_string())


if __name__ == "__main__":
    e1(); e2(); e3()
