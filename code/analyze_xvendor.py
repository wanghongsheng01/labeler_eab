"""Labelers added on 2026-11-06 to answer the vendor-coverage limitation: GPT-6-Sol and Claude-Opus-5.5 (through the
company gateway) and the open-weight Qwen2.5-3B-Instruct served by llama.cpp on a CPU server (greedy decoding, one
request at a time). Same quantities as the main analysis, B = 10:
  comp   composition sensitivity: canonical vs. every other random order that was run (cluster bootstrap over calls)
  rerun  rerun flip rate of the canonical calls
  pos    label change when the canonical calls are only permuted (perm1)
  sorted / keyed  accuracy change against the mean of the random orders, z-test clustered by call
Writes results/xv_summary.csv."""
import glob, os, sys
import numpy as np, pandas as pd
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot
from tasks import load

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = f"{HERE}/results/main"
MODELS = ("gpt-6-sol", "claude-opus-5.5", "qwen2.5-3b")
rng = np.random.default_rng(0)


def files(model):
    return [f for f in glob.glob(f"{M}/*__{model}__*.csv") if os.path.basename(f).split("__")[0] in ("B10", "B10s", "R10", "P", "L")]


def boot(v, cl, n=2000):
    df = pd.DataFrame(dict(v=v, c=cl)).groupby("c").v.agg(["sum", "count"])
    s, c = df["sum"].values, df["count"].values
    idx = rng.integers(0, len(s), (n, len(s)))
    b = s[idx].sum(1) / c[idx].sum(1)
    return np.nanmean(v), np.percentile(b, 2.5), np.percentile(b, 97.5)


def zdiff(ok_o, base, cl):
    d = ok_o.astype(float) - base; m = d.mean()
    s = pd.Series(d - m).groupby(cl).sum().values
    se = np.sqrt((s ** 2).sum()) / len(d)
    return 100 * m, 2 * norm.sf(abs(m / se)) if se > 0 else 1.0


def main():
    rows = []
    for model in MODELS:
        fs = files(model)
        if not fs:
            continue
        d = pd.concat([pd.read_csv(f) for f in fs])
        d = d[d.B == 10]
        for t, x in d.groupby("task"):
            df = load(t)
            run0 = x[x.rep == 0]
            piv = run0.pivot_table(index="tid", columns="order", values="pred", aggfunc="first")
            gold = run0.groupby("tid").gold.first().loc[piv.index]
            n = len(piv)
            canon = piv["shuffled"]
            cl = pd.Series({tid: i // 10 for i, tid in enumerate(pilot.order_of(df, "shuffled", n))}).reindex(piv.index).values
            rnd = [o for o in ["shuffled", "shuffled1", "shuffled2", "shuffled3"] if o in piv]
            ok = canon >= 0  # valid labels only (unparsed, refused and failed answers are not labels)
            fl = lambda a, b: np.where((a >= 0) & (b >= 0), (a != b).astype(float), np.nan)
            with np.errstate(all="ignore"):
                comp_v = np.nanmean([fl(piv[o].values, canon.values) for o in rnd[1:]], axis=0)
            r = dict(model=model, task=t, n=n, acc=float((canon == gold).mean()), random_orders=len(rnd))
            r["comp"], r["comp_lo"], r["comp_hi"] = boot(comp_v, cl)
            rr = x[x.rep == 1]
            if len(rr):
                rv = rr.set_index("tid").pred.reindex(piv.index)
                rer = fl(rv.values, canon.values)
                r["rerun"], r["rerun_lo"], r["rerun_hi"] = boot(rer, cl)
                r["diff"], r["diff_lo"], r["diff_hi"] = boot(comp_v - rer, cl)
            if "perm1" in piv:
                with np.errstate(all="ignore"):
                    pv = np.nanmean([fl(piv[o].values, canon.values) for o in ["perm1", "perm2"] if o in piv], axis=0)
                r["pos"] = np.nanmean(pv)
            base = np.mean([(piv[o] == gold).values for o in rnd], axis=0)
            for o in ["sorted", "keyed"]:
                if o in piv:
                    clo = pd.Series({tid: i // 10 for i, tid in enumerate(pilot.order_of(df, o, n))}).reindex(piv.index).values
                    r[f"{o}_pp"], r[f"{o}_p"] = zdiff((piv[o] == gold).values, base, clo)
            r["refused"] = float((run0.status == "refused").mean())
            r["served"] = ",".join(sorted(set(run0.served.dropna().astype(str))))
            rows.append(r)
    D = pd.DataFrame(rows)
    for o in ["sorted", "keyed"]:
        p = D[f"{o}_p"].values; k = np.argsort(p); adj = np.empty(len(p))
        adj[k] = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[k])); D[f"{o}_p_holm"] = adj
    D.to_csv(f"{HERE}/results/xv_summary.csv", index=False)
    print(D.round(4).to_string())


if __name__ == "__main__":
    main()
