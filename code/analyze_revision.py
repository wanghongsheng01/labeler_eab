"""Revised statistics after the simulated review (2026-11-05). Replaces the tests of analyze_main.py/analyze_main_a.py
in the paper:
- H1: per-tuple composition sensitivity (mean of the three new-neighbour comparisons) and rerun flips, with
  cluster bootstrap over canonical batches; position-only control (same members, permuted positions).
- H2: order effect against the mean of the four random compositions, cluster-robust z-test over the batches of the
  clustered order, Holm over all order contrasts; also the original McNemar against the canonical run.
- H4: format and context effects with Holm inside each family; context effects cluster-robust over batches.
- Contrast effect: placebo randomization null, Holm per cell, robustness variants.
- Query level: class-size-weighted filter Jaccard; composition spread vs. binomial sampling error.
- Position: task-stratified chi-square.
- New blocks: dose response (N), prompt remedy (I), late rerun (R2).
Writes results/rev_*.csv and prints a summary."""
import glob, os, sys
import numpy as np, pandas as pd
from scipy.stats import binomtest, chi2, chi2_contingency, norm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot
from tasks import load

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = f"{HERE}/results"
RND = ["shuffled", "shuffled1", "shuffled2", "shuffled3"]
rng = np.random.default_rng(0)


PRIMARY = ("ds-flash", "ds-pro", "glm-5.3-flash", "kimi-k2.6")  # labelers of the analysis plan; others: analyze_xvendor.py


def L(pattern):
    fs = [f for f in glob.glob(f"{R}/main/{pattern}") if os.path.basename(f).split("__")[1] in PRIMARY]
    return pd.concat([pd.read_csv(f) for f in fs]) if fs else pd.DataFrame()


def holm(p):
    p = np.asarray(p, float); k = len(p); o = np.argsort(p); adj = np.empty(k)
    adj[o] = np.minimum(1, np.maximum.accumulate((k - np.arange(k)) * p[o]))
    return adj


def cluster_mean_ci(v, cl, n=2000):
    """Mean of v with a cluster bootstrap CI (clusters resampled with replacement)."""
    df = pd.DataFrame(dict(v=v, c=cl)).groupby("c").v.agg(["sum", "count"])
    s, c = df["sum"].values, df["count"].values
    idx = rng.integers(0, len(s), (n, len(s)))
    m = s[idx].sum(1) / c[idx].sum(1)
    return v.mean(), *np.percentile(m, [2.5, 97.5])


def cluster_z(v, cl):
    """Cluster-robust z-test of mean(v) = 0."""
    g = pd.DataFrame(dict(v=v, c=cl)).groupby("c").v.agg(["sum", "count"])
    m = g["sum"].sum() / g["count"].sum()
    resid = g["sum"] - m * g["count"]
    G = len(g); se = np.sqrt(G / (G - 1) * (resid ** 2).sum()) / g["count"].sum()
    z = m / se if se > 0 else 0.0
    return m, se, 2 * norm.sf(abs(z))


def batch_id(df, order, n, B):
    t = pilot.order_of(df, order, n)
    return pd.Series(np.repeat(np.arange((len(t) + B - 1) // B), B)[:len(t)], index=t)


Bk = pd.concat([L("B__*.csv"), L("B10__*.csv")])
Rr = pd.concat([L("R__*.csv"), L("R10__*.csv")])
Pm = L("P__*.csv")


# --- H1 -----------------------------------------------------------------------------------------------------------
def h1():
    rows = []
    for (m, t, b), x in Bk.groupby(["model", "task", "B"]):
        df = load(t)
        piv = x.pivot_table(index="tid", columns="order", values="pred", aggfunc="first")
        n = len(piv); ok = piv["shuffled"] >= -1
        comp = np.mean([((piv[o] != piv["shuffled"]) & (piv[o] >= -1)).values for o in RND[1:]], axis=0)
        cl = batch_id(df, "shuffled", n, b).reindex(piv.index).values
        row = dict(model=m, task=t, B=b, n=n)
        row["comp"], row["comp_lo"], row["comp_hi"] = cluster_mean_ci(comp[ok.values], cl[ok.values])
        rr = Rr[(Rr.model == m) & (Rr.task == t) & (Rr.B == b)]
        if len(rr):
            r1 = rr.set_index("tid").pred.reindex(piv.index)
            rer = ((r1 != piv["shuffled"]) & (r1 >= -1)).values.astype(float)
            row["rerun"], row["rerun_lo"], row["rerun_hi"] = cluster_mean_ci(rer[ok.values], cl[ok.values])
            d = comp - rer
            row["diff"], row["diff_lo"], row["diff_hi"] = cluster_mean_ci(d[ok.values], cl[ok.values])
        pm = Pm[(Pm.model == m) & (Pm.task == t) & (Pm.B == b)] if len(Pm) else Pm
        if len(pm):
            pp = pm.pivot_table(index="tid", columns="order", values="pred", aggfunc="first").reindex(piv.index)
            pos = np.mean([((pp[o] != piv["shuffled"]) & (pp[o] >= -1)).values for o in ["perm1", "perm2"]], axis=0)
            row["pos"], row["pos_lo"], row["pos_hi"] = cluster_mean_ci(pos[ok.values], cl[ok.values])
        rows.append(row)
    C = pd.DataFrame(rows); C.to_csv(f"{R}/rev_composition.csv", index=False)
    h = C.dropna(subset=["rerun"])
    print(f"H1: comp > rerun, cluster-bootstrap CI of the difference above 0 in {(h.diff_lo > 0).sum()}/{len(h)} cells; "
          f"point estimate larger in {(h.comp > h.rerun).sum()}/{len(h)}; disjoint CIs {(h.comp_lo > h.rerun_hi).sum()}/{len(h)}")
    if "pos" in C:
        p = C.dropna(subset=["pos"])
        print(f"position-only control: pos < comp in {(p.pos < p.comp).sum()}/{len(p)}; pos CI above rerun CI in "
              f"{(p.pos_lo > p.rerun_hi).sum()}/{len(p)}; share of comp explained by position (median) "
              f"{np.median(p.pos / p.comp):.2f}")
        print(p[["model", "task", "comp", "pos", "rerun"]].round(4).to_string())
    print("ranges B=10: comp", C.query("B==10").comp.min().round(4), C.query("B==10").comp.max().round(4),
          "rerun", h.query("B==10").rerun.min().round(4), h.query("B==10").rerun.max().round(4))
    return C


# --- H2 -----------------------------------------------------------------------------------------------------------
def h2():
    rows = []
    for (m, t, b), x in Bk.groupby(["model", "task", "B"]):
        df = load(t)
        piv = x.pivot_table(index="tid", columns="order", values="pred", aggfunc="first")
        gold = x.groupby("tid").gold.first().loc[piv.index]
        base = np.mean([(piv[o] == gold).values for o in RND], axis=0)
        for o in ["sorted", "keyed"]:
            c = (piv[o] == gold).values.astype(float)
            cl = batch_id(df, o, len(piv), b).reindex(piv.index).values
            mu, se, p = cluster_z(c - base, cl)
            a, cc = piv["shuffled"] == gold, piv[o] == gold
            n01, n10 = int((a & ~cc).sum()), int((~a & cc).sum())
            rows.append(dict(model=m, task=t, B=b, order=o, acc_base=base.mean(), acc_order=c.mean(), diff_pp=100 * mu,
                             se_pp=100 * se, p=p, diff_canon_pp=100 * (cc.mean() - a.mean()),
                             p_mcnemar=binomtest(n01, n01 + n10).pvalue if n01 + n10 else 1.0))
    O = pd.DataFrame(rows); O["p_holm"] = holm(O.p); O["p_mcnemar_holm"] = holm(O.p_mcnemar)
    O.to_csv(f"{R}/rev_order.csv", index=False)
    for o in ["sorted", "keyed"]:
        s = O[O.order == o]
        print(f"H2 {o}: cluster-robust vs mean of 4 random, Holm p<.05 in {(s.p_holm < .05).sum()}/{len(s)} "
              f"(McNemar vs canonical, Holm: {(s.p_mcnemar_holm < .05).sum()}/{len(s)}); range of significant "
              f"{s[s.p_holm < .05].diff_pp.min():.2f}..{s[s.p_holm < .05].diff_pp.max():.2f} pp")
    nt = O[(O.order == "sorted") & (O.task != "imdb")]
    print("H2 per labeler (non-trivial tasks, B=10, Holm<.05):",
          nt[nt.B == 10].groupby("model").apply(lambda z: int((z.p_holm < .05).sum())).to_dict(),
          "DS-Flash B=50:", int((nt[nt.B == 50].p_holm < .05).sum()))
    print(O[O.order == "keyed"].sort_values("p").head(4)[["model", "task", "B", "diff_pp", "p", "p_holm"]].round(4).to_string())
    return O


# --- H4 -----------------------------------------------------------------------------------------------------------
def h4():
    d = pd.concat([L("A__*.csv"), L("A4__*.csv")])
    rows = []
    for (m, t), x in d.groupby(["model", "task"]):
        df = load(t)
        get = lambda f, b: x[(x.fmt == f) & (x.B == b)].set_index("tid")
        s, b1 = get("single", 1), get("batch", 1)
        idx = s.index.intersection(b1.index)
        cs, cb = (s.pred.loc[idx] == s.gold.loc[idx]).values, (b1.pred.loc[idx] == b1.gold.loc[idx]).values
        n01, n10 = int((cs & ~cb).sum()), int((~cs & cb).sum())
        r = dict(model=m, task=t, format_pp=100 * (cb.mean() - cs.mean()), format_p=binomtest(n01, n01 + n10).pvalue)
        for B in (10, 50):
            bb = get("batch", B)
            if not len(bb): continue
            v = (bb.pred.loc[idx] == bb.gold.loc[idx]).values.astype(float) - cb
            cl = batch_id(df, "shuffled", len(idx), B).reindex(idx).values
            mu, se, p = cluster_z(v, cl)
            r[f"context{B}_pp"], r[f"context{B}_p"] = 100 * mu, p
        rows.append(r)
    D = pd.DataFrame(rows)
    for c in ["format_p", "context10_p", "context50_p"]:
        D[c + "_holm"] = holm(D[c])
    D.to_csv(f"{R}/rev_decomposition.csv", index=False)
    for c in ["format", "context10", "context50"]:
        print(f"H4 {c}: p<.05 {(D[c + '_p'] < .05).sum()}/{D[c + '_p'].notna().sum()}, Holm {(D[c + '_p_holm'] < .05).sum()}/{D[c + '_p'].notna().sum()}; "
              f"significant after Holm: {D[D[c + '_p_holm'] < .05][['model', 'task', c + '_pp']].round(2).values.tolist()}")
    return D


# --- contrast effect ----------------------------------------------------------------------------------------------
def contrast(n_placebo=200):
    out, null = [], np.zeros(n_placebo)
    nn = np.zeros(n_placebo)
    for (m, t), x in Bk[Bk.B == 10].groupby(["model", "task"]):
        df = load(t); gold = df.gold.values; K = gold.max() + 1
        piv = x.pivot_table(index="tid", columns="order", values="pred", aggfunc="first")
        n = len(piv); tids = piv.index.values

        def shares(order_tids):  # array [tid, class]: class shares among the tuple's 9 neighbours
            order_tids = np.asarray(order_tids)
            bid = np.repeat(np.arange(n // 10 + 1), 10)[:n]
            oh = np.eye(K)[gold[order_tids]]
            cnt = np.zeros((bid.max() + 1, K)); np.add.at(cnt, bid, oh)
            sh = np.full((len(gold), K), np.nan); sh[order_tids] = (cnt[bid] - oh) / 9.0
            return sh

        S = {o: shares(pilot.order_of(df, o, n)) for o in RND}
        ch = [(o, u, piv.at[u, o]) for o in RND[1:]
              for u in tids[((piv[o] != piv["shuffled"]) & (piv[o] >= 0) & (piv["shuffled"] >= 0)).values]]
        if not ch: continue
        o_ = np.array([c[0] for c in ch]); u_ = np.array([c[1] for c in ch], int); b_ = np.array([c[2] for c in ch], int)

        def diff(SS):
            d = np.empty(len(u_))
            for o in RND[1:]:
                k = o_ == o
                d[k] = SS[o][u_[k], b_[k]] - SS["shuffled"][u_[k], b_[k]]
            return d

        d = diff(S)
        pos, neg = int((d > 0).sum()), int((d < 0).sum())
        out.append(dict(model=m, task=t, changed=len(d), pos=pos, neg=neg, mean_shift=d.mean(),
                        p=binomtest(pos, pos + neg).pvalue if pos + neg else 1.0))
        for k in range(n_placebo):  # placebo: same label changes, neighbour sets from independent random orders
            pr = np.random.default_rng(1000 + k)
            dp = diff({o: shares(pr.permutation(tids)) for o in RND})
            null[k] += (dp < 0).sum(); nn[k] += (dp != 0).sum()
    D = pd.DataFrame(out); D["p_holm"] = holm(D.p)
    D.to_csv(f"{R}/rev_contrast.csv", index=False)
    P, N = int(D.pos.sum()), int(D.neg.sum())
    obs = N / (P + N); nullf = null / nn
    z = (obs - nullf.mean()) / nullf.std()
    print(f"contrast: away {N} vs toward {P} (share away {obs:.4f}); placebo null share away {nullf.mean():.4f} "
          f"+- {nullf.std():.4f} (max {nullf.max():.4f}); z = {z:.1f}; placebo p < {1 / (len(nullf) + 1):.3f}")
    print(f"cells negative {int((D.mean_shift < 0).sum())}/{len(D)}; Holm-significant negative "
          f"{int(((D.p_holm < .05) & (D.mean_shift < 0)).sum())}, positive {int(((D.p_holm < .05) & (D.mean_shift > 0)).sum())}")
    return D


# --- query level --------------------------------------------------------------------------------------------------
def querylevel():
    rows = []
    for (m, t, b), x in Bk.groupby(["model", "task", "B"]):
        piv = x.pivot_table(index="tid", columns="order", values="pred", aggfunc="first")
        gold = x.groupby("tid").gold.first().loc[piv.index]
        K = int(gold.max()) + 1; n = len(piv)
        wj, w, spread, se = 0.0, 0.0, [], []
        for c in range(K):
            S0 = set(piv.index[piv["shuffled"] == c])
            for o in RND[1:]:
                S = set(piv.index[piv[o] == c])
                if S0 | S:
                    wj += len(S0 & S); w += len(S0 | S)
            cs = np.array([(piv[o] == c).mean() for o in RND])
            spread.append(cs.std(ddof=1)); p = cs.mean(); se.append(np.sqrt(p * (1 - p) / n))
        rows.append(dict(model=m, task=t, B=b, jac_weighted=wj / w, comp_sd_over_sampling_se=np.mean(np.array(spread) / np.array(se))))
    Q = pd.DataFrame(rows); Q.to_csv(f"{R}/rev_querylevel.csv", index=False)
    print("weighted filter Jaccard (B=10): min", Q.query("B==10").jac_weighted.min().round(3),
          "median", Q.query("B==10").jac_weighted.median().round(3))
    print("composition SD of class shares / binomial sampling SE at the same n: median",
          Q.comp_sd_over_sampling_se.median().round(2), "range", Q.comp_sd_over_sampling_se.min().round(2),
          Q.comp_sd_over_sampling_se.max().round(2))
    return Q


def position():
    d = pd.concat([L("A__*.csv")])
    for b in (50, 100):
        x = d[(d.fmt == "batch") & (d.B == b)].copy()
        x["q"] = pd.cut(x.pos, [-1] + [b * k / 5 - 1 for k in range(1, 6)], labels=[1, 2, 3, 4, 5])
        x["ok"] = x.pred == x.gold
        stat = sum(chi2_contingency(pd.crosstab(g.q, g.ok))[0] for _, g in x.groupby("task"))
        acc = x.groupby("q").ok.mean()
        print(f"position B={b}: task-stratified chi2 = {stat:.1f}, df = 28, p = {chi2.sf(stat, 28):.3f}; "
              f"pooled quintile accuracy {acc.round(4).tolist()}")


# --- new blocks ---------------------------------------------------------------------------------------------------
def new_blocks():
    N, I, R2 = L("N__*.csv"), L("I__*.csv"), L("R2__*.csv")
    from scipy.stats import spearmanr
    if len(N):
        rows = []
        for (m, t), x in N.groupby(["model", "task"]):
            df = load(t)
            b = Bk[(Bk.model == m) & (Bk.task == t) & (Bk.B == 10)]
            piv = b.pivot_table(index="tid", columns="order", values="pred", aggfunc="first")
            gold = b.groupby("tid").gold.first().loc[piv.index]
            base = np.mean([(piv[o] == gold).values for o in RND], axis=0)
            for o, y in x.groupby("order"):
                yy = y.set_index("tid").reindex(piv.index)
                c = (yy.pred == gold).values.astype(float)
                tl = pilot.order_of(df, o, len(piv))
                rho = spearmanr(df.gold.values[tl], np.arange(len(tl)))[0]
                mu, se, p = cluster_z(c - base, batch_id(df, o, len(piv), 10).reindex(piv.index).values)
                rows.append(dict(model=m, task=t, order=o, rho=rho, diff_pp=100 * mu, se_pp=100 * se, p=p))
            for o in ["sorted", "keyed"]:
                c = (piv[o] == gold).values.astype(float)
                tl = pilot.order_of(df, o, len(piv))
                mu, se, p = cluster_z(c - base, batch_id(df, o, len(piv), 10).reindex(piv.index).values)
                rows.append(dict(model=m, task=t, order=o, rho=spearmanr(df.gold.values[tl], np.arange(len(tl)))[0],
                                 diff_pp=100 * mu, se_pp=100 * se, p=p))
        Dn = pd.DataFrame(rows); Dn.to_csv(f"{R}/rev_dose.csv", index=False)
        print("dose response (DS-Flash, B=10):\n", Dn.pivot(index="task", columns="order", values="diff_pp").round(2).to_string())
        print(Dn.pivot(index="task", columns="order", values="rho").round(2).to_string())
    if len(I):
        rows = []
        for (m, t), x in I.groupby(["model", "task"]):
            b = Bk[(Bk.model == m) & (Bk.task == t) & (Bk.B == 10)]
            g = b.groupby("tid").gold.first()
            acc = lambda y: (y.set_index("tid").pred.reindex(g.index) == g).mean()
            rows.append(dict(model=m, task=t,
                             std_canon=acc(b[b.order == "shuffled"]), std_sorted=acc(b[b.order == "sorted"]),
                             ind_canon=acc(x[x.order == "shuffled"]), ind_sorted=acc(x[x.order == "sorted"])))
        Di = pd.DataFrame(rows)
        Di["std_gap_pp"] = 100 * (Di.std_sorted - Di.std_canon); Di["ind_gap_pp"] = 100 * (Di.ind_sorted - Di.ind_canon)
        Di.to_csv(f"{R}/rev_remedy.csv", index=False)
        print("prompt remedy:\n", Di.round(3).to_string())
    if len(R2):
        rows = []
        for (m, t), x in R2.groupby(["model", "task"]):
            b = Bk[(Bk.model == m) & (Bk.task == t) & (Bk.B == 10) & (Bk.order == "shuffled")].set_index("tid").pred
            r1 = Rr[(Rr.model == m) & (Rr.task == t) & (Rr.B == 10)].set_index("tid")
            y = x.set_index("tid")
            ok = b >= -1
            rows.append(dict(model=m, task=t, rerun_same_day=((r1.pred.reindex(b.index) != b) & ok).mean(),
                             rerun_next_day=((y.pred.reindex(b.index) != b) & ok).mean(),
                             cache_hit_same_day=(r1.cache_hit / r1.in_tok).mean(), cache_hit_next_day=(y.cache_hit / y.in_tok).mean()))
        D2 = pd.DataFrame(rows); D2.to_csv(f"{R}/rev_rerun2.csv", index=False)
        print("late rerun:\n", D2.round(4).to_string())


if __name__ == "__main__":
    which = sys.argv[1:] or ["h1", "h2", "h4", "contrast", "querylevel", "position", "new_blocks"]
    for w in which:
        print(f"\n===== {w}"); globals()[w]()
