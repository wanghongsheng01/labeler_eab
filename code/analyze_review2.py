"""Analyses added after the second internal review (2026-11-06). Writes results/rev2_*.csv and prints a summary.

- template:   label disagreement between the single and the batch (B=1) template, both with one text and no neighbours
- h2_orig:    H2 under the originally planned test (exact McNemar against the canonical order, Holm over 70 contrasts)
- contrast:   per-task share of label changes that move away from the neighbours
- gt_subsets: precision and recall of the group-testing decode when only k of the seven runs are available
- heldout:    DS-Flash results on the 1,000 tuples per dataset that the pilot study never used
- glm:        share of GLM-5.3-Flash calls with reasoning tokens and the reasoning share of its output tokens
- block_k:    DS-Flash on a table clustered on DS-Pro's predicted label (Block K)
- block_x:    DS-Flash with a delimited template (Block X)
"""
import collections, glob, gzip, itertools, json, os, sys
import numpy as np, pandas as pd
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot
from tasks import load

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R, M = f"{HERE}/results", f"{HERE}/results/main"
RND = ["shuffled", "shuffled1", "shuffled2", "shuffled3"]
TASKS7 = ["agnews", "amazon", "emotion", "hate", "imdb", "offensive", "yelp"]


def L(pattern):
    fs = glob.glob(f"{M}/{pattern}")
    return pd.concat([pd.read_csv(f) for f in fs]) if fs else pd.DataFrame()


def cluster_diff(d, cl):
    """Mean of d and its cluster-robust standard error (clusters = calls)."""
    d = np.asarray(d, float); m = d.mean()
    s = pd.Series(d - m).groupby(np.asarray(cl)).sum().values
    return m, np.sqrt((s ** 2).sum()) / len(d)


def template():
    rows = []
    for f in sorted(glob.glob(f"{M}/A__ds-flash__*.csv") + glob.glob(f"{M}/A4__ds-pro__*.csv")):
        d = pd.read_csv(f)
        s = d[(d.fmt == "single")].set_index("tid").pred
        b = d[(d.fmt == "batch") & (d.B == 1)].set_index("tid").pred.reindex(s.index)
        ok = (s >= 0) & (b >= 0)
        rows.append(dict(model=d.model.iloc[0], task=d.task.iloc[0], n=int(ok.sum()), template_flip=(s[ok] != b[ok]).mean()))
    D = pd.DataFrame(rows); D.to_csv(f"{R}/rev2_template.csv", index=False)
    C = pd.read_csv(f"{R}/rev_composition.csv").query("B == 10")[["model", "task", "comp", "pos", "rerun"]]
    D = D.merge(C, on=["model", "task"])
    print(D.round(4).to_string())
    for m, x in D.groupby("model"):
        print(f"{m}: template flip {x.template_flip.min():.3f}-{x.template_flip.max():.3f}; "
              f"cells where template flip > composition sensitivity: {int((x.template_flip > x.comp).sum())}/{len(x)}")
    return D


def h2_orig():
    o = pd.read_csv(f"{R}/rev_order.csv")
    s = o[(o.order == "sorted") & (o.B == 10) & (o.task != "imdb")]
    print("H2 per labeler (non-trivial tasks, B=10): revised test", s.groupby("model").apply(lambda z: int((z.p_holm < .05).sum())).to_dict(),
          "original McNemar test", s.groupby("model").apply(lambda z: int((z.p_mcnemar_holm < .05).sum())).to_dict())
    so = o[o.order == "sorted"]
    print("sorted cells significant: revised", int((so.p_holm < .05).sum()), "original", int((so.p_mcnemar_holm < .05).sum()), "of", len(so))


def contrast():
    c = pd.read_csv(f"{R}/rev_contrast.csv")
    g = c.groupby("task")[["pos", "neg"]].sum(); g["share_away"] = g.neg / (g.pos + g.neg)
    print(g.round(3).to_string())
    a = g.loc["agnews"]; r = g.drop("agnews").sum()
    print(f"AG News: {a.neg:.0f} away vs {a.pos:.0f} toward (share {a.neg / (a.neg + a.pos):.3f}); "
          f"other six tasks: {r.neg:.0f} vs {r.pos:.0f} (share {r.neg / (r.neg + r.pos):.3f})")
    g.to_csv(f"{R}/rev2_contrast_by_task.csv")


def gt_subsets():
    D = pd.read_csv(f"{M}/D__glm-5.3-flash.csv")
    rows = []
    for task in ["agnews", "hate"]:
        df = load(task)
        x = D[D.task == task]
        truth = set(x.tid[(x.alone_rep0 == "refused") & (x.alone_rep1 == "refused")])
        runs = {}
        for f in glob.glob(f"{M}/B10__glm-5.3-flash__{task}.csv") + glob.glob(f"{M}/R10__glm-5.3-flash__{task}.csv"):
            for o, x in pd.read_csv(f).groupby("order"):
                if o in RND + ["sorted", "keyed"]:
                    runs[os.path.basename(f)[0] + ":" + o] = x.set_index("tid").pred
        names = sorted(runs)
        bat = {}
        for run in names:
            tids = pilot.order_of(df, run.split(":")[1], 1000)
            bat[run] = [(tids[i:i + 10], bool((runs[run].loc[tids[i:i + 10]] == -2).all())) for i in range(0, 1000, 10)]
        for k in range(1, len(names) + 1):
            P, Rc = [], []
            for sub in itertools.combinations(names, k):
                ref = [g for r in sub for g, x in bat[r] if x]
                rem, trig = list(ref), []
                while rem:
                    t = collections.Counter(t for g in rem for t in g).most_common(1)[0][0]
                    trig.append(t); rem = [g for g in rem if t not in g]
                hit = len(set(trig) & truth)
                P.append(hit / len(trig) if trig else 1.0); Rc.append(hit / len(truth))
            rows.append(dict(task=task, k=k, subsets=len(P), precision=np.mean(P), recall=np.mean(Rc),
                             exact=np.mean([p == 1 and r == 1 for p, r in zip(P, Rc)])))
    G = pd.DataFrame(rows); G.to_csv(f"{R}/rev2_gt_subsets.csv", index=False)
    print(G.round(3).to_string())


def heldout():
    B = L("B__ds-flash__*.csv"); Rr = L("R__ds-flash__*.csv")
    rows = []
    for t in TASKS7:
        df = load(t)
        x = B[(B.task == t) & (B.B == 10)]
        n = x.tid.nunique()
        seen = set(np.random.default_rng(0).permutation(len(df))[:min(1000, n)])
        piv = x.pivot_table(index="tid", columns="order", values="pred", aggfunc="first")
        gold = x.groupby("tid").gold.first()
        rr = Rr[(Rr.task == t) & (Rr.B == 10)].set_index("tid").pred.reindex(piv.index)
        for part, idx in [("pilot", [i for i in piv.index if i in seen]), ("held-out", [i for i in piv.index if i not in seen])]:
            if not idx: continue
            p, gd = piv.loc[idx], gold.loc[idx]
            comp = np.mean([(p[o] != p["shuffled"]).mean() for o in RND[1:]])
            acc_r = np.mean([(p[o] == gd).mean() for o in RND])
            rows.append(dict(task=t, part=part, n=len(idx), comp=comp, rerun=(rr.loc[idx] != p["shuffled"]).mean(),
                             sorted_pp=100 * ((p["sorted"] == gd).mean() - acc_r)))
    H = pd.DataFrame(rows); H.to_csv(f"{R}/rev2_heldout.csv", index=False)
    print(H.round(4).to_string())


def glm():
    n = r = out = rt = 0
    for l in open(f"{HERE}/cache/usage.log"):
        d = json.loads(l)
        if d.get("m") != "glm-5.3-flash" or d.get("status") != "ok": continue
        k = d.get("reasoning_tokens") or 0
        n += 1; r += k > 0; out += d.get("completion_tokens", 0); rt += k
    print(f"GLM-5.3-Flash answered calls {n}: with reasoning {r / n:.3f}; reasoning share of output tokens {rt / out:.3f}")


def block_k():
    K = L("K__ds-flash__*.csv"); B = L("B__ds-flash__*.csv")
    rows = []
    for t, x in K.groupby("task"):
        df = load(t)
        b = B[(B.task == t) & (B.B == 10)]
        idx = x.tid.values
        gold = df.gold.values[idx]
        acc_r = np.mean([b[b.order == o].set_index("tid").pred.reindex(idx).values == gold for o in RND], axis=0)
        okk = x.set_index("tid").pred.reindex(idx).values == gold
        okk_sorted = b[b.order == "sorted"].set_index("tid").pred.reindex(idx).values == gold
        cl = np.arange(len(idx)) // 10  # calls of the K order (x is in prompt order per call)
        order_k = pilot.order_of(df, "byds-pro", len(idx), t)
        pos = {tid: i // 10 for i, tid in enumerate(order_k)}
        cl = np.array([pos[i] for i in idx])
        m, se = cluster_diff(okk.astype(float) - acc_r, cl)
        ref = pd.read_csv(f"{M}/B10__ds-pro__{t}.csv").query("order == 'shuffled'").set_index("tid").pred.reindex(idx)
        rows.append(dict(task=t, n=len(idx), key_accuracy=(ref.values == gold).mean(),
                         rho=pd.Series(ref.values).corr(pd.Series(gold), method="spearman"),
                         diff_pp=100 * m, se_pp=100 * se, p=2 * norm.sf(abs(m / se)) if se > 0 else 1.0,
                         sorted_same_tuples_pp=100 * (okk_sorted.mean() - acc_r.mean())))
    D = pd.DataFrame(rows)
    p = D.p.values; o = np.argsort(p); adj = np.empty(len(p))
    adj[o] = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[o])); D["p_holm"] = adj
    D.to_csv(f"{R}/rev2_block_k.csv", index=False)
    print(D.round(4).to_string())


def block_x():
    X = L("X__ds-flash__*.csv"); B = L("B__ds-flash__*.csv")
    rows = []
    for t, x in X.groupby("task"):
        b = B[(B.task == t) & (B.B == 10)]
        for name, y in [("plain", b), ("delimited", x)]:
            piv = y.pivot_table(index="tid", columns="order", values="pred", aggfunc="first")
            gold = y.groupby("tid").gold.first().loc[piv.index]
            base = np.mean([(piv[o] == gold).mean() for o in ["shuffled", "shuffled1"]])
            rows.append(dict(task=t, template=name, acc_canonical=(piv["shuffled"] == gold).mean(),
                             comp_1=(piv["shuffled1"] != piv["shuffled"]).mean(),
                             sorted_pp=100 * ((piv["sorted"] == gold).mean() - base)))
    D = pd.DataFrame(rows); D.to_csv(f"{R}/rev2_block_x.csv", index=False)
    print(D.round(4).to_string())


def block_s():
    """AG News clustered on the news agency named in the text (Block S), DS-Flash, against the mean of the random orders."""
    x = pd.read_csv(f"{M}/S__ds-flash__agnews.csv"); b = L("B__ds-flash__agnews.csv"); b = b[b.B == 10]
    df = load("agnews"); idx = x.tid.values; gold = df.gold.values[idx]
    base = np.mean([b[b.order == o].set_index("tid").pred.reindex(idx).values == gold for o in RND], axis=0)
    ok = x.set_index("tid").pred.reindex(idx).values == gold
    pos = {t: i // 10 for i, t in enumerate(pilot.order_of(df, "source", len(idx), "agnews"))}
    m, se = cluster_diff(ok.astype(float) - base, np.array([pos[i] for i in idx]))
    src = df.text.str.extract(r"\((Reuters|AP|AFP|Canadian Press|Sports Network|PC World|washingtonpost\.com|NewsFactor|TechWeb|SPACE\.com)\)", expand=False)
    print(f"Block S: accuracy {ok.mean():.4f} vs random mean {base.mean():.4f}: {100 * m:+.2f} pp, se {100 * se:.2f}, "
          f"p {2 * norm.sf(abs(m / se)):.3f}; tuples with an agency {src.iloc[idx].notna().mean():.3f}")


if __name__ == "__main__":
    for w in sys.argv[1:] or ["template", "h2_orig", "contrast", "gt_subsets", "heldout", "glm", "block_k", "block_x", "block_s"]:
        print(f"\n===== {w}"); globals()[w]()
