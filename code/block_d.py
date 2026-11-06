"""Block D of the main runs: refusal bisection for glm-5.3-flash (B10 + R10 on agnews and hate). Tests H3 (refusal
triggers are not missing at random).

1. Group-testing decode (no calls): the 7 runs (6 orders + 1 rerun) place every tuple in 7 different batches; a
   tuple that ever sits in an answered batch is cleared, and the triggers are a minimum hitting set of the refused
   batches, chosen greedily.
2. Single-call confirmation: every tuple of every refused batch is re-sent alone to glm-5.3-flash (rep 0 and 1).
   Skipped when there is no key for the Zhipu subscription endpoint.
3. The triggers are re-sent alone to the other vendors.

Usage: python block_d.py
Output: results/main/D__decode.csv, D__glm-5.3-flash.csv (step 2), D__vendors.csv, results/main_refusal.txt.
"""
import os, sys, glob, collections
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
from scipy.stats import binomtest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot, llm
from tasks import TASKS, load

HERE = pilot.HERE
M = f"{HERE}/results/main"
GLM, OTHERS, TASKS_D, N, B = "glm-5.3-flash", ["ds-flash", "ds-pro", "kimi-k2.6"], ["agnews", "hate"], 1000, 10
out = open(f"{HERE}/results/main_refusal.txt", "w")


def say(*a):
    print(*a, flush=True); print(*a, file=out, flush=True)


def single(model, task, tid, rep=0):
    spec, df = TASKS[task], load(task)
    try:
        r = llm.chat(model, pilot.prompt_single(spec, df.text[tid]), max_tokens=2000, rep=rep, retries=12)
    except RuntimeError:
        return "error", -3, 0.0
    return r["status"], pilot.parse(r["text"], spec["labels"]) if r["status"] == "ok" else -2, r["usd"]


dec, conf, trig_all = [], [], []
for task in TASKS_D:
    spec, df = TASKS[task], load(task)
    K = len(spec["labels"])
    runs = {}
    for f in glob.glob(f"{M}/B10__{GLM}__{task}.csv") + glob.glob(f"{M}/R10__{GLM}__{task}.csv"):
        for o, x in pd.read_csv(f).groupby("order"):
            runs[os.path.basename(f)[0] + ":" + o] = x.set_index("tid").pred
    batches = []
    for run, pred in runs.items():
        tids = pilot.order_of(df, run.split(":")[1], N)
        batches += [(run, tids[i:i + B], bool((pred.loc[tids[i:i + B]] == -2).all())) for i in range(0, N, B)]
    ref = [(r, g) for r, g, x in batches if x]
    app = collections.Counter(t for _, g, _ in batches for t in g)
    nref = collections.Counter(t for _, g in ref for t in g)
    rem, trig = list(ref), []
    while rem:  # greedy minimum hitting set of the refused batches
        t = collections.Counter(t for _, g in rem for t in g).most_common(1)[0][0]
        trig.append(t); rem = [(r, g) for r, g in rem if t not in g]
    say(f"\n== {task}: {len(ref)} refused batches of {len(batches)} in {len(runs)} runs; "
        f"{len(nref)} tuples sit in a refused batch; triggers (hitting set): {len(trig)}")
    for t in trig:
        say(f"  tid {t} gold={spec['labels'][df.gold[t]]} refused in {nref[t]}/{app[t]} of its batches")  # trigger texts are never printed
        dec.append(dict(task=task, tid=t, gold=int(df.gold[t]), n_batches=app[t], n_refused=nref[t], text=df.text[t]))
        trig_all.append((task, t))
    can_b = {tuple(g) for r, g in ref if r == "B:shuffled"}; can_r = {tuple(g) for r, g in ref if r == "R:shuffled"}
    say(f"canonical order refused batches: run 1 {len(can_b)}, rerun {len(can_r)}, differing {len(can_b ^ can_r)}")
    sub = pilot.order_of(df, "shuffled", N)
    pop = np.bincount(df.gold.values[sub], minlength=K) / N
    tg = np.bincount(df.gold.values[trig], minlength=K)
    c = int(np.argmax(tg))
    say(f"H3: {tg[c]}/{tg.sum()} triggers are '{spec['labels'][c]}' (sample share {pop[c]:.3f}); "
        f"one-sided binomial p under MAR = {binomtest(int(tg[c]), int(tg.sum()), pop[c], alternative='greater').pvalue:.2e}")
    kept = runs["B:shuffled"][runs["B:shuffled"] != -2].index
    say(f"canonical run drops {N - len(kept)} tuples; gold shares all vs kept: "
        f"{np.round(pop, 3).tolist()} vs {np.round(np.bincount(df.gold.values[kept], minlength=K) / len(kept), 3).tolist()}")
    if "glmplan" in llm.KEYS:  # step 2: confirm alone
        cand = sorted(nref)
        with ThreadPoolExecutor(4) as ex:
            r0 = list(ex.map(lambda t: single(GLM, task, t, 0), cand))
            r1 = list(ex.map(lambda t: single(GLM, task, t, 1), cand))
        C = pd.DataFrame([dict(task=task, tid=t, gold=int(df.gold[t]), alone_rep0=a[0], alone_rep1=b[0], hitting_set=t in trig)
                          for t, a, b in zip(cand, r0, r1)])
        conf.append(C)
        alone = set(C[(C.alone_rep0 == "refused") | (C.alone_rep1 == "refused")].tid)
        say(f"alone: {len(alone)} of {len(cand)} tuples refused (rep 0 or 1); hitting set ∩ alone = "
            f"{len(alone & set(trig))}/{len(trig)}; refused alone but not in hitting set: {sorted(alone - set(trig))}")
    else:
        say("step 2 skipped: no key for the GLM subscription endpoint (key entry 'glmplan')")

pd.DataFrame(dec).to_csv(f"{M}/D__decode.csv", index=False)
if conf: pd.concat(conf).to_csv(f"{M}/D__{GLM}.csv", index=False)
if all(llm.MODELS[m][0] in llm.KEYS for m in OTHERS):
    V = pd.DataFrame([dict(task=task, tid=t, model=m, **dict(zip(["status", "pred", "usd"], single(m, task, t))),
                           gold=int(load(task).gold[t])) for task, t in trig_all for m in OTHERS])
    V.to_csv(f"{M}/D__vendors.csv", index=False)
else:  # no keys: report the released results of step 3
    V = pd.read_csv(f"{M}/D__vendors.csv")
say("\n== triggers sent alone to the other vendors")
say(V.groupby(["task", "model"]).apply(lambda x: pd.Series(dict(
    n=len(x), refused=int((x.status == "refused").sum()), correct=int((x.pred == x.gold).sum()), usd=x.usd.sum()))).to_string())
