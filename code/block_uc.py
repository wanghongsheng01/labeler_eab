"""Two blocks requested by the third internal review (2026-11-06), DS-Flash and DS-Pro, B = 10.

Block U (uncached rerun): a fresh random grouping (seed 9) that no earlier call used, sent twice at the same moment
(rep 0 and rep 1 in parallel), so that neither copy can be served from the vendor's prefix cache of the other. The
disagreement between the two copies is a rerun flip rate without prefix caching.

Block C (controlled neighbors): for 500 target tuples, the target sits at position 5 of a call whose nine neighbors are
(same) all of the target's gold class, (other) all of one other class, or (mixed) a uniform random draw. A contrast
effect predicts lower accuracy under "same" than under "mixed", and that labels under "other" avoid the neighbors' class.
Usage: python block_uc.py <model> U|C <task> [<task> ...]
Output: results/main/U__<model>__<task>.csv, results/main/C__<model>__<task>.csv"""
import os, re, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm, pilot

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = f"{HERE}/results/main"


def ask(model, spec, texts, rep):
    r = llm.chat(model, pilot.prompt_batch(spec, texts), max_tokens=60 * len(texts) + 200, rep=rep)
    found = dict((int(k), v) for k, v in re.findall(r"\[(\d+)\]\s*([^\n]+)", r["text"]))
    preds = [pilot.parse(found.get(j + 1, ""), spec["labels"]) for j in range(len(texts))] if r["status"] == "ok" \
        else [-2] * len(texts)
    return preds, (r.get("usage") or {}).get("prompt_cache_hit_tokens", 0), (r.get("usage") or {}).get("prompt_tokens", 0)


def block_u(model, task, n):
    spec, df = pilot.TASKS[task], pilot.load(task)
    tids = pilot.order_of(df, "shuffled9", n)
    groups = [tids[i:i + 10] for i in range(0, len(tids), 10)]
    rows = []

    def pair(g):
        texts = [df.text[i] for i in g]
        with ThreadPoolExecutor(2) as ex:  # both copies leave at the same time
            a, b = ex.submit(ask, model, spec, texts, 0), ex.submit(ask, model, spec, texts, 1)
            (pa, ca, ta), (pb, cb, tb) = a.result(), b.result()
        return [dict(tid=t, pos=j, pred0=x, pred1=y, cache0=ca / max(ta, 1), cache1=cb / max(tb, 1))
                for j, (t, x, y) in enumerate(zip(g, pa, pb))]

    with ThreadPoolExecutor(4) as ex:
        for res in ex.map(pair, groups):
            rows += res
    d = pd.DataFrame(rows); d["gold"] = df.gold.values[d.tid.values]; d["task"] = task; d["model"] = model
    d.to_csv(f"{M}/U__{model}__{task}.csv", index=False)
    ok = (d.pred0 >= 0) & (d.pred1 >= 0)
    print(model, task, "uncached rerun flip", round(float((d.pred0 != d.pred1)[ok].mean()), 4),
          "cache share", round(float(d[["cache0", "cache1"]].mean().mean()), 3), flush=True)


def block_c(model, task, n_targets=500):
    spec, df = pilot.TASKS[task], pilot.load(task)
    rng = np.random.default_rng(11)
    targets = pilot.order_of(df, "shuffled", n_targets)
    by_class = {c: np.array(df.tid[df.gold == c]) for c in range(len(spec["labels"]))}
    jobs = []
    for t in targets:
        y = int(df.gold[t]); others = [c for c in by_class if c != y]
        oc = int(rng.choice(others))
        pool = np.setdiff1d(np.arange(len(df)), [t])
        conds = {"same": rng.choice(np.setdiff1d(by_class[y], [t]), 9, replace=False),
                 "other": rng.choice(by_class[oc], 9, replace=False),
                 "mixed": rng.choice(pool, 9, replace=False)}
        for c, nb in conds.items():
            g = list(nb[:4]) + [t] + list(nb[4:])
            jobs.append((t, y, c, oc if c == "other" else -1, g))

    def run(job):
        t, y, c, oc, g = job
        preds, _, _ = ask(model, spec, [df.text[i] for i in g], 0)
        return dict(tid=t, gold=y, cond=c, nbr_class=oc, pred=preds[4])

    with ThreadPoolExecutor(6) as ex:
        rows = list(ex.map(run, jobs))
    d = pd.DataFrame(rows); d["task"] = task; d["model"] = model
    d.to_csv(f"{M}/C__{model}__{task}.csv", index=False)
    acc = d.groupby("cond").apply(lambda z: (z.pred == z.gold).mean()).round(4).to_dict()
    o = d[d.cond == "other"]
    print(model, task, "accuracy by condition", acc, "other: label = neighbors' class",
          round(float((o.pred == o.nbr_class).mean()), 4), flush=True)


if __name__ == "__main__":
    model, block, tasks = sys.argv[1], sys.argv[2], sys.argv[3:]
    for t in tasks:
        if block == "U":
            block_u(model, t, 2000 if model == "ds-flash" else 1000)
        else:
            block_c(model, t)
