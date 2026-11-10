"""Block F (2026-11-09): selective semantic filters in a random order, no clustering.

A semantic filter keeps the tuples with A(t) = positive class. Real filters are selective, so most neighbors in a batch
are negatives even in a random order. For Hate, Offensive and IMDB we draw 1,000 tuples (500 for Offensive, which has 860) from the sample
with 5%, 20% and 50% positives (seed 13), label them one per call (single format, and the batch template with B = 1) and with B = 10 under two random
groupings, and measure recall, false-positive rate and the relative error of COUNT(*) of the filter.
Usage: python block_f.py <model> [task ...]
Output: results/main/F__<model>__<task>.csv"""
import os, re, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot, llm

HERE = pilot.HERE
M = f"{HERE}/results/main"
POS = {"hate": 1, "offensive": 1, "imdb": 1}
N, B = 1000, 10


def sample(df, task, prev):
    rng = np.random.default_rng(13)
    pos = np.flatnonzero(df.gold.values == POS[task]); neg = np.flatnonzero(df.gold.values != POS[task])
    n = min(N, int(len(pos) / 0.5), int(len(neg) / 0.95)) // 100 * 100  # Offensive has 860 tuples: 500 per table
    k = int(round(prev * n))
    return np.concatenate([rng.choice(pos, k, replace=False), rng.choice(neg, n - k, replace=False)])


def label(model, spec, df, tids, B, rep):
    groups = [list(tids[i:i + max(B, 1)]) for i in range(0, len(tids), max(B, 1))]

    def work(g):
        texts = [df.text[i] for i in g]
        if B == 0:  # one tuple per call in the batch template (the format control of Block A)
            r = llm.chat(model, pilot.prompt_batch(spec, texts), max_tokens=60 * len(g) + 200, rep=rep)
            found = dict((int(k), v) for k, v in re.findall(r"\[(\d+)\]\s*([^\n]+)", r["text"]))
            return [(g[0], pilot.parse(found.get(1, ""), spec["labels"]) if r["status"] == "ok" else -2)]
        if B == 1:
            r = llm.chat(model, pilot.prompt_single(spec, texts[0]), max_tokens=2000, rep=rep)
            return [(g[0], pilot.parse(r["text"], spec["labels"]) if r["status"] == "ok" else -2)]
        r = llm.chat(model, pilot.prompt_batch(spec, texts), max_tokens=60 * len(g) + 200, rep=rep)
        found = dict((int(k), v) for k, v in re.findall(r"\[(\d+)\]\s*([^\n]+)", r["text"]))
        return [(t, pilot.parse(found.get(j + 1, ""), spec["labels"]) if r["status"] == "ok" else -2) for j, t in enumerate(g)]

    with ThreadPoolExecutor(pilot.WORKERS.get(model, 6)) as ex:
        return [x for res in ex.map(work, groups) for x in res]


def main(model, tasks):
    for task in tasks:
        spec, df = pilot.TASKS[task], pilot.load(task)
        rows = []
        for prev in [0.05, 0.2, 0.5]:
            base = sample(df, task, prev)
            for cond, seed, Bc in [("single", None, 1), ("batch_b1", None, 0), ("batch_g1", 1, B), ("batch_g2", 2, B)]:
                tids = base if seed is None else np.random.default_rng(seed).permutation(base)
                for t, p in label(model, spec, df, tids, Bc, rep=0):
                    rows.append(dict(task=task, prev=prev, cond=cond, tid=int(t), pred=p, gold=int(df.gold[t])))
        d = pd.DataFrame(rows); d["model"] = model
        d.to_csv(f"{M}/F__{model}__{task}.csv", index=False)
        P = POS[task]
        for (prev, cond), x in d.groupby(["prev", "cond"]):
            pos = x.gold == P; sel = x.pred == P
            print(model, task, prev, cond, "recall", round(float(sel[pos].mean()), 3), "fpr", round(float(sel[~pos].mean()), 3),
                  "count_err", round(float((sel.sum() - pos.sum()) / pos.sum()), 3), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] or list(POS))
