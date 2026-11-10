"""EAB pilot: batch-size curves, run-to-run non-determinism, and batch-composition/physical-order effects.

Usage: python pilot.py <model> <exp> [<exp> ...]     exp in {E1, E2, E3}
Output: results/pilot/<exp>__<model>.csv, one row per (task, B, fmt, order, rep, tid).

Prompts and the parser are copied verbatim from ICDE2027/code/label.py so that B=1 and B=10 at rep 0 are the
same physical labelers as in the existing catalog.
"""
import json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
from tasks import TASKS, load  # read-only import: dataset specs and samples
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm
from tasks_ext import TASKS_EXT, load_ext  # entity matching and a non-English task (2026-11-06)
TASKS.update(TASKS_EXT)
_load_base = load


def load(name, *a, **k):
    return load_ext(name) if name in TASKS_EXT else _load_base(name, *a, **k)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "results", "pilot")
os.makedirs(OUT, exist_ok=True)
WORKERS = {"ds-flash": 6, "ds-pro": 6, "glm-air": 2, "glm-flash": 1, "glm-air-k3": 3, "glm-5.3": 4, "glm-5.3-flash": 4, "kimi-k2.6": 3, "kimi-k3": 2, "gpt-6-sol": 8, "gpt-6-sol-nr": 8, "gpt-6-sol-nr2": 8, "gpt-6.1-sol": 6, "gpt-6-astra": 6, "kimi-k3-gw": 4, "qwen3.8-27b": 4, "glm-5.3-gw": 4, "kimi-k3.1-ark": 6, "glm-5.3-zp": 3, "claude-opus-5.5": 6, "qwen2.5-3b": 1, "qwen2.5-7b": 1}


def parse(text, labels):
    t = text.strip().lower().strip(" .\"'*`")
    for i, l in enumerate(labels):
        if t == l: return i
    hits = []
    for i, l in enumerate(labels):
        m = re.search(r"(^|\W)" + re.escape(l) + r"($|\W)", t)
        if m: hits.append((m.start(), -len(l), i))
    return min(hits)[2] if hits else -1


def prompt_single(spec, text):
    opts = ", ".join(spec["labels"])
    return [{"role": "system", "content": "You are a careful data annotator."},
            {"role": "user", "content": f"{spec['question']}\nAnswer with exactly one of: {opts}.\n\nText:\n\"\"\"{text}\"\"\"\n\nAnswer:"}]


def prompt_batch(spec, texts):
    opts = ", ".join(spec["labels"])
    body = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(texts))
    return [{"role": "system", "content": "You are a careful data annotator."},
            {"role": "user", "content": f"{spec['question']}\nFor each of the {len(texts)} texts below, answer with exactly one of: {opts}.\n"
                                        f"Return one line per text in the form `[k] answer`, nothing else.\n\n{body}"}]


def prompt_batch_ind(spec, texts):
    """Batch prompt plus an explicit statement that the texts are independent (remedy for the contrast effect)."""
    m = prompt_batch(spec, texts)
    m[1]["content"] = m[1]["content"].replace("nothing else.\n\n", "nothing else.\nThe texts are unrelated to each other: "
                                              "judge each text on its own. Labels may repeat, and a batch need not contain every label.\n\n", 1)
    return m


def prompt_batch_xml(spec, texts):
    """Batch prompt with explicit delimiters and ids around every text (robustness check of the template)."""
    opts = ", ".join(spec["labels"])
    body = "\n".join(f'<text id="{i+1}">\n{t}\n</text>' for i, t in enumerate(texts))
    return [{"role": "system", "content": "You are a careful data annotator."},
            {"role": "user", "content": f"{spec['question']}\nBelow are {len(texts)} texts, each enclosed in <text id=\"k\"> ... </text>. "
                                        f"For each text, answer with exactly one of: {opts}.\n"
                                        f"Return one line per text in the form `[k] answer`, where k is the text's id, nothing else.\n\n{body}"}]


def prompt_batcher(spec, texts):
    """BatchER's question-batching prompt for entity matching (Fan et al., ICDE 2024; github.com/fmh1art/BatchER,
    src/prompting.py and exp.py), zero-shot: no demonstrations. Each text holds "Product A is ...\nProduct B is ..."."""
    task = "When determining whether two Products are the same, you should only focus on critical properties and overlook noisy factors.\n"
    qs = "\n".join(f"Question {i + 1}:\n " + t.replace("\n", "\n ") for i, t in enumerate(texts))
    sent = "question above" if len(texts) == 1 else f"above {len(texts)} questions"
    trailer = (f"\nUse domain knowledge of Products to help understand the text and answer the {sent} in the format: "
               "For Question i, Yes, Product A and Product B are the same product./No, Product A and Product B are "
               "different products. For Question i+1, (repeat the above procedures)")
    return [{"role": "user", "content": "\n\n".join([task, qs + trailer])}]


def parse_batcher(text, n):
    found = {int(k): v.lower() for k, v in re.findall(r"Question\s*(\d+)\s*[,:.]?\s*(Yes|No)", text, flags=re.I)}
    return [(1 if found[j + 1] == "yes" else 0) if (j + 1) in found else -1 for j in range(n)], \
           [(j + 1) not in found for j in range(n)]


def prompt_semajson(spec, texts):
    """Sema-style batch prompt (Qi et al., PVLDB 19(11), 2026): the tuples are passed as a JSON array and the API's
    structured output forces a JSON array of results of the same length."""
    opts = ", ".join(spec["labels"])
    items = json.dumps([{"id": i + 1, "text": t} for i, t in enumerate(texts)], ensure_ascii=False)
    return [{"role": "system", "content": "You are a careful data annotator. Respond in JSON."},
            {"role": "user", "content": f"{spec['question']} Answer with exactly one of: {opts}.\n"
                                        f"Input (a JSON array of {len(texts)} items):\n{items}\n\n"
                                        f'Return a JSON object {{"results": [...]}} whose array has exactly {len(texts)} '
                                        f"answers, one per item, in input order."}]


def parse_semajson(text, labels, n):
    try:
        t = text.strip()
        m = re.search(r"\{.*\}", t, flags=re.S)
        res = json.loads(m.group(0) if m else t)["results"]
        if len(res) != n:
            return [-1] * n, [True] * n
        return [parse(str(r), labels) for r in res], [False] * n
    except Exception:
        return [-1] * n, [True] * n


def prompt_json(spec, texts):
    opts = ", ".join(spec["labels"])
    body = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(texts))
    return [{"role": "system", "content": "You are a careful data annotator."},
            {"role": "user", "content": f"{spec['question']}\nFor each of the {len(texts)} texts below, answer with exactly one of: {opts}.\n"
                                        f'Return only a JSON array of the form [{{"k": 1, "label": "..."}}, ...], one object per text, nothing else.\n\n{body}'}]


def parse_json(text, labels, n):
    """Strict JSON parse (code fences removed); malformed output means every tuple of the call is unparsed."""
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m: t = m.group(1).strip()
    try:
        arr = json.loads(t)
        found = {int(o["k"]): str(o["label"]) for o in arr if isinstance(o, dict) and "k" in o and "label" in o}
    except (ValueError, TypeError):
        return [-1] * n, [True] * n
    return [parse(found.get(j + 1, ""), labels) for j in range(n)], [(j + 1) not in found for j in range(n)]


def order_of(df, order, n, task=None):
    """'shuffled': random order (seed 0); 'sorted': by gold label (a table physically clustered on a key that
    correlates with the attribute); 'shuffledK': another random order (seed K) for composition tests."""
    tids = np.arange(len(df))
    if order.startswith("shuffled"):
        seed = int(order[8:] or 0)
        tids = np.random.default_rng(seed).permutation(tids)[:n] if seed == 0 else \
            np.random.default_rng(seed).permutation(np.random.default_rng(0).permutation(tids)[:n])
    elif order == "stored":  # the canonical tuples in the order in which the distributed file stores them
        tids = np.sort(np.random.default_rng(0).permutation(tids)[:n])
    elif order == "sorted":
        sub = np.random.default_rng(0).permutation(tids)[:n]
        tids = sub[np.argsort(df.gold.values[sub], kind="stable")]
    elif order.startswith("perm"):  # 'perm<K>': canonical batch membership (B=10), positions permuted with seed K
        base = np.random.default_rng(0).permutation(tids)[:n]
        rng = np.random.default_rng(int(order[4:]))
        tids = np.concatenate([rng.permutation(base[i:i + 10]) for i in range(0, len(base), 10)])
    elif order.startswith("noisy"):  # 'noisy<s>': clustered on gold + N(0, s^2) noise, a partly attribute-correlated key
        sub = np.random.default_rng(0).permutation(tids)[:n]
        key = df.gold.values[sub] + np.random.default_rng(7).normal(0, float(order[5:]), len(sub))
        tids = sub[np.argsort(key, kind="stable")]
    elif order.startswith("by"):  # 'by<model>': clustered on another labeler's predicted label (a materialized
        # semantic column); uses that labeler's canonical B=10 predictions, which cover the first n tuples
        sub = np.random.default_rng(0).permutation(tids)[:n]
        ref = pd.read_csv(os.path.join(HERE, "results", "main", f"B10__{order[2:]}__{task}.csv"))
        ref = ref[(ref.order == "shuffled") & (ref.B == 10)].set_index("tid").pred
        tids = sub[np.argsort(ref.reindex(sub).fillna(-9).values, kind="stable")]
    elif order == "source":  # AG News clustered on a real metadata key: the news agency named in the text
        sub = np.random.default_rng(0).permutation(tids)[:n]
        src = df.text.str.extract(r"\((Reuters|AP|AFP|Canadian Press|Sports Network|PC World|washingtonpost\.com|"
                                  r"NewsFactor|TechWeb|SPACE\.com)\)", expand=False).fillna("~none").values
        tids = sub[np.argsort(src[sub], kind="stable")]
    elif order == "keyed":  # clustered on an attribute-independent key (text length)
        sub = np.random.default_rng(0).permutation(tids)[:n]
        tids = sub[np.argsort(df.text.str.len().values[sub], kind="stable")]
    return list(tids[:n])


def run_cell(model, task, B, fmt, order, rep, n):
    spec, df = TASKS[task], load(task)
    tids = order_of(df, order, n, task)
    groups = [tids[i:i + B] for i in range(0, len(tids), B)]

    def work(g):
        texts = [df.text[i] for i in g]
        try:
            if fmt == "single":
                r = llm.chat(model, prompt_single(spec, texts[0]), max_tokens=2000, rep=rep)
                preds = [parse(r["text"], spec["labels"])] if r["status"] == "ok" else [-2]
                miss = [False]
            elif fmt == "batcher":
                r = llm.chat(model, prompt_batcher(spec, texts), max_tokens=40 * len(g) + 200, rep=rep)
                preds, miss = parse_batcher(r["text"], len(g))
                if r["status"] != "ok": preds = [-2] * len(g)
            elif fmt == "semajson":
                r = llm.chat(model, prompt_semajson(spec, texts), max_tokens=60 * len(g) + 200, rep=rep,
                             extra={"response_format": {"type": "json_object"}})
                preds, miss = parse_semajson(r["text"], spec["labels"], len(g))
                if r["status"] != "ok": preds = [-2] * len(g)
            elif fmt == "json":
                r = llm.chat(model, prompt_json(spec, texts), max_tokens=60 * len(g) + 200, rep=rep)
                preds, miss = parse_json(r["text"], spec["labels"], len(g))
                if r["status"] != "ok": preds = [-2] * len(g)
            else:
                pr = prompt_batch_ind(spec, texts) if fmt == "batchind" else \
                    prompt_batch_xml(spec, texts) if fmt == "batchxml" else prompt_batch(spec, texts)
                r = llm.chat(model, pr, max_tokens=60 * len(g) + 200, rep=rep)
                found = dict((int(k), v) for k, v in re.findall(r"\[(\d+)\]\s*([^\n]+)", r["text"]))
                miss = [(j + 1) not in found for j in range(len(g))]
                preds = [parse(found.get(j + 1, ""), spec["labels"]) for j in range(len(g))] if r["status"] == "ok" \
                    else [-2] * len(g)
            u = r["usage"]
            return [dict(tid=i, pos=j, pred=p, missing=m, status=r["status"], usd=r["usd"] / len(g),
                         in_tok=u.get("prompt_tokens", 0) / len(g), out_tok=u.get("completion_tokens", 0) / len(g),
                         cache_hit=u.get("prompt_cache_hit_tokens", 0) / len(g), call_lat=r["latency"],
                         finish=r.get("finish"), served=r.get("served"), B_eff=len(g)) for j, (i, p, m) in enumerate(zip(g, preds, miss))]
        except Exception as e:
            print("ERR", model, task, B, rep, g[0], str(e)[:150], flush=True)
            return [dict(tid=i, pos=j, pred=-3, missing=True, status="error", usd=np.nan) for j, i in enumerate(g)]

    t0 = time.time()
    with ThreadPoolExecutor(WORKERS.get(model, 8)) as ex:
        rows = [x for res in ex.map(work, groups) for x in res]
    wall = time.time() - t0
    out = pd.DataFrame(rows)
    out["gold"] = df.gold.values[out.tid.values]
    for k, v in dict(task=task, model=model, B=B, fmt=fmt, order=order, rep=rep, wall=wall).items():
        out[k] = v
    acc = (out.pred == out.gold).mean()
    print(f"{model:10s} {task:8s} B={B:<3d} {fmt:6s} {order:9s} rep={rep} n={len(out)} acc={acc:.3f} "
          f"refused={(out.status == 'refused').mean():.3f} missing={out.missing.mean():.3f} "
          f"$={out.usd.sum():.4f} wall={wall:.0f}s", flush=True)
    return out


def cells(model, exp):
    big = model.startswith("ds")
    if exp == "E1":  # batch-size curve on four tasks
        n1 = 1000 if big else (200 if model.startswith("glm") else 500)
        for task in ["agnews", "emotion", "hate", "yelp"]:
            yield task, 1, "single", "shuffled", 0, n1
            if big: yield task, 1, "batch", "shuffled", 0, 1000  # batch format with one tuple: format vs. context
            for B in [5, 10, 20, 50] + ([100] if big else []):
                yield task, B, "batch", "shuffled", 0, 1000
    if exp == "E1b":  # format control (batch format, one tuple) for the rate-limited vendors
        for task in ["agnews", "emotion", "hate", "yelp"]:
            yield task, 1, "batch", "shuffled", 0, 500 if model.startswith("kimi") else 200
    if exp == "E23k":  # reduced composition test for slow endpoints: two reruns and two new neighbourhoods
        for task in ["agnews", "emotion", "hate", "yelp"]:
            for rep in range(2):
                yield task, 10, "batch", "shuffled", rep, 500
            for order in ["shuffled1", "shuffled2"]:
                yield task, 10, "batch", order, 0, 500
    if exp == "E2":  # repeats at the vendor's lowest temperature
        for task in ["agnews", "emotion", "hate", "yelp"]:
            for rep in range(5):
                if model in ("ds-flash", "ds-pro"): yield task, 1, "single", "shuffled", rep, 500
                if model != "ds-pro": yield task, 10, "batch", "shuffled", rep, 500
    if exp == "E3":  # batch composition: other random neighbours, and label-clustered physical order
        for task in ["agnews", "emotion", "hate", "yelp"]:
            for order in ["shuffled1", "shuffled2"]:
                yield task, 10, "batch", order, 0, 500
        for task in ["imdb", "amazon", "yelp", "agnews"]:
            for B in [10, 50]:
                yield task, B, "batch", "sorted", 0, 1000
                yield task, B, "batch", "shuffled", 0, 1000


if __name__ == "__main__":
    model = sys.argv[1]
    for exp in sys.argv[2:]:
        path = os.path.join(OUT, f"{exp}__{model}.csv")
        res = [run_cell(model, *c) for c in cells(model, exp)]
        pd.concat(res).to_csv(path, index=False)
        print("wrote", path, "spent today:", {k: round(v, 3) for k, v in llm._spent.items()}, flush=True)
