"""Per-call log of the main runs, rebuilt from the response cache with the network disabled.

One JSON line per API call of Blocks A, A4, B, B10, R, R10, P, N, I, R2, K and X: block, cell (model, task, B, format,
order, repetition), call index, the tuple ids of the call in prompt order, the request SHA-256 (the cache key), status,
served model, finish reason, token counts, list-price cost, latency, the vendor's error body for refused calls, and the
raw response text (the answer lines only; responses never repeat the input texts). Every replayed call re-parses its
response and is checked against the released per-tuple labels.

No API key is needed. GLM3_KEY is set to a placeholder so that the GLM request body (Responses API) is rebuilt exactly;
DAILY_CAP = -1 makes a cache miss raise instead of calling a vendor.
Output: results/main/calls_main.jsonl.gz"""
import os
os.environ.setdefault("GLM3_KEY", "placeholder-never-sent")
os.environ["DAILY_CAP"] = "-1"
os.environ["CACHE_AFTER"] = "0"
import glob, gzip, json, re, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm, pilot
from tasks import TASKS, load

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = f"{HERE}/results/main"
BLOCKS = ("A", "A4", "B", "B10", "B10s", "R", "R10", "P", "N", "I", "R2", "K", "X", "S", "L", "TE", "TZ", "TJ")


def prompt(fmt, spec, texts):
    if fmt == "single":
        return pilot.prompt_single(spec, texts[0]), 2000
    if fmt == "json":
        return pilot.prompt_json(spec, texts), 60 * len(texts) + 200
    if fmt == "batcher":
        return pilot.prompt_batcher(spec, texts), 40 * len(texts) + 200
    if fmt == "semajson":
        return pilot.prompt_semajson(spec, texts), 60 * len(texts) + 200
    f = {"batchind": pilot.prompt_batch_ind, "batchxml": pilot.prompt_batch_xml}.get(fmt, pilot.prompt_batch)
    return f(spec, texts), 60 * len(texts) + 200


def main():
    n_calls, n_bad, n_miss = 0, 0, 0
    with gzip.open(f"{M}/calls_main.jsonl.gz", "wt") as out:
        for path in sorted(glob.glob(f"{M}/*__*__*.csv")):
            block = os.path.basename(path).split("__")[0]
            if block not in BLOCKS:
                continue
            d = pd.read_csv(path)
            for (model, task, B, fmt, order, rep), x in d.groupby(["model", "task", "B", "fmt", "order", "rep"]):
                spec, df = pilot.TASKS[task], pilot.load(task)
                tids = pilot.order_of(df, order, len(x), task)
                released = x.set_index("tid").pred
                for c, i in enumerate(range(0, len(tids), int(B))):
                    g = [int(t) for t in tids[i:i + int(B)]]
                    msgs, mt = prompt(fmt, spec, [df.text[t] for t in g])
                    extra = {"response_format": {"type": "json_object"}} if fmt == "semajson" else None
                    fl = True
                    key = llm.request(model, msgs, max_tokens=mt, rep=int(rep), extra=extra)[3]
                    if not os.path.exists(os.path.join(llm.CACHE_DIR, key[:2], key + ".json")):  # ran before the token floor
                        fl = False
                        key = llm.request(model, msgs, max_tokens=mt, rep=int(rep), extra=extra, floor=False)[3]
                    try:
                        r = llm.chat(model, msgs, max_tokens=mt, rep=int(rep), extra=extra, floor=fl)
                    except RuntimeError:  # not in the cache: the call failed in the original run (pred = -3)
                        r = {"status": "error", "text": None}
                        n_miss += 1
                    if r["status"] == "error":
                        preds = [-3] * len(g)
                    elif r["status"] != "ok":
                        preds = [-2] * len(g)
                    elif fmt == "single":
                        preds = [pilot.parse(r["text"], spec["labels"])]
                    elif fmt == "json":
                        preds = pilot.parse_json(r["text"], spec["labels"], len(g))[0]
                    elif fmt == "batcher":
                        preds = pilot.parse_batcher(r["text"], len(g))[0]
                    elif fmt == "semajson":
                        preds = pilot.parse_semajson(r["text"], spec["labels"], len(g))[0]
                    else:
                        found = dict((int(k), v) for k, v in re.findall(r"\[(\d+)\]\s*([^\n]+)", r["text"]))
                        preds = [pilot.parse(found.get(j + 1, ""), spec["labels"]) for j in range(len(g))]
                    n_bad += int(any(int(released.get(t, -9)) != int(p) for t, p in zip(g, preds)))
                    u = r.get("usage") or {}
                    out.write(json.dumps(dict(
                        block=block, model=model, task=task, B=int(B), fmt=fmt, order=order, rep=int(rep), call=c,
                        tids=g, request_sha256=key, status=r["status"], served=r.get("served"), finish=r.get("finish"),
                        prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"),
                        cached_tokens=u.get("prompt_cache_hit_tokens", u.get("cached_tokens", (u.get("prompt_tokens_details") or {}).get("cached_tokens"))),
                        reasoning_tokens=u.get("reasoning_tokens", (u.get("completion_tokens_details") or u.get("output_tokens_details") or {}).get("reasoning_tokens")),
                        usd=r.get("usd"), latency=r.get("latency"), vendor_error=r.get("err"), response=r.get("text"))) + "\n")
                    n_calls += 1
    print(f"exported {n_calls} calls ({n_miss} not in the cache); calls whose re-parsed labels differ from the released "
          f"labels: {n_bad}")


if __name__ == "__main__":
    main()
