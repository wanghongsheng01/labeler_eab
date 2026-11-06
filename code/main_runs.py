"""Main-run blocks A (batch curve x prompt format) and B (composition and physical order) of
protocol/benchmark_protocol.md, on every existing public dataset.

Usage: python main_runs.py <model> <block> [n]      block in {A, A4, B, B10, R, R10, P, N, I, R2} (A4 = the reduced A for costly labelers)
Output: results/main/<block>__<model>__<task>.csv (one file per task, so finished tasks survive a stop).
"""
import os, sys
import pandas as pd
import pilot, llm

TASKS7 = ["imdb", "yelp", "agnews", "emotion", "hate", "offensive", "amazon"]
if os.environ.get("TASKS"): TASKS7 = os.environ["TASKS"].split(",")  # optional subset, e.g. for blocks N and I
OUT = os.path.join(pilot.HERE, "results", "main")
os.makedirs(OUT, exist_ok=True)


def block_cells(block, n):
    if block == "A":
        yield 1, "single", "shuffled"
        for fmt in ["batch", "json"]:
            for B in [1, 5, 10, 20, 50, 100]:
                yield B, fmt, "shuffled"
    if block == "A4":
        yield 1, "single", "shuffled"
        for B in [1, 10, 50]:
            yield B, "batch", "shuffled"
    if block in ("R", "R10"):  # rerun baseline for H1: same prompts as the canonical cells, second independent call
        for B in ([10] if block == "R10" else [10, 50]):
            yield B, "batch", "shuffled", 1
    if block in ("B", "B10"):
        for B in ([10] if block == "B10" else [10, 50]):
            for order in ["shuffled", "shuffled1", "shuffled2", "shuffled3", "sorted", "keyed"]:
                yield B, "batch", order


    # blocks added after the simulated review (2026-11-05); see protocol change log v0.2
    if block == "P":  # same batch members, permuted positions (separates position from neighbour membership)
        for order in ["perm1", "perm2"]:
            yield 10, "batch", order
    if block == "N":  # partly attribute-correlated clustering keys (dose response)
        for order in ["noisy0.5", "noisy1", "noisy2"]:
            yield 10, "batch", order
    if block == "I":  # prompt remedy: "texts are unrelated, labels may repeat"
        for order in ["shuffled", "sorted"]:
            yield 10, "batchind", order
    if block == "R2":  # late rerun, one day later (prefix cache expired), B = 10
        yield 10, "batch", "shuffled", 2

    # blocks added after the second simulated review (2026-11-06)
    if block == "B10s":  # reduced Block B for the most expensive labeler: four random orders would cost too much
        for order in ["shuffled", "shuffled1", "shuffled2", "sorted", "keyed"]:
            yield 10, "batch", order
    # blocks added on 2026-11-06: other task types and the batch templates of published systems
    if block == "TE":  # entity matching (abtbuy): our template and BatchER's template
        for fmt in ["batch", "batcher"]:
            for order in ["shuffled", "shuffled1", "sorted", "keyed"]:
                yield 10, fmt, order
            yield 10, fmt, "shuffled", 1
    if block == "TZ":  # a non-English task (amazon_zh)
        for order in ["shuffled", "shuffled1", "sorted", "keyed"]:
            yield 10, "batch", order
        yield 10, "batch", "shuffled", 1
    if block == "TJ":  # Sema-style structured-output (JSON array) batch template
        for order in ["shuffled", "shuffled1", "sorted"]:
            yield 10, "semajson", order
        yield 10, "semajson", "shuffled", 1
    if block == "L":  # compact block for the slow local open-weight labeler: composition, order, position and rerun
        for order in ["shuffled", "shuffled1", "sorted", "keyed", "perm1"]:
            yield 10, "batch", order
        yield 10, "batch", "shuffled", 1
    if block == "S":  # AG News clustered on its news agency (a metadata key); run with TASKS=agnews
        yield 10, "batch", "source"
    if block == "K":  # clustered on a materialized semantic column: DS-Pro's predicted label (first 1,000 tuples)
        yield 10, "batch", "byds-pro"
    if block == "X":  # delimited template with ids: canonical, one more composition, and the sorted order
        for order in ["shuffled", "shuffled1", "sorted"]:
            yield 10, "batchxml", order


def vendor_up(model):
    """One uncached tiny call; False when the vendor is overloaded or unreachable (logged as availability data)."""
    import time, json, datetime, requests
    prov, name = llm.MODELS[model]
    cfg = llm.KEYS[prov]
    t0 = time.time()
    try:
        r = requests.post(cfg["base_url"].rstrip("/") + "/chat/completions", headers={"Authorization": "Bearer " + cfg["api_key"]},
                          json={"model": name, "messages": [{"role": "user", "content": "Say OK"}], "max_tokens": 5,
                                **{k: v for k, v in llm.FIXED.get(model, {}).items() if k != "temperature"}}, timeout=45)
        code = r.status_code
    except requests.RequestException:
        code = -1
    with open(os.path.join(OUT, "availability.log"), "a") as f:
        f.write(json.dumps({"t": time.time(), "when": datetime.datetime.now().isoformat(timespec="seconds"),
                            "model": model, "http": code, "lat": round(time.time() - t0, 2)}) + "\n")
    return code == 200


if __name__ == "__main__":
    model, block = sys.argv[1], sys.argv[2]
    if llm.KEYS.get(llm.MODELS[model][0], {}).get("api") != "responses" and not vendor_up(model):
        print("VENDOR DOWN", model, flush=True); sys.exit(2)
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    for task in TASKS7:
        path = os.path.join(OUT, f"{block}__{model}__{task}.csv")
        if os.path.exists(path): continue
        res = []
        for B, fmt, order, *rep in block_cells(block, n):
            if llm._spent.get(model, 0) > llm.DAILY_CAP:
                break
            res.append(pilot.run_cell(model, task, B, fmt, order, rep[0] if rep else 0, n))
        d = pd.concat(res) if res else pd.DataFrame()
        if len(res) < len(list(block_cells(block, n))) or (d.status == "error").any():
            # incomplete (cap or persistent errors): keep for inspection, rerun later; successful calls are cached
            d.to_csv(path + ".partial", index=False)
            print("INCOMPLETE", task, "cells", len(res), "errors", int((d.status == "error").sum()) if len(d) else 0, flush=True)
            break
        d.to_csv(path, index=False)
        print("wrote", path, "spent today:", {k: round(v, 3) for k, v in llm._spent.items()}, flush=True)
