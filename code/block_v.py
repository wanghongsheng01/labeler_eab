"""Block V (2026-11-09): the three pilot observations re-measured with fresh calls under the fixed protocol.

V1  refusals of the glm-4.5 endpoints: every tuple that the pilot re-sent alone (E4, 402 tuples from refused batches of
    AG News and Hate) is sent alone again to glm-4.5-flash and glm-4.5-air, with a new repetition index so that each
    request is a real call.
V2  served model names: the model name returned by the vendor for each V1 call, and for five tiny requests per
    requested model (glm-4.5-flash, glm-4.5-air, glm-4.7) under each of the two accounts.
V3  throughput of batching: DS-Flash labels 1,000 AG News tuples one per call and with B = 10 and B = 50, at the same
    concurrency (6 requests in flight), with a new repetition index; tuples per second = tuples / wall-clock time.
Usage: python block_v.py V1|V2|V3
Output: results/main/V1__<model>.csv, V2_served.csv, V3__ds-flash.csv"""
import os, sys
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot, llm
from tasks import TASKS, load

HERE = pilot.HERE
M = f"{HERE}/results/main"
REP = 31


def v1(model):
    E4 = pd.read_csv(f"{HERE}/results/pilot/E4__glm-flash.csv")

    def work(k):
        task, tid = k
        spec, df = TASKS[task], load(task)
        try:
            r = llm.chat(model, pilot.prompt_single(spec, df.text[tid]), max_tokens=2000, rep=REP, retries=12)
        except RuntimeError:
            return dict(task=task, tid=tid, status="error", served=None, pred=-3, gold=int(df.gold[tid]))
        return dict(task=task, tid=tid, status=r["status"], served=r.get("served"), gold=int(df.gold[tid]),
                    pred=pilot.parse(r["text"], spec["labels"]) if r["status"] == "ok" else -2,
                    pilot_status=E4.set_index(["task", "tid"]).status.get((task, tid)))

    with ThreadPoolExecutor(pilot.WORKERS.get(model, 1)) as ex:
        rows = list(ex.map(work, list(zip(E4.task, E4.tid))))
    d = pd.DataFrame(rows); d["model"] = model
    d.to_csv(f"{M}/V1__{model}.csv", index=False)
    print(model, d.groupby(["task", "status"]).size().to_dict(), "served", d.served.value_counts().to_dict(),
          "agree with pilot", round(float((d.status == d.pilot_status).mean()), 4), flush=True)


def v2():
    """Served model names of the two Zhipu accounts for the requested pilot models, five tiny requests each."""
    import requests, time
    rows = []
    for acct in ["glm", "glm2"]:
        cfg = llm.KEYS[acct]
        for name in ["glm-4.5-flash", "glm-4.5-air", "glm-4.7"]:
            for i in range(5):
                r = requests.post(cfg["base_url"].rstrip("/") + "/chat/completions", timeout=120,
                                  headers={"Authorization": "Bearer " + cfg["api_key"], "User-Agent": "Mozilla/5.0"},
                                  json={"model": name, "messages": [{"role": "user", "content": f"Reply with the number {i}."}],
                                        "max_tokens": 20, "temperature": 0, "thinking": {"type": "disabled"}})
                d = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
                rows.append(dict(account=acct, requested=name, http=r.status_code, served=d.get("model"),
                                 when=time.strftime("%Y-%m-%d %H:%M")))
    d = pd.DataFrame(rows); d.to_csv(f"{M}/V2_served.csv", index=False)
    print(d.groupby(["account", "requested", "http", "served"], dropna=False).size(), flush=True)


def v3():
    out = []
    for B, fmt in [(1, "single"), (10, "batch"), (50, "batch")]:
        x = pilot.run_cell("ds-flash", "agnews", B, fmt, "shuffled", REP, 1000)
        out.append(dict(B=B, fmt=fmt, n=len(x), wall=x.wall.iloc[0], tuples_per_s=len(x) / x.wall.iloc[0],
                        acc=float((x.pred == x.gold).mean())))
    d = pd.DataFrame(out); d.to_csv(f"{M}/V3__ds-flash.csv", index=False); print(d.round(2).to_string(), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "V1":
        for m in ["glm-flash", "glm-air"]:
            v1(m)
    elif sys.argv[1] == "V2":
        v2()
    else:
        v3()
