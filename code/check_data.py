"""Write (--write) or verify the frozen prompt templates and the SHA-256 of every dataset file (2026-11-09).

prompts/<task>__<format>.txt holds the exact messages of each prompt format with the placeholder texts <TEXT 1> ...
<TEXT B>; prompts/SHA256SUMS lists their hashes and the hashes of the data files read by tasks.py and tasks_ext.py, so
that an update of a dataset or a prompt is detected before any analysis.
Usage: python check_data.py [--write]"""
import hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot

HERE = pilot.HERE
P = os.path.join(HERE, "prompts")
RAW = ["agnews_test", "amazon_test", "emotion_test", "imdb_test", "tweet_hate_test", "tweet_offensive_test", "yelp_test"]
EXT = ["abtbuy_tableA.csv", "abtbuy_tableB.csv", "abtbuy_test.csv", "amazon_zh_test.jsonl", "dbpedia_test.parquet"]
FORMATS = {"single": lambda s: pilot.prompt_single(s, "<TEXT 1>"),
           "batch": lambda s: pilot.prompt_batch(s, ["<TEXT 1>", "<TEXT 2>", "<TEXT 3>"]),
           "batch_xml": lambda s: pilot.prompt_batch_xml(s, ["<TEXT 1>", "<TEXT 2>", "<TEXT 3>"]),
           "json": lambda s: pilot.prompt_json(s, ["<TEXT 1>", "<TEXT 2>", "<TEXT 3>"])}


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def files():
    root = os.environ.get("DATA_RAW") or next((r for r in [os.path.join(HERE, "data", "raw"),
                                                     os.path.join(os.path.dirname(HERE), "data", "raw")] if os.path.isdir(r)), "")
    out = {f"data/raw/{f}.parquet": os.path.join(root, f + ".parquet") for f in RAW}
    out.update({f"data_ext/{f}": os.path.join(HERE, "data_ext", f) for f in EXT})
    return out


def main(write):
    os.makedirs(P, exist_ok=True)
    sums = {}
    for task, spec in pilot.TASKS.items():
        if task in ("rma", "abtbuy"):
            continue
        for fmt, make in FORMATS.items():
            name = f"{task}__{fmt}.txt"
            text = json.dumps(make(spec), ensure_ascii=False, indent=1)
            if write:
                open(os.path.join(P, name), "w").write(text + "\n")
            sums[f"prompts/{name}"] = hashlib.sha256((text + "\n").encode()).hexdigest()
    for rel, path in files().items():
        if os.path.exists(path):
            sums[rel] = sha(path)
    if write:
        open(os.path.join(P, "SHA256SUMS"), "w").write("".join(f"{v}  {k}\n" for k, v in sorted(sums.items())))
        print("wrote", len(sums), "hashes")
        return
    ref = dict(l.split()[::-1] for l in open(os.path.join(P, "SHA256SUMS")))
    bad = [k for k, v in sums.items() if ref.get(k) != v]
    print("checked", len(sums), "files;", "all match" if not bad else f"MISMATCH: {bad}")


if __name__ == "__main__":
    main("--write" in sys.argv)
