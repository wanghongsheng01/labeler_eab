"""Tasks added on 2026-11-06 to widen the task coverage beyond English text classification.

- abtbuy:    entity matching on the Abt-Buy product pairs (DeepMatcher benchmark, textual version, test split:
             1,916 pairs, 206 matches); one tuple is a pair "Product A is ... / Product B is ...".
- amazon_zh: Chinese Amazon review stars (Multilingual Amazon Reviews Corpus, SetFit copy, test split).
- dbpedia:   DBpedia-14 topics (test split of fancyzhx/dbpedia_14, 70,000 rows, SHA-256 of the parquet b956f5c6...),
             added 2026-11-09 because the distributed test split is stored sorted by class (Block H).
Files live in ../data_ext (see README); samples use the same seed rule as tasks.py.
"""
import json, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(HERE, "data_ext")
SEED = 20261103

TASKS_EXT = {
    "abtbuy": dict(labels=["different", "same"],
                   question="Do Product A and Product B refer to the same real-world product?"),
    "amazon_zh": dict(labels=["1", "2", "3", "4", "5"], numeric=True,
                      question="How many stars (1-5) did the author of this Chinese Amazon product review most likely give?"),
    "dbpedia": dict(labels=["company", "educational institution", "artist", "athlete", "office holder",
                            "mean of transportation", "building", "natural place", "village", "animal", "plant", "album",
                            "film", "written work"],
                    question="What kind of entity does this Wikipedia article describe?"),
}


def _product(r):
    s = f"{r['name']}"
    if isinstance(r.get("description"), str) and r["description"].strip():
        s += f" {r['description'].strip()}"
    if isinstance(r.get("price"), (int, float)) and not np.isnan(r["price"]):
        s += f" (price {r['price']})"
    return s[:600]


def load_ext(name, n=2000):
    if name == "abtbuy":
        a = pd.read_csv(os.path.join(D, "abtbuy_tableA.csv")).set_index("id")
        b = pd.read_csv(os.path.join(D, "abtbuy_tableB.csv")).set_index("id")
        t = pd.read_csv(os.path.join(D, "abtbuy_test.csv"))
        text = [f"Product A is {_product(a.loc[l].to_dict())}\nProduct B is {_product(b.loc[r].to_dict())}"
                for l, r in zip(t.ltable_id, t.rtable_id)]
        return pd.DataFrame({"tid": np.arange(len(t)), "text": text, "gold": t.label.astype(int).values})
    if name == "amazon_zh":
        rows = [json.loads(l) for l in open(os.path.join(D, "amazon_zh_test.jsonl"))]
        df = pd.DataFrame(rows)[["text", "label"]]
        idx = np.random.default_rng(SEED).permutation(len(df))[: min(n, len(df))]
        df = df.iloc[np.sort(idx)].reset_index(drop=True)
        return pd.DataFrame({"tid": np.arange(len(df)), "text": df.text.astype(str).str.slice(0, 2500),
                             "gold": df.label.astype(int).values})
    if name == "dbpedia":  # uniform sample; tids keep the stored row order, which is sorted by class
        df = pd.read_parquet(os.path.join(D, "dbpedia_test.parquet"))
        idx = np.sort(np.random.default_rng(SEED).permutation(len(df))[:n])
        df = df.iloc[idx].reset_index(drop=True)
        text = (df.title.astype(str) + ". " + df.content.astype(str).str.strip()).str.slice(0, 2500)
        return pd.DataFrame({"tid": np.arange(len(df)), "text": text, "gold": df.label.astype(int).values})
    raise KeyError(name)
