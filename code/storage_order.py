"""How public text-classification test splits are stored (2026-11-09; no LLM calls).

For each dataset we download the parquet files of its evaluation split from the Hugging Face hub (through the hf-mirror
endpoint) and measure, in the stored row order: the share of consecutive batches of B = 10 rows that hold a single
class, the same share expected under a random order, and the Spearman correlation between row position and class id.
A split counts as stored clustered on its label when single-class batches are at least five times as frequent as under
a random order. A scan that batches rows in storage order then sees the attribute-correlated layout of Section V-B.
Output: results/storage_order.csv"""
import io, os, sys
import numpy as np, pandas as pd, requests
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HUB = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
UA = {"User-Agent": "Mozilla/5.0"}
B = 10
# (repo, config, split, label column): widely used single-label text-classification benchmarks
SETS = [("stanfordnlp/imdb", "plain_text", "test", "label"), ("fancyzhx/ag_news", "default", "test", "label"),
        ("Yelp/yelp_review_full", "yelp_review_full", "test", "label"), ("fancyzhx/yelp_polarity", "plain_text", "test", "label"),
        ("fancyzhx/amazon_polarity", "amazon_polarity", "test", "label"), ("SetFit/amazon_reviews_multi_en", "default", "test", "label"),
        ("fancyzhx/dbpedia_14", "dbpedia_14", "test", "label"), ("cornell-movie-review-data/rotten_tomatoes", "default", "test", "label"),
        ("dair-ai/emotion", "split", "test", "label"), ("cardiffnlp/tweet_eval", "hate", "test", "label"),
        ("cardiffnlp/tweet_eval", "offensive", "test", "label"), ("cardiffnlp/tweet_eval", "sentiment", "test", "label"),
        ("cardiffnlp/tweet_eval", "irony", "test", "label"), ("cardiffnlp/tweet_eval", "emotion", "test", "label"),
        ("CogComp/trec", "default", "test", "coarse_label"), ("PolyAI/banking77", "default", "test", "label"),
        ("clinc/clinc_oos", "plus", "test", "intent"), ("community-datasets/yahoo_answers_topics", "yahoo_answers_topics", "test", "topic"),
        ("SetFit/20_newsgroups", "default", "test", "label"), ("SetFit/sst5", "default", "test", "label"),
        ("SetFit/sst2", "default", "test", "label"), ("SetFit/enron_spam", "default", "test", "label"),
        ("stanfordnlp/snli", "plain_text", "test", "label"), ("nyu-mll/multi_nli", "default", "validation_matched", "label"),
        ("google-research-datasets/poem_sentiment", "default", "test", "label"), ("SetFit/subj", "default", "test", "label"),
        ("SetFit/TREC-QC", "default", "test", "label"), ("financial_phrasebank", "sentences_allagree", "train", "label"),
        ("cardiffnlp/tweet_topic_single", "tweet_topic_single", "test_2021", "label"), ("SetFit/CR", "default", "test", "label")]


def fetch(repo, config, split, col):
    urls = requests.get(f"{HUB}/api/datasets/{repo}/parquet/{config}/{split}", headers=UA, timeout=60).json()
    if not isinstance(urls, list):
        raise RuntimeError(str(urls)[:100])
    parts = []
    for u in urls:
        u = u.replace("https://huggingface.co", HUB)
        r = requests.get(u, headers=UA, timeout=600); r.raise_for_status()
        parts.append(pd.read_parquet(io.BytesIO(r.content), columns=[col]))
    return pd.concat(parts)[col].values


def main():
    rows = []
    for repo, config, split, col in SETS:
        try:
            y = fetch(repo, config, split, col)
        except Exception as e:
            print("skip", repo, config, str(e)[:120], flush=True); continue
        y = pd.Series(y).astype("category").cat.codes.values
        nb = len(y) // B
        single = np.mean([len(set(y[i * B:(i + 1) * B])) == 1 for i in range(nb)])
        p = np.bincount(y) / len(y); expect = float((p ** B).sum())
        rho = spearmanr(np.arange(len(y)), y)[0]
        rows.append(dict(dataset=f"{repo}:{config}", split=split, rows=len(y), classes=int(y.max() + 1),
                         single_class_batches=single, expected_random=expect, rho=rho,
                         clustered=bool(single >= max(5 * expect, 0.05))))
        print(rows[-1], flush=True)
    D = pd.DataFrame(rows); D.to_csv(f"{HERE}/results/storage_order.csv", index=False)
    print(D.round(3).to_string(), "\nclustered:", int(D.clustered.sum()), "of", len(D))


if __name__ == "__main__":
    main()
