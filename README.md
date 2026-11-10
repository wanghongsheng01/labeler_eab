# Artifacts for "Semantic Operators Are Not Tuple Functions: Batching, Physical Order, and Refusals in LLM Labelers"

Hongsheng Wang (Zhejiang University). Repository: <https://github.com/wanghongsheng01/labeler_eab>.

This repository contains the protocol, code, per-tuple outputs, per-call logs and costs needed to reproduce every number, table and figure of the paper. No API key is needed to reproduce the analyses.

## Contents

| Path | What |
|---|---|
| `protocol/benchmark_protocol.md` | the analysis plan with hypotheses H1--H4, written after the pilot study and fixed on 2026-11-03 before the main runs, its change log of every later deviation, and the map of confirmatory and exploratory blocks |
| `code/tasks.py` | the seven public datasets, the uniform samples (fixed seed), label sets and questions |
| `code/fetch_data.py` | downloads the public test splits from the Hugging Face hub into `data/raw/` |
| `code/pilot.py` | prompts (single, indexed batch, JSON), answer parsing, physical orders, one experimental cell |
| `code/main_runs.py` | the protocol blocks A, A4, B, B10, R, R10 and the added blocks P, N, I, R2, K, X of the paper |
| `code/block_d.py` | Block D: group-testing decoding of refusal triggers, singleton confirmation, cross-vendor resend |
| `code/llm.py` | cached OpenAI-compatible client (chat and Responses APIs) that logs the served model, tokens, cost, latency and error codes |
| `code/analyze_main.py` | H1 (composition vs. rerun) and H2 (physical order) |
| `code/analyze_main_a.py` | Block A: batch curves and the format/context decomposition (H4) |
| `code/analyze_extra.py` | query-level instability, the contrast effect, position effects |
| `code/analyze_revision.py` | revised statistics (cluster bootstrap, cluster-robust order tests, Holm families, placebo null for the contrast effect) and the added Blocks P, N, I and R2 -> `results/rev_*.csv` |
| `code/analyze_review2.py` | analyses added after the second internal review (template baseline, original H2 test, contrast by task, group-testing subsets, held-out tuples, Blocks K and X) -> `results/rev2_*.csv` |
| `code/export_calls.py` | rebuilds the per-call log of the main runs from the response cache without network access |
| `code/run_revision.sh` | the fresh-call re-runs and the added blocks (protocol change log v0.2) |
| `code/figures.py`, `code/tables.py` | all figures and the generated tables of the paper |
| `code/analyze_pilot.py`, `code/analyze_stats.py` | analyses of the pilot study |
| `results/main/<block>__<labeler>__<dataset>.csv.gz` | one row per (tuple, run): `tid, pos, pred, missing, status, usd, in_tok, out_tok, cache_hit, call_lat, finish, served, B_eff, gold, task, model, B, fmt, order, rep` |
| `results/main/D__*.csv.gz` | Block D: decoded triggers, singleton confirmation, cross-vendor resend |
| `results/pilot/*.csv.gz` | per-tuple outputs of the pilot study |
| `results/calls.jsonl.gz` | one JSON line per real API call of the whole study, pilot included (199,050 calls): day, time, labeler, served model, status, tokens, list-price USD, latency |
| `results/main/calls_main.jsonl.gz` | one JSON line per call of the main runs: block, cell, call index, tuple ids in prompt order, request SHA-256, status, served model, finish reason, tokens, cost, latency, vendor error body and raw response; rebuilt by `code/export_calls.py`, which also checks every response against the released labels |
| `prompts/` | the exact messages of every prompt format and task, and `SHA256SUMS` of the prompts and of every dataset file; `python code/check_data.py` verifies them |
| `results/archive/` | superseded runs to which the protocol's change log refers (first main runs that reused pilot answers; partial Claude runs at temperature 0) |
| `results/*.csv`, `results/*.txt` | the summaries that the paper reports |

`pred` encodes a label index, `-1` for an unparsable answer, `-2` for a refused call and `-3` for a call that failed after all retries. `order` is `shuffled` (canonical random order), `shuffled1..3` (three further random orders of the same tuples), `sorted` (clustered on the gold label) or `keyed` (clustered on text length). The texts are not redistributed; `tid` indexes the samples drawn by `code/tasks.py`. Texts of refusal triggers are omitted on purpose; the paper describes them by paraphrased topic.

## Reproduce the paper without API keys (minutes on a laptop)

```bash
pip install -r requirements.txt
gunzip -k results/main/*.gz results/pilot/*.gz
python code/fetch_data.py          # public test splits -> data/raw/ (needed for neighbor and trigger analyses)
# Block T data -> data_ext/ (read by code/tasks_ext.py):
#   Abt-Buy (DeepMatcher, textual): tableA.csv, tableB.csv, test.csv from
#     http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Textual/Abt-Buy/exp_data/  saved as abtbuy_<name>.csv
#   Chinese Amazon reviews: test.jsonl of the Hugging Face dataset SetFit/amazon_reviews_multi_zh saved as amazon_zh_test.jsonl
#   DBpedia-14 (Block H): the test split of fancyzhx/dbpedia_14 as one parquet file, saved as dbpedia_test.parquet
python code/check_data.py          # SHA-256 of every data file and prompt against prompts/SHA256SUMS
cd code
python analyze_main.py             # H1, H2  -> results/main_composition.csv, results/main_order.csv
python analyze_main_a.py           # H4      -> results/main_batchcurve.csv, results/main_decomposition.csv
python analyze_extra.py            # Table III, contrast effect, position effect
python analyze_revision.py         # statistics reported in the paper and Blocks P, N, I, R2 -> results/rev_*.csv
python analyze_review2.py          # template baseline, original H2 test, Blocks K and X -> results/rev2_*.csv
python block_d.py                  # H3 and refusal bias (re-uses released Block D calls when no key is configured)
python analyze_xvendor.py          # Table VI (GPT-6, Claude, Qwen-3B, Qwen-7B)   -> results/xv_summary.csv
python analyze_tasks_ext.py        # Block T (entity matching, Chinese, Sema-style) -> results/xt_summary.csv
python analyze_f.py                # Blocks F and H (selective filters, DBpedia as stored) -> results/f_summary.csv, h_summary.csv
python analyze_1109.py             # two groupings suffice for CS (L2); GLM reasoning vs. rerun flips (F5)
python refusal_mc.py               # 20,000 random layouts of the refusal bias (Fig. 4)
python gt_scale.py                 # group-testing decode with synthetic triggers (Section V-D)
python storage_order.py            # how 28 public test splits are stored (downloads them; Section III)
python figures.py && python tables.py   # -> paper/figs/*.pdf, paper/tables/*.tex
```

| Paper element | Script | Output |
|---|---|---|
| Fig. 1, Table III (composition, reruns, permutations) | `analyze_main.py`, `analyze_revision.py`, `analyze_extra.py` | `results/main_composition.csv`, `rev_composition.csv`, `main_querylevel.csv` |
| Fig. 2, Blocks N, K, S, X, I, C (physical order, mechanism) | `analyze_main.py`, `analyze_revision.py`, `analyze_review2.py`, `block_uc.py` | `results/main_order.csv`, `rev_dose.csv`, `rev2_block_k.csv`, `rev2_block_x.csv`, `results/main/C__*.csv.gz` |
| Blocks F and H (selective filters, DBpedia as stored) | `block_f.py`, `block_h.py`, `analyze_f.py` | `results/f_summary.csv`, `h_summary.csv` |
| Table IV, Fig. 3 (format and context) | `analyze_main_a.py` | `results/main_decomposition.csv`, `main_batchcurve.csv` |
| Table V, Fig. 4 (refusals) | `block_d.py`, `refusal_mc.py`, `gt_scale.py` | `results/main_refusal.txt`, `refusal_mc.csv`, `gt_scale.csv` |
| Table VI (added labelers), Block T, Block U | `analyze_xvendor.py`, `analyze_tasks_ext.py`, `block_uc.py` | `results/xv_summary.csv`, `xt_summary.csv`, `results/main/U__*.csv.gz` |
| Block V (throughput, served names, GLM-4.5 refusals) | `block_v.py` | `results/main/V1__*.csv.gz`, `V2_served.csv.gz`, `V3__ds-flash.csv.gz` |
| L2 (two groupings), F5 (GLM reasoning) | `analyze_1109.py` | `results/l2_two_groupings.csv`, `glm_reasoning_rerun.csv` |

## Re-run the experiments (needs API keys)

Create a JSON file outside the repository, e.g. `~/.config/llm_keys.json` (chmod 600), and point `LLM_KEYS_JSON` at it:

```json
{"deepseek": {"api_key": "...", "base_url": "https://api.deepseek.com/v1"},
 "kimi":     {"api_key": "...", "base_url": "https://api.moonshot.cn/v1"},
 "glmplan":  {"api_key": "...", "base_url": "https://open.bigmodel.cn/api/v1", "api": "responses"},
 "gateway":  {"api_key": "...", "base_url": "<any OpenAI-compatible endpoint that serves gpt-6-sol and claude-opus-5-5>"},
 "local":    {"api_key": "none", "base_url": "http://<host>:8080/v1"},
 "local7":   {"api_key": "none", "base_url": "http://<gpu-host>:8080/v1"}}
```

GPT-6 is reported as `gpt-6-sol-nr`: `gpt-6-sol` requested with `reasoning_effort: "none"`, because its default reasons in 80% of the calls (`code/run_gpt_nr.sh`, `code/run_gpt_nr_ext.sh`; the earlier default-reasoning runs remain in `results/main` as `gpt-6-sol`). On 2026-11-10 the first gateway went down and the remaining calls went to a second OpenAI-compatible gateway serving the same model; the served model of every call is logged.
The added labelers of Section V-E use `gateway` (GPT-6, Claude) and `local`, a llama.cpp server
(`python -m llama_cpp.server --model qwen2.5-3b-instruct-q4_k_m.gguf --n_ctx 8192 --seed 0`, the GGUF file from the
`Qwen/Qwen2.5-3B-Instruct-GGUF` repository on Hugging Face, SHA-256 `626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d`), queried one request at a time.
`local7` serves Qwen2.5-7B-Instruct the same way on one NVIDIA A10 GPU (llama-cpp-python 0.3.16 with CUDA 12.4,
`--n_gpu_layers -1`; the two shards `qwen2.5-7b-instruct-q4_k_m-0000{1,2}-of-00002.gguf` of `Qwen/Qwen2.5-7B-Instruct-GGUF`,
SHA-256 `dfce12e3862a5283ccfb88221b48480e58745165de856439950d0f22590580db` and
`539cf93f78e887edea1c04e2d7d8cdaca9d01dae9c9025bcb8accbe29df3d72a`; pass the first shard as `--model`); `code/run_gpu7b.sh`
runs its block on all seven datasets.
The only rerun flip of these greedy labelers (Qwen2.5-3B, Hate, three tuples of one call) comes from the server's
prompt-prefix cache: the first copy followed the availability probe of `main_runs.py`; `code/prefix_probe.py` resends
the call after the probe and after a Hate call and reproduces both answers.

Then, for example:

```bash
cd code
python main_runs.py ds-flash B 2000     # Block B: B in {10, 50} x six orders
python main_runs.py kimi-k2.6 B10 1000  # Block B with B = 10 only
python main_runs.py kimi-k2.6 R10 1000  # rerun baseline
python main_runs.py ds-flash A 2000     # Block A
python block_d.py                       # Block D
./run_revision.sh                       # fresh-call re-runs and Blocks P, N, I, R2
```

Every call is cached under `cache/` (the cache key includes the repetition index, so reruns are real calls) and logged to `cache/usage.log`. A per-labeler daily spending guard (`DAILY_CAP`, default USD 15) stops a run before it overspends. The client accepts any OpenAI-compatible endpoint, so other vendors can be added in `code/llm.py` (`MODELS`, `PRICES`, `FIXED`).

Vendors change their models, filters and prices over time. The released files are the outputs collected between 2026-11-03 and 2026-11-09; the served model of every call is recorded in `served`.

## Datasets and licenses

The code is released under the MIT License (see `LICENSE`). The datasets below keep their own licenses.

IMDB (Maas et al., 2011), AG News and Yelp Review Full (Zhang et al., 2015), TweetEval hate and offensive (Barbieri et al., 2020), CARER emotion (Saravia et al., 2018) and English Amazon reviews (Keung et al., 2020) are downloaded from their public Hugging Face copies under their original terms. This repository redistributes only tuple indices, model outputs and gold label indices.

## Content warning

The Hate and Offensive datasets contain hateful and offensive text.
