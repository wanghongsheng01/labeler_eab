# Benchmark Protocol (draft v0.1, 2026-11-03)

Working title: *Semantic Operators Are Not Tuple Functions: Batching, Order, Refusals, and Non-Determinism in Commercial LLM Labelers*.
This protocol fixes every choice before the main runs so that the paper reports pre-registered contrasts. Changes after the freeze (target 2026-12-15) are logged in Section 10 with a reason.

## 1. Scope

An LLM labeler is a *physical semantic operator*: a model, a vendor endpoint, a prompting mode and a batch size that maps a tuple to a label from a closed set. We study it inside three operator types:
- filter (`sem_filter`): a binary predicate;
- map/classify (`sem_map`): a closed label set;
- aggregate: COUNT, AVG or GROUP BY over a semantic attribute.

All tasks are closed-set classification or rating, so quality is measured exactly against gold labels.

## 2. Labelers

| Vendor | Request model id | Modes | Notes |
|---|---|---|---|
| DeepSeek | `deepseek-flash`, `deepseek-v4-pro` | thinking off; thinking on (`reasoning_effort` low / high) | temperature 0 |
| Zhipu | `glm-4.5-flash`, `glm-4.5-air`, `glm-5.3` | thinking off | temperature 0. Log the served model: `glm-4.5-air` is served as `glm-5.3-flash` (observed 2026-11-03) |
| Moonshot | `kimi-k2.6`, `kimi-k3` | thinking off | vendor forces temperature 0.6 |
| International (C2, pending) | at least two tiers of one of OpenAI, Google or Anthropic | thinking off / on | a direct vendor API is preferred over the gateway, because the gateway hides prices and adds a routing layer |

For every call we record the response's `model` field (the served model) next to the requested id. A labeler whose served model changes during the study is split into two labelers.

## 3. Prompts

- **P-single**: the free-form single-tuple prompt (verbatim from the pilot, `code/pilot.py::prompt_single`).
- **P-batch**: the indexed `[k] answer` prompt (`prompt_batch`), used for every B ≥ 1, including B = 1 as the format control.
- **P-json** (new): the same content, asking for a JSON array `[{"k": 1, "label": "..."}]`. Malformed JSON counts as a parse failure; we do not repair it.

Parsing is the pilot parser, unchanged: exact match first, then the earliest whole-word label match; anything else is unparsed. Prompts are frozen as text files with SHA-256 hashes in the artifact.

## 4. Datasets and samples

- **Current**: IMDB, Yelp, AG News, CARER Emotion, TweetEval hate, TweetEval offensive, Amazon reviews (n = 2000 each; offensive n = 860), sampled exactly as in `ICDE2027/code/tasks.py` (seed 20261103).
- **To add (≥ 3, licences to be checked before inclusion)**:
  - one semantic-filter task taken from the SemBench / LOTUS / Palimpzest evaluations;
  - one entity-matching pair task (e.g. Abt-Buy), as a semantic join;
  - one attribute-extraction task with a closed value set;
  - one long-document task.
- **Ordering**:
  - the canonical order is a random permutation (seed 0) of the sample;
  - "sorted" orders the same tuples by gold label;
  - "keyed" orders them by an attribute-independent key (text length);
  - "shuffled-s" uses permutation seeds s = 1, 2, 3.

## 5. Factors and cells

| Block | RQ | Factors | Cells per (labeler, dataset) |
|---|---|---|---|
| A. Batch curve | RQ1 | B ∈ {1, 5, 10, 20, 50, 100} × {P-single (B = 1 only), P-batch, P-json} | 13 |
| B. Composition and order | RQ2 | B ∈ {10, 50} × order ∈ {canonical, shuffled-1..3, sorted, keyed} | 12 |
| C. Repeats | RQ3 | 5 repeats of {P-single B = 1, P-batch B = 10}; 3 temperatures on one cheap labeler | 10 (+3) |
| D. Refusals | RQ4 | all refused calls, bisected down to the single trigger tuples; triggers re-sent to the other vendors | variable |
| E. Reasoning | RQ5 | thinking {off, low, high} × B ∈ {1, 10} | 6 |
| F. Throughput and price | RQ6 | concurrency ∈ {1, 4, 16} for one dataset; prefix-cache-friendly layout vs. the default layout | 4 |
| G. Drift | — | block A at B ∈ {1, 10}, rerun at +4 and +8 weeks | 4 |
| H. Gold audit | RQ7 | 300 tuples on which every labeler agrees against gold, re-judged by two humans | — |
| I. Remedies | RQ8 | shuffle before batching; bisect-and-resend; vendor fallback for triggers; majority-of-3 reruns | 4 |

Rate-limited vendors (Zhipu, Moonshot) run blocks A–C on n = 1000 tuples per dataset; all other cells use the full sample.

## 6. Logging schema (one JSON line per call)

`day, t_start, latency_s, vendor, requested_model, served_model, request_sha256, prompt_id, B, order, rep, temperature, thinking, status ∈ {ok, refused, error}, http_code, vendor_error_code, finish_reason, prompt_tokens, completion_tokens, reasoning_tokens, cache_hit_tokens, usd_list, usd_billed, raw_text`

Plus one row per (call, tuple): `tid, pos, pred, missing, gold`.
- The cache key includes `rep`, so repeats are real calls.
- Refusals are cached as results.
- Errors (429 or 5xx after all retries) are not cached; they are retried in a repair pass and counted.

## 7. Metrics

- **Quality**: accuracy against the original gold and against the audited gold (block H); macro-F1 for imbalanced tasks.
- **Stability**:
  - per-tuple flip rate across reps;
  - pairwise disagreement between runs;
  - composition sensitivity = the share of tuples whose label changes when only the batch neighbours change;
  - order effect = the accuracy difference and the GROUP BY L1 error difference between sorted and canonical order.
- **Availability**: tuple-level and call-level refusal rates; trigger rate q; blast radius (tuples lost per trigger); missing, misaligned and unparsed rates.
- **Cost**: USD per 1k tuples at list price and as billed (cache discounts applied); fit of in_tok(B) = c + a/B.
- **Time**: tuples per second at fixed concurrency; P50 and P99 latency; 429 count and retry overhead.
- **Query level**:
  - filter result-set Jaccard across reps and orders;
  - relative error of COUNT, AVG and GROUP BY against gold;
  - bias introduced by dropping refused tuples.

## 8. Statistics

- Paired accuracy contrasts use the exact McNemar test. Composition and flip rates get bootstrap 95% CIs over tuples (2000 resamples).
- Labeler rankings get Kendall τ with a bootstrap CI.
- Multiple comparisons are handled with Holm correction inside each block; we report raw and adjusted p.
- The pilot effects are the pre-registered hypotheses:
  - H1: composition sensitivity exceeds the rerun flip rate, for every labeler that supports batching;
  - H2: label-sorted order changes accuracy on at least half of the non-trivial tasks;
  - H3: refusal triggers are not missing at random with respect to the gold label;
  - H4: format and context effects are each significant on at least one (labeler, task) pair, with opposite dominance across vendors.

## 9. Budget and safety guards

- Each process reads today's spend per model from the call log and stops at $15 per model per day, inside the shared $20 cap.
- GLM and Kimi run at 1–3 workers.
- Keys are only read from the secrets file or environment variables; they are never written into the repository or the artifact.
- Content-filter trigger texts are reported by class and topic only, never quoted in the paper.

## 10. Change log

- v0.1 (2026-11-03): first draft after the pilot (41,120 calls, $9.76).
- v0.2 (2026-11-05): changes after the main runs, made in response to an internal mock review. Each change is listed with its reason; the hypotheses H1–H4 above are unchanged.
  1. **Cache reuse corrected.** The client caches answers by request. Because the main runs reused the pilot's prompts, 62,360 of the 518,720 per-tuple rows (12.0%) of v0.1's main results, in cells of DS-Flash Blocks A, B and R, DS-Pro A4 and Kimi B10 and R10, were answers cached during the pilot. All affected cells were re-run with fresh calls (`CACHE_AFTER` = the time this protocol was first saved, 2026-11-03 17:29:38 +08:00). The v0.1 results are kept in `results/main_v1_with_pilot_cache/` for comparison.
  2. **Statistics.** Composition and rerun rates now get cluster-bootstrap intervals over canonical batches. The order effect is measured against the mean of the four random compositions, and is tested with a cluster-robust z-test over the batches of the clustered order, with Holm correction over all order contrasts. Block A effects get Holm correction within each family; the original tests are kept as secondary results.
  3. **Blocks added:** P (same batch members, permuted positions), N (partly attribute-correlated keys: gold label plus Gaussian noise with s ∈ {0.5, 1, 2}), I (prompt that states that the texts are unrelated and that labels may repeat), and R2 (a second rerun one day later).
  4. **Blocks not run** for the ICDE 2027 R2 submission: C (repeats and temperatures), E (reasoning), F (throughput), G (drift), H (gold audit), I-remedies other than the prompt remedy, and the international vendor.
  5. **Final execution (v0.2).** With the re-runs and the added blocks, the study made 139,379 calls ($46.29 at list prices, GLM flat-rate calls excluded); the main runs, including the added blocks, made 97,691 calls ($36.53) and produced 637,312 per-tuple labels. All calls ran between 2026-11-03 and 2026-11-05.
  6. **Status of this document.** The header of v0.1 calls it a draft and names a freeze target of 2026-12-15, the date planned for a later submission. The main runs used v0.1 exactly as saved at 17:29:38 (+08:00) on 2026-11-03, after the last pilot call (17:25:37) and before the first main call (17:35:49), according to the per-call log. The protocol has no external timestamp, so the paper does not call it a pre-registration.
  7. **Planned items that were not run** (in addition to item 4): the additional tasks of Section 4 (a semantic-filter task from the SemBench, LOTUS or Palimpzest evaluations, entity matching such as Abt-Buy, attribute extraction, a long-document task); Block A for GLM and Kimi (only DS-Flash and DS-Pro were run, so the cross-vendor part of H4 cannot be tested); bisection in Block D, replaced by group-testing decoding over the existing runs plus a confirmation of every candidate sent alone.
  8. **Confirmatory tuples.** The main runs used the same seed, tuples and prompts as the pilot for the first 1,000 tuples of the canonical permutation. DS-Flash also labels tuples 1,001–2,000, which the pilot never used; the paper reports H1 and the sorted-order effect separately on these held-out tuples.
- v0.3 (2026-11-05, late evening): changes after a second internal mock review.
  1. **Blocks added:** K (DS-Flash on the first 1,000 tuples clustered on DS-Pro's canonical predicted label, a realistic materialized-column key that does not use the gold label) and X (DS-Flash with a delimited template that wraps every text in `<text id="k">…</text>`, orders canonical, shuffled1 and sorted, on AG News, Emotion, Yelp and Hate).
  2. **Reporting.** Table IV now uses the revised tests (format: exact McNemar; context: z-test clustered by call); H2 is reported under both the revised and the originally planned test; the query-level table counts only valid labels and adds the rerun baseline; trigger texts were removed from all released logs; the per-call log of the main runs (`results/main/calls_main.jsonl.gz`) now records cell, tuples, request hash, raw response and the vendor's error body.
  3. **Final execution (v0.3).** The study made 142,461 calls ($47.32 at list prices, of which $0.70 for metered GLM calls in the pilot; flat-rate GLM-5.3-Flash calls are not priced). After the protocol was saved, the main runs, the added blocks and Block D made 101,341 calls ($37.57). The main runs and added blocks produced 668,460 per-tuple labels. Item 5 of v0.2 is superseded.
- v0.4 (2026-11-06): additions to address the remaining limitations; the four labelers and hypotheses of the plan are unchanged.
  1. **Block A for GLM and Kimi** (single, batch with B ∈ {1, 10, 50}, 1,000 tuples, all datasets), so the cross-vendor part of H4 can be tested; H4 is now evaluated on 28 (labeler, dataset) pairs.
  2. **Added labelers, reported separately and not pooled with the 35 planned cells:** GPT-6-Sol and Claude-Opus-5.5 through the company gateway (an OpenAI-compatible router authorized by the author on 2026-11-06; Gemini is not offered there), and the open-weight Qwen2.5-3B-Instruct (Q4_K_M GGUF, SHA-256 626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d) served by llama.cpp on a 4-core CPU server with greedy decoding and one request at a time. Claude rejects `temperature` on part of the gateway, so it runs with its default sampling; the first Claude attempt at temperature 0 was discarded (`results/main_claude_temp0_partial/`). Designs: GPT-6 Blocks B10, R10, P (7 datasets, 1,000 tuples); Claude canonical + 2 random + sorted + keyed + rerun (AG News, Emotion, Yelp, Hate; 500 tuples); Qwen canonical + 1 random + sorted + keyed + perm1 + rerun (500 tuples).
  3. **Block S:** AG News clustered on a real metadata key, the news agency named in the text (DS-Flash, 2,000 tuples).
- v0.5 (2026-11-06, evening): Block T on DS-Flash, extending the protocol beyond English classification: (1) entity matching on the Abt-Buy test pairs of the DeepMatcher benchmark (1,916 pairs) with our batch template and with BatchER's zero-shot question-batching template taken from its code (github.com/fmh1art/BatchER, src/prompting.py); (2) 2,000 Chinese Amazon reviews (SetFit copy of the Multilingual Amazon Reviews Corpus, test split, file size checked against the Hugging Face listing); (3) a Sema-style template (tuples as a JSON array, answers forced into a JSON array by the API's `response_format`), following Sema's description of its batch prompting (arXiv 2603.11622, Sec. 5.1), on AG News, Emotion, Yelp and Hate. Orders: canonical, shuffled1, sorted, keyed (not for the Sema-style template), rerun. A second open-weight labeler (Qwen2.5-7B-Instruct, Q4_K_M, on an A10 GPU, greedy, one request at a time) is set up as `qwen2.5-7b`.
- v0.6 (2026-11-06, night): after a third internal review. (1) Composition sensitivity and rerun flip rates now count valid labels only (unparsed, refused and failed answers are not labels); the H1 count is unchanged (35/35, 32 significant). (2) Block U: fresh groupings (seed 9) sent twice at the same moment on DS-Flash and DS-Pro (AG News, Emotion, Yelp, Hate), a rerun without prefix caching. (3) Block C: 500 targets per task placed among nine same-class, other-class or random neighbors (DS-Flash; same four tasks), a direct test of the contrast mechanism. (4) Refusal bias in expectation over 20,000 random layouts (`code/refusal_mc.py`) replaces single-run ratios. (5) Two-way clustered robustness check of H1 and H2 (`analyze_review2.py twoway`).
  6. The Qwen2.5-7B labeler announced in v0.5 was not run on 2026-11-06 afternoon: the GPU server's link (about 0.1–0.7 MB/s from the mirror used then) could not fetch the 4.7 GB model file in time (superseded by v0.7).
- v0.7 (2026-11-07, 00:30): Qwen2.5-7B-Instruct run as announced in v0.5. Model from ModelScope (`Qwen/Qwen2.5-7B-Instruct-GGUF`, Q4_K_M, two shards, SHA-256 dfce12e3… and 539cf93f…, identical to the Hugging Face listing), served by llama-cpp-python 0.3.16 (the version that serves Qwen2.5-3B) with the CUDA 12.4 build on one NVIDIA A10, all layers on the GPU, `--n_ctx 8192 --seed 0`, greedy decoding, one request at a time, reached through an SSH tunnel. Design: the Qwen block L (canonical, shuffled1, sorted, keyed, perm1, rerun; 500 tuples) on all seven datasets, 23:37 on 2026-11-06 to 00:24 on 2026-11-07, 2,100 calls (`code/run_gpu7b.sh`). Reported in Table VI next to Qwen2.5-3B, not pooled with the 35 planned cells. Its rerun flip rate is 0 on all seven datasets.
- v0.8 (2026-11-07, 10:30): Qwen2.5-3B block L finished on Yelp and Hate (01:38 and 03:35 on 2026-11-07), as planned in v0.5; Qwen-3B now covers AG News, Emotion, Yelp and Hate. Its rerun flip rate is 0 on three tasks and 0.6% on Hate: three tuples of one call. The canonical copy of that call followed the availability probe ("Say OK") and the server reused 5 cached prompt tokens; the rerun followed a Hate call and reused 66. Resending the call after the probe and after a Hate call reproduces both answers twice each (`code/prefix_probe.py`). Reported as measured, not removed.
