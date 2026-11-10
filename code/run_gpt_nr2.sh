#!/bin/bash
# GPT-6 without reasoning, every block through the second gateway only (one channel per labeler; 2026-11-10).
# LLMHUB_API_KEY and LLMHUB_BASE_URL (second gateway) must be set in the environment.
cd "$(dirname "$0")"
P=${PYTHON:-python}
export CACHE_AFTER=1793698178
for b in B10 R10 P; do $P main_runs.py gpt-6-sol-nr2 $b 1000; done
$P block_h.py gpt-6-sol-nr2
$P block_uc.py gpt-6-sol-nr2 C agnews emotion yelp hate
TASKS=abtbuy $P main_runs.py gpt-6-sol-nr2 TE 2000
TASKS=amazon_zh $P main_runs.py gpt-6-sol-nr2 TZ 2000
echo "GPT-NR2 DONE $(date)"
