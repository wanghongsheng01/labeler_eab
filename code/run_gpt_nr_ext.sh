#!/bin/bash
# GPT-6 with reasoning disabled on Block T (entity matching, Chinese reviews) and Block C (controlled neighbors).
# LLMHUB_API_KEY and LLMHUB_BASE_URL must be set in the environment.
cd "$(dirname "$0")"
P=${PYTHON:-python}
export CACHE_AFTER=1793698178
TASKS=abtbuy $P main_runs.py gpt-6-sol-nr TE 2000
TASKS=amazon_zh $P main_runs.py gpt-6-sol-nr TZ 2000
$P block_uc.py gpt-6-sol-nr C agnews emotion yelp hate
echo "GPT-NR EXT DONE $(date)"
