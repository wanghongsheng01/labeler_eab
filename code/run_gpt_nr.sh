#!/bin/bash
# GPT-6 with reasoning disabled (reasoning_effort "none", accepted by the gateway; checked 2026-11-09), Blocks B, R and P
# on all seven datasets. LLMHUB_API_KEY and LLMHUB_BASE_URL must be set in the environment.
cd "$(dirname "$0")"
P=${PYTHON:-python}
export CACHE_AFTER=1793698178
for b in B10 R10 P; do $P main_runs.py gpt-6-sol-nr $b 1000; done
echo "GPT-NR DONE $(date)"
