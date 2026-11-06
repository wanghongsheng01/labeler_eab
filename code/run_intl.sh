#!/bin/bash
# International labelers through the company gateway (2026-11-06). LLMHUB_API_KEY must be set in the environment.
cd "$(dirname "$0")"
P=${PYTHON:-python}
export CACHE_AFTER=1793698178
( for b in B10 R10 P; do $P main_runs.py gpt-6-sol $b 1000; done; echo "GPT DONE $(date)" ) > ../results/main/intl_gpt.log 2>&1 &
( for b in B10s R10; do TASKS=agnews,emotion,yelp,hate $P main_runs.py claude-opus-5.5 $b 500; done; echo "CLAUDE DONE $(date)" ) > ../results/main/intl_claude.log 2>&1 &
wait
