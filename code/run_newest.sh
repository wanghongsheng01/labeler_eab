#!/bin/bash
# Newest models on the company gateway (2026-11-10): Block L (canonical, one further random order, sorted, keyed, one
# permutation, rerun; 500 tuples) on AG News, Emotion, Yelp and Hate, each at its lowest accepted reasoning setting.
# LLMHUB_API_KEY and LLMHUB_BASE_URL must be set in the environment.
cd "$(dirname "$0")"
P=${PYTHON:-python}
export CACHE_AFTER=1793698178
for m in "$@"; do
  ( TASKS=agnews,emotion,yelp,hate $P main_runs.py $m L 500; echo "NEWEST DONE $m $(date)" ) > ../results/main/newest_$m.log 2>&1 &
done
wait
