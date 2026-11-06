#!/bin/bash
# Open-weight deterministic labeler on the local CPU server (2026-11-06); set LOCAL_LLM_URL.
cd "$(dirname "$0")"
export CACHE_AFTER=1793698178
for t in agnews emotion yelp hate; do TASKS=$t ${PYTHON:-python} main_runs.py qwen2.5-3b L 500; done
echo "LOCAL DONE $(date)"
