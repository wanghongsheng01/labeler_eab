#!/bin/bash
# Second open-weight deterministic labeler, Qwen2.5-7B-Instruct on one A10 GPU (2026-11-07); set LOCAL7_LLM_URL.
cd "$(dirname "$0")"
export CACHE_AFTER=1793698178
for t in agnews emotion yelp hate imdb offensive amazon; do TASKS=$t ${PYTHON:-python} main_runs.py qwen2.5-7b L 500; done
echo "GPU7B DONE $(date)"
