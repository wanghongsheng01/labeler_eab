#!/bin/bash
# Other task types and published batch templates (2026-11-06), DS-Flash.
cd "$(dirname "$0")"
P=${PYTHON:-python}
export CACHE_AFTER=1793698178
TASKS=abtbuy $P main_runs.py ds-flash TE 2000
TASKS=amazon_zh $P main_runs.py ds-flash TZ 2000
TASKS=agnews,emotion,yelp,hate $P main_runs.py ds-flash TJ 2000
echo "TASKS EXT DONE $(date)"
