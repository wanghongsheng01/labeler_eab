#!/bin/sh
# Revision runs after the simulated review (2026-11-05). CACHE_AFTER = time the protocol was fixed, so that no
# pilot-era cached answer enters the main runs; main-era cache entries are reused (identical requests).
cd "$(dirname "$0")"
PY=../../.venv/bin/python
export CACHE_AFTER=1793698178
run() { $PY -W ignore main_runs.py "$@"; }
for pass in 1 2 3 4 5 6; do
  run ds-flash B 2000; run ds-flash R 2000; run ds-flash A 2000
  run ds-pro A4 1000
  run kimi-k2.6 B10 1000; run kimi-k2.6 R10 1000
  run ds-flash P 2000; run ds-pro P 1000; run kimi-k2.6 P 1000
  TASKS=agnews,emotion,yelp,hate run ds-flash N 2000
  TASKS=agnews,emotion,yelp,hate run ds-flash I 2000
  run ds-flash R2 2000; run ds-pro R2 1000
  n=$(ls ../results/main/*.csv | grep -v partial | wc -l)
  echo "pass $pass done, files $n $(date)"
  [ "$n" -ge 109 ] && break
  sleep 600
done
echo "REVISION DONE $(date)"
