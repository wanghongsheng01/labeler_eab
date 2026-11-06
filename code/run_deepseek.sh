#!/bin/bash
# Supervisor for the DeepSeek main runs: retry every 15 min while the vendor is overloaded (503), for up to 12 h.
# Each pass skips finished task files and replays cached calls, so retries cost nothing extra.
cd "$(dirname "$0")"
PY=../../.venv/bin/python
LOG=../results/main/logs/deepseek_supervisor.log
for pass in $(seq 1 48); do
  echo "pass $pass $(date '+%F %T')" >> $LOG
  # composition/order and rerun blocks first (they test H1/H2), then the batch curves
  $PY -W ignore main_runs.py ds-flash B >> ../results/main/logs/ds-flash.log 2>&1
  $PY -W ignore main_runs.py ds-flash R >> ../results/main/logs/ds-flash.log 2>&1
  $PY -W ignore main_runs.py ds-pro B10 1000 >> ../results/main/logs/ds-pro.log 2>&1
  $PY -W ignore main_runs.py ds-pro R10 1000 >> ../results/main/logs/ds-pro.log 2>&1
  $PY -W ignore main_runs.py ds-pro A4 1000 >> ../results/main/logs/ds-pro.log 2>&1 &
  $PY -W ignore main_runs.py ds-flash A >> ../results/main/logs/ds-flash.log 2>&1
  wait
  n=$(ls ../results/main/{B,R,A}__ds-flash__*.csv ../results/main/{B10,R10,A4}__ds-pro__*.csv 2>/dev/null | wc -l)
  echo "  finished task files: $n / 42" >> $LOG
  [ "$n" -ge 42 ] && { echo "all done $(date '+%F %T')" >> $LOG; exit 0; }
  sleep 900  # vendor down or cap reached: wait
done
