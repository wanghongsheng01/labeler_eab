#!/bin/bash
# Block A (A4 cells) for Kimi and GLM, so that the cross-vendor part of H4 can be tested (2026-11-06).
# GLM needs GLM3_KEY in the environment.
cd "$(dirname "$0")"
P=${PYTHON:-python}
export CACHE_AFTER=1793698178
( $P main_runs.py kimi-k2.6 A4 1000; echo "KIMI A DONE $(date)" ) > ../results/main/blockA_kimi.log 2>&1 &
( for i in 1 2 3; do $P main_runs.py glm-5.3-flash A4 1000; sleep 900; done; echo "GLM A DONE $(date)" ) > ../results/main/blockA_glm.log 2>&1 &
wait
