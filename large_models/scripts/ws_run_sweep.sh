#!/bin/bash
# Runs on the workstation, from ~/phd/biosym_myofull/large_models: the speed sweep of MyoFullBody (metabolic cost, seeds 1-3).
#   ws_run_sweep.sh [N_PARALLEL]        (default 2 speeds at a time)
# Standing of the seeds first, then all speeds 0.8 to 4.5 m/s in 0.1 steps (the paper's eight first), N at a time, niced (one process per speed, mumps, one
# thread). Safe to start again after a restart or a stop: results that exist are skipped (scripts/sweep.py).
cd "$(dirname "$0")/.."
source scripts/ws_env.sh
mkdir -p logs
echo "$(date '+%F %T') runner start (pid $$, $1 parallel)" >> logs/sweep_runner.log
$PY -u scripts/sweep.py standing 1 2 3 --linear-solver mumps >> logs/sweep_standing.log 2>&1
{ printf '%s\n' 0.8 1.0 1.2 1.4 1.6 2.5 3.5 4.5
  for i in $(seq 8 45); do printf '%.1f\n' "$(echo "$i/10" | bc -l)"; done; } | awk '!seen[$0]++' | xargs -P "${1:-2}" -I{} bash -c \
  "nice -n 10 $PY -u scripts/sweep.py walking {} --seeds 1 2 3 --objective metabolic --linear-solver mumps --max-iter 3000 >> logs/sweep_v{}.log 2>&1"
echo "$(date '+%F %T') runner end" >> logs/sweep_runner.log
