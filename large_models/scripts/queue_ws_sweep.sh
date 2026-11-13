#!/bin/bash
# Local watchdog of the MyoFullBody speed sweep on the workstation (asm-biomac-ws01).
#
#   caffeinate -is nohup bash scripts/queue_ws_sweep.sh [HH:MM] > logs/queue_ws_sweep.log 2>&1 &
#
# Waits until HH:MM (default 04:00), then every 5 minutes: checks that the workstation answers, starts
# ``scripts/ws_run_sweep.sh`` there when no sweep process is running (after a restart of the workstation the processes are
# gone, and the sweep resumes: results that exist are skipped), and pulls the results and logs into the local project.
# Ends when all 38 speeds x 3 seeds have a result (or a ``.failed.txt``), after a last pull.
# Local copies (nothing of the existing results is touched):
#   results/myofullbody/sweep_ws/metabolic/v<speed>/seed<k>.pkl, summary.csv     results/myofullbody/sweep_ws/standing/
#   logs/ws_sweep/
# Stop it with:  kill $(cat logs/queue_ws_sweep.pid)    (the sweep on the workstation keeps running)
# The Mac must not sleep: keep it on power with the lid open (caffeinate stops idle and system sleep on power).
cd "$(dirname "$0")/.."
WS=ri94mihu@asm-biomac-ws01.aibe.uni-erlangen.de
REMOTE=phd/biosym_myofull/large_models
EXPECTED=114           # 38 speeds (0.8 to 4.5 in 0.1 steps) x 3 seeds
PARALLEL=${PARALLEL:-2}
AT=${1:-04:00}
echo $$ > logs/queue_ws_sweep.pid
log() { echo "$(date '+%F %T') $*"; }

target=$(date -j -f "%Y-%m-%d %H:%M:%S" "$(date +%F) $AT:00" +%s)
[ "$target" -le "$(date +%s)" ] && target=$((target + 86400))
log "queued: starts at $(date -r $target '+%F %T') (in $((target - $(date +%s))) s), $PARALLEL speeds in parallel on the workstation"
while [ "$(date +%s)" -lt "$target" ]; do sleep 30; done      # short sleeps: a wake-up after sleep ends the wait on time
[ -n "$DRY_RUN" ] && { log "dry run: would start now"; exit 0; }

pull() {
  mkdir -p results/myofullbody/sweep_ws/standing logs/ws_sweep
  rsync -au -e "ssh -o BatchMode=yes -o ConnectTimeout=20" "$WS:$REMOTE/results/myofullbody/sweep/metabolic/" results/myofullbody/sweep_ws/metabolic/ 2>&1 | tail -1
  rsync -au -e "ssh -o BatchMode=yes -o ConnectTimeout=20" --include='standing_seed*.pkl' --exclude='*' "$WS:$REMOTE/results/myofullbody/" results/myofullbody/sweep_ws/standing/ 2>&1 | tail -1
  rsync -au -e "ssh -o BatchMode=yes -o ConnectTimeout=20" "$WS:$REMOTE/logs/" logs/ws_sweep/ 2>&1 | tail -1
}

launches=0
while true; do
  if ! ssh -o BatchMode=yes -o ConnectTimeout=20 "$WS" true 2>/dev/null; then
    log "workstation not reachable (restarting?), trying again in 5 min"; sleep 300; continue
  fi
  finished=$(ssh -o BatchMode=yes "$WS" "cd $REMOTE && ls results/myofullbody/sweep/metabolic/v*/seed*.pkl results/myofullbody/sweep/metabolic/v*/seed*.failed.txt 2>/dev/null | wc -l")
  running=$(ssh -o BatchMode=yes "$WS" "pgrep -fc '[s]cripts/sweep.py|[w]s_run_sweep.sh'")
  log "workstation up: $finished of $EXPECTED results, $running sweep processes"
  if [ "${finished:-0}" -ge "$EXPECTED" ]; then pull; log "all results there, pulled to results/myofullbody/sweep_ws/ - done"; break; fi
  if [ "${running:-0}" -eq 0 ]; then
    launches=$((launches + 1))
    if [ "$launches" -gt 60 ]; then log "started 60 times without finishing, giving up"; break; fi
    log "no sweep running: starting it (launch $launches)"
    ssh -o BatchMode=yes "$WS" "cd $REMOTE && (setsid nohup bash scripts/ws_run_sweep.sh $PARALLEL >> logs/sweep_driver.log 2>&1 < /dev/null &)"
  fi
  pull
  sleep 300
done
