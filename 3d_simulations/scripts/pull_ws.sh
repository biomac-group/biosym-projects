#!/usr/bin/env bash
# Pull results/logs/configs from the workstation into this project (no deletes, newer wins).
set -euo pipefail
cd "$(dirname "$0")/.."
REMOTE=ri94mihu@asm-biomac-ws01.aibe.uni-erlangen.de:phd/biosym_ws_fixed/3d_simulations
rsync -au --stats "$REMOTE/results/" results/
rsync -au --stats "$REMOTE/logs/" logs/
rsync -au --stats "$REMOTE/configs/" configs/
