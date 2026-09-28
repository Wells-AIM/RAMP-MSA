#!/bin/bash
# Run experiment files one after another (after any running grid manager finishes).
# usage: bash tools/chain.sh "SEEDS" exps/a.txt exps/b.txt ...   (SEEDS e.g. 1111,2222,3333)
cd "$(dirname "$0")/.."
PY=/media/disk3/muxy/envs/HME/bin/python
SEEDS=$1; shift
WAIT_FOR=${WAIT_FOR:-"^$PY tools/grid.py"}
while pgrep -f "$WAIT_FOR" >/dev/null; do sleep 60; done
for f in "$@"; do
  echo "$(date +%H:%M:%S) chain: $f"
  $PY tools/grid.py "$f" --seeds "$SEEDS" --per_gpu 2
done
echo "$(date +%H:%M:%S) chain: ALL DONE"
