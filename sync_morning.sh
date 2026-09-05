#!/bin/bash
# Morning sync: pull every overnight result, then regenerate figures+tables.
set -e
cd "$(dirname "$0")"
export MSYS2_ARG_CONV_EXCL='*'
mkdir -p ../results/innovation ../results/fullsup
python server_exec.py "mkdir -p /tmp/x" > /dev/null
# full-sup runs
for f in train_metrics_v2_paviau_s10000_inj_seed0.json train_metrics_v2_indianpines_s10000_inj_seed0.json train_metrics_paviau_s10000_seed0_ctx1.json pred_map_v2_paviau_s10000_inj_seed0.npy pred_map_v2_indianpines_s10000_inj_seed0.npy; do
  python server_exec.py --get /home/ubuntu/spectrum_pilot/results/$f ../results/fullsup/$f 2>/dev/null || echo "missing $f"
done
# innovation queue outputs
for f in $(python server_exec.py "ls /home/ubuntu/spectrum_pilot/results/innovation/ 2>/dev/null" | grep json | tr -d '\r'); do
  python server_exec.py --get "/home/ubuntu/spectrum_pilot/results/innovation/$f" "../results/innovation/$f" 2>/dev/null || true
done
python make_figures.py
echo "=== sync done ==="
ls ../results/fullsup ../results/innovation
