#!/bin/bash
# background monitor: poll server training status every 5 min, log locally
cd "$(dirname "$0")"
for i in $(seq 1 120); do
  ts=$(date "+%H:%M:%S")
  st=$(python server_exec.py "tail -1 /home/ubuntu/spectrum_pilot/results/fullsup_pavia_run.log 2>/dev/null | tr -d '\r'; ls /tmp/fullsup_pavia.flag >/dev/null 2>&1 && echo PAVIA_DONE; tail -1 /home/ubuntu/spectrum_pilot/results/fullsup_ip_run.log 2>/dev/null | tr -d '\r'; ls /tmp/fullsup_ip.flag >/dev/null 2>&1 && echo IP_DONE" 2>/dev/null | tr '\n' ' | ')
  echo "[$ts] $st" >> ../results/fullsup_watch.log
  if echo "$st" | grep -q "IP_DONE"; then echo "[$ts] ALL_DONE" >> ../results/fullsup_watch.log; break; fi
  sleep 300
done
