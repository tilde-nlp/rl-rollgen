#!/bin/bash
# Token throughput of a job's vLLM servers, from the stats line vLLM logs about every 10 s
# ("Avg prompt throughput: ... tokens/s, Avg generation throughput: ..., Running: ..., Waiting: ...,
# GPU KV cache usage: ...%"). Works on a login node, during the job or after it.
#
#   bash lumi/throughput.sh                          # logs/latest_run
#   bash lumi/throughput.sh logs/20261108-141557-local-40964
#
# Arguments: [RUN_DIR] a job's log folder (default: logs/latest_run).
# Prints:
#   1. per server: averages over the samples where it was busy (Running > 0)
#   2. all servers together, per hour, idle time included (drops show gaps, e.g. between files)
# Reading it: Running near --concurrency (default 64) with Waiting 0 = the client is the limit (raise
# --concurrency); Waiting > 0 or KV cache near 100% = the servers are full.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
RUN_DIR=${1:-$REPO/logs/latest_run}
[ -d "$RUN_DIR" ] || { echo "FATAL: no run folder $RUN_DIR" >&2; exit 1; }
cd "$RUN_DIR"
shopt -s nullglob
LOGS=(*-vllm-*.log)
(( ${#LOGS[@]} )) || { echo "FATAL: no *-vllm-*.log in $RUN_DIR" >&2; exit 1; }
STATS="Avg prompt throughput"
grep -q "$STATS" "${LOGS[@]}" || { echo "FATAL: no '$STATS' lines in the vLLM logs (server still starting?)" >&2; exit 1; }

echo "Run: $(pwd -P)"
echo
echo "Per server (busy samples only):"
for f in "${LOGS[@]}"; do
  grep -h "$STATS" "$f" | sed -E 's/.*prompt throughput: ([0-9.]+).*generation throughput: ([0-9.]+).*Running: ([0-9]+).*Waiting: ([0-9]+).*KV cache usage: ([0-9.]+)%.*/\1 \2 \3 \4 \5/' |
    awk -v f="${f%.log}" '$3>0 {n++; p+=$1; g+=$2; r+=$3; w+=$4; kv+=$5}
      END {if (n) printf "  %-20s %5d samples  prompt %8.0f tok/s  gen %6.0f tok/s  running %5.1f  waiting %6.1f  KV %5.1f%%\n", f, n, p/n, g/n, r/n, w/n, kv/n
           else printf "  %-20s never busy\n", f}'
done

echo
echo "All servers together, per hour (idle included):"
grep "$STATS" "${LOGS[@]}" | sed -E 's/^([^:]+):.* ([0-9]{2}-[0-9]{2} [0-9]{2}):[0-9]{2}:[0-9]{2} .*prompt throughput: ([0-9.]+).*generation throughput: ([0-9.]+).*/\1 \2 \3 \4/' |
  awk '{k=$2" "$3; p[k,$1]+=$4; g[k,$1]+=$5; n[k,$1]++; hours[k]; files[$1]}
    END {for (h in hours) {P=G=0; for (f in files) if (n[h,f]) {P+=p[h,f]/n[h,f]; G+=g[h,f]/n[h,f]}
         printf "  %s:00  prompt %9.0f tok/s  gen %7.0f tok/s\n", h, P, G}}' | sort
