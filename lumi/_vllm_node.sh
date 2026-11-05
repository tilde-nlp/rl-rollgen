#!/bin/bash
# Runs INSIDE the container, one copy per node (started by start_vllm_servers in lumi/_common.sh):
#   1. stages the model to $NODE_TMP in /tmp (node-local RAM; much faster to load than from Lustre)
#   2. starts GPUS_PER_NODE / TP independent vLLM servers on this node
#   3. waits until they answer /health, then writes their URLs to $RUN_DIR/urls/<node>.txt
#   4. keeps serving until killed (the batch script kills it when the client is done)
# Server config (MODEL, SERVED_NAME, TP, VLLM_ARGS, STAGE_MODEL) comes from the environment,
# set by parse_launcher_args in lumi/_common.sh from the sbatch flags.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
apply_env
: "${MODEL:?MODEL not set (this script is started by lumi/*.sbatch, not by hand)}"

NODE="${SLURMD_NODENAME:-$(hostname)}"
# Address other nodes use to reach this one: the Slingshot interface if present.
: "${NODE_ADDR:=$(ip -4 -o addr show hsn0 2>/dev/null | awk '{print $4}' | cut -d/ -f1)}"
: "${NODE_ADDR:=$NODE}"
mkdir -p "$RUN_DIR"
log() { echo "[$NODE] $*"; }

mapfile -t GPUS < <(tr ',' '\n' <<< "$HIP_VISIBLE_DEVICES" | grep .)
(( ${#GPUS[@]} == GPUS_PER_NODE )) || die "[$NODE] expected $GPUS_PER_NODE GPUs, got ${#GPUS[@]} (HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES)"

# --- 1. stage model ---
MODEL_PATH="$MODEL"
if [ "$STAGE_MODEL" = 1 ] && [ -d "$MODEL" ]; then
  MODEL_PATH="$NODE_TMP/model/$(basename "$MODEL")"
  mkdir -p "$MODEL_PATH"
  SECONDS=0
  ls "$MODEL" | xargs -P 16 -I{} cp -rL "$MODEL/{}" "$MODEL_PATH/"
  log "staged $MODEL -> $MODEL_PATH in ${SECONDS}s"
fi

PIDS=()
cleanup() {
  local rc=$?
  if (( ${#PIDS[@]} )); then kill "${PIDS[@]}" 2>/dev/null || true; wait "${PIDS[@]}" 2>/dev/null || true; fi
  rm -rf "$NODE_TMP"   # staged model + caches
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 143' TERM INT HUP

# --- 2. start servers ---
NUM_SERVERS=$(( GPUS_PER_NODE / TP ))
for (( i = 0; i < NUM_SERVERS; i++ )); do
  port=$(( BASE_PORT + i ))
  gpus=$(IFS=,; echo "${GPUS[*]:i*TP:TP}")
  log "server $i: GPUs $gpus, port $port"
  # shellcheck disable=SC2086
  HIP_VISIBLE_DEVICES="$gpus" vllm serve "$MODEL_PATH" \
    --served-model-name "$SERVED_NAME" \
    --tensor-parallel-size "$TP" \
    --host 0.0.0.0 --port "$port" \
    $VLLM_ARGS > "$RUN_DIR/$NODE-vllm-$i.log" 2>&1 &
  PIDS+=($!)
done

# --- 3. wait for health, publish URLs ---
healthy() { python -c 'import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=5)' "$1" 2>/dev/null; }
SECONDS=0
for (( i = 0; i < NUM_SERVERS; i++ )); do
  until healthy "http://127.0.0.1:$(( BASE_PORT + i ))/health"; do
    kill -0 "${PIDS[i]}" 2>/dev/null || die "[$NODE] vLLM server $i died during startup, see $RUN_DIR/$NODE-vllm-$i.log"
    sleep 10
  done
done
for (( i = 0; i < NUM_SERVERS; i++ )); do echo "http://$NODE_ADDR:$(( BASE_PORT + i ))/v1"; done > "$RUN_DIR/urls/$NODE.tmp"
mv "$RUN_DIR/urls/$NODE.tmp" "$RUN_DIR/urls/$NODE.txt"   # atomic: the batch script never reads a partial file
log "$NUM_SERVERS server(s) healthy after ${SECONDS}s at $NODE_ADDR"

# --- 4. serve until killed; exit (and fail the step) if any server dies ---
wait -n "${PIDS[@]}"
die "[$NODE] a vLLM server exited unexpectedly, see $RUN_DIR/$NODE-vllm-*.log"
