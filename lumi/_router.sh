#!/bin/bash
# Runs INSIDE the container on the first node (started by start_router in lumi/_common.sh), after all
# vLLM servers are healthy:
#   1. starts vllm-router (lumi/install_router.sh) in front of all servers in SERVER_URLS
#   2. waits until a 1-token completion through it works, and (with --api-key-file) that a request
#      without the key is refused
#   3. writes its URL to $RUN_DIR/urls/router.txt and keeps routing until killed
# Policy ($ROUTER_POLICY, launcher flag --router-policy): round_robin for generate/judge/classify (even
# spread), cache_aware for serve.sbatch (requests that share a prompt prefix, e.g. the turns of one agent
# trial, go to the same server and reuse its prefix cache; otherwise the least loaded server gets it).
# Request timeout 3600s, like the rollgen client's (the router's default 1800s could cut long generations).
# Arguments: none (not run by hand).
# Environment: SERVER_URLS, ROUTER_POLICY, API_KEY_FILE, ROUTER_DIR, ROUTER_PORT, RUN_DIR, SERVED_NAME.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
: "${SERVER_URLS:?SERVER_URLS not set (this script is started by lumi/*.sbatch, not by hand)}"
export PYTHONNOUSERSITE=1 PYTHONPATH="$ROUTER_DIR"
NODE_ADDR=$(node_addr)
log() { echo "[router] $*"; }

mapfile -t WORKERS < <(tr ',' '\n' <<< "$SERVER_URLS" | sed -E 's#/v1/?$##' | grep .)   # router wants base URLs
API_KEY=""
if [ -n "${API_KEY_FILE:-}" ]; then API_KEY=$(head -1 "$API_KEY_FILE"); fi
# Read by the launcher below; keeps the key out of `ps` output and logs.
export ROUTER_API_KEY="$API_KEY" ROUTER_KEY_CHECK_URL="${WORKERS[0]}/v1/models"

# --- 1. start ---
# Same as the `vllm-router` command, plus the API key. vllm-router's own --api-key only sets the key
# it sends to the servers; checking the key of incoming requests (api_key_validation_urls: the
# client's key must get a 200 from that URL, here a vLLM server's /v1/models) has no CLI flag in
# vllm-router 0.1.16, so we pass both to its Rust core directly.
python - --worker-urls "${WORKERS[@]}" --policy "$ROUTER_POLICY" --request-timeout-secs 3600 \
  --host 0.0.0.0 --port "$ROUTER_PORT" \
  > "$RUN_DIR/router.log" 2>&1 <<'EOF' &
import os, sys
from vllm_router import router
key = os.environ["ROUTER_API_KEY"]
if key:
    _Router = router._Router
    router._Router = lambda **kw: _Router(**{**kw, "api_key": key,
                                             "api_key_validation_urls": [os.environ["ROUTER_KEY_CHECK_URL"]]})
from vllm_router.launch_router import main
sys.argv[0] = "vllm-router"
main()
EOF
PID=$!
trap 'kill "$PID" 2>/dev/null || true' EXIT
trap 'exit 143' TERM INT HUP

# --- 2. check ---
# HTTP status of a 1-token completion through the router (0 = no connection). $1 = API key or "".
status() {
  python - "$1" <<'EOF'
import json, os, sys, urllib.error, urllib.request
key = sys.argv[1]
req = urllib.request.Request(
    f"http://127.0.0.1:{os.environ['ROUTER_PORT']}/v1/completions",
    data=json.dumps({"model": os.environ["SERVED_NAME"], "prompt": "Hi", "max_tokens": 1}).encode(),
    headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})})
try:
    print(urllib.request.urlopen(req, timeout=60).status)
except urllib.error.HTTPError as e:
    print(e.code)
except OSError:
    print(0)
EOF
}
SECONDS=0
until [ "$(status "$API_KEY")" = 200 ]; do
  kill -0 "$PID" 2>/dev/null || die "[router] vllm-router died during startup, see $RUN_DIR/router.log"
  (( SECONDS < 300 )) || die "[router] no working completion through vllm-router after ${SECONDS}s (last HTTP status $(status "$API_KEY")), see $RUN_DIR/router.log"
  sleep 5
done
if [ -n "$API_KEY" ]; then
  code=$(status "")
  [[ "$code" == 401 || "$code" == 403 ]] \
    || die "[router] vllm-router answered $code to a request WITHOUT the API key; refusing to publish it (it would open the servers to the LUMI network)"
else
  log "WARNING: no --api-key-file: anyone on the LUMI network can use the router while the job runs"
fi

# --- 3. publish, then route until killed ---
URL="http://$NODE_ADDR:$ROUTER_PORT/v1"
VERSION=$(basename "$(ls -d "$ROUTER_DIR"/vllm_router-*.dist-info)" .dist-info)
echo "router:      ${VERSION/-/ } ($ROUTER_POLICY) at $URL over ${#WORKERS[@]} server(s)" >> "$RUN_DIR/run.txt"
echo "$URL" > "$RUN_DIR/urls/router.tmp"
mv "$RUN_DIR/urls/router.tmp" "$RUN_DIR/urls/router.txt"
log "ready after ${SECONDS}s: $URL -> ${WORKERS[*]}"
wait "$PID"
die "[router] vllm-router exited unexpectedly, see $RUN_DIR/router.log"
