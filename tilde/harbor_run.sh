#!/bin/bash
# Run Harbor (multi-step agent rollouts) on Tilde: sandboxes are local Docker containers, the model
# is a vLLM server reached over HTTP (usually a LUMI job forwarded by tilde/tunnel.sh).
#
#   tilde/tunnel.sh &                                  # or in another terminal / tmux
#   tilde/harbor_run.sh -d terminal-bench@2.0 -n 8     # any extra args go to `harbor run`
#   tilde/harbor_run.sh -p path/to/tasks --n-attempts 4
#
# Results: Harbor job dirs under $JOBS_DIR (trajectories, verifier rewards, logs).
# Note: Harbor sends all LLM calls to ONE base URL, so only the first of VLLM_URLS is used.
# To use several servers, run several harbor_run.sh with different HARBOR_URL and task subsets.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
die() { echo "FATAL: $*" >&2; exit 1; }

# --- config (override by exporting) ---
: "${VLLM_URLS:=$(cat "$REPO/logs/tunnel.urls" 2>/dev/null || true)}"   # written by tilde/tunnel.sh
: "${HARBOR_URL:=${VLLM_URLS%%,*}}"   # the one server Harbor talks to
: "${VLLM_API_KEY:=EMPTY}"            # only matters if vLLM was started with --api-key
: "${AGENT:=terminus-2}"
: "${ENV_TYPE:=docker}"
: "${N_CONCURRENT:=8}"                # concurrent trials = concurrent containers on this machine
: "${MAX_OUTPUT_TOKENS:=4096}"        # per agent turn
: "${JOBS_DIR:=$REPO/logs/harbor}"
: "${HARBOR_VERSION:=0.20.0}"         # used only if `harbor` isn't on PATH (then run via uvx)
# SERVED_NAME and MAX_INPUT_TOKENS default to what the server reports in /v1/models.

[ -n "$HARBOR_URL" ] || die "no vLLM URL: run tilde/tunnel.sh first, or export VLLM_URLS / HARBOR_URL"
command -v docker >/dev/null || die "docker not found (Harbor's docker environment needs it)"
if command -v harbor >/dev/null; then HARBOR=(harbor)
elif command -v uvx >/dev/null; then HARBOR=(uvx --from "harbor==$HARBOR_VERSION" harbor)
else die "neither harbor nor uvx found: install uv (https://docs.astral.sh/uv/)"; fi

# --- ask the server for the model name and context length (also checks the tunnel works) ---
MODELS=$(curl -sf --max-time 10 -H "Authorization: Bearer $VLLM_API_KEY" "$HARBOR_URL/models") \
  || die "$HARBOR_URL/models not reachable (tunnel down? LUMI job ended?)"
read -r name ctx < <(python3 -c 'import json,sys; m=json.load(sys.stdin)["data"][0]; print(m["id"], m.get("max_model_len") or 0)' <<< "$MODELS")
: "${SERVED_NAME:=$name}"
: "${MAX_INPUT_TOKENS:=$ctx}"
(( MAX_INPUT_TOKENS > 0 )) || die "server did not report max_model_len; export MAX_INPUT_TOKENS"

# Terminus-2 wants model_info for hosted_vllm models (context limits; zero cost).
MODEL_INFO=$(printf '{"max_input_tokens":%d,"max_output_tokens":%d,"input_cost_per_token":0,"output_cost_per_token":0}' \
  "$MAX_INPUT_TOKENS" "$MAX_OUTPUT_TOKENS")

echo "harbor: agent=$AGENT env=$ENV_TYPE model=hosted_vllm/$SERVED_NAME ($MAX_INPUT_TOKENS ctx) at $HARBOR_URL"
mkdir -p "$JOBS_DIR"
export HOSTED_VLLM_API_KEY="$VLLM_API_KEY"   # read by LiteLLM
exec "${HARBOR[@]}" run \
  --env "$ENV_TYPE" \
  --agent "$AGENT" \
  --model "hosted_vllm/$SERVED_NAME" \
  --ak "api_base=$HARBOR_URL" \
  --ak "model_info=$MODEL_INFO" \
  --n-concurrent "$N_CONCURRENT" \
  --jobs-dir "$JOBS_DIR" \
  "$@"
