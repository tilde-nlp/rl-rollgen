#!/bin/bash
# Multi-step agent rollouts with Harbor on Tilde. Tasks come from a local dataset folder
# (tilde/download_dataset.sh), sandboxes are local Docker containers, and the model is a vLLM server
# reached over HTTP (usually a LUMI serve.sbatch job forwarded by tilde/tunnel.sh). See HARBOR_EXAMPLE.md.
#
#   tilde/harbor_run.sh -p data/harbor/terminal-bench@2.0 --api-key-file ~/.vllm_key -l 5 -k 4
#
# Flags handled here (everything else goes to `harbor run`, e.g. -l/--n-tasks, -k/--n-attempts,
# -i/-x task-name globs, --max-retries):
#   -p, --path DIR           local dataset or task folder (required; registry datasets -d/-t are not used)
#   --url URL                vLLM base URL ending in /v1 (default: first URL in logs/tunnel.urls)
#   --api-key-file FILE      file with the vLLM API key (default: none)
#   --agent NAME             Harbor agent (default: terminus-2)
#   -n, --n-concurrent N     concurrent trials = Docker containers on this machine (default: 8)
#   --max-output-tokens N    per agent turn (default: 4096)
#   --max-input-tokens N     default: the server's max_model_len minus --max-output-tokens
#   --job-name NAME          default: <date>-<time>-<dataset>-<model>
#
# Results: logs/harbor/<job-name>/ = Harbor's config.json, result.json, job.log and one folder per
# trial, plus our run.txt. logs/harbor/latest points to the newest job. Rerunning with the same
# --job-name and the same settings resumes that job (finished trials are kept).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
JOBS_DIR="$REPO/logs/harbor"
die() { echo "FATAL: $*" >&2; exit 1; }

# --- flags ---
ORIG_ARGS=("$@")
TASKS="" URL="" API_KEY_FILE="" AGENT=terminus-2 N_CONCURRENT=8 MAX_OUTPUT_TOKENS=4096
MAX_INPUT_TOKENS="" JOB_NAME="" PASS=()
while (( $# )); do
  case "$1" in
    -h|--help) sed -n '2,/^set -e/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
    -p|--path)           TASKS=$2; shift 2 ;;
    --url)               URL=$2; shift 2 ;;
    --api-key-file)      API_KEY_FILE=$2; shift 2 ;;
    --agent)             AGENT=$2; shift 2 ;;
    -n|--n-concurrent)   N_CONCURRENT=$2; shift 2 ;;
    --max-output-tokens) MAX_OUTPUT_TOKENS=$2; shift 2 ;;
    --max-input-tokens)  MAX_INPUT_TOKENS=$2; shift 2 ;;
    --job-name)          JOB_NAME=$2; shift 2 ;;
    -d|--dataset|-t|--task) die "$1: download the dataset with tilde/download_dataset.sh and pass -p data/harbor/<name>@<version>" ;;
    -o|--jobs-dir)       die "$1: jobs always go to logs/harbor/ (choose the folder name with --job-name)" ;;
    *) PASS+=("$1"); shift ;;
  esac
done

[ -n "$TASKS" ] || die "-p DIR is required (e.g. -p data/harbor/terminal-bench@2.0; see tilde/download_dataset.sh)"
[ -d "$TASKS" ] || die "$TASKS is not a directory"
TASKS=$(cd "$TASKS" && pwd)
if [ -z "$URL" ]; then
  [ -s "$REPO/logs/tunnel.urls" ] || die "no --url and no logs/tunnel.urls: start tilde/tunnel.sh first"
  URL=$(cut -d, -f1 "$REPO/logs/tunnel.urls")   # Harbor talks to ONE server
fi
API_KEY=EMPTY
if [ -n "$API_KEY_FILE" ]; then
  [ -r "$API_KEY_FILE" ] || die "cannot read --api-key-file $API_KEY_FILE"
  API_KEY=$(< "$API_KEY_FILE")
fi
command -v harbor >/dev/null || die "harbor not found: run tilde/install.sh"
docker info >/dev/null 2>&1 || die "docker not usable by $USER (is the daemon running, are you in the docker group?)"

# --- ask the server for the model name and context length (also checks the tunnel and the key) ---
MODELS=$(curl -sS --fail-with-body --max-time 10 -H "Authorization: Bearer $API_KEY" "$URL/models") \
  || die "$URL/models failed (tunnel down? LUMI job ended? wrong API key?): $MODELS"
read -r MODEL CTX < <(python3 -c 'import json,sys; m=json.load(sys.stdin)["data"][0]; print(m["id"], m.get("max_model_len") or 0)' <<< "$MODELS")
if [ -z "$MAX_INPUT_TOKENS" ]; then
  (( CTX > 0 )) || die "server did not report max_model_len; pass --max-input-tokens"
  MAX_INPUT_TOKENS=$(( CTX - MAX_OUTPUT_TOKENS ))
fi
(( MAX_INPUT_TOKENS > 0 )) || die "max input tokens $MAX_INPUT_TOKENS <= 0 (context $CTX, output $MAX_OUTPUT_TOKENS)"
: "${JOB_NAME:=$(date +%Y%m%d-%H%M%S)-$(basename "$TASKS")-${MODEL##*/}}"
JOB_DIR="$JOBS_DIR/$JOB_NAME"

# Terminus-2 wants model_info for hosted_vllm models (context limits; zero cost).
MODEL_INFO=$(printf '{"max_input_tokens":%d,"max_output_tokens":%d,"input_cost_per_token":0,"output_cost_per_token":0}' \
  "$MAX_INPUT_TOKENS" "$MAX_OUTPUT_TOKENS")
CMD=(harbor run
  --path "$TASKS"
  --env docker
  --agent "$AGENT"
  --model "hosted_vllm/$MODEL"
  --ak "api_base=$URL"
  --ak "model_info=$MODEL_INFO"
  --n-concurrent "$N_CONCURRENT"
  --jobs-dir "$JOBS_DIR"
  --job-name "$JOB_NAME"
  "${PASS[@]}")

# --- record the run (Harbor ignores plain files in the job dir; appended on resume) ---
mkdir -p "$JOB_DIR"
ln -sfn "$JOB_NAME" "$JOBS_DIR/latest"
{
  echo "=== $(date -Is) on $(hostname)"
  echo "command:     $0 $(printf '%q ' "${ORIG_ARGS[@]:-}")"
  echo "harbor:      $(harbor --version) | $(printf '%q ' "${CMD[@]}")"
  echo "tasks:       $TASKS ($(find "$TASKS" -mindepth 1 -maxdepth 1 -type d | wc -l) entries)"
  echo "server:      $URL, model '$MODEL', max_model_len $CTX, api key file: ${API_KEY_FILE:-none}"
  echo "tunnel:      $(cat "$REPO/logs/tunnel.run" 2>/dev/null || echo unknown) (LUMI run dir)"
  echo "git commit:  $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "git status:"
  git -C "$REPO" status --short 2>/dev/null | sed 's/^/  /'
  echo "git diff:"
  git -C "$REPO" diff HEAD 2>/dev/null
} >> "$JOB_DIR/run.txt"

echo "harbor: $AGENT on $TASKS, model hosted_vllm/$MODEL ($MAX_INPUT_TOKENS in / $MAX_OUTPUT_TOKENS out) at $URL"
echo "results: $JOB_DIR"
export HOSTED_VLLM_API_KEY="$API_KEY"   # read by LiteLLM
exec "${CMD[@]}"
