#!/bin/bash
# Multi-step agent rollouts with Harbor on Tilde, with the model served on LUMI (lumi/serve.sbatch).
# Tasks come from a local dataset folder (tilde/download_dataset.sh), sandboxes are local Docker containers.
# One command: wait for the LUMI job's router, open an ssh tunnel to it through a LUMI login node,
# run Harbor, close the tunnel when Harbor exits. See docs/harbor_tilde.md.
#
#   tilde/harbor_run.sh --path data/harbor/terminal-bench@2.0 --n-tasks 5 --n-attempts 4 \
#       --lumi-repo /scratch/project_465003062/<you>/rl-rollgen --lumi-run 12345678   # the serve.sbatch job id
#   tilde/harbor_run.sh --resume <job-name> --lumi-repo ... --lumi-run ...
#
# What to run:
#   -p, --path DIR           local dataset (or single task) folder, from tilde/download_dataset.sh (required)
#   --n-tasks N              at most N tasks, after the name filters below (Harbor's -l; default: all)
#   --n-attempts K           attempts per task = samples per prompt (Harbor's -k; default 1)
#   --include-task-name GLOB / --exclude-task-name GLOB
#                            pick tasks by name, repeatable (Harbor's -i / -x)
#   -n, --n-concurrent N     trials at once = Docker containers on this machine (default: 8)
#   --agent NAME             Harbor agent (default: terminus-2)
#   --max-output-tokens N    per agent turn (default: 4096)
#   --max-input-tokens N     default: the server's max_model_len minus --max-output-tokens
#   --n-tasks, --n-attempts, --include/--exclude-task-name are Harbor's own flags, passed through like any
#   other flag not listed here (e.g. --max-retries N: retry failed trials). All of them: harbor run --help
#
# Where the model is (LUMI serve job, reached through an ssh tunnel):
#   --lumi-repo PATH         this repo's clone on LUMI, where serve.sbatch was submitted (required unless --url)
#   --lumi-run JOBID|RUN     the serve job: Slurm job id, or run folder name under logs/ on LUMI
#                            (default: latest_run). A job id works right after sbatch, even while queued.
#   --lumi-host [USER@]HOST  LUMI login node for lookups and the tunnel (default: lumi-uan01.csc.fi;
#                            the user name can also come from ~/.ssh/config). Needs the key in ssh-agent.
#   --wait MIN               max minutes to wait for the job's router (queue + vLLM startup; default 60)
#   --local-port N           local end of the tunnel, 127.0.0.1:N (default 18000; must be free)
#   --api-key-file FILE      file with the vLLM API key (default: ~/.vllm_key)
#   --url URL                use this server directly (base URL ending in /v1): no LUMI lookup, no tunnel
#
# The job:
#   --job-name NAME          results folder logs/harbor/NAME (default: <date>-<time>-<dataset>-<model>)
#   --resume NAME            continue logs/harbor/NAME: keeps finished trials, reruns cancelled ones. Its
#                            settings come from its config.json (only the --lumi-*/--wait/--api-key-file
#                            flags apply); the server must serve the same model.
#   -h, --help               print this header
#   Not accepted: -d/--dataset, -t/--task (use tilde/download_dataset.sh + -p), -o/--jobs-dir (always
#   logs/harbor/; pick the name with --job-name).
#
# Results: logs/harbor/<job-name>/ = Harbor's config.json, result.json, job.log and one folder per
# trial, plus our run.txt. logs/harbor/latest points to the newest job. Stop with Ctrl-C.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
JOBS_DIR="$REPO/logs/harbor"
die() { echo "FATAL: $*" >&2; exit 1; }

# --- flags ---
ORIG_ARGS=("$@")
TASKS="" URL="" API_KEY_FILE="$HOME/.vllm_key" AGENT=terminus-2 N_CONCURRENT=8 MAX_OUTPUT_TOKENS=4096
MAX_INPUT_TOKENS="" JOB_NAME="" RESUME="" PASS=()
LUMI_HOST=lumi-uan01.csc.fi LUMI_REPO="" LUMI_RUN=latest_run WAIT_MIN=60 LOCAL_PORT=18000
RUN_SETTINGS=()   # flags that --resume takes from config.json instead
while (( $# )); do
  case "$1" in
    -h|--help) sed -n '2,/^set -e/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
    -p|--path)           TASKS=$2; RUN_SETTINGS+=("$1"); shift 2 ;;
    --url)               URL=$2; shift 2 ;;
    --api-key-file)      API_KEY_FILE=$2; shift 2 ;;
    --agent)             AGENT=$2; RUN_SETTINGS+=("$1"); shift 2 ;;
    -n|--n-concurrent)   N_CONCURRENT=$2; RUN_SETTINGS+=("$1"); shift 2 ;;
    --max-output-tokens) MAX_OUTPUT_TOKENS=$2; RUN_SETTINGS+=("$1"); shift 2 ;;
    --max-input-tokens)  MAX_INPUT_TOKENS=$2; RUN_SETTINGS+=("$1"); shift 2 ;;
    --job-name)          JOB_NAME=$2; RUN_SETTINGS+=("$1"); shift 2 ;;
    --resume)            RESUME=$2; shift 2 ;;
    --lumi-host)         LUMI_HOST=$2; shift 2 ;;
    --lumi-repo)         LUMI_REPO=$2; shift 2 ;;
    --lumi-run)          LUMI_RUN=$2; shift 2 ;;
    --wait)              WAIT_MIN=$2; shift 2 ;;
    --local-port)        LOCAL_PORT=$2; shift 2 ;;
    -d|--dataset|-t|--task) die "$1: download the dataset with tilde/download_dataset.sh and pass -p data/harbor/<name>@<version>" ;;
    -o|--jobs-dir)       die "$1: jobs always go to logs/harbor/ (choose the folder name with --job-name)" ;;
    *) PASS+=("$1"); shift ;;
  esac
done

[ -r "$API_KEY_FILE" ] || die "cannot read the API key file $API_KEY_FILE (copy ~/.vllm_key from LUMI, or pass --api-key-file)"
API_KEY=$(head -1 "$API_KEY_FILE")
command -v harbor >/dev/null || die "harbor not found: run tilde/install.sh"
docker info >/dev/null 2>&1 || die "docker not usable by $USER (is the daemon running, are you in the docker group?)"

if [ -n "$RESUME" ]; then
  (( ${#RUN_SETTINGS[@]} == 0 )) || die "--resume takes its settings from the job's config.json; drop ${RUN_SETTINGS[*]}"
  JOB_NAME=$RESUME
  JOB_DIR="$JOBS_DIR/$JOB_NAME"
  [ -f "$JOB_DIR/config.json" ] || die "no Harbor job at $JOB_DIR (see ls logs/harbor/)"
  # The job talks to the api_base it was started with, so the server must come back at the same URL.
  read -r OLD_MODEL OLD_URL < <(python3 -c 'import json, sys
a = json.load(open(sys.argv[1]))["agents"][0]
print(a["model_name"].removeprefix("hosted_vllm/"), a["kwargs"]["api_base"])' "$JOB_DIR/config.json")
  [ -z "$URL" ] || [ "$URL" = "$OLD_URL" ] || die "--url $URL differs from the job's api_base $OLD_URL"
  if [[ "$OLD_URL" =~ ^http://127\.0\.0\.1:([0-9]+)/v1$ ]]; then
    LOCAL_PORT=${BASH_REMATCH[1]}   # it was a tunnel: reopen it on the same port
  else
    URL=$OLD_URL                    # it was a --url server: use it directly
  fi
else
  [ -n "$TASKS" ] || die "-p DIR is required (e.g. -p data/harbor/terminal-bench@2.0; see tilde/download_dataset.sh)"
  [ -d "$TASKS" ] || die "$TASKS is not a directory"
  TASKS=$(cd "$TASKS" && pwd)
fi

# --- find the LUMI job's router and tunnel to it (skipped with --url) ---
LUMI_INFO="none (--url)"
if [ -z "$URL" ]; then
  [ -n "$LUMI_REPO" ] || die "--lumi-repo PATH is required (the repo clone on LUMI), or pass --url"
  [[ "$WAIT_MIN" =~ ^[0-9]+$ && "$LOCAL_PORT" =~ ^[0-9]+$ ]] || die "--wait and --local-port take numbers"
  command -v setsid >/dev/null || die "setsid not found (util-linux)"
  # BatchMode: fail instead of asking for a passphrase (a background ssh asking would just hang).
  SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20)
  "${SSH[@]}" "$LUMI_HOST" true \
    || die "cannot ssh to $LUMI_HOST without a prompt. Load your key into an agent first: eval \"\$(ssh-agent)\"; ssh-add ~/.ssh/<lumi key> (and accept the host key once with a plain ssh $LUMI_HOST). See docs/harbor_tilde.md, steps 0b and 0c"

  # Prints "<run folder or -> <router URL or -> <job state: PENDING, RUNNING, ..., or GONE>".
  lumi_lookup() {
    "${SSH[@]}" "$LUMI_HOST" bash -s -- "$(printf '%q' "$LUMI_REPO")" "$(printf '%q' "$LUMI_RUN")" <<'EOF'
cd "$1/logs" 2>/dev/null || { echo "$1/logs does not exist on $(hostname)" >&2; exit 2; }
if [[ "$2" =~ ^[0-9]+$ ]]; then id=$2; run=$(ls -d -- *-"$2" 2>/dev/null | tail -1)
elif [ "$2" = latest_run ]; then run=$(readlink latest_run || true)
else [ -d "$2" ] || { echo "no run folder $1/logs/$2" >&2; exit 2; }; run=$2
fi
id=${id:-${run##*-}}
state=$(squeue -h -j "$id" -o %T 2>/dev/null || true)
echo "${run:--} $(cat "$run/urls/router.txt" 2>/dev/null || echo -) ${state:-GONE}"
EOF
  }

  SECONDS=0 LAST=""
  while :; do
    INFO=$(lumi_lookup) || die "lookup on $LUMI_HOST failed (see above)"
    read -r RUN ROUTER STATE <<< "$INFO"
    [ "$ROUTER" = - ] || break
    if [ "$STATE" = GONE ]; then
      if [ "$RUN" = - ]; then
        [[ "$LUMI_RUN" =~ ^[0-9]+$ ]] && die "LUMI job $LUMI_RUN is not in the queue and has no run folder in $LUMI_REPO/logs/ (wrong job id or --lumi-repo?)"
        die "no LUMI run found for --lumi-run $LUMI_RUN (if your serve job is still queued, pass its job id)"
      fi
      die "LUMI run $RUN has no router and its job is not in the queue: it ended (see logs/$RUN/slurm.out on LUMI). If a new serve job is still queued, pass its job id with --lumi-run"
    fi
    if [ "$INFO" != "$LAST" ]; then
      echo "waiting for the router of LUMI job ${RUN/#-/$LUMI_RUN} ($STATE; up to $WAIT_MIN min)"
      LAST=$INFO
    fi
    (( SECONDS < WAIT_MIN * 60 )) || die "no router URL after $WAIT_MIN min (job $STATE; see logs/${RUN/#-/<run>}/slurm.out on LUMI)"
    sleep 30
  done
  HOSTPORT=${ROUTER#*://}; HOSTPORT=${HOSTPORT%%/*}   # 10.253.0.12:9000
  LUMI_INFO="$LUMI_HOST:$LUMI_REPO/logs/$RUN, router $ROUTER"

  # setsid: Ctrl-C goes to Harbor only, so the tunnel stays up while Harbor cancels trials.
  # ServerAlive*: notice a dead link. ExitOnForwardFailure: fail if the local port is taken.
  # ControlPath=none: own connection, even with ControlMaster in ~/.ssh/config. Through a shared master the
  # forward would belong to the master and outlive this ssh (ControlPersist), keeping the port taken.
  setsid "${SSH[@]}" -N -o ControlPath=none -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    -L "127.0.0.1:$LOCAL_PORT:$HOSTPORT" "$LUMI_HOST" &
  TUNNEL_PID=$!
  trap 'kill "$TUNNEL_PID" 2>/dev/null || true' EXIT
  URL="http://127.0.0.1:$LOCAL_PORT/v1"
  echo "tunnel: $URL -> $ROUTER (LUMI run $RUN) via $LUMI_HOST"
fi

# --- ask the server for the model name and context length (also checks the tunnel and the key) ---
for (( try = 1; ; try++ )); do
  MODELS=$(curl -sS --fail-with-body --max-time 10 -H "Authorization: Bearer $API_KEY" "$URL/models" 2>&1) && break
  if [ -n "${TUNNEL_PID:-}" ] && ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
    die "the ssh tunnel exited (local port $LOCAL_PORT in use? pass --local-port; 'administratively prohibited' = the login node forbids forwarding)"
  fi
  (( try < 6 )) || die "$URL/models failed (LUMI job ended? wrong API key?): $MODELS"
  sleep 5
done
read -r MODEL CTX < <(python3 -c 'import json,sys; m=json.load(sys.stdin)["data"][0]; print(m["id"], m.get("max_model_len") or 0)' <<< "$MODELS")

if [ -n "$RESUME" ]; then
  [ "$MODEL" = "$OLD_MODEL" ] || die "the server serves '$MODEL', but job $JOB_NAME used '$OLD_MODEL'"
  CMD=(harbor jobs resume --job-path "$JOB_DIR" "${PASS[@]}")
else
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
fi

# --- record the run (Harbor ignores plain files in the job dir; appended on every (re)start) ---
mkdir -p "$JOB_DIR"
ln -sfn "$JOB_NAME" "$JOBS_DIR/latest"
{
  echo "=== $(date -Is) on $(hostname)${RESUME:+ (resume)}"
  echo "command:     $0 $(printf '%q ' "${ORIG_ARGS[@]:-}")"
  echo "harbor:      $(harbor --version) | $(printf '%q ' "${CMD[@]}")"
  [ -n "$RESUME" ] || echo "tasks:       $TASKS ($(find "$TASKS" -mindepth 1 -maxdepth 1 -type d | wc -l) entries)"
  echo "server:      $URL, model '$MODEL', max_model_len $CTX, api key file: $API_KEY_FILE"
  echo "lumi:        $LUMI_INFO"
  echo "git commit:  $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "git status:"
  git -C "$REPO" status --short 2>/dev/null | sed 's/^/  /'
  echo "git diff:"
  git -C "$REPO" diff HEAD 2>/dev/null
} >> "$JOB_DIR/run.txt"

if [ -n "$RESUME" ]; then echo "harbor: resuming $JOB_DIR, model $MODEL at $URL"
else echo "harbor: $AGENT on $TASKS, model hosted_vllm/$MODEL ($MAX_INPUT_TOKENS in / $MAX_OUTPUT_TOKENS out) at $URL"
fi
echo "results: $JOB_DIR"
export HOSTED_VLLM_API_KEY="$API_KEY"   # read by LiteLLM
rc=0
"${CMD[@]}" || rc=$?
echo "harbor exited with $rc; results: $JOB_DIR${TUNNEL_PID:+ (closing the tunnel)}"
exit "$rc"
