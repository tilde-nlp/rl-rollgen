# Shared LUMI code, sourced by lumi/*.sbatch (on the host) and lumi/_vllm_node.sh (inside the container).
# Users don't need to read this; the entry points are lumi/*.sbatch.

# --- site config (rarely changes; override by exporting before sbatch if you must) ---
set -a
: "${SIF:=/appl/local/laifs/containers/lumi-multitorch-u24r72f21m50t211-20261028_090706/lumi-multitorch-plus-u24r72f21m50t211-20261028_090706.sif}"
: "${BINDS:=/var/spool/slurmd,/scratch/project_465003062,/scratch/project_465002038}"
: "${HF_HOME:=/scratch/project_465003062/environment/containers/fsdp-skyrl-harbor/hf}"
: "${GPUS_PER_NODE:=8}"
: "${BASE_PORT:=8000}"
# --- per-job paths (on shared storage) ---
: "${REPO:=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
: "${RUN_DIR:=$REPO/logs/${SLURM_JOB_NAME:-local}-${SLURM_JOB_ID:-$$}}"   # run.txt, vLLM logs, server URLs
set +a

die() { echo "FATAL: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Host side (called from lumi/*.sbatch)
# ---------------------------------------------------------------------------

LAUNCHER_HELP='Launcher flags (consumed here, everything else goes to the Python script):
  --model PATH            model dir or HF hub id (required; hub ids must already be in HF_HOME)
  --served-name NAME      name used in API requests (default: basename of --model)
  --tp N                  tensor parallel size per vLLM server (default 1); must divide 8
  --vllm-args "ARGS"      extra args for `vllm serve`, e.g. "--max-model-len 16384 --gpu-memory-utilization 0.9"
  --no-stage-model        load the model from Lustre instead of copying it to node-local /tmp first
  --startup-timeout SEC   max wait for all vLLM servers to become healthy (default 1800)
Node count, time limit, partition: regular sbatch options, e.g. `sbatch --nodes=4 --time=04:00:00 ...`.'

# Usage: parse_launcher_args <python script> "$@"
# Sets and exports the server config; leaves the remaining args in CLIENT_ARGS.
# `--help` prints the launcher flags plus the Python script's own --help, then exits.
parse_launcher_args() {
  local script=$1; shift
  MODEL="" SERVED_NAME="" TP=1 VLLM_ARGS="" STAGE_MODEL=1 STARTUP_TIMEOUT=1800
  CLIENT_ARGS=()
  local want_help=0
  while (( $# )); do
    case "$1" in
      --model)              MODEL=$2; shift 2 ;;
      --model=*)            MODEL=${1#*=}; shift ;;
      --served-name)        SERVED_NAME=$2; shift 2 ;;
      --served-name=*)      SERVED_NAME=${1#*=}; shift ;;
      --tp)                 TP=$2; shift 2 ;;
      --tp=*)               TP=${1#*=}; shift ;;
      --vllm-args)          VLLM_ARGS=$2; shift 2 ;;
      --vllm-args=*)        VLLM_ARGS=${1#*=}; shift ;;
      --no-stage-model)     STAGE_MODEL=0; shift ;;
      --startup-timeout)    STARTUP_TIMEOUT=$2; shift 2 ;;
      --startup-timeout=*)  STARTUP_TIMEOUT=${1#*=}; shift ;;
      -h|--help)            want_help=1; shift ;;
      *)                    CLIENT_ARGS+=("$1"); shift ;;
    esac
  done
  if (( want_help )); then
    echo "$LAUNCHER_HELP"; echo
    echo "Python script flags ($script):"
    run_in_container python "$REPO/$script" --help
    exit 0
  fi
  [ -n "$MODEL" ] || die "--model is required (see --help)"
  [[ "$TP" =~ ^[0-9]+$ ]] && (( GPUS_PER_NODE % TP == 0 )) || die "--tp $TP must divide GPUS_PER_NODE=$GPUS_PER_NODE"
  : "${SERVED_NAME:=$(basename "$MODEL")}"
  [[ "$SERVED_NAME" != */* ]] || die "--served-name must not contain '/'"
  export MODEL SERVED_NAME TP VLLM_ARGS STAGE_MODEL STARTUP_TIMEOUT

  # Validate the Python script's arguments (typos, missing input, changed settings on resume)
  # now, instead of after minutes of vLLM startup.
  run_in_container python "$REPO/$script" "${CLIENT_ARGS[@]}" --check-args || die "invalid arguments for $script (see above)"
}

# Write logs/<job>/run.txt: what was submitted and on what code, so runs can be diffed later.
# Usage: record_run <entry point name> "$@"   (the original, unparsed arguments)
record_run() {
  local entry=$1; shift
  mkdir -p "$RUN_DIR"
  {
    echo "date:        $(date -Is)"
    echo "command:     sbatch --nodes=$SLURM_JOB_NUM_NODES $entry $(printf '%q ' "$@")"
    echo "job:         $SLURM_JOB_ID ($SLURM_JOB_NAME) on partition ${SLURM_JOB_PARTITION:-?}"
    echo "nodes:       $SLURM_JOB_NUM_NODES ($SLURM_JOB_NODELIST)"
    echo "model:       $MODEL (served as '$SERVED_NAME'), TP=$TP, vllm args: ${VLLM_ARGS:-none}"
    echo "container:   $SIF"
    echo "git commit:  $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "git status:"
    git -C "$REPO" status --short 2>/dev/null | sed 's/^/  /'
    echo "git diff (uncommitted changes at submit time):"
    git -C "$REPO" diff HEAD 2>/dev/null
  } > "$RUN_DIR/run.txt"
  echo "run record: $RUN_DIR/run.txt"
}

# Start vLLM on every node of the allocation (one srun task per node, in the background),
# wait until all of them are healthy, then export VLLM_URLS (comma-separated) for the client.
# The servers are stopped when the batch script exits.
start_vllm_servers() {
  local nodes=$SLURM_JOB_NUM_NODES
  mkdir -p "$RUN_DIR/urls"
  echo "starting vLLM: $nodes node(s) x $(( GPUS_PER_NODE / TP )) server(s), TP=$TP, model $MODEL as '$SERVED_NAME'"
  echo "vLLM logs: $RUN_DIR"

  # --kill-on-bad-exit: one node failing (OOM, bad model) takes down the whole server step,
  # which the wait loop below notices.
  srun --label --kill-on-bad-exit=1 --mem=0 \
       --nodes="$nodes" --ntasks="$nodes" --ntasks-per-node=1 \
       --gpus-per-node="$GPUS_PER_NODE" --cpus-per-task="$SLURM_CPUS_PER_TASK" \
       singularity run -B "$BINDS,$REPO" "$SIF" bash "$REPO/lumi/_vllm_node.sh" &
  SERVERS_PID=$!
  trap stop_vllm_servers EXIT
  trap 'exit 143' TERM INT HUP

  SECONDS=0
  until (( $(find "$RUN_DIR/urls" -name '*.txt' | wc -l) == nodes )); do
    kill -0 "$SERVERS_PID" 2>/dev/null || die "vLLM server step exited during startup, see $RUN_DIR"
    (( SECONDS < STARTUP_TIMEOUT )) || die "vLLM not ready on all nodes after ${STARTUP_TIMEOUT}s, see $RUN_DIR"
    sleep 10
  done
  VLLM_URLS=$(cat "$RUN_DIR"/urls/*.txt | paste -sd,)
  export VLLM_URLS
  echo "all vLLM servers healthy after ${SECONDS}s: $VLLM_URLS"
}

stop_vllm_servers() {
  local rc=$?
  if [ -n "${SERVERS_PID:-}" ] && kill -0 "$SERVERS_PID" 2>/dev/null; then
    echo "stopping vLLM servers"
    kill "$SERVERS_PID" 2>/dev/null || true
    wait "$SERVERS_PID" 2>/dev/null || true
  fi
  exit "$rc"
}

# Run a command in the container on this (the first) node. The client needs no GPUs.
run_in_container() {
  singularity run -B "$BINDS,$REPO" "$SIF" "$@"
}

# ---------------------------------------------------------------------------
# Container side (called from lumi/_vllm_node.sh)
# ---------------------------------------------------------------------------
apply_env() {
  export PYTHONNOUSERSITE=1
  export HF_HUB_OFFLINE=1                    # compute nodes have no internet: pre-download models

  # Slurm sets ROCR_VISIBLE_DEVICES; vLLM/torch select GPUs via HIP_VISIBLE_DEVICES.
  export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
  export HWLOC_COMPONENTS=-amd_smi,-rsmi

  # node-local caches (/tmp is RAM, per node)
  export MIOPEN_USER_DB_PATH="/tmp/$USER-miopen-${SLURM_JOB_ID:-x}"
  export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
  export TORCH_EXTENSIONS_DIR="/tmp/$USER-torch-ext-${SLURM_JOB_ID:-x}"
  mkdir -p "$MIOPEN_USER_DB_PATH" "$TORCH_EXTENSIONS_DIR"

  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  export OMP_NUM_THREADS=7
}
