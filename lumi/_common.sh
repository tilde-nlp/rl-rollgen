# Shared LUMI code, sourced by lumi/*.sbatch (on the host) and lumi/_vllm_node.sh (inside the container).
# Users don't need to read this; the entry points are lumi/*.sbatch.
# Arguments: none (sourced). The site config below can be overridden by exporting the variables before sbatch.
# The flags parsed here (parse_launcher_args) are documented at the top of lumi/generate.sbatch.

# --- site config (rarely changes; override by exporting before sbatch if you must) ---
set -a
: "${SIF:=/appl/local/laifs/containers/lumi-multitorch-u24r72f21m50t211-20261028_090706/lumi-multitorch-plus-u24r72f21m50t211-20261028_090706.sif}"
: "${BINDS:=/var/spool/slurmd,/scratch/project_465003062,/scratch/project_465002038}"
: "${HF_HOME:=/scratch/project_465003062/environment/containers/fsdp-skyrl-harbor/hf}"
: "${GPUS_PER_NODE:=8}"
: "${BASE_PORT:=8000}"
# --- per-job paths (on shared storage) ---
: "${REPO:=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
: "${LOGS_DIR:=$REPO/logs}"
: "${ROUTER_DIR:=$REPO/lumi/vllm-router}"   # vllm-router (every job), from lumi/install_router.sh
: "${ROUTER_PORT:=9000}"
# One folder per job: slurm.out, run.txt, vLLM logs, server URLs. Set once in the batch script,
# inherited by every node. Date first, so `ls logs/` is chronological.
: "${RUN_DIR:=$LOGS_DIR/$(date +%Y%m%d-%H%M%S)-${SLURM_JOB_NAME:-local}-${SLURM_JOB_ID:-$$}}"
set +a

die() { echo "FATAL: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Host side (called from lumi/*.sbatch)
# ---------------------------------------------------------------------------

LAUNCHER_HELP='Launcher flags (consumed here, everything else goes to the Python script):
  --model PATH|HUB_ID     model dir, or HF hub id already downloaded to HF_HOME (e.g. Qwen/Qwen3-32B)
  --served-name NAME      name used in API requests (default: basename of --model)
  --tp N                  tensor parallel size per vLLM server (default 1); must divide 8
  --vllm-args "ARGS"      extra args for `vllm serve`, e.g. "--max-model-len 16384 --gpu-memory-utilization 0.9"
  --no-stage-model        load the model from Lustre instead of copying it to node-local /tmp first
  --startup-timeout SEC   max wait for all vLLM servers to become healthy (default 1800)
  --compile-cache DIR     keep vLLM/Triton/MIOpen compile caches in DIR (e.g. on /scratch) so later jobs
                          start faster; default: node-local /tmp, recompiled every job
  --api-key-file FILE     servers require this key (file holds one line; chmod 600). Without it, anyone
                          on the LUMI network can use the servers while the job runs
  --router-policy NAME    how vllm-router spreads requests over the servers: round_robin (default), or
                          cache_aware (default of serve.sbatch: requests sharing a prompt prefix, like the
                          turns of one agent trial, stay on one server), random, power_of_two, consistent_hash
Node count, time limit, partition: regular sbatch options, e.g. `sbatch --nodes=4 --time=04:00:00 ...`.'

# Local directory for --model: the dir itself, or the snapshot of a hub id in $HF_HOME/hub
# (refs/main if present, else the newest snapshot). Empty if not found.
resolve_model_dir() {
  local m=$1
  if [ -d "$m" ]; then (cd "$m" && pwd); return; fi
  local cache="$HF_HOME/hub/models--${m//\//--}"
  [ -d "$cache/snapshots" ] || return 0
  if [ -f "$cache/refs/main" ] && [ -d "$cache/snapshots/$(cat "$cache/refs/main")" ]; then
    echo "$cache/snapshots/$(cat "$cache/refs/main")"
  else
    ls -dt "$cache"/snapshots/*/ 2>/dev/null | head -1 | sed 's:/$::'
  fi
}

# Usage: parse_launcher_args <python script, or "" for none (serve.sbatch)> "$@"
# Sets and exports the server config; leaves the remaining args in CLIENT_ARGS.
# `--help` prints the launcher flags plus the Python script's own --help, then exits.
parse_launcher_args() {
  local script=$1; shift
  MODEL="" SERVED_NAME="" TP=1 VLLM_ARGS="" STAGE_MODEL=1 STARTUP_TIMEOUT=1800 COMPILE_CACHE="" API_KEY_FILE=""
  ROUTER_POLICY=${DEFAULT_ROUTER_POLICY:-round_robin}   # serve.sbatch sets DEFAULT_ROUTER_POLICY
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
      --compile-cache)      COMPILE_CACHE=$2; shift 2 ;;
      --compile-cache=*)    COMPILE_CACHE=${1#*=}; shift ;;
      --api-key-file)       API_KEY_FILE=$2; shift 2 ;;
      --api-key-file=*)     API_KEY_FILE=${1#*=}; shift ;;
      --router-policy)      ROUTER_POLICY=$2; shift 2 ;;
      --router-policy=*)    ROUTER_POLICY=${1#*=}; shift ;;
      -h|--help)            want_help=1; shift ;;
      *)                    CLIENT_ARGS+=("$1"); shift ;;
    esac
  done
  if (( want_help )); then
    echo "$LAUNCHER_HELP"
    if [ -n "$script" ]; then
      echo; echo "Python script flags ($script):"
      run_in_container python "$REPO/$script" --help
    fi
    exit 0
  fi
  [ -n "$script" ] || (( ${#CLIENT_ARGS[@]} == 0 )) || die "unknown flag(s): ${CLIENT_ARGS[*]} (see --help)"
  [ -n "$MODEL" ] || die "--model is required (see --help)"
  [ -d "$ROUTER_DIR/vllm_router" ] || die "vllm-router not installed: run 'bash lumi/install_router.sh' once on a login node"
  case "$ROUTER_POLICY" in
    round_robin|cache_aware|random|power_of_two|consistent_hash) ;;
    *) die "--router-policy $ROUTER_POLICY: expected round_robin, cache_aware, random, power_of_two or consistent_hash" ;;
  esac
  [[ "$TP" =~ ^[0-9]+$ ]] && (( GPUS_PER_NODE % TP == 0 )) || die "--tp $TP must divide GPUS_PER_NODE=$GPUS_PER_NODE"
  MODEL_DIR=$(resolve_model_dir "$MODEL")
  [ -n "$MODEL_DIR" ] && [ -f "$MODEL_DIR/config.json" ] \
    || die "--model $MODEL: not a model dir and not found in $HF_HOME/hub (compute nodes are offline: download it first, e.g. HF_HOME=$HF_HOME hf download $MODEL)"
  : "${SERVED_NAME:=$(basename "$MODEL")}"
  [[ "$SERVED_NAME" != */* ]] || die "--served-name must not contain '/'"
  if [ -n "$COMPILE_CACHE" ]; then
    mkdir -p "$COMPILE_CACHE" || die "cannot create --compile-cache $COMPILE_CACHE"
    COMPILE_CACHE=$(cd "$COMPILE_CACHE" && pwd)   # absolute: the path must mean the same on every node
    [[ "$COMPILE_CACHE" != "$HOME"* ]] || die "--compile-cache must not be in \$HOME (100K file quota); use /scratch"
  fi
  if [ -n "$API_KEY_FILE" ]; then
    [ -s "$API_KEY_FILE" ] || die "--api-key-file $API_KEY_FILE: missing or empty"
    API_KEY_FILE=$(cd "$(dirname "$API_KEY_FILE")" && pwd)/$(basename "$API_KEY_FILE")
    [[ "$(stat -c %a "$API_KEY_FILE")" == *00 ]] || die "--api-key-file is readable by others: chmod 600 $API_KEY_FILE"
  fi
  export MODEL MODEL_DIR SERVED_NAME TP VLLM_ARGS STAGE_MODEL STARTUP_TIMEOUT COMPILE_CACHE API_KEY_FILE ROUTER_POLICY

  # Validate the Python script's arguments (typos, missing input, changed settings on resume)
  # now, instead of after minutes of vLLM startup.
  [ -z "$script" ] || run_in_container python "$REPO/$script" "${CLIENT_ARGS[@]}" --check-args \
    || die "invalid arguments for $script (see above)"
}

# Create this job's log folder, move Slurm's .out file into it, and point logs/latest_run at it.
# Slurm can't write into a folder that doesn't exist at submit time, so the .out file starts in logs/
# and is moved here; Slurm keeps writing to it (same file, new name).
init_run_dir() {
  [ -n "${SLURM_JOB_ID:-}" ] || return 0   # not in a job (e.g. `bash lumi/generate.sbatch --help`)
  mkdir -p "$RUN_DIR"
  local out
  out=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | grep -oP 'StdOut=\K\S+' || true)
  if [ -n "$out" ] && [ -f "$out" ]; then mv "$out" "$RUN_DIR/slurm.out"; fi
  ln -sfn "$(basename "$RUN_DIR")" "$LOGS_DIR/latest_run"
  echo "logs: $RUN_DIR (also logs/latest_run)"
}

# Write <run dir>/run.txt: what was submitted and on what code, so runs can be diffed later.
# Usage: record_run <entry point name> "$@"   (the original, unparsed arguments)
record_run() {
  local entry=$1; shift
  {
    echo "date:        $(date -Is)"
    echo "command:     sbatch --nodes=$SLURM_JOB_NUM_NODES $entry $(printf '%q ' "$@")"
    echo "job:         $SLURM_JOB_ID ($SLURM_JOB_NAME) on partition ${SLURM_JOB_PARTITION:-?}"
    echo "nodes:       $SLURM_JOB_NUM_NODES ($SLURM_JOB_NODELIST)"
    echo "model:       $MODEL (served as '$SERVED_NAME'), TP=$TP, vllm args: ${VLLM_ARGS:-none}"
    echo "model dir:   $MODEL_DIR"
    echo "compile cache: ${COMPILE_CACHE:-node-local /tmp (per job)}"
    echo "api key:     ${API_KEY_FILE:-none (servers open to the LUMI network)}"
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
# wait until all of them are healthy, then export SERVER_URLS (comma-separated) for the router.
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
       singularity run -B "$BINDS,$REPO${COMPILE_CACHE:+,$COMPILE_CACHE}${API_KEY_FILE:+,$API_KEY_FILE}" \
         "$SIF" bash "$REPO/lumi/_vllm_node.sh" &
  SERVERS_PID=$!
  trap stop_vllm_servers EXIT
  trap 'exit 143' TERM INT HUP

  SECONDS=0
  until (( $(find "$RUN_DIR/urls" -name '*.txt' | wc -l) == nodes )); do
    kill -0 "$SERVERS_PID" 2>/dev/null || die "vLLM server step exited during startup, see $RUN_DIR"
    (( SECONDS < STARTUP_TIMEOUT )) || die "vLLM not ready on all nodes after ${STARTUP_TIMEOUT}s, see $RUN_DIR"
    sleep 10
  done
  SERVER_URLS=$(cat "$RUN_DIR"/urls/*.txt | paste -sd,)
  export SERVER_URLS
  echo "all vLLM servers healthy after ${SECONDS}s: $SERVER_URLS"
}

# Start vllm-router (lumi/_router.sh) on this (the first) node in front of all servers in SERVER_URLS,
# wait until it works, then export VLLM_URL (its URL) and VLLM_SERVERS (how many servers are behind
# it) for the client. Call after start_vllm_servers; stopped with the servers.
start_router() {
  run_in_container bash "$REPO/lumi/_router.sh" &
  ROUTER_PID=$!
  until [ -f "$RUN_DIR/urls/router.txt" ]; do
    kill -0 "$ROUTER_PID" 2>/dev/null || die "vllm-router failed to start, see above and $RUN_DIR/router.log"
    sleep 5
  done
  VLLM_URL=$(< "$RUN_DIR/urls/router.txt")
  VLLM_SERVERS=$(tr ',' '\n' <<< "$SERVER_URLS" | grep -c .)
  export VLLM_URL VLLM_SERVERS
}

stop_vllm_servers() {
  local rc=$?
  rm -rf "$RUN_DIR/urls"   # only valid while the servers run; a leftover would look like a live job
  if [ -n "${ROUTER_PID:-}" ]; then kill "$ROUTER_PID" 2>/dev/null || true; fi
  if [ -n "${SERVERS_PID:-}" ] && kill -0 "$SERVERS_PID" 2>/dev/null; then
    echo "stopping vLLM servers"
    kill "$SERVERS_PID" 2>/dev/null || true
    wait "$SERVERS_PID" 2>/dev/null || true
  fi
  exit "$rc"
}

# Run a command in the container on this (the first) node. The client needs no GPUs.
run_in_container() {
  singularity run -B "$BINDS,$REPO${API_KEY_FILE:+,$API_KEY_FILE}" "$SIF" "$@"
}

# ---------------------------------------------------------------------------
# Container side (called from lumi/_vllm_node.sh and lumi/_router.sh)
# ---------------------------------------------------------------------------

# Address other nodes use to reach this one: the Slingshot interface (hsn0) if present, else the hostname.
# (The container has no `ip` command; ask the kernel directly: SIOCGIFADDR = 0x8915.)
node_addr() {
  python -c 'import fcntl, socket, struct
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
print(socket.inet_ntoa(fcntl.ioctl(s.fileno(), 0x8915, struct.pack("256s", b"hsn0"))[20:24]))' 2>/dev/null \
    || echo "${SLURMD_NODENAME:-$(hostname)}"
}

apply_env() {
  export PYTHONNOUSERSITE=1
  export HF_HUB_OFFLINE=1                    # compute nodes have no internet: pre-download models

  # Slurm sets ROCR_VISIBLE_DEVICES; vLLM/torch select GPUs via HIP_VISIBLE_DEVICES.
  export HIP_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  unset ROCR_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
  export HWLOC_COMPONENTS=-amd_smi,-rsmi

  # All caches and the staged model go to one node-local dir (/tmp is RAM), deleted when the job ends.
  # Never let anything default to ~/.cache: the LUMI home quota is 100K files, and vLLM's
  # torch.compile/Triton caches alone can exceed it ("Disk quota exceeded" during startup).
  export NODE_TMP="/tmp/$USER-rollgen-${SLURM_JOB_ID:-$$}"
  export XDG_CACHE_HOME="$NODE_TMP/cache"            # catch-all for anything using ~/.cache
  # Compile caches: per job in /tmp by default, or shared across jobs with --compile-cache.
  local cc="${COMPILE_CACHE:-$NODE_TMP/cache}"
  export VLLM_CACHE_ROOT="$cc/vllm"                   # torch.compile cache
  export TORCHINDUCTOR_CACHE_DIR="$cc/inductor"
  export TRITON_CACHE_DIR="$cc/triton"
  export MIOPEN_USER_DB_PATH="$cc/miopen"
  export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
  export TORCH_EXTENSIONS_DIR="$NODE_TMP/cache/torch-ext"
  mkdir -p "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$MIOPEN_USER_DB_PATH" "$TORCH_EXTENSIONS_DIR"

  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  export OMP_NUM_THREADS=7
}
