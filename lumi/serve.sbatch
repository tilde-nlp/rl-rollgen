#!/bin/bash
#SBATCH --job-name=rollgen-serve
#SBATCH --account=project_465003062
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=0                       # important on dev-g, OOM otherwise
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#
# Serve a model with vLLM on every node of the allocation and keep it up until the time limit
# (or scancel), with no client. For clients outside the job, e.g. Harbor on Tilde (tilde/harbor_run.sh).
# A vllm-router on the first node puts all servers behind one URL (policy cache_aware: requests that
# share a prompt prefix, like the turns of one agent trial, stay on one server).
#
# Once per clone, on a login node:  bash lumi/install_router.sh
#
# Submit from the repo root:
#   sbatch --nodes=2 --time=04:00:00 lumi/serve.sbatch --model Qwen/Qwen3-32B --tp 4 \
#       --api-key-file ~/.vllm_key
#
# The URL to use: logs/<run>/urls/router.txt once the router works (also printed in slurm.out);
# the single servers are in urls/<node>.txt. Removed when the job ends. Check from a login node:
#   curl -H "Authorization: Bearer $(cat ~/.vllm_key)" $(cat logs/latest_run/urls/router.txt)/models
#
# Arguments. Slurm options go before the script name, all other flags after it.
#   sbatch options (common ones):
#     --nodes=N                 vLLM servers start on every node (8 / --tp servers per node)
#     --time=HH:MM:SS           time limit; a timed-out generate/judge job can be resubmitted (it resumes)
#     -p, --partition=NAME      dev-g (testing, ONE job per project), small-g, standard-g
#     --dependency=after:JOBID  start once JOBID has started (afterok:JOBID = once it finished OK)
#     --job-name=NAME           shows up in squeue and in the logs/ folder name
#   Server flags (read by the launcher in lumi/_common.sh, not passed to Python):
#     --model PATH|HUB_ID       (required) model dir, or HF hub id already downloaded to $HF_HOME,
#                               e.g. Qwen/Qwen3-32B (compute nodes are offline)
#     --served-name NAME        model name used in API requests (default: basename of --model)
#     --tp N                    tensor parallel size per vLLM server, must divide 8 (default: 1)
#     --vllm-args "ARGS"        extra `vllm serve` flags as ONE quoted string. The most useful ones:
#                                 --max-model-len N           context length (prompt + output); smaller
#                                                             leaves more memory for KV cache
#                                 --gpu-memory-utilization F  share of GPU memory vLLM takes (vLLM default 0.9)
#                                 --max-num-seqs N            max sequences one server runs at once
#                                 --enforce-eager             no torch.compile / graphs: faster startup,
#                                                             slower generation (good for quick tests)
#                                 --dtype bfloat16            weight / activation dtype
#                                 --seed N                    sampling seed
#                                 --enable-auto-tool-choice --tool-call-parser hermes
#                                                             native tool calls (agents that use them)
#                                 --reasoning-parser qwen3    split thinking into reasoning_content
#                                 full list: `vllm serve --help` in the container
#     --no-stage-model          load weights from Lustre instead of copying them to node-local /tmp first
#     --startup-timeout SEC     max wait for all servers to answer /health (default: 1800)
#     --compile-cache DIR       keep compile caches in DIR (on /scratch, never $HOME) so later jobs start
#                               faster (default: node-local /tmp, recompiled every job)
#     --api-key-file FILE       servers require the key in FILE (one line, chmod 600); without it anyone on
#                               the LUMI network can use them while the job runs
#     --router-policy NAME      how vllm-router spreads requests: cache_aware (default here: requests sharing a
#                               prompt prefix, like the turns of one agent trial, stay on one server),
#                               round_robin, random, power_of_two, consistent_hash
#   No other flags: this job runs no client.
#
# All flags:  bash lumi/serve.sbatch --help     (works on a login node)
# Logs:       logs/latest_run/ (slurm.out, run.txt, vLLM logs)
set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
[ -f "$REPO/lumi/_common.sh" ] || { echo "FATAL: submit from the repo root (sbatch lumi/serve.sbatch ...)" >&2; exit 1; }
source "$REPO/lumi/_common.sh"

init_run_dir
DEFAULT_ROUTER_POLICY=cache_aware   # agent trials: keep each trial's turns on one server (prefix cache)
parse_launcher_args "" "$@"
record_run lumi/serve.sbatch "$@"
start_vllm_servers
start_router
echo "serving at $VLLM_URL until the time limit or scancel $SLURM_JOB_ID (tilde/harbor_run.sh --lumi-run $SLURM_JOB_ID)"
wait -n "$SERVERS_PID" "$ROUTER_PID" || true
die "vLLM servers or vllm-router exited, see $RUN_DIR (router.log, *-vllm-*.log)"
