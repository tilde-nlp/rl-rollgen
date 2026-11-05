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
# (or scancel), with no client. For clients outside the job, e.g. Harbor on Tilde (tilde/tunnel.sh).
#
# Submit from the repo root:
#   sbatch --nodes=2 --time=04:00:00 lumi/serve.sbatch --model Qwen/Qwen3-32B --tp 4 \
#       --api-key-file ~/.vllm_key
#
# Server URLs: logs/<run>/urls/*.txt once all servers are healthy (also printed in slurm.out);
# removed when the job ends. Check from a login node:
#   curl -H "Authorization: Bearer $(cat ~/.vllm_key)" $(head -1 logs/latest_run/urls/*.txt)/models
#
# All flags:  bash lumi/serve.sbatch --help     (works on a login node)
# Logs:       logs/latest_run/ (slurm.out, run.txt, vLLM logs)
set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
[ -f "$REPO/lumi/_common.sh" ] || { echo "FATAL: submit from the repo root (sbatch lumi/serve.sbatch ...)" >&2; exit 1; }
source "$REPO/lumi/_common.sh"

init_run_dir
parse_launcher_args "" "$@"
record_run lumi/serve.sbatch "$@"
start_vllm_servers
echo "serving until the time limit or scancel $SLURM_JOB_ID; run name for tilde/tunnel.sh: $(basename "$RUN_DIR")"
wait "$SERVERS_PID" || die "vLLM servers exited (rc $?), see $RUN_DIR"
die "vLLM servers exited, see $RUN_DIR"
