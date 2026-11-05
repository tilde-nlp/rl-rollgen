#!/bin/bash
#SBATCH --job-name=rollgen-generate
#SBATCH --account=project_465003062
#SBATCH --partition=dev-g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=0                       # important on dev-g, OOM otherwise
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#
# Generate rollouts with the policy model: starts vLLM on every node of the allocation, then runs
# rollgen/generate.py once (on the first node) against all servers.
#
# Submit from the repo root:
#   sbatch --nodes=4 lumi/generate.sbatch --model /scratch/.../my-model --tp 2 \
#       --input data/prompts.jsonl --output data/rollouts.jsonl --n 8 --max-tokens 2048
#
# All flags:  bash lumi/generate.sbatch --help     (works on a login node)
# Run record: logs/<job>/run.txt and <output>.meta.json
set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
[ -f "$REPO/lumi/_common.sh" ] || { echo "FATAL: submit from the repo root (sbatch lumi/generate.sbatch ...)" >&2; exit 1; }
source "$REPO/lumi/_common.sh"

parse_launcher_args rollgen/generate.py "$@"
record_run lumi/generate.sbatch "$@"
start_vllm_servers
run_in_container python "$REPO/rollgen/generate.py" "${CLIENT_ARGS[@]}"
