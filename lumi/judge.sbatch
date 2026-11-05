#!/bin/bash
#SBATCH --job-name=rollgen-judge
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
# Score rollouts with a judge model: starts the judge in vLLM on every node of the allocation,
# then runs rollgen/judge.py once (on the first node) against all servers.
#
# Submit from the repo root:
#   sbatch --nodes=2 lumi/judge.sbatch --model /scratch/.../judge-model --tp 4 \
#       --input data/rollouts.jsonl --output data/judged.jsonl --template rollgen/judge_prompts/correctness.txt
#
# Chain after a generation job:
#   jid=$(sbatch --parsable lumi/generate.sbatch ...)
#   sbatch --dependency=afterok:$jid lumi/judge.sbatch ...
#
# All flags:  bash lumi/judge.sbatch --help     (works on a login node)
# Run record: logs/<job>/run.txt and <output>.meta.json
set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
[ -f "$REPO/lumi/_common.sh" ] || { echo "FATAL: submit from the repo root (sbatch lumi/judge.sbatch ...)" >&2; exit 1; }
source "$REPO/lumi/_common.sh"

parse_launcher_args rollgen/judge.py "$@"
record_run lumi/judge.sbatch "$@"
start_vllm_servers
run_in_container python "$REPO/rollgen/judge.py" "${CLIENT_ARGS[@]}"
