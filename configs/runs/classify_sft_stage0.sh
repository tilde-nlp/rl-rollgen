#!/bin/bash
# Difficulty labels (configs/classifiers/difficulty) for two SFT candidate files, judged by Gemma 4 31B-it.
# Small first run on dev-g (1 hour) before classify_sft_stage1.sh. One job for both files; the labels go
# next to each input: <name>.jsonl -> <name>.difficulty.jsonl (+ <name>.difficulty.jsonl.meta.json).
#
# Usage (login node, from anywhere; it cds to the repo root):
#   bash configs/runs/classify_sft_stage0.sh --check   only check both files (record 0 through the
#                                                      classifier); no job
#   bash configs/runs/classify_sft_stage0.sh           check, then submit the job
# Timed out? Run it again: finished files and records are skipped.
# dev-g allows ONE job per project (pending included): nothing else of ours may be queued there.
set -euo pipefail
cd "$(dirname "$0")/../.."
source lumi/_common.sh   # SIF, BINDS, run_in_container (for the check)

DATA=/scratch/project_465003062/data/tilde-think
# vLLM/Inductor/Triton/MIOpen compile caches kept across jobs (the first job fills it, later ones start faster)
COMPILE_CACHE=/scratch/project_465003062/environment/rollgen-compile-cache
FILES=(
    _merged/science/syn_mcq.reason.no_tools.jsonl
    _merged/science/syn_mcq.reason.tools.jsonl
)
INPUTS=("${FILES[@]/#/$DATA/}")
ARGS=(--input "${INPUTS[@]}" --output-suffix .difficulty.jsonl
      --classifier difficulty
      --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}')

# 1. Check both files before submitting (the job checks again at start, before vLLM loads).
run_in_container python rollgen/classify.py --check-args "${ARGS[@]}"
[ "${1:-}" = "--check" ] && exit 0

# 2. Submit. 2 nodes x 4 servers (TP=2) = 8 Gemma servers behind the router.
sbatch -p dev-g --nodes=2 --time=01:00:00 lumi/classify.sbatch --model google/gemma-4-31B-it --tp 2 \
    --compile-cache "$COMPILE_CACHE" "${ARGS[@]}"
