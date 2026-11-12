#!/bin/bash
# Difficulty labels (configs/classifiers/difficulty) for the SFT stage-1 candidates, judged by Gemma 4 31B-it
# (same model as classify_sft_stage0.sh). --keep-fields uuid is a new setting: outputs written without it
# (e.g. stage 0's two syn_mcq files) can't be resumed, so move them aside before running this.
# File list copied on 2026-11-06 from /scratch/project_465003062/data/tilde-think/sft-stage1-candidates.txt.
# ONE job for all files (vLLM starts once); the labels go next to each input:
#   <name>.jsonl -> <name>.difficulty.jsonl (+ <name>.difficulty.jsonl.meta.json)
#
# Usage (login node, from anywhere; it cds to the repo root):
#   bash configs/runs/classify_sft_stage1.sh --check   only check every file (record 0 through the
#                                                      classifier); no job
#   bash configs/runs/classify_sft_stage1.sh           check, then submit the job
# Timed out? Run it again: finished files and records are skipped.
set -euo pipefail
cd "$(dirname "$0")/../.."
source lumi/_common.sh   # SIF, BINDS, run_in_container (for the check)

DATA=/scratch/project_465003062/data/tilde-think
MODEL=google/gemma-4-31B-it
FILES=(
    _merged/math/all.reason.no_tools.jsonl
    _merged/math/all.reason.tools.jsonl
    _merged/science/rqa.reason.no_tools.jsonl
    _merged/science/rqa.reason.tools.jsonl
    _merged/science/so.reason.no_tools.jsonl
    _merged/science/so.reason.tools.jsonl
    _merged/science/syn_mcq.reason.no_tools.jsonl
    _merged/science/syn_mcq.reason.tools.jsonl
    _merged/science/vendor.reason.no_tools.jsonl
    _merged/coding-v2/competitive_programming_cpp.reason.no_tools.jsonl
    _merged/coding-v2/competitive_programming_python.reason.no_tools.jsonl
    _merged/coding-v2/text_to_sql.reason.no_tools.jsonl
    _merged/infinibyte/all.reason.no_tools.jsonl
    _merged/finance-doc/all.reason.no_tools_uuid.jsonl   # made by add_uuid.sh (the original has no uuid)
    _merged/arc/raw/small_reasoning_and_tools.jsonl
    _merged/arc/raw/small_reasoning_no_tools.jsonl
    _merged/agentic-v4/raw/tool_calling.jsonl
)
INPUTS=("${FILES[@]/#/$DATA/}")
ARGS=(--input "${INPUTS[@]}" --output-suffix .difficulty.jsonl
      --classifier difficulty --keep-fields uuid
      --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}')

# 1. Check every file before submitting (a missing field or another record shape shows up here,
#    not after waiting in the queue). The job checks again at start, before vLLM loads.
# MODEL_DIR/SERVED_NAME as the launcher sets them in the job, so resuming a started output passes the check.
MODEL_DIR=$(resolve_model_dir "$MODEL") SERVED_NAME=$(basename "$MODEL") \
    run_in_container python rollgen/classify.py --check-args "${ARGS[@]}"
[ "${1:-}" = "--check" ] && exit 0

# 2. Submit. 32 nodes x 4 servers (TP=2) = 128 Gemma servers behind the router.
# Expected (2026-11-11): a busy server does ~6,500-7,500 input chars/s, so 128 servers need ~4-4.5 h for the
# ~12.6 billion chars left (cpp rest, python, text_to_sql, infinibyte, finance-doc, arc, tool_calling),
# plus ~15 min vLLM startup per job. More nodes: scale --time down, but the single client process sends
# at most ~300-400 requests/s (these files need ~165/s on 32 nodes), so beyond ~64 nodes servers would idle.
# --concurrency 24 (per server): the servers are full with ~20 requests each (prefill-bound), and fewer
# open connections keep the client fast. Resume-safe, so started outputs continue.
sbatch -p standard-g --nodes=32 --time=06:00:00 lumi/classify.sbatch --model "$MODEL" --tp 2 \
    --concurrency 24 "${ARGS[@]}"
