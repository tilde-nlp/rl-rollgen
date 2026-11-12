#!/bin/bash
# Make the difficulty labels from classify_sft_stage1.sh readable by everyone:
#   files 644 (rw-r--r--), the directories leading to them 755 (rwxr-xr-x).
# chmod -c prints each change with the old mode, so you can undo it from the output.
# Only files and directories you own can be changed; the others are reported and skipped.
# Note: 755 removes group write from a directory that had it (colleagues then can't create files there).
#
# Usage (LUMI login node): bash configs/runs/share_sft_stage1.sh
set -uo pipefail

DATA=/scratch/project_465003062/data/tilde-think

DIRS=(
    /scratch/project_465003062/data
    $DATA
    $DATA/_merged
    $DATA/_merged/math
    $DATA/_merged/science
    $DATA/_merged/coding-v2
    $DATA/_merged/infinibyte
    $DATA/_merged/finance-doc
    $DATA/_merged/arc
    $DATA/_merged/arc/raw
    $DATA/_merged/agentic-v4
    $DATA/_merged/agentic-v4/raw
)

FILES=(
    $DATA/_merged/math/all.reason.no_tools.difficulty.jsonl
    $DATA/_merged/math/all.reason.no_tools.difficulty.jsonl.meta.json
    $DATA/_merged/math/all.reason.tools.difficulty.jsonl
    $DATA/_merged/math/all.reason.tools.difficulty.jsonl.meta.json
    $DATA/_merged/science/rqa.reason.no_tools.difficulty.jsonl
    $DATA/_merged/science/rqa.reason.no_tools.difficulty.jsonl.meta.json
    $DATA/_merged/science/rqa.reason.tools.difficulty.jsonl
    $DATA/_merged/science/rqa.reason.tools.difficulty.jsonl.meta.json
    $DATA/_merged/science/so.reason.no_tools.difficulty.jsonl
    $DATA/_merged/science/so.reason.no_tools.difficulty.jsonl.meta.json
    $DATA/_merged/science/so.reason.tools.difficulty.jsonl
    $DATA/_merged/science/so.reason.tools.difficulty.jsonl.meta.json
    $DATA/_merged/science/syn_mcq.reason.no_tools.difficulty.jsonl
    $DATA/_merged/science/syn_mcq.reason.no_tools.difficulty.jsonl.meta.json
    $DATA/_merged/science/syn_mcq.reason.tools.difficulty.jsonl
    $DATA/_merged/science/syn_mcq.reason.tools.difficulty.jsonl.meta.json
    $DATA/_merged/science/vendor.reason.no_tools.difficulty.jsonl
    $DATA/_merged/science/vendor.reason.no_tools.difficulty.jsonl.meta.json
    $DATA/_merged/coding-v2/competitive_programming_cpp.reason.no_tools.difficulty.jsonl
    $DATA/_merged/coding-v2/competitive_programming_cpp.reason.no_tools.difficulty.jsonl.meta.json
    $DATA/_merged/coding-v2/competitive_programming_python.reason.no_tools.difficulty.jsonl
    $DATA/_merged/coding-v2/competitive_programming_python.reason.no_tools.difficulty.jsonl.meta.json
    $DATA/_merged/coding-v2/text_to_sql.reason.no_tools.difficulty.jsonl
    $DATA/_merged/coding-v2/text_to_sql.reason.no_tools.difficulty.jsonl.meta.json
    $DATA/_merged/infinibyte/all.reason.no_tools.difficulty.jsonl
    $DATA/_merged/infinibyte/all.reason.no_tools.difficulty.jsonl.meta.json
    $DATA/_merged/finance-doc/all.reason.no_tools_uuid.difficulty.jsonl
    $DATA/_merged/finance-doc/all.reason.no_tools_uuid.difficulty.jsonl.meta.json
    $DATA/_merged/arc/raw/small_reasoning_and_tools.difficulty.jsonl
    $DATA/_merged/arc/raw/small_reasoning_and_tools.difficulty.jsonl.meta.json
    $DATA/_merged/arc/raw/small_reasoning_no_tools.difficulty.jsonl
    $DATA/_merged/arc/raw/small_reasoning_no_tools.difficulty.jsonl.meta.json
    $DATA/_merged/agentic-v4/raw/tool_calling.difficulty.jsonl
    $DATA/_merged/agentic-v4/raw/tool_calling.difficulty.jsonl.meta.json
)

failed=0
for d in "${DIRS[@]}"; do chmod -c 755 "$d" || failed=1; done
for f in "${FILES[@]}"; do chmod -c 666 "$f" || failed=1; done
[ "$failed" = 0 ] || { echo "Some paths were not changed (see the errors above)." >&2; exit 1; }
