#!/bin/bash
# GSM8K rollouts from TildeOpen, judged by Qwen3-32B while generation runs.
# Generate on dev-g, judge on small-g (dev_g allows only one job per project).
# Usage: bash runs/gsm8k_tildeopen.sh   (from anywhere; it cds to the repo root)
set -euo pipefail
cd "$(dirname "$0")/.."

jid=$(sbatch --parsable -p dev-g --nodes=2 --time=01:00:00 lumi/generate.sbatch \
    --model /scratch/project_465003062/environment/containers/grpo-pregen-rl/tildeopen --tp 2 \
    --input examples/gsm8k.jsonl --output examples/gsm8k_rollouts.jsonl \
    --samples-per-prompt 8 --max-tokens 2048)
echo "generate: $jid"

sbatch --dependency=after:"$jid" -p small-g --nodes=2 --time=01:00:00 lumi/judge.sbatch \
    --model Qwen/Qwen3-32B --tp 4 \
    --input examples/gsm8k_rollouts.jsonl --output examples/gsm8k_judged.jsonl \
    --judge judges/correctness --follow \
    --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}'
