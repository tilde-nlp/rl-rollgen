sbatch --nodes=2 --time=01:00:00 -p dev-g lumi/classify.sbatch \
    --model google/gemma-4-31B-it \
    --tp 2 \
    --input data/so.reason.no_tools.short.jsonl \
    --output results/so.reason.no_tools.short.classified.jsonl \
    --classifier difficulty \
    --keep-fields uuid \
    --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}'
