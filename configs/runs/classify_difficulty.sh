sbatch --nodes=2 lumi/classify.sbatch \
    --model /scratch/.../gemma-4-12B-it \
    --input data/so_openq.short.jsonl \
    --output data/CLASSIFIED_so_openq.short.jsonl \
    --classifier difficulty \
    --keep-fields uuid \
    --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}'
