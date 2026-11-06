# Examples (personal, gitignored)

Commands actually used to launch runs on LUMI. Submit from the repo root.

## Generate: example prompts, 2 nodes, TP=4

```bash
sbatch --nodes=2 lumi/generate.sbatch \
    --model /scratch/project_465003062/environment/containers/grpo-pregen-rl/tildeopen --tp 4 \
    --input examples/prompts.jsonl --output examples/rollouts.jsonl \
    --samples-per-prompt 8 --max-tokens 2048
```

## Generate + judge while generating: GSM8K, dev-g

```bash
jid=$(sbatch --parsable -p dev-g --nodes=2 --time=01:00:00 lumi/generate.sbatch \
    --model /scratch/project_465003062/environment/containers/grpo-pregen-rl/tildeopen --tp 2 \
    --input examples/gsm8k.jsonl --output examples/gsm8k_rollouts.jsonl \
    --samples-per-prompt 8 --max-tokens 2048)

sbatch --dependency=after:$jid --nodes=2 -p small-g --time=01:00:00 lumi/judge.sbatch \
    --model Qwen/Qwen3-32B --tp 4 \
    --input examples/gsm8k_rollouts.jsonl --output examples/gsm8k_judged.jsonl \
    --judge judges/correctness --follow
```

## Generate, then judge after it finishes (old judge interface)

`--template` no longer exists; today this is `--judge judges/correctness`.

```bash
jid=$(sbatch --parsable --nodes=2 lumi/generate.sbatch ...)
sbatch --dependency=afterok:$jid --nodes=1 lumi/judge.sbatch \
    --model Qwen/Qwen3-32B --tp 4 \
    --input examples/rollouts.jsonl --output examples/judged.jsonl \
    --template judge_prompts/correctness.txt \
    --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}'
```

## Network test: can the login node reach a compute node's port?

```bash
sbatch --account=project_465003062 --partition=dev-g --nodes=1 --gpus-per-node=1 \
       --time=00:30:00 --job-name=nettest --output=logs/nettest-%j.out --wrap '
echo "node: $(hostname)"
ip -4 -o addr show
echo "in container:"; singularity run /appl/local/laifs/containers/lumi-multitorch-u24r72f21m50t211-20261028_090706/lumi-multitorch-plus-u24r72f21m50t211-20261028_090706.sif \
    ip -4 -o addr show hsn0 || echo "NO ip IN CONTAINER"
python3 -m http.server 8000 --bind 0.0.0.0'

# then, from uan01:
curl -sS -m 5 -o /dev/null -w '%{http_code}\n' http://<hsn0-ip>:8000/
```
