# Running on LUMI


```bash
git clone <repo> && cd rl-rollgen      # always submit from the repo root
bash lumi/install_router.sh            # once per clone, on a login node: vllm-router (see below)

# 1. generate: 8 samples per prompt, vLLM on 4 nodes, TP=2
sbatch --nodes=4 lumi/generate.sbatch --model /scratch/.../policy-model --tp 2 \
    --input data/prompts.jsonl --output data/rollouts.jsonl --samples-per-prompt 8 --max-tokens 2048

# 2. judge
sbatch --nodes=2 lumi/judge.sbatch --model /scratch/.../judge-model --tp 4 \
    --input data/rollouts.jsonl --output data/judged.jsonl --judge correctness
```

Label a dataset (e.g. difficulty of each prompt; see [classifiers.md](classifiers.md)):
```bash
sbatch --nodes=2 lumi/classify.sbatch --model /scratch/.../gemma-4-12B-it \
    --input data/so_openq.jsonl --output data/so_openq.difficulty.jsonl \
    --classifier difficulty --keep-fields uuid \
    --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}'
```

Run the judge right after generation:
```bash
jid=$(sbatch --parsable --nodes=4 lumi/generate.sbatch ...)
sbatch --dependency=afterok:$jid lumi/judge.sbatch ...
```

Or judge **while** generating (see [Judging while generating](judges.md#judging-while-generating)):
```bash
jid=$(sbatch --parsable --nodes=4 lumi/generate.sbatch ... --output data/rollouts.jsonl)
sbatch --dependency=after:$jid --nodes=2 lumi/judge.sbatch ... --input data/rollouts.jsonl --follow
```

All flags:
```bash
bash lumi/generate.sbatch --help
bash lumi/judge.sbatch --help
bash lumi/classify.sbatch --help
```

Keep a model served for clients outside the job (e.g. Harbor on Tilde), until `--time` runs out or `scancel`:
```bash
(umask 077; openssl rand -hex 24 > ~/.vllm_key)       # once; the servers then require this key
sbatch --nodes=2 --time=04:00:00 lumi/serve.sbatch --model Qwen/Qwen3-32B --tp 4 --api-key-file ~/.vllm_key
cat logs/latest_run/urls/router.txt                    # one URL for all servers (vllm-router), gone when the job ends
```
`--api-key-file` works for generate/judge/classify too.

Every job puts its vLLM servers (on all nodes) behind one vllm-router on the first node; the Python script, or a
client outside the job, talks only to the router. `--router-policy` sets how it spreads the requests:
`round_robin` (default), `cache_aware` (default of serve.sbatch: requests sharing a prompt prefix, like the turns
of one agent trial, stay on one server and reuse its prefix cache). `--concurrency` stays per server: the script
keeps `concurrency x number of servers` requests in flight.

`--model` takes a model directory or a Hugging Face id already downloaded to the shared HF cache
(`/scratch/project_465003062/environment/containers/fsdp-skyrl-harbor/hf`), e.g. `--model Qwen/Qwen3-32B`.

A timed-out job can be resubmitted with the same command; it skips finished items.

Per-experiment launch scripts: [`configs/runs/`](../configs/runs/) (`bash configs/runs/<name>.sh`).
