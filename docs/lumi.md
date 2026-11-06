# Running on LUMI


```bash
git clone <repo> && cd rl-rollgen      # nothing to install; always submit from the repo root

# 1. generate: 8 samples per prompt, vLLM on 4 nodes, TP=2
sbatch --nodes=4 lumi/generate.sbatch --model /scratch/.../policy-model --tp 2 \
    --input data/prompts.jsonl --output data/rollouts.jsonl --samples-per-prompt 8 --max-tokens 2048

# 2. judge
sbatch --nodes=2 lumi/judge.sbatch --model /scratch/.../judge-model --tp 4 \
    --input data/rollouts.jsonl --output data/judged.jsonl --judge judges/correctness

# 3. changed the reward formula in judges/correctness/rubric.toml? recompute, no GPUs needed
#    (needs Python >= 3.11; on a LUMI login node: module load cray-python)
python3 rollgen/rescore.py --input data/judged.jsonl --output data/judged_v2.jsonl --judge judges/correctness
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
```

Keep a model served for clients outside the job (e.g. Harbor on Tilde), until `--time` runs out or `scancel`:
```bash
(umask 077; openssl rand -hex 24 > ~/.vllm_key)       # once; the servers then require this key
sbatch --nodes=2 --time=04:00:00 lumi/serve.sbatch --model Qwen/Qwen3-32B --tp 4 --api-key-file ~/.vllm_key
cat logs/latest_run/urls/router.txt                    # one URL for all servers (vllm-router), gone when the job ends
```
Once per clone, before the first serve job: `bash lumi/install_router.sh` (on a login node).
`--api-key-file` works for generate/judge too.

`--model` takes a model directory or a Hugging Face id already downloaded to the shared HF cache
(`/scratch/project_465003062/environment/containers/fsdp-skyrl-harbor/hf`), e.g. `--model Qwen/Qwen3-32B`.

A timed-out job can be resubmitted with the same command; it skips finished items.

Per-experiment launch scripts: [`runs/`](../runs/) (`bash runs/<name>.sh`).
