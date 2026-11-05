# rl-rollgen

Generate RL rollouts with self-hosted vLLM and score them with an LLM judge, on LUMI and Tilde.

## Quickstart (LUMI)

```bash
git clone <repo> && cd rl-rollgen      # nothing to install; always submit from the repo root

# 1. generate: 8 samples per prompt, vLLM on 4 nodes, TP=2
sbatch --nodes=4 lumi/generate.sbatch --model /scratch/.../policy-model --tp 2 \
    --input data/prompts.jsonl --output data/rollouts.jsonl --samples-per-prompt 8 --max-tokens 2048

# 2. judge
sbatch --nodes=2 lumi/judge.sbatch --model /scratch/.../judge-model --tp 4 \
    --input data/rollouts.jsonl --output data/judged.jsonl \
    --template judge_prompts/correctness.txt
```

Run the judge right after generation:
```bash
jid=$(sbatch --parsable --nodes=4 lumi/generate.sbatch ...)
sbatch --dependency=afterok:$jid lumi/judge.sbatch ...
```

All flags:
```bash
bash lumi/generate.sbatch --help
bash lumi/judge.sbatch --help
```

A timed-out job can be resubmitted with the same command; it skips finished items.

## Tilde

Not yet available.

## Data

Input (`--input` for generate): `.jsonl` or `.parquet` (SkyRL/verl datasets work as-is).
Each task needs `prompt` (string or list of chat messages) or `messages`; other fields are kept:
```json
{"id": "q1", "prompt": "What is 17 * 23?", "answer": "391"}
{"id": "q2", "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}], "answer": "..."}
{"prompt": [{"role": "user", "content": "..."}], "reward_spec": {"ground_truth": "..."}}
```

Outputs:
- `rollouts.jsonl`: one line per sample, input fields + `task_idx`, `sample_idx`, `completion`, `finish_reason`
- `judged.jsonl`: rollout fields + `judge_output`, `judge_score`
- `<output>.meta.json`: settings, model, git commit, timing of every run that wrote the file
- `logs/<date>-<job name>-<job id>/`: `slurm.out`, `run.txt` (submitted command, code state), vLLM logs per node.
  `logs/latest_run` points to the newest one.

Judge templates are plain text with `{field}` placeholders filled from each rollout (e.g. `{prompt}`, `{completion}`, `{answer}`, `{reward_spec[ground_truth]}`).
The score is the last match of `--score-regex` (default `SCORE: <number>`). See `judge_prompts/`.

## Layout

```
lumi/           LUMI launchers (sbatch)
tilde/          Tilde launchers
rollgen/        the Python that does the work, same on both clusters
judge_prompts/  judge prompt templates
examples/       small example inputs
```
Files starting with `_` are internals.

## Goal

A collection of scripts/python code that:
* launches vLLM inference
* generates rollouts for one-step RL tasks
* generates rollouts for multi-step RL tasks (harbor)
* tokenizes the final rollouts
* operates on two clusters (LUMI, which is hpc with AMD, and Tilde servers with NVIDIA gpus and without slurm)
* has llm-as-a-judge support for RL tasks
* other useful scripts for filtering, etc.

## Priorities

1. No overengineering: it's better to just have a neat collection of documented scripts, than to have a very complex structure no one will understand/need.
2. Using already built solutions instead of rewriting them by hand
3. Documentation - I will write it myself mostly, but scripts need to have clear names, and the structure of the project should also be clear
