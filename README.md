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

Or judge **while** generating (see [Judging while generating](#judging-while-generating)):
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
cat logs/latest_run/urls/*.txt                         # server URLs, once all are up (gone when the job ends)
```
`--api-key-file` works for generate/judge too.

`--model` takes a model directory or a Hugging Face id already downloaded to the shared HF cache
(`/scratch/project_465003062/environment/containers/fsdp-skyrl-harbor/hf`), e.g. `--model Qwen/Qwen3-32B`.

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
- `judged.jsonl`: rollout fields + `judge_output` (raw reply), `judge_scores` (answer per criterion), `reward`,
  `judge_error` (why a reply was unusable; then scores and reward are null)
- `<output>.meta.json`: settings, model, git commit, timing of every run that wrote the file
- `logs/<date>-<job name>-<job id>/`: `slurm.out`, `run.txt` (submitted command, code state), vLLM logs per node.
  `logs/latest_run` points to the newest one.

## Judges

A judge is a folder `judges/<name>/` with two files; copy one to make a new judge.
- `prompt.txt`: the judge prompt. `{field}` placeholders are filled from each rollout (`{prompt}`, `{completion}`,
  `{answer}`, `{reward_spec[ground_truth]}`, ...); `{criteria}` becomes the list of questions. Literal braces: `{{ }}`.
- `rubric.toml`: the questions and the reward formula:
  ```toml
  reward = "0 if not correct else max(0, 10 - 5*bad_reasoning - (5 - clarity))"   # before [criteria]

  [criteria]
  correct       = { type = "bool", question = "Is the final answer correct?" }
  bad_reasoning = { type = "bool", question = "Is the reasoning wrong even if the answer is right?" }
  clarity       = { type = "int", min = 1, max = 5, question = "How clear is the explanation?" }
  ```
  The judge answers every question in JSON (enforced by vLLM structured outputs); the reward is computed from those
  answers. `reward` is a Python expression over the criterion names (bools are 1/0; `min`, `max`, `abs`, `round`).

Changing only `reward` → `rollgen/rescore.py`. Changing questions → rerun the judge.

## Judging while generating

With `--follow`, the judge job runs at the same time as the generation job and grades rollouts as they are
written, instead of waiting for the whole file.

```bash
jid=$(sbatch --parsable --nodes=4 --time=04:00:00 lumi/generate.sbatch --model ... \
    --input data/prompts.jsonl --output data/rollouts.jsonl)
sbatch --dependency=after:$jid --nodes=2 --time=04:00:00 lumi/judge.sbatch --model ... \
    --input data/rollouts.jsonl --output data/judged.jsonl --judge judges/correctness --follow
```

- `--dependency=after:$jid` starts the judge once generation has *started* (`afterok` would wait for it to *finish*).
- Every 30 s the judge reads `rollouts.jsonl` again and grades the new lines.
- It stops on its own when the generation run is finished (it reads `data/rollouts.jsonl.meta.json`).
- If generation crashes or times out, the judge stops after `--follow-timeout` minutes (default 60) with no new
  lines and exits with an error. Resubmit both jobs with the same commands: each skips what it already did.
- Give the judge job the same `--time` as generation: it is busy until generation ends.
- Both jobs hold GPUs at the same time, and the judge's GPUs idle while it waits for rollouts. You get results
  sooner, not cheaper. If nothing is urgent, `afterok` uses fewer GPU hours.
- Needs a `.jsonl` input (generate.py always writes one).

## Layout

```
lumi/           LUMI launchers (sbatch)
tilde/          Tilde launchers
rollgen/        the Python that does the work, same on both clusters
judges/         judge prompts + rubrics, one folder per judge
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
