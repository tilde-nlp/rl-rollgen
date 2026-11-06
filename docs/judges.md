# Judges

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
