# Judges

A judge is a folder `configs/judges/<name>/` with a `config.py`; copy one to make a new judge. Use it as
`--judge <name>` (or pass a folder path). Start from `configs/judges/example/config.py`: a working judge where
every part is explained. Same format as a [classifier](classifiers.md#writing-a-classifier):

```python
# configs/judges/correctness/config.py
from pydantic import BaseModel
from _ask import last_user_message


class Output(BaseModel):     # stored per rollout as judge_scores
    reasoning: str           # first, so the judge reasons before it answers
    correct: bool


def messages(r):             # one rollout from generate.py -> the request
    return f"""You are grading a model's answer to a problem.

Problem:
{last_user_message(r)}

Reference answer:
{r["answer"]}

Model's response:
{r["completion"]}

Answer as a JSON object:
- "reasoning": a brief explanation
- "correct" (true/false): Is the model's final answer equivalent to the reference answer?"""
```

- `Output`: what the judge answers (types: see [classifiers.md](classifiers.md#writing-a-classifier), e.g.
  `clarity: int = Field(ge=1, le=5)`). vLLM enforces it as a JSON schema. The answers are stored per rollout as
  `judge_scores`, e.g. `{"reasoning": "...", "correct": true}`, the raw reply as `judge_output`. No reward is
  computed; turning answers into a training reward is left to the training side.
- `messages(r)`: the rollout has the input task's fields (`prompt`, `answer`, ...) plus `completion` (see
  [data.md](data.md)). `last_user_message(r)` is the prompt as text (the last user message if the prompt is
  chat messages). Return `None` to skip a rollout (it gets no output line). If it raises, that rollout gets a
  `judge_error`; on the first rollout the job stops before it starts.

Judges: `correctness` (final answer vs `answer`), `gsm8k` (also uses `reference_solution`; reasoning, answer
line, clarity 1-5), `tool_use` (tool-call rollouts).

## Judging while generating

With `--follow`, the judge job runs at the same time as the generation job and grades rollouts as they are
written, instead of waiting for the whole file.

```bash
jid=$(sbatch --parsable --nodes=4 --time=04:00:00 lumi/generate.sbatch --model Qwen/Qwen3-32B --tp 4 \
    --input data/prompts.jsonl --output data/rollouts.jsonl)
sbatch --dependency=after:$jid --nodes=2 --time=04:00:00 lumi/judge.sbatch --model Qwen/Qwen3-32B --tp 4 \
    --input data/rollouts.jsonl --output data/judged.jsonl --judge correctness --follow \
    --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}'
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
