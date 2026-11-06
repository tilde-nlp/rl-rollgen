# Judges

A judge is a folder `judges/<name>/` with two or three files; copy one to make a new judge.
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
- `system.txt` (optional): a system message.

Changing only `reward` → `rollgen/rescore.py`. Changing questions → rerun the judge.

## Folder format

Judges and [classifiers](classifiers.md) use the same folder format (`rollgen/_rubric.py`); only a judge has `reward`.

| file | |
|---|---|
| `prompt.txt` | user message template, see placeholders below |
| `prompt.json` | instead of `prompt.txt`: the user message as a JSON object, e.g. `{"problem": "{problem}", "tools": "{tools?}"}`. A value that is exactly one placeholder becomes the record's value (any JSON type); other values stay as written. Sent as compact JSON |
| `system.txt` | optional system message, sent as-is (no placeholders, braces need no escaping) |
| `rubric.toml` | settings (above `[criteria]`, a TOML rule) and the questions |

Placeholders: `{name}`, nested `{reward_spec[ground_truth]}`, list items `{messages[0][content]}`.
Strings are inserted as they are, other values as JSON. A field missing from a record makes that record an error
line (`judge_error` / `classify_error`); a missing field in the *first* record stops the job before it starts
(typo check). `{name?}` is optional: nothing (or no key, in `prompt.json`) if the field is missing or empty.
`{prompt}`: for chat-message prompts, the last user message. `{criteria}`: the questions, required unless
`system.txt` exists.

`rubric.toml` settings:
- `reward = "..."`: judges only (required there).
- `reasoning = true` (default): the model writes a free-text `reasoning` field before its answers. `false`: answers only.
- `max_field_chars = N`: strings inserted into the prompt longer than N characters are cut (2/3 head, 1/3 tail kept,
  `[... truncated ...]` between).

Criteria (`[criteria]`, answered in this order):
- `{ type = "bool", question = "..." }`
- `{ type = "int", min = 1, max = 5, question = "..." }`
- `{ type = "enum", values = ["a", "b"], question = "..." }`
- any of them with `when = "field"` (e.g. `"responses_create_params[tools]"`): asked only if the record has that
  field (non-empty), otherwise `null`. In a `reward` formula it is then `None`.

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
