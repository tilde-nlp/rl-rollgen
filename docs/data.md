# Data: inputs, outputs, run records

Input (`--input` for generate): `.jsonl` or `.parquet` (SkyRL/verl datasets work as-is).
Each task needs `prompt` (string or list of chat messages) or `messages`; other fields are kept:
```json
{"id": "q1", "prompt": "What is 17 * 23?", "answer": "391"}
{"id": "q2", "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}], "answer": "..."}
{"prompt": [{"role": "user", "content": "..."}], "reward_spec": {"ground_truth": "..."}}
```

Outputs:
- `rollouts.jsonl`: one line per sample, input fields + `task_idx`, `sample_idx`, `completion`, `finish_reason`
- `judged.jsonl`: rollout fields + `judge_output` (raw reply), `judge_scores` (the `Output` fields of the judge),
  `judge_error` (why a reply was unusable; then `judge_scores` is null)
- `labels.jsonl` (classify): one line per input record (except skipped ones), `record_idx` + `--keep-fields` (e.g. `uuid`) + one field
  per `Output` field + `classify_error`; in completion order, join on `record_idx` or the kept id
- `<output>.meta.json`: settings, model, git commit, timing of every run that wrote the file
- `logs/<date>-<job name>-<job id>/`: `slurm.out`, `run.txt` (submitted command, code state), vLLM logs per node.
  `logs/latest_run` points to the newest one.
