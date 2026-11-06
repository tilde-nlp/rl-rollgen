# Classifiers

A classifier labels each record of a dataset (e.g. domain and difficulty of every prompt) with an LLM.
It is a folder `classifiers/<name>/` in the same format as a [judge](judges.md), without a reward.
Copy one to make a new classifier.

```bash
sbatch --nodes=2 lumi/classify.sbatch --model /scratch/.../gemma-4-12B-it \
    --input data/so_openq.jsonl --output data/so_openq.difficulty.jsonl \
    --classifier classifiers/difficulty --keep-fields uuid \
    --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}'
```
Without Slurm, against vLLM servers that already run (e.g. on Tilde): one server is `--url http://host:8000/v1`;
for several, put a vllm-router in front of them and pass its URL and the number of servers:
```bash
uvx --from vllm-router==0.1.16 vllm-router --policy round_robin --port 9000 \
    --worker-urls http://daugava:8002 http://daugava:8003 http://daugava:8004 &
uv run --with openai rollgen/classify.py --url http://localhost:9000/v1 --servers 3 --served-name NAME ...
```

Output: a sidecar `.jsonl`, one line per input record:
```json
{"record_idx": 0, "uuid": "...", "domain": "physics", "task_type": "conceptual_qa", "difficulty": "medium", "tools_needed": false, "classify_error": null}
{"record_idx": 1, "uuid": "...", "domain": "math", "task_type": "proof", "difficulty": "hard", "tools_needed": null, "classify_error": null}
{"record_idx": 2, "uuid": "...", "domain": null, "task_type": null, "difficulty": null, "tools_needed": null, "classify_error": "record has no field {expected_answer} ..."}
```
- `record_idx`: input record number (from 0, blank lines not counted). Lines are in completion order, not input
  order: join on `record_idx` or a `--keep-fields` id.
- One field per criterion; `null` for a `when` criterion that wasn't asked (`tools_needed` without tools).
- `classify_error`: why a record has no labels (missing prompt field, unusable reply incl. the reply itself).
- `reasoning` too, if the rubric has `reasoning = true`.

The input is streamed (files of many GB are fine). Identical requests are sent once per run. A killed job is
resumed by resubmitting the same command; resuming with other settings or an edited classifier is refused.

## Writing a classifier

`classifiers/difficulty/` as the example:
- `system.txt`: the instructions and the full rubric, sent as-is.
- `prompt.json`: the user message as a JSON object, values filled from the record:
  ```json
  {"problem": "{problem}", "expected_answer": "{expected_answer}", "tools": "{responses_create_params[tools]?}"}
  ```
  `?` = optional: the key is left out if the record has no tools. (A `prompt.txt` text template works too.)
- `rubric.toml`: the labels.
  ```toml
  reasoning = false        # just the labels, no free-text reasoning first
  max_field_chars = 12000  # longer strings are cut, head and tail kept

  [criteria]
  domain       = { type = "enum", values = ["math", "physics", "..."], question = "The main subject knowledge needed." }
  difficulty   = { type = "enum", values = ["easy", "medium", "hard", "expert"], question = "..." }
  tools_needed = { type = "bool", when = "responses_create_params[tools]", question = "..." }
  ```
  Fields are generated in this order. The questions are listed in the prompt where `{criteria}` stands; a
  classifier whose `system.txt` describes the fields can leave `{criteria}` out.

All file options: [judges.md](judges.md#folder-format).
