# Classifiers

A classifier labels each record of a dataset (e.g. domain and difficulty of every prompt) with an LLM.
It is a folder `configs/classifiers/<name>/` with a `config.py` (same format as a [judge](judges.md)).
Copy one to make a new classifier; use it as `--classifier <name>` (or pass a folder path).

```bash
sbatch --nodes=2 lumi/classify.sbatch --model Qwen/Qwen3-32B --tp 4 \
    --input data/so_openq.jsonl --output data/so_openq.difficulty.jsonl \
    --classifier difficulty --keep-fields uuid \
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
{"record_idx": 0, "uuid": "...", "domain": "physics", "task_type": "conceptual_qa", "difficulty": "medium", "classify_error": null}
{"record_idx": 1, "uuid": "...", "domain": "math", "task_type": "proof", "difficulty": "hard", "classify_error": null}
{"record_idx": 2, "uuid": "...", "domain": null, "task_type": null, "difficulty": null, "classify_error": "configs/classifiers/difficulty/config.py line 57: messages() failed: KeyError: 'problem'"}
```
- `record_idx`: input record number (from 0, blank lines not counted). Lines are in completion order, not input
  order: join on `record_idx` or a `--keep-fields` id.
- One field per field of `Output` (see below).
- `classify_error`: why a record has no labels (`messages()` failed, or the reply doesn't fit `Output`; the
  reply itself is included).
- Records that `messages()` skips (returns `None`) get no line.

The input is streamed (files of many GB are fine). Identical requests are sent once per run.

Several files in one job (the model starts once): `--input a.jsonl b.jsonl ... --output-suffix .difficulty.jsonl`
writes each file's labels next to it (`a.jsonl` -> `a.difficulty.jsonl`, each with its own `.meta.json`). All
files are checked before anything is sent (record 0, `--keep-fields`, resume settings), then classified one
after another. A timed-out job is resumed by resubmitting the same command (finished files are skipped
quickly); one file resumes the same way with `--input a.jsonl --output a.difficulty.jsonl`.
Example: `configs/runs/classify_sft_stage1.sh`. A killed job is
resumed by resubmitting the same command; resuming with other settings or an edited classifier is refused.

## Writing a classifier

Start from `configs/classifiers/example/config.py`: a working classifier where every part is explained
(`cp -r configs/classifiers/example configs/classifiers/my_labels`).

A classifier is plain Python: `config.py` says what the model must answer (`Output`) and builds the request
for one record (`messages`). The smallest one:

```python
# configs/classifiers/is_math/config.py
from typing import Literal
from pydantic import BaseModel


class Output(BaseModel):          # the labels; one output field each
    is_math: bool
    level: Literal["school", "university", "research"]


def messages(record):             # one input record (dict) -> the request
    return f"""Is this a math problem, and at what level?

{record["problem"]}

Answer as JSON: "is_math" (true/false), "level" (school, university or research)."""
```
Check it on the first record without GPUs or a server (`--check-args`; needs openai (pydantic comes with it),
so on LUMI run it in the container, elsewhere `uv run --with openai python ...`):
```bash
python rollgen/classify.py --check-args --classifier is_math --input data/x.jsonl --output data/x.is_math.jsonl
```
Run it on LUMI: `--classifier` takes the folder name in `configs/classifiers/`, or a path to any folder with a
`config.py` (relative to the repo root, where you submit from). The launcher passes it to `classify.py`:
```bash
sbatch --nodes=1 lumi/classify.sbatch --model Qwen/Qwen3-32B --tp 4 \
    --input data/x.jsonl --output data/x.is_math.jsonl --classifier is_math --keep-fields uuid
# or a folder elsewhere:  --classifier /scratch/project_465003062/my_configs/is_math
```
At job start, before vLLM loads, the job runs this check on record 0 and stops on a mistake (see `slurm.out`).

`Output` is a [pydantic](https://docs.pydantic.dev/) model. It becomes the JSON schema that vLLM enforces
(structured outputs), and the reply is checked against it. Types:
- `bool`, `str`, `int`, `float`
- `int = Field(ge=1, le=5)`: an integer 1-5 (`from pydantic import Field`)
- `Literal["a", "b"]`: one of the given values
- `int | None`: may be null

Fields are generated in the order they're written, so a free-text `reasoning: str` first makes the model
think before it answers (it then needs a larger `--max-tokens`). The schema enforces the shape, but the model
only knows what a field *means* from your prompt, so describe the fields there.

`messages(record)` returns:
- a string: sent as one user message
- a list of chat messages, e.g. `[{"role": "system", "content": ...}, {"role": "user", "content": ...}]`
- `None`: skip this record (it gets no output line); filter whatever you like

If it raises (e.g. `KeyError` for a field the record doesn't have), that record gets a `classify_error`.
On the first record the job stops before it starts, so typos show up in `--check-args`. Different dataset
shapes are handled right there with ordinary `if`s (see `configs/classifiers/difficulty/config.py`: plain,
Nemotron-RL and chat `messages` records).

Helpers (`from _ask import truncate`):
- `truncate(text, n)`: cut text longer than `n` characters (head 2/3 + tail 1/3, `[... truncated ...]` between).

Every file of the folder is stored in `<output>.meta.json`, so resuming with an edited classifier is refused.

`configs/classifiers/difficulty/` as the full example:
`config.py`: `SYSTEM`, the instructions and the rubric as one string, with `{tools_note}` replaced by a
sentence saying whether the solver has tools; `Output` with `domain`, `task_type`, `difficulty` (each a
`Literal`); `messages()` reads the record's problem, reference answer and tools (from whichever shape it has),
cuts long texts to 12000 characters, and sends the system prompt plus the fields as a JSON user message.

The input must be JSONL (one record per line) or parquet; a pretty-printed JSON dump can be converted with
`jq -c . dump.json > data.jsonl`.
