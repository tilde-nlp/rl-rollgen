"""Example classifier: copy this folder to make your own (cp -r configs/classifiers/example configs/classifiers/NAME).

classify.py loads this file and needs exactly two things from it:

  Output          a pydantic model: the labels the model must answer. Its JSON schema is sent to vLLM, which
                  forces the reply into that shape; the reply is then checked against it.
  messages(rec)   builds the request for one input record (a dict: one line of the input .jsonl/.parquet).

Everything else here (constants, helper functions, imports) is up to you: it's plain Python.
All files in this folder are saved in <output>.meta.json; resuming with an edited config is refused.

Output file: one line per record, {"record_idx": ..., <--keep-fields>, <one field per Output field>,
"classify_error": null or why there are no labels}.

Try it on the example data (no GPUs needed for --check-args):
  python rollgen/classify.py --check-args --classifier example \\
      --input configs/examples/prompts.jsonl --output data/example.labels.jsonl --keep-fields id
"""

from typing import Literal

from pydantic import BaseModel, Field

# Helpers from rollgen/_ask.py (optional):
#   truncate(text, n)  cut text longer than n characters (head 2/3 + tail 1/3 kept, a marker between)
from _ask import truncate


# --- What the model answers ----------------------------------------------------------------------
# Each field becomes one field of the output line. Field order = generation order, so a free-text
# `reasoning: str` first makes the model think before it labels (then raise --max-tokens).
# The model doesn't see these names' meaning from the schema: describe each field in the prompt.
class Output(BaseModel):
    topic: Literal["math", "science", "code", "other"]   # one of these strings
    needs_calculation: bool                               # true / false
    steps: int = Field(ge=1, le=10)                       # integer 1-10
    # also possible: str (free text), float, `int | None` (may be null)


SYSTEM = """You label tasks from a training dataset. You do not solve them.

Reply with a JSON object:
- "topic": the subject: math, science, code or other
- "needs_calculation": true if solving it requires arithmetic or other computation
- "steps": roughly how many reasoning steps a solution needs, 1-10"""


# --- The request for one record ------------------------------------------------------------------
# Return one of:
#   a str              sent as one user message
#   a list of messages [{"role": "system" | "user" | "assistant", "content": ...}, ...]
#   None               skip this record: no request, no output line (filter whatever you like)
# If it raises (e.g. KeyError for a missing field), this record gets a classify_error line instead;
# on the first record the whole job stops before it starts, so typos show up in --check-args.
def messages(record):
    # Different dataset shapes are just ifs. Here: "prompt" as a string, or chat "messages".
    if isinstance(record.get("prompt"), str):
        task = record["prompt"]
    else:
        task = "\n\n".join(m["content"] for m in record["messages"] if m["role"] in ("system", "user"))

    if not task.strip():
        return None  # nothing to classify

    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Task:\n{truncate(task, 12000)}"},
    ]
