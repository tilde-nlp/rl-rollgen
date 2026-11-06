"""Example judge: copy this folder to make your own (cp -r configs/judges/example configs/judges/NAME).

judge.py loads this file and needs exactly two things from it:

  Output        a pydantic model: what the judge must answer. Its JSON schema is sent to vLLM, which
                forces the reply into that shape; the reply is then checked against it.
  messages(r)   builds the request for one rollout (a dict: one line of the rollouts .jsonl).

Everything else here (constants, helper functions, imports) is up to you: it's plain Python.
All files in this folder are saved in <output>.meta.json; resuming with an edited config is refused.

A rollout from generate.py has the input task's fields (e.g. "prompt" or "messages", "answer", "id")
plus task_idx, sample_idx, completion, finish_reason, prompt_tokens.

Output file: the rollout's fields plus
  judge_scores  the reply as a dict, one key per Output field (null on error)
  judge_output  the raw reply text
  judge_error   null, or why there are no scores (messages() failed, or the reply doesn't fit Output)
No reward is computed: turning judge_scores into a reward is up to the training side.

Try it on rollouts of the example data (no GPUs needed for --check-args):
  python rollgen/judge.py --check-args --judge example --input data/rollouts.jsonl --output data/judged.jsonl
"""

from pydantic import BaseModel, Field

# Helpers from rollgen/_ask.py (optional):
#   last_user_message(r)  the rollout's prompt as text ("prompt" if it's a string, else the last user message)
#   truncate(text, n)     cut text longer than n characters (head 2/3 + tail 1/3 kept, a marker between)
from _ask import last_user_message, truncate


# --- What the judge answers ----------------------------------------------------------------------
# Field order = generation order: `reasoning` first, so the judge explains before it decides.
# Stored as judge_scores, e.g. {"reasoning": "...", "correct": true, "clarity": 4}.
# The judge doesn't see these names' meaning from the schema: describe each field in the prompt.
class Output(BaseModel):
    reasoning: str                     # free text
    correct: bool                      # true / false
    clarity: int = Field(ge=1, le=5)   # integer 1-5
    # also possible: Literal["a", "b"] (one of these strings), float, `int | None` (may be null)


# --- The request for one rollout -----------------------------------------------------------------
# Return one of:
#   a str              sent as one user message
#   a list of messages [{"role": "system" | "user" | "assistant", "content": ...}, ...]
#   None               skip this rollout: no request, no output line (filter whatever you like)
# If it raises (e.g. KeyError for a missing field), this rollout gets a judge_error line instead;
# on the first rollout the whole job stops before it starts, so typos show up in --check-args.
def messages(r):
    if r["finish_reason"] == "length":
        return None  # cut off by --max-tokens during generation: nothing to judge

    return f"""You are grading a model's answer to a problem.

Problem:
{last_user_message(r)}

Reference answer:
{r["answer"]}

Model's response:
{truncate(r["completion"], 20000)}

Reply with a JSON object:
- "reasoning": a brief explanation
- "correct" (true/false): is the model's final answer equivalent to the reference answer?
- "clarity" (integer 1-5): how clear is the solution? 1 = confusing, 5 = clear and to the point"""
