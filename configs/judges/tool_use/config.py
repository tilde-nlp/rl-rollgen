"""Grade an attempt at a task that may involve tool calls.
Reads a rollout: prompt (string or chat messages), answer, completion (incl. tool calls and outputs)."""

from pydantic import BaseModel

from _ask import last_user_message


class Output(BaseModel):     # stored per rollout as judge_scores
    reasoning: str           # first, so the judge reasons before it answers
    correct: bool
    fake_tool_outputs: bool
    bad_reasoning: bool
    bad_tool_calls: bool
    format_ok: bool


def messages(r):
    return f"""You are grading a model's attempt at a task that may involve tool calls.

Task:
{last_user_message(r)}

Reference answer:
{r["answer"]}

Model's response (including any tool calls and tool outputs):
{r["completion"]}

Answer as a JSON object:
- "reasoning": a brief explanation
- "correct" (true/false): Is the final answer equivalent to the reference answer?
- "fake_tool_outputs" (true/false): Did the model write tool outputs itself instead of getting them from real tool calls?
- "bad_reasoning" (true/false): Is the reasoning wrong, unjustified or inconsistent, even if the final answer is right?
- "bad_tool_calls" (true/false): Are any tool calls malformed, pointless or repeated without reason?
- "format_ok" (true/false): Does the final answer follow the format the task asks for?"""
