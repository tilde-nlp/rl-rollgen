"""Is the rollout's final answer equivalent to the reference answer?
Reads a rollout from generate.py: prompt (string or chat messages), answer, completion."""

from pydantic import BaseModel

from _ask import last_user_message


class Output(BaseModel):     # stored per rollout as judge_scores
    reasoning: str           # first, so the judge reasons before it answers
    correct: bool


def messages(r):
    return f"""You are grading a model's answer to a problem.

Problem:
{last_user_message(r)}

Reference answer:
{r["answer"]}

Model's response:
{r["completion"]}

Answer as a JSON object:
- "reasoning": a brief explanation
- "correct" (true/false): Is the model's final answer equivalent to the reference answer?

Ignore formatting differences and the reasoning style; judge only the final answer."""
