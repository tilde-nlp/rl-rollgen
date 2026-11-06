"""Grade a GSM8K solution against the reference answer and solution.
Reads a rollout from generate.py on configs/examples/gsm8k.jsonl: prompt, answer, reference_solution, completion."""

from pydantic import BaseModel, Field

from _ask import last_user_message


class Output(BaseModel):     # stored per rollout as judge_scores
    reasoning: str           # first, so the judge reasons before it answers
    correct: bool
    valid_reasoning: bool
    answer_line: bool
    clarity: int = Field(ge=1, le=5)


def messages(r):
    return f"""You are grading a model's solution to a grade-school math word problem.

Problem (including the instructions the model was given):
{last_user_message(r)}

Reference final answer: {r["answer"]}

Reference solution:
{r["reference_solution"]}

Model's response:
{r["completion"]}

Answer as a JSON object:
- "reasoning": a brief explanation
- "correct" (true/false): Is the model's final answer numerically equal to the reference final answer?
- "valid_reasoning" (true/false): Are all of the model's steps and calculations correct, with no lucky guesses or errors that cancel out?
- "answer_line" (true/false): Does the response end with a line of exactly the form "Answer: <number>", as instructed?
- "clarity" (integer 1-5): How clear and concise is the solution? 1 = confusing or padded, 5 = clear and to the point.

The model may solve the problem differently from the reference solution; that is fine if its steps are valid."""
