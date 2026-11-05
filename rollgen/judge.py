"""Score rollouts with an LLM judge served by vLLM.

Input: rollouts .jsonl (e.g. from generate.py), one rollout per line, or .parquet.
Template: a text file sent to the judge as a single user message. {field} placeholders are
filled from each rollout's fields, e.g. {prompt}, {completion}, {answer}. If a rollout has
its prompt as chat messages ("messages", or "prompt" as a list), {prompt} is the last user message. Write literal braces as {{ }}.

The judge's reply is searched with --score-regex; the LAST match's first group is the score.

Output JSONL, one line per rollout:
  <rollout fields> + rollout_idx (input record number), judge_output, judge_score (float, or null
  if the regex didn't match)

Resumable: rerun with the same arguments and finished rollouts are skipped.
Each run is recorded in <output>.meta.json.

Example:
  python rollgen/judge.py --input rollouts.jsonl --output judged.jsonl \\
      --template judge_prompts/correctness.txt
"""

import argparse
import json
import re
from pathlib import Path

from _common import add_server_args, get_messages, read_records, run_resumable


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--template", required=True, help="judge prompt template file")
    p.add_argument("--score-regex", default=r"SCORE:\s*([0-9]+(?:\.[0-9]+)?)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--extra-body", type=json.loads, default={},
                   help="extra vLLM sampling params as JSON")
    add_server_args(p)
    args = p.parse_args()

    template = Path(args.template).read_text()
    score_re = re.compile(args.score_regex)
    rollouts = read_records(args.input)
    for i, r in enumerate(rollouts):
        r["rollout_idx"] = i

    def render(r):
        fields = dict(r)
        if not isinstance(fields.get("prompt"), str):
            messages = get_messages(r) or []
            fields["prompt"] = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        try:
            return template.format_map(fields)
        except KeyError as e:
            raise SystemExit(f"template placeholder {e} not found in rollout fields: {sorted(fields)}")

    render(rollouts[0])  # fail fast on a bad template, before sending anything

    async def worker(r, client):
        resp = await client.chat.completions.create(
            model=args.served_name,
            messages=[{"role": "user", "content": render(r)}],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            extra_body=args.extra_body,
        )
        text = resp.choices[0].message.content or ""
        matches = score_re.findall(text)
        return [{**r, "judge_output": text, "judge_score": float(matches[-1]) if matches else None}]

    run_resumable(rollouts, worker, args, "rollout_idx")


if __name__ == "__main__":
    main()
