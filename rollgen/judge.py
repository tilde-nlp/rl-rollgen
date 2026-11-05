"""Score rollouts with an LLM judge served by vLLM.

Input: rollouts .jsonl (e.g. from generate.py), one rollout per line, or .parquet.
Judge: a folder judges/<name>/ with prompt.txt and rubric.toml (see rollgen/_rubric.py and
judges/). The judge answers each rubric question as JSON; the reward is then computed from
those answers with the rubric's formula. Change the formula later without re-judging:
rollgen/rescore.py.

{prompt} in prompt.txt: if a rollout has its prompt as chat messages ("messages", or "prompt"
as a list), it is the last user message.

Output JSONL, one line per rollout:
  <rollout fields> + rollout_idx (input record number), judge_output (raw reply),
  judge_scores ({criterion: answer}), reward, judge_error (null, or why the reply was unusable;
  then judge_scores and reward are null)

Judge while generating (--follow): start this job while generate.py is still writing --input;
new rollouts are judged as they appear (checked every 30s). Stops when the generation run
finishes (per <input>.meta.json), or after --follow-timeout minutes with no new rollouts.
Needs a .jsonl input.

Resumable: rerun with the same arguments and finished rollouts are skipped.
Each run is recorded in <output>.meta.json, including the judge's prompt.txt and rubric.toml.

Example:
  python rollgen/judge.py --input rollouts.jsonl --output judged.jsonl --judge judges/correctness
"""

import argparse
import json

from _common import add_server_args, get_messages, read_jsonl, read_records, run_resumable
from _rubric import Judge


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--judge", required=True, help="judge folder with prompt.txt and rubric.toml")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--extra-body", type=json.loads, default={},
                   help="extra vLLM sampling params as JSON")
    p.add_argument("--follow", action="store_true",
                   help="--input is still being generated: judge new rollouts as they appear, "
                        "stop when the generation run finishes")
    p.add_argument("--follow-timeout", type=float, default=60,
                   help="with --follow: give up after this many minutes without new rollouts (default 60)")
    add_server_args(p)
    args = p.parse_args()
    if args.follow and not args.input.endswith(".jsonl"):
        p.error("--follow needs a .jsonl --input")

    judge = Judge(args.judge)
    judge.check_reward()
    criteria = judge.criteria_text()

    def load():
        records = read_jsonl(args.input, growing=True) if args.follow else read_records(args.input)
        for i, r in enumerate(records):
            r["rollout_idx"] = i  # line number: stable, since generate.py only appends
        if records:
            render(records[0])  # fail fast on a bad prompt, before sending anything
        return records

    def render(r):
        fields = dict(r)
        if not isinstance(fields.get("prompt"), str):
            messages = get_messages(r) or []
            fields["prompt"] = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        fields["criteria"] = criteria
        try:
            return judge.prompt.format_map(fields)
        except KeyError as e:
            raise SystemExit(f"prompt.txt placeholder {e} not found in rollout fields: {sorted(fields)}")

    async def worker(r, client):
        resp = await client.chat.completions.create(
            model=args.served_name,
            messages=[{"role": "user", "content": render(r)}],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            response_format=judge.response_format(),
            extra_body=args.extra_body,
        )
        text = resp.choices[0].message.content or ""
        scores, error = judge.parse(text)
        reward = judge.reward(scores) if scores is not None else None
        return [{**r, "judge_output": text, "judge_scores": scores, "reward": reward, "judge_error": error}]

    run_resumable(load(), worker, args, "rollout_idx",
                  record={"judge_prompt": judge.prompt, "judge_rubric": judge.rubric_text},
                  follow=(load, args.input + ".meta.json", args.follow_timeout) if args.follow else None)


if __name__ == "__main__":
    main()
