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

Resumable: rerun with the same arguments and finished rollouts are skipped.
Each run is recorded in <output>.meta.json, including the judge's prompt.txt and rubric.toml.

Example:
  python rollgen/judge.py --input rollouts.jsonl --output judged.jsonl --judge judges/correctness
"""

import argparse
import json

from _common import add_server_args, get_messages, read_records, run_resumable
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
    add_server_args(p)
    args = p.parse_args()

    judge = Judge(args.judge)
    judge.check_reward()
    criteria = judge.criteria_text()
    rollouts = read_records(args.input)
    for i, r in enumerate(rollouts):
        r["rollout_idx"] = i

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

    render(rollouts[0])  # fail fast on a bad prompt, before sending anything

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

    run_resumable(rollouts, worker, args, "rollout_idx",
                  record={"judge_prompt": judge.prompt, "judge_rubric": judge.rubric_text})


if __name__ == "__main__":
    main()
