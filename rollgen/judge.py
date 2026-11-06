"""Score rollouts with an LLM judge served by vLLM.

Input: rollouts .jsonl (e.g. from generate.py), one rollout per line, or .parquet.
Judge: a folder judges/<name>/ with prompt.txt (or prompt.json), rubric.toml and optionally
system.txt (see rollgen/_rubric.py and docs/judges.md). The judge answers each rubric question as
JSON; the reward is then computed from those answers with the rubric's formula. Change the formula
later without re-judging: rollgen/rescore.py.

{prompt} in the prompt: if a rollout has its prompt as chat messages ("messages", or "prompt"
as a list), it is the last user message.

Output JSONL, one line per rollout:
  <rollout fields> + rollout_idx (input record number), judge_output (raw reply),
  judge_scores ({criterion: answer}), reward, judge_error (null, or why there is no usable reply,
  e.g. a missing prompt field; then judge_scores and reward are null. If only the reward formula
  failed, judge_scores are kept: fix the formula and run rescore.py)

Judge while generating (--follow): start this job while generate.py is still writing --input;
new rollouts are judged as they appear (checked every 30s). Stops when the generation run
finishes (per <input>.meta.json), or after --follow-timeout minutes with no new rollouts.
Needs a .jsonl input.

Resumable: rerun with the same arguments and finished rollouts are skipped.
Each run is recorded in <output>.meta.json, including the judge folder's files.

Example:
  python rollgen/judge.py --input rollouts.jsonl --output judged.jsonl --judge judges/correctness
"""

import argparse
import json

from _common import add_server_args, count_records, iter_records, read_growing_jsonl, run_resumable
from _rubric import Rubric


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

    judge = Rubric(args.judge, need_reward=True)
    judge.check_reward()

    def rollouts():
        # rollout_idx = record number: stable, since generate.py only appends
        return ({**r, "rollout_idx": i} for i, r in enumerate(iter_records(args.input)))

    def load_growing():
        records = [{**r, "rollout_idx": i} for i, r in enumerate(read_growing_jsonl(args.input))]
        if records:
            judge.check_record(records[0], f"{args.input} record 0")
        return records

    async def worker(r, client):
        scores, text, error = await judge.ask(client, args.served_name, r, args.temperature,
                                              args.max_tokens, args.extra_body)
        reward = None
        if scores is not None:
            try:
                reward = judge.reward(scores)
            except Exception as e:  # scores are kept: fix the formula, then rollgen/rescore.py
                error = f"reward formula failed: {type(e).__name__}: {e}"
        return [{**r, "judge_output": text, "judge_scores": scores, "reward": reward, "judge_error": error}]

    record = judge.record("judge")
    if args.follow:
        run_resumable(load_growing(), worker, args, "rollout_idx", record=record,
                      follow=(load_growing, args.input + ".meta.json", args.follow_timeout))
    else:
        first = next(rollouts(), None)
        if first is not None:
            judge.check_record(first, f"{args.input} record 0")  # fail fast, before sending anything
        run_resumable(rollouts(), worker, args, "rollout_idx", record=record, total=count_records(args.input))


if __name__ == "__main__":
    main()
