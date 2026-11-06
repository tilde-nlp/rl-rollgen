"""Score rollouts with an LLM judge served by vLLM.

Input: rollouts .jsonl (e.g. from generate.py), one rollout per line, or .parquet.
Judge: a folder configs/judges/<name>/ with config.py: class Output (what the judge answers) and
messages(record) (the request; None skips the rollout). See rollgen/_ask.py and docs/judges.md.
The answers are stored as they are (no reward is computed).

Output JSONL, one line per rollout:
  <rollout fields> + rollout_idx (input record number), judge_output (raw reply),
  judge_scores (the reply's Output fields), judge_error (null, or why there is no usable reply,
  e.g. messages() failed or the reply doesn't fit Output; then judge_scores is null)
Rollouts that messages() skips get no line.

Judge while generating (--follow): start this job while generate.py is still writing --input;
new rollouts are judged as they appear (checked every 30s). Stops when the generation run
finishes (per <input>.meta.json), or after --follow-timeout minutes with no new rollouts.
Needs a .jsonl input.

Resumable: rerun with the same arguments and finished rollouts are skipped.
Each run is recorded in <output>.meta.json, including the judge folder's files.

Example:
  python rollgen/judge.py --input rollouts.jsonl --output judged.jsonl --judge correctness
"""

import argparse
import json
import sys

from _common import add_server_args, count_records, iter_records, read_growing_jsonl, resolve_config, run_resumable
from _ask import Config


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--judge", required=True, help="name in configs/judges/ (e.g. correctness) or a folder path")
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

    args.judge = resolve_config(args.judge, "judges")
    judge = Config(args.judge)

    def rollouts():
        # rollout_idx = record number: stable, since generate.py only appends
        return ({**r, "rollout_idx": i} for i, r in enumerate(iter_records(args.input)))

    def load_growing():
        records = [{**r, "rollout_idx": i} for i, r in enumerate(read_growing_jsonl(args.input))]
        if records:
            judge.check_record(records[0], f"{args.input} record 0")
        return records

    async def worker(r, client):
        result = await judge.ask(client, args.served_name, r, args.temperature,
                                 args.max_tokens, args.extra_body)
        if result is None:
            return None  # skipped by messages()
        scores, text, error = result
        return [{**r, "judge_output": text, "judge_scores": scores, "judge_error": error}]

    record = judge.record("judge")
    if args.follow:
        ok = run_resumable(load_growing(), worker, args, "rollout_idx", record=record,
                           follow=(load_growing, args.input + ".meta.json", args.follow_timeout))
    else:
        first = next(rollouts(), None)
        if first is not None:
            judge.check_record(first, f"{args.input} record 0")  # fail fast, before sending anything
        ok = run_resumable(rollouts(), worker, args, "rollout_idx", record=record, total=count_records(args.input))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
