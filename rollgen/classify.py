"""Label records (e.g. the prompts of a dataset) with an LLM classifier served by vLLM.

Input: .jsonl or .parquet, one record per line. Read as a stream, so files of many GB are fine.
Classifier: a folder classifiers/<name>/ with prompt.txt or prompt.json, rubric.toml (the labels
as [criteria], no reward) and optionally system.txt. See docs/classifiers.md.

Output: a sidecar .jsonl, one line per input record (without the input fields, except --keep-fields):
  record_idx (input record number, from 0; blank lines don't count), the --keep-fields (e.g. uuid),
  one field per criterion (null if a `when` criterion wasn't asked), reasoning (only if the
  rubric has reasoning = true), classify_error (null, or why there are no labels; then they are null)
Lines are written as requests finish, not in input order: join on record_idx or a kept id field.

Identical requests (e.g. duplicated questions) are sent once per run; the answer is reused.
Resumable: rerun with the same arguments and finished records are skipped.
Each run is recorded in <output>.meta.json, including the classifier folder's files.

Example:
  python rollgen/classify.py --input data/so_openq.jsonl --output data/so_openq.difficulty.jsonl \\
      --classifier classifiers/difficulty --keep-fields uuid
"""

import argparse
import json
import sys

from _common import add_server_args, count_records, iter_records, run_resumable
from _rubric import REASONING, Rubric


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--classifier", required=True, help="classifier folder with prompt.txt/.json and rubric.toml")
    p.add_argument("--keep-fields", default="",
                   help="comma-separated input fields copied to every output line, e.g. uuid")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=512,
                   help="max tokens of the reply (default 512; the reply is just the JSON labels "
                        "unless the rubric has reasoning = true)")
    p.add_argument("--extra-body", type=json.loads, default={},
                   help='extra vLLM sampling params as JSON, e.g. for Qwen3/Gemma: '
                        '\'{"chat_template_kwargs": {"enable_thinking": false}}\'')
    add_server_args(p)
    args = p.parse_args()

    clf = Rubric(args.classifier, need_reward=False)
    keep = [f.strip() for f in args.keep_fields.split(",") if f.strip()]
    clash = set(keep) & {"record_idx", "classify_error", REASONING, *clf.criteria}
    if clash:
        p.error(f"--keep-fields {sorted(clash)}: these names are output fields already")

    first = next(iter_records(args.input), None)
    if first is not None:  # fail fast, before sending anything
        clf.check_record(first, f"{args.input} record 0")
        missing = [f for f in keep if f not in first]
        if missing:
            sys.exit(f"--keep-fields {missing} not in {args.input} record 0 (it has {sorted(first)})")

    def records():
        for i, r in enumerate(iter_records(args.input)):
            r["record_idx"] = i
            yield r

    cache = {}

    async def worker(r, client):
        answers, text, error = await clf.ask(client, args.served_name, r, args.temperature,
                                             args.max_tokens, args.extra_body, cache=cache)
        if error and text is not None:
            error += f"; reply: {text[:500]!r}"
        out = {"record_idx": r["record_idx"], **{f: r.get(f) for f in keep}}
        out.update(answers or dict.fromkeys(clf.criteria))
        if clf.reasoning:
            out[REASONING] = json.loads(text).get(REASONING) if answers else None
        out["classify_error"] = error
        return [out]

    run_resumable(records(), worker, args, "record_idx", record=clf.record("classifier"),
                  total=count_records(args.input))


if __name__ == "__main__":
    main()
