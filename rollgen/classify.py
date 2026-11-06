"""Label records (e.g. the prompts of a dataset) with an LLM classifier served by vLLM.

Input: .jsonl or .parquet, one record per line. Read as a stream, so files of many GB are fine.
Classifier: a folder configs/classifiers/<name>/ with config.py: class Output (the labels) and
messages(record) (the request; None skips the record). See rollgen/_ask.py and docs/classifiers.md.

Output: a sidecar .jsonl, one line per classified record (without the input fields, except --keep-fields):
  record_idx (input record number, from 0; blank lines don't count), the --keep-fields (e.g. uuid),
  one field per Output field, classify_error (null, or why there are no labels; then they are null)
Lines are written as requests finish, not in input order: join on record_idx or a kept id field.
Records that messages() skips get no line.

Identical requests (e.g. duplicated questions) are sent once per run; the answer is reused.
Resumable: rerun with the same arguments and finished records are skipped.
Each run is recorded in <output>.meta.json, including the classifier folder's files.

Example:
  python rollgen/classify.py --input data/so_openq.jsonl --output data/so_openq.difficulty.jsonl \\
      --classifier difficulty --keep-fields uuid
"""

import argparse
import json
import sys

from _ask import Config
from _common import add_server_args, count_records, iter_records, resolve_config, run_resumable


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--classifier", required=True, help="name in configs/classifiers/ (e.g. difficulty) or a folder path")
    p.add_argument("--keep-fields", default="",
                   help="comma-separated input fields copied to every output line, e.g. uuid")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=512,
                   help="max tokens of the reply (default 512; enough for the JSON labels, more if "
                        "Output has a free-text reasoning field)")
    p.add_argument("--extra-body", type=json.loads, default={},
                   help='extra vLLM sampling params as JSON, e.g. for Qwen3/Gemma: '
                        '\'{"chat_template_kwargs": {"enable_thinking": false}}\'')
    add_server_args(p)
    args = p.parse_args()

    args.classifier = resolve_config(args.classifier, "classifiers")
    clf = Config(args.classifier)
    keep = [f.strip() for f in args.keep_fields.split(",") if f.strip()]
    clash = set(keep) & {"record_idx", "classify_error", *clf.fields}
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
        result = await clf.ask(client, args.served_name, r, args.temperature,
                               args.max_tokens, args.extra_body, cache=cache)
        if result is None:
            return None  # skipped by messages()
        answers, text, error = result
        if error and text is not None:
            error += f"; reply: {text[:500]!r}"
        out = {"record_idx": r["record_idx"], **{f: r.get(f) for f in keep}}
        out.update(answers or dict.fromkeys(clf.fields))
        out["classify_error"] = error
        return [out]

    run_resumable(records(), worker, args, "record_idx", record=clf.record("classifier"),
                  total=count_records(args.input))


if __name__ == "__main__":
    main()
