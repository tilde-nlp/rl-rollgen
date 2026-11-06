"""Label records (e.g. the prompts of a dataset) with an LLM classifier served by vLLM.

Input: .jsonl or .parquet, one record per line. Read as a stream, so files of many GB are fine.
Several files in one run: --input A B C --output-suffix .difficulty.jsonl writes A's labels next to A
(a.jsonl -> a.difficulty.jsonl), and so on. All files are checked first (record 0, --keep-fields,
resume settings), then classified one after another by the same servers: one job, one vLLM startup.
Classifier: a folder configs/classifiers/<name>/ with config.py: class Output (the labels) and
messages(record) (the request; None skips the record). See rollgen/_ask.py and docs/classifiers.md.

Output: a sidecar .jsonl, one line per classified record (without the input fields, except --keep-fields):
  record_idx (input record number, from 0; blank lines don't count), the --keep-fields (e.g. uuid),
  one field per Output field, classify_error (null, or why there are no labels; then they are null)
Lines are written as requests finish, not in input order: join on record_idx or a kept id field.
Records that messages() skips get no line.

Identical requests (e.g. duplicated questions) are sent once per run, also across files.
Resumable: rerun with the same arguments and finished records are skipped.
Each output has its own <output>.meta.json (its input, the arguments, the classifier folder's files).

Examples:
  python rollgen/classify.py --input data/so_openq.jsonl --output data/so_openq.difficulty.jsonl \\
      --classifier difficulty --keep-fields uuid
  python rollgen/classify.py --input data/a.jsonl data/b.jsonl --output-suffix .difficulty.jsonl \\
      --classifier difficulty
"""

import argparse
import json
import sys
from pathlib import Path

from _ask import Config
from _common import add_server_args, count_records, iter_records, resolve_config, run_resumable, start_meta


def output_path(input_path, suffix):
    """data/a.jsonl + .difficulty.jsonl -> data/a.difficulty.jsonl (same for .parquet inputs)."""
    stem = str(input_path)
    for ext in (".jsonl", ".parquet"):
        stem = stem.removesuffix(ext)
    return stem + suffix


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, nargs="+", metavar="FILE", help="one or more .jsonl/.parquet files")
    out = p.add_mutually_exclusive_group(required=True)
    out.add_argument("--output", help="labels .jsonl (one --input only)")
    out.add_argument("--output-suffix", metavar="SUFFIX",
                     help="write each input's labels next to it: a.jsonl -> a<SUFFIX>, e.g. .difficulty.jsonl")
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

    if args.output and len(args.input) > 1:
        p.error("several --input files need --output-suffix (each gets its own output next to it)")
    args.classifier = resolve_config(args.classifier, "classifiers")
    clf = Config(args.classifier)
    record = clf.record("classifier")
    keep = [f.strip() for f in args.keep_fields.split(",") if f.strip()]
    clash = set(keep) & {"record_idx", "classify_error", *clf.fields}
    if clash:
        p.error(f"--keep-fields {sorted(clash)}: these names are output fields already")

    # One argument set per file: what <output>.meta.json records for it is the same as for a
    # single-file run (input and output are that file's), so either form resumes the other.
    jobs = []
    for path in args.input:
        output = args.output or output_path(path, args.output_suffix)
        if Path(output).resolve() == Path(path).resolve():
            p.error(f"--output-suffix {args.output_suffix}: the output would overwrite the input {path}")
        a = argparse.Namespace(**{**vars(args), "input": path, "output": output})
        del a.output_suffix
        jobs.append(a)
    outputs = [a.output for a in jobs]
    if len(set(outputs)) < len(outputs):
        p.error("two inputs map to the same output; the --input list has duplicates")

    # Check every file before sending anything, so a bad file doesn't fail hours into the job.
    for a in jobs:
        first = next(iter_records(a.input), None)
        if first is not None:
            clf.check_record(first, f"{a.input} record 0")
            missing = [f for f in keep if f not in first]
            if missing:
                sys.exit(f"--keep-fields {missing} not in {a.input} record 0 (it has {sorted(first)})")
        start_meta(a, dry_run=True, record=record)  # refuses resuming with other settings
    if args.check_args:
        print(f"arguments OK ({len(jobs)} file{'s' * (len(jobs) > 1)})")
        return

    cache = {}  # shared by all files: a question seen in an earlier file isn't asked again

    def records(path):
        for i, r in enumerate(iter_records(path)):
            r["record_idx"] = i
            yield r

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

    failed = []
    for n, a in enumerate(jobs, 1):
        if len(jobs) > 1:
            print(f"\n=== file {n}/{len(jobs)}: {a.input} -> {a.output}", flush=True)
        if not run_resumable(records(a.input), worker, a, "record_idx", record=record,
                             total=count_records(a.input)):
            failed.append(a.input)
    if failed:
        sys.exit(f"{len(failed)} of {len(jobs)} files had failed records (rerun to retry them): {failed}")


if __name__ == "__main__":
    main()
