"""Recompute rewards of judged rollouts with a (changed) rubric formula. No GPUs, no judge calls.

Reads the stored judge_scores from judge.py output and applies the reward formula from
--judge's rubric.toml. The rubric must have the same criteria as the one used for judging
(change questions -> rerun judge.py; change only the formula -> rescore).
Rollouts the judge couldn't score (judge_error set) keep reward null.

Output: the input records with `reward` replaced. Overwrites --output; the run is recorded in
<output>.meta.json, including the rubric.

Example:
  python rollgen/rescore.py --input judged.jsonl --output judged_v2.jsonl --judge judges/tool_use
"""

import argparse
import json
from pathlib import Path

from _common import finish_meta, read_records, start_meta
from _rubric import Rubric


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="judge.py output")
    p.add_argument("--output", required=True)
    p.add_argument("--judge", required=True, help="judge folder whose rubric.toml has the new formula")
    args = p.parse_args()

    judge = Rubric(args.judge, need_reward=True)
    judge.check_reward()
    records = read_records(args.input)
    if Path(args.input).resolve() == Path(args.output).resolve():
        raise SystemExit("--output must differ from --input")

    rescored = []
    for i, r in enumerate(records):
        scores = r.get("judge_scores")
        if "judge_scores" not in r:
            raise SystemExit(f"record {i} has no judge_scores; is {args.input} judge.py output?")
        if scores is not None and scores.keys() != judge.criteria.keys():
            raise SystemExit(f"record {i}: judged criteria {sorted(scores)} != rubric criteria "
                             f"{sorted(judge.criteria)}; changed questions need a new judge.py run")
        rescored.append({**r, "reward": judge.reward(scores) if scores is not None else None})

    Path(args.output + ".meta.json").unlink(missing_ok=True)  # a rescore is a fresh, cheap rewrite
    meta_path, meta = start_meta(args, record={"judge_rubric": judge.rubric_text})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as out:
        for r in rescored:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    finish_meta(meta_path, meta, items_ok=len(rescored),
                items_unscored=sum(r["reward"] is None for r in rescored))
    print(f"rescored {len(rescored)} rollouts -> {args.output}")


if __name__ == "__main__":
    main()
