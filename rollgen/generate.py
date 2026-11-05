"""Generate N completions per prompt from vLLM (one-step RL rollouts).

Input: .jsonl (one task per line) or .parquet (e.g. a SkyRL/verl dataset). Each task has one of:
  "messages": [{"role": ..., "content": ...}, ...]   chat messages
  "prompt":   [{"role": ..., "content": ...}, ...]   chat messages (SkyRL/verl style)
  "prompt":   "..."                                    sent as a single user message
Any other fields (answer, reward_spec, id, ...) are copied unchanged to every output record.

Output JSONL, one line per sample:
  <input fields> + task_idx (input record number), sample_idx, completion, finish_reason, prompt_tokens

Resumable: rerun with the same arguments and finished tasks are skipped.
Each run is recorded in <output>.meta.json.

Example:
  python rollgen/generate.py --input prompts.jsonl --output rollouts.jsonl --samples-per-prompt 8 --max-tokens 2048
"""

import argparse
import json

from _common import add_server_args, get_messages, read_records, run_resumable


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--samples-per-prompt", type=int, default=8,
                   help="completions generated per prompt (the GRPO group size)")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--extra-body", type=json.loads, default={},
                   help='extra vLLM sampling params as JSON, e.g. \'{"top_k": 50, "min_p": 0.05}\'')
    add_server_args(p)
    args = p.parse_args()

    tasks = read_records(args.input)
    for i, t in enumerate(tasks):
        t["task_idx"] = i
        if get_messages(t) is None:
            raise SystemExit(f"record {i} of {args.input} has no 'messages' or 'prompt' (string or message list)")

    async def worker(task, client):
        messages = get_messages(task)
        resp = await client.chat.completions.create(
            model=args.served_name,
            messages=messages,
            n=args.samples_per_prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            extra_body=args.extra_body,
        )
        return [
            {
                **task,
                "sample_idx": ch.index,
                "completion": ch.message.content,
                "finish_reason": ch.finish_reason,
                "prompt_tokens": resp.usage.prompt_tokens,
            }
            for ch in resp.choices
        ]

    run_resumable(tasks, worker, args, "task_idx")


if __name__ == "__main__":
    main()
