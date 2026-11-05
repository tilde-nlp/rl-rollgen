"""Small helpers shared by the scripts in this folder: JSONL I/O, vLLM clients, and a
resumable async loop that writes results as they finish.

Scripts here never know which cluster they run on. They get vLLM endpoints from
--urls or the VLLM_URLS env var (comma-separated, e.g. "http://127.0.0.1:8000/v1"),
which the cluster launchers (lumi/, tilde/) set.

Every run is recorded in <output>.meta.json (arguments, model, git commit, timing). Resuming into
an existing output with different arguments is refused, so one file never mixes settings.
"""

import argparse
import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# Args that may differ between resumed runs without changing what the output contains.
RESUME_SAFE_ARGS = {"urls", "concurrency", "check_args"}


def read_jsonl(path):
    if not Path(path).is_file():
        sys.exit(f"file not found: {path}")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def add_server_args(parser):
    parser.add_argument("--urls", default=os.environ.get("VLLM_URLS"),
                        help="comma-separated OpenAI-compatible base URLs (default: $VLLM_URLS)")
    parser.add_argument("--served-name", default=os.environ.get("SERVED_NAME"),
                        help="model name to put in API requests (default: $SERVED_NAME)")
    parser.add_argument("--concurrency", type=int, default=64,
                        help="max in-flight requests per server")
    # Used by the cluster launchers: validate arguments and the resume state, then exit,
    # so mistakes show up before minutes of vLLM startup.
    parser.add_argument("--check-args", action="store_true", help=argparse.SUPPRESS)


def make_clients(args):
    from openai import AsyncOpenAI  # imported here so --help works without openai installed

    if not args.urls:
        sys.exit("no vLLM endpoints: pass --urls or set VLLM_URLS")
    if not args.served_name:
        sys.exit("no model name: pass --served-name or set SERVED_NAME")
    return [AsyncOpenAI(base_url=u.strip(), api_key="EMPTY", timeout=3600, max_retries=3)
            for u in args.urls.split(",") if u.strip()]


def _git(*cmd):
    try:
        return subprocess.run(["git", "-C", str(REPO), *cmd], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def start_meta(args, dry_run=False):
    """Append this run to <output>.meta.json; refuse if an earlier run used different settings."""
    path = Path(args.output + ".meta.json")
    meta = json.loads(path.read_text()) if path.exists() else {"runs": []}
    settings = {k: v for k, v in vars(args).items() if k not in RESUME_SAFE_ARGS}
    settings["model_path"] = os.environ.get("MODEL")  # set by the cluster launcher, if any
    if meta["runs"] and Path(args.output).exists():
        prev = meta["runs"][-1]["settings"]
        diff = {k: (prev.get(k), settings.get(k)) for k in prev.keys() | settings.keys()
                if prev.get(k) != settings.get(k)}
        if diff:
            lines = "\n".join(f"  {k}: before={a!r} now={b!r}" for k, (a, b) in sorted(diff.items()))
            sys.exit(f"{args.output} was started with different settings:\n{lines}\n"
                     f"Use a new --output, or delete {args.output} and {path} to start over.")
    if dry_run:
        return path, meta
    status = _git("status", "--porcelain")
    run = {
        "started": _now(),
        "finished": None,
        "settings": settings,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "n_servers": len([u for u in (args.urls or "").split(",") if u.strip()]),
    }
    meta["runs"].append(run)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2) + "\n")
    return path, meta


def finish_meta(path, meta, **result):
    meta["runs"][-1].update(finished=_now(), **result)
    path.write_text(json.dumps(meta, indent=2) + "\n")


def run_resumable(items, worker, args, key):
    """Call `await worker(item, client) -> list[dict]` for every item and append the returned
    records to `output` (JSONL) as soon as each item finishes.

    Each item must have a unique `key` field. Items whose key already appears in `output`
    are skipped, so a job that timed out can simply be resubmitted with the same arguments.
    Items that fail are logged and skipped; rerun to retry them.
    `args` needs .output and .concurrency; all of it is recorded in <output>.meta.json.
    """
    if args.check_args:
        start_meta(args, dry_run=True)
        print(f"arguments OK ({len(items)} items)")
        sys.exit(0)
    clients = make_clients(args)
    meta_path, meta = start_meta(args)
    n_ok, n_fail = asyncio.run(_run(items, worker, Path(args.output), key, clients, args.concurrency))
    finish_meta(meta_path, meta, items_ok=n_ok, items_failed=n_fail)
    if n_fail:
        sys.exit(1)


async def _run(items, worker, output, key, clients, concurrency):
    done = set()
    if output.exists():
        done = {r[key] for r in read_jsonl(output)}
    todo = [it for it in items if it[key] not in done]
    print(f"{len(items)} items, {len(done)} already in {output}, {len(todo)} to do, "
          f"{len(clients)} server(s)", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    sems = [asyncio.Semaphore(concurrency) for _ in clients]
    n_ok = n_fail = 0
    start = time.time()

    with open(output, "a") as out:
        async def one(i, item):
            nonlocal n_ok, n_fail
            c = i % len(clients)
            async with sems[c]:
                try:
                    records = await worker(item, clients[c])
                except Exception as e:
                    n_fail += 1
                    print(f"FAILED {key}={item[key]}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                    return
            for r in records:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
            out.flush()
            n_ok += 1
            if n_ok % 100 == 0:
                print(f"{n_ok}/{len(todo)} done, {n_ok / (time.time() - start):.2f} items/s", flush=True)

        await asyncio.gather(*(one(i, it) for i, it in enumerate(todo)))

    print(f"finished: {n_ok} ok, {n_fail} failed, {time.time() - start:.0f}s", flush=True)
    return n_ok, n_fail
