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
RESUME_SAFE_ARGS = {"urls", "concurrency", "check_args", "follow", "follow_timeout"}
FOLLOW_POLL_SECONDS = 30


def read_jsonl(path, growing=False):
    """growing=True: the file may still be being written by another job, so a missing file
    is empty and an unfinished last line (no trailing newline yet) is left for later."""
    if not Path(path).is_file():
        if growing:
            return []
        sys.exit(f"file not found: {path}")
    lines = Path(path).read_text().split("\n")
    if growing:
        lines = lines[:-1]  # text after the last newline: empty, or a line still being written
    return [json.loads(line) for line in lines if line.strip()]


def read_records(path):
    """Read input records from .jsonl or .parquet (e.g. a SkyRL/verl dataset)."""
    if str(path).endswith(".parquet"):
        if not Path(path).is_file():
            sys.exit(f"file not found: {path}")
        try:
            import pyarrow.parquet as pq
        except ImportError:
            sys.exit("reading .parquet needs pyarrow (pip install pyarrow)")
        return pq.read_table(path).to_pylist()
    return read_jsonl(path)


def get_messages(record):
    """Chat messages of a task: "messages", or "prompt" as a list of messages (SkyRL/verl
    datasets), or "prompt" as a string (sent as one user message). None if there's no prompt."""
    if record.get("messages"):
        return record["messages"]
    prompt = record.get("prompt")
    if isinstance(prompt, list):
        return prompt
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return None


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


def start_meta(args, dry_run=False, record=None):
    """Append this run to <output>.meta.json; refuse if an earlier run used different settings.
    `record`: extra settings to store and compare, e.g. the contents of a judge's files."""
    path = Path(args.output + ".meta.json")
    meta = json.loads(path.read_text()) if path.exists() else {"runs": []}
    settings = {k: v for k, v in vars(args).items() if k not in RESUME_SAFE_ARGS}
    # set by the cluster launcher, if any; MODEL_DIR pins the exact HF snapshot
    settings["model_path"] = os.environ.get("MODEL_DIR") or os.environ.get("MODEL")
    settings.update(record or {})
    if meta["runs"] and Path(args.output).exists():
        prev = meta["runs"][-1]["settings"]
        diff = {k: (prev.get(k), settings.get(k)) for k in prev.keys() | settings.keys()
                if prev.get(k) != settings.get(k)}
        if diff:
            lines = "\n".join(f"  {k}: changed (file contents; see {path})" if "\n" in f"{a}{b}" else
                              f"  {k}: before={a!r} now={b!r}" for k, (a, b) in sorted(diff.items()))
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
        "n_servers": len([u for u in (getattr(args, "urls", None) or "").split(",") if u.strip()]),
    }
    meta["runs"].append(run)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2) + "\n")
    return path, meta


def finish_meta(path, meta, **result):
    meta["runs"][-1].update(finished=_now(), **result)
    path.write_text(json.dumps(meta, indent=2) + "\n")


def run_resumable(items, worker, args, key, record=None, follow=None):
    """Call `await worker(item, client) -> list[dict]` for every item and append the returned
    records to `output` (JSONL) as soon as each item finishes.

    Each item must have a unique `key` field. Items whose key already appears in `output`
    are skipped, so a job that timed out can simply be resubmitted with the same arguments.
    Items that fail are logged and skipped; rerun to retry them.
    `args` needs .output and .concurrency; all of it (plus `record`) is recorded in <output>.meta.json.

    follow=(load, producer_meta, timeout_min): the input is still being written by another job
    (e.g. judging while generating). See _follow.
    """
    if args.check_args:
        start_meta(args, dry_run=True, record=record)
        print(f"arguments OK ({len(items)} items)")
        sys.exit(0)
    clients = make_clients(args)
    meta_path, meta = start_meta(args, record=record)
    complete = True
    if follow:
        n_ok, n_fail, complete = asyncio.run(
            _follow(*follow, worker, Path(args.output), key, clients, args.concurrency))
    else:
        n_ok, n_fail = asyncio.run(_run(items, worker, Path(args.output), key, clients, args.concurrency))
    finish_meta(meta_path, meta, items_ok=n_ok, items_failed=n_fail, **({} if complete else {"incomplete": True}))
    if n_fail or not complete:
        sys.exit(1)


def _producer_finished(meta_path, since):
    """True if the newest run in the producer's <output>.meta.json finished at or after `since`."""
    try:
        runs = json.loads(Path(meta_path).read_text())["runs"]
    except (OSError, ValueError, KeyError):  # not there yet, or caught mid-write
        return False
    return bool(runs) and (runs[-1].get("finished") or "") >= since


async def _follow(load, producer_meta, timeout_min, worker, output, key, clients, concurrency):
    """Process items as another job appends them to the input: every FOLLOW_POLL_SECONDS, load()
    the input again and process the new items. Stop when the producer's run (producer_meta)
    finishes after we started and everything is processed, or when nothing new has appeared for
    timeout_min minutes (producer crashed or timed out; rerun to continue).
    Items that fail are not retried in this process; rerun to retry them."""
    since = _now()
    attempted = {r[key] for r in read_jsonl(output)} if output.exists() else set()
    n_ok = n_fail = 0
    last_new = time.time()
    print(f"following {producer_meta.removesuffix('.meta.json')} (new lines every {FOLLOW_POLL_SECONDS}s)", flush=True)
    while True:
        finished = _producer_finished(producer_meta, since)  # check before loading: no line is missed
        new = [it for it in load() if it[key] not in attempted]
        if new:
            ok, fail = await _run(new, worker, output, key, clients, concurrency)
            attempted.update(it[key] for it in new)
            n_ok, n_fail = n_ok + ok, n_fail + fail
            last_new = time.time()
        elif finished:
            print("the generation run finished and all its lines are processed", flush=True)
            return n_ok, n_fail, True
        elif time.time() - last_new > timeout_min * 60:
            print(f"WARNING: nothing new for {timeout_min:g} min and {producer_meta} shows no finished run; "
                  f"stopping. Rerun to process the rest.", flush=True)
            return n_ok, n_fail, False
        await asyncio.sleep(FOLLOW_POLL_SECONDS)


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
