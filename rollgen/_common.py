"""Small helpers shared by the scripts in this folder: JSONL/parquet I/O, the vLLM client, and a
resumable async loop that writes results as they finish.

Scripts here never know which cluster they run on. They talk to one OpenAI-compatible URL
(--url or $VLLM_URL): a single vLLM server, or a vllm-router that spreads the requests over many
servers. The LUMI launchers start the router in front of all servers of a job and set VLLM_URL,
VLLM_SERVERS (how many servers are behind it) and SERVED_NAME.

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
# (judge/classifier: only their location; their file contents are compared instead)
RESUME_SAFE_ARGS = {"url", "servers", "concurrency", "check_args", "follow", "follow_timeout",
                    "judge", "classifier"}
FOLLOW_POLL_SECONDS = 30
PROGRESS_SECONDS = 30  # print progress at most this often


def read_growing_jsonl(path):
    """Records of a .jsonl that another job may still be writing: a missing file is empty, and an
    unfinished last line (no trailing newline yet) is left for later."""
    if not Path(path).is_file():
        return []
    lines = Path(path).read_text().split("\n")[:-1]  # after the last newline: empty, or still being written
    return [json.loads(line) for line in lines if line.strip()]


def _parquet(path):
    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("reading .parquet needs pyarrow (pip install pyarrow)")
    return pq.ParquetFile(path)


def iter_records(path):
    """Yield input records one by one from .jsonl (blank lines skipped) or .parquet, without loading
    the whole file, so inputs of many GB are fine."""
    if not Path(path).is_file():
        sys.exit(f"file not found: {path}")
    if str(path).endswith(".parquet"):
        for batch in _parquet(path).iter_batches(batch_size=1000):
            yield from batch.to_pylist()
        return
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    sys.exit(f"{path}: line {n} is not valid JSON: {e}")


def count_records(path):
    """Number of records iter_records(path) yields (a quick pass over the file)."""
    if not Path(path).is_file():
        sys.exit(f"file not found: {path}")
    if str(path).endswith(".parquet"):
        return _parquet(path).metadata.num_rows
    with open(path, "rb") as f:
        return sum(1 for line in f if line.strip())


def read_records(path):
    """Read all input records from .jsonl or .parquet (e.g. a SkyRL/verl dataset)."""
    return list(iter_records(path))


def resolve_config(value, kind, suffix=""):
    """A config given by short name or path: `--judge gsm8k` -> configs/judges/gsm8k.
    An existing path is used as given."""
    if value is None or Path(value).exists():
        return value
    path = REPO / "configs" / kind / (value + suffix)
    if path.exists():
        return str(path.relative_to(Path.cwd())) if path.is_relative_to(Path.cwd()) else str(path)
    names = sorted(p.name.removesuffix(suffix) for p in (REPO / "configs" / kind).iterdir()
                   if not p.name.startswith(("_", ".")) and p.name.endswith(suffix))
    sys.exit(f"{value}: no such path, and not one of configs/{kind}/: {', '.join(names)}")


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
    parser.add_argument("--url", default=os.environ.get("VLLM_URL"),
                        help="OpenAI-compatible base URL: one vLLM server, or a vllm-router in front of "
                             "several (default: $VLLM_URL, set by the launcher)")
    parser.add_argument("--servers", type=int, default=int(os.environ.get("VLLM_SERVERS") or 1),
                        help="number of vLLM servers behind --url (default: $VLLM_SERVERS, set by the "
                             "launcher, else 1)")
    parser.add_argument("--served-name", default=os.environ.get("SERVED_NAME"),
                        help="model name to put in API requests (default: $SERVED_NAME)")
    parser.add_argument("--concurrency", type=int, default=64,
                        help="max in-flight requests per server (in total: concurrency x servers)")
    # Used by the cluster launchers: validate arguments and the resume state, then exit,
    # so mistakes show up before minutes of vLLM startup.
    parser.add_argument("--check-args", action="store_true", help=argparse.SUPPRESS)


def make_client(args, connections):
    # imported here so --help works without openai installed
    from openai import AsyncOpenAI, DefaultAsyncHttpxClient
    try:  # the HTTP library under openai: httpx2 in newer releases, httpx before
        import httpx2 as httpx
        if not issubclass(DefaultAsyncHttpxClient, httpx.AsyncClient):
            raise ImportError
    except ImportError:
        import httpx

    if not args.url:
        sys.exit("no vLLM URL: pass --url or set VLLM_URL")
    if "," in args.url:
        sys.exit("--url takes one URL. For several servers, start a vllm-router in front of them, e.g.\n"
                 "  uvx --from vllm-router==0.1.16 vllm-router --policy round_robin --port 9000 "
                 "--worker-urls http://host:8000 http://host:8001\n"
                 "and pass --url http://<router host>:9000/v1 --servers 2")
    if not args.served_name:
        sys.exit("no model name: pass --served-name or set SERVED_NAME")
    # API_KEY_FILE: set by the launcher's --api-key-file (the servers then require the key)
    key_file = os.environ.get("API_KEY_FILE")
    api_key = Path(key_file).read_text().splitlines()[0].strip() if key_file else "EMPTY"
    # one kept-open connection per in-flight request of this client (openai's default: at most 1000, 100 kept open)
    http = DefaultAsyncHttpxClient(limits=httpx.Limits(max_connections=connections,
                                                       max_keepalive_connections=connections))
    return AsyncOpenAI(base_url=args.url, api_key=api_key, timeout=3600, max_retries=3, http_client=http)


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
        "n_servers": getattr(args, "servers", None),
    }
    meta["runs"].append(run)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2) + "\n")
    return path, meta


def finish_meta(path, meta, **result):
    meta["runs"][-1].update(finished=_now(), **result)
    path.write_text(json.dumps(meta, indent=2) + "\n")


def run_resumable(items, worker, args, key, record=None, follow=None, total=None):
    """Call `await worker(item, client) -> list[dict]` for every item and append the returned
    records to `output` (JSONL) as soon as each item finishes (so in completion order).
    A worker returning None skips the item (nothing written; it's looked at again on a rerun).

    `items`: a list, or any iterable (e.g. a generator streaming a big file; then pass `total`,
    the number of items, for progress and ETA). --concurrency x --servers requests are in flight.
    Each item must have a unique `key` field. Items whose key already appears in `output`
    are skipped, so a job that timed out can simply be resubmitted with the same arguments.
    Items that fail are logged and skipped; rerun to retry them.
    `args` needs .output, .concurrency and .servers; all of it (plus `record`) is recorded in <output>.meta.json.

    follow=(load, producer_meta, timeout_min): the input is still being written by another job
    (e.g. judging while generating). See _follow.

    Returns True if every item succeeded (and, with follow, the producer finished); callers exit 1 otherwise.
    """
    if total is None and hasattr(items, "__len__"):
        total = len(items)
    if args.check_args:
        start_meta(args, dry_run=True, record=record)
        print(f"arguments OK ({total if total is not None else '?'} items)")
        return True
    # One client (= one connection pool) per server, each with --concurrency connections; all point at the
    # same URL. Not one big pool: httpx scans all of a pool's connections on every request, so with 8192
    # connections (128 servers) one pool capped the client at ~60 requests/s and the servers sat idle
    # (job 23640579, 2026-11-11).
    n_loops = args.concurrency * args.servers
    clients = [make_client(args, args.concurrency) for _ in range(args.servers)]
    output = Path(args.output)
    done = _done_keys(output, key)  # before start_meta: a refusal here leaves no unfinished run behind
    meta_path, meta = start_meta(args, record=record)
    complete = True
    if follow:
        n_ok, n_fail, n_skip, complete = asyncio.run(
            _follow(*follow, worker, output, key, clients, n_loops, done))
    else:
        n_ok, n_fail, n_skip = asyncio.run(_run(items, worker, output, key, clients, n_loops, done, total))
    finish_meta(meta_path, meta, items_ok=n_ok, items_failed=n_fail, items_skipped=n_skip,
                **({} if complete else {"incomplete": True}))
    return not n_fail and complete


def _producer_finished(meta_path, since):
    """True if the newest run in the producer's <output>.meta.json finished at or after `since`."""
    try:
        runs = json.loads(Path(meta_path).read_text())["runs"]
    except (OSError, ValueError, KeyError):  # not there yet, or caught mid-write
        return False
    return bool(runs) and (runs[-1].get("finished") or "") >= since


async def _follow(load, producer_meta, timeout_min, worker, output, key, clients, n_loops, done):
    """Process items as another job appends them to the input: every FOLLOW_POLL_SECONDS, load()
    the input again and process the new items. Stop when the producer's run (producer_meta)
    finishes after we started and everything is processed, or when nothing new has appeared for
    timeout_min minutes (producer crashed or timed out; rerun to continue).
    Items that fail are not retried in this process; rerun to retry them."""
    since = _now()
    attempted = set(done)
    n_ok = n_fail = n_skip = 0
    last_new = time.time()
    print(f"following {producer_meta.removesuffix('.meta.json')} (new lines every {FOLLOW_POLL_SECONDS}s)", flush=True)
    while True:
        finished = _producer_finished(producer_meta, since)  # check before loading: no line is missed
        new = [it for it in load() if it[key] not in attempted]
        if new:
            ok, fail, skip = await _run(new, worker, output, key, clients, n_loops, set(), len(new))
            attempted.update(it[key] for it in new)
            n_ok, n_fail, n_skip = n_ok + ok, n_fail + fail, n_skip + skip
            last_new = time.time()
        elif finished:
            print("the generation run finished and all its lines are processed", flush=True)
            return n_ok, n_fail, n_skip, True
        elif time.time() - last_new > timeout_min * 60:
            print(f"WARNING: nothing new for {timeout_min:g} min and {producer_meta} shows no finished run; "
                  f"stopping. Rerun to process the rest.", flush=True)
            return n_ok, n_fail, n_skip, False
        await asyncio.sleep(FOLLOW_POLL_SECONDS)


def _done_keys(output, key):
    """Keys of the items already in `output`. A last line cut off by a killed job is removed first
    (its item is then redone)."""
    done = set()
    if not output.exists():
        return done
    good = 0  # bytes of complete lines
    with open(output, "rb+") as f:
        for n, line in enumerate(f, 1):
            try:
                rec = json.loads(line) if line.endswith(b"\n") else None
            except ValueError:
                rec = None
            if rec is None:
                if f.read(1):
                    sys.exit(f"{output}: line {n} is broken and not the last line; fix or remove it by hand")
                print(f"WARNING: {output}: removing the cut-off last line {n} (from a killed run)", flush=True)
                f.truncate(good)
                break
            done.add(rec[key])
            good += len(line)
    return done


async def _run(items, worker, output, key, clients, n_loops, done, total=None):
    """Process the items not in `done` with `n_loops` concurrent loops, each taking the next item from
    the shared iterator, and append the results to `output`. The loops are spread evenly over `clients`."""
    print(f"{total if total is not None else '?'} items, {len(done)} already in {output}, "
          f"{n_loops} requests at a time", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = iter(items)
    n_ok = n_fail = n_done = n_skip = 0  # n_done: already in output; n_skip: worker returned None
    start = last_report = time.time()

    with open(output, "a", encoding="utf-8") as out:
        async def loop(client):
            nonlocal n_ok, n_fail, n_done, n_skip, last_report
            for item in pending:  # shared: every item is taken by exactly one loop
                if item[key] in done:
                    n_done += 1
                    continue
                try:
                    records = await worker(item, client)
                except Exception as e:
                    n_fail += 1
                    print(f"FAILED {key}={item[key]}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                    continue
                if records is None:
                    n_skip += 1
                    continue
                for r in records:
                    out.write(json.dumps(r, ensure_ascii=False) + "\n")
                out.flush()
                n_ok += 1
                if time.time() - last_report >= PROGRESS_SECONDS:
                    last_report = time.time()
                    rate = (n_ok + n_fail + n_skip) / (last_report - start)
                    n_left = None if total is None else total - n_ok - n_fail - n_skip - n_done
                    left = "" if total is None else f", {n_left} left, ETA {n_left / 60 / rate:.0f} min"
                    print(f"{n_ok} done, {n_fail} failed, {n_skip} skipped, {rate:.2f} items/s{left}", flush=True)

        await asyncio.gather(*(loop(clients[i % len(clients)]) for i in range(n_loops)))

    print(f"finished: {n_ok} ok, {n_fail} failed, {n_skip} skipped, {n_done} already done, "
          f"{time.time() - start:.0f}s", flush=True)
    return n_ok, n_fail, n_skip
