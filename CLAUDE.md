# rl-rollgen

A collection of scripts for generating RL rollouts with self-hosted vLLM and preparing them for training.
See `docs/README.md` for the goal statement and priorities. **`README.md` is written only by the owner: never edit it; put docs in `docs/`.**

## Priorities (from the owner)
1. **No overengineering.** A neat collection of documented scripts beats a complex framework.
2. **Reuse existing tools** (vLLM, openai client, HF transformers/datasets, Harbor) instead of hand-rolling.
3. **Clarity.** Clear script names and an obvious repo structure. The owner writes most of the docs.
4. **Easy install and use.** Ideally: clone the repo and run the scripts, nothing more.

## Environments
- **LUMI**: HPC, AMD GPUs, Slurm. Uses the latest **LAIFS container** with **vLLM 0.26**.
  Scripts run inside the container, so they must not need extra packages beyond what the container ships.
- **Tilde servers**: NVIDIA GPUs, **no Slurm**. Launch processes directly (bash).

## Decisions
- **Inference**: self-hosted vLLM server (OpenAI-compatible API). Generation and judging both talk to it over HTTP.
- **LLM-as-a-judge**: also a self-hosted vLLM server, not an external API, run as a **separate Slurm job**.
- **Multi-node is the normal case.** One job = vLLM servers on every node (one srun step) + a vllm-router on the
  first node in front of all of them + one client process there that talks only to the router (decided 2026-11-06:
  balancing moved out of rollgen/). No Ray, no NCCL between nodes: servers are independent.
- **Output target**: rollouts end up in **SkyRL** for training. The owner assumes a verl-style format; confirm the exact
  schema against SkyRL before finalizing the tokenizer/export script.

## LUMI facts (from the owner's SkyRL training launcher, shared once; no longer in the repo)
That script was the owner's working SkyRL training job on LUMI; the facts below were taken from it.
- Container: the **plus** variant of LAIFS `lumi-multitorch-u24r72f21m50t211-20261028_090706` (vLLM 0.26):
  `/appl/local/laifs/containers/lumi-multitorch-u24r72f21m50t211-20261028_090706/lumi-multitorch-plus-u24r72f21m50t211-20261028_090706.sif`
  (other variants in that dir: rocm, torch, mpich, libfabric, full). The SkyRL job uses its own sandbox
  (`$ROOT/multitorch_sandbox.sif`, venv at `/opt/venv`).
  The full image (and so plus) ships `vllm` 0.26.0 and `openai` 3.40.0
  (release notes: github.com/lumi-ai-factory/laifs-container-recipes/releases).
  Since 20260513 the runtime env is set by an **OCI entrypoint**, so use **`singularity run`**, not `exec`
  (`exec` skips the entrypoint).
- dev-g QOS `dev_g`: MaxSubmitPA=1, MaxSubmitPU=1, MaxJobsPU=1 (checked 2026-11-05). Only ONE job per project
  on dev-g, pending jobs included: no `--dependency` chains and no generate + `judge --follow` there; put the
  second job on small-g/standard-g, or run them one after the other.
- Slurm: account `project_465003062`, partition `dev-g` for testing (easy to change), 8 GCDs per node,
  `--cpus-per-task=56`, `--exclusive`, **`--mem=0` (required on dev-g, OOM otherwise; also pass it to srun)**.
- Paths: `ROOT=/scratch/project_465003062/environment/containers/fsdp-skyrl-harbor`, `HF_HOME=$ROOT/hf`,
  binds `/var/spool/slurmd,/scratch/project_465003062,/scratch/project_465002038`.
- Env lessons to reuse:
  - `HF_HUB_OFFLINE=1`: compute nodes have no internet, so pre-download models.
  - Slurm sets `ROCR_VISIBLE_DEVICES`. Export `HIP_VISIBLE_DEVICES` from it and unset ROCR/CUDA variants (Ray needs this).
  - **Never let caches default to `~/.cache`**: the home quota is 100K files and vLLM's torch.compile cache
    blew it on the first real run ("Disk quota exceeded"). All caches (VLLM_CACHE_ROOT, TORCHINDUCTOR_CACHE_DIR,
    TRITON_CACHE_DIR, MIOpen, XDG_CACHE_HOME) and the staged model live in `$NODE_TMP=/tmp/$USER-rollgen-$SLURM_JOB_ID`
    (node-local RAM), removed when the job ends. Cost: torch.compile runs from scratch every job.
    Optional `--compile-cache DIR` (e.g. on /scratch, never $HOME) keeps vLLM/Inductor/Triton/MIOpen caches across
    jobs; it's bind-mounted into the container. Concurrent writes from many servers into one shared cache
    are untested on LUMI.
  - Stage model weights to `/tmp` before loading (much faster than mmap from Lustre).
  - `PYTHONNOUSERSITE=1`, `VLLM_WORKER_MULTIPROC_METHOD=spawn`.
  - Use the `hsn0` (Slingshot) IP for cross-node traffic. The container has **no `ip` command** (checked
    2026-11-05), so `_vllm_node.sh` reads the address with a Python ioctl. Compute nodes have nmn0 (10.120.x,
    management) and hsn0-3 (10.253.x, Slingshot).
  - Login node `uan01` reaches a compute node's port 8000 by hsn0 IP and by hostname (checked 2026-11-05, dev-g).
    No direct ssh into compute nodes on LUMI. Networking details for Tilde: see "Harbor on Tilde" below.
  - LUMI login-node system `python3` is too old / lacks openai/pydantic (judge/classifier configs, e.g. --check-args, fail there). Run Python tools in the
    container (`singularity run -B /scratch/project_465003062 $SIF python ...`). Offered, not yet built:
    `lumi/run.sh <cmd>` wrapper around `run_in_container` for login-node use.
  - The RCCL/libfabric tuning is only needed for multi-node NCCL (training weight sync).
    Rollout generation with independent per-node vLLM servers shouldn't need it.
- `--model` accepts a dir or an HF hub id; hub ids resolve to `$HF_HOME/hub/models--Org--Name/snapshots/<refs/main>`
  on the host (fail early if missing), then are staged with `cp -L` (snapshots are symlinks into blobs/).
- Style the owner likes: `: "${VAR:=default}"` config overridable from the submitting shell, a config section at the top,
  `die` checks with clear messages, archiving the script and config per run.

## Install / usage model
- **Clone and run.** No package install, no `src/` layout, no console entry points.
- LUMI: `singularity run <laifs.sif> python rollgen/<name>.py ...` (the sbatch wrappers do this for you).
  Dependencies must stay within what the LAIFS container already ships.
- Tilde: `uv run rollgen/<name>.py ...` (uv builds the env from `pyproject.toml`; pin vLLM 0.26 to match LUMI).

## Repo structure
```
rollgen/           # "native" to both clusters: the actual work (generate, judge, later filter/tokenize).
  _common.py       #   JSONL/parquet I/O (streamed), one vLLM client for --url/$VLLM_URL (a server or a router),
                   #   resumable async loop (concurrency x --servers loops, cut-off last output line repaired),
                   #   meta.json, --check-args
  generate.py      #   prompts .jsonl/.parquet -> n samples per prompt -> rollouts JSONL
                   #   (prompt: string | message list (SkyRL/verl) | "messages"); outputs stay JSONL (append/resume)
  judge.py         #   rollouts JSONL -> + judge_output, judge_scores, judge_error (--judge configs/judges/<name>)
  classify.py      #   records .jsonl/.parquet -> sidecar labels JSONL (record_idx, --keep-fields, one field per
                   #   Output field, classify_error) (--classifier configs/classifiers/<name>); dedups identical requests
  _ask.py          #   shared engine for judges + classifiers: loads <folder>/config.py (Output + messages()),
                   #   JSON schema from Output, validates replies; helpers truncate/last_user_message
lumi/              # LUMI-specific glue: Slurm, container, starting vLLM
  _common.sh       #   site config, parse_launcher_args, record_run, start_vllm_servers, start_router,
                   #   run_in_container (host side); apply_env, node_addr (container side)
  _vllm_node.sh    #   inside container, one per node: stage model to /tmp, start GPUS/TP vLLM servers,
                   #   wait for health, write URLs to <run dir>/urls/<node>.txt, serve until killed
  generate.sbatch  #   policy model + rollgen/generate.py
  judge.sbatch     #   judge model + rollgen/judge.py (separate job, chain with --dependency=afterok)
  classify.sbatch  #   classifier model + rollgen/classify.py
  serve.sbatch     #   servers + vllm-router (one URL), no client, held until walltime/scancel (Harbor on Tilde)
  _router.sh       #   inside container, first node: vllm-router (cache_aware) over all servers, self-test
                   #   (completion with key = 200, without = 401), write <run dir>/urls/router.txt
  install_router.sh #  one-time per clone, login node (every job needs it): pip --no-deps --target
                   #   lumi/vllm-router/ (gitignored), pinned
tilde/             # Tilde-specific glue (plain bash, no Slurm)
  install.sh       #   one-time: uv + `uv tool install harbor==<pin>` (Docker assumed present)
  download_dataset.sh #   Harbor registry dataset -> data/harbor/<name>@<version>/<task>/
  harbor_run.sh    #   one command: wait for a LUMI serve job's router (ssh lookup of <run>/urls/router.txt),
                   #   ssh -L tunnel to it, `harbor run -p data/harbor/...` with local Docker sandboxes,
                   #   close tunnel on exit -> logs/harbor/<job>/ (+ run.txt); --resume. Walkthrough: docs/harbor_tilde.md
configs/           # what users edit per experiment (user content, not code). Flags take short names:
                   #   --judge gsm8k = configs/judges/gsm8k (rollgen/_common.py resolve_config; a path works too)
  judges/          #   one folder per judge: config.py (example = annotated template; correctness, gsm8k, tool_use)
  classifiers/     #   one folder per classifier: config.py (example = annotated template; difficulty/: from the owner's
                   #   difficulty_classifiers.py, gitignored at the repo root)
  examples/        #   tiny example inputs for smoke tests
  runs/            #   per-experiment launch scripts (bash configs/runs/<name>.sh)
docs/              # all docs (index: docs/README.md). Top-level README.md is owner-written only
logs/              # Slurm and vLLM logs (gitignored)
```
Top level is kept to lumi/, tilde/, rollgen/, configs/, docs/ (owner's wish, 2026-11-06); new user content goes
under configs/. Moving a config folder doesn't break resuming: the judge/classifier paths are
resume-safe args (their file contents are compared instead).
Rule: `rollgen/` holds the real logic and runs unchanged on both clusters; it only knows vLLM URLs.
Anything cluster-specific (Slurm, containers, GPU env vars, paths) lives in `lumi/` or `tilde/`.
Naming: files users run have plain names; internals have a `_` prefix and live next to what uses them
(`lumi/_common.sh`, `rollgen/_common.py`). Keep it that way when adding files.
Sbatch files must be submitted from the repo root (they use `$SLURM_SUBMIT_DIR`).

## Interface conventions
- **Flags only.** Every setting is a command-line flag; no env vars in normal use.
  `sbatch --nodes=4 lumi/generate.sbatch --model PATH --tp 2 --input ... --output ... --samples-per-prompt 8`
  The launcher consumes server flags (`--model`, `--served-name`, `--tp`, `--vllm-args`, `--no-stage-model`,
  `--startup-timeout`, `--compile-cache`, `--api-key-file`, `--router-policy`) and passes everything else to the Python script
  (serve.sbatch has no script and rejects other flags). Slurm things (nodes, time, partition)
  stay regular sbatch options. Site config (SIF, BINDS, HF_HOME) is env-overridable but not part of the interface.
- `bash lumi/<entry>.sbatch --help` works on a login node: launcher flags plus the Python script's --help.
- **Same flags on both clusters** (tilde/ launchers must mirror lumi/).
- **Fail before vLLM starts:** launchers run the Python script with the hidden `--check-args` first
  (argparse typos, missing input, bad judge config / messages() failing on record 0, resume-with-different-settings).
- **Every run is recorded** (the owner's main pain was tracking changes between runs):
  - `<output>.meta.json` (written by rollgen/, so cluster-agnostic): list of runs with settings, model path,
    git commit + dirty flag, host, job id, timing, ok/failed counts.
  - `logs/<date>-<jobname>-<jobid>/` per job: `slurm.out` (moved there by init_run_dir at job start; Slurm keeps
    writing to the moved file), `run.txt` (full command, nodes, model/TP, container, git commit/status/diff),
    `<node>-vllm-<i>.log`, `urls/` (only while the servers run; removed on exit so a dead job never looks alive).
    `logs/latest_run` is a symlink to the newest run folder.
  - Resuming into an existing output with different settings is refused (only `url`, `servers`, `concurrency` may differ).
- Per-experiment run files live in `configs/runs/<name>.sh`: plain bash holding the sbatch command(s), `cd` to the repo
  root first, run with `bash configs/runs/<name>.sh`. No special support; what actually ran is in `logs/<run>/run.txt`.
  If colleagues want more, revisit "one run file per experiment" (variant B).

## Judges and classifiers (config.py format decided 2026-11-06, replaced prompt.txt/rubric.toml)
- A judge/classifier = a folder `configs/{judges,classifiers}/<name>/` with `config.py` (system prompt included
  as a string; owner rejected jinja templates, 2026-11-06: no extra dependency). config.py defines `class Output(pydantic.BaseModel)` (the answer fields; its JSON schema
  goes to vLLM structured outputs via `response_format` json_schema, the reply is validated with it) and
  `messages(record)` -> chat messages | str (one user message) | None (skip: no output line, `items_skipped`
  in meta.json; re-evaluated on rerun, cheap). Exceptions -> error line; on record 0 -> stop (--check-args).
- Why (owner, 2026-11-06): rubric.toml mixed schema, question text, `when`/`[text]` conditionals and settings
  into a small DSL; `--input-format` Python was already the intuitive part. Now one way: plain Python.
  `configs/formats/` and `--input-format` are gone; dataset shapes are handled inside messages()
  (difficulty reads plain/Nemotron-RL/chat-messages records itself, so tools can no longer be silently missing).
- Deps: only pydantic, which comes with openai (already needed). `from _ask import truncate, last_user_message`
  in configs (rollgen/ is on sys.path when the scripts run).
- The judge only answers questions (no arithmetic, no reward: removed 2026-11-06, a colleague said the computed
  reward isn't what we want). Output: `judge_scores` (all Output fields, incl. reasoning if declared) +
  `judge_output` (raw reply) + `judge_error`. Reward is up to the training side.
- Every file of the folder is stored in meta.json settings as `<judge|classifier>_file <name>`, so resuming
  with an edited config is refused (the message names the changed file). Outputs started with the old
  rubric format can't be resumed with the new one (settings keys differ).
- difficulty: SYSTEM (in config.py) is byte-identical to the old system.txt with `{tools_note}` filled
  (checked 2026-11-06). Requests match difficulty_classifiers.py no longer (system text + schema since the tools
  change; old folder in git, commit 4a62bbb). Pydantic's schema adds `title` keys (harmless).
  The tools sentence is factual only: how tools should weigh on difficulty is not specified (owner's call).
  Owner doesn't want `tools_needed` graded (knows it from the data).
- `judge.py --follow`: judge job runs alongside generation (`--dependency=after:`), re-reads the growing rollouts
  file every 30s (ignores an unfinished last line; rollout_idx = line number, stable because generate only appends),
  stops when the newest run in `<input>.meta.json` finished after the judge started, or after `--follow-timeout`
  minutes without new lines (then exit 1, run marked `incomplete`). Failed items aren't retried within one run.
- classify.py: sidecar lines in completion order with `record_idx`, `classify_error`, failed server requests
  are retried on rerun, thinking off via `--extra-body`. Dedups identical requests (sha1 of messages; only
  after a reply, so identical requests in flight at the same time are both sent; errors aren't cached).
- Multi-file classify (decided 2026-11-06: owner rejected one job per file because every job pays the vLLM
  startup): `--input A B ... --output-suffix .difficulty.jsonl` -> outputs next to inputs, one job. All files
  checked first (record 0, keep fields, start_meta dry run), then run_resumable per file in sequence (short
  concurrency drain between files, accepted). Per-file Namespace (input/output = that file, no output_suffix)
  so meta.json settings equal a single-file run's (either form resumes the other; adding files to the list
  doesn't block resuming). Dedup cache shared across files. run_resumable now returns ok instead of
  sys.exit (callers exit 1). Only classify.py has this; judge.py doesn't (yet).
- Client throughput (measured with the old engine): ~450 req/s per process, CPU-bound in the openai client;
  memory flat on large inputs (672 MB input -> 133 MB RSS).
- Tested locally (2026-11-06) with a fake vLLM server: classify on plain/Nemotron/chat records, dedup, error
  lines, resume, judge gsm8k, skip via None, edited-config refusal. Not yet run on LUMI with a real model.
- No tilde/ launcher yet (no one-step Tilde launchers exist); there: `uvx --from vllm-router==0.1.16 vllm-router`
  in front of the servers, then classify.py --url <router> --servers N (docs/classifiers.md; tested locally).

## Current scope
- Focus: **one-step RL rollouts + LLM judge, on LUMI first**. Tilde comes after.
- **Harbor (multi-step)**: see "Harbor on Tilde" below.
- **SkyRL output format is deferred.** Don't design the export/tokenize step yet.

## Harbor on Tilde (status 2026-11-05)
Topology: Harbor orchestrator + Docker sandboxes on **Tilde**, vLLM on **LUMI**; only the vLLM HTTP port crosses
sites: `Tilde 127.0.0.1:18000 --ssh -L--> lumi-uan01.csc.fi --TCP--> <first node hsn0 IP>:9000 (vllm-router)`.
Rejected: `DOCKER_HOST=ssh://tilde` (Harbor bind-mounts trial log dirs, which would resolve on Tilde).

Facts established:
- Login node ssh names are `lumi-uan01.csc.fi` .. `lumi-uan04.csc.fi` (`uan01.lumi.csc.fi` does NOT resolve).
  `lumi.csc.fi` round-robins, so tunnels pin `lumi-uan01.csc.fi` (harbor_run.sh `--lumi-host` default).
- LUMI side verified: uan01 -> compute node hsn0 IP / hostname port 8000 works (curl 200 from a nettest job).
- Tilde (host `perkons`, repo at `/local_data/rusevich/rl-rollgen`, LUMI user `pinnismr`): outbound ssh to LUMI
  works, key registered, `ssh -o BatchMode=yes pinnismr@lumi-uan01.csc.fi hostname` works once the key is in an
  ssh-agent and the host key for `lumi-uan01.csc.fi` was accepted once interactively.
- The key has a **passphrase** -> use ssh-agent. Since harbor_run.sh is one command in one tmux window, the docs
  (docs/harbor_tilde.md 3a) start the agent inside that window (`eval "$(ssh-agent)"; ssh-add -t 12h <key>`);
  once per window, re-`ssh-add` after expiry (open tunnels survive it). Docs suggest `~/.ssh/config`
  `Host lumi-uan01.csc.fi / User / IdentityFile` so `--lumi-host` can stay default. A backgrounded `ssh ... &` that needs the
  passphrase gets "Stopped" (SIGTTIN) and the port never opens. harbor_run.sh uses `ssh -o BatchMode=yes`
  (fails fast with a hint instead) and checks `ssh <host> true` first.
- Tilde has Docker and uv. Harbor: `bash tilde/install.sh`. No CUDA/vLLM needed
  on Tilde. Watch /var/lib/docker disk space; task image builds need internet from containers.
- **NOT YET VERIFIED: the Tilde -> uan01 port forward itself** (was mid-test: tunnel in tmux window 0, then
  `curl 127.0.0.1:18000/` -> expect 200; "administratively prohibited" = uan01 forbids forwarding -> plan B).
  Also unverified: vLLM 0.26 enforcing the key from the VLLM_API_KEY env (expect 401 without key).

Runbook for the first end-to-end run: **`docs/harbor_tilde.md`** (install, download, serve, run (incl. tunnel),
resume, results, cleanup). The tunnel test is now just the first real harbor_run.sh. Commit/push first; `git pull` on LUMI
(`/scratch/project_465003062/FAKE_MARCIS/rl-rollgen`) and Tilde.

Harbor layout (decided 2026-11-05; facts from harbor 0.20.0 source):
- Tasks: `tilde/download_dataset.sh <name>@<ver>` -> `data/harbor/<name>@<ver>/<task>/` (wraps `harbor datasets download`,
  whose export mode would drop the version). Runs use `-p` on that folder; **`harbor run -d` is not used**
  (it fetches into the content-addressed `~/.cache/harbor`, hardcoded, hard to browse). harbor_run.sh rejects -d/-t.
- Jobs: `logs/harbor/<date>-<time>-<dataset>-<model>/` (Harbor's config.json, result.json, job.log, trial dirs
  `<task>__<id>/` with agent/, verifier/, result.json) + our `run.txt` (appended per (re)start); `logs/harbor/latest`.
  Harbor ignores plain files in the job dir; same job name + same config = continue (cancelled trials kept as
  failed), different config = error. `harbor jobs resume -p <job>` drops CancelledError trials and reruns them
  (needs HOSTED_VLLM_API_KEY in env). Ctrl-C/SIGTERM cancel trials, recover outputs, tear down containers.
- harbor_run.sh (decided 2026-11-06: attach to a serve job, never submits/scancels it; LUMI config by flags, put
  them in a configs/runs/<name>.sh): `-p`, `--lumi-host [user@]host` (default lumi-uan01.csc.fi), `--lumi-repo` (required
  unless `--url`), `--lumi-run JOBID|RUN` (default latest_run; job id works while queued: run dir = logs/*-<jobid>),
  `--wait MIN` (60), `--local-port` (18000), `--url` (no LUMI/tunnel), `--api-key-file` (default ~/.vllm_key,
  required), `--agent`, `-n`, `--max-output-tokens`, `--max-input-tokens` (default ctx - max output), `--job-name`,
  `--resume NAME`; rest goes to `harbor run` / `harbor jobs resume`. Requires `harbor` on PATH (install.sh: 0.20.0).
  - Lookup: one ssh per 30s poll running a heredoc script on the login node: prints run folder, router URL,
    `squeue -j` state (GONE = not in queue -> die if no router). latest_run = newest *started* job of any kind.
  - Tunnel: `setsid ssh -N -o ControlPath=none -L 127.0.0.1:PORT:<router>` in the background (ControlPath=none:
    the owner's ~/.ssh/config has `Host lumi` (lumi.csc.fi) with ControlMaster auto + ControlPersist 10m; through
    a shared master the forward would outlive our ssh and keep the port taken. Lookups may share the master.
    The preflight `ssh true` also uses ControlPath=none: through a live master it passed without the key in
    ssh-agent, then the tunnel failed "Permission denied (publickey)" (seen on Tilde 2026-11-06). That first real
    run also showed: lookup over ssh + router URL from a real serve job (23510754) work.
    `--lumi-host lumi` works with that alias; any login node reaches compute nodes, round-robin is fine) (setsid: Ctrl-C reaches Harbor only, tunnel
    stays up while Harbor cancels trials), ExitOnForwardFailure + ServerAlive; EXIT trap kills it. Harbor runs in
    the foreground (no exec) so the trap fires; the script exits with Harbor's code.
  - --resume: model + api_base from `config.json` (`agents[0].model_name`, `.kwargs.api_base`); a 127.0.0.1:PORT
    api_base -> tunnel reopened on that port; refuses run-setting flags and a server with a different model.
  - Tested locally (2026-11-06) with stub ssh (local forwarder), squeue, docker, harbor + the real router over fake
    vLLM servers: PENDING -> RUNNING -> router wait, tunnel, key, run.txt, resume, port in use, ended job, wrong
    job id/run name, Ctrl-C (tunnel up during Harbor's SIGINT handling, closed after). Never run for real.
- tunnel.sh removed (2026-11-06); `logs/tunnel.urls`/`tunnel.run` are no longer read or written.

vllm-router (decided 2026-11-06): multi-server balancing happens in vllm-router 0.1.16 (`pip install vllm-router`,
fork of SGLang's model gateway; Rust core), started by start_router in **every** LUMI job on the first node, port 9000.
For Harbor that means balancing on LUMI, not Tilde. Policy: launcher flag `--router-policy`, default round_robin
(generate/judge/classify: independent one-step requests, predictable even spread), serve.sbatch sets
DEFAULT_ROUTER_POLICY=cache_aware (prefix-tree routing: an agent trial's turns stay on one server).
`--request-timeout-secs 3600` (= the openai client's timeout; router default 1800 could cut long generations).
The client gets VLLM_URL (router) + VLLM_SERVERS (count) from start_router; `--concurrency` stays per server and
rollgen keeps concurrency x servers requests in flight over one AsyncOpenAI client whose connection pool is sized
to that (openai's default pool: 1000 connections, 100 kept alive; openai 3.46 uses httpx2, older httpx).
Installed by lumi/install_router.sh into lumi/vllm-router/ with --no-deps (container has fastapi, uvicorn, aiohttp,
requests, setproctitle; orjson installed alongside) and run in the container with PYTHONPATH.
- Gotcha: vllm-router's `--api-key` is only the key it sends TO the workers; incoming requests are unchecked unless
  `api_key_validation_urls` is set (client's bearer token must get 200 from that URL), which the Rust core accepts
  but the Python CLI doesn't pass. _router.sh patches `router._Router` to pass both (key from the env, not argv),
  validating against the first worker's /v1/models. The self-test refuses to publish the URL if a keyless request
  isn't rejected. With validation on, every endpoint (also /health) needs the key.
- Router `/v1/models` passes vLLM's response through (incl. max_model_len), which harbor_run.sh relies on.
- Request bodies (checked 2026-11-06 with a body-logging fake server): every field arrives unchanged, unknown ones
  too (response_format, chat_template_kwargs, n, top_k, min_p, ...). The router ADDS its typed-struct defaults:
  continue_final_message/ignore_eos/logprobs/stream false, include_reasoning/skip_special_tokens true (= vLLM's
  defaults) and SGLang-only no_stop_trim, separate_reasoning, stream_reasoning, return_hidden_states (vLLM should
  ignore unknown fields, maybe with a warning per request: check vLLM logs on LUMI). Classify labels through the
  router == without it. round_robin split 60 requests 30/30; cache_aware split 3000 classify requests 1424/1366.
- Two routers on one host collide on the Prometheus port 29000 (router dies at startup). Can't happen on LUMI
  (--exclusive nodes, one router per job); matters only for local tests.
- Tested locally (2026-11-06) with two fake key-checking vLLM servers: auth 200/401 as expected; 8 concurrent
  distinct trials split 4/4 and each trial's next turn hit the same server. Sequential single requests all
  went to one server (load balancing kicks in under load). Not yet run on LUMI.
- Single point: if worker 0 (key validation target) dies, all requests fail auth. The router itself dying ends
  the job (serve.sbatch `wait -n`); in generate/judge/classify the client's requests then fail (rerun resumes).
- Known: `run_in_container ... &` in start_router backgrounds a shell function, so ROUTER_PID is a subshell and
  stop_vllm_servers' `kill` misses the router (seen in a local stub test 2026-11-06). Harmless on LUMI: the batch
  script exits right after and Slurm kills all of the job's processes. Fix if it ever matters (e.g. `exec` the
  singularity call in a subshell so $! is the container process).

Remaining in tilde/: a tunnel dying mid-run is not restarted (trials then fail; resume after). Next: measure vLLM
prefix-cache hit rate per server (`vllm:prefix_cache_*` in /metrics); later
`rollgen/harbor_export.py` -> rollouts JSONL (then configs/judges/tool_use applies).
Agents running inside the sandbox (not terminus-2) would need the tunnel bound to the docker bridge, not 127.0.0.1.
127.0.0.1 on Tilde is shared by all users there -> always use the API key.

## Considered and not adopted
- **NeMo Gym** (github.com/NVIDIA-NeMo/gym): covers rollout collection, verifiers and LLM judges,
  but needs Python >= 3.13, Ray and several local servers (Responses API).
  That conflicts with clone-and-run inside the LAIFS container. It's fine to borrow ideas from it,
  such as judge prompts and verifier logic.

## Open questions
- Exact SkyRL input schema (fields, token ids vs. text, loss masks). Deferred.
- Verified on LUMI (first dev-g run, 2026-11-05): `singularity run` works, vllm from the plus image starts,
  GPU split and model staging work. Not yet verified: cross-node HTTP to port 8000+, the srun step being
  stopped cleanly at the end, `serve.sbatch` on real hardware, the judge's JSON structured outputs with
  Qwen3-32B (check `judge_error` in judged output; pass `--extra-body '{"chat_template_kwargs":
  {"enable_thinking": false}}'`).
- Launch commands: `configs/runs/*.sh` (GSM8K: `configs/runs/gsm8k_tildeopen.sh`, judge on small-g). `docs/examples.md` (gitignored)
  still holds older notes; its GSM8K example puts both jobs on dev-g, which the dev_g QOS forbids.
