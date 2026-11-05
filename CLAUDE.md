# rl-rollgen

A collection of scripts for generating RL rollouts with self-hosted vLLM and preparing them for training.
See `README.md` for the goal statement and priorities.

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
- **Multi-node is the normal case.** One job = vLLM servers on every node (one srun step) + one client process
  on the first node that load-balances over all servers. No Ray, no NCCL between nodes: servers are independent.
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
  - LUMI login-node system `python3` is too old for `tomllib` (rescore.py fails there). Run Python tools in the
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
  _common.py       #   JSONL I/O, vLLM clients from --urls/$VLLM_URLS, resumable async loop, meta.json, --check-args
  generate.py      #   prompts .jsonl/.parquet -> n samples per prompt -> rollouts JSONL
                   #   (prompt: string | message list (SkyRL/verl) | "messages"); outputs stay JSONL (append/resume)
  judge.py         #   rollouts JSONL -> + judge_output, judge_scores, reward, judge_error (--judge judges/<name>)
  rescore.py       #   recompute `reward` from stored judge_scores with an edited formula (no GPUs)
  _rubric.py       #   loads judges/<name>/ (prompt.txt + rubric.toml), JSON schema, reward formula
lumi/              # LUMI-specific glue: Slurm, container, starting vLLM
  _common.sh       #   site config, parse_launcher_args, record_run, start_vllm_servers,
                   #   run_in_container (host side); apply_env (container side)
  _vllm_node.sh    #   inside container, one per node: stage model to /tmp, start GPUS/TP vLLM servers,
                   #   wait for health, write URLs to <run dir>/urls/<node>.txt, serve until killed
  generate.sbatch  #   policy model + rollgen/generate.py
  judge.sbatch     #   judge model + rollgen/judge.py (separate job, chain with --dependency=afterok)
  serve.sbatch     #   servers only, no client, held until walltime/scancel (for Harbor on Tilde via tunnel)
tilde/             # Tilde-specific glue (plain bash, no Slurm)
  install.sh       #   one-time: uv + `uv tool install harbor==<pin>` (Docker assumed present)
  download_dataset.sh #   Harbor registry dataset -> data/harbor/<name>@<version>/<task>/
  tunnel.sh        #   ssh -L through a LUMI login node to a LUMI job's vLLM servers (reads <run>/urls/*.txt
                   #   on LUMI), writes local URLs to logs/tunnel.urls
  harbor_run.sh    #   `harbor run -p data/harbor/...` with local Docker sandboxes + hosted_vllm model at the
                   #   tunneled URL -> logs/harbor/<job>/ (+ run.txt). Walkthrough: HARBOR_EXAMPLE.md
judges/            # one folder per judge: prompt.txt ({prompt}, {completion}, {answer}, {criteria}) +
                   #   rubric.toml (reward formula + criteria); user content, not code
examples/          # tiny example inputs for smoke tests
logs/              # Slurm and vLLM logs (gitignored)
```
Rule: `rollgen/` holds the real logic and runs unchanged on both clusters; it only knows vLLM URLs.
Anything cluster-specific (Slurm, containers, GPU env vars, paths) lives in `lumi/` or `tilde/`.
Naming: files users run have plain names; internals have a `_` prefix and live next to what uses them
(`lumi/_common.sh`, `rollgen/_common.py`). Keep it that way when adding files.
Sbatch files must be submitted from the repo root (they use `$SLURM_SUBMIT_DIR`).

## Interface conventions
- **Flags only.** Every setting is a command-line flag; no env vars in normal use.
  `sbatch --nodes=4 lumi/generate.sbatch --model PATH --tp 2 --input ... --output ... --samples-per-prompt 8`
  The launcher consumes server flags (`--model`, `--served-name`, `--tp`, `--vllm-args`, `--no-stage-model`,
  `--startup-timeout`, `--compile-cache`, `--api-key-file`) and passes everything else to the Python script
  (serve.sbatch has no script and rejects other flags). Slurm things (nodes, time, partition)
  stay regular sbatch options. Site config (SIF, BINDS, HF_HOME) is env-overridable but not part of the interface.
- `bash lumi/<entry>.sbatch --help` works on a login node: launcher flags plus the Python script's --help.
- **Same flags on both clusters** (tilde/ launchers must mirror lumi/).
- **Fail before vLLM starts:** launchers run the Python script with the hidden `--check-args` first
  (argparse typos, missing input, bad judge prompt/rubric, resume-with-different-settings).
- **Every run is recorded** (the owner's main pain was tracking changes between runs):
  - `<output>.meta.json` (written by rollgen/, so cluster-agnostic): list of runs with settings, model path,
    git commit + dirty flag, host, job id, timing, ok/failed counts.
  - `logs/<date>-<jobname>-<jobid>/` per job: `slurm.out` (moved there by init_run_dir at job start; Slurm keeps
    writing to the moved file), `run.txt` (full command, nodes, model/TP, container, git commit/status/diff),
    `<node>-vllm-<i>.log`, `urls/` (only while the servers run; removed on exit so a dead job never looks alive).
    `logs/latest_run` is a symlink to the newest run folder.
  - Resuming into an existing output with different settings is refused (only `urls`, `concurrency` may differ).
- Optional per-experiment run files are fine as plain shell scripts holding the sbatch command; no special support.
  If colleagues want more, revisit "one run file per experiment" (variant B).

## Judging design (decided 2026-11-05)
- Scoring must be configurable outside the code: a judge = `judges/<name>/{prompt.txt,rubric.toml}`.
- The judge only answers rubric questions (bool, or int with min/max) as JSON; vLLM structured outputs enforce
  the schema (`response_format` json_schema, with a free-text `reasoning` field first). The judge never does arithmetic.
- `reward` is a one-line Python expression over criterion names, evaluated with no builtins except
  min/max/abs/round; names and attribute access are checked up front. Trusted-team tool, not a sandbox.
- Reweighting is free: `rescore.py` recomputes rewards from stored `judge_scores`. Changed questions need a re-judge.
- prompt.txt and rubric.toml contents are stored in meta.json settings, so resuming with an edited judge is refused.
- TOML (stdlib `tomllib`) because nothing extra is needed in the container. Gotcha: `reward` must be above `[criteria]`.
- Old `--template`/`--score-regex` interface removed (one way to judge).
- `judge.py --follow`: judge job runs alongside generation (`--dependency=after:`), re-reads the growing rollouts
  file every 30s (ignores an unfinished last line; rollout_idx = line number, stable because generate only appends),
  stops when the newest run in `<input>.meta.json` finished after the judge started, or after `--follow-timeout`
  minutes without new lines (then exit 1, run marked `incomplete`). Failed items aren't retried within one run.

## Current scope
- Focus: **one-step RL rollouts + LLM judge, on LUMI first**. Tilde comes after.
- **Harbor (multi-step)**: see "Harbor on Tilde" below.
- **SkyRL output format is deferred.** Don't design the export/tokenize step yet.

## Harbor on Tilde (status 2026-11-05)
Topology: Harbor orchestrator + Docker sandboxes on **Tilde**, vLLM on **LUMI**; only the vLLM HTTP port crosses
sites: `Tilde 127.0.0.1:18000 --ssh -L--> lumi-uan01.csc.fi --TCP--> <compute hsn0 IP>:8000`.
Rejected: `DOCKER_HOST=ssh://tilde` (Harbor bind-mounts trial log dirs, which would resolve on Tilde).

Facts established:
- Login node ssh names are `lumi-uan01.csc.fi` .. `lumi-uan04.csc.fi` (`uan01.lumi.csc.fi` does NOT resolve).
  `lumi.csc.fi` round-robins, so tunnels pin `lumi-uan01.csc.fi` (tunnel.sh default fixed accordingly).
- LUMI side verified: uan01 -> compute node hsn0 IP / hostname port 8000 works (curl 200 from a nettest job).
- Tilde (host `perkons`, repo at `/local_data/rusevich/rl-rollgen`, LUMI user `pinnismr`): outbound ssh to LUMI
  works, key registered, `ssh -o BatchMode=yes pinnismr@lumi-uan01.csc.fi hostname` works once the key is in an
  ssh-agent and the host key for `lumi-uan01.csc.fi` was accepted once interactively.
- The key has a **passphrase** -> use ssh-agent. Start the agent BEFORE `tmux new` so all tmux windows inherit
  it (`eval "$(ssh-agent)"; ssh-add -t 12h <key>; tmux new -s lumi`). A backgrounded `ssh ... &` that needs the
  passphrase gets "Stopped" (SIGTTIN) and the port never opens; use foreground in tmux or `ssh -f`.
- Tilde has Docker and uv. Harbor: `bash tilde/install.sh`. No CUDA/vLLM needed
  on Tilde. Watch /var/lib/docker disk space; task image builds need internet from containers.
- **NOT YET VERIFIED: the Tilde -> uan01 port forward itself** (was mid-test: tunnel in tmux window 0, then
  `curl 127.0.0.1:18000/` -> expect 200; "administratively prohibited" = uan01 forbids forwarding -> plan B).
  Also unverified: vLLM 0.26 enforcing the key from the VLLM_API_KEY env (expect 401 without key).

Runbook for the first end-to-end run (after the tunnel test passes): **`HARBOR_EXAMPLE.md`** (install,
download, serve, tunnel, run, results, cleanup). Commit/push first; `git pull` on LUMI
(`/scratch/project_465003062/FAKE_MARCIS/rl-rollgen`) and Tilde.

Harbor layout (decided 2026-11-05; facts from harbor 0.20.0 source):
- Tasks: `tilde/download_dataset.sh <name>@<ver>` -> `data/harbor/<name>@<ver>/<task>/` (wraps `harbor datasets download`,
  whose export mode would drop the version). Runs use `-p` on that folder; **`harbor run -d` is not used**
  (it fetches into the content-addressed `~/.cache/harbor`, hardcoded, hard to browse). harbor_run.sh rejects -d/-t.
- Jobs: `logs/harbor/<date>-<time>-<dataset>-<model>/` (Harbor's config.json, result.json, job.log, trial dirs
  `<task>__<id>/` with agent/, verifier/, result.json) + our `run.txt` (appended per (re)start); `logs/harbor/latest`.
  Harbor ignores plain files in the job dir; same job name + same config = resume, different config = error.
- harbor_run.sh: flags only (`-p`, `--url`, `--api-key-file`, `--agent`, `-n`, `--max-output-tokens`,
  `--max-input-tokens` default ctx - max output, `--job-name`); rest goes to `harbor run`. Requires `harbor`
  on PATH (tilde/install.sh pins 0.20.0). tunnel.sh writes `logs/tunnel.run` (LUMI run name) for run.txt.
- Tested locally with a fake /v1/models server, stub docker and `--print-config`; never run against real Docker/vLLM.

Remaining known issues in tilde/: tunnel.sh still uses env-var config (LUMI_USER, LUMI_REPO);
only the first URL is used; `2>/dev/null` on tunnel.sh's URL lookup hides ssh errors. Plan: one `harbor_run.sh` owning tunnel + balancer + harbor (trap cleanup); balancer = LiteLLM
proxy first, measure vLLM prefix-cache hit rate (agent loops want session affinity), then a session/prefix-aware
router if needed; later `rollgen/harbor_export.py` -> rollouts JSONL (then judges/tool_use applies).
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
- Personal launch commands live in `EXAMPLES.md` (gitignored). Its GSM8K example still puts both jobs on dev-g,
  which the dev_g QOS forbids; the judge should go to small-g.
