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
  - Use the `hsn0` (Slingshot) IP for cross-node traffic.
  - The RCCL/libfabric tuning is only needed for multi-node NCCL (training weight sync).
    Rollout generation with independent per-node vLLM servers shouldn't need it.
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
  generate.py      #   prompts JSONL -> n samples per prompt -> rollouts JSONL
  judge.py         #   rollouts JSONL -> + judge_output, judge_score (template + score regex)
  judge_prompts/   #   judge prompt templates ({prompt}, {completion}, {answer}, ...)
lumi/              # LUMI-specific glue: Slurm, container, starting vLLM
  _common.sh       #   site config, parse_launcher_args, record_run, start_vllm_servers,
                   #   run_in_container (host side); apply_env (container side)
  _vllm_node.sh    #   inside container, one per node: stage model to /tmp, start GPUS/TP vLLM servers,
                   #   wait for health, write URLs to logs/<job>/urls/<node>.txt, serve until killed
  generate.sbatch  #   policy model + rollgen/generate.py
  judge.sbatch     #   judge model + rollgen/judge.py (separate job, chain with --dependency=afterok)
tilde/             # Tilde-specific glue (not written yet)
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
  `--startup-timeout`) and passes everything else to the Python script. Slurm things (nodes, time, partition)
  stay regular sbatch options. Site config (SIF, BINDS, HF_HOME) is env-overridable but not part of the interface.
- `bash lumi/<entry>.sbatch --help` works on a login node: launcher flags plus the Python script's --help.
- **Same flags on both clusters** (tilde/ launchers must mirror lumi/).
- **Fail before vLLM starts:** launchers run the Python script with the hidden `--check-args` first
  (argparse typos, missing input, bad judge template, resume-with-different-settings).
- **Every run is recorded** (the owner's main pain was tracking changes between runs):
  - `<output>.meta.json` (written by rollgen/, so cluster-agnostic): list of runs with settings, model path,
    git commit + dirty flag, host, job id, timing, ok/failed counts.
  - `logs/<job>/run.txt` (LUMI launcher): full command, nodes, model/TP, container, git commit/status/diff.
  - Resuming into an existing output with different settings is refused (only `urls`, `concurrency` may differ).
- Optional per-experiment run files are fine as plain shell scripts holding the sbatch command; no special support.
  If colleagues want more, revisit "one run file per experiment" (variant B).

## Current scope
- Focus: **one-step RL rollouts + LLM judge, on LUMI first**. Tilde comes after.
- **Harbor (multi-step) is deferred.** Don't build it until the owner brings it up.
- **SkyRL output format is deferred.** Don't design the export/tokenize step yet.

## Considered and not adopted
- **NeMo Gym** (github.com/NVIDIA-NeMo/gym): covers rollout collection, verifiers and LLM judges,
  but needs Python >= 3.13, Ray and several local servers (Responses API).
  That conflicts with clone-and-run inside the LAIFS container. It's fine to borrow ideas from it,
  such as judge prompts and verifier logic.

## Open questions
- Exact SkyRL input schema (fields, token ids vs. text, loss masks). Deferred.
- Verified on LUMI (first dev-g run, 2026-11-05): `singularity run` works, vllm from the plus image starts,
  GPU split and model staging work. Not yet verified:
  `ip` available in the container for the hsn0 address
  (falls back to hostname), cross-node HTTP to port 8000+, and the srun step being stopped cleanly at the end.
