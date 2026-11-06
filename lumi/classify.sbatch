#!/bin/bash
#SBATCH --job-name=rollgen-classify
#SBATCH --account=project_465003062
#SBATCH --partition=dev-g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=0                       # important on dev-g, OOM otherwise
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#
# Label the records of a dataset (e.g. difficulty of each prompt) with a classifier model: starts
# the model in vLLM on every node of the allocation, then runs rollgen/classify.py once (on the
# first node) through a vllm-router in front of all servers.
#
# Once per clone, on a login node:  bash lumi/install_router.sh
#
# Submit from the repo root:
#   sbatch --nodes=2 lumi/classify.sbatch --model /scratch/.../gemma-4-12B-it \
#       --input data/so_openq.jsonl --output data/so_openq.difficulty.jsonl \
#       --classifier classifiers/difficulty --keep-fields uuid \
#       --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}'
#
# Arguments. Slurm options go before the script name, all other flags after it.
#   sbatch options (common ones):
#     --nodes=N                 vLLM servers start on every node (8 / --tp servers per node)
#     --time=HH:MM:SS           time limit; a timed-out generate/judge job can be resubmitted (it resumes)
#     -p, --partition=NAME      dev-g (testing, ONE job per project), small-g, standard-g
#     --dependency=after:JOBID  start once JOBID has started (afterok:JOBID = once it finished OK)
#     --job-name=NAME           shows up in squeue and in the logs/ folder name
#   Server flags (read by the launcher in lumi/_common.sh, not passed to Python):
#     --model PATH|HUB_ID       (required) model dir, or HF hub id already downloaded to $HF_HOME,
#                               e.g. Qwen/Qwen3-32B (compute nodes are offline)
#     --served-name NAME        model name used in API requests (default: basename of --model)
#     --tp N                    tensor parallel size per vLLM server, must divide 8 (default: 1)
#     --vllm-args "ARGS"        extra `vllm serve` flags as ONE quoted string. The most useful ones:
#                                 --max-model-len N           context length (prompt + output); smaller
#                                                             leaves more memory for KV cache
#                                 --gpu-memory-utilization F  share of GPU memory vLLM takes (vLLM default 0.9)
#                                 --max-num-seqs N            max sequences one server runs at once
#                                 --enforce-eager             no torch.compile / graphs: faster startup,
#                                                             slower generation (good for quick tests)
#                                 --dtype bfloat16            weight / activation dtype
#                                 --seed N                    sampling seed
#                                 full list: `vllm serve --help` in the container
#     --no-stage-model          load weights from Lustre instead of copying them to node-local /tmp first
#     --startup-timeout SEC     max wait for all servers to answer /health (default: 1800)
#     --compile-cache DIR       keep compile caches in DIR (on /scratch, never $HOME) so later jobs start
#                               faster (default: node-local /tmp, recompiled every job)
#     --api-key-file FILE       servers require the key in FILE (one line, chmod 600); without it anyone on
#                               the LUMI network can use them while the job runs
#     --router-policy NAME      how vllm-router spreads requests over the servers: round_robin (default),
#                               cache_aware, random, power_of_two, consistent_hash
#   rollgen/classify.py flags (passed through):
#     --input FILE              (required) records .jsonl/.parquet (streamed: many GB is fine)
#     --output FILE             (required) labels .jsonl, one line per record (record_idx + kept fields +
#                               labels + classify_error); an existing file is resumed, same settings only
#                               (--concurrency may differ; an edited classifier counts as different)
#     --classifier DIR          (required) folder with prompt.txt/.json + rubric.toml (+ system.txt),
#                               e.g. classifiers/difficulty
#     --keep-fields A,B         input fields copied to every output line, e.g. uuid (to join on)
#     --temperature F           (default: 0.0)
#     --max-tokens N            max tokens of the reply (default: 512)
#     --extra-body JSON         extra sampling params for vLLM; for Qwen3/Gemma:
#                               '{"chat_template_kwargs": {"enable_thinking": false}}'
#     --concurrency N           max in-flight requests per server (default: 64; in total x servers)
#     (--url and --servers are filled in by the launcher: the router and the number of servers)
#
# All flags:  bash lumi/classify.sbatch --help     (works on a login node)
# Logs:       logs/latest_run/ (slurm.out, run.txt, vLLM logs); run record also in <output>.meta.json
set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
[ -f "$REPO/lumi/_common.sh" ] || { echo "FATAL: submit from the repo root (sbatch lumi/classify.sbatch ...)" >&2; exit 1; }
source "$REPO/lumi/_common.sh"

init_run_dir
parse_launcher_args rollgen/classify.py "$@"
record_run lumi/classify.sbatch "$@"
start_vllm_servers
start_router
run_in_container python "$REPO/rollgen/classify.py" "${CLIENT_ARGS[@]}"
