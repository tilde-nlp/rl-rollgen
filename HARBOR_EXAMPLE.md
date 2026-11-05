# Harbor example: agent rollouts on Tilde with a model served from LUMI

Harbor runs the agent loop and the Docker sandboxes on **Tilde**. The model is a vLLM server on **LUMI**.
They are connected by an ssh tunnel through a LUMI login node:
`Tilde 127.0.0.1:18000 --ssh--> lumi-uan01.csc.fi --> <compute node>:8000`.

Where things live (Tilde, inside the repo, both gitignored):

```
data/harbor/<dataset>@<version>/<task>/   # tasks, from tilde/download_dataset.sh (we never use `harbor run -d`)
logs/harbor/<job-name>/                   # one Harbor job: config.json, result.json, job.log, <trial>/, run.txt
logs/harbor/latest -> <job-name>          # newest job
```

Harbor also keeps Docker images (`hb__*`) and `~/.cache/harbor`. `harbor cache clean` removes both.

## 0. One-time setup

```bash
# Tilde
git clone <repo> && cd rl-rollgen
bash tilde/install.sh                       # uv + harbor 0.20.0; checks that docker works

# LUMI: an API key for the vLLM server (127.0.0.1 on Tilde is shared by all users there)
(umask 077; openssl rand -hex 24 > ~/.vllm_key)
# Tilde: copy the key over
scp YOUR_username@lumi-uan01.csc.fi:.vllm_key ~/.vllm_key && chmod 600 ~/.vllm_key
```

The ssh key to LUMI has a passphrase, so load it into an agent **before** starting tmux. That way every
tmux window can use it:

```bash
eval "$(ssh-agent)"; ssh-add -t 12h ~/.ssh/<lumi key>; tmux new -s lumi
```

## 1. Download a dataset (Tilde)

```bash
harbor datasets list --legacy               # what is available
tilde/download_dataset.sh terminal-bench@2.0        # -> data/harbor/terminal-bench@2.0/<task>/
ls data/harbor/terminal-bench@2.0/<task>    # instruction.md, task.toml, environment/, tests/, solution/
```

Always pin the version. To change tasks, edit them in place or copy the folder under a new name.

## 2. Serve the model (LUMI)

```bash
cd /scratch/project_465003062/FAKE_MARCIS/rl-rollgen && git pull
sbatch -p dev-g --nodes=1 --time=01:00:00 lumi/serve.sbatch --model Qwen/Qwen3-32B --tp 8 \
  --vllm-args "--max-model-len 32768" --api-key-file ~/.vllm_key
tail -f logs/latest_run/slurm.out           # wait for "serving until ...; run name for tilde/tunnel.sh: <run>"
```

`--tp 8` gives one server per node. Harbor talks to a single URL, so more servers would sit idle.

## 3. Open the tunnel (Tilde, tmux window 1)

```bash
export LUMI_USER=pinnismr LUMI_REPO=/scratch/project_465003062/FAKE_MARCIS/rl-rollgen
tilde/tunnel.sh <run>                       # or no argument = newest LUMI run; writes logs/tunnel.urls
# check from another window: expect the model list (and 401 without the key)
curl -H "Authorization: Bearer $(cat ~/.vllm_key)" http://127.0.0.1:18000/v1/models
```

## 4. Run Harbor (Tilde, tmux window 2)

```bash
tilde/harbor_run.sh -p data/harbor/terminal-bench@2.0 --api-key-file ~/.vllm_key -l 5 -k 4
```

- `-l 5`: the first 5 tasks. `-k 4`: 4 attempts per task, i.e. samples per prompt.
- `-i '<glob>'` / `-x '<glob>'`: include or exclude tasks by name.
- `-n 8`: concurrent trials, which is the number of Docker containers running at once. Watch CPU, RAM and `/var/lib/docker`.

The script reads the model name and context length from the server. It sets the agent's input budget to the
context length minus `--max-output-tokens` (default 4096). See `tilde/harbor_run.sh --help` for all flags.
Any other flag goes straight to `harbor run`.

If a job is interrupted, rerun the same command with `--job-name <job-name>` (same settings) to resume it.
Finished trials are kept.

## 5. Results (Tilde)

```bash
J=logs/harbor/latest
cat $J/run.txt                              # our record: command, tasks, server/model, LUMI run, git commit + diff
python3 -m json.tool $J/result.json | head -40   # Harbor's summary: trial counts, mean reward, pass@k, errors
# one line per trial: task, reward(s), error
python3 -c 'import json,glob
for f in sorted(glob.glob("'$J'/*/result.json")):
    r = json.load(open(f)); v = r.get("verifier_result") or {}; e = r.get("exception_info") or {}
    print(r["task_name"], r["trial_name"], v.get("rewards"), e.get("exception_type", ""))'
harbor view logs/harbor                     # web viewer for trajectories, http://127.0.0.1:8080
```

Inside one trial folder (`$J/<task>__<id>/`):
- `agent/`: the agent's logs and trajectory (`trajectory.json` for terminus-2)
- `verifier/`: test output and `reward.txt` / `reward.json`
- `result.json`, `config.json`: everything about that trial

To open the viewer from your laptop: `ssh -L 8080:127.0.0.1:8080 <tilde host>`, then browse to `localhost:8080`.

## 6. Clean up

```bash
# Tilde: Ctrl-C the tunnel (window 1)
# LUMI:
scancel <jobid>
```
