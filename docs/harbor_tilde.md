# Harbor example: agent rollouts on Tilde with a model served from LUMI

Harbor runs the agent loop and the Docker sandboxes on **Tilde**. The model is a vLLM server on **LUMI**.
They are connected by an ssh tunnel through a LUMI login node to a vllm-router, which spreads the requests over
all vLLM servers of the job:
`Tilde 127.0.0.1:18000 --ssh--> LUMI login node --> <first node>:9000 (router) --> <all nodes>:8000..`.
`tilde/harbor_run.sh` opens and closes the tunnel itself.

Where things live (Tilde, inside the repo, both gitignored):

```
data/harbor/<dataset>@<version>/<task>/   # tasks, from tilde/download_dataset.sh (we never use `harbor run -d`)
logs/harbor/<job-name>/                   # one Harbor job: config.json, result.json, job.log, <trial>/, run.txt
logs/harbor/latest -> <job-name>          # newest job
```

Harbor also keeps Docker images (`hb__*`) and `~/.cache/harbor`. `harbor cache clean` removes both.

## 0a. One-time setup on LUMI (login node)

```bash
# 1. clone the repo on /scratch (not in $HOME: the home quota is small)
cd /scratch/project_465003062/<you> && git clone <repo> && cd rl-rollgen

# 2. the router that puts all vLLM servers of a job behind one URL (once per clone)
bash lumi/install_router.sh                 # -> lumi/vllm-router/

# 3. an API key for the vLLM servers. Without it, anyone on the LUMI network could use them, and on
#    Tilde anyone on the machine could use the tunnel (127.0.0.1 is shared by all users there).
#    serve.sbatch reads it on LUMI, harbor_run.sh on Tilde: you copy it to Tilde in step 0b.6.
(umask 077; openssl rand -hex 24 > ~/.vllm_key)

# 4. the model, if it isn't in the shared HF cache yet (compute nodes have no internet)
ls /scratch/project_465003062/environment/containers/fsdp-skyrl-harbor/hf/hub   # models--Org--Name = already there
```

To download a model, see [lumi.md](lumi.md) (`--model` and the shared HF cache).

## 0b. One-time setup on Tilde

```bash
# 1. Docker must work without sudo (otherwise ask an admin to add you to the docker group)
docker ps

# 2. the repo and Harbor
git clone <repo> && cd rl-rollgen
bash tilde/install.sh                       # uv + harbor 0.20.0; checks that docker works
```

**3. An ssh key from Tilde to LUMI.** Skip this if you already have one on Tilde (e.g. `~/.ssh/id_rsa`) that LUMI
accepts.

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_lumi     # choose a passphrase
cat ~/.ssh/id_lumi.pub
```

Add the public key to your LUMI account the same way you added your laptop's key (LUMI docs: "Setting up SSH key
pair"). It can take a while before LUMI accepts it.

**4. A short name for LUMI.** Add this to `~/.ssh/config` on Tilde (use your key from step 3):

```
Host lumi
    HostName lumi.csc.fi
    User <lumi user>
    IdentityFile ~/.ssh/id_lumi
    IdentitiesOnly yes
```

All commands below use this name: `ssh lumi`, and `--lumi-host lumi` for `harbor_run.sh`. Extra lines like
`ControlMaster` are fine: `harbor_run.sh` opens its own connection for the tunnel anyway.

**5. Log in once by hand.** Answer `yes` to the host key question and type the passphrase. `harbor_run.sh`
can't answer either of them.

```bash
ssh lumi hostname                           # -> uan01 (or another login node)
```

**6. Copy the API key from LUMI to Tilde.** This is the key you created in step 0a.3. Both sides need the same
key: the servers on LUMI check it, and `harbor_run.sh` on Tilde sends it (it reads `~/.vllm_key` by default).

```bash
scp lumi:.vllm_key ~/.vllm_key && chmod 600 ~/.vllm_key
diff <(ssh lumi cat .vllm_key) ~/.vllm_key && echo "same key on both sides"
```

If you ever make a new key on LUMI, copy it again. With different keys, `harbor_run.sh` stops with
"`.../models` failed (... wrong API key?)".

## 1. Download a dataset (Tilde)

Takes seconds to minutes; no tmux or ssh-agent needed (doing it in the tmux window of step 3 is fine too).

```bash
harbor datasets list --legacy               # what is available
tilde/download_dataset.sh terminal-bench@2.0        # -> data/harbor/terminal-bench@2.0/<task>/
ls data/harbor/terminal-bench@2.0/<task>    # instruction.md, task.toml, environment/, tests/, solution/
```

Always pin the version. To change tasks, edit them in place or copy the folder under a new name.

## 2. Serve the model (LUMI)

On a LUMI login node. No tmux needed: `sbatch` returns at once and the job runs without you.

```bash
cd /scratch/project_465003062/<you>/rl-rollgen && git pull
sbatch -p dev-g --nodes=2 --time=01:00:00 lumi/serve.sbatch --model Qwen/Qwen3-32B --tp 4 \
  --vllm-args "--max-model-len 32768" --api-key-file ~/.vllm_key
# -> "Submitted batch job <jobid>". You don't have to wait for it: go on with step 3.
```

This starts 2 servers per node (`--tp 4`) on 2 nodes. A vllm-router on the first node puts all of them behind one
URL (`logs/<run>/urls/router.txt`). It keeps each trial on one server, so the server can reuse its prefix cache
across the trial's turns. Router log: `logs/<run>/router.log`.

## 3. Run Harbor (Tilde, in tmux)

### 3a. Open tmux and load your ssh key (every time you log in to Tilde)

Harbor jobs run for hours, so run them in **tmux**: the job keeps running when you disconnect.

`harbor_run.sh` logs in to LUMI over ssh several times, and it **can't type your key's passphrase**. So first put
the key into an **ssh-agent**, in the same tmux window where you will run `harbor_run.sh`:

```bash
tmux new -s harbor                          # after a disconnect: tmux attach -t harbor
eval "$(ssh-agent)"                         # once per tmux window
ssh-add -t 12h ~/.ssh/id_lumi               # type the passphrase here; the agent forgets it after 12 h
ssh -o BatchMode=yes -o ControlPath=none lumi true && echo ok   # must print "ok" without asking anything
```

Don't start the run until the last line prints `ok`. If it says `Permission denied (publickey)`, the key is not in
the agent of *this* window. Run the `eval` and `ssh-add` lines again here. `ssh-add -l` shows which keys the agent
holds. With `IdentitiesOnly yes`, it must be the key named in `~/.ssh/config`.

The agent belongs to the window it was started in. A new tmux window needs its own `eval` and `ssh-add`.
After 12 hours, a running job is not affected, because its tunnel is already open. Before the next run, repeat
`ssh-add -t 12h ~/.ssh/id_lumi`.

### 3b. Run

```bash
tilde/harbor_run.sh --path data/harbor/terminal-bench@2.0 --n-tasks 5 --n-attempts 4 \
  --lumi-host lumi --lumi-repo /scratch/project_465003062/<you>/rl-rollgen --lumi-run <jobid>
```

The script:
1. Waits for the LUMI job's router: queue time plus vLLM startup, up to `--wait` minutes (default 60).
   It prints the job state when it changes.
2. Opens the ssh tunnel `127.0.0.1:18000 -> router`. Use `--local-port` if 18000 is taken: 127.0.0.1 on Tilde is
   shared by all users.
3. Reads the model name and context length from the server. The agent's input budget is the context length
   minus `--max-output-tokens` (default 4096).
4. Runs Harbor. When Harbor exits, the script closes the tunnel. The LUMI job keeps running until its
   `--time` runs out or you `scancel` it.

`--lumi-run` takes the job id (works while the job is still queued) or a run folder name from `logs/` on LUMI.
Without it, the script uses LUMI's `logs/latest_run`. That is the newest job that has *started*, so a job id is
safer. The API key is read from `~/.vllm_key` (`--api-key-file` to change). To skip retyping the LUMI flags,
put the command in a run file, e.g. `runs/harbor_tb2.sh`, like the LUMI runs.

Choosing what to run:
- `--n-tasks 5`: at most 5 tasks, picked after the name filters below (default: all).
- `--n-attempts 4`: 4 attempts per task, i.e. 4 samples per prompt (default 1).
- `--include-task-name '<glob>'` / `--exclude-task-name '<glob>'`: pick tasks by name; repeatable.
- `--n-concurrent 8`: trials at once, i.e. Docker containers running at the same time (default 8).
  Watch CPU, RAM and `/var/lib/docker`.

These are Harbor's own flags (short forms `-l`, `-k`, `-i`, `-x`, `-n` also work). `harbor_run.sh` passes
any flag it doesn't handle itself straight to `harbor run`; see `harbor run --help` for those, and
`tilde/harbor_run.sh --help` for ours.
While it runs, you can check the server from another window:
`curl -H "Authorization: Bearer $(cat ~/.vllm_key)" http://127.0.0.1:18000/v1/models`.

**Stop** a job with Ctrl-C, or `kill <harbor pid>` (SIGTERM behaves the same). Running trials are cancelled,
their outputs so far are saved, and their containers are removed. The tunnel stays up until Harbor has
finished, then closes. Check for leftover containers with `docker ps` (names start with the trial name).
If any are left, `docker rm -f` them.

**Resume** a stopped job. This works with the same or a new serve job, as long as it serves the same model:

```bash
tilde/harbor_run.sh --resume <job-name> \
  --lumi-host lumi --lumi-repo /scratch/project_465003062/<you>/rl-rollgen --lumi-run <jobid>
```

This keeps finished trials and reruns the cancelled ones. The job's settings come from its `config.json`.
The tunnel is reopened on the job's original local port, because Harbor stores that URL.

## 4. Results (Tilde)

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

## 5. Clean up

```bash
# LUMI (the tunnel on Tilde is already closed when harbor_run.sh exits):
scancel <jobid>
```
