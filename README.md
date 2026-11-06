# rl-rollgen

Scripts for launching RL rollout generation on LUMI and Tilde servers. Human written README.md.

## Structure

1) You will ONLY need scripts from folders: lumi, tilde and runs.
2) You will ONLY need to use scripts, that do not start with an underscore (lumi/generate.sbatch etc.)

```
lumi/ - collection of scripts to run on lumi
tilde/ - collection of scripts to run on tilde servers (I assume you have docker rights)
runs/ - examples of running the scripts (shows how to pass arguments)

judges/ - collection of prompts and instructions for judges
logs/ - job outputs and vllm logs will be here (look at the latest_run) 
examples/ - dummy jsonl files for test runs
rollgen/ - python implementation of common functions
```

## One-step RL


### LUMI
lorem ipsum

### Tilde
TODO

## Multi-step RL

### Tilde sandboxes (docker) & LUMI vLLM

```
0. use docs/harbor_tilde.md to set it up
1. lumi/serve.sbatch on LUMI
2. tilde/harbor_run.sh on Tilde
```


### LUMI sandboxes (apptainer) & LUMI vLLM
TODO

### Tilde sandboxes (docker) & Tilde vLLM
TODO
