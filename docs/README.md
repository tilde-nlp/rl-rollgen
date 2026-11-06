# rl-rollgen docs

Generate RL rollouts with self-hosted vLLM and score them with an LLM judge, on LUMI and Tilde.

- [lumi.md](lumi.md): quickstart on LUMI (generate, judge, classify, serve)
- [data.md](data.md): input format, output fields, `meta.json` and `logs/` run records
- [judges.md](judges.md): writing a judge (`config.py`: the answer fields + the prompt), judging while generating
- [classifiers.md](classifiers.md): labeling datasets (e.g. difficulty of each prompt), same `config.py` format
- [harbor_tilde.md](harbor_tilde.md): multi-step agent rollouts with Harbor on Tilde, model served from LUMI

Tilde (one-step): Not yet available.

## Layout

```
lumi/           LUMI launchers (sbatch)
tilde/          Tilde launchers
rollgen/        the Python that does the work, same on both clusters
configs/        what you edit per experiment:
  judges/         one folder per judge: config.py           (--judge correctness)
  classifiers/    one folder per classifier: config.py      (--classifier difficulty)
  examples/       small example inputs
  runs/           per-experiment launch scripts (bash configs/runs/<name>.sh)
docs/           documentation
```
Files starting with `_` are internals.

## Goal

A collection of scripts/python code that:
* launches vLLM inference
* generates rollouts for one-step RL tasks
* generates rollouts for multi-step RL tasks (harbor)
* tokenizes the final rollouts
* operates on two clusters (LUMI, which is hpc with AMD, and Tilde servers with NVIDIA gpus and without slurm)
* has llm-as-a-judge support for RL tasks
* other useful scripts for filtering, etc.

## Priorities

1. No overengineering: it's better to just have a neat collection of documented scripts, than to have a very complex structure no one will understand/need.
2. Using already built solutions instead of rewriting them by hand
3. Documentation - I will write it myself mostly, but scripts need to have clear names, and the structure of the project should also be clear
