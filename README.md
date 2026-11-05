# rl-rollgen

### GOAL
A collection of scripts/python code that:
* launches vLLM inference 
* generates rollouts for one-step RL tasks
* generates rollouts for multi-step RL tasks (harbor)
* tokenizes the final rollouts
* operates on two clusters (LUMI, which is hpc with AMD, and Tilde servers with NVIDIA gpus and without slurm)
* has llm-as-a-judge support for RL tasks
* other useful scripts for filtering, etc.

### PRIORITIES
1. No overengineering: it's better to just have a neat collection of documented scripts, than to have a very complex structure no one will understand/need.
2. Using already built solutions instead of rewriting them by hand
3. Documentation - I will write it myself mostly, but scripts need to have clear names, and the structure of the project should also be clear
