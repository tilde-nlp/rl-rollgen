#!/bin/bash
# One-time setup on a Tilde server: uv + the harbor CLI (into ~/.local/bin). Docker must already work.
#   bash tilde/install.sh
# Arguments: none. Harbor version pinned below (override: HARBOR_VERSION=0.21.0 bash tilde/install.sh).
# Rerunning it is safe (upgrades/downgrades harbor to the pin).
set -euo pipefail
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv tool install --upgrade "harbor==${HARBOR_VERSION:-0.20.0}"   # pinned for reproducible runs
docker info >/dev/null && harbor --help >/dev/null && echo "OK: uv, harbor and docker ready"
