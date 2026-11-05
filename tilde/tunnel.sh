#!/bin/bash
# Forward the vLLM servers of a running LUMI job to this (Tilde) machine over SSH.
#
# LUMI jobs write their server URLs to <repo on LUMI>/logs/<run>/urls/<node>.txt
# (see lumi/_vllm_node.sh). This script reads them over ssh, opens one local port per server
# through a LUMI login node, and writes the local URLs to logs/tunnel.urls, where
# tilde/harbor_run.sh picks them up. It runs in the foreground; stop it with Ctrl-C.
#
#   tilde/tunnel.sh                    # tunnel the newest LUMI run (logs/latest_run)
#   tilde/tunnel.sh 20261105-...-123   # a specific run dir name under logs/ on LUMI
#   tilde/tunnel.sh --urls http://10.253.0.12:8000/v1,http://10.253.0.12:8001/v1   # explicit, no lookup
#
# Needs: passwordless ssh to $LUMI_HOST (key registered with LUMI); the login node must reach
# the compute nodes' ports. Run it inside tmux/screen, or with `&`, so it outlives your shell.
set -euo pipefail

# --- config (override by exporting) ---
: "${LUMI_HOST:=lumi-uan01.csc.fi}"   # one fixed login node (lumi-uan01..04); lumi.csc.fi round-robins
: "${LUMI_USER:=}"                    # LUMI username if it differs from the local one (or use ~/.ssh/config)
: "${LUMI_REPO:=}"                    # path of this repo's clone on LUMI, e.g. /scratch/project_465003062/$USER/rl-rollgen
: "${LOCAL_BASE_PORT:=18000}"         # first local port; server i gets LOCAL_BASE_PORT + i

REPO="$(cd "$(dirname "$0")/.." && pwd)"
URLS_FILE="$REPO/logs/tunnel.urls"
die() { echo "FATAL: $*" >&2; exit 1; }
SSH_TARGET="${LUMI_USER:+$LUMI_USER@}$LUMI_HOST"

# --- 1. remote URLs ---
if [ "${1:-}" = "--urls" ]; then
  [ -n "${2:-}" ] || die "--urls needs a comma-separated list"
  REMOTE_URLS=$(tr ',' '\n' <<< "$2")
else
  [ -n "$LUMI_REPO" ] || die "set LUMI_REPO to the repo path on LUMI (or pass --urls)"
  RUN="${1:-latest_run}"
  REMOTE_URLS=$(ssh "$SSH_TARGET" "cat '$LUMI_REPO/logs/$RUN/urls/'*.txt" 2>/dev/null) \
    || die "no URL files in $SSH_TARGET:$LUMI_REPO/logs/$RUN/urls/ (job not running yet, or servers still starting?)"
fi
mapfile -t REMOTE < <(grep . <<< "$REMOTE_URLS")
(( ${#REMOTE[@]} )) || die "no URLs found"

# --- 2. one -L per server ---
FORWARDS=() LOCAL=()
for i in "${!REMOTE[@]}"; do
  hostport=$(sed -E 's#^https?://([^/]+).*#\1#' <<< "${REMOTE[i]}")   # 10.253.0.12:8000
  [[ "$hostport" == *:* ]] || die "cannot parse host:port from ${REMOTE[i]}"
  port=$(( LOCAL_BASE_PORT + i ))
  FORWARDS+=(-L "127.0.0.1:$port:$hostport")
  LOCAL+=("http://127.0.0.1:$port/v1")
  echo "${REMOTE[i]}  ->  http://127.0.0.1:$port/v1"
done

mkdir -p "$(dirname "$URLS_FILE")"
(IFS=,; echo "${LOCAL[*]}") > "$URLS_FILE"
echo "${RUN:-explicit --urls}" > "$REPO/logs/tunnel.run"   # recorded by harbor_run.sh
echo "local URLs written to $URLS_FILE"
echo "use:  export VLLM_URLS=$(cat "$URLS_FILE")"
echo "tunnel open via $SSH_TARGET (Ctrl-C to close)"

# --- 3. hold the tunnel ---
# ExitOnForwardFailure: fail loudly if a local port is taken. ServerAlive*: notice a dead link.
exec ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  "${FORWARDS[@]}" "$SSH_TARGET"
