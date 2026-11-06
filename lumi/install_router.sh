#!/bin/bash
# One-time, on a LUMI login node: install vllm-router into lumi/vllm-router/ (gitignored).
# Every lumi/*.sbatch job runs it in front of all its vLLM servers: the client (rollgen/*.py, or Harbor on
# Tilde for serve.sbatch) talks to one URL and the router spreads the requests.
#
#   bash lumi/install_router.sh
#
# Arguments: none. Versions are pinned below (ROUTER_VERSION, ORJSON_VERSION; override by exporting them,
# e.g. ROUTER_VERSION=0.1.17 bash lumi/install_router.sh). To reinstall, delete lumi/vllm-router/ first.
#
# Installed with --no-deps into a plain folder (put on PYTHONPATH): the container already ships the
# router's dependencies (fastapi, uvicorn, aiohttp, requests, setproctitle) except orjson, which is
# installed next to it. The router runs inside the same container.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO/lumi/_common.sh"
: "${ROUTER_VERSION:=0.1.16}"   # pinned for reproducible runs
: "${ORJSON_VERSION:=3.13.0}"

[ ! -e "$ROUTER_DIR" ] || die "$ROUTER_DIR exists (delete it to reinstall)"
TMP=$(mktemp -d "$REPO/lumi/.vllm-router-XXXXXX")
trap 'rm -rf "$TMP"' EXIT
# --no-cache-dir: pip's cache would go to ~/.cache (100K-file home quota)
run_in_container python -m pip install --no-cache-dir --no-deps --target "$TMP" \
  "vllm-router==$ROUTER_VERSION" "orjson==$ORJSON_VERSION"
run_in_container env PYTHONNOUSERSITE=1 PYTHONPATH="$TMP" python -c \
  'import vllm_router.router, vllm_router.launch_router, orjson, setproctitle, aiohttp, uvicorn, fastapi, requests' \
  || die "vllm-router does not import in the container (missing dependency? see above)"
mv "$TMP" "$ROUTER_DIR"
echo "OK: vllm-router $ROUTER_VERSION in $ROUTER_DIR"
