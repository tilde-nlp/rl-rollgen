#!/bin/bash
# Download a Harbor registry dataset into data/harbor/<name>@<version>/<task>/ (browsable, editable, offline).
# Runs then use the local copy (tilde/harbor_run.sh -p data/harbor/<name>@<version>), never `harbor run -d`.
#
#   tilde/download_dataset.sh terminal-bench@2.0
#   harbor datasets list --legacy          # what exists
set -euo pipefail
die() { echo "FATAL: $*" >&2; exit 1; }
[[ "${1:-}" == ?*@?* ]] || die "usage: tilde/download_dataset.sh <name>@<version> (pin the version)"
DEST="$(cd "$(dirname "$0")/.." && pwd)/data/harbor/$1"
[ ! -e "$DEST" ] || die "$DEST exists (delete it to re-download)"
mkdir -p "$(dirname "$DEST")"
TMP=$(mktemp -d "$(dirname "$DEST")/.download-XXXXXX")
trap 'rm -rf "$TMP"' EXIT
harbor datasets download "$1" -o "$TMP"        # export mode: $TMP/<name>/<task>/
NAME=${1%@*}
mv "$TMP/${NAME##*/}" "$DEST"
echo "$(ls "$DEST" | wc -l) tasks in $DEST"
