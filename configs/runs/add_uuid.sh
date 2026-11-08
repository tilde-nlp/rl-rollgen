#!/bin/bash
# Add a top-level "uuid" to the one SFT stage-1 file that has none (found by test.sh).
#   all.reason.no_tools.jsonl -> all.reason.no_tools_uuid.jsonl (the input is not touched)
# The uuid goes first in each record; the rest of the line is copied byte for byte.
# It is derived from the input path + record number (from 0, blank lines don't count, same as
# classify's record_idx), so a rerun gives the same uuids.
#
# Usage (LUMI login node, plain python3 is enough): bash configs/runs/add_uuid.sh
set -euo pipefail

DATA=/scratch/project_465003062/data/tilde-think
INPUT=$DATA/_merged/finance-doc/all.reason.no_tools.jsonl
OUTPUT=${INPUT%.jsonl}_uuid.jsonl

[ -f "$INPUT" ] || { echo "ERROR: input not found: $INPUT" >&2; exit 1; }
[ ! -e "$OUTPUT" ] || { echo "ERROR: output already exists, not overwriting: $OUTPUT" >&2; exit 1; }

# Written under a temporary name, so a failed run never leaves a partial output.
trap 'rm -f "$OUTPUT.tmp"' EXIT
python3 - "$INPUT" "$OUTPUT.tmp" <<'EOF'
import json, sys, uuid

src, dst = sys.argv[1:]
n = 0
with open(src, "rb") as f, open(dst, "wb") as out:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if not isinstance(rec, dict) or not rec:
            sys.exit("record %d: not a non-empty JSON object" % n)
        if "uuid" in rec:
            sys.exit("record %d already has a uuid" % n)
        new_id = uuid.uuid5(uuid.NAMESPACE_URL, "%s:%d" % (src, n))
        out.write(b'{"uuid": "%s", ' % str(new_id).encode() + line[1:].lstrip() + b"\n")
        n += 1
print("%d records" % n)
EOF
mv "$OUTPUT.tmp" "$OUTPUT"
echo "Wrote $OUTPUT"
