#!/bin/bash
# How much text would the difficulty classifier receive per SFT stage-1 file, and how much would a lower
# MAX_CHARS (configs/classifiers/difficulty/config.py) save? Only reads the files; no model or GPU involved.
# About as fast as test.sh: reads each file once in order, 8 files at a time, and parses only every Nth record
# (~2000 per file). Sampling by record number, not by file position, so long records aren't over-represented.
# Rows print as files finish, so not in list order.
# Lengths follow the classifier's logic: problem = system + user messages before the first assistant reply,
# answer = the last assistant message of a single-turn record, tools are never cut. System prompt not included.
#
# Usage (LUMI login node, from anywhere, plain python3 is enough): bash configs/runs/text_lengths_sft_stage1.sh
set -euo pipefail
cd /scratch/project_465003062/data/tilde-think
python3 - <<'EOF'
import json
from multiprocessing import Pool

SAMPLES = 2000     # per file, about
CAPS = [12000, 6000, 3000]
PROCESSES = 8

# path, records (counted by test.sh). finance-doc: the original, so this works before add_uuid.sh; same texts.
FILES = """
_merged/math/all.reason.no_tools.jsonl                                  285516
_merged/math/all.reason.tools.jsonl                                     259915
_merged/science/rqa.reason.no_tools.jsonl                               354241
_merged/science/rqa.reason.tools.jsonl                                   59268
_merged/science/so.reason.no_tools.jsonl                               2053500
_merged/science/so.reason.tools.jsonl                                    51995
_merged/science/syn_mcq.reason.no_tools.jsonl                            96293
_merged/science/syn_mcq.reason.tools.jsonl                                 315
_merged/science/vendor.reason.no_tools.jsonl                            221598
_merged/coding-v2/competitive_programming_cpp.reason.no_tools.jsonl     332186
_merged/coding-v2/competitive_programming_python.reason.no_tools.jsonl  333561
_merged/coding-v2/text_to_sql.reason.no_tools.jsonl                      96564
_merged/infinibyte/all.reason.no_tools.jsonl                           1174694
_merged/finance-doc/all.reason.no_tools.jsonl                            63056
_merged/arc/raw/small_reasoning_and_tools.jsonl                          15729
_merged/arc/raw/small_reasoning_no_tools.jsonl                           43980
_merged/agentic-v4/raw/tool_calling.jsonl                               549707
""".split()
FILES = list(zip(FILES[::2], map(int, FILES[1::2])))


def text(c):
    if isinstance(c, list):
        return "\n".join(p.get("text") or "" for p in c if isinstance(p, dict) and p.get("type") == "text")
    return c or ""


def lengths(r):
    """Characters of (problem, answer, tools) in this record."""
    tools = r.get("tools") or (r.get("responses_create_params") or {}).get("tools")
    tools = len(json.dumps(tools)) if tools else 0
    if "messages" not in r:
        return len(text(r.get("problem"))), len(text(r.get("expected_answer"))), tools
    m = r["messages"]
    replies = [i for i, x in enumerate(m) if x["role"] == "assistant"]
    first = replies[0] if replies else len(m)
    problem = "\n\n".join(text(x["content"]) for x in m[:first] if x["role"] in ("system", "user"))
    single_turn = replies == [len(m) - 1] and not m[-1].get("tool_calls")
    return len(problem), len(text(m[-1]["content"])) if single_turn else 0, tools


def sample_lines(path, records):
    step = max(1, records // SAMPLES)
    with open(path, "rb", buffering=1 << 20) as f:
        return [line for i, line in enumerate(f) if i % step == 0 and line.strip()]


def measure(item):
    path, records = item
    rows = [lengths(json.loads(line)) for line in sample_lines(path, records)]
    full = sorted(p + a for p, a, t in rows)
    over = 100.0 * sum(p > CAPS[0] or a > CAPS[0] for p, a, t in rows) / len(rows)
    tools = sum(t for p, a, t in rows) / len(rows)
    sent = [sum(min(p, c) + min(a, c) + t for p, a, t in rows) / len(rows) for c in CAPS]
    return path, records, len(rows), full[len(full) // 2], full[len(full) * 9 // 10], over, tools, sent


print("%-71s %7s %7s %7s %6s %7s  mean chars sent at cap %s" % ("file", "sampled", "p50", "p90", "%>12k", "tools", CAPS), flush=True)
total = [0.0] * len(CAPS)
with Pool(PROCESSES) as pool:
    for path, records, n, p50, p90, over, tools, sent in pool.imap_unordered(measure, FILES):
        total = [t + s * records for t, s in zip(total, sent)]
        print("%-71s %7d %7d %7d %6.1f %7d  %s" % (path, n, p50, p90, over, tools, " ".join("%7d" % s for s in sent)),
              flush=True)
print("\nTOTAL billions of chars sent (no system prompt):", {c: round(t / 1e9, 1) for c, t in zip(CAPS, total)})
EOF
