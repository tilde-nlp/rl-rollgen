"""Load a judge folder (judges/<name>/) and turn the judge's answers into a reward.

A judge folder holds two files:
  prompt.txt   the judge prompt. {field} placeholders are filled from each rollout
               ({prompt}, {completion}, {answer}, {reward_spec[ground_truth]}, ...);
               {criteria} is replaced by the list of questions from rubric.toml.
               Write literal braces as {{ }}.
  rubric.toml  the reward formula and the questions (criteria):

    reward = "0 if not correct else 5 + clarity"   # must come before [criteria] (TOML rule)

    [criteria]
    correct = { type = "bool", question = "Is the final answer correct?" }
    clarity = { type = "int", min = 1, max = 5, question = "How clear is the explanation?" }

The judge must answer with JSON: {"reasoning": "...", "correct": true, "clarity": 4}
(vLLM structured outputs enforce this). The reward is a Python expression over the criterion
names; only min, max, abs and round are available. Bools count as 1/0 in arithmetic.
"""

import ast
import json
import sys
from pathlib import Path

try:
    import tomllib
except ImportError:  # Python < 3.11
    tomllib = None

TYPES = {"bool", "int"}
FUNCS = {"min": min, "max": max, "abs": abs, "round": round}
REASONING = "reasoning"  # free-text field the judge fills first, so it reasons before answering


class Judge:
    def __init__(self, folder):
        folder = Path(folder)
        prompt_file, rubric_file = folder / "prompt.txt", folder / "rubric.toml"
        for f in (prompt_file, rubric_file):
            if not f.is_file():
                sys.exit(f"judge folder {folder} needs {f.name}")
        if tomllib is None:
            sys.exit("reading rubric.toml needs Python >= 3.11 (tomllib)")
        self.prompt = prompt_file.read_text()
        self.rubric_text = rubric_file.read_text()
        try:
            rubric = tomllib.loads(self.rubric_text)
        except tomllib.TOMLDecodeError as e:
            sys.exit(f"{rubric_file}: {e}")

        def bad(msg):
            sys.exit(f"{rubric_file}: {msg}")

        unknown = rubric.keys() - {"criteria", "reward"}
        if unknown:
            bad(f"unknown keys {sorted(unknown)} (allowed: criteria, reward)")
        self.criteria = rubric.get("criteria") or bad("no [criteria]")
        if isinstance(self.criteria.get("reward"), str):
            bad("move `reward = ...` above [criteria]: in TOML, keys after a [table] line belong to that table")
        for name, c in self.criteria.items():
            if not name.isidentifier() or name == REASONING or name in FUNCS:
                bad(f"criterion name {name!r} must be a Python identifier, not {REASONING!r} or {sorted(FUNCS)}")
            if not isinstance(c, dict):
                bad(f"criterion {name}: expected {{ type = ..., question = ... }}")
            allowed = {"type", "question"} | ({"min", "max"} if c.get("type") == "int" else set())
            if c.keys() - allowed:
                bad(f"criterion {name}: unknown keys {sorted(c.keys() - allowed)}")
            if c.get("type") not in TYPES:
                bad(f"criterion {name}: type must be one of {sorted(TYPES)}")
            if not c.get("question"):
                bad(f"criterion {name}: missing question")
            if c["type"] == "int" and not (isinstance(c.get("min"), int) and isinstance(c.get("max"), int)):
                bad(f"criterion {name}: int needs integer min and max")

        self.reward_expr = rubric.get("reward") or bad('no reward = "..." formula')
        try:
            tree = ast.parse(self.reward_expr, mode="eval")
        except SyntaxError as e:
            bad(f"reward formula: {e}")
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        if names - self.criteria.keys() - FUNCS.keys():
            bad(f"reward formula uses unknown names {sorted(names - self.criteria.keys() - FUNCS.keys())}; "
                f"criteria are {sorted(self.criteria)}")
        if any(isinstance(n, ast.Attribute) for n in ast.walk(tree)):
            bad("reward formula: attribute access (x.y) is not allowed")
        self._code = compile(tree, "reward", "eval")

        if "{criteria}" not in self.prompt:
            sys.exit(f"{prompt_file}: must contain {{criteria}} (where the questions go)")

    def criteria_text(self):
        lines = []
        for name, c in self.criteria.items():
            kind = "true/false" if c["type"] == "bool" else f"integer {c['min']}-{c['max']}"
            lines.append(f'- "{name}" ({kind}): {c["question"]}')
        return "\n".join(lines)

    def response_format(self):
        props = {REASONING: {"type": "string"}}
        for name, c in self.criteria.items():
            props[name] = ({"type": "boolean"} if c["type"] == "bool" else
                           {"type": "integer", "minimum": c["min"], "maximum": c["max"]})
        schema = {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}
        return {"type": "json_schema", "json_schema": {"name": "judgement", "schema": schema, "strict": True}}

    def parse(self, text):
        """Judge reply -> (scores dict, None) or (None, error message)."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            return None, f"reply is not JSON ({e}); truncated by --max-tokens?"
        if not isinstance(data, dict):
            return None, "reply is not a JSON object"
        scores = {}
        for name, c in self.criteria.items():
            v = data.get(name)
            if c["type"] == "bool" and not isinstance(v, bool):
                return None, f"{name}: expected true/false, got {v!r}"
            if c["type"] == "int" and not (isinstance(v, int) and not isinstance(v, bool)
                                           and c["min"] <= v <= c["max"]):
                return None, f"{name}: expected integer {c['min']}-{c['max']}, got {v!r}"
            scores[name] = v
        return scores, None

    def reward(self, scores):
        return eval(self._code, {"__builtins__": {}, **FUNCS}, dict(scores))

    def check_reward(self):
        """Evaluate the formula on the lowest and highest answers, to catch errors early."""
        for pick in ("min", "max"):
            scores = {n: (pick == "max") if c["type"] == "bool" else c[pick] for n, c in self.criteria.items()}
            try:
                r = self.reward(scores)
            except Exception as e:
                sys.exit(f"reward formula fails on {scores}: {type(e).__name__}: {e}")
            if not isinstance(r, (int, float)):
                sys.exit(f"reward formula returned {r!r} on {scores}, expected a number")
