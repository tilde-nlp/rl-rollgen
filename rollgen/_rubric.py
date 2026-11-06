"""Load a judge or classifier folder and ask the model about one record with it.

A folder (judges/<name>/ or classifiers/<name>/) holds 2-3 files:
  prompt.txt   the user message. {field} placeholders are filled from each record
               ({prompt}, {completion}, {answer}, {reward_spec[ground_truth]}, ...);
               {criteria} is replaced by the list of questions from rubric.toml.
               {field?} = optional: empty if the record doesn't have it. Write literal braces as {{ }}.
    or
  prompt.json  the user message as a JSON object, e.g.
                 {"problem": "{problem}", "tools": "{responses_create_params[tools]?}"}
               A value that is exactly one placeholder is replaced by the record's value (any JSON
               type); with "?" the key is left out if the record doesn't have it. Other values are
               kept as they are. Sent as compact JSON.
  system.txt   (optional) the system message, sent as-is (no placeholders, braces need no escaping).
  rubric.toml  what the model must answer, plus settings:

    reward = "0 if not correct else 5 + clarity"   # judges only; top-level keys before [criteria] (TOML rule)
    reasoning = true          # default: the model writes a free-text "reasoning" field first
    max_field_chars = 12000   # optional: cut longer strings inserted into the prompt (head and tail kept)

    [criteria]
    correct = { type = "bool", question = "Is the final answer correct?" }
    clarity = { type = "int", min = 1, max = 5, question = "How clear is the explanation?" }
    level   = { type = "enum", values = ["easy", "hard"], question = "How hard is the task?" }
    uses_tools = { type = "bool", when = "tools", question = "..." }   # asked only if the record has `tools`

The model must answer with JSON, e.g. {"reasoning": "...", "correct": true, "clarity": 4}
(vLLM structured outputs enforce this). A criterion with `when` that isn't asked is null.
The reward is a Python expression over the criterion names; only min, max, abs and round are
available. Bools count as 1/0 in arithmetic.

Missing (required) placeholder or a bad reply: that record gets an error message instead of answers.
"""

import ast
import hashlib
import json
import re
import string
import sys
from pathlib import Path

try:
    import tomllib
except ImportError:  # Python < 3.11
    tomllib = None

from _common import get_messages

TYPES = {"bool", "int", "enum"}
FUNCS = {"min": min, "max": max, "abs": abs, "round": round}
REASONING = "reasoning"  # free-text field the model fills first, so it reasons before answering
CRITERIA = "criteria"    # placeholder for the list of questions
SETTINGS = {"reward", "reasoning", "max_field_chars"}  # top-level keys of rubric.toml besides [criteria]
TRUNCATION_MARKER = "\n[... truncated ...]\n"
CACHE_SIZE = 1_000_000  # ask(cache=...): answers kept for reuse
PLACEHOLDER = re.compile(r"(\w+)((?:\[[^\[\]]+\])*)(\??)")  # name[key][0]... with an optional "?"


class PromptError(Exception):
    """The prompt can't be built for this record (e.g. a placeholder field is missing)."""


def _path(text, where):
    """'a[b][0]?' -> (('a', 'b', '0'), optional)."""
    m = PLACEHOLDER.fullmatch(text)
    if not m:
        sys.exit(f"{where}: bad placeholder {{{text}}}: expected {{name}}, {{name[key]}} or {{name?}}")
    return (m[1], *re.findall(r"\[([^\[\]]+)\]", m[2])), m[3] == "?"


def _lookup(record, path):
    """Value at path (dict keys, list indices), or None if it isn't there."""
    v = record
    for k in path:
        if isinstance(v, dict):
            v = v.get(k)
        elif isinstance(v, list) and k.isdigit() and int(k) < len(v):
            v = v[int(k)]
        else:
            return None
    return v


def _present(v):
    return v is not None and v != "" and v != [] and v != {}


class Rubric:
    """A judge or classifier folder. need_reward: True for judges (reward formula required),
    False for classifiers (no reward allowed)."""

    def __init__(self, folder, need_reward):
        folder = Path(folder)
        if not folder.is_dir():
            sys.exit(f"{folder}: not a folder")
        self.folder = folder
        prompts = [f for f in (folder / "prompt.txt", folder / "prompt.json") if f.is_file()]
        rubric_file = folder / "rubric.toml"
        if len(prompts) == 2:
            sys.exit(f"{folder} has both prompt.txt and prompt.json; keep one")
        if not prompts or not rubric_file.is_file():
            sys.exit(f"{folder} needs rubric.toml and one of prompt.txt / prompt.json")
        prompt_file = prompts[0]
        self.prompt = prompt_file.read_text()
        system_file = folder / "system.txt"
        self.system = system_file.read_text().strip() if system_file.is_file() else None
        self.rubric_text = rubric_file.read_text()

        self._load_rubric(rubric_file, need_reward)
        if prompt_file.suffix == ".json":
            self._load_json_prompt(prompt_file)
        else:
            self._load_text_prompt(prompt_file)
        if CRITERIA not in self._placeholder_names and self.system is None:
            sys.exit(f"{prompt_file}: must contain {{{CRITERIA}}} (where the questions go), "
                     f"unless system.txt describes the fields")

    # --- loading -------------------------------------------------------------------------------

    def _load_rubric(self, rubric_file, need_reward):
        if tomllib is None:
            sys.exit("reading rubric.toml needs Python >= 3.11 (tomllib)")
        try:
            rubric = tomllib.loads(self.rubric_text)
        except tomllib.TOMLDecodeError as e:
            sys.exit(f"{rubric_file}: {e}")

        def bad(msg):
            sys.exit(f"{rubric_file}: {msg}")

        unknown = rubric.keys() - SETTINGS - {"criteria"}
        if unknown:
            bad(f"unknown keys {sorted(unknown)} (allowed: {sorted(SETTINGS)}, [criteria])")
        self.criteria = rubric.get("criteria") or bad("no [criteria]")
        misplaced = sorted(k for k in SETTINGS if k in self.criteria and not isinstance(self.criteria[k], dict))
        if misplaced:
            bad(f"move {misplaced} above [criteria]: in TOML, keys after a [table] line belong to that table")
        for name, c in self.criteria.items():
            if not name.isidentifier() or name == REASONING or name in FUNCS:
                bad(f"criterion name {name!r} must be a Python identifier, not {REASONING!r} or {sorted(FUNCS)}")
            if not isinstance(c, dict):
                bad(f"criterion {name}: expected {{ type = ..., question = ... }}")
            if c.get("type") not in TYPES:
                bad(f"criterion {name}: type must be one of {sorted(TYPES)}")
            allowed = {"type", "question", "when"} | {"int": {"min", "max"}, "enum": {"values"}}.get(c["type"], set())
            if c.keys() - allowed:
                bad(f"criterion {name}: unknown keys {sorted(c.keys() - allowed)}")
            if not c.get("question"):
                bad(f"criterion {name}: missing question")
            if c["type"] == "int" and not (isinstance(c.get("min"), int) and isinstance(c.get("max"), int)):
                bad(f"criterion {name}: int needs integer min and max")
            if c["type"] == "enum":
                values = c.get("values")
                if not (isinstance(values, list) and values and all(isinstance(v, str) and v for v in values)
                        and len(set(values)) == len(values)):
                    bad(f"criterion {name}: enum needs values = [\"a\", \"b\", ...] (distinct, non-empty strings)")
            if "when" in c:
                if not isinstance(c["when"], str):
                    bad(f"criterion {name}: when must be a field name like \"tools\" or \"a[b]\"")
                path, optional = _path(c["when"], f"{rubric_file}: criterion {name}: when")
                if optional:
                    bad(f"criterion {name}: when takes a field name without '?'")
                c["_when"] = path

        self.reasoning = rubric.get("reasoning", True)
        if not isinstance(self.reasoning, bool):
            bad("reasoning must be true or false")
        self.max_field_chars = rubric.get("max_field_chars")
        if self.max_field_chars is not None and not (isinstance(self.max_field_chars, int)
                                                     and self.max_field_chars > 2 * len(TRUNCATION_MARKER)):
            bad("max_field_chars must be an integer (a few hundred or more)")

        self.reward_expr = rubric.get("reward")
        if not need_reward:
            if self.reward_expr is not None:
                bad("a classifier has no reward; remove `reward = ...`")
            return
        if not self.reward_expr:
            bad('no reward = "..." formula')
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

    def _load_text_prompt(self, prompt_file):
        self._json = None
        self._parts = []  # (literal text, placeholder path or None, optional)
        try:
            parsed = list(string.Formatter().parse(self.prompt))
        except ValueError as e:
            sys.exit(f"{prompt_file}: {e} (write literal braces as {{{{ }}}})")
        for literal, field, spec, conversion in parsed:
            if field is None:
                self._parts.append((literal, None, False))
                continue
            if spec or conversion:
                sys.exit(f"{prompt_file}: placeholder {{{field}{'!' + conversion if conversion else ''}"
                         f"{':' + spec if spec else ''}}}: format specs and conversions are not supported")
            self._parts.append((literal, *_path(field, str(prompt_file))))
        self._placeholder_names = {p[0] for _, p, _ in self._parts if p}

    def _load_json_prompt(self, prompt_file):
        try:
            template = json.loads(self.prompt)
        except json.JSONDecodeError as e:
            sys.exit(f"{prompt_file}: not valid JSON: {e}")
        if not isinstance(template, dict):
            sys.exit(f"{prompt_file}: must be a JSON object")
        self._json = []  # (key, placeholder path or None, optional, literal value)
        for k, v in template.items():
            m = re.fullmatch(r"\{([^{}]*)\}", v) if isinstance(v, str) else None
            if m:
                self._json.append((k, *_path(m[1], str(prompt_file)), None))
            elif isinstance(v, str) and any(PLACEHOLDER.fullmatch(x) for x in re.findall(r"\{([^{}]*)\}", v)):
                sys.exit(f"{prompt_file}: key {k!r}: a placeholder must be the whole value, like \"{{field}}\"")
            else:
                self._json.append((k, None, False, v))
        self._placeholder_names = {p[0] for _, p, _, _ in self._json if p}

    # --- one record ----------------------------------------------------------------------------

    def asked(self, record):
        """Criteria the model answers for this record (those with `when` only if the field is there)."""
        return {n: c for n, c in self.criteria.items() if "_when" not in c or _present(_lookup(record, c["_when"]))}

    def criteria_text(self, asked):
        lines = []
        for name, c in asked.items():
            kind = {"bool": "true/false",
                    "int": f"integer {c.get('min')}-{c.get('max')}",
                    "enum": "one of: " + ", ".join(c.get("values") or [])}[c["type"]]
            lines.append(f'- "{name}" ({kind}): {c["question"]}')
        return "\n".join(lines)

    def _value(self, record, path, optional, asked):
        """(found, value) for one placeholder."""
        if path == (CRITERIA,):
            return True, self.criteria_text(asked)
        if path == ("prompt",) and not isinstance(record.get("prompt"), str) and get_messages(record):
            # rollouts with chat messages: {prompt} is the last user message
            v = next((m.get("content") for m in reversed(get_messages(record)) if m.get("role") == "user"), None)
        else:
            v = _lookup(record, path)
        if v is None and not optional:
            raise PromptError(f"record has no field {self._name(path)} (used in {self.folder}'s prompt)")
        if optional and not _present(v):
            return False, None
        if isinstance(v, str) and self.max_field_chars and len(v) > self.max_field_chars:
            keep = self.max_field_chars - len(TRUNCATION_MARKER)
            head = keep * 2 // 3
            v = v[:head] + TRUNCATION_MARKER + v[len(v) - (keep - head):]
        return True, v

    @staticmethod
    def _name(path):
        return "{" + path[0] + "".join(f"[{k}]" for k in path[1:]) + "}"

    def user_message(self, record, asked):
        if self._json is not None:
            out = {}
            for key, path, optional, literal in self._json:
                if path is None:
                    out[key] = literal
                    continue
                found, v = self._value(record, path, optional, asked)
                if found:
                    out[key] = v
            return json.dumps(out, ensure_ascii=False, separators=(",", ":"))
        text = []
        for literal, path, optional in self._parts:
            text.append(literal)
            if path is not None:
                found, v = self._value(record, path, optional, asked)
                if found:
                    text.append(v if isinstance(v, str) else json.dumps(v, ensure_ascii=False))
        return "".join(text)

    def messages(self, record, asked):
        system = [{"role": "system", "content": self.system}] if self.system else []
        return system + [{"role": "user", "content": self.user_message(record, asked)}]

    def response_format(self, asked):
        props = {REASONING: {"type": "string"}} if self.reasoning else {}
        for name, c in asked.items():
            props[name] = {"bool": {"type": "boolean"},
                           "int": {"type": "integer", "minimum": c.get("min"), "maximum": c.get("max")},
                           "enum": {"type": "string", "enum": c.get("values")}}[c["type"]]
        schema = {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}
        return {"type": "json_schema", "json_schema": {"name": "answers", "schema": schema, "strict": True}}

    def parse(self, text, asked):
        """Model reply -> (answers for all criteria (null if not asked), None) or (None, error message)."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            return None, f"reply is not JSON ({e}); truncated by --max-tokens?"
        if not isinstance(data, dict):
            return None, "reply is not a JSON object"
        answers = {}
        for name, c in self.criteria.items():
            v = data.get(name)
            if name not in asked:
                v = None
            elif c["type"] == "bool" and not isinstance(v, bool):
                return None, f"{name}: expected true/false, got {v!r}"
            elif c["type"] == "int" and not (isinstance(v, int) and not isinstance(v, bool)
                                             and c["min"] <= v <= c["max"]):
                return None, f"{name}: expected integer {c['min']}-{c['max']}, got {v!r}"
            elif c["type"] == "enum" and v not in c["values"]:
                return None, f"{name}: expected one of {c['values']}, got {v!r}"
            answers[name] = v
        return answers, None

    async def ask(self, client, model, record, temperature, max_tokens, extra_body, cache=None):
        """Ask the model about one record -> (answers or None, raw reply or None, error or None).
        Server errors raise (the record is then retried on the next run); prompt and reply problems
        are returned as the error. cache: a dict; identical requests are then sent only once."""
        asked = self.asked(record)
        try:
            messages = self.messages(record, asked)
        except PromptError as e:
            return None, None, str(e)
        if cache is not None:
            key = hashlib.sha1(json.dumps([messages, list(asked)], ensure_ascii=False).encode()).hexdigest()
            if key in cache:
                return cache[key]
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=self.response_format(asked),
            extra_body=extra_body,
        )
        text = resp.choices[0].message.content or ""
        answers, error = self.parse(text, asked)
        if cache is not None and error is None:
            if len(cache) >= CACHE_SIZE:
                del cache[next(iter(cache))]  # oldest first
            cache[key] = answers, text, error
        return answers, text, error

    def check_record(self, record, where):
        """Build the prompt for one record now, to fail before starting on a typo'd placeholder."""
        try:
            self.messages(record, self.asked(record))
        except PromptError as e:
            sys.exit(f"{where}: {e}")

    def record(self, prefix):
        """The folder's files, stored in <output>.meta.json (an edited folder then refuses to resume)."""
        return {f"{prefix}_prompt": self.prompt, f"{prefix}_system": self.system, f"{prefix}_rubric": self.rubric_text}

    # --- reward (judges) -----------------------------------------------------------------------

    def reward(self, answers):
        return eval(self._code, {"__builtins__": {}, **FUNCS}, dict(answers))

    def check_reward(self):
        """Evaluate the formula on the lowest and highest answers (and with `when` criteria unasked,
        i.e. None), to catch errors early."""
        cases = []
        for pick in ("min", "max"):
            cases.append({n: (pick == "max") if c["type"] == "bool" else
                          c["values"][0 if pick == "min" else -1] if c["type"] == "enum" else c[pick]
                          for n, c in self.criteria.items()})
        if any("_when" in c for c in self.criteria.values()):
            cases += [{n: None if "_when" in self.criteria[n] else v for n, v in case.items()} for case in cases]
        for answers in cases:
            try:
                r = self.reward(answers)
            except Exception as e:
                sys.exit(f"reward formula fails on {answers}: {type(e).__name__}: {e}")
            if not isinstance(r, (int, float)):
                sys.exit(f"reward formula returned {r!r} on {answers}, expected a number")
