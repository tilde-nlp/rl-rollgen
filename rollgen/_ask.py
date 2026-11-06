"""Judges and classifiers: a config folder that turns one record into a request, and asks vLLM.

A config is a folder (configs/judges/<name>/ or configs/classifiers/<name>/) with a `config.py`
(and any files it reads). config.py defines:

    class Output(BaseModel):          # pydantic: what the model must answer. Becomes the JSON schema
        reasoning: str                #   vLLM enforces (structured outputs); fields are generated in
        correct: bool                 #   this order, so put free-text reasoning first if you want it.
        clarity: int = Field(ge=1, le=5)
        level: Literal["easy", "hard"]

    def messages(record):             # one input record (dict) -> the request
        return [{"role": "user", "content": f"..."}]   # chat messages, or a str (one user message),
                                                       # or None: skip this record (no output line)

If messages() raises, that record gets an error line (judge_error / classify_error); on the first
record the job stops before it starts (typo check). A reply that doesn't fit Output is an error too.

Helpers for config.py (`from _ask import truncate, last_user_message`):
  truncate(text, n)         cut text longer than n chars (head 2/3 + tail 1/3 kept, marker between)
  last_user_message(record) the prompt of a rollout: last user message, or "prompt" if it's a string

The contents of every file in the folder are stored in <output>.meta.json, so resuming with an
edited config is refused.
"""

import hashlib
import importlib.util
import json
import sys
import traceback
from pathlib import Path

from _common import get_messages

CONFIG_FILE = "config.py"
TRUNCATION_MARKER = "\n[... truncated ...]\n"
CACHE_SIZE = 1_000_000  # ask(cache=...): answers kept for reuse


# --- helpers for config.py ---------------------------------------------------------------------

def truncate(text, max_chars):
    """text, or if longer than max_chars: its head (2/3) and tail (1/3) with a marker between."""
    if not isinstance(text, str) or len(text) <= max_chars:
        return text
    keep = max_chars - len(TRUNCATION_MARKER)
    head = keep * 2 // 3
    return text[:head] + TRUNCATION_MARKER + text[len(text) - (keep - head):]


def last_user_message(record):
    """The prompt of a rollout from generate.py: "prompt" if it's a string, otherwise the last user
    message of its chat messages ("messages", or "prompt" as a list)."""
    if isinstance(record.get("prompt"), str):
        return record["prompt"]
    return next((m.get("content") for m in reversed(get_messages(record) or []) if m.get("role") == "user"), None)


# --- the engine --------------------------------------------------------------------------------

class ConfigError(Exception):
    """messages() failed for this record."""


class Config:
    """A judge or classifier folder with config.py."""

    def __init__(self, folder):
        folder = Path(folder)
        self.folder = folder
        self.file = folder / CONFIG_FILE
        if not self.file.is_file():
            sys.exit(f"{folder}: needs a {CONFIG_FILE} (see rollgen/_ask.py, or copy a folder from configs/)")
        spec = importlib.util.spec_from_file_location(f"config_{folder.name}", self.file)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.exit(f"{self.file} fails to load:\n{traceback.format_exc()}")
        from pydantic import BaseModel
        self.Output = getattr(module, "Output", None)
        if not (isinstance(self.Output, type) and issubclass(self.Output, BaseModel)):
            sys.exit(f"{self.file}: must define class Output(BaseModel) (the fields the model answers)")
        if not callable(getattr(module, "messages", None)):
            sys.exit(f"{self.file}: must define messages(record) -> chat messages, a str, or None")
        self._messages = module.messages
        self.fields = list(self.Output.model_fields)
        schema = self.Output.model_json_schema()
        schema["additionalProperties"] = False
        self.response_format = {"type": "json_schema",
                                "json_schema": {"name": "output", "schema": schema, "strict": True}}

    def messages(self, record):
        """Chat messages for one record, or None (skip it). Raises ConfigError."""
        try:
            m = self._messages(record)
        except Exception as e:
            frame = next((f for f in reversed(traceback.extract_tb(e.__traceback__))
                          if Path(f.filename).resolve() == self.file.resolve()), None)
            where = f"{self.file}" + (f" line {frame.lineno}" if frame else "")
            raise ConfigError(f"{where}: messages() failed: {type(e).__name__}: {e}") from None
        if m is None:
            return None
        if isinstance(m, str):
            return [{"role": "user", "content": m}]
        if not (isinstance(m, list) and m and all(isinstance(x, dict) and "role" in x for x in m)):
            raise ConfigError(f"{self.file}: messages() returned {type(m).__name__}; expected a list of "
                              f"{{'role': ..., 'content': ...}}, a str or None")
        return m

    def parse(self, text):
        """Model reply -> (answers dict, None) or (None, error message)."""
        from pydantic import ValidationError
        try:
            return self.Output.model_validate_json(text).model_dump(mode="json"), None
        except ValidationError as e:
            if any(err["type"] == "json_invalid" for err in e.errors()):
                return None, "reply is not valid JSON; truncated by --max-tokens?"
            return None, "reply doesn't fit Output: " + "; ".join(
                f"{'.'.join(map(str, err['loc'])) or 'reply'}: {err['msg']}" for err in e.errors())

    async def ask(self, client, model, record, temperature, max_tokens, extra_body, cache=None):
        """Ask the model about one record -> (answers or None, raw reply or None, error or None),
        or None if messages() skips the record. Server errors raise (the record is then retried on
        the next run); config and reply problems are returned as the error.
        cache: a dict; identical requests are then sent only once."""
        try:
            messages = self.messages(record)
        except ConfigError as e:
            return None, None, str(e)
        if messages is None:
            return None
        if cache is not None:
            key = hashlib.sha1(json.dumps(messages, ensure_ascii=False, default=str).encode()).hexdigest()
            if key in cache:
                return cache[key]
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=self.response_format,
            extra_body=extra_body,
        )
        text = resp.choices[0].message.content or ""
        answers, error = self.parse(text)
        result = answers, text, error
        if cache is not None and error is None:
            if len(cache) >= CACHE_SIZE:
                del cache[next(iter(cache))]  # oldest first
            cache[key] = result
        return result

    def check_record(self, record, where):
        """Build the request for one record now, to fail before starting on a typo."""
        try:
            self.messages(record)
        except ConfigError as e:
            sys.exit(f"{where}: {e}\n(fields of this record: {sorted(record)})")

    def record(self, prefix):
        """Every file of the folder, for <output>.meta.json (an edited config then refuses to resume)."""
        return {f"{prefix}_file {p.relative_to(self.folder)}": p.read_text(errors="replace")
                for p in sorted(self.folder.rglob("*"))
                if p.is_file() and "__pycache__" not in p.parts and not p.name.startswith(".")}
