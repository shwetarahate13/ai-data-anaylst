"""LLM access. The agent works with or without a model:

  * with ANTHROPIC_API_KEY  -> planner, text-to-SQL and narrator use Claude
  * without a key           -> deterministic rule-based fallbacks (everything still runs offline)

Numbers never come from the LLM: SQL + pandas compute them, the Critic verifies them, and the
narrator's output is rejected if it contains numbers that are not in the verified facts.
"""
from __future__ import annotations

import json
import re
from typing import Protocol

from .config import Settings


class LLM(Protocol):
    def complete(self, system: str, user: str, max_tokens: int = 1500) -> str: ...


class AnthropicLLM:
    def __init__(self, api_key: str, model: str):
        import anthropic  # lazy import so offline mode needs no SDK

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def complete(self, system: str, user: str, max_tokens: int = 1500) -> str:
        resp = self.client.messages.create(
            model=self.model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


def get_llm(settings: Settings) -> LLM | None:
    if not settings.api_key:
        return None
    return AnthropicLLM(settings.api_key, settings.model)


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply (handles ```json fences and chatter)."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in model reply")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("unterminated JSON object in model reply")
