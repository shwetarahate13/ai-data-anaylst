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


class OpenAILLM:
    """OpenAI-compatible LLM client using httpx.
    Supports OpenAI, Ollama, Groq, OpenRouter, and local vLLM/LM Studio.
    """
    def __init__(self, api_key: str, model: str, base_url: str = "https://api.openai.com/v1", timeout: float = 60.0):
        self.api_key = api_key or "sk-local-no-key-required"
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def complete(self, system: str, user: str, max_tokens: int = 1500) -> str:
        import httpx
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]


def get_llm(settings: Settings) -> LLM | None:
    provider = (settings.provider or "").strip().lower()

    # Explicit or auto-detected OpenAI
    if provider in ("openai", "groq", "openrouter") or (not provider and settings.openai_api_key and not settings.api_key):
        key = settings.openai_api_key or settings.api_key or ""
        if not key:
            return None
        model = settings.model if (settings.model and settings.model != "claude-sonnet-5-5") else settings.openai_model
        return OpenAILLM(api_key=key, model=model, base_url=settings.openai_base_url)

    # Local Ollama
    if provider == "ollama":
        model = settings.model if (settings.model and settings.model != "claude-sonnet-5-5") else settings.ollama_model
        return OpenAILLM(api_key="ollama", model=model, base_url=settings.ollama_base_url)

    # Anthropic (default if api_key present)
    if provider == "anthropic" or settings.api_key:
        if not settings.api_key:
            return None
        model = settings.model or "claude-sonnet-5-5"
        return AnthropicLLM(settings.api_key, model)

    # Fallback to OpenAI if openai_api_key set
    if settings.openai_api_key:
        model = settings.model if (settings.model and settings.model != "claude-sonnet-5-5") else settings.openai_model
        return OpenAILLM(api_key=settings.openai_api_key, model=model, base_url=settings.openai_base_url)

    return None


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
