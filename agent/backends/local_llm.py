"""
agent/backends/local_llm.py
---------------------------
LLM backend that calls a locally-running llama.cpp server.

Start your local server with:
    ./llama-server -m models/Llama-3.1-8B-Instruct.Q4_K_M.gguf \
        --port 8080 --n-gpu-layers 99 --ctx-size 8192

The server exposes an OpenAI-compatible /v1/chat/completions endpoint,
so this client works with any llama.cpp-served model.

Cost: always $0.00 — local inference has no per-token billing.
Latency on RTX 4080 with Q4_K_M: ~80 tokens/s.
"""

from __future__ import annotations

import os
import urllib.request
import json

from loguru import logger

from .base import BaseLLMBackend, LLMResponse


_DEFAULT_URL = os.getenv("LOCAL_LLM_URL", "http://localhost:8080")

# Rough token→char ratio for the cost-free prompt-token estimate.
# llama.cpp does not always return usage counts for all model formats.
_CHARS_PER_TOKEN = 4


class LocalLLMBackend(BaseLLMBackend):
    """
    Calls a llama.cpp server at LOCAL_LLM_URL.

    Falls back gracefully: if the server is unreachable, raises
    ConnectionError so the agent's backend selector can try Azure.
    """

    def __init__(self, base_url: str = _DEFAULT_URL, timeout: int = 120) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "local"

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> LLMResponse:
        payload = {
            "model": "local",          # llama.cpp ignores this field
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        url = f"{self._base_url}/v1/chat/completions"
        body = json.dumps(payload).encode()

        try:
            req = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read().decode())
        except OSError as exc:
            raise ConnectionError(
                f"Local LLM unreachable at {self._base_url}: {exc}"
            ) from exc

        choice = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", len(system + user) // _CHARS_PER_TOKEN)
        completion_tokens = usage.get("completion_tokens", len(choice) // _CHARS_PER_TOKEN)

        logger.debug(
            f"local LLM: {prompt_tokens}+{completion_tokens} tokens, "
            f"reply len={len(choice)}"
        )
        return LLMResponse(
            text=choice.strip(),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=0.0,
            backend="local",
        )

    def health_check(self) -> bool:
        """Returns True if the server responds to /health."""
        try:
            req = urllib.request.Request(f"{self._base_url}/v1/models")
            with urllib.request.urlopen(req, timeout=3):
                return True
        except OSError:
            return False

    def is_available(self) -> bool:
        return self.health_check()
