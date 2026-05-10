"""
agent/backends/azure_oai.py
---------------------------
LLM backend that calls Azure OpenAI (GPT-4o by default).

Used only when:
  1. cloud_allowed is True (SENSITIVE route hard-disables this)
  2. The local backend raised ConnectionError OR returned low-confidence

Required environment variables:
    AZURE_OAI_ENDPOINT   e.g. https://<resource>.openai.azure.com/
    AZURE_OAI_KEY        your Azure OAI API key
    AZURE_OAI_DEPLOYMENT deployment name (default: gpt-4o)
    AZURE_OAI_API_VER    API version (default: 2024-08-01-preview)

Pricing (GPT-4o, as of mid-2025):
    Input:  $5.00 / 1M tokens  →  $0.000005 / token
    Output: $15.00 / 1M tokens →  $0.000015 / token
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.parse

from loguru import logger

from .base import BaseLLMBackend, LLMResponse


_ENDPOINT   = os.getenv("AZURE_OAI_ENDPOINT", "")
_KEY        = os.getenv("AZURE_OAI_KEY", "")
_DEPLOYMENT = os.getenv("AZURE_OAI_DEPLOYMENT", "gpt-4o")
_API_VER    = os.getenv("AZURE_OAI_API_VER", "2024-08-01-preview")

# GPT-4o pricing per token (USD)
_INPUT_COST_PER_TOKEN  = 5.00  / 1_000_000
_OUTPUT_COST_PER_TOKEN = 15.00 / 1_000_000


class AzureOAIBackend(BaseLLMBackend):
    """
    Calls Azure OpenAI chat completions endpoint.

    Raises RuntimeError on missing credentials so the caller gets a
    clear error rather than a silent HTTP 401.
    """

    def __init__(
        self,
        endpoint: str = _ENDPOINT,
        api_key: str = _KEY,
        deployment: str = _DEPLOYMENT,
        api_version: str = _API_VER,
        timeout: int = 60,
    ) -> None:
        if not endpoint or not api_key:
            raise RuntimeError(
                "AzureOAIBackend requires AZURE_OAI_ENDPOINT and AZURE_OAI_KEY "
                "environment variables to be set."
            )
        self._endpoint = endpoint.rstrip("/")
        self._api_key = api_key
        self._deployment = deployment
        self._api_version = api_version
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "azure"

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> LLMResponse:
        url = (
            f"{self._endpoint}/openai/deployments/{self._deployment}"
            f"/chat/completions?api-version={self._api_version}"
        )
        payload = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "api-key": self._api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode(errors="replace")
            raise RuntimeError(
                f"Azure OAI HTTP {exc.code}: {body_text[:300]}"
            ) from exc
        except OSError as exc:
            raise ConnectionError(f"Azure OAI unreachable: {exc}") from exc

        choice = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        cost = (
            prompt_tokens * _INPUT_COST_PER_TOKEN
            + completion_tokens * _OUTPUT_COST_PER_TOKEN
        )

        logger.debug(
            f"azure OAI: {prompt_tokens}+{completion_tokens} tokens, "
            f"cost=${cost:.6f}"
        )
        return LLMResponse(
            text=choice.strip(),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            backend="azure",
        )