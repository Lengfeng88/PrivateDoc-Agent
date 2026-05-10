from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Unified response contract returned by every LLM backend."""
    text: str
    prompt_tokens: int
    completion_tokens: int
    backend: str        # "local" | "azure"
    cost_usd: float     # 0.0 for local


class BaseLLMBackend(ABC):
    """Abstract base class for all LLM backends."""

    @abstractmethod
    def complete(self, system: str, user: str) -> LLMResponse:
        """Send a completion request and return a unified LLMResponse."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the backend is reachable."""
        ...