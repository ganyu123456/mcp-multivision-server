"""Base interface for cloud vision-language model providers.

A provider takes an ordered list of OpenAI-style ``content`` parts (interleaved
``text`` and ``image_url`` items) and returns the model's textual analysis. The
interleaving lets callers label video frames with timestamps so the model can
ground its answers in time.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class VisionResult:
    """Unified result returned by a vision provider."""

    text: str = ""
    provider: str = ""
    model: str = ""
    usage: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "provider": self.provider,
            "model": self.model,
            "usage": self.usage,
        }


class BaseVisionProvider(ABC):
    """Abstract base for all vision-language model providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Machine-readable provider name (e.g. 'openai')."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Whether the provider is fully configured (base_url + api_key + model)."""
        ...

    @abstractmethod
    async def analyze(
        self,
        parts: list[dict],
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> VisionResult:
        """Send content parts to the model and return its analysis.

        parts: ordered OpenAI content items, e.g.
            [{"type": "text", "text": "..."},
             {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}]
        """
        ...


class ProviderError(Exception):
    """Raised when a vision provider operation fails."""

    def __init__(self, provider: str, message: str):
        self.provider = provider
        self.message = message
        super().__init__(f"[{provider}] {message}")
