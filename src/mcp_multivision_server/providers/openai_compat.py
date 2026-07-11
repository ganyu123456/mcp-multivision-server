"""OpenAI-compatible vision provider.

Works with any chat-completions endpoint that accepts multimodal ``image_url``
content parts: OpenAI (gpt-4o), 通义千问VL (DashScope compatible-mode),
智谱 GLM-4V, 豆包 (Volcengine Ark), local servers (Ollama / LM Studio), etc.
Configure via base_url + api_key + model.
"""

import asyncio

import httpx

from .base import BaseVisionProvider, ProviderError, VisionResult


class OpenAICompatProvider(BaseVisionProvider):
    def __init__(
        self,
        base_url: str = "",
        api_key: str = "",
        model: str = "",
        *,
        timeout: float = 60.0,
        max_retries: int = 3,
        default_max_tokens: int = 1024,
        default_temperature: float = 0.2,
    ):
        self._base_url = (base_url or "").rstrip("/")
        self._api_key = api_key or ""
        self._model = model or ""
        self._timeout = timeout
        self._max_retries = max(1, max_retries)
        self._default_max_tokens = default_max_tokens
        self._default_temperature = default_temperature

    @property
    def provider_name(self) -> str:
        return "openai"

    def is_available(self) -> bool:
        return bool(self._base_url and self._api_key and self._model)

    async def analyze(
        self,
        parts: list[dict],
        *,
        max_tokens=None,
        temperature=None,
    ) -> VisionResult:
        if not self.is_available():
            raise ProviderError(
                self.provider_name,
                "Provider not configured. Set MCP_VISION_BASE_URL, MCP_VISION_API_KEY and MCP_VISION_MODEL.",
            )

        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": parts}],
            "max_tokens": int(max_tokens if max_tokens is not None else self._default_max_tokens),
            "temperature": float(
                temperature if temperature is not None else self._default_temperature
            ),
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self._base_url}/chat/completions"

        last_error = None
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for attempt in range(self._max_retries):
                try:
                    resp = await client.post(url, json=payload, headers=headers)
                except (httpx.TimeoutException, httpx.TransportError) as e:
                    last_error = f"request failed: {e}"
                else:
                    if resp.status_code == 200:
                        return self._parse(resp.json())
                    # 4xx (auth/bad request) 不重试
                    if 400 <= resp.status_code < 500:
                        raise ProviderError(
                            self.provider_name,
                            f"HTTP {resp.status_code}: {self._error_snippet(resp)}",
                        )
                    last_error = f"HTTP {resp.status_code}: {self._error_snippet(resp)}"

                if attempt < self._max_retries - 1:
                    await asyncio.sleep(2 ** attempt)

        raise ProviderError(self.provider_name, last_error or "unknown error")

    def _parse(self, data: dict) -> VisionResult:
        try:
            text = data["choices"][0]["message"]["content"]
            if isinstance(text, list):
                # 某些实现返回 content 数组，拼接其中的文本
                text = "".join(
                    part.get("text", "") for part in text if isinstance(part, dict)
                )
        except (KeyError, IndexError, TypeError) as e:
            raise ProviderError(self.provider_name, f"unexpected response shape: {e}")
        return VisionResult(
            text=text or "",
            provider=self.provider_name,
            model=self._model,
            usage=data.get("usage"),
        )

    @staticmethod
    def _error_snippet(resp: httpx.Response) -> str:
        try:
            body = resp.json()
            err = body.get("error")
            if isinstance(err, dict):
                return str(err.get("message", err))
            return str(err or body)[:300]
        except Exception:
            return resp.text[:300]
