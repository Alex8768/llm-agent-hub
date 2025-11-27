from __future__ import annotations

import os

from openai import APIError, APITimeoutError, AuthenticationError, OpenAI, RateLimitError

from .base import GenerateRequest, GenerateResponse, LLMEngine
from .registry import register


class OpenAIEngine(LLMEngine):
    name = "api"

    def _client(self) -> OpenAI:
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key:
            raise RuntimeError("OPENAI_API_KEY missing")
        return OpenAI(api_key=key)

    def health(self) -> bool:
        return bool(os.getenv("OPENAI_API_KEY", "").strip())

    def generate(self, req: GenerateRequest) -> GenerateResponse:
        client = self._client()
        model_name = req.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        messages = []
        if req.system:
            messages.append({"role": "system", "content": req.system})
        messages.append({"role": "user", "content": req.prompt})
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=(req.options or {}).get("temperature", 0.7),
            )
        except AuthenticationError as exc:
            raise RuntimeError("openai_auth_error: invalid or missing OPENAI_API_KEY") from exc
        except RateLimitError as exc:
            raise RuntimeError("openai_ratelimit: usage exceeded or throttled") from exc
        except APITimeoutError as exc:
            raise RuntimeError("openai_timeout") from exc
        except APIError as exc:
            raise RuntimeError(f"openai_api_error: {exc}") from exc
        text = (response.choices[0].message.content or "").strip()
        return GenerateResponse(text=text, raw=response.model_dump())


register(OpenAIEngine())
