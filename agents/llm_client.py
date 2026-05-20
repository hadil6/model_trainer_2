from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from openai import AzureOpenAI

from core.config import Settings, get_settings

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = self._build_client()

    def _build_client(self) -> AzureOpenAI:
        if not self.settings.azure_api_key:
            raise ValueError("Missing AZURE_API_KEY in .env")
        endpoint = (self.settings.azure_endpoint or "").strip()
        if not endpoint:
            raise ValueError("Missing AZURE_ENDPOINT in .env")
        return AzureOpenAI(
            api_version=self.settings.azure_api_version,
            azure_endpoint=endpoint,
            api_key=self.settings.azure_api_key,
            timeout=self.settings.llm_timeout_seconds,
            max_retries=self.settings.llm_max_retries,
        )

    def _resolved_model(self, override: str | None = None) -> str:
        return override or self.settings.deployment_name

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        response = self._client.chat.completions.create(
            model=self._resolved_model(model),
            messages=messages,
            max_completion_tokens=max_tokens or self.settings.llm_max_tokens,
        )
        content = (response.choices[0].message.content or "").strip()
        if not content:
            raise ValueError("Empty content returned from LLM")
        return content

    def complete(self, system: str, user: str, model: str | None = None) -> str:
        return self.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            model=model,
        )

    def call_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str | None = None,
    ) -> dict:
        """Call LLM with tool definitions (function calling). Returns assistant message dict."""
        all_messages: list[dict] = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        response = self._client.chat.completions.create(
            model=self._resolved_model(),
            messages=all_messages,
            tools=tools,
            tool_choice="auto",
            max_completion_tokens=self.settings.llm_max_tokens,
        )
        msg = response.choices[0].message
        return {
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in (msg.tool_calls or [])
            ],
        }


@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    return LLMClient()
