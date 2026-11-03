"""Sarvam AI LLM adapter implementing LlmPort for structured JSON and text completions."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
from typing import Any

import httpx
from sarvamai import SarvamAI

from ingest.models import IngestError


logger = logging.getLogger(__name__)


class SarvamLlm:
    """LLM client wrapping Sarvam AI chat completions."""

    def __init__(
        self,
        client: Any = None,
        api_key: str | None = None,
        model: str = "sarvam-105b-conversations",
        base_url: str = "https://api.sarvam.ai/v1",
        temperature: float | None = None,
        max_tokens: int | None = 4096,
    ) -> None:
        self.client = client
        self.api_key = api_key or os.getenv("SARVAM_API_KEY", "")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens or 4096

    def _sync_call(self, messages: list[dict[str, str]], json_mode: bool = False) -> str:
        """Synchronous chat completions call via client if available."""
        if self.client is None:
            raise ValueError("Sarvam AI client is not configured (missing api_key or client)")
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens

        response = self.client.chat.completions(**kwargs)
        if not response.choices:
            raise ValueError("No response choices returned by Sarvam LLM")
        return response.choices[0].message.content or ""

    @staticmethod
    def _parse_and_extract_json(raw: str) -> dict[str, Any]:
        """Strip markdown fences and extract structured JSON object."""
        clean = raw.strip()
        if clean.startswith("```json"):
            clean = clean[7:]
        elif clean.startswith("```"):
            clean = clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        clean = clean.strip()

        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            pass

        # Try finding outer braces { ... } if response has conversational padding
        start = clean.find("{")
        end = clean.rfind("}")
        if start != -1 and end != -1 and end > start:
            substring = clean[start : end + 1]
            return json.loads(substring)

        raise ValueError(f"Could not extract valid JSON from response: {raw[:120]}")

    async def _execute_chat(self, messages: list[dict[str, str]], json_mode: bool = False) -> str:
        """Execute chat completions using either self.client or httpx.AsyncClient."""
        if self.client is not None:
            call_fn = self.client.chat.completions
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            if self.temperature is not None:
                kwargs["temperature"] = self.temperature
            if self.max_tokens is not None:
                kwargs["max_tokens"] = self.max_tokens

            if inspect.iscoroutinefunction(call_fn):
                response = await call_fn(**kwargs)
            else:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop and loop.is_running():
                    try:
                        response = await asyncio.to_thread(call_fn, **kwargs)
                    except Exception:
                        response = call_fn(**kwargs)
                else:
                    response = call_fn(**kwargs)

            if inspect.isawaitable(response):
                response = await response

            if hasattr(response, "choices"):
                if not response.choices:
                    raise ValueError("No response choices returned by Sarvam LLM")
                choice = response.choices[0]
                if hasattr(choice, "message"):
                    return choice.message.content or ""
                return str(choice)
            elif isinstance(response, dict):
                return response.get("choices", [{}])[0].get("message", {}).get("content", "")
            return str(response)

        # Direct HTTP call via httpx
        async with httpx.AsyncClient(timeout=30.0) as http_client:
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
            }
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
            if self.temperature is not None:
                payload["temperature"] = self.temperature
            if self.max_tokens is not None:
                payload["max_tokens"] = self.max_tokens

            res = await http_client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "api-subscription-key": self.api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            res.raise_for_status()
            data = res.json()
            return data["choices"][0]["message"]["content"] or ""

    async def complete_json(self, system: str, user: str) -> dict[str, Any]:
        """Request structured JSON completion with a single automatic repair attempt."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            raw = await self._execute_chat(messages, json_mode=True)
            try:
                return self._parse_and_extract_json(raw)
            except Exception as parse_err:
                # One automatic repair attempt
                repair_messages = list(messages) + [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            f"The previous output was not valid JSON ({parse_err}). "
                            "Return only a valid JSON object matching the schema without markdown fences or commentary."
                        ),
                    },
                ]
                repair_raw = await self._execute_chat(repair_messages, json_mode=True)
                return self._parse_and_extract_json(repair_raw)
        except Exception as e:
            raise IngestError("classify", e) from e

    async def complete_text(self, system: str, user: str, stage: str = "classify") -> str:
        """Request markdown text completion."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            raw = await self._execute_chat(messages, json_mode=False)
            return raw.strip()
        except IngestError:
            raise
        except Exception as e:
            raise IngestError(stage, e) from e

