from __future__ import annotations

import json
import os
import re
from typing import Any

from anthropic import AsyncAnthropic

from .cache import JsonCache
from .config import Config
from .data import ResolvedQuestion
from .prompts import (
    SYSTEM_POSTERIOR,
    SYSTEM_PRIOR,
    render_evidence,
    render_question,
)


class ForecastClient:
    def __init__(self, cfg: Config) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self.cfg = cfg
        self.client = AsyncAnthropic(api_key=api_key)
        self.cache = JsonCache(cfg.cache_dir / "llm")

    async def _call(self, system: str, user: str) -> dict[str, Any]:
        payload = {
            "model": self.cfg.model,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "system": system,
            "user": user,
        }
        cached = self.cache.get(payload)
        if cached is not None:
            return cached

        last_err: Exception | None = None
        for _ in range(self.cfg.llm_max_retries + 1):
            try:
                msg = await self.client.messages.create(
                    model=self.cfg.model,
                    temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                text = "".join(
                    b.text for b in msg.content if getattr(b, "type", "") == "text"
                )
                parsed = _parse_json(text)
                value = {"raw": text, **parsed}
                self.cache.put(payload, value)
                return value
            except Exception as e:  # noqa: BLE001 — propagate after retries
                last_err = e
        raise RuntimeError(f"LLM call failed after retries: {last_err}")

    async def estimate_p_h(self, q: ResolvedQuestion) -> dict[str, Any]:
        return await self._call(SYSTEM_PRIOR, render_question(q))

    async def estimate_p_h_given_e(
        self, q: ResolvedQuestion, evidence: list[dict[str, Any]]
    ) -> dict[str, Any]:
        user = render_question(q) + "\n\nEvidence:\n" + render_evidence(evidence)
        return await self._call(SYSTEM_POSTERIOR, user)


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json(text: str) -> dict[str, Any]:
    match = _JSON_RE.search(text)
    if not match:
        raise ValueError(f"no JSON object in model output: {text!r}")
    obj = json.loads(match.group(0))
    p = obj.get("probability")
    if not isinstance(p, (int, float)):
        raise ValueError(f"probability missing or non-numeric: {obj!r}")
    p = float(p)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"probability out of range: {p}")
    return {"probability": p, "reasoning": obj.get("reasoning", "")}
