# -*- coding: utf-8 -*-
"""
LLMClient wraps OpenAI Chat Completions for agents.
- Standardizes token/time logging (English keys)
- Carries execution context (agent, base, level, cid, cluster_id)
"""
from __future__ import annotations
import os

from typing import Optional

from ..config.model_config import get_agent_llm

try:
    from openai import AsyncOpenAI
except Exception:  
    AsyncOpenAI = None  

# 비동기
class LLMClient:
    def __init__(
        self,
        agent_key: str,
        level: Optional[int] = None,
        cid: Optional[int] = None,
        cluster_id: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        top_p: float = 1.0,
        max_tokens: int = 512,
        api_key_env: str = "OPENAI_API_KEY",
    ) -> None:
        self.agent = agent_key
        self.level = level
        self.cid = cid
        self.cluster_id = cluster_id or ""

        cfg = get_agent_llm(self.agent)
        self.model = model or cfg.MODEL
        self.temperature = float(cfg.TEMPERATURE) if cfg.TEMPERATURE is not None else float(temperature)
        self.top_p = float(cfg.TOP_P) if cfg.TOP_P is not None else float(top_p)
        # No max token limit is sent to the API
        self.max_tokens = None
        self._api_key = os.getenv(api_key_env, "")
        self._client = AsyncOpenAI(api_key=self._api_key) if (AsyncOpenAI and self._api_key) else None
    # ---- calls ----
    async def chat(self, system: str, user: str, step: str) -> str:
        """Return plain text; log tokens/time with unified schema."""
        if self._client is None:
            # offline fallback
            txt = "(offline) " + (user[:200] + ("..." if len(user) > 200 else ""))
            
            return txt


        resp = await self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )

        txt = (resp.choices[0].message.content or "").strip()

        return txt