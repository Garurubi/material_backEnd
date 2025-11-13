
# -*- coding: utf-8 -*-
from __future__ import annotations
"""LLMClient (proposal-review edition)
- Reads API key & defaults from config.api_config and config.model_config.
- Logs tokens/time via core.io_utils_pr.append_metric (global JSONL).
- Provides chat() and chat_with_files() for PDF attachments.
"""
import os
from typing import Optional

try:
    from openai import AsyncOpenAI
except Exception:
    AsyncOpenAI = None

from ..config import model_config as MC


class LLMClient:
    def __init__(
        self,
        agent_key: str,
        cluster_id: Optional[str] = None,
        base_ascii: str = "proposal_review_base",
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        self.agent = agent_key
        self.cluster_id = cluster_id or ""
        self.base_ascii = base_ascii
        cfg = MC.get_agent_llm(agent_key)
        self.model = model or cfg.MODEL
        self.temperature = float(temperature if temperature is not None else (cfg.TEMPERATURE if cfg.TEMPERATURE is not None else 0.2))
        self.top_p = float(top_p if top_p is not None else (cfg.TOP_P if cfg.TOP_P is not None else 1.0))
        self.max_tokens = int(max_tokens if max_tokens is not None else (cfg.MAX_TOKENS if cfg.MAX_TOKENS is not None else 4096))
       
        api_key = os.getenv("OPENAI_API_KEY", "")
        self._client = AsyncOpenAI(api_key=api_key) if (AsyncOpenAI and api_key) else None

    async def chat(self, system: str, user: str, step: str) -> str:
        if self._client is None:
            txt = "(offline) " + (user[:200] + ("..." if len(user) > 200 else ""))
    
            return txt

        kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }

        # Do NOT include any token limit for evaluation or judge agents
        if self.agent not in ("proposal_evaluation", "proposal_judge"):
            kwargs["max_tokens"] = self.max_tokens

        resp = await self._client.chat.completions.create(**kwargs)
        txt = (resp.choices[0].message.content or "").strip()

        return txt
