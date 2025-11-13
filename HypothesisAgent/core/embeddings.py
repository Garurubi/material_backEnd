import os

import asyncio

import numpy as np

from typing import List
from tqdm import tqdm
from openai import AsyncOpenAI


def _client() -> AsyncOpenAI:
    return AsyncOpenAI(base_url=os.getenv("QWEN_EMBED_ADDR"), api_key="")


async def _embed_one(client: AsyncOpenAI, text: str) -> List[float]:
    """단일 문장 임베딩"""
    text = (text or "").strip()
    if not text:
        return [0.0] * 1536
    resp = await client.embeddings.create(
        model=os.getenv("QWEN_EMBED_MODEL"),
        input=text
    )
    return resp.data[0].embedding

async def _embed(texts: List[str], concurrency: int = 8) -> np.ndarray:
    client = _client()
    sem = asyncio.Semaphore(concurrency)

    async def sem_task(t):
        async with sem:
            return await _embed_one(client, t)

    tasks = [sem_task(t) for t in texts]

    results = []
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="[Emb] progress", unit="doc"):
        await f  # 단순 진행 표시용

    # 결과는 순서대로 다시 gather (순서 유지)
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 예외 처리 (None 또는 예외를 [0]*1536로 대체)
    vecs = []
    for r in results:
        if isinstance(r, Exception):
            vecs.append([0.0] * 1536)
        else:
            vecs.append(r)

    return np.array(vecs, dtype="float32")

async def ensure_embeddings_cached(texts: List[str]) -> np.ndarray:
    """Load if npy exists; else compute per-document embeddings and save.
       Also appends a metric line with model, elapsed, and prompt_tokens (estimate).
    """
    X = await _embed(texts)

    return X