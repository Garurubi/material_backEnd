from __future__ import annotations

import asyncio
from typing import Any, Optional

from langgraph.graph import END, START, StateGraph
from langgraph.types import CachePolicy

from .node import (
    clarify_with_user,
    criteria_generation,
    re_question,
    supervisor_agent,
    write_research_brief,
    make_final_report
)
from .state import AgentState

# from langgraph.checkpoint.redis.aio import AsyncRedisSaver  # Original Redis checkpointer
from langgraph.checkpoint.memory import InMemorySaver

# Build the scoping workflow
main_workflow = StateGraph(AgentState)
# cache_policy=CachePolicy(
#  ttl=10800, # 3시간 동안 캐시 유지, None 이면 만료시간 없음
#  # key_func=lambda x: hash(x["x"]) # 커스텀 캐시 키 생성 함수
# )

# Add workflow nodes
main_workflow.add_node("clarify_with_user", clarify_with_user)
main_workflow.add_node("write_research_brief", write_research_brief)
main_workflow.add_node("re_question", re_question)
main_workflow.add_node("criteria_generation", criteria_generation)
main_workflow.add_node("supervisor_agent", supervisor_agent)
main_workflow.add_node("final_report", make_final_report)

# Add workflow edges
main_workflow.add_edge(START, "clarify_with_user")
main_workflow.add_edge("final_report", END)

# DB_URI = "redis://192.168.2.135:6379"  # Original Redis endpoint

_main_graph: Any | None = None
_checkpointer: Optional[InMemorySaver] = None
_compile_lock: Optional[asyncio.Lock] = None

async def get_main_graph():
    """Return compiled workflow, compiling once with async Redis checkpointer."""
    global _main_graph, _checkpointer, _compile_lock

    if _main_graph is not None:
        return _main_graph

    if _compile_lock is None:
        _compile_lock = asyncio.Lock()

    async with _compile_lock:
        if _main_graph is None:
            # Original Redis-based setup:
            # checkpointer = AsyncRedisSaver(redis_url=DB_URI)
            # await checkpointer.setup()
            # Memory-based checkpointer for local development / debugging
            checkpointer = InMemorySaver()
            _checkpointer = checkpointer
            _main_graph = main_workflow.compile(checkpointer=checkpointer)

    return _main_graph
