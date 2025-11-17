from dotenv import load_dotenv
from typing_extensions import Literal
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from .state import (
    DebateState,
    DebateUtteranceLLM,
    DebateUtterance,
    Role,
    ModeratorDecision,
    Decision,
    DebateSummary
)
from langgraph.types import Command
from .prompt import (
    debate_turn_jinja_format,
    proponent_prompt,
    opponent_prompt,
    moderator_prompt,
    hypothesis_summary_prompt,
    debate_summarize_prompt
)
from jinja2 import Template
import os

model = init_chat_model(model=os.getenv("DEBATE_MODEL"))

# 토론 이력 포맷 생성
def make_debate_history(turns):
    return Template(debate_turn_jinja_format).render(turns = turns)

def summary_hypothesis(state: DebateState) -> Command[Literal["proponent"]]:
    """토론 주제(가설) 요약 노드"""
    response = model.invoke([
        HumanMessage(content=hypothesis_summary_prompt.format(hypothesis=state.get("hypothesis", "")))          
    ])

    return Command(
        goto="proponent",
        update={"topic": response.content.strip()}
    )

def proponent(state: DebateState) -> Command[Literal["opponent"]]:
    """찬성 토론자 역할을 수행하는 노드"""
    # debate_history 정리
    debate_history = make_debate_history(turns = state.get("turns", []))

    structured_output_model = model.with_structured_output(DebateUtteranceLLM)
    response = structured_output_model.invoke([
        HumanMessage(content=proponent_prompt.format(
            topic=state.get("topic", ""),
            debate_history=debate_history
        ))
    ])
    turn_id = state.get("turn_id", 0) + 1
    turn_result = DebateUtterance(**response.model_dump(), turn=turn_id, role=Role.PROPONENT)

    return Command(
        goto="opponent",
        update={
            "turn_id": turn_id,
            "turns": [turn_result]
        }
    )

def opponent(state: DebateState) -> Command[Literal["proponent", "moderator"]]:
    """반대 토론자 역할을 수행하는 노드"""
    debate_history = Template(debate_turn_jinja_format).render(turns = state.get("turns", []))

    structured_output_model = model.with_structured_output(DebateUtteranceLLM)
    response = structured_output_model.invoke([
        HumanMessage(content=opponent_prompt.format(
            topic=state.get("topic", ""),
            debate_history=debate_history
        ))
    ])
    turn_id = state["turn_id"] + 1
    turn_result = DebateUtterance(**response.model_dump(), turn=turn_id, role=Role.OPPONENT)

    # proponent, opponent가 최소 각자 3번씩 발언한 후에 사회자에게 판단 넘김
    if turn_id >= 6:
        return Command(
            goto="moderator",
            update={
                "turn_id": turn_id,
                "turns": [turn_result]
            }
        )
    else:
        return Command(
            goto="proponent",
            update={
                "turn_id": turn_id,
                "turns": [turn_result]
            }
        )

def moderator(state: DebateState) -> Command[Literal["proponent", "summary_debate"]]:
    """사회자 역할을 수행하는 노드"""
    debate_history = Template(debate_turn_jinja_format).render(turns = state.get("turns", []))

    structured_output_model = model.with_structured_output(ModeratorDecision)
    response = structured_output_model.invoke([
        HumanMessage(content=moderator_prompt.format(
            debate_history=debate_history
        ))
    ])

    if response.decision == Decision.CONTINUE and state["turn_id"] < 12:
        return Command(
            goto="proponent",
            update={"moderator_decisions": response}
        )
    elif response.decision == Decision.END or state["turn_id"] >= 12:
        return Command(
            goto="summary_debate",
            update={"moderator_decisions": response}
        )

def summary_debate(state: DebateState) -> dict:
    """토론 결론을 정리하고 노드"""
    debate_history = Template(debate_turn_jinja_format).render(turns = state.get("turns", []))
    structured_output_model = model.with_structured_output(DebateSummary)
    response = structured_output_model.invoke([
        HumanMessage(content=debate_summarize_prompt.format(
            debate_history=debate_history
        ))
    ])

    return {"debate_summary": response}

# def candidate_select(state: DebateState) -> dict:
#     """토론 결과를 기반으로 후보 선정 노드"""
#     search_results = state.get("search_results")
#     candidate = search_results.get("search_reactions") + "\n" + search_results.get("search_from_mongoDB")
#     response = model.invoke([
#         HumanMessage(content=candidate_selection_prompt.format(
#             debate_summary_json=state.get("debate_summary", ""),
#             candidate_list_json=candidate
#         ))
#     ])

#     return {"candidate_selection": response.content.strip()}