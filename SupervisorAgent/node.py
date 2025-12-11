from datetime import datetime
from typing_extensions import Literal
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, get_buffer_string
from langgraph.types import Command
from .prompts import (
    clarify_with_user_instructions, 
    transform_messages_into_research_topic_prompt,
    paper_system_instructions,
    paper_clarify_instructions,
    clarify_with_user_and_paper,
    evaluate_criteria_prompt,
    feedback_intent_prompt,
    sac_supervisor_prompt,
    perovskite_supervisor_prompt,
    final_anwser_prompt
)
from .state import (
    AgentState, 
    ClarifyWithUser, 
    ResearchQuestion,
    ClarifyPaper,
    ClassifiedPaperWithUser,
    Criteria,
    UserFeedbackIntent,
    RouteResponse,
    Domain
)
from jinja2 import Template
from DataCollectAgent import build_data_collect_graph
from HypothesisAgent import hypothesis_workflow
from DebateAgent import debate_graph
from PerovskiteSearchAgent import perovskite_workflow
import os
from langgraph.types import interrupt


model = init_chat_model(model=os.getenv("SUPERVISOR_MODEL"), temperature=0.0)

async def clarify_with_user(state: AgentState) -> Command[Literal["re_question"]]:
    # 질문만 들어오는 경우
    if state.get("messages") and not state.get("pdfs"):
        structured_output_model = model.with_structured_output(ClarifyWithUser)
        response = await structured_output_model.ainvoke([
            HumanMessage(content=clarify_with_user_instructions.format(
                messages=get_buffer_string(messages=state["messages"])
            ))
        ])

        return Command(
            goto="re_question",
            update={"messages": [AIMessage(content=response.question)]
                    ,"classified_input": response}
        )
    # PDF만 들어오는 경우
    elif state.get("pdfs") and not state.get("messages"):
        pdfs = state.get("pdfs")
        papers_str = "\n".join([f"Paper_id: {pdf['id']}\nTitle: {pdf['title']}\nAbstract: {pdf['abstract']}\n" for pdf in pdfs])
        structured_output_model = model.with_structured_output(ClarifyPaper)
        response = await structured_output_model.ainvoke([
            SystemMessage(content=paper_system_instructions),
            HumanMessage(content=paper_clarify_instructions.format(
                papers=papers_str
            ))
        ])
    # 질문과 PDF가 모두 들어오는 경우
    elif state.get("messages") and state.get("pdfs"):
        pdfs = state.get("pdfs")
        papers_str = "\n".join([f"Paper_id: {pdf['id']}\nTitle: {pdf['title']}\nAbstract: {pdf['abstract']}\n" for pdf in pdfs])
        structured_output_model = model.with_structured_output(ClassifiedPaperWithUser)
        response = await structured_output_model.ainvoke([
            SystemMessage(content=paper_system_instructions),
            HumanMessage(content=clarify_with_user_and_paper.format(
                messages=get_buffer_string(messages=state["messages"]),
                papers=papers_str
            ))
        ])

    # 분류 결과를 state에 반영
    classified_result = {p.paper_id:p.classification for p in response.classifyed_papers 
                         if p.classification in ["Single-Atom Catalysts", "Photovoltaic Tandem Devices", "Biopolymer Materials", "Other"]}
    for p in state.get("pdfs", []):
        if (clsf := classified_result.get(p["id"])) is not None:
            p["classification"] = clsf

    return Command(
        goto="re_question",
        update={"messages": [AIMessage(content=response.question)],
                "classified_input": response}
    )

async def write_research_brief(state: AgentState) -> Command[Literal["criteria_generation"]]:
    structured_output_model = model.with_structured_output(ResearchQuestion)

    tmpl = Template(transform_messages_into_research_topic_prompt)

    response = await structured_output_model.ainvoke([
        HumanMessage(content=tmpl.render(
            messages=get_buffer_string(state.get("messages", [])),
            papers="\n".join([f"{pdf.title}\n{pdf.abstract}\n" for pdf in state.get("pdfs", [])])
        ))
    ])

    if isinstance(state.get("classified_input"), ClarifyWithUser):
        classified = state.get("classified_input")
        if classified.query_domain == Domain.PV_TANDEM:
            return Command(
                goto="supervisor_agent",
                update={"research_brief": response.research_brief}
            )

    # Update state with generated research brief and pass it to the supervisor
    return Command(
        goto="criteria_generation",
        update={"research_brief": response.research_brief,
        "supervisor_messages": [HumanMessage(content=f"{response.research_brief}")]}
    )

async def re_question(state: AgentState)-> Command[Literal["write_research_brief", "supervisor_agent"]]:
    ai_message = state.get("messages", [AIMessage(content="")])[-1]
    
    # 유저에게 질문한후 피드백 받기
    user_feedback = interrupt(ai_message.content)
    # 평가기준 피드백
    if state.get("criteria"):
        # 사용자의 피드백에서 평가기준을 승인
        structured_output_model = model.with_structured_output(UserFeedbackIntent)
        response = await structured_output_model.ainvoke([
            HumanMessage(content=feedback_intent_prompt.format(
                user_feedback=user_feedback,
            ))
        ])

        if response.feedback_intent.lower() == "approve":
            return Command(
                goto="supervisor_agent",
                update={"messages": [HumanMessage(content=user_feedback)]}
            )
        else:
            return Command(
                goto="criteria_generation",
                update={"messages": [HumanMessage(content=user_feedback)],
                        "user_feedback": user_feedback}
            )
    # 도메인 피드백(사용자의 피드백을 여기서 반영할 필요 없음 - write_research_brief에서 반영)   
    else:
        return Command(
            goto="write_research_brief",
            update={"messages": [HumanMessage(content=user_feedback)]}
        )

async def criteria_generation(state: AgentState) -> Command[Literal["re_question"]]:
    user_feedback = state.get("user_feedback", "")
    # 평가기준 생성(research_brief로 생성)
    structured_output_model = model.with_structured_output(Criteria)

    tmpl = Template(evaluate_criteria_prompt)

    response = await structured_output_model.ainvoke([
        HumanMessage(content=tmpl.render(
            research_brief = state.get("research_brief", ""),
            user_feedback = user_feedback
        ))
    ])
    # 표형식으로 가중치 기준 표현하기 
    criteria_description = {
        "activity" : "촉매의 반응 성능",
        "stability" : "장기 안정성",
        "synthesis" : "합성 난이도 및 재현성",
        "cost" : "재료 및 공정 비용",
        "evidence" : "문헌·실험 데이터 신뢰도",
        "ml_lit_agree" : "머신러닝 예측과 기존 연구의 일치성"
    }
    criteria_table = "| 항목 | 가중치 | 설명 |\n|------|----|--------|\n"\
        + '\n'.join([f"| {w} | {v} | {criteria_description[w]} |" for w, v in response.weight.model_dump().items()])    
    criteria_str = criteria_table + f"\n\n{response.feedback_question}"
    
    # 평가기준에 대한 피드백을 한번이라도 받았으면 다시 피드백 받지 않음
    if user_feedback:
        return Command(
            goto="supervisor_agent",
            update={"criteria": response}
        )
    else:
        return Command(
            goto="re_question",
            update={"messages": [AIMessage(content=criteria_str)]
                    ,"criteria": response}
        )
    
async def supervisor_agent(state: AgentState):
    # 무한 재귀 방지(하위 에이전트 호출 횟수 제한)
    supervisor_recursion = state.get("supervisor_recursion", 0)
    if supervisor_recursion >= 3:
        return {"next_agents": ["final_report"]}
    
    # pdf가 있다면 데이터 추출 agent 호출
    if state.get("pdfs"):
        pass
    
    classified_input = state.get("classified_input")
    if classified_input.query_domain == Domain.SAC:
        # sac supervisor agent
        tmpl = Template(sac_supervisor_prompt)
        structured_output_model = model.with_structured_output(RouteResponse)
        response = await structured_output_model.ainvoke([
            HumanMessage(content=tmpl.render(
                query = state.get("messages")[0].content,
                search_results = state.get("search_results"),
                hypothesis_results = state.get("hypothesis_results"),
                debate_summary = state.get("debate_summary"),
            ))
        ])
    elif classified_input.query_domain == Domain.PV_TANDEM:
        # perovskite supervisor agent
        tmpl = Template(perovskite_supervisor_prompt)
        structured_output_model = model.with_structured_output(RouteResponse)
        response = await structured_output_model.ainvoke([
            HumanMessage(content=tmpl.render(
                query = state.get("messages")[0].content,
                search_results = state.get("search_results")
            ))
        ])
    
    return {"next_agents": [response.next_agent],
            "supervisor_recursion": supervisor_recursion + 1}

async def sac_search_agent(state: AgentState):
    data_collect_graph = build_data_collect_graph()
    data_collect_state = await data_collect_graph.ainvoke(
        {"requirements": state.get("messages")[0].content},
        config={"run_name": "data_collect_agent"}
    )
    return Command(
        goto="supervisor_agent",
        update={"search_results" : {"sac_search_results" : data_collect_state.get("response", [])}}
    )

async def hypothesis_agent(state: AgentState):
    hypothesis_state = await hypothesis_workflow.ainvoke(
        {
            "hypothesis_query": state.get("messages")[0].content,
            "step_timestamp": {},
        },
        config={"run_name": "hypothesis_agent"}
    )

    if proposal_output:=hypothesis_state.get("proposal_output"):
        hypothesis_result = []
        for val1 in proposal_output.values():
            for val2 in val1.values():
                hypothesis_result.append(val2.get("output"))

        return Command(
            goto="supervisor_agent",
            update={"hypothesis_results" : hypothesis_result}
        )
    else:
        return Command(
            goto="supervisor_agent",
            update={"hypothesis_results" : []}
        )

async def debate_agent(state: AgentState):
    debate_state = await debate_graph.ainvoke(
        {
            "user_query" : state.get("messages")[0].content,
            "hypothesis_results" : state.get("hypothesis_results", []),
            "search_results" : state.get("search_results", {}),
        },
        config={"run_name": "debate_agent"}
    )
    
    return Command(
        goto="supervisor_agent",
        update={"debate_summary" : debate_state.get("debate_summary")}
    )

async def perovskite_search_agent(state: AgentState):
    perovskite_state = await perovskite_workflow.ainvoke(
        {
            "messages" : [state.get("messages")[0].content],
        },
        config={"run_name": "perovskite_search_agent"}
    )
    
    return Command(
        goto="supervisor_agent",
        update={"search_results" : {"perovskite_search_results" : perovskite_state["messages"][-1].content}}
    )

async def make_final_report(state: AgentState):
    tmpl = Template(final_anwser_prompt)
    response = await model.ainvoke([
        HumanMessage(content=tmpl.render(
            query = state.get("messages")[0].content,
            criteria = state.get("criteria").weight if state.get("criteria") else "",
            search_results = state.get("search_results"),
            hypothesis_results = state.get("hypothesis_results"),
            debate_summary = state.get("debate_summary"),
        ))
    ])

    return {"final_report": response.content}