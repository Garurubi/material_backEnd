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
    feedback_intent_prompt
)
from .state import (
    AgentState, 
    ClarifyWithUser, 
    ResearchQuestion,
    ClarifyPaper,
    ClassifiedPaperWithUser,
    Criteria,
    UserFeedbackIntent
)
from jinja2 import Template
from DataCollectAgent import build_data_collect_graph
from HypothesisAgent import hypothesis_workflow
from DebateAgent import debate_graph
import os
import asyncio
from langgraph.types import interrupt

model = init_chat_model(model=os.getenv("SUPERVISOR_MODEL"), temperature=0.0)

def clarify_with_user(state: AgentState) -> Command[Literal["re_question"]]:
    # 질문만 들어오는 경우
    if state.get("messages") and not state.get("pdfs"):
        structured_output_model = model.with_structured_output(ClarifyWithUser)
        response = structured_output_model.invoke([
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
        response = structured_output_model.invoke([
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
        response = structured_output_model.invoke([
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

def write_research_brief(state: AgentState) -> Command[Literal["criteria_generation"]]:
    structured_output_model = model.with_structured_output(ResearchQuestion)

    tmpl = Template(transform_messages_into_research_topic_prompt)

    response = structured_output_model.invoke([
        HumanMessage(content=tmpl.render(
            messages=get_buffer_string(state.get("messages", [])),
            papers="\n".join([f"{pdf.title}\n{pdf.abstract}\n" for pdf in state.get("pdfs", [])])
        ))
    ])

    # Update state with generated research brief and pass it to the supervisor
    return Command(
        goto="criteria_generation",
        update={"research_brief": response.research_brief,
        "supervisor_messages": [HumanMessage(content=f"{response.research_brief}")]}
    )

def re_question(state: AgentState)-> Command[Literal["write_research_brief", "supervisor_agent"]]:
    ai_message = state.get("messages", [AIMessage(content="")])[-1]
    
    # 유저에게 질문한후 피드백 받기
    user_feedback = interrupt(ai_message.content)
    # 평가기준 피드백
    if state.get("criteria"):
        # 사용자의 피드백에서 평가기준을 승인
        structured_output_model = model.with_structured_output(UserFeedbackIntent)
        response = structured_output_model.invoke([
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

def criteria_generation(state: AgentState) -> Command[Literal["re_question"]]:
    user_feedback = state.get("user_feedback", "")
    # 평가기준 생성(research_brief로 생성)
    structured_output_model = model.with_structured_output(Criteria)

    tmpl = Template(evaluate_criteria_prompt)

    response = structured_output_model.invoke([
        HumanMessage(content=tmpl.render(
            research_brief = state.get("research_brief", ""),
            user_feedback = user_feedback
        ))
    ])
    criteria_str = "Weight: \n" \
        + str({'\n'.join([f"{w}={v}" for w, v in response.weight.model_dump().items()])})\
        + f"\n\n{response.feedback_question}"
    
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
    
def supervisor_agent(state: AgentState):
    # pdf가 있다면 데이터 추출 agent 호출
    if state.get("pdfs"):
        pass
    
    data_collect_graph = build_data_collect_graph()
    data_collect_state = await data_collect_graph.ainvoke(
        {"requirements": state.get("messages")[0].content},
        config={"run_name": "data_collect_agent"}
    )

    hypothesis_state = await hypothesis_workflow.ainvoke(
        {
            "hypothesis_query": state.get("messages")[0].content,
            "step_timestamp": {},
        },
        config={"run_name": "hypothesis_agent"}
    )

    if hypothesis_state.get("proposal_output"):
        # 토론 agent 호출
        debate_state = debate_graph.invoke(
            {
                "hypothesis": hypothesis_state["proposal_output"],
                "search_results": data_collect_state.get("response", {})
            },
            config={"run_name": "debate_agent"}
        )

        return {"final_report": debate_state.get("debate_summary")}
    else:
        return {"final_report": "No hypothesis proposal generated."}


    # # 데이터 수집 agent 호출
    # data_collect_graph = build_data_collect_graph()
    # collect_coro = data_collect_graph.ainvoke(
    #     {"requirements": state.get("research_brief")},
    #     config={"run_name": "data_collect_agent"}
    # )
    # # 가설생성 agent 호출
    # hypo_coro = hypothesis_workflow.ainvoke(
    #     {
    #         "hypothesis_query": state.get("research_brief"),
    #         "step_timestamp": {},
    #     },
    #     config={"run_name": "hypothesis_agent"}
    # )
    # # 데이터 수집 에이전트, 가설생성 에이전트 동시 호출한후 대기
    # data_collect_state, hypothesis_state = await asyncio.gather(
    #     collect_coro, hypo_coro, return_exceptions=False
    # )

    # if hypothesis_state.get("proposal_output"):
    #     # 토론 agent 호출
    #     debate_state = debate_graph.invoke(
    #         {
    #             "hypothesis": hypothesis_state["proposal_output"],
    #             "search_results": data_collect_state.get("response", {})
    #         },
    #         config={"run_name": "debate_agent"}
    #     )

    #     return {"final_report": debate_state.get("debate_summary")}
    # else:
    #     return {"final_report": "No hypothesis proposal generated."}