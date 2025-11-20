from langgraph.graph import StateGraph, END, START
from .state import DebateState
from .node import (
    debate_topic_summary,
    proponent,
    opponent,
    moderator,
    summary_debate
)

# ---- Graph wiring ----
debate_workflow = StateGraph(DebateState)
debate_workflow.add_node("debate_topic_summary", debate_topic_summary)
debate_workflow.add_node("proponent", proponent)
debate_workflow.add_node("opponent", opponent)
debate_workflow.add_node("moderator", moderator)
debate_workflow.add_node("summary_debate", summary_debate)

debate_workflow.add_edge(START, "debate_topic_summary")
debate_workflow.add_edge("summary_debate", END)

debate_graph = debate_workflow.compile(name="debate_agent")
