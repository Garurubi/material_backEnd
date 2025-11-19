import os
from dotenv import load_dotenv
load_dotenv()
from pydantic import BaseModel

from fastapi import FastAPI, Query, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from langchain_core.messages import HumanMessage

from langfuse import Langfuse, propagate_attributes
from langfuse.langchain import CallbackHandler

from langgraph.types import Command

from graph_stream.streamer import sse_event_generator
from SupervisorAgent.main_graph import get_main_graph

import logging.config
from logging_config import LOGGING_CONFIG

# uvicorn access log 설정
# logging.config.dictConfig(LOGGING_CONFIG) 

# langfuse 설정
langfuse = Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host="http://192.168.2.134:3000",
)
langfuse_handler = CallbackHandler()

origins = [
    "http://localhost:8501",
    "http://192.168.2.135:8501"
    "http://220.89.167.202:53984"
]

# FastAPI
app = FastAPI(title="Material Agent", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MaterialAgentRequest(BaseModel):
    query : str
    conversation_id : str

class MaterialAgentResponse(BaseModel):
    response_str: str

@app.post("/material_chat", response_model=MaterialAgentResponse)
async def material_chat(request: MaterialAgentRequest) -> MaterialAgentResponse:
    """
    사용자의 질문을 받아서 스텝별로 수행
    """
    config = {"configurable": {"thread_id": request.conversation_id}, "callbacks": [langfuse_handler]}
    graph = await get_main_graph()

    # 새로운 대화 세션인 경우 초기화 작업 수행
    has_history = False
    async for _ in graph.aget_state_history(config, limit=1):
        has_history = True
        break
    
    with langfuse.start_as_current_span(name="langgraph-call"):
        # Propagate session_id to all observations
        with propagate_attributes(session_id=request.conversation_id):
            if not has_history:
                initial_result = await graph.ainvoke(
                    {"messages" : [HumanMessage(content=request.query)]}, 
                    config = config
                )
                
                interrupt_result = initial_result["__interrupt__"][-1].value
                return MaterialAgentResponse(response_str=interrupt_result)
            else:
                resume_result = await graph.ainvoke(Command(resume=request.query), config=config)
                if "__interrupt__" in resume_result:
                    interrupt_result = resume_result["__interrupt__"][-1].value
                    return MaterialAgentResponse(response_str=interrupt_result)
                else:
                    return MaterialAgentResponse(response_str=resume_result["final_report"])

@app.get("/material_chat/{conversation_id}")
async def material_chat_stream(
    conversation_id: str,
    # request: Request,
    query: str = Query(..., description="Answer to resume interrupted workflow")
    ) -> StreamingResponse:
    """Stream in SSE(Server-Sent Events) mode"""

    config = {"configurable": {"thread_id": conversation_id}, "callbacks": [langfuse_handler]}
    graph = await get_main_graph()

    print(conversation_id, "/", query)
    async def event_gen():
        with langfuse.start_as_current_span(name="langgraph-call"):
            with propagate_attributes(session_id=conversation_id):
                async for event in sse_event_generator(graph, {"configurable": {"thread_id": conversation_id}, "callbacks": [langfuse_handler]}, query):
                    yield event
    
    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("FASTAPI_HOST", 8593)))
