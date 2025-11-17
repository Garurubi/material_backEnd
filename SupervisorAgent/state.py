import operator
from typing_extensions import Optional, Annotated, Sequence, TypedDict, Literal, List, Dict
from enum import Enum
from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from DebateAgent import DebateSummary

# ===== STRUCTURED OUTPUT SCHEMAS =====
class Domain(str, Enum):
    SAC = "Single-Atom Catalysts"
    PV_TANDEM = "Photovoltaic Tandem Devices"
    BIOPOLYMER = "Biopolymer Materials"
    OTHER = "Other"

# 유저 질의만 있는 경우
class ClarifyWithUser(BaseModel):
    """Classification schema for user questions"""
    
    query_intent: str = Field(
        description="Summary of query intentions for user messages",
        examples=["단일 원자 촉매를 이용한 CO2 환원 반응 분석 요청"]
    )
    query_domain: Domain = Field(
        description="Domain classification of user messages",
        examples=[Domain.SAC]
    )
    question: str = Field(
        description="Final question to check with the user",
        examples=["단일 원자 촉매 분야로 분석을 진행할까요?"]
    )

class ClassifiedPaper(BaseModel):
    """Individual Paper Classification Results Schema"""

    paper_id: str = Field(
        description="paper id",
    )
    classification: Domain = Field(
        description="Classification to which the paper belongs",
        examples=[Domain.SAC]
    )

# pdf 논문만 있는 경우
class ClarifyPaper(BaseModel):
    """Schema for multiple thesis classifications"""
    
    classifyed_papers: List[ClassifiedPaper] = Field(
        description="a list of categorized papers"
    )
    question: str = Field(
        description="Final question to check with the user",
        examples=["논문 도메인(단일 원자 촉매)으로 조사를 진행할까요?"]
    )

# 유저 질의와 pdf 논문이 모두 있는 경우
class ClassifiedPaperWithUser(ClarifyWithUser, ClarifyPaper):
    """Multiple paper classification results and user verification question schema"""

class ResearchQuestion(BaseModel):
    """Schema for generating structured research summaries"""
    
    research_brief: str = Field(
        description="A research question that will be used to guide the research",
    )

class UserFeedbackIntent(BaseModel):
    """Understanding User Feedback Intent Schema"""
    
    feedback_intent: str = Field(
        description="Understand the intent of user feedback",
        examples=["approve", "revise"]
    )

class Weight(BaseModel):
    activity: float
    stability : float
    synthesis : float
    cost : float
    evidence : float
    ml_lit_agree : float

class Rationale(BaseModel):
    activity: str
    stability : str
    synthesis : str
    cost : str
    evidence : str
    ml_lit_agree : str

class Criteria(BaseModel):
    weight : Weight
    rationale : Rationale
    feedback_question : str

# ===== state classes =====
class PdfReference(TypedDict):
    id: str
    title: str
    abstract : str
    classification: Optional[str]

class AgentState(MessagesState):
    pdfs : List[PdfReference]
    classified_input : Optional[ClarifyWithUser | ClarifyPaper | ClassifiedPaperWithUser]
    research_brief: Optional[str]
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]
    criteria: Optional[Criteria]
    user_feedback: Optional[str]
    search_results: Optional[List[str]]
    hypothesis_results: Optional[List[str]]
    debate_summary: DebateSummary
    final_report: str