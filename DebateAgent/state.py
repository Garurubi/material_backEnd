from typing import TypedDict, List, Dict, Any, Literal, Optional
from typing_extensions import Annotated
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field, conint
from enum import Enum
import operator

# ===== STRUCTURED OUTPUT SCHEMAS =====
class Role(str, Enum):
    PROPONENT = "Proponent"
    OPPONENT = "Opponent"

class DebateUtteranceLLM(BaseModel):
    """Schema for argument grounds in a debate"""

    claim: str = Field(
        description="The main claim made by the debater",
        examples=["Increased renewable energy leads to grid instability."]
    )
    reasoning: str = Field(
        description="Logical reasoning or evidence supporting the claim"
    )

class Decision(str, Enum):
    CONTINUE = "continue"
    END = "end"

Score = Annotated[int, Field(ge=0, le=2, description="allowed: 0,1,2")]
class Scores(BaseModel):
    new_evidence: Score = Field(description="0–2")
    rebuttal_strength_coherence: Score = Field(description="0–2")
    convergence: Score = Field(description="0–2")

class ModeratorDecision(BaseModel):
    """Schema for moderator decisions in a debate"""
    score: Scores
    decision: Decision = Field(..., description='"continue" or "end"')
    reason: str = Field(..., min_length=10, description="2–3 sentence concise rationale")

class AgreementStatus(str, Enum):
    CONVERGED = "Converged"
    PARTIALLY_CONVERGED = "Partially Converged"
    UNRESOLVED = "Unresolved"

class DebateSummary(BaseModel):
    issue: str = Field(..., min_length=1, description="Short title of the issue")
    proponent_summary: str = Field(..., description="Summary from the proponent")
    opponent_summary: str = Field(..., description="Summary from the opponent")
    agreement_status: AgreementStatus = Field(..., description='One of "Converged", "Partially Converged", "Unresolved"')

# ===== state classes =====
class DebateUtterance(DebateUtteranceLLM):
    turn: int
    role: Role

class SearchResult(TypedDict):
    search_reactions: str
    search_from_mongoDB: str

class DebateState(MessagesState, total=False):
    hypothesis: str
    search_results: SearchResult
    topic: str
    turn_id: int = 0
    turns: Annotated[List[DebateUtterance], operator.add] = []
    moderator_decisions: ModeratorDecision
    debate_summary: DebateSummary

    