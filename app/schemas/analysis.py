from typing import Any, Optional, List

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    # Frontend sends 'question' as the primary field
    question: str = ""
    # 'message' is kept for backward compatibility but not required
    message: str = ""
    conversation: List[ChatMessage] = []
    findings: List[dict[str, Any]] = []
    quality_score: dict[str, Any] = {}
    ml_readiness: dict[str, Any] = {}
    profile: dict[str, Any] = {}
    dataset_name: str = ""
    
    class Config:
        extra = "allow"


class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    grounded: bool = True


class AnalysisResponse(BaseModel):

    status: str

    message: str

    filename: str

    profile: dict[str, Any]

    understanding: dict[str, Any]

    findings: list[dict[str, Any]]

    quality_score: dict[str, Any]

    ml_readiness: dict[str, Any]

    snapshot: dict[str, Any]

    drift: Optional[dict[str, Any]]

    visualizations: list[dict[str, Any]]

    ai_insights: Optional[dict[str, Any]] = None

    preview: Optional[list[dict[str, Any]]] = None