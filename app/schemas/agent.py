from pydantic import BaseModel, ConfigDict
from uuid import UUID
from datetime import datetime
from typing import List, Optional

class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    role: str
    content: str
    created_at: datetime

class CitationResponse(BaseModel):
    guideline_title: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    distance: Optional[float] = None
    confidence: Optional[int] = None

class SessionStateResponse(BaseModel):
    risk_category: Optional[str] = None
    retrieval_confidence: Optional[int] = None
    evidence_sufficient: Optional[bool] = None
    evaluation_complete: Optional[bool] = None
    citations: List[CitationResponse] = []
    recommendations: List[str] = []
    missing_fields: List[str] = []

class SessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    case_id: Optional[UUID]
    created_at: datetime
    updated_at: datetime
    messages: List[MessageResponse] = []
    state: SessionStateResponse = SessionStateResponse()

class SessionCreateRequest(BaseModel):
    case_id: UUID

class MessageSendRequest(BaseModel):
    content: str
