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

class SessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    case_id: Optional[UUID]
    created_at: datetime
    updated_at: datetime
    messages: List[MessageResponse] = []

class SessionCreateRequest(BaseModel):
    case_id: UUID

class MessageSendRequest(BaseModel):
    content: str
