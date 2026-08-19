from pydantic import BaseModel, ConfigDict
from uuid import UUID
from datetime import datetime, date
from typing import Optional

class GuidelineResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    version: Optional[str] = None
    author: Optional[str] = None
    publication_date: Optional[date] = None
    created_at: datetime
    updated_at: datetime
